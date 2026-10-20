"""
==============================================================================
rma.py - online adaptation to the robot's physical parameters (Rapid Motor Adaptation, ND2 / ND3)
==============================================================================
Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots" (RSS 2021), in its explicit form
(the adaptation module estimates the physical parameters themselves, like an online system identification):

  phase 1  teacher pi(o_t, e): PPO (train_vertical_pipe.py --robust train --priv both) with the normalised
           parameters e of the robot (uncertainty.privileged: cable stiffness / spread, pre-tension and pair
           asymmetry, backbone stiffness / damping, cable damping, capstan friction, compliance split,
           backlash, motor bandwidth, payload, latency) as extra input;
  phase 2  adaptation module e_hat = phi(o_{t-H+1..t}): a temporal convolution over the last H = 50 steps
           (1 s) of the observation (which holds the previous action), trained by regression on e over the
           states the deployed student itself visits (DAgger, train_rma.py);
  deploy   pi(o_t, phi(history)): no privileged input, the robot identifies itself while it moves.

RMAPolicy follows the controllers.Controller interface (per-env history, reset at episode start).
==============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

import uncertainty as U
from controllers import Controller, MLPPolicy, OBS_T

HISTORY = 50


class AdaptationModule(nn.Module):
    """(N, H, d_obs) -> (N, PRIV_DIM): per-step embedding + 3 strided 1-D convolutions + linear head."""

    def __init__(self, obs_dim: int, out_dim: int = U.PRIV_DIM, hist: int = HISTORY, width: int = 64):
        super().__init__()
        self.obs_dim, self.hist = obs_dim, hist
        self.embed = nn.Sequential(nn.Linear(obs_dim, width), nn.ELU(), nn.Linear(width, 32), nn.ELU())
        self.conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.ELU())
        with torch.no_grad():
            n_out = self.conv(torch.zeros(1, 32, hist)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_out, 64), nn.ELU(), nn.Linear(64, out_dim))

    def forward(self, hist):
        z = self.embed(hist).transpose(1, 2)                      # (N, 32, H)
        return self.head(self.conv(z))


class RMAPolicy(Controller):
    """Teacher actor + adaptation module; keeps the observation history of every env."""

    def __init__(self, teacher: MLPPolicy, adapter: AdaptationModule, num_envs: int = 0, device="cpu", name="rma"):
        self.teacher, self.adapter, self.device, self.name = teacher, adapter.to(device).eval(), device, name
        self.obs_dim = adapter.obs_dim
        self.uses_tension = self.obs_dim == OBS_T
        self.hist = None
        if num_envs:
            self._alloc(num_envs)
        self.last_e_hat = None

    def _alloc(self, n):
        self.hist = torch.zeros(n, self.adapter.hist, self.obs_dim, device=self.device)
        self._fresh = torch.ones(n, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if self.hist is None:
            return
        if env_ids is None:
            self.hist.zero_()
            self._fresh.fill_(True)
        else:
            idx = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
            self.hist[idx] = 0.0
            self._fresh[idx] = True

    def push(self, obs_policy):
        x = obs_policy[:, : self.obs_dim].to(self.device).float()
        if self.hist is None or self.hist.shape[0] != x.shape[0]:
            self._alloc(x.shape[0])
        # first step of an episode: fill the whole history with it (no zero-padding artefacts)
        fresh = self._fresh
        if fresh.any():
            self.hist[fresh] = x[fresh][:, None, :].expand(-1, self.adapter.hist, -1)
            self._fresh[:] = False
        self.hist = torch.cat([self.hist[:, 1:], x[:, None]], dim=1)

    @torch.no_grad()
    def estimate(self):
        return self.adapter(self.hist)

    @torch.no_grad()
    def __call__(self, obs: dict, env=None) -> torch.Tensor:
        self.push(obs["policy"])
        e_hat = self.estimate()
        self.last_e_hat = e_hat
        x = obs["policy"][:, : OBS_T if self.teacher.uses_tension else 33].to(self.device).float()
        a = self.teacher.net(torch.cat([x, e_hat], dim=1))
        return a if env is None else a.to(env.device)

    def save(self, path, teacher_path):
        torch.save({"kind": "rma", "teacher": teacher_path, "obs_dim": self.obs_dim, "hist": self.adapter.hist,
                    "adapter_state_dict": self.adapter.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu", name=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        teacher = MLPPolicy(d["teacher"], device=device)
        adapter = AdaptationModule(d["obs_dim"], hist=d["hist"])
        adapter.load_state_dict(d["adapter_state_dict"])
        return cls(teacher, adapter, device=device, name=name or path)
