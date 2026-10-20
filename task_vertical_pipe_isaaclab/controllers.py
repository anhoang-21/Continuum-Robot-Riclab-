"""
==============================================================================
controllers.py - the motor-level controllers of Stage 6 behind one interface
==============================================================================
Every controller maps the observation dict of robust_env.py ({"policy": (N, 33 or 45), "priv": (N, 13)})
to the 7-D action (6 motor speeds + elevator), and can be reset per env:

  MLPPolicy   an rsl_rl actor (train_vertical_pipe.py): its input is told by the first layer's width
              33 = task observation, 45 = + 12 load cells, 46 / 58 = + the 13 privileged parameters
  (rma.RMAPolicy and safety_filter.FilteredController follow the same interface)

`load_controller("path.pt")` picks the class from the checkpoint content.
==============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

import mdp
import uncertainty as U

OBS_T = mdp.OBS_DIM + 12             # task observation + load cells


class Controller:
    name = "controller"
    uses_tension = False
    uses_priv = False

    def reset(self, env_ids=None):
        pass

    def __call__(self, obs: dict, env=None) -> torch.Tensor:
        raise NotImplementedError


class MLPPolicy(Controller):
    """Deterministic mean of an rsl_rl MLP actor (tanh, the architecture of agent_cfg.py)."""

    def __init__(self, path: str, device="cpu", name=None):
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state["actor_state_dict"] if "actor_state_dict" in state else state
        weights = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
        layers = []
        for j, idx in enumerate(weights):
            w = sd[f"mlp.{idx}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(sd[f"mlp.{idx}.bias"])
            layers.append(lin)
            if j < len(weights) - 1:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers).to(device).eval()
        self.device = device
        self.in_dim = layers[0].in_features
        if self.in_dim not in (mdp.OBS_DIM, OBS_T, mdp.OBS_DIM + U.PRIV_DIM, OBS_T + U.PRIV_DIM):
            raise ValueError(f"{path}: unexpected actor input width {self.in_dim}")
        self.uses_priv = self.in_dim in (mdp.OBS_DIM + U.PRIV_DIM, OBS_T + U.PRIV_DIM)
        self.uses_tension = self.in_dim in (OBS_T, OBS_T + U.PRIV_DIM)
        self.name = name or path

    def features(self, obs: dict) -> torch.Tensor:
        x = obs["policy"][:, : OBS_T if self.uses_tension else mdp.OBS_DIM]
        if x.shape[1] < (OBS_T if self.uses_tension else mdp.OBS_DIM):
            raise ValueError(f"{self.name} needs the load-cell tensions in the observation (obs_tension=True)")
        if self.uses_priv:
            x = torch.cat([x, obs["priv"]], dim=1)
        return x

    @torch.no_grad()
    def __call__(self, obs: dict, env=None) -> torch.Tensor:
        x = self.features(obs).to(self.device).float()
        a = self.net(x)
        return a if env is None else a.to(env.device)


def load_controller(path: str, device="cpu", name=None) -> Controller:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and state.get("kind") == "rma":
        from rma import RMAPolicy
        return RMAPolicy.load(path, device=device, name=name)
    return MLPPolicy(path, device=device, name=name)
