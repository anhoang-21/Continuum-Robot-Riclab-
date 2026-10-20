"""
==============================================================================
ppo_lag.py - constrained PPO: SB3PPO + cost critic + PID-Lagrange multipliers (ND3 / ND4)
==============================================================================
The constrained problem

    max_pi  E[ sum_t gamma^t r_t ]     s.t.   J_ci = mean per-step cost i  <=  d_i,   i = slack, over, contact

(costs from robust_env.py: extras["cost"] (N, 3)) is solved with the Lagrangian

    L(pi, lam) = J_r(pi) - sum_i lam_i (J_ci(pi) - d_i),      lam_i >= 0.

Policy step: the PPO-clip surrogate of the combined advantage

    A = (A_r - sum_i lam_i A_ci) / (1 + sum_i lam_i)

with reward / cost advantages normalised separately over the rollout (GAE, the same gamma / lambda;
the cost GAE uses its own critic V_c (an MLP with one output per cost) trained by MSE on the cost
returns). Truncated episodes are bootstrapped with V_c(terminal observation), like the rewards.

Multiplier step: PID Lagrangian (Stooke, Achiam, Abbeel, "Responsive Safety in Reinforcement Learning
by PID Lagrangian Methods", ICML 2020) on the mean per-step cost of the rollout, which damps the
oscillation of the plain gradient-ascent multiplier:
    e = J_c - d,  I = max(0, I + e),  lam = max(0, K_p EMA(e) + K_i I + K_d max(0, EMA(J_c) - EMA(J_c)_prev)).

Everything else is SB3PPO (ppo_sb3.py): the update rule of the MuJoCo trainer.
==============================================================================
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from ppo_sb3 import SB3PPO, sb3_orthogonal_init

COST_NAMES = ("slack", "over", "contact")


class CostCritic(nn.Module):
    """V_c(s): one value per cost, on the concatenated observation groups of the critic."""

    def __init__(self, obs_dim: int, n_costs: int, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, n_costs))
        self.mlp = nn.Sequential(*layers)
        sb3_orthogonal_init(self.mlp, out_gain=1.0)

    def forward(self, x):
        return self.mlp(x)


class SB3PPOLag(SB3PPO):
    def __init__(self, actor, critic, storage, *, cost_limits=(0.02, 0.01, 0.01), cost_groups=("policy",),
                 pid=(1.0, 0.1, 0.0), lam_init=0.0, lam_max=50.0, ema=0.8, cost_lr=3e-4,
                 constrain=(True, True, True), **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        self.cost_limits = torch.as_tensor(cost_limits, dtype=torch.float32, device=self.device)
        self.n_costs = len(cost_limits)
        self.constrain = torch.as_tensor(constrain, dtype=torch.bool, device=self.device)
        self.cost_groups = tuple(cost_groups)
        obs0 = storage.observations
        obs_dim = sum(obs0[g].shape[-1] for g in self.cost_groups)
        self.cost_critic = CostCritic(obs_dim, self.n_costs).to(self.device)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_lr, eps=1e-5)
        T, N = storage.num_transitions_per_env, storage.num_envs
        self.cost_buf = torch.zeros(T, N, self.n_costs, device=self.device)
        self.cost_values = torch.zeros(T, N, self.n_costs, device=self.device)
        self.cost_returns = torch.zeros(T, N, self.n_costs, device=self.device)
        self.kp, self.ki, self.kd = pid
        self.lam_max, self.ema_beta = lam_max, ema
        self.lagr = torch.full((self.n_costs,), float(lam_init), device=self.device)   # Lagrange multipliers
        self.pid_i = torch.zeros(self.n_costs, device=self.device)
        self.err_ema = torch.zeros(self.n_costs, device=self.device)
        self.cost_ema = torch.zeros(self.n_costs, device=self.device)
        self.cost_ema_prev = torch.zeros(self.n_costs, device=self.device)
        self._ema_ready = False
        self.last_cost_mean = torch.zeros(self.n_costs, device=self.device)

    # ------------------------------------------------------------------
    def _cost_input(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.cost_groups], dim=-1)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        step = self.storage.step
        cost = extras.get("cost")
        if cost is None:
            cost = torch.zeros(dones.shape[0], self.n_costs, device=self.device)
        cost = cost.to(self.device).float().clone()
        time_outs = extras.get("time_outs")
        if time_outs is not None and "terminal_obs" in extras:
            to = time_outs.to(self.device).bool().view(-1)
            if to.any():
                final = TensorDict({"policy": extras["terminal_obs"].to(self.device)}, batch_size=[to.shape[0]])
                for g in self.cost_groups:
                    if g != "policy":
                        # (the env has already drawn the next robot: its parameters stand in for the ended one's)
                        final[g] = obs[g].to(self.device)
                with torch.no_grad():
                    cost = cost + self.gamma * self.cost_critic(self._cost_input(final)) * to[:, None].float()
        if step < self.cost_buf.shape[0]:
            self.cost_buf[step].copy_(cost)
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        super().compute_returns(obs)
        st = self.storage
        with torch.no_grad():
            flat = st.observations.flatten(0, 1)
            self.cost_values.copy_(self.cost_critic(self._cost_input(flat)).view(st.num_transitions_per_env, st.num_envs, -1))
            last = self.cost_critic(self._cost_input(obs.to(self.device)))
            adv = torch.zeros_like(last)
            for t in reversed(range(st.num_transitions_per_env)):
                nxt = last if t == st.num_transitions_per_env - 1 else self.cost_values[t + 1]
                not_done = 1.0 - st.dones[t].float()                                   # (N, 1)
                delta = self.cost_buf[t] + not_done * self.gamma * nxt - self.cost_values[t]
                adv = delta + not_done * self.gamma * self.lam * adv                     # self.lam: GAE lambda
                self.cost_returns[t] = adv + self.cost_values[t]
            cost_adv = self.cost_returns - self.cost_values                            # (T, N, k)
            # multipliers from this rollout's mean per-step cost
            self.last_cost_mean = self.cost_buf.mean(dim=(0, 1))
            self._update_multipliers(self.last_cost_mean)
            # combined advantage, reward / cost parts normalised separately
            a_r = st.advantages
            a_r = (a_r - a_r.mean()) / (a_r.std() + 1e-8)
            a_c = (cost_adv - cost_adv.mean(dim=(0, 1))) / (cost_adv.std(dim=(0, 1)) + 1e-8)
            lam = self.lagr * self.constrain.float()
            st.advantages = (a_r - (a_c * lam).sum(-1, keepdim=True)) / (1.0 + lam.sum())

    def _update_multipliers(self, cost_mean):
        err = cost_mean - self.cost_limits
        if not self._ema_ready:
            self.err_ema.copy_(err)
            self.cost_ema.copy_(cost_mean)
            self.cost_ema_prev.copy_(cost_mean)
            self._ema_ready = True
        b = self.ema_beta
        self.err_ema.mul_(b).add_((1 - b) * err)
        self.cost_ema_prev.copy_(self.cost_ema)
        self.cost_ema.mul_(b).add_((1 - b) * cost_mean)
        self.pid_i.copy_(torch.clamp(self.pid_i + err, min=0.0))
        deriv = torch.clamp(self.cost_ema - self.cost_ema_prev, min=0.0)
        self.lagr.copy_(torch.clamp(self.kp * self.err_ema + self.ki * self.pid_i + self.kd * deriv, 0.0, self.lam_max))

    def update(self) -> dict[str, float]:
        # cost critic: MSE on the cost returns, same epochs / minibatches as the policy
        st = self.storage
        flat_in = self._cost_input(st.observations.flatten(0, 1)).detach()
        flat_ret = self.cost_returns.flatten(0, 1)
        n = flat_in.shape[0]
        mb = max(1, n // self.num_mini_batches)
        cost_loss, k = 0.0, 0                        # (own constant learning rate: the critic may start fresh)
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.num_mini_batches):
                idx = perm[i * mb:(i + 1) * mb]
                loss = (self.cost_critic(flat_in[idx]) - flat_ret[idx]).pow(2).mean()
                self.cost_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_optimizer.step()
                cost_loss += loss.item()
                k += 1
        out = super().update()                       # policy + reward critic (clears the storage)
        out["cost_value"] = cost_loss / max(k, 1)
        for i, name in enumerate(COST_NAMES[: self.n_costs]):
            out[f"lagrange_{name}"] = float(self.lagr[i])
            out[f"cost_{name}"] = float(self.last_cost_mean[i])
        return out

    # ------------------------------------------------------------------
    def save(self) -> dict:
        d = super().save()
        d["cost_critic_state_dict"] = self.cost_critic.state_dict()
        d["cost_optimizer_state_dict"] = self.cost_optimizer.state_dict()
        d["lagrange"] = {"lam": self.lagr.cpu(), "pid_i": self.pid_i.cpu(), "err_ema": self.err_ema.cpu(),
                         "cost_ema": self.cost_ema.cpu(), "cost_limits": self.cost_limits.cpu()}
        return d

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        out = super().load(loaded_dict, load_cfg, strict)
        if "cost_critic_state_dict" in loaded_dict and (load_cfg is None or load_cfg.get("critic")):
            self.cost_critic.load_state_dict(loaded_dict["cost_critic_state_dict"])
        return out
