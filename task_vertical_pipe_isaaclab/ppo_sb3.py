"""
==============================================================================
ppo_sb3.py - rsl_rl PPO with the update rule of Stable-Baselines3 PPO
==============================================================================
The MuJoCo version trains with SB3 PPO (train_vertical_pipe.py). This port uses
rsl_rl (runner, rollout storage, logging) but makes the PPO update the one SB3 does,
where rsl_rl's defaults differ:

  SB3 (MuJoCo trainer)                         rsl_rl default -> here
  lr linear 3e-4 -> 1e-5 over the run           fixed / KL-adaptive -> linear, same formula
  Adam eps 1e-5                                 1e-8 -> 1e-5
  orthogonal init: hidden sqrt(2), pi 0.01,     PyTorch default -> SB3 ortho_init
    value 1, biases 0
  minibatches reshuffled every epoch            one permutation for all epochs -> reshuffled
  advantages normalised per minibatch           per rollout (option) -> per minibatch
  value loss MSE, no clipping (vf_coef 0.5)     clipped -> plain MSE
  one grad-norm clip over all parameters        separate actor / critic clips -> one clip
  truncated episodes bootstrapped with          V(s_t) of the last step -> V(final obs),
    V(info["terminal_observation"])               read from extras["terminal_obs"]
==============================================================================
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


def sb3_orthogonal_init(mlp: nn.Sequential, out_gain: float) -> None:
    """SB3 ActorCriticPolicy(ortho_init=True): hidden layers gain sqrt(2), output layer out_gain, zero biases."""
    linears = [m for m in mlp if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears):
        nn.init.orthogonal_(layer.weight, gain=out_gain if i == len(linears) - 1 else math.sqrt(2.0))
        nn.init.zeros_(layer.bias)


class SB3PPO(PPO):
    def __init__(self, actor, critic, storage, *, lr_initial: float = 3e-4, lr_final: float = 1e-5,
                 total_timesteps: int = 3_000_000, adam_eps: float = 1e-5, ortho_init: bool = True, **kwargs):
        for key in ("learning_rate", "schedule", "desired_kl", "use_clipped_value_loss",
                    "normalize_advantage_per_mini_batch", "optimizer"):
            kwargs.pop(key, None)
        super().__init__(actor, critic, storage, learning_rate=lr_initial, schedule="fixed", desired_kl=None,
                         use_clipped_value_loss=False, normalize_advantage_per_mini_batch=True, **kwargs)
        if ortho_init:
            sb3_orthogonal_init(self.actor.mlp, out_gain=0.01)
            sb3_orthogonal_init(self.critic.mlp, out_gain=1.0)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr_initial, eps=adam_eps)
        self.set_schedule(lr_initial, lr_final, total_timesteps)

    def set_schedule(self, lr_initial: float, lr_final: float, total_timesteps: int) -> None:
        """SB3 linear_schedule(initial, final) over `total_timesteps` (restarts the step counter, like
        model.learn(reset_num_timesteps=True))."""
        self.lr_initial, self.lr_final = float(lr_initial), float(lr_final)
        self.total_timesteps = int(total_timesteps)
        self.num_timesteps = 0
        self.learning_rate = self.lr_initial
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    # ------------------------------------------------------------------
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        time_outs = extras.get("time_outs")
        if time_outs is not None and "terminal_obs" in extras:
            extras = {k: v for k, v in extras.items() if k != "time_outs"}      # no V(s_t) bootstrap in super()
            time_outs = time_outs.to(self.device).bool().view(-1)
            if time_outs.any():
                final = TensorDict({"policy": extras["terminal_obs"].to(self.device)}, batch_size=[time_outs.shape[0]])
                final_values = self.critic(final).detach().view(-1)
                rewards = rewards + self.gamma * final_values * time_outs.to(rewards.dtype)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        # learning rate: SB3 evaluates the schedule after the rollout, with num_timesteps already advanced
        self.num_timesteps += self.storage.num_envs * self.storage.num_transitions_per_env
        progress_remaining = 1.0 - self.num_timesteps / self.total_timesteps
        self.learning_rate = self.lr_final + progress_remaining * (self.lr_initial - self.lr_final)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

        stats = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for _ in range(self.num_learning_epochs):
            # a new permutation every epoch (SB3 RolloutBuffer.get)
            for batch in self.storage.mini_batch_generator(self.num_mini_batches, 1):
                advantages = batch.advantages.view(-1)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.actor(batch.observations, stochastic_output=True)
                log_prob = self.actor.get_output_log_prob(batch.actions)
                entropy = self.actor.output_entropy
                values = self.critic(batch.observations).view(-1)

                log_ratio = log_prob - batch.old_actions_log_prob.view(-1)
                ratio = torch.exp(log_ratio)
                policy_loss = -torch.min(advantages * ratio,
                                         advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
                value_loss = F.mse_loss(batch.returns.view(-1), values)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.entropy_coef * entropy_loss + self.value_loss_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["value"] += value_loss.item()
                    stats["surrogate"] += policy_loss.item()
                    stats["entropy"] += entropy.mean().item()
                    stats["clip_fraction"] += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    stats["approx_kl"] += ((ratio - 1.0) - log_ratio).mean().item()
                n_updates += 1
        self.storage.clear()
        return {k: v / max(n_updates, 1) for k, v in stats.items()}
