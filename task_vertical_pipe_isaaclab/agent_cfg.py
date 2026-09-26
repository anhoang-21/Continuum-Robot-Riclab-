"""
==============================================================================
agent_cfg.py - rsl_rl configuration equal to the SB3 PPO setup of the MuJoCo trainer
==============================================================================
MuJoCo (task_vertical_pipe/train_vertical_pipe.py)      here (rsl_rl OnPolicyRunner + ppo_sb3.SB3PPO)
  PPO("MlpPolicy", ...)                                  MLPModel actor / critic, Gaussian policy
  learning_rate=linear_schedule(3e-4, 1e-5)              lr_initial / lr_final (linear, per timestep)
  n_steps=512                                            num_steps_per_env=512
  batch_size=1024                                        num_mini_batches = n_envs * 512 / 1024
  n_epochs=10                                            num_learning_epochs=10
  gamma=0.99, gae_lambda=0.95                            gamma, lam
  clip_range=0.2                                         clip_param
  ent_coef=0.0, vf_coef=0.5 (SB3 default)                entropy_coef, value_loss_coef
  max_grad_norm=0.5                                      max_grad_norm
  net_arch pi=[256,256], vf=[256,256], Tanh              hidden_dims, activation="tanh"
  log_std_init=-0.5                                      init_std=exp(-0.5), std_type="log"
  no VecNormalize                                        obs_normalization=False
  seed=0, device="cpu"                                   seed, runner device
==============================================================================
"""

import math

N_STEPS = 512
BATCH_SIZE = 1024


def make_agent_cfg(n_envs: int, total_timesteps: int, gamma: float = 0.99, seed: int = 0,
                   lr_initial: float = 3e-4, lr_final: float = 1e-5) -> dict:
    rollout = N_STEPS * n_envs
    return {
        "seed": seed,
        "num_steps_per_env": N_STEPS,
        "save_interval": 10 ** 9,                # checkpoints are written by PipeRunner (every 200k steps)
        "check_for_nan": True,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "tanh",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": math.exp(-0.5), "std_type": "log"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "tanh",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "ppo_sb3:SB3PPO",
            "num_learning_epochs": 10,
            "num_mini_batches": max(1, rollout // BATCH_SIZE),
            "clip_param": 0.2,
            "gamma": gamma,
            "lam": 0.95,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.0,
            "max_grad_norm": 0.5,
            "lr_initial": lr_initial,
            "lr_final": lr_final,
            "total_timesteps": total_timesteps,
            "adam_eps": 1e-5,
            "ortho_init": True,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def num_iterations(total_timesteps: int, n_envs: int) -> int:
    """SB3 keeps collecting rollouts while num_timesteps < total_timesteps."""
    return math.ceil(total_timesteps / (N_STEPS * n_envs))
