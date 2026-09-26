"""
==============================================================================
train_vertical_pipe.py - PPO (rsl_rl) training in Isaac Lab: continuum robot
threads a vertical pipe. Port of D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\train_vertical_pipe.py
==============================================================================
Examples (run from this folder with the Isaac Lab python):
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe train_vertical_pipe.py                    # 3M steps, mixed scenes
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe train_vertical_pipe.py --pipe-mode rig    # only the plate-hole pipe
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe train_vertical_pipe.py --resume models/ppo_vpipe/best_model.pt

Same options and defaults as the MuJoCo trainer (--timesteps 3M, --n-envs 16, --pipe-mode
mixed = even envs on the rig pipe / odd envs on a random pipe, --assist 0.4, --contact-penalty 0.1,
--max-offset 0.17, --gamma 0.99, --seed 0, PPO on cpu: --ppo-device). Hyperparameters: agent_cfg.py.

Outputs:
    models/<run-name>/best_model.pt   (best eval reward)     models/<run-name>/final_model.pt
    models/<run-name>/model_<N>_steps.pt (every 200k steps)  logs/<run-name>/ (TensorBoard)
==============================================================================
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file

# Windows: load torch / tensordict / rsl_rl DLLs before Kit starts (avoids access violations)
import torch  # noqa: F401
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="PPO (rsl_rl): continuum robot moves down through a vertical pipe")
parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total training steps (default 3M)")
parser.add_argument("--n-envs", type=int, default=16, help="Parallel environments (default 16, as the MuJoCo trainer)")
parser.add_argument("--pipe-mode", choices=["rig", "random", "mixed"], default="mixed",
                    help="rig = pipe on the plate hole, random = pipe anywhere in reach, mixed = both")
parser.add_argument("--max-episode-steps", type=int, default=400, help="Steps per episode (20 ms each)")
parser.add_argument("--assist", type=float, default=0.4,
                    help="Share of training episodes that start part-way along the solution (evaluation: 0)")
parser.add_argument("--run-name", type=str, default="ppo_vpipe", help="Sub-folder for models / logs")
parser.add_argument("--resume", type=str, default="", help="Continue training from this .pt (lr 1e-4 -> 1e-5)")
parser.add_argument("--max-offset", type=float, default=0.17,
                    help="random scene: max distance of the pipe from the robot axis (m, <= 0.19)")
parser.add_argument("--contact-penalty", type=float, default=0.1, help="Cost per step touching the inner wall")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ppo-device", type=str, default="cpu",
                    help="PPO device (the MuJoCo trainer's --device, cpu). The PhysX device is --device")
parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation every N steps (0 = off)")
parser.add_argument("--checkpoint-freq", type=int, default=200_000, help="Checkpoint every N steps")
parser.add_argument("--stop-at", type=int, default=0,
                    help="Stop after this many steps; the learning-rate schedule still spans --timesteps (0 = off)")
parser.add_argument("--viewer", action="store_true", help="Open the Isaac Sim window (training runs headless by default)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")      # PhysX GPU pipeline (the CPU pipeline is not supported, see README)
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.headless = not args.viewer
app = AppLauncher(args).app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from agent_cfg import make_agent_cfg, num_iterations  # noqa: E402
from pipe_runner import PipeRunner  # noqa: E402
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg  # noqa: E402


def env_modes(pipe_mode):
    return ("rig", "random") if pipe_mode == "mixed" else (pipe_mode,)


def main():
    # never overwrite a trained model (same rule as the MuJoCo trainer)
    run_name, suffix = args.run_name, 1
    while any(os.path.exists(os.path.join(TASK_DIR, "models", run_name, f)) for f in ("best_model.pt", "final_model.pt")):
        suffix += 1
        run_name = f"{args.run_name}_{suffix}"
    if run_name != args.run_name:
        print(f"models/{args.run_name} already holds a trained model -> this run is saved as '{run_name}'")
    models_dir = os.path.join(TASK_DIR, "models", run_name)
    logs_dir = os.path.join(TASK_DIR, "logs", run_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    cfg = VerticalPipeEnvCfg()
    cfg.scene.num_envs = args.n_envs
    cfg.pipe_modes = env_modes(args.pipe_mode)
    cfg.max_episode_steps = args.max_episode_steps
    cfg.assist_prob = args.assist
    cfg.contact_penalty = args.contact_penalty
    cfg.random_offset = (0.0, args.max_offset)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    modes = [cfg.pipe_modes[i % len(cfg.pipe_modes)] for i in range(args.n_envs)]

    print("=" * 75)
    print("PPO (rsl_rl) TRAINING - CONTINUUM ROBOT THROUGH A VERTICAL PIPE - ISAAC LAB")
    print(f"  Scenes          : {args.pipe_mode} ({modes.count('rig')} rig / {modes.count('random')} random envs)")
    print(f"  Timesteps       : {args.timesteps:,}  |  episode <= {args.max_episode_steps} steps x 20 ms "
          f"(dt 2 ms x decimation 10)")
    print(f"  Assisted starts : {args.assist:.0%} of training episodes (eval: 0%)")
    print(f"  Wall contact    : -{args.contact_penalty} per step  |  random pipe offset <= {args.max_offset * 100:.0f} cm")
    print(f"  Devices         : PPO {args.ppo_device} | PhysX {cfg.sim.device}")
    print(f"  Models -> {models_dir}")
    print(f"  Logs   -> {logs_dir}   (tensorboard --logdir {os.path.join(TASK_DIR, 'logs')})")
    print("=" * 75)

    env = RslRlVecEnvWrapper(VerticalPipeEnv(cfg))
    agent_cfg = make_agent_cfg(args.n_envs, args.timesteps, gamma=args.gamma, seed=args.seed)
    runner = PipeRunner(env, agent_cfg, log_dir=logs_dir, device=args.ppo_device, models_dir=models_dir,
                        eval_freq=args.eval_freq, n_eval_episodes=20, checkpoint_freq=args.checkpoint_freq)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        print(f"\nResuming from {args.resume}")
        has_optimizer = "optimizer_state_dict" in torch.load(args.resume, map_location="cpu", weights_only=False)
        runner.load(args.resume, load_cfg={"actor": True, "critic": True, "optimizer": has_optimizer, "iteration": False},
                    map_location=args.ppo_device)
        runner.alg.set_schedule(1e-4, 1e-5, args.timesteps)

    iterations = num_iterations(args.stop_at or args.timesteps, args.n_envs)
    start = time.time()
    try:
        runner.learn(num_learning_iterations=iterations)
        print(f"\nDone in {(time.time() - start) / 60:.1f} min. Saved {os.path.join(models_dir, 'final_model.pt')}")
    except KeyboardInterrupt:
        runner.save(os.path.join(models_dir, "interrupted_model.pt"))
        print(f"\nInterrupted - saved {os.path.join(models_dir, 'interrupted_model.pt')}")
    finally:
        env.close()
    best_path = os.path.join(models_dir, "best_model.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(models_dir, "final_model.pt")
    print(f"Watch it:  D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_vertical_pipe.py --checkpoint {best_path}")


if __name__ == "__main__":
    main()
    app.close()
