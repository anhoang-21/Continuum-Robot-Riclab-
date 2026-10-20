"""
==============================================================================
train_vertical_pipe.py - PPO (rsl_rl) training in Isaac Lab: continuum robot
threads a vertical pipe. Port of D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\train_vertical_pipe.py
==============================================================================
Examples (run from this folder with the Isaac Lab python):
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe train_vertical_pipe.py                    # 3M steps, mixed scenes
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe train_vertical_pipe.py --pipe-mode rig    # only the plate-hole pipe
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe train_vertical_pipe.py --resume models/ppo_vpipe/best_model.pt

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

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser(description="PPO (rsl_rl): continuum robot moves down through a vertical pipe")
parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total training steps (default 3M)")
parser.add_argument("--n-envs", type=int, default=16, help="Parallel environments (default 16, as the MuJoCo trainer)")
parser.add_argument("--steps-per-env", type=int, default=0,
                    help="rollout length per env and update (default 512 = SB3 n_steps; with many envs use "
                         "8192 / n-envs to keep 8192 samples per update, e.g. --n-envs 128 --steps-per-env 64)")
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
parser.add_argument("--eval-episodes", type=int, default=20,
                    help="Deterministic episodes per evaluation (with many envs more cost no extra time)")
parser.add_argument("--checkpoint-freq", type=int, default=200_000, help="Checkpoint every N steps")
parser.add_argument("--stop-at", type=int, default=0,
                    help="Stop after this many steps; the learning-rate schedule still spans --timesteps (0 = off)")
parser.add_argument("--viewer", action="store_true", help="Open the Isaac Sim window (training runs headless by default)")
parser.add_argument("--cable", action="store_true",
                    help="cable-driven robot (cable_env.py): 6 lead-screw motors pull 12 cables, no joint servos")
parser.add_argument("--action-mode", choices=["motor", "bend"], default="motor",
                    help="--cable: motor = the policy drives the 6 motors, bend = bend rates + the lab's cable formula")
parser.add_argument("--no-gravity", action="store_true", help="--cable: no gravity on the arm")
# Stage 6: uncertainty, safety constraints (robust_env.py, ppo_lag.py)
parser.add_argument("--robust", choices=["nominal", "train", "ood"], default=None,
                    help="cable-driven robot with per-env physical parameters drawn from this uncertainty set "
                         "(robust_env.py; implies --cable)")
parser.add_argument("--uncertainty-json", type=str, default="", help="--robust: identified ranges (sysid.py)")
parser.add_argument("--constrained", action="store_true",
                    help="PPO-Lagrangian (ppo_lag.py) on the tension / contact costs of robust_env.py")
parser.add_argument("--cost-limits", type=str, default="0.02,0.01,0.01",
                    help="--constrained: allowed mean per-step cost of slack,over,contact")
parser.add_argument("--constrain", type=str, default="slack,over,contact", help="--constrained: which costs")
parser.add_argument("--pid", type=str, default="1.0,0.1,0.0", help="--constrained: K_p,K_i,K_d of the multipliers")
parser.add_argument("--t-min", type=float, default=0.5, help="--robust: slack threshold of the cable tension (N)")
parser.add_argument("--t-max", type=float, default=10.0, help="--robust: tension limit (N)")
parser.add_argument("--f-max", type=float, default=2.0, help="--robust: limit of the robot's force on the pipe (N)")
parser.add_argument("--pretension", type=float, default=None,
                    help="--robust: nominal cable pre-tension T0 (N) (pre-tension design study)")
parser.add_argument("--obs-tension", action="store_true", help="--robust: the 12 load-cell tensions in the observation")
parser.add_argument("--priv", choices=["none", "critic", "both"], default="none",
                    help="--robust: normalised physical parameters as input of the critic (asymmetric) or of "
                         "actor and critic (privileged teacher for train_rma.py)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")      # PhysX GPU pipeline (the CPU pipeline is not supported, see README)
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
if args.robust:
    args.cable = True
app = AppLauncher(launcher_args(args, gui=args.viewer, cameras=False)).app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from agent_cfg import N_STEPS, make_agent_cfg, num_iterations  # noqa: E402
from pipe_runner import PipeRunner  # noqa: E402
from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg, scene_view_cfg  # noqa: E402


def floats(text):
    return tuple(float(v) for v in text.split(","))


def widen_checkpoint(path, runner, out_path):
    """
    A checkpoint whose actor / critic read fewer inputs than this run's (e.g. the 33-D task observation, now
    + 12 load cells or + 13 privileged parameters): the first layers get zero columns for the new inputs, so
    the run starts from exactly the old policy. Returns the path of the widened checkpoint (or `path`).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    changed = False
    for key, model in (("actor_state_dict", runner.alg.actor), ("critic_state_dict", runner.alg.critic)):
        sd = ckpt[key]
        want = model.state_dict()["mlp.0.weight"]
        have = sd["mlp.0.weight"]
        if have.shape[1] < want.shape[1]:
            w = torch.zeros(have.shape[0], want.shape[1], dtype=have.dtype)
            w[:, :have.shape[1]] = have
            sd["mlp.0.weight"] = w
            for k in [k for k in sd if "normalizer" in k]:
                sd.pop(k)
            changed = True
    if not changed:
        return path
    ckpt.pop("optimizer_state_dict", None)
    torch.save(ckpt, out_path)
    print(f"  widened the first layers of {path} to this run's inputs (new inputs start with zero weight)")
    return out_path


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

    cfg = RobustCableEnvCfg() if args.robust else (CableVerticalPipeEnvCfg() if args.cable else VerticalPipeEnvCfg())
    if args.cable:
        cfg.action_mode = args.action_mode
        cfg.gravity = not args.no_gravity
    if args.robust:
        cfg.uncertainty = args.robust
        cfg.uncertainty_json = args.uncertainty_json
        cfg.t_min, cfg.t_max, cfg.f_max = args.t_min, args.t_max, args.f_max
        cfg.obs_tension = args.obs_tension
        cfg.priv_obs = args.priv != "none"
        if args.pretension is not None:
            cfg.cable_pretension = args.pretension
    cfg.scene.num_envs = args.n_envs
    cfg.pipe_modes = env_modes(args.pipe_mode)
    cfg.max_episode_steps = args.max_episode_steps
    cfg.assist_prob = args.assist
    cfg.contact_penalty = args.contact_penalty
    cfg.random_offset = (0.0, args.max_offset)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.viewer:                                      # Isaac Lab 3: the window is a Kit visualizer
        cfg.sim.visualizer_cfgs = [scene_view_cfg(headless=False)]
    modes = [cfg.pipe_modes[i % len(cfg.pipe_modes)] for i in range(args.n_envs)]
    n_steps = args.steps_per_env or N_STEPS

    print("=" * 75)
    print("PPO (rsl_rl) TRAINING - CONTINUUM ROBOT THROUGH A VERTICAL PIPE - ISAAC LAB")
    print(f"  Robot           : " + (f"cable-driven, {args.action_mode} actions, gravity {not args.no_gravity}"
                                     if args.cable else "joint servos (bend commands)"))
    print(f"  Scenes          : {args.pipe_mode} ({modes.count('rig')} rig / {modes.count('random')} random envs)")
    print(f"  Timesteps       : {args.timesteps:,}  |  episode <= {args.max_episode_steps} steps x 20 ms "
          f"(dt 2 ms x decimation 10)")
    print(f"  Assisted starts : {args.assist:.0%} of training episodes (eval: 0%)")
    print(f"  Wall contact    : -{args.contact_penalty} per step  |  random pipe offset <= {args.max_offset * 100:.0f} cm")
    print(f"  Rollout         : {args.n_envs} envs x {n_steps} steps = {args.n_envs * n_steps:,} samples per update")
    print(f"  Devices         : PPO {args.ppo_device} | PhysX {cfg.sim.device}")
    print(f"  Models -> {models_dir}")
    print(f"  Logs   -> {logs_dir}   (tensorboard --logdir {os.path.join(TASK_DIR, 'logs')})")
    print("=" * 75)

    if args.robust:
        print(f"  Uncertainty     : {args.robust}{' + ' + args.uncertainty_json if args.uncertainty_json else ''} | "
              f"tension band [{args.t_min}, {args.t_max}] N, pipe force <= {args.f_max} N"
              + (f" | pre-tension {args.pretension} N" if args.pretension is not None else ""))
        print(f"  Observation     : 33{' + 12 load cells' if args.obs_tension else ''}"
              f"{' | privileged parameters -> ' + args.priv if args.priv != 'none' else ''}")
        if args.constrained:
            print(f"  Constraints     : PPO-Lagrangian on {args.constrain}, limits {args.cost_limits} "
                  f"(per step), PID {args.pid}")
        print("=" * 75)
    env_cls = RobustCableEnv if args.robust else (CableVerticalPipeEnv if args.cable else VerticalPipeEnv)
    env = RslRlVecEnvWrapper(env_cls(cfg))
    agent_cfg = make_agent_cfg(args.n_envs, args.timesteps, gamma=args.gamma, seed=args.seed, n_steps=n_steps)
    if args.priv in ("critic", "both"):
        agent_cfg["obs_groups"] = {"actor": ["policy", "priv"] if args.priv == "both" else ["policy"],
                                   "critic": ["policy", "priv"]}
    if args.constrained:
        names = ("slack", "over", "contact")
        wanted = args.constrain.split(",")
        agent_cfg["algorithm"].update({
            "class_name": "ppo_lag:SB3PPOLag", "cost_limits": floats(args.cost_limits),
            "pid": floats(args.pid), "constrain": tuple(n in wanted for n in names),
            "cost_groups": tuple(agent_cfg["obs_groups"]["critic"]),
        })
    runner = PipeRunner(env, agent_cfg, log_dir=logs_dir, device=args.ppo_device, models_dir=models_dir,
                        eval_freq=args.eval_freq, n_eval_episodes=args.eval_episodes, checkpoint_freq=args.checkpoint_freq)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        print(f"\nResuming from {args.resume}")
        resume = widen_checkpoint(args.resume, runner, os.path.join(models_dir, "init_widened.pt"))
        has_optimizer = "optimizer_state_dict" in torch.load(resume, map_location="cpu", weights_only=False)
        runner.load(resume, load_cfg={"actor": True, "critic": True, "optimizer": has_optimizer, "iteration": False},
                    map_location=args.ppo_device)
        runner.alg.set_schedule(1e-4, 1e-5, args.timesteps)

    iterations = num_iterations(args.stop_at or args.timesteps, args.n_envs, n_steps)
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
    cable = f" --cable --action-mode {args.action_mode}" if args.cable else ""
    print(f"Watch it:  D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe play_vertical_pipe.py --checkpoint {best_path}{cable}")


if __name__ == "__main__":
    main()
    app.close()
