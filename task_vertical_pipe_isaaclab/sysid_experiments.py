"""
==============================================================================
sysid_experiments.py - identification experiments on simulated robots with hidden parameters (ND1)
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe sysid_experiments.py --robots 8 --level train
    python sysid.py data/sysid/robot_*.npz --study

The protocol the real K22 would run once it has load cells and mocap markers (P0.1 / P0.2), executed on
robust_env.RobustCableEnv robots whose parameters are drawn from an uncertainty set and kept hidden
(they are only written into the files as true_* for checking):

  1. every motor alone: triangle 0 -> +A -> 0 -> -A -> 0 (A = 6 / 5 / 4 mm for sections 1 / 2 / 3):
     stiffness, coupling, the friction loop of every cable and the backlash at the reversals;
  2. six random multi-motor waypoints and back to 0;
  3. three bent poses, each with 0 -> 20 -> 50 -> 0 g on the tip: backbone vs cable stiffness under load.

The motors move at --speed (1 mm/s) and stop --hold seconds before every sample (every <= 0.3 mm of nut
travel), so the samples are quasi-static. Recorded (noise-free; sysid.py adds the sensor noise): encoder
positions, nut-side tensions (load cells), centres of the discs Seg5 / Seg10 / Seg15 relative to Seg1
(mocap), tilt of those discs (IMU), payload, the true joint angles. No pipes in reach.
==============================================================================
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser(description="Quasi-static identification experiments in Isaac Sim")
parser.add_argument("--robots", type=int, default=8)
parser.add_argument("--level", choices=["nominal", "train", "ood"], default="train")
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--speed", type=float, default=1.0, help="nut speed between samples (mm/s)")
parser.add_argument("--hold", type=float, default=0.5, help="rest before every sample (s)")
parser.add_argument("--sample-step", type=float, default=0.3, help="max nut travel between samples (mm)")
parser.add_argument("--out", type=str, default=os.path.join(TASK_DIR, "data", "sysid"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args, cameras=False)).app

import cables as CB  # noqa: E402
import uncertainty as U  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402

AMPLITUDE = np.array([6.0, 6.0, 5.0, 5.0, 4.0, 4.0]) * 1e-3          # m, per motor (M1 M2 | M5 M6 | M3 M4)


def protocol(rng):
    """Waypoints [(u (6,), payload kg)]; the robot moves linearly between them."""
    wp = [(np.zeros(6), 0.0)]
    for m in range(6):
        for a in (1.0, 0.0, -1.0, 0.0):
            u = np.zeros(6)
            u[m] = a * AMPLITUDE[m]
            wp.append((u, 0.0))
    lim = 0.5 * CB.MOTOR_RANGE / (CB.MOTOR_SECTION + 1)
    for _ in range(6):
        wp.append((rng.uniform(-lim, lim), 0.0))
    wp.append((np.zeros(6), 0.0))
    for pose in ([-5, 0, -4, 0, 3, 0], [0, 5, 0, -4, 0, 3], [4, -4, 0, 4, -3, 0]):
        u = np.array(pose, dtype=np.float64) * 1e-3
        wp.append((u, 0.0))
        for kg in (0.020, 0.050, 0.0):
            wp.append((u, kg))
        wp.append((np.zeros(6), 0.0))
    return wp


def main():
    cfg = RobustCableEnvCfg()
    cfg.scene.num_envs = args.robots
    cfg.uncertainty = args.level
    cfg.pipe_modes = ("random",)
    cfg.max_episode_steps = 10 ** 7
    cfg.seed = args.seed
    cfg.sim.device = args.device
    env = RobustCableEnv(cfg)
    env.auto_reset = False
    env.reset()
    n, dev = env.num_envs, env.device
    # latency / sensor noise do not matter here (motors are set directly; noise is added by sysid.py)
    env.park_pipes()
    env.straighten()
    zero = torch.zeros(n, 7, device=dev)
    for _ in range(50):
        env.step(zero)
    seg = env.robot.find_bodies(["Seg1", "Seg5", "Seg10", "Seg15"], preserve_order=True)[0]

    rng = np.random.default_rng(args.seed + 1)
    wps = protocol(rng)
    dt = env.step_dt
    hold = max(1, int(round(args.hold / dt)))
    rec = {k: [] for k in ("u", "payload", "markers", "imu", "tension", "q")}
    t0 = time.time()
    u_now, kg_now = wps[0]
    for u_next, kg in wps[1:]:
        if kg != kg_now:
            env.set_payload(range(n), kg)
            kg_now = kg
        dist = np.abs(u_next - u_now).max()
        n_samples = max(1, int(np.ceil(dist * 1e3 / args.sample_step)))
        for k in range(1, n_samples + 1):
            target = u_now + (u_next - u_now) * k / n_samples
            prev = env.motor_cmd[0].cpu().numpy()
            moves = max(1, int(np.ceil(np.abs(target - prev).max() * 1e3 / (args.speed * dt))))
            for j in range(1, moves + 1):
                cmd = prev + (target - prev) * j / moves
                env.motor_cmd[:] = torch.as_tensor(cmd, device=dev, dtype=torch.float64)
                env.step(zero)                                  # zero action: the motor targets stay
            for _ in range(hold):
                env.step(zero)
            pos = env.robot.data.body_link_pos_w.torch[:, seg].double()
            rot = matrix_from_quat(env.robot.data.body_link_quat_w.torch[:, seg[1:]].double())
            rec["u"].append(env.motor.cpu().numpy().copy())
            rec["payload"].append(np.full(n, kg_now))
            rec["markers"].append((pos[:, 1:] - pos[:, :1]).cpu().numpy())
            rec["imu"].append(rot[..., :2, 2].cpu().numpy())
            rec["tension"].append(env.ucm.T_n.cpu().numpy().copy())
            rec["q"].append(env.joint_q().cpu().numpy())
        u_now = u_next
        print(f"  waypoint {np.round(u_next * 1e3, 1)} mm, payload {kg_now * 1e3:.0f} g: {len(rec['u'])} samples "
              f"({time.time() - t0:.0f} s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    data = {k: np.stack(v, axis=1) for k, v in rec.items()}               # (n, T, ...)
    p = env.prm
    for i in range(n):
        path = os.path.join(args.out, f"robot_{args.level}_{i}.npz")
        np.savez(path, u=data["u"][i], payload=data["payload"][i], markers=data["markers"][i], imu=data["imu"][i],
                 tension=data["tension"][i], q_true=data["q"][i],
                 true_k_c=p["k_c"][i], true_t0=p["t0"][i], true_k_j=p["k_j"][i], true_mu_theta=p["mu_theta"][i],
                 true_phi=p["phi"][i], true_backlash=p["backlash"][i], true_c_c=p["c_c"][i], true_c_j=p["c_j"][i],
                 level=args.level)
        print(f"robot {i}: k_c {p['k_c'][i].mean():.0f} N/m, T0 {p['t0'][i].mean():.2f} N, k_j {p['k_j'][i]:.2f}, "
              f"mu theta {p['mu_theta'][i].mean():.2f}, backlash {p['backlash'][i].mean() * 1e3:.2f} mm -> {path}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
