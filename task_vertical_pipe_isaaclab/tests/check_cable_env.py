"""
==============================================================================
tests/check_cable_env.py - checks of the cable-driven robot (cable_env.py) in PhysX
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe tests\\check_cable_env.py [--no-gravity]

1. statics: random motor targets -> let PhysX settle -> the pose must be the equilibrium of
   cables.CableModel (+ GravityModel), and the motor torque the screw torque of the nut load
2. single motors: pulling one motor bends its section towards the motor angle; the cables of
   sections 2 / 3 also bend the sections below (cable coupling)
3. the lab's controller formula ("bend" mode): commanded vs achieved bend
4. dynamics: random policy actions for 400 steps without instability; start states at rest
==============================================================================
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--no-gravity", action="store_true")
parser.add_argument("--num-envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args, cameras=False)).app

import cables as CB  # noqa: E402
import motors as MT  # noqa: E402
from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def make_env(mode):
    cfg = CableVerticalPipeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.action_mode = mode
    cfg.gravity = not args.no_gravity
    cfg.pipe_modes = ("random",)
    cfg.sim.device = args.device
    cfg.max_episode_steps = 10 ** 6
    env = CableVerticalPipeEnv(cfg)
    env.auto_reset = False
    env.reset()
    return env


def move_pipes_away(env):
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, -3.0], device=env.device)
    pose[:, 6] = 1.0                                       # identity quaternion (x, y, z, w)
    env.pipe.write_root_pose_to_sim_index(root_pose=pose)


def settle(env, seconds=5.0):
    """Zero actions keep the motor targets; let the physics come to rest."""
    for _ in range(int(seconds / 0.02)):
        env.step(torch.zeros(env.num_envs, 7, device=env.device))
    return env.joint_q(), env.robot.data.joint_vel.torch[:, env._jchain].abs().max().item()


def bend_of(q):
    q = q.view(-1, 3, 5, 2).sum(dim=2)
    return torch.stack([q[..., 1], -q[..., 0]], dim=-1)


def deg(x):
    return np.rad2deg(x.detach().cpu().numpy())


def main():
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    screw = MT.CABLE_SCREW
    print(f"\nlead screw: lead {screw.lead * 1000:.0f} mm, lead angle {np.rad2deg(screw.lead_angle):.1f} deg, "
          f"efficiency {screw.efficiency:.2f} (back-driven {screw.back_efficiency:.2f})")
    for spec in (MT.EZM_42M, MT.EZM_56M):
        print(f"{spec.name}: holding {spec.holding_torque} N m -> max cable pull at standstill "
              f"{spec.holding_torque * 2 * np.pi * screw.efficiency / screw.lead:.0f} N")

    env = make_env("motor")
    move_pipes_away(env)
    n = env.num_envs
    model = env._cable_model_cpu
    print(f"gravity {env.cfg.gravity} (arm mass {env.gravity_model.total_mass if env.gravity_model else 0:.3f} kg), "
          f"one-sided cables {env.cfg.one_sided_cables}")

    # 1. statics
    print("\n1. statics: random motor targets, PhysX equilibrium vs cables.CableModel")
    rng = np.random.default_rng(0)
    u = torch.as_tensor(rng.uniform(-0.6, 0.6, (n, 6)) * CB.MOTOR_RANGE / (CB.MOTOR_SECTION + 1), device=env.device)
    u[0] = 0.0
    u[1] = 0.0
    u[1, 0] = -0.010                                    # M1 winds cable a in by 10 mm
    env.set_motors(slice(None), u)
    q_sim, vmax = settle(env)
    q_mod = model.equilibrium(env.motor.cpu()).to(env.device)
    err = (q_sim - q_mod).abs().max(dim=1).values
    for i in range(n):
        print(f"  env {i}: nut [mm] {env.motor[i].cpu().numpy() * 1000}\n"
              f"     bend sim   [deg] {deg(bend_of(q_sim)[i]).ravel()}\n"
              f"     bend model [deg] {deg(bend_of(q_mod)[i]).ravel()}   max |dq| {err[i]:.1e} rad")
    # slack cables leave only the backbone spring + damping (overdamped, time constant of seconds)
    check("settled after 5 s", vmax < 2e-2, f"(max |qdot| {vmax:.1e} rad/s)")
    check("PhysX statics == cable model", err.max().item() < 5e-3, f"(max |dq| {err.max().item():.1e} rad)")
    check("motor follows its target", (env.motor - u).abs().max().item() < 5e-5,
          f"(max nut error {(env.motor - u).abs().max().item() * 1e6:.1f} um)")
    # a screw at standstill holds its load with any torque in the friction band [F k eta_back, F k / eta]
    k = screw.lead / (2 * np.pi)
    f = -env.motors.load                                     # force of the nut on the cables
    tau = env.motors.torque
    lo, hi = f.abs() * k * screw.back_efficiency, f.abs() * k / screw.efficiency
    inside = ((tau * torch.sign(f) >= lo - 2e-4) & (tau * torch.sign(f) <= hi + 2e-4)) | (f.abs() < 0.05)
    check("motor torque inside the screw friction band of the cable load", bool(inside.all()),
          f"(max |torque| {tau.abs().max().item() * 1000:.1f} mN m, max |load| {f.abs().max().item():.1f} N)")

    # 2. single motors and coupling
    print("\n2. one motor at a time winds its cable in by 12 mm (the others hold)")
    for m in range(6):
        u = torch.zeros(n, 6, device=env.device, dtype=torch.float64)
        u[:, m] = -0.012
        env.set_motors(slice(None), u)
        q_sim, _ = settle(env, 2.0)
        b = deg(bend_of(q_sim)[0])
        s = CB.MOTOR_SECTION[m]
        ang = np.degrees(np.arctan2(b[s, 1], b[s, 0])) % 360
        tens = env.tension[0].cpu().numpy()
        print(f"  {CB.MOTOR_NAMES[m]} (section {s + 1}, {np.degrees(CB.MOTOR_ALPHA[m]):.0f} deg): bends "
              f"{np.linalg.norm(b, axis=1)} deg, section {s + 1} towards {ang:.0f} deg "
              f"(rigid cable: {np.degrees(0.012 / CB.CABLE_RADIUS[s]):.1f} deg) | tension a/b "
              f"{tens[2 * m]:.1f}/{tens[2 * m + 1]:.1f} N | motor {env.motors.torque[0, m].item() * 1000:+.1f} mN m")
        check(f"{CB.MOTOR_NAMES[m]} bends towards its motor angle",
              abs((ang - np.degrees(CB.MOTOR_ALPHA[m]) + 180) % 360 - 180) < 3.0)

    # 3. the lab's controller formula (Isaac Sim runs one simulation per process: same env, other mode)
    print("\n3. bend mode (controller formula, thesis eq. 3.32): commanded vs achieved bend")
    env.cfg.action_mode = "bend"
    th = np.array([0.6, 0.9, 1.2])
    for i in range(n):
        ph = 2 * np.pi * i / n
        env.bend_cmd[i] = torch.as_tensor(np.stack([th * np.cos(ph + np.arange(3)), th * np.sin(ph + np.arange(3))], 1),
                                          device=env.device)
    env.set_motors(slice(None), env.cable_model.compensate(env.bend_cmd))
    q_sim, _ = settle(env)
    ratio = (torch.linalg.norm(bend_of(q_sim), dim=-1) / torch.linalg.norm(env.bend_cmd, dim=-1)).cpu().numpy()
    print(f"  achieved / commanded bend per section (mean over {n} directions): {ratio.mean(0)} "
          f"(min {ratio.min(0)})")
    print(f"  motor torques [mN m]: max {env.motors.torque.abs().max().item() * 1000:.1f}")

    # 4. dynamics with random actions
    print("\n4. random actions for 400 steps (motor mode, pipes in place)")
    env.cfg.action_mode = "motor"
    env.reset()
    unstable, max_tau = 0, 0.0
    for _ in range(400):
        env.step(torch.rand(n, 7, device=env.device) * 2 - 1)
        qd = env.robot.data.joint_vel.torch
        unstable += int((~torch.isfinite(qd)).any() or qd.abs().max() > 1e3)
        max_tau = max(max_tau, env.motors.torque.abs().max().item())
    tens = env.tension
    check("no instability", unstable == 0)
    check("no motor alarm", not bool(env.motors.alarm.any()))
    print(f"  bends [deg]: {np.degrees(torch.linalg.norm(bend_of(env.joint_q()), dim=-1).cpu().numpy()[:3])}")
    print(f"  tensions [N]: max {tens.max().item():.1f}, slack {int((tens < 0.05).sum())}/{tens.numel()} | "
          f"max motor torque {max_tau:.3f} N m | elevator torque {env.elev_torque.cpu().numpy()[:4]} N m")

    env.reset()
    q0 = env.joint_q().clone()
    env.step(torch.zeros(n, 7, device=env.device))
    drift = (env.joint_q() - q0).abs().max().item()
    check("start state is at rest", drift < 2e-3, f"(max joint change in the first step {drift:.1e} rad)")
    print("\nsummary:", "all checks passed" if not FAILS else f"FAILED: {FAILS}")


if __name__ == "__main__":
    main()
    app.close()
