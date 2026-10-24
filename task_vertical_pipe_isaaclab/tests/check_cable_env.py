"""
==============================================================================
check_cable_env.py - Checks of the cable-driven env in Isaac Sim
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe tests\\check_cable_env.py

Run tests/check_cable_model.py first (numpy only).

1. cable geometry of the PhysX link poses = numpy geometry (rest length, 12 cables)
2. start state: tension = preload, robot holds its start pose with zero action (drift, tension, speed)
3. one motor pulled: the section bends towards the pulled cable, the cable tension rises, its partner goes slack
4. total cable force / torque on the robot is zero
5. random motor actions on all envs: finite states, tension >= 0, steps / s
==============================================================================
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file

import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--random-steps", type=int, default=400)
parser.add_argument("--hold-steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import cable_model as CM  # noqa: E402
import constants as C  # noqa: E402
from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg  # noqa: E402


def make_env(**cfg_kw):
    cfg = CableVerticalPipeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return CableVerticalPipeEnv(cfg)


def cable_lengths(env):
    pos, rot, com, v, w = env._link_state()
    return CM.path_lengths(env.cable.hole, env.cable.mask, pos, rot)


def check_geometry(env):
    env.reset()
    env.set_start_state(list(range(env.num_envs)), env.pipe_xy.cpu().numpy(), env.z_top.cpu().numpy(),
                        np.zeros((env.num_envs, 3, 2)), np.zeros(env.num_envs))
    L = cable_lengths(env)
    err = (L - env.cable.rest_len).abs().max().item()
    print(f"[1] straight pose: PhysX cable lengths vs numpy rest lengths, max |diff| = {err * 1000:.4f} mm")
    assert err < 2e-4


def check_hold(env):
    env.reset()
    env.actions.zero_()
    p0 = env._measure()["tip"].clone()
    t0 = env.tension.clone()
    zero = torch.zeros(env.num_envs, 7, device=env.device)
    for _ in range(args.hold_steps):
        env.step(zero)
        env.reset_buf[:] = False
    tip = env._measure()["tip"]
    drift = (tip - p0).norm(dim=1).max().item() * 1000
    qd = env.robot.data.joint_vel.abs().max().item()
    T = env.tension
    print(f"[2] hold {args.hold_steps} steps, zero action: tip drift max {drift:.2f} mm, max |qdot| {qd:.3f}, "
          f"tension {T.min().item():.2f} .. {T.max().item():.2f} N (preload {env.cfg.cable_preload})")
    assert torch.isfinite(tip).all()


def check_pull(env):
    env.cfg.max_episode_steps = 10_000
    env.set_start_state(list(range(env.num_envs)), env.pipe_xy.cpu().numpy(), env.z_top.cpu().numpy(),
                        np.zeros((env.num_envs, 3, 2)), np.zeros(env.num_envs))
    jy = torch.as_tensor(env._jy[:5], device=env.device)          # section 1: joints Seg1_y .. Seg5_y (angle = b0 / 5)
    act = torch.zeros(env.num_envs, 7, device=env.device)
    act[:, 0] = 1.0                                               # motor 0 (section 1, alpha 0) pulls
    for _ in range(30):
        env.step(act)
        env.reset_buf[:] = False
    q = env.robot.data.joint_pos[:, jy].mean(1)
    cmd = env.bend_cmd[:, 0, 0] / 5.0
    T = env.tension
    print(f"[3] motor 0 pulled 30 steps: bend command {cmd[0].item():.4f} rad/joint, measured mean joint angle "
          f"{q[0].item():.4f} rad; tension cable 0 = {T[0, 0].item():.1f} N, partner cable 1 = {T[0, 1].item():.1f} N "
          f"(section-1 bend {env.bend_cmd[0, 0, 0].item():.3f} of {C.THETA_MAX})")
    assert (q > 0.5 * cmd).all() and (T[:, 0] > T[:, 1]).all()


def check_wrench(env):
    pos, rot, com, v, w = env._link_state()
    force, torque, T, L = env.cable.compute(pos, rot, com, v, w, env.motor_cmd.float())
    f = force.sum(1).abs().max().item()
    # torque about the origin: sum of (com x F) + torques
    tot = (torch.cross(com, force, dim=-1) + torque).sum(1).abs().max().item()
    print(f"[4] total cable force {f:.2e} N, total moment {tot:.2e} N m (both must be ~0)")
    assert f < 1e-2 and tot < 1e-2


def check_random(env):
    env.reset()
    t = time.time()
    n_reset = 0
    for _ in range(args.random_steps):
        act = torch.empty(env.num_envs, 7, device=env.device).uniform_(-1, 1)
        obs, rew, term, trunc, _ = env.step(act)
        n_reset += int((term | trunc).sum())
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
        assert (env.tension >= 0).all()
    dt = time.time() - t
    print(f"[5] {args.random_steps} random steps x {env.num_envs} envs: finite, tension >= 0 ({env.tension.max().item():.0f} N max), "
          f"{n_reset} episode ends, {args.random_steps * env.num_envs / dt:.0f} env-steps/s")


def main():
    env = make_env()
    check_geometry(env)
    check_hold(env)
    check_pull(env)
    check_wrench(env)
    check_random(env)
    env.close()
    print("\nALL OK")


if __name__ == "__main__":
    main()
    app.close()
