"""
==============================================================================
play_shape.py - shape tracking on uncertain robots: model-based controllers, with / without the CBF (ND2)
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe play_shape.py --levels nominal,train --robots 32

Every robot (robust_env.RobustCableEnv, pipes parked) tracks the reference bends of shape_control.reference:
"scurve" (form the S-curve that puts the tip over a random pipe in 2 s, hold, straighten), "circle" (bend
directions turning at 0.25 Hz), "steps"; with --payload-step a 50 g mass is hung on the tip at t = 5 s.
The controllers give motor targets; the env moves the motors towards them at the motors' rate limit, and
with "+cbf" the tension CBF (safety_filter.TensionCBF) filters every motor step. The shape sensor is the
true section bends + mocap noise (sigma 0.5 mm over ~15 cm per section -> 0.0033 rad), the load cells the
nut-side tensions + 0.1 N noise.

Metrics per cell (same robots / references for all controllers): RMS / max section-bend error (deg), RMS
tip-position error (mm, against the tip of the reference shape), fraction of steps with a cable outside
[T_min, T_max], highest tension. Output: reports/shape/<tag>.json, a printed table.
==============================================================================
"""

import argparse
import json
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

parser = argparse.ArgumentParser(description="Shape tracking with model-based controllers on uncertain robots")
parser.add_argument("--controllers", type=str, default="formula,statics,statics+PI,adaptive,adaptive+loadcell,statics+PI+cbf")
parser.add_argument("--levels", type=str, default="nominal,train")
parser.add_argument("--trajectories", type=str, default="scurve,circle")
parser.add_argument("--robots", type=int, default=32)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--payload-step", action="store_true", help="hang 50 g on the tip at t = 5 s")
parser.add_argument("--pretension", type=float, default=None)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--tag", type=str, default="shape")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args, cameras=False)).app

import cables as CB  # noqa: E402
import constants as C  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
import uncertainty as U  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402
from safety_filter import TensionCBF  # noqa: E402
from shape_control import (AdaptiveController, FormulaController, StaticsController, StaticsPIController,  # noqa: E402
                           reference)

SIGMA_B = 0.5e-3 / 0.15


def make_controller(name, env):
    base = name.replace("+cbf", "")
    kw = dict(t0=env.cfg.cable_pretension)
    n = env.num_envs
    if base == "formula":
        c = FormulaController(env.geom, env.statics, n, **kw)
    elif base == "statics":
        c = StaticsController(env.geom, env.statics, n, **kw)
    elif base == "statics+PI":
        c = StaticsPIController(env.geom, env.statics, n, **kw)
    elif base == "adaptive":
        c = AdaptiveController(env.geom, env.statics, n, use_tension=False, **kw)
    elif base == "adaptive+loadcell":
        c = AdaptiveController(env.geom, env.statics, n, use_tension=True, **kw)
    else:
        raise ValueError(name)
    c.name = name
    c.cbf = TensionCBF(env, source="loadcell") if name.endswith("+cbf") else None
    return c


def tip_ref(kin, bends):
    """Tip of the reference shapes (n, 3) relative to Seg1 (kinematics of the bend vectors)."""
    xpos, xmat = kin.forward(kin.qpos_from(bends, 0.0))
    tip = xpos[:, kin.tip_id] + C.TIP_OFFSET * xmat[:, kin.tip_id, :, 2]
    return tip - xpos[:, kin.seg_ids[0]]


def tip_sim(env):
    """Tip of the simulated robots (n, 3) relative to Seg1."""
    pos = env.robot.data.body_link_pos_w.torch.double()
    rot = matrix_from_quat(env.robot.data.body_link_quat_w.torch[:, env._tip_body].double())
    tip = pos[:, env._tip_body] + C.TIP_OFFSET * rot[:, :, 2]
    return (tip - pos[:, env._seg_ids[0]]).cpu().numpy()


def run_cell(env, ctrl, level, traj, goals):
    n, dev = env.num_envs, env.device
    env.set_uncertainty(level)
    env._prm_rngs = [np.random.default_rng(args.seed + 5000 + i) for i in range(n)]
    env._rngs = [np.random.default_rng(args.seed + i) for i in range(n)]
    env.safety_filter = None
    env.reset()
    env.park_pipes()
    env.straighten()
    zero = torch.zeros(n, 7, device=dev)
    for _ in range(25):
        env.step(zero)
    env.safety_filter = ctrl.cbf
    if ctrl.cbf is not None:
        ctrl.cbf.reset()
    ctrl.reset()
    kin = env.kin
    steps = int(args.seconds / env.step_dt)
    u_rate = torch.as_tensor(CB.MOTOR_RATE, device=dev, dtype=torch.float64)
    rng = np.random.default_rng(args.seed + 99)
    err_b, err_tip, viol, t_hi = [], [], [], []
    t0 = time.time()

    for k in range(steps):
        t = k * env.step_dt
        if args.payload_step and abs(t - 5.0) < 1e-9:
            env.set_payload(range(n), 0.050)
        b_ref = reference(traj, t, goals)
        b_true = env.measured_bend().cpu().numpy()
        b_meas = b_true + rng.normal(0.0, SIGMA_B, b_true.shape)
        t_meas = env.ucm.T_n.cpu().numpy() + rng.normal(0.0, 0.1, (n, 12))
        u_now = env.motor.cpu()
        if isinstance(ctrl, AdaptiveController):
            u_tgt = ctrl(b_ref, b_meas, u_now, t_meas=t_meas)
        else:
            u_tgt = ctrl(b_ref, b_meas, u_now)
        act = torch.zeros(n, 7, device=dev)
        act[:, :6] = torch.clamp((u_tgt.to(dev) - env.motor_cmd) / u_rate, -1.0, 1.0).float()
        env.step(act)
        # errors after the step
        b_true = env.measured_bend().cpu().numpy()
        err_b.append(np.degrees(np.linalg.norm(b_true - b_ref, axis=-1)))                     # (n, 3)
        err_tip.append(np.linalg.norm(tip_sim(env) - tip_ref(kin, b_ref), axis=-1) * 1e3)

        T_r, T_n = env.tension.cpu().numpy(), env.ucm.T_n.cpu().numpy()
        viol.append(((T_r < env.cfg.t_min) | (np.maximum(T_r, T_n) > env.cfg.t_max)).any(1))
        t_hi.append(np.maximum(T_r, T_n).max(1))
    err_b, err_tip, viol, t_hi = map(np.array, (err_b, err_tip, viol, t_hi))
    if args.payload_step:
        env.set_payload(range(n), 0.0)
    settle = int(0.5 / env.step_dt)                                    # skip the first 0.5 s
    return {"bend_rms_deg": float(np.sqrt((err_b[settle:] ** 2).mean())), "bend_max_deg": float(err_b.max()),
            "tip_rms_mm": float(np.sqrt((err_tip[settle:] ** 2).mean())), "tip_p95_mm": float(np.percentile(err_tip, 95)),
            "viol_frac": float(viol.mean()), "t_max_p95": float(np.percentile(t_hi.max(0), 95)),
            "per_robot_tip_rms": np.sqrt((err_tip[settle:] ** 2).mean(0)).tolist(), "seconds": time.time() - t0}


def main():
    cfg = RobustCableEnvCfg()
    cfg.scene.num_envs = args.robots
    cfg.pipe_modes = ("random",)
    cfg.max_episode_steps = 10 ** 7
    cfg.seed = args.seed
    if args.pretension is not None:
        cfg.cable_pretension = args.pretension
    cfg.sim.device = args.device
    env = RobustCableEnv(cfg)
    env.auto_reset = False
    env.randomized = True
    rng = np.random.default_rng(args.seed)
    from reset_logic import ResetSampler
    sampler = ResetSampler(env.kin, "random", C.PIPE_HEIGHT, (0.05, 0.17))
    goals = np.stack([env.kin.aligned_bend(sampler.place_pipe(rng)[0]) for _ in range(env.num_envs)])
    report = {"args": vars(args), "cells": []}
    out_dir = os.path.join(TASK_DIR, "reports", "shape")
    os.makedirs(out_dir, exist_ok=True)
    for level in args.levels.split(","):
        for traj in args.trajectories.split(","):
            for name in args.controllers.split(","):
                ctrl = make_controller(name, env)
                r = run_cell(env, ctrl, level, traj, goals)
                report["cells"].append({"level": level, "trajectory": traj, "controller": name, **r})
                print(f"[{level}/{traj}{'/payload' if args.payload_step else ''}] {name:>20}: bend RMS {r['bend_rms_deg']:5.2f} deg "
                      f"(max {r['bend_max_deg']:5.1f}) | tip RMS {r['tip_rms_mm']:5.1f} mm (p95 {r['tip_p95_mm']:5.1f}) | "
                      f"tension outside the band {100 * r['viol_frac']:5.1f}% | T_max p95 {r['t_max_p95']:4.1f} N | "
                      f"{r['seconds']:.0f} s", flush=True)
                with open(os.path.join(out_dir, f"{args.tag}.json"), "w", encoding="utf-8") as f:
                    json.dump(report, f)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
