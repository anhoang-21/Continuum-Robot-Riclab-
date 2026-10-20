"""
==============================================================================
benchmark_safety.py - K22-SafeBench: controllers x uncertainty levels on the cable-driven robot (ND5)
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe benchmark_safety.py \\
        --policy ppo_nominal=models/ppo_cable_128/best_model.pt \\
        --policy ppo_dr=models/ppo_dr/best_model.pt --policy ppo_dr_cbf=models/ppo_dr/best_model.pt+cbf \\
        --levels nominal,train,ood --episodes 100 --tag stage6

Every (controller, level) cell runs on the SAME robots and scenes: the env's random streams (start state
and physical parameters) are re-seeded per cell and every env runs a fixed number of episodes, so env i's
k-th episode is the same robot / pipe for all controllers. Isaac Sim runs one simulation per process: the
levels are switched on the same env (robust_env.RobustCableEnv draws the parameters at every reset).

Per episode: outcome, length, wall-contact steps, the robot's parameters, and the safety statistics of
robust_env.py (fraction of steps with a cable outside [T_min, T_max], slack / over-tension steps, extremes
of the tension and of the force on the pipe). Rates come with Wilson 95 % intervals.
Output: reports/safebench/<tag>.json (all episodes + summary), a printed table. plot_safebench.py draws it.

Stress tests (--stress): a 2 N push on the tip for 0.2 s at a random step of every episode (push), the cable
stiffness of every robot divided by 2 (creep), the backlash of one motor x3 (backlash).
==============================================================================
"""

import argparse
import json
import math
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

parser = argparse.ArgumentParser(description="K22-SafeBench: success and safety of controllers under uncertainty")
parser.add_argument("--policy", action="append", default=[],
                    help="NAME=PATH[+cbf|+cbf-model|+cbf-oracle]; repeatable. +cbf: the tension CBF filter "
                         "(safety_filter.py) on top, with load-cell / model / true tensions")
parser.add_argument("--levels", type=str, default="nominal,train,ood")
parser.add_argument("--uncertainty-json", type=str, default="", help="identified ranges as an extra level 'identified'")
parser.add_argument("--pipe-mode", choices=["rig", "random", "both"], default="both")
parser.add_argument("--episodes", type=int, default=100, help="episodes per pipe mode and cell (rounded up to the envs)")
parser.add_argument("--num-envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=9000)
parser.add_argument("--max-steps", type=int, default=400)
parser.add_argument("--t-min", type=float, default=0.5)
parser.add_argument("--t-max", type=float, default=10.0)
parser.add_argument("--f-max", type=float, default=2.0)
parser.add_argument("--pretension", type=float, default=None, help="nominal pre-tension T0 (N)")
parser.add_argument("--stress", type=str, default="", help="comma list of push,creep,backlash (one cell per stress)")
parser.add_argument("--tag", type=str, default="safebench")
parser.add_argument("--out-dir", type=str, default=os.path.join(TASK_DIR, "reports", "safebench"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args, cameras=False)).app

import uncertainty as U  # noqa: E402
from controllers import load_controller  # noqa: E402
from pipe_runner import OUTCOMES, SAFETY_KEYS, episode_safety  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402

PRM_SUMMARY = ("k_c", "t0", "k_j", "mu_theta", "backlash", "payload", "latency")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def make_env():
    cfg = RobustCableEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.pipe_modes = ("rig", "random") if args.pipe_mode == "both" else (args.pipe_mode,)
    cfg.max_episode_steps = args.max_steps
    cfg.assist_prob = 0.0
    cfg.seed = args.seed
    cfg.t_min, cfg.t_max, cfg.f_max = args.t_min, args.t_max, args.f_max
    cfg.obs_tension = True
    cfg.priv_obs = True
    if args.pretension is not None:
        cfg.cable_pretension = args.pretension
    cfg.sim.device = args.device
    env = RobustCableEnv(cfg)
    env.randomized = True
    return env


def make_controller(spec, env):
    """NAME=PATH[+cbf | +cbf-model | +cbf-oracle]: the policy, optionally behind the tension CBF (estimate source)."""
    name, _, path = spec.partition("=")
    path, _, filt = path.partition("+")
    ctrl = load_controller(path if os.path.isabs(path) else os.path.join(TASK_DIR, path), name=name)
    if filt:
        from safety_filter import FilteredController, TensionCBF
        source = filt.split("-")[1] if "-" in filt else "loadcell"
        ctrl = FilteredController(ctrl, TensionCBF(env, source=source))
        env.safety_filter = None
    ctrl.name = name
    return ctrl


def set_level(env, level):
    if level == "identified":
        env.set_uncertainty("train", args.uncertainty_json)
    else:
        env.set_uncertainty(level)


def apply_stress(env, stress, prm):
    """Change the drawn parameters of a stress cell (in place) before they are written to the robots."""
    if stress == "creep":
        prm["k_c"] *= 0.5
    elif stress == "backlash":
        prm["backlash"][:, 0] = np.maximum(prm["backlash"][:, 0], 0.2e-3) * 3.0


def run_cell(env, ctrl, level, stress=""):
    n = env.num_envs
    modes = env.env_modes
    per_mode = {m: sum(1 for x in modes if x == m) for m in set(modes)}
    quota = {m: math.ceil(args.episodes / per_mode[m]) for m in per_mode}
    set_level(env, level)
    env._rngs = [np.random.default_rng(args.seed + i) for i in range(n)]
    env._prm_rngs = [np.random.default_rng(args.seed + 5000 + i) for i in range(n)]
    if stress in ("creep", "backlash"):
        draw = env._draw_params

        def stressed(ids):
            p = draw(ids)
            apply_stress(env, stress, p)
            return p
        env._draw_params = stressed
    push_rng = np.random.default_rng(args.seed + 77)
    env.safety_filter = getattr(ctrl, "cbf", None)
    ctrl.reset()
    if hasattr(env, "_act_buf"):
        env._act_buf.zero_()
    obs, _ = env.reset()
    cur = {k: np.array(env.prm[k]).copy() for k in PRM_SUMMARY}
    counts = np.zeros(n, dtype=int)
    target = np.array([quota[modes[i]] for i in range(n)])
    push_at = push_rng.integers(10, 80, n) if stress == "push" else None
    ep_len = np.zeros(n, dtype=int)
    episodes, filt_steps, infeas_steps, total_steps = [], 0, 0, 0
    t0 = time.time()
    while (counts < target).any():
        act = ctrl(obs, env)
        if stress == "push":
            hit = np.nonzero(ep_len == push_at)[0]
            if len(hit):
                ang = push_rng.uniform(0, 2 * np.pi, len(hit))
                f = np.stack([2.0 * np.cos(ang), 2.0 * np.sin(ang), np.zeros(len(hit))], 1)
                env.push(hit.tolist(), f, 10)
        obs, _, _, _, ex = env.step(act)
        ep_len += 1
        total_steps += n
        if env.safety_filter is not None:
            filt_steps += int(env.safety_filter.last_active.sum())
            infeas_steps += int((env.safety_filter.last_slack > 0.05).sum())
        info = ex["done_info"]
        oc = info["outcome"].cpu().numpy()
        ended = np.nonzero(oc >= 0)[0]
        if len(ended):
            safe = episode_safety(info, torch.as_tensor(ended, device=info["outcome"].device))
            for j, i in enumerate(ended):
                if counts[i] < target[i]:
                    rec = {"env": int(i), "k": int(counts[i]), "mode": modes[i], "outcome": OUTCOMES[oc[i]],
                           "length": int(info["episode_length"][i]), "contact_steps": int(info["contact_steps"][i]),
                           "params": {k: float(np.mean(cur[k][i])) for k in PRM_SUMMARY}}
                    rec.update({k: float(safe[k][j]) for k in SAFETY_KEYS})
                    episodes.append(rec)
                    counts[i] += 1
                for k in PRM_SUMMARY:
                    cur[k][i] = env.prm[k][i]
                ep_len[i] = 0
                if push_at is not None:
                    push_at[i] = push_rng.integers(10, 80)
            ctrl.reset(ended.tolist())
    if stress in ("creep", "backlash"):
        del env._draw_params
    env.safety_filter = None
    return episodes, {"wall_s": time.time() - t0, "filter_active_frac": filt_steps / max(total_steps, 1),
                      "filter_infeasible_frac": infeas_steps / max(total_steps, 1)}


def summarize(episodes):
    out = {}
    for mode in sorted({e["mode"] for e in episodes}) + ["all"]:
        eps = [e for e in episodes if mode == "all" or e["mode"] == mode]
        n = len(eps)
        k = sum(e["outcome"] == "success" for e in eps)
        ok = [e for e in eps if e["outcome"] == "success"]
        fails = {}
        for e in eps:
            if e["outcome"] != "success":
                fails[e["outcome"]] = fails.get(e["outcome"], 0) + 1
        s = {"n": n, "success": k / n if n else float("nan"), "success_ci": wilson(k, n), "failures": fails,
             "time_s": float(np.mean([e["length"] for e in ok]) * 0.02) if ok else float("nan"),
             "contact_free": float(np.mean([e["contact_steps"] == 0 for e in eps])) if n else float("nan")}
        for key in ("viol_frac", "slack_frac", "over_frac", "contact_viol_frac", "cost_slack", "cost_over", "cost_contact"):
            s[key] = float(np.mean([e[key] for e in eps]))
        s["episodes_with_violation"] = float(np.mean([e["viol_frac"] > 0 for e in eps]))
        s["t_max_mean"] = float(np.mean([e["t_max"] for e in eps]))
        s["t_max_p95"] = float(np.percentile([e["t_max"] for e in eps], 95))
        s["t_min_p05"] = float(np.percentile([e["t_min"] for e in eps], 5))
        s["f_max_p95"] = float(np.percentile([e["f_max"] for e in eps], 95))
        out[mode] = s
    return out


def main():
    if not args.policy:
        raise SystemExit("give at least one --policy NAME=PATH")
    env = make_env()
    ctrls = [make_controller(spec, env) for spec in args.policy]
    levels = args.levels.split(",") + (["identified"] if args.uncertainty_json else [])
    stresses = [s for s in args.stress.split(",") if s]
    cells = [(lv, "") for lv in levels] + [("train", s) for s in stresses]
    os.makedirs(args.out_dir, exist_ok=True)
    report = {"args": vars(args), "cells": []}
    print(f"\nK22-SafeBench: {len(ctrls)} controllers x {len(cells)} cells, {args.episodes} episodes per pipe mode, "
          f"{args.num_envs} envs, seed {args.seed} | tension band [{args.t_min}, {args.t_max}] N, pipe force <= {args.f_max} N")
    for level, stress in cells:
        for ctrl in ctrls:
            episodes, meta = run_cell(env, ctrl, level, stress)
            summ = summarize(episodes)
            report["cells"].append({"controller": ctrl.name, "level": level, "stress": stress,
                                    "summary": summ, "meta": meta, "episodes": episodes})
            a = summ["all"]
            lo, hi = a["success_ci"]
            per_mode = " | ".join(f"{m} {100 * summ[m]['success']:.0f}%" for m in summ if m != "all")
            print(f"[{level}{'/' + stress if stress else ''}] {ctrl.name:>18}: success {100 * a['success']:5.1f}% "
                  f"[{100 * lo:.0f}, {100 * hi:.0f}] ({per_mode}) | time {a['time_s']:.2f} s | tension outside the band "
                  f"{100 * a['viol_frac']:.1f}% of steps (slack {100 * a['slack_frac']:.1f}, over {100 * a['over_frac']:.1f}), "
                  f"in {100 * a['episodes_with_violation']:.0f}% of episodes | T_max p95 {a['t_max_p95']:.1f} N | "
                  f"pipe force p95 {a['f_max_p95']:.2f} N | fails {a['failures']} | {meta['wall_s']:.0f} s", flush=True)
            path = os.path.join(args.out_dir, f"{args.tag}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f)
    print(f"\nreport: {path}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
