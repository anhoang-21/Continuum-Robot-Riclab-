"""
==============================================================================
pretension_study.py - which pre-tension lets the robot thread the pipe with every cable in its band? (ND3/ND4)
==============================================================================
    python pretension_study.py [--poses 400] [--out reports/pretension]

With one motor per antagonistic pair the robot has 6 actuators for 6 bend coordinates: the motor
positions of a pose are fixed (cables.CableModel.motors_for_pose), hence so are the 12 tensions. The only
free design variable is the pre-tension T0 (fixed by the tension-box springs today, motorised in the P2
upgrade of CAI_TIEN_ROBOT_K22_CHO_DE_TAI.md). For a pair with both cables taut T_a + T_b ~ 2 T0 (the cables
lengthen / shorten by the same amount), so the differential tension that holds a bend is bounded by

    |T_a - T_b| <= D_max(T0) = min(2 (T0 - T_min), 2 (T_max - T0)),

largest at T0* = (T_min + T_max) / 2. This script checks that on the poses the task needs: the S-curves
that put the tip over the pipe (kinematics.aligned_bend for pipes drawn like the task, rig + random) and
the way there (0.3 - 1.0 of the S-curve), for the nominal robot and for robots of the train uncertainty set:
for every T0 of a grid, the motor positions holding each pose, the static pose they give, the 12 tensions,
and whether every cable is inside [T_min, T_max] (and the pose reached within 1 deg per section).
Output: reports/pretension/pretension_study.json + .png. numpy / torch, no Isaac Sim.
==============================================================================
"""

import argparse
import json
import os

import numpy as np
import torch

import cables as CB
import constants as C
import uncertainty as U
from kinematics import ContinuumKinematics
from reset_logic import ResetSampler
from uncertain_cables import StaticSolver


def task_poses(kin, n, rng):
    """Bends (n, 3, 2) the insertion needs: the S-curve over the pipe, and 30-100 % of it on the way."""
    samplers = [ResetSampler(kin, m, C.PIPE_HEIGHT, (0.0, 0.17)) for m in ("rig", "random")]
    out, kinds = [], []
    for i in range(n):
        s = samplers[i % 2]
        pipe_xy, _ = s.place_pipe(rng)
        aligned = kin.aligned_bend(pipe_xy)
        frac = 1.0 if i % 4 < 2 else rng.uniform(0.3, 1.0)
        out.append(frac * aligned)
        kinds.append(("aligned" if frac == 1.0 else "partial") + "/" + s.pipe_mode)
    return np.stack(out), kinds


def evaluate(ss, geom, bends, t0, k_c, k_j, t_min, t_max):
    """Feasibility of every pose at pre-tension t0 (per cable (n, 12)) for robots k_c (n, 12), k_j (n,)."""
    q_des = torch.as_tensor(geom.q_bend(bends))
    u = ss.motors_for_pose(q_des, k_c, t0, k_j)
    q = ss.equilibrium(u, k_c, t0, k_j, q0=q_des)
    T = ss.tension(q, u, k_c, t0).numpy()
    b = q.view(-1, 3, 5, 2).sum(dim=2).numpy()
    b = np.stack([b[..., 1], -b[..., 0]], -1)
    pose_err = np.degrees(np.linalg.norm(b - bends, axis=-1).max(-1))
    in_band = (T >= t_min).all(1) & (T <= t_max).all(1)
    pair = T.reshape(-1, 6, 2)
    return {"in_band": in_band, "reached": pose_err < 1.0, "pose_err": pose_err, "t_min": T.min(1), "t_max": T.max(1),
            "diff": np.abs(pair[..., 0] - pair[..., 1]).max(1), "slack": (T < t_min).sum(1)}


def main():
    ap = argparse.ArgumentParser(description="pre-tension design study")
    ap.add_argument("--poses", type=int, default=400)
    ap.add_argument("--robots", type=int, default=4, help="robots of the train uncertainty set per pose")
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=10.0)
    ap.add_argument("--grid", type=str, default="1,1.5,2,2.5,3,3.5,4,4.5,5,5.25,5.5,6,6.5,7,7.5,8,9,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "pretension"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    kin = ContinuumKinematics()
    geom = CB.CableGeometry(kin)
    ss = StaticSolver(geom, CB.GravityModel(geom))
    bends, kinds = task_poses(kin, a.poses, rng)
    n = len(bends)
    grid = [float(v) for v in a.grid.split(",")]
    kc_nom = torch.full((n, 12), CB.CABLE_STIFFNESS, dtype=torch.float64)
    kj_nom = torch.full((n,), CB.BACKBONE_STIFFNESS, dtype=torch.float64)
    # robots of the train set (their own k_c, k_j and the relative spread of T0; T0 scaled to the grid value)
    uset = U.UncertaintySet.make("train")
    robots = [uset.sample(rng, n) for _ in range(a.robots)]
    res = {"t_min": a.t_min, "t_max": a.t_max, "grid": grid, "nominal": [], "train": [], "kinds": kinds}
    print(f"{n} task poses, tension band [{a.t_min}, {a.t_max}] N; analytic optimum T0* = {(a.t_min + a.t_max) / 2:.2f} N")
    print(f"{'T0':>5} | {'nominal: in band':>16} {'reached':>8} {'slack cables':>12} {'T_max p95':>9} | train robots: in band | D_max")
    for t0 in grid:
        r = evaluate(ss, geom, bends, torch.full((n, 12), t0, dtype=torch.float64), kc_nom, t0 * 0 + kj_nom,
                     a.t_min, a.t_max)
        ok_train = []
        for p in robots:
            t0_rob = torch.as_tensor(p["t0"] / p["t0"].mean(1, keepdims=True) * t0)
            rr = evaluate(ss, geom, bends, t0_rob, torch.as_tensor(p["k_c"]), torch.as_tensor(p["k_j"]), a.t_min, a.t_max)
            ok_train.append(rr["in_band"] & rr["reached"])
        ok_train = np.concatenate(ok_train)
        d_max = min(2 * (t0 - a.t_min), 2 * (a.t_max - t0))
        rec = {"t0": t0, "in_band": float(np.mean(r["in_band"] & r["reached"])), "reached": float(np.mean(r["reached"])),
               "slack_mean": float(r["slack"].mean()), "t_max_p95": float(np.percentile(r["t_max"], 95)),
               "diff_p95": float(np.percentile(r["diff"], 95)), "d_max": d_max,
               "by_kind": {k: float(np.mean((r["in_band"] & r["reached"])[np.array(kinds) == k])) for k in sorted(set(kinds))}}
        res["nominal"].append(rec)
        res["train"].append({"t0": t0, "in_band": float(ok_train.mean())})
        print(f"{t0:5.2f} | {100 * rec['in_band']:15.1f}% {100 * rec['reached']:7.1f}% {rec['slack_mean']:12.2f} "
              f"{rec['t_max_p95']:8.1f} N | {100 * ok_train.mean():19.1f}% | {d_max:4.1f} N "
              f"(needed p95 {rec['diff_p95']:.1f} N)")
    best = max(res["nominal"], key=lambda r: r["in_band"])
    print(f"\nbest pre-tension on the grid: {best['t0']} N ({100 * best['in_band']:.1f}% of the task poses inside the band); "
          f"today's 3 N: {100 * [r for r in res['nominal'] if r['t0'] == 3.0][0]['in_band']:.1f}%")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pretension_study.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        g = np.array(grid)
        ax[0].plot(g, [100 * r["in_band"] for r in res["nominal"]], "o-", label="nominal robot")
        ax[0].plot(g, [100 * r["in_band"] for r in res["train"]], "s--", label="robots of the train set")
        ax[0].axvline((a.t_min + a.t_max) / 2, color="gray", ls=":", label="T0* = (T_min + T_max) / 2")
        ax[0].axvline(3.0, color="tab:red", ls=":", label="today (3 N)")
        ax[0].set_xlabel("pre-tension T0 (N)")
        ax[0].set_ylabel("task poses with every cable in the band (%)")
        ax[0].legend(fontsize=8)
        ax[1].plot(g, [r["d_max"] for r in res["nominal"]], "k-", label="D_max(T0) (both cables taut)")
        ax[1].plot(g, [r["diff_p95"] for r in res["nominal"]], "o-", label="pair differential the poses need (p95)")
        ax[1].set_xlabel("pre-tension T0 (N)")
        ax[1].set_ylabel("tension (N)")
        ax[1].legend(fontsize=8)
        fig.suptitle(f"Pre-tension design: tension band [{a.t_min}, {a.t_max}] N, {n} insertion poses")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "pretension_study.png"), dpi=130)
        print(f"figure: {os.path.join(a.out, 'pretension_study.png')}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
