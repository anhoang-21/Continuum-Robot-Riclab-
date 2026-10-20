"""
==============================================================================
plot_sysid.py - figure of the sensor-set study of sysid.py (ND1): identification error per sensor set
==============================================================================
    python plot_sysid.py [reports/sysid/sysid_results.json] [--out reports/sysid/sysid_sensors.png]

One panel per parameter group (cable stiffness, pre-tension, backbone spring, capstan friction, backlash):
the error of every robot's fit (dots), the median over the robots (bar) and the median 95 % half-width the
fit claims (black tick). Where the tick is far below the bar, the error is model mismatch, not noise.
Also prints the table of the README. matplotlib only, no Isaac Sim.
==============================================================================
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SETS = ("encoders+imu", "encoders+mocap", "encoders+loadcell", "encoders+mocap+loadcell")
LABELS = ("IMU", "mocap", "load\ncells", "mocap +\nload cells")   # all with the 6 motor encoders
PARAMS = (("k_c", "cable stiffness k_c", "relative error (%)", 100.0, "rel"),
          ("T0", "pre-tension T0", "relative error (%)", 100.0, "rel"),
          ("k_j", "backbone spring k_j", "relative error (%)", 100.0, "rel"),
          ("mu_theta", "capstan friction mu theta", "absolute error (-)", 1.0, "abs"),
          ("backlash", "nut backlash", "absolute error (mm)", 1.0, "abs"))
COLORS = ("#9aa5b1", "#5b8def", "#f0a030", "#2ca25f")


def errors(results, s, p, scale, kind):
    fits = results.get(s, [])
    e = np.array([f["errors"][p]["rel_err" if kind == "rel" else "abs_err"] for f in fits]) * scale
    c = np.array([f["errors"][p]["rel_ci_half" if kind == "rel" else "ci_half"] for f in fits]) * scale
    return e, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=os.path.join(TASK_DIR, "reports", "sysid", "sysid_results.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    with open(a.results, encoding="utf-8") as f:
        res = json.load(f)["results"]
    out = a.out or os.path.join(os.path.dirname(a.results), "sysid_sensors.png")
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(19, 4.6))
    rng = np.random.default_rng(0)
    print("| sensors | " + " | ".join(p[1] for p in PARAMS) + " |")
    print("|---|" + "---|" * len(PARAMS))
    rows = {s: [] for s in SETS}
    for ax, (p, title, ylabel, scale, kind) in zip(axes, PARAMS):
        for k, s in enumerate(SETS):
            e, c = errors(res, s, p, scale, kind)
            if not len(e):
                continue
            med = float(np.median(e))
            ax.bar(k, med, color=COLORS[k], width=0.7, alpha=0.85)
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(e)), e, s=14, color="k", alpha=0.55, zorder=3)
            ax.plot([k - 0.3, k + 0.3], [np.median(c)] * 2, color="k", lw=2.5, zorder=4)
            unit = "%" if kind == "rel" else (" mm" if p == "backlash" else "")
            rows[s].append(f"{med:.1f}{unit} (+-{np.median(c):.1f})" if kind == "rel"
                           else f"{med:.3f}{unit} (+-{np.median(c):.3f})")
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    for s in SETS:
        print(f"| {s} | " + " | ".join(rows[s]) + " |")
    fig.suptitle("Identification on 8 hidden robots of the train set: error per sensor set, all with the motor encoders\n"
                 "(bar: median, dots: robots, black tick: median 95 % half-width claimed by the fit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
