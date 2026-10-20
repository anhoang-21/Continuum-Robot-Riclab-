"""
==============================================================================
plot_safebench.py - figures and a Markdown table from K22-SafeBench reports (no Isaac Sim)
==============================================================================
    python plot_safebench.py reports/safebench/stage6.json [more.json ...] [--out reports/safebench/stage6]

Merges the cells of the given reports (benchmark_safety.py) and draws:
  success.png   success rate (Wilson 95 % interval) per controller and uncertainty level
  safety.png    share of steps with a cable outside [T_min, T_max] (slack / over-tension stacked)
  pareto.png    success vs tension violations, one point per controller and level
and prints the Markdown table used in the README.
==============================================================================
"""

import argparse
import json
import os

import numpy as np


def load(paths):
    cells = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        for c in rep["cells"]:
            c = dict(c)
            c.pop("episodes", None)
            cells.append(c)
    return cells


def table(cells):
    lines = ["| controller | robots | success (95 % CI) | time | steps outside the band (slack / over) | "
             "episodes with a violation | T_max p95 | pipe force p95 | CBF active / infeasible |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        a = c["summary"]["all"]
        lo, hi = a["success_ci"]
        level = c["level"] + (f" / {c['stress']}" if c.get("stress") else "")
        meta = c.get("meta", {})
        cbf = (f"{100 * meta.get('filter_active_frac', 0):.0f} % / {100 * meta.get('filter_infeasible_frac', 0):.0f} %"
               if "+" in c["controller"] or "cbf" in c["controller"] else "-")
        lines.append(f"| {c['controller']} | {level} | {100 * a['success']:.1f} % ({100 * lo:.0f}-{100 * hi:.0f}) | "
                     f"{a['time_s']:.2f} s | {100 * a['viol_frac']:.1f} % ({100 * a['slack_frac']:.1f} / "
                     f"{100 * a['over_frac']:.1f}) | {100 * a['episodes_with_violation']:.0f} % | {a['t_max_p95']:.1f} N | "
                     f"{a['f_max_p95']:.2f} N | {cbf} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    cells = load(a.reports)
    print(table(cells))
    out = a.out or os.path.splitext(a.reports[0])[0]
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctrls = list(dict.fromkeys(c["controller"] for c in cells))
    levels = list(dict.fromkeys(c["level"] + (f"/{c['stress']}" if c.get("stress") else "") for c in cells))
    get = {(c["controller"], c["level"] + (f"/{c['stress']}" if c.get("stress") else "")): c["summary"]["all"] for c in cells}
    w = 0.8 / max(len(ctrls), 1)
    x = np.arange(len(levels))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(levels) * max(1, len(ctrls) / 3), 4))
    for i, cname in enumerate(ctrls):
        vals = [get.get((cname, lv)) for lv in levels]
        s = np.array([100 * v["success"] if v else np.nan for v in vals])
        err = np.array([[100 * (v["success"] - v["success_ci"][0]), 100 * (v["success_ci"][1] - v["success"])] if v
                        else [0, 0] for v in vals]).T
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, s, w, yerr=err, capsize=2, label=cname, color=colors[i % len(colors)])
    ax.set_xticks(x, levels)
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Pipe insertion: success on uncertain robots")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(levels) * max(1, len(ctrls) / 3), 4))
    for i, cname in enumerate(ctrls):
        vals = [get.get((cname, lv)) for lv in levels]
        sl = np.array([100 * v["slack_frac"] if v else np.nan for v in vals])
        ov = np.array([100 * v["over_frac"] if v else np.nan for v in vals])
        xs = x + (i - (len(ctrls) - 1) / 2) * w
        ax.bar(xs, sl, w, color=colors[i % len(colors)], label=f"{cname}: slack")
        ax.bar(xs, ov, w, bottom=sl, color=colors[i % len(colors)], alpha=0.45, hatch="//")
    ax.set_xticks(x, levels)
    ax.set_ylabel("steps with a cable outside the band (%)")
    ax.set_title("Tension safety (solid: slack < T_min, hatched: over-tension > T_max)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "safety.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, cname in enumerate(ctrls):
        for j, lv in enumerate(levels):
            v = get.get((cname, lv))
            if v:
                ax.scatter(100 * v["viol_frac"], 100 * v["success"], color=colors[i % len(colors)],
                           marker=markers[j % len(markers)], s=50, label=cname if j == 0 else None)
    for j, lv in enumerate(levels):
        ax.scatter([], [], color="gray", marker=markers[j % len(markers)], label=f"robots: {lv}")
    ax.set_xlabel("steps with a cable outside [T_min, T_max] (%)")
    ax.set_ylabel("success (%)")
    ax.set_title("Success vs safety (top left is best)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pareto.png"), dpi=130)
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as f:
        f.write(table(cells) + "\n")
    print(f"\nfigures + table.md -> {out}")


if __name__ == "__main__":
    main()
