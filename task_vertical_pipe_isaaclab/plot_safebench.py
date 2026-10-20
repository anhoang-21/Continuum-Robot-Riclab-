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
             "episodes with a violation | T_max p95 | steps pushing the pipe > F_max | pipe force p95 | "
             "CBF active / infeasible |",
             "|---|---|---|---|---|---|---|---|---|---|"]
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
                     f"{100 * a['contact_viol_frac']:.2f} % | {a['f_max_p95']:.1f} N | {cbf} |")
    return "\n".join(lines)


def role(name):
    """Controller name without the pre-tension tag (lag_t525+cbf -> lag+cbf): same colour in every panel."""
    import re

    return re.sub(r"_t(3|525|[0-9]+)(?=$|[+_])", "", name)


def pareto_compare(paths, out_png, only=None):
    """One success-vs-violations panel per report (e.g. per pre-tension), no-stress cells; only: roles to draw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reps.append(json.load(f))
    roles = list(dict.fromkeys(role(c["controller"]) for r in reps for c in r["cells"] if not c.get("stress")))
    if only:
        roles = [r for r in only if r in roles]
    colors = [plt.get_cmap("tab20")(k) for k in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7)]
    col = {r: colors[i % len(colors)] for i, r in enumerate(roles)}
    markers = {"nominal": "o", "train": "s", "ood": "^", "identified": "D"}
    fig, axes = plt.subplots(1, len(reps), figsize=(6.2 * len(reps), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reps):
        t0 = rep.get("args", {}).get("pretension")
        for c in rep["cells"]:
            if c.get("stress") or role(c["controller"]) not in col:
                continue
            a = c["summary"]["all"]
            ax.scatter(100 * a["viol_frac"], 100 * a["success"], color=col[role(c["controller"])],
                       marker=markers.get(c["level"], "x"), s=60, edgecolor="k", linewidth=0.4)
        ax.set_title(f"pre-tension T0 = {t0:g} N" if t0 else "", fontsize=12)
        ax.set_xlabel("steps with a cable outside [T_min, T_max] (%)")
        ax.grid(alpha=0.3)
        ax.set_xlim(-2, 65)
    axes[0].set_ylabel("success (%)")
    for r in roles:
        axes[-1].scatter([], [], color=col[r], s=60, label=r)
    for lv, m in markers.items():
        if any(c["level"] == lv for rep in reps for c in rep["cells"]):
            axes[-1].scatter([], [], facecolor="none", edgecolor="k", marker=m, s=60, label=f"robots: {lv}")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("K22-SafeBench: success vs tension safety (top left is best)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"-> {out_png}")


def by_param(path, param, level, out_png, controllers=None, bins=5):
    """Success and steps outside the band per bin (quantiles) of one true robot parameter, one line per controller."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    cells = [c for c in rep["cells"] if c["level"] == level and not c.get("stress")
             and (not controllers or c["controller"] in controllers)]
    allv = np.concatenate([[e["params"][param] for e in c["episodes"]] for c in cells])
    edges = np.quantile(allv, np.linspace(0, 1, bins + 1))
    mid = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rows = []
    for i, c in enumerate(cells):
        v = np.array([e["params"][param] for e in c["episodes"]])
        ok = np.array([e["outcome"] == "success" for e in c["episodes"]])
        viol = np.array([e["viol_frac"] for e in c["episodes"]])
        k = np.clip(np.digitize(v, edges[1:-1]), 0, bins - 1)
        s = [100 * ok[k == b].mean() if (k == b).any() else np.nan for b in range(bins)]
        w = [100 * viol[k == b].mean() if (k == b).any() else np.nan for b in range(bins)]
        axes[0].plot(mid, s, "o-", color=colors[i % len(colors)], label=c["controller"])
        axes[1].plot(mid, w, "o-", color=colors[i % len(colors)], label=c["controller"])
        rows.append((c["controller"], s, w))
    unit = {"t0": "N", "k_c": "N/m", "k_j": "N m/rad", "payload": "kg", "backlash": "m"}.get(param, "")
    for ax, yl in zip(axes, ("success (%)", "steps with a cable outside the band (%)")):
        ax.set_xlabel(f"robot's true {param} ({unit}, mean over its cables; {bins} quantile bins)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    t0 = rep.get("args", {}).get("pretension")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"'{level}' robots at nominal pre-tension {t0:g} N: who fails, who violates" if t0 else level)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"bins of {param}: " + "  ".join(f"{lo:.3g}-{hi:.3g}" for lo, hi in zip(edges[:-1], edges[1:])))
    for name, s, w in rows:
        print(f"  {name:20s} success " + " ".join(f"{x:5.1f}" for x in s) + " | viol " + " ".join(f"{x:5.1f}" for x in w))
    print(f"-> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--compare", type=str, default="",
                    help="also draw one Pareto panel per report into this PNG (e.g. one report per pre-tension)")
    ap.add_argument("--roles", type=str, default="", help="--compare: comma list of controller roles to draw "
                    "(names without the pre-tension tag, e.g. ppo_nominal,lag,lag+cbf)")
    ap.add_argument("--by", type=str, default="", help="PARAM[:LEVEL]: success / violations per bin of a robot parameter "
                    "(first report), e.g. t0:train")
    ap.add_argument("--controllers", type=str, default="", help="--by: comma list of controllers (default all)")
    a = ap.parse_args()
    if a.compare:
        pareto_compare(a.reports, a.compare, [x for x in a.roles.split(",") if x])
    if a.by:
        param, _, level = a.by.partition(":")
        out = a.out or os.path.splitext(a.reports[0])[0]
        os.makedirs(out, exist_ok=True)
        by_param(a.reports[0], param, level or "train", os.path.join(out, f"by_{param}_{level or 'train'}.png"),
                 [x for x in a.controllers.split(",") if x])
        return
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
