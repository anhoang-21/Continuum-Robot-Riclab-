"""
==============================================================================
analyze_torque.py - motor torques needed to insert the cable-driven robot into the pipe
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe analyze_torque.py                        # joint-trained policy, "bend" mode
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe analyze_torque.py --checkpoint models\\ppo_cable\\best_model.pt --action-mode motor
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe analyze_torque.py --episodes 40 --pipe-mode random

Runs insertion episodes in cable_env.py (6 lead-screw motors -> 12 cables -> arm, gravity,
pipe contacts) and records every 20 ms step: motor torque, speed and nut load of the 6 cable
motors (motors.py: EzM-42M / EzM-56M, 8 mm lead screws), cable tensions, elevator motor torque,
stage (ALIGN / INSERT / EXIT) and wall contacts.

Output (reports/torque/<run>/):
    summary.txt   peak / 99th percentile / RMS torque per motor and stage, the motor limits
                  (holding and pull-out torque at the speed reached) and the safety factors
    steps.csv     every recorded step of every episode
    torque.png    time series of one successful episode (torques, tensions, depth)
==============================================================================
"""

import argparse
import csv
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Motor torques of the cable-driven robot during pipe insertion")
parser.add_argument("--checkpoint", default="models/mujoco_ppo_vpipe_wide/model.pt", help="rsl_rl .pt")
parser.add_argument("--action-mode", choices=["bend", "motor"], default="bend",
                    help="bend: policy trained on the joint env + the lab's cable formula; motor: cable-trained policy")
parser.add_argument("--pipe-mode", choices=["rig", "random", "both"], default="both")
parser.add_argument("--episodes", type=int, default=20, help="episodes per pipe mode")
parser.add_argument("--num-envs", type=int, default=8, help="parallel envs per pipe mode")
parser.add_argument("--max-steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--no-gravity", action="store_true")
parser.add_argument("--output", default="", help="output folder (default reports/torque/<checkpoint>_<mode>)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import cables as CB  # noqa: E402
import constants as C  # noqa: E402
import mdp  # noqa: E402
import motors as MT  # noqa: E402
from agent_cfg import make_agent_cfg  # noqa: E402
from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402

STAGES = C.STAGE_NAMES


def load_actor(path):
    """Deterministic rsl_rl actor with the architecture of agent_cfg.py (as play_vertical_pipe.load_actor)."""
    actor_cfg = dict(make_agent_cfg(1, 1)["actor"])
    actor_cfg.pop("class_name")
    actor_cfg["distribution_cfg"] = dict(actor_cfg["distribution_cfg"])
    dummy = TensorDict({"policy": torch.zeros(1, mdp.OBS_DIM)}, batch_size=[1])
    actor = MLPModel(dummy, {"actor": ["policy"], "critic": ["policy"]}, "actor", mdp.ACT_DIM, **actor_cfg)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"])
    actor.eval()
    return actor


def run(actor, modes):
    cfg = CableVerticalPipeEnvCfg()
    cfg.scene.num_envs = args.num_envs * len(modes)
    cfg.pipe_modes = tuple(modes)
    cfg.action_mode = args.action_mode
    cfg.gravity = not args.no_gravity
    cfg.max_episode_steps = args.max_steps
    cfg.assist_prob = 0.0
    cfg.seed = args.seed
    cfg.sim.device = args.device
    env = CableVerticalPipeEnv(cfg)
    env.auto_reset = False
    env.set_eval(True)
    n = env.num_envs
    obs, _ = env.reset()
    traces = [[] for _ in range(n)]
    episodes = {m: [] for m in modes}
    t0 = time.time()
    with torch.inference_mode():
        while any(len(episodes[m]) < args.episodes for m in modes):
            td = TensorDict({"policy": obs["policy"].to("cpu")}, batch_size=[n])
            obs, _, _, _, extras = env.step(actor(td).to(env.device))
            d = {k: v.cpu().numpy() for k, v in extras["drive"].items()}
            meas = {k: v.cpu().numpy() for k, v in env._meas.items() if k in ("depth", "n_inner", "lat")}
            stage = (meas["depth"] > 0).astype(int) + (env._meas["tip"][:, 2] <= env.z_exit).cpu().numpy().astype(int)
            for i in range(n):
                traces[i].append({
                    "stage": int(stage[i]), "depth_mm": float(meas["depth"][i]) * 1000, "lat_mm": float(meas["lat"][i]) * 1000,
                    "contacts": int(meas["n_inner"][i]), "tau": d["motor_torque"][i], "rpm": d["motor_rpm"][i],
                    "load": d["nut_load"][i], "tension": d["tension"][i], "elev_tau": float(d["elevator_torque"][i]),
                    "elev_force": float(d["elevator_force"][i]), "alarm": bool(d["motor_alarm"][i].any()),
                })
            outcome = extras["done_info"]["outcome"].cpu().numpy()
            ended = np.nonzero(outcome >= 0)[0].tolist()
            for i in ended:
                mode = env.env_modes[i]
                if len(episodes[mode]) < args.episodes:
                    episodes[mode].append((OUTCOMES[outcome[i]], traces[i]))
                    print(f"  [{mode:6s}] episode {len(episodes[mode]):3d}: {OUTCOMES[outcome[i]]:12s} "
                          f"{len(traces[i]):3d} steps, peak cable-motor torque "
                          f"{max(np.abs(s['tau']).max() for s in traces[i]) * 1000:6.1f} mN m")
                traces[i] = []
            if ended:
                obs = env.reset_envs(ended)
    print(f"  ({time.time() - t0:.0f} s)")
    hold = env.motors.hold.cpu().numpy()
    specs = env.motors.specs
    env.close()
    return episodes, hold, specs


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarize(episodes, hold, specs, out_dir):
    lines = []
    p = lines.append
    screw = MT.CABLE_SCREW
    p("MOTOR TORQUES DURING PIPE INSERTION - cable-driven continuum robot (Isaac Sim / PhysX)")
    p(f"checkpoint {args.checkpoint} | action mode {args.action_mode} | gravity {not args.no_gravity}")
    p(f"cable motors: {', '.join(f'{nm}={s.name}' for nm, s in zip(CB.MOTOR_NAMES, specs))}; "
      f"lead screw {screw.lead * 1000:.0f} mm lead, efficiency {screw.efficiency:.2f} "
      f"(back-driven {screw.back_efficiency:.2f})")
    p(f"elevator: {MT.ELEVATOR_MOTOR.name}, {MT.ELEVATOR_SCREW.lead * 1000:.0f} mm lead, moving mass "
      f"{MT.ELEVATOR_MOVING_MASS} kg")
    p(f"cables: k {CB.CABLE_STIFFNESS:.0f} N/m, pre-tension {CB.CABLE_PRETENSION} N | backbone {CB.BACKBONE_STIFFNESS} "
      "N m/rad per joint axis (calibration values)")
    all_steps = [s for m in episodes for _, tr in episodes[m] for s in tr]
    for mode in episodes:
        res = [o for o, _ in episodes[mode]]
        p(f"[{mode}] {len(res)} episodes: success {res.count('success')}/{len(res)} "
          + " ".join(f"{k} {res.count(k)}" for k in sorted(set(res)) if k != "success"))
    succ = [s for m in episodes for o, tr in episodes[m] if o == "success" for s in tr]
    p(f"steps recorded: {len(all_steps)} ({len(succ)} in successful episodes)")
    p("")
    tau = np.array([s["tau"] for s in all_steps])
    rpm = np.array([s["rpm"] for s in all_steps])
    stage = np.array([s["stage"] for s in all_steps])
    contact = np.array([s["contacts"] for s in all_steps]) > 0
    p("cable motors [mN m]      peak    p99    RMS | ALIGN  INSERT  EXIT (peak) | in wall contact | "
      "holding  pull-out@max rpm  safety")
    for m in range(6):
        a = np.abs(tau[:, m]) * 1000
        per_stage = [a[stage == k].max() if (stage == k).any() else float("nan") for k in range(3)]
        rpm_max = np.abs(rpm[:, m]).max()
        pull = specs[m].pullout_torque(rpm_max) * 1000
        p(f"  {CB.MOTOR_NAMES[m]} ({specs[m].name})  {a.max():7.1f} {pct(a, 99):6.1f} {np.sqrt((a ** 2).mean()):6.1f} | "
          f"{per_stage[0]:6.1f} {per_stage[1]:6.1f} {per_stage[2]:6.1f} | "
          f"{(a[contact].max() if contact.any() else float('nan')):8.1f}        | {hold[m] * 1000:6.0f}  "
          f"{pull:6.0f} ({rpm_max:4.0f} rpm)  x{pull / max(a.max(), 1e-9):6.1f}")
    load = np.abs(np.array([s["load"] for s in all_steps]))
    tens = np.array([s["tension"] for s in all_steps])
    p(f"  nut load peak [N]: {np.round(load.max(0), 1)}   cable tension peak [N]: {tens.max():.1f} "
      f"(cable {CB.CABLE_NAMES[int(tens.max(0).argmax())]}), mean {tens.mean():.1f}")
    et = np.array([s["elev_tau"] for s in all_steps])
    ef = np.array([s["elev_force"] for s in all_steps])
    p(f"elevator motor [N m]: peak {np.abs(et).max():.3f}, p99 {pct(np.abs(et), 99):.3f}, mean {et.mean():.3f} "
      f"(force peak {np.abs(ef).max():.0f} N) | holding {MT.ELEVATOR_MOTOR.holding_torque} N m -> "
      f"x{MT.ELEVATOR_MOTOR.holding_torque / np.abs(et).max():.1f}")
    lift = MT.breakaway_torque(np.abs(ef).max(), MT.ELEVATOR_SCREW)
    p(f"  (the carriage moves down during insertion, so its weight back-drives the screw; to lift / start it "
      f"against the peak force: {lift:.3f} N m -> x{MT.ELEVATOR_MOTOR.holding_torque / lift:.1f})")
    alarms = sum(s["alarm"] for s in all_steps)
    p(f"motor position-error alarms (stall / lost steps): {alarms} steps")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    with open(os.path.join(out_dir, "steps.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pipe_mode", "episode", "outcome", "step", "stage", "depth_mm", "lat_mm", "contacts"]
                   + [f"tau_{m}_mNm" for m in CB.MOTOR_NAMES] + [f"rpm_{m}" for m in CB.MOTOR_NAMES]
                   + [f"load_{m}_N" for m in CB.MOTOR_NAMES] + [f"T_{c}_N" for c in CB.CABLE_NAMES]
                   + ["elev_tau_Nm", "elev_force_N"])
        for mode in episodes:
            for e, (o, tr) in enumerate(episodes[mode]):
                for k, s in enumerate(tr):
                    w.writerow([mode, e, o, k, STAGES[s["stage"]], f"{s['depth_mm']:.2f}", f"{s['lat_mm']:.2f}", s["contacts"]]
                               + [f"{v * 1000:.3f}" for v in s["tau"]] + [f"{v:.1f}" for v in s["rpm"]]
                               + [f"{v:.3f}" for v in s["load"]] + [f"{v:.3f}" for v in s["tension"]]
                               + [f"{s['elev_tau']:.4f}", f"{s['elev_force']:.2f}"])
    plot(episodes, hold, specs, out_dir)


def plot(episodes, hold, specs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = next((t for m in episodes for o, t in episodes[m] if o == "success"), None)
    if tr is None:
        tr = next(t for m in episodes for _, t in episodes[m])
    t = np.arange(len(tr)) * 0.02
    tau = np.array([s["tau"] for s in tr]) * 1000
    tens = np.array([s["tension"] for s in tr])
    stage = np.array([s["stage"] for s in tr])
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    colors = ["#d9480f", "#f08c00", "#2b8a3e", "#66a80f", "#1864ab", "#4dabf7"]
    for k, name in ((1, "INSERT"), (2, "EXIT")):
        if (stage == k).any():
            t0 = t[np.argmax(stage == k)]
            for a in ax:
                a.axvline(t0, color="0.6", ls="--", lw=0.8)
            ax[0].text(t0, 1.0, f" {name}", transform=ax[0].get_xaxis_transform(), va="top", fontsize=8, color="0.4")
    for m in range(6):
        ax[0].plot(t, tau[:, m], color=colors[m], lw=1.2, label=f"{CB.MOTOR_NAMES[m]} ({specs[m].name})")
    ax[0].set_ylabel("cable motor torque [mN m]")
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_title("Motor torques while the policy inserts the robot into the pipe")
    for c in range(12):
        ax[1].plot(t, tens[:, c], color=colors[CB.CABLE_MOTOR[c]], lw=1.0, ls="-" if CB.CABLE_SIGN[c] > 0 else ":")
    ax[1].set_ylabel("cable tension [N]\n(solid a, dotted b)")
    ax[2].plot(t, [s["elev_tau"] for s in tr], color="k", lw=1.2)
    ax[2].axhline(MT.ELEVATOR_MOTOR.holding_torque, color="r", ls=":", lw=0.8)
    ax[2].set_ylabel(f"elevator torque [N m]\n({MT.ELEVATOR_MOTOR.name}, red: holding)")
    ax[3].plot(t, [s["depth_mm"] for s in tr], color="k", lw=1.2, label="tip depth below the pipe top")
    ax[3].plot(t, [s["lat_mm"] for s in tr], color="#1864ab", lw=1.0, label="tip offset from the pipe axis")
    ax3b = ax[3].twinx()
    ax3b.fill_between(t, 0, [s["contacts"] for s in tr], color="#e03131", alpha=0.25, step="mid")
    ax3b.set_ylabel("wall contacts", color="#e03131")
    ax[3].set_ylabel("[mm]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "torque.png")
    fig.savefig(path, dpi=130)
    print(f"plot: {path}")


def main():
    path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(TASK_DIR, args.checkpoint)
    if not os.path.isfile(path):
        print(f"checkpoint not found: {path}")
        return
    run_name = os.path.basename(os.path.dirname(path)) + f"_{args.action_mode}"
    out_dir = args.output or os.path.join(TASK_DIR, "reports", "torque", run_name)
    os.makedirs(out_dir, exist_ok=True)
    modes = ["rig", "random"] if args.pipe_mode == "both" else [args.pipe_mode]
    print(f"policy {path} | {args.action_mode} mode | pipes {modes} | {args.episodes} episodes each")
    episodes, hold, specs = run(load_actor(path), modes)
    summarize(episodes, hold, specs, out_dir)


if __name__ == "__main__":
    main()
    app.close()
