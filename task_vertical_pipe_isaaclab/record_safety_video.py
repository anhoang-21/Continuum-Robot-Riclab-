"""
==============================================================================
record_safety_video.py - 1920 x 1080 clips: a policy on uncertain robots, with the tension band monitor
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe record_safety_video.py --policy models/ppo_cable_128/best_model.pt \\
        --label "PPO (Stage 4)" --pretension 3 --output videos/safety_before.mp4
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe record_safety_video.py --policy models/lag_t525/best_safe_model.pt \\
        --label "PPO-Lagrangian + CBF" --pretension 5.25 --cbf --output videos/safety_after.mp4

Left: Isaac Sim with the 12 cables (grey = slack) and the lead-screw drive over the arm (record_cable_video.py's
scene). Right: the tension band monitor - the 12 cable tensions (robot side; tick: nut side, what a load cell
reads) against [T_min, T_max], green inside / red outside, the share of steps outside the band, the highest
tension, what the CBF does (active / infeasible), the robot's drawn parameters, and a timeline of the episode.
The robots and pipes come from fixed seeds, so clips of different controllers show the same robots.
==============================================================================
"""

import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser(description="Clips with the tension band monitor")
parser.add_argument("--policy", required=True)
parser.add_argument("--label", default="policy")
parser.add_argument("--cbf", action="store_true", help="tension CBF under the policy")
parser.add_argument("--cbf-source", choices=["loadcell", "loadcell_id", "model", "oracle"], default="loadcell_id")
parser.add_argument("--pretension", type=float, default=3.0)
parser.add_argument("--level", choices=["nominal", "train", "ood"], default="train")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=21)
parser.add_argument("--output", default=os.path.join(TASK_DIR, "videos", "safety.mp4"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args)).app

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402

import cables as CB  # noqa: E402
from controllers import load_controller  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402
from safety_filter import TensionCBF  # noqa: E402
from vertical_pipe_env import scene_frame, scene_view_cfg  # noqa: E402

W, H, FPS = 1920, 1080, 25
PANEL_W = 640
FONT, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
FULL_VIEW = ((W - PANEL_W, H), (0.11, 0.20, 1.16), (1.14, 1.28, 1.44))
BG, GREEN, RED, GREY, WHITE, CYAN = (26, 22, 18), (70, 190, 70), (60, 60, 230), (150, 150, 150), (238, 238, 238), (255, 220, 0)
SECTION_COL = ((40, 110, 245), (90, 205, 60), (255, 140, 60))


def open_writer(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def hide_frame(env):
    from pxr import UsdGeom

    for path in env.scene.env_prim_paths:
        prim = env.sim.stage.GetPrimAtPath(f"{path}/Robot/base_link/base_link")
        for child in prim.GetChildren():
            if "visual" in child.GetName():
                UsdGeom.Imageable(child).MakeInvisible()


def banner(img, text, y=18):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    over = img.copy()
    cv2.rectangle(over, (18, y), (18 + tw + 28, y + th + 24), (18, 20, 26), -1)
    cv2.addWeighted(over, 0.8, img, 0.2, 0, img)
    cv2.putText(img, text, (32, y + th + 12), FONT, 0.7, CYAN, 2, AA)


def badge(img, text, ok):
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
    x, y = (img.shape[1] - tw) // 2, 110
    cv2.rectangle(img, (x - 22, y - th - 16), (x + tw + 22, y + 16), (40, 170, 40) if ok else (40, 40, 200), -1)
    cv2.putText(img, text, (x, y), FONT, 1.0, (255, 255, 255), 2, AA)


def monitor(env, i, hist, cbf, t_min, t_max):
    """The tension band panel (PANEL_W x H) of env i."""
    p = np.full((H, PANEL_W, 3), BG, np.uint8)
    cv2.putText(p, "TENSION BAND MONITOR", (20, 42), FONT, 0.85, CYAN, 2, AA)
    cv2.putText(p, f"{args.label}   |   pre-tension T0 = {args.pretension:g} N", (20, 78), FONT, 0.55, WHITE, 1, AA)
    # bars
    T_r, T_n = env.tension[i].cpu().numpy(), env.ucm.T_n[i].cpu().numpy()
    top, bot, x0, bw, gap = 130, 560, 70, 32, 10
    tmax_axis = 16.0
    ypix = lambda v: int(bot - (bot - top) * min(v, tmax_axis) / tmax_axis)  # noqa: E731
    for v in range(0, 17, 4):
        cv2.line(p, (x0 - 8, ypix(v)), (PANEL_W - 30, ypix(v)), (60, 56, 50), 1)
        cv2.putText(p, f"{v}", (28, ypix(v) + 5), FONT, 0.45, GREY, 1, AA)
    cv2.putText(p, "N", (30, top - 12), FONT, 0.45, GREY, 1, AA)
    x = x0
    for c in range(CB.N_CABLES):
        s = CB.CABLE_SECTION[c]
        inside = t_min <= T_r[c] and max(T_r[c], T_n[c]) <= t_max
        col = GREEN if inside else RED
        cv2.rectangle(p, (x, ypix(T_r[c])), (x + bw, bot), col, -1)
        cv2.line(p, (x - 3, ypix(T_n[c])), (x + bw + 3, ypix(T_n[c])), WHITE, 2)
        cv2.putText(p, CB.CABLE_NAMES[c], (x + 1, bot + 22), FONT, 0.42, SECTION_COL[s], 1, AA)
        x += bw + gap + (12 if c % 4 == 3 else 0)
    for v, name in ((t_min, "T_min"), (t_max, "T_max")):
        cv2.line(p, (x0 - 8, ypix(v)), (PANEL_W - 30, ypix(v)), CYAN, 2)
        cv2.putText(p, f"{name} {v:g} N", (PANEL_W - 140, ypix(v) - 8 if name == "T_max" else ypix(v) + 20), FONT, 0.5, CYAN, 1, AA)
    cv2.putText(p, "bars: robot-side tension   white tick: nut side (load cell)", (20, bot + 52), FONT, 0.45, GREY, 1, AA)
    # numbers
    viol = np.array(hist["viol"]) if hist["viol"] else np.zeros(1)
    lines = [(f"steps with a cable outside the band: {100 * viol.mean():4.1f} %", RED if viol.mean() > 0.05 else GREEN),
             (f"slack cables now: {int((T_r < t_min).sum())}    highest tension so far: {max(hist['tmax'] or [0]):.1f} N", WHITE)]
    if cbf is not None:
        act = bool(cbf.last_active[i])
        inf = float(cbf.last_slack[i]) > 0.05
        lines.append((f"tension CBF: {'INFEASIBLE (least violation)' if inf else ('ACTIVE' if act else 'idle')}",
                      RED if inf else (CYAN if act else GREY)))
    else:
        lines.append(("no safety filter", GREY))
    pr = env.prm
    lines.append((f"this robot: k_c {pr['k_c'][i].mean():.0f} N/m  T0 {pr['t0'][i].mean():.2f} N  "
                  f"mu theta {pr['mu_theta'][i].mean():.2f}", GREY))
    lines.append((f"            k_j {pr['k_j'][i]:.2f} N m/rad  backlash {pr['backlash'][i].mean() * 1e3:.2f} mm  "
                  f"payload {pr['payload'][i] * 1e3:.0f} g", GREY))
    y = 640
    for text, col in lines:
        cv2.putText(p, text, (20, y), FONT, 0.55, col, 1, AA)
        y += 34
    # timeline
    cv2.putText(p, "episode timeline (red: a cable outside the band)", (20, 880), FONT, 0.5, GREY, 1, AA)
    cv2.rectangle(p, (20, 895), (PANEL_W - 20, 935), (50, 46, 40), -1)
    n = max(len(hist["viol"]), 1)
    span = PANEL_W - 40
    for k, v in enumerate(hist["viol"]):
        xa, xb = 20 + int(span * k / max(n, 150)), 20 + int(span * (k + 1) / max(n, 150))
        cv2.rectangle(p, (xa, 895), (max(xb, xa + 1), 935), RED if v else GREEN, -1)
    return p


def main():
    cfg = RobustCableEnvCfg()
    cfg.scene.num_envs = 1
    cfg.visual = True
    cfg.show_cables = cfg.show_actuators = True
    cfg.cable_draw_radius = 0.0012
    cfg.uncertainty = args.level
    cfg.cable_pretension = args.pretension
    cfg.obs_tension = True
    cfg.priv_obs = True
    cfg.assist_prob = 0.0
    cfg.seed = args.seed
    cfg.pipe_modes = ("random",)
    cfg.sim.device = args.device
    resolution, lookat, eye = FULL_VIEW
    cfg.sim.visualizer_cfgs = [scene_view_cfg(eye, lookat, resolution)]
    env = RobustCableEnv(cfg)
    hide_frame(env)
    env.auto_reset = False
    env.set_eval(True)
    ctrl = load_controller(os.path.join(TASK_DIR, args.policy) if not os.path.isabs(args.policy) else args.policy,
                           device=env.device)
    cbf = TensionCBF(env, source=args.cbf_source) if args.cbf else None
    env.safety_filter = cbf
    ff = open_writer(args.output)
    obs, _ = env.reset()
    for _ in range(3):
        scene_frame(env)
    with torch.inference_mode():
        for ep in range(args.episodes):
            ctrl.reset()
            hist = {"viol": [], "tmax": []}
            step, name = 0, ""
            while not name:
                obs, _, _, _, ex = env.step(ctrl(obs, env))
                step += 1
                s = ex["safety"]
                hist["viol"].append(bool((s["t_min"][0] < cfg.t_min) or (s["t_max"][0] > cfg.t_max)))
                hist["tmax"].append(float(s["t_max"][0]))
                o = int(ex["done_info"]["outcome"][0])
                name = OUTCOMES[o] if o >= 0 else ""
                if step % 2 and not name:
                    continue
                scene = cv2.resize(cv2.cvtColor(np.ascontiguousarray(scene_frame(env)), cv2.COLOR_RGB2BGR),
                                   FULL_VIEW[0], interpolation=cv2.INTER_AREA)
                banner(scene, f"Isaac Sim  |  {args.label}  |  robot {ep + 1}/{args.episodes} ({args.level} set)  |  "
                              f"t = {step * 0.02:4.2f} s")
                if name:
                    badge(scene, "THROUGH THE PIPE!" if name == "success" else name.upper().replace("_", " "),
                          name == "success")
                frame = np.hstack([scene, monitor(env, 0, hist, cbf, cfg.t_min, cfg.t_max)])
                for _ in range(int(2.0 * FPS) if name else 1):
                    ff.stdin.write(frame.tobytes())
            print(f"robot {ep + 1}: {name} after {step} steps, tension outside the band in "
                  f"{100 * np.mean(hist['viol']):.1f}% of the steps, max {max(hist['tmax']):.1f} N", flush=True)
            obs = env.reset_envs([0])
    ff.stdin.close()
    ff.wait()
    print(f"video: {os.path.abspath(args.output)}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
