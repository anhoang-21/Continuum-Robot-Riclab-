"""
==============================================================================
play_vision.py - Tip camera finds the pipe, then the RL policy goes through it
==============================================================================
The policy only knows the pipe pose that the tip camera measured (perception.py);
the search / hand-over logic is vision_pipeline.py. Needs cameras (always enabled here).

    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_vision.py --episodes 50              # statistics
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_vision.py --obs-source gt --episodes 50   # baseline, true pose
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_vision.py --depth-noise 0.005 --rgb-noise 0.05
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_vision.py --meshes     # CAD frame / plate in the camera view
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_vision.py --video --pipe-mode random --episodes 8 --seed 1   # --view close: near the pipe

Statistics: success / failure per pipe mode, how the pipe was found (start view, after
lifting, scan), search time, and the error of the estimated pipe mouth vs the truth.
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
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Tip camera search + RL insertion through a vertical pipe")
parser.add_argument("--checkpoint", type=str, default=os.path.join("models", "mujoco_ppo_vpipe_wide", "model.pt"))
parser.add_argument("--pipe-mode", choices=["rig", "random", "both"], default="both")
parser.add_argument("--episodes", type=int, default=20, help="episodes per pipe mode")
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=9000)
parser.add_argument("--max-offset", type=float, default=0.17, help="random scene: max pipe offset (m)")
parser.add_argument("--max-steps", type=int, default=1000, help="episode limit (search + insertion)")
parser.add_argument("--obs-source", choices=["vision", "gt"], default="vision")
parser.add_argument("--student", type=str, default="",
                    help="image-based student (train_student.py) inserts from a pre-insertion pose instead of the PPO policy")
parser.add_argument("--depth-noise", type=float, default=0.0, help="std of the depth noise (m)")
parser.add_argument("--rgb-noise", type=float, default=0.0, help="std of the colour noise (0..1)")
parser.add_argument("--meshes", action="store_true",
                    help="statistics with the CAD meshes (frame, plate, robot) in the camera view; the video always has them")
parser.add_argument("--video", action="store_true", help="MP4: scene view + tip camera (one robot)")
parser.add_argument("--view", choices=["wide", "close"], default="wide",
                    help="video scene view: wide = whole rig incl. the elevator (prismatic joint) and its tower, "
                         "portrait, with a joint gauge; close = near the pipe")
parser.add_argument("--output", type=str, default="")
parser.add_argument("--hold", type=float, default=1.0, help="video: seconds to hold the end of each episode")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import cv2  # noqa: E402

import constants as C  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from vision_env import VisionPipeEnv, VisionPipeEnvCfg  # noqa: E402
from vision_pipeline import GIVE_UP, INSERT, PHASE_NAMES, SearchInsertController, load_actor  # noqa: E402
from perception import collar_mask  # noqa: E402

CAM_PANEL = 720
# scene camera per --view: (resolution, lookat, eye); the wide view is portrait so that the elevator tower
# above the frame (it rises and sinks with the prismatic joint) and the pipe below both fit
VIEWS = {
    "wide": ((720, 960), (0.10, 0.02, 1.04), (0.945, -0.989, 1.340)),
    "close": ((960, 720), (0.09, 0.21, 0.58), (0.62, -0.42, 0.86)),
}
VIEW_W, VIEW_H = VIEWS[args.view][0]
GAUGE_H = VIEW_H - CAM_PANEL                # wide: joint gauge below the tip camera


def make_env(modes, num_envs, render, meshes):
    cfg = VisionPipeEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.pipe_modes = tuple(modes)
    cfg.max_episode_steps = args.max_steps
    cfg.random_offset = (0.0, args.max_offset)
    cfg.assist_prob = 0.0
    cfg.seed = args.seed
    cfg.obs_source = args.obs_source
    cfg.depth_noise_std = args.depth_noise
    cfg.rgb_noise_std = args.rgb_noise
    cfg.visual = meshes
    cfg.sim.device = args.device
    if render:
        res, lookat, eye = VIEWS[args.view]
        cfg.viewer.resolution = res
        cfg.viewer.lookat = tuple(float(v) for v in lookat)
        cfg.viewer.eye = tuple(float(v) for v in eye)
    return VisionPipeEnv(cfg, render_mode="rgb_array" if render else None)


# ---------------------------------------------------------------------------
def tip_panel(env, ctrl, i):
    """Tip-camera image of env i, magnified, with the collar mask and the estimated mouth drawn in."""
    rgb, depth, K, cam_pos, cam_rot = (t[i].cpu().numpy() for t in env.tip_camera())
    img = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    s = CAM_PANEL / img.shape[1]
    img = cv2.resize(img, (CAM_PANEL, CAM_PANEL), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(collar_mask(rgb).astype(np.uint8), (CAM_PANEL, CAM_PANEL), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 255), 1)

    def project(p):
        x = cam_rot.T @ (p - cam_pos)
        if x[2] <= 1e-3:
            return None
        return int(s * (K[0, 0] * x[0] / x[2] + K[0, 2])), int(s * (K[1, 1] * x[1] / x[2] + K[1, 2]))

    if bool(env.est_valid[i]):
        c, z = env.pipe_xy_hat[i].cpu().numpy(), float(env.z_top_hat[i])
        ring = [project(np.array([c[0] + C.PLATE_HOLE_RADIUS * np.cos(a), c[1] + C.PLATE_HOLE_RADIUS * np.sin(a), z]))
                for a in np.linspace(0, 2 * np.pi, 49)]
        for p, q in zip(ring[:-1], ring[1:]):
            if p is not None and q is not None:
                cv2.line(img, p, q, (0, 255, 0), 2, cv2.LINE_AA)
        ctr = project(np.array([c[0], c[1], z]))
        if ctr is not None:
            cv2.drawMarker(img, ctr, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (0, 0), (CAM_PANEL, 70), (18, 20, 26), -1)
    cv2.putText(img, "TIP CAMERA (RGB-D, 120 deg)", (14, 26), font, 0.6, (0, 230, 255), 1, cv2.LINE_AA)
    txt = f"phase: {PHASE_NAMES[ctrl.phase[i]]}"
    if ctrl.found_by[i]:
        exy, ez = ctrl.estimate_error(i)
        txt += f" | found: {ctrl.found_by[i]}"
        if np.isfinite(exy):
            txt += f" | err xy {exy * 1000:.1f} mm, z {ez * 1000:+.1f} mm"
    cv2.putText(img, txt, (14, 56), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def bar(img, x, y, w, h, lo, hi, value, cmd=None, color=(0, 200, 255)):
    """Horizontal gauge: filled up to `value`, white tick at `cmd`."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 72, 80), -1)
    f = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + int(f * w), y + h), color, -1)
    if cmd is not None:
        c = x + int(float(np.clip((cmd - lo) / (hi - lo), 0.0, 1.0)) * w)
        cv2.line(img, (c, y - 4), (c, y + h + 4), (255, 255, 255), 2)


def gauge_panel(env, ctrl, i):
    """Prismatic joint (elevator) position / command / speed and the bend of the 3 sections."""
    img = np.full((GAUGE_H, CAM_PANEL, 3), (26, 24, 20), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    j = env._jelev[0]
    q = float(env.robot.data.joint_pos[i, j])
    qd = float(env.robot.data.joint_vel[i, j])
    cmd = float(env.elev_cmd[i])
    lo, hi = C.ELEV_RANGE
    cv2.putText(img, "PRISMATIC JOINT (elevator: lead screw lifts the whole robot + tower)", (14, 28), font, 0.5,
                (0, 230, 255), 1, cv2.LINE_AA)
    arrow = "  UP" if qd > 0.002 else ("  DOWN" if qd < -0.002 else "")
    cv2.putText(img, f"{q * 1000:+7.1f} mm   speed {qd * 1000:+6.1f} mm/s{arrow}", (14, 60), font, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    bar(img, 14, 76, CAM_PANEL - 28, 22, lo, hi, q, cmd)
    cv2.putText(img, f"{lo * 1000:.0f} mm", (14, 118), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"+{hi * 1000:.0f} mm", (CAM_PANEL - 80, 118), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "white tick: command", (CAM_PANEL // 2 - 70, 118), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    theta = torch.linalg.norm(env.bend_cmd[i], dim=1).cpu().numpy()
    for s_, th in enumerate(theta):
        y = 140 + 30 * s_
        cv2.putText(img, f"section {s_ + 1} bend {np.rad2deg(th):5.1f} deg", (14, y + 15), font, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        bar(img, 250, y, CAM_PANEL - 264, 16, 0.0, C.THETA_MAX, th, color=(80, 200, 80))
    return img


def scene_panel(env, ctrl, i, ep, n_eps, step, finished):
    frame = cv2.cvtColor(np.ascontiguousarray(env.render()), cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (VIEW_W, VIEW_H))
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (650, 110), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    info = env.info(i)
    title = "Tip camera search + image-based student policy (Isaac Sim)" if ctrl.student is not None         else "Tip camera search + PPO insertion (Isaac Sim)"
    cv2.putText(frame, title, (20, 34), font, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{env.env_modes[i]}] episode {ep}/{n_eps}  step {step:4d}  t = {step * 0.02:5.2f} s",
                (20, 60), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"tip offset {info['lat_mm']:5.1f} mm  depth {max(info['depth_mm'], 0):5.1f} mm  "
                       f"elevator {info['elevator_mm']:+6.1f} mm", (20, 86), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    if finished:
        (tw, th), _ = cv2.getTextSize(finished, font, 0.9, 2)
        x, y = (VIEW_W - tw) // 2, VIEW_H - 50
        color = (40, 170, 40) if finished.startswith("THROUGH") else (40, 40, 200)
        cv2.rectangle(frame, (x - 20, y - th - 14), (x + tw + 20, y + 14), color, -1)
        cv2.putText(frame, finished, (x, y), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    right = tip_panel(env, ctrl, i)
    if GAUGE_H > 0:
        right = np.concatenate([right, gauge_panel(env, ctrl, i)], axis=0)
    return np.concatenate([frame, right], axis=1)


# ---------------------------------------------------------------------------
def run(actor, modes):
    video = args.video
    n = len(modes) if video else max(args.num_envs, len(modes))
    env = make_env(modes, n, render=video, meshes=video or args.meshes)
    student = None
    if args.student:
        from student_policy import load_student

        path = args.student if os.path.isabs(args.student) else os.path.join(TASK_DIR, args.student)
        student = load_student(path, device=env.device)
    ctrl = SearchInsertController(env, actor, use_vision=args.obs_source == "vision" or student is not None,
                                  student=student)
    env.auto_reset = False
    obs, _ = env.reset()
    ctrl.reset(range(n))
    if video:                                   # the first viewport image comes out black
        for _ in range(3):
            env.render()
    writer, out = None, ""
    if video:
        tag = "student" if args.student else "vision"
        out = args.output or os.path.join(TASK_DIR, "videos", f"{tag}_{args.pipe_mode}_{args.view}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (VIEW_W + CAM_PANEL, VIEW_H))
    results = {m: [] for m in modes}
    steps = np.zeros(n, dtype=int)
    t0 = time.time()
    k = 0                                       # video: env shown
    while any(len(results[m]) < args.episodes for m in modes):
        if video and len(results[modes[k]]) >= args.episodes:
            k += 1
            env.viewport_camera_controller.set_view_env_index(k)
        obs, _, _, _, extras = env.step(ctrl.step_actions(obs))
        steps += 1
        outcome = extras["done_info"]["outcome"].cpu().numpy()
        ended = []
        for i in range(n):
            name = OUTCOMES[outcome[i]] if outcome[i] >= 0 else ("not_found" if ctrl.phase[i] == GIVE_UP else "")
            if not name:
                continue
            ended.append(i)
            mode = env.env_modes[i]
            exy, ez = ctrl.estimate_error(i)
            if len(results[mode]) < args.episodes:
                results[mode].append(dict(outcome=name, found_by=ctrl.found_by[i], search=int(ctrl.search_steps[i]),
                                          steps=int(steps[i]), err_xy=exy, err_z=ez))
                print(f"[env {i:2d} {mode:6s}] {name.upper():11s} {steps[i]:4d} steps (search {ctrl.search_steps[i]:3d}, "
                      f"found: {ctrl.found_by[i] or '-':18s}) est. error xy {exy * 1000:5.1f} mm z {ez * 1000:+5.1f} mm")
            if video and i == k:
                done_text = "THROUGH THE PIPE!" if name == "success" else name.upper().replace("_", " ")
                for _ in range(int(args.hold * 25)):
                    writer.write(scene_panel(env, ctrl, i, len(results[mode]), args.episodes, steps[i], done_text))
            steps[i] = 0
        if video and k not in ended:
            writer.write(scene_panel(env, ctrl, k, len(results[modes[k]]) + 1, args.episodes, steps[k], ""))
        if ended:
            obs = env.reset_envs(ended)
            ctrl.reset(ended)
        if not app.is_running():
            break
    if writer is not None:
        writer.release()
        print(f"video: {os.path.abspath(out)}")
    report(results, time.time() - t0)
    env.close()


def report(results, wall):
    scene = "CAD meshes" if args.meshes or args.video else "colliders only"
    policy = f"student {args.student}" if args.student else f"PPO, obs source {args.obs_source}"
    print(f"\n=== insertion: {policy} | {scene} | depth noise {args.depth_noise * 1000:.1f} mm | "
          f"rgb noise {args.rgb_noise:.2f} | {wall:.0f} s ===")
    for mode, res in results.items():
        if not res:
            continue
        names = [r["outcome"] for r in res]
        rates = "  ".join(f"{k}: {names.count(k)}" for k in sorted(set(names)) if k != "success")
        print(f"[{mode}] {len(res)} episodes: success {100 * names.count('success') / len(res):.1f}%  {rates}")
        found = {}
        for r in res:
            key = r["found_by"].split("+")[0] or "-"
            found[key] = found.get(key, 0) + 1
        print("  pipe found by: " + ", ".join(f"{k} {v}" for k, v in sorted(found.items())))
        search = np.array([r["search"] for r in res])
        print(f"  search + hand-over: {search.mean():.0f} steps on average ({search.mean() * 0.02:.2f} s), max {search.max()}")
        exy = np.array([r["err_xy"] for r in res]) * 1000
        ez = np.array([r["err_z"] for r in res]) * 1000
        ok = np.isfinite(exy)
        if ok.any():
            print(f"  estimate error xy: median {np.median(exy[ok]):.2f} mm, 95% {np.percentile(exy[ok], 95):.2f} mm, "
                  f"max {exy[ok].max():.2f} mm | z: median {np.median(np.abs(ez[ok])):.2f} mm, max {np.abs(ez[ok]).max():.2f} mm")
        succ = [r["steps"] for r in res if r["outcome"] == "success"]
        if succ:
            print(f"  successful episodes: {np.mean(succ):.0f} steps ({np.mean(succ) * 0.02:.2f} s) in total")


def main():
    path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(TASK_DIR, args.checkpoint)
    if not os.path.isfile(path):
        print(f"checkpoint not found: {path}")
        return
    print(f"Model: {path} | scene: {args.pipe_mode} | obs source: {args.obs_source}")
    actor = load_actor(path)
    modes = ["rig", "random"] if args.pipe_mode == "both" else [args.pipe_mode]
    with torch.inference_mode():
        run(actor, modes)


if __name__ == "__main__":
    main()
    app.close()
