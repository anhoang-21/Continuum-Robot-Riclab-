"""
==============================================================================
play_language.py - "Go through the red pipe": local VLM + image-based student (Stage 5)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_language.py --episodes 100          # statistics
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe play_language.py --video --episodes 8    # MP4

Three pipes with different tube colours; each episode an instruction names one of them by
colour or by position in the overview image. Qwen3-VL-2B (local) picks the pipe in the
overview camera image, the overview depth locates it, the robot goes to its pre-insertion
pose and the image-based student threads it (language_pipeline.py).

Statistics: grounding (right / wrong pipe / none) and task success, per instruction kind;
VLM time per instruction. --vlm-url: use a running `python vlm.py --serve` instead of
loading the model in this process (e.g. the model on the CPU, the simulation on the GPU).
==============================================================================
"""

import argparse
import os
import re
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

parser = argparse.ArgumentParser(description="Language-conditioned pipe insertion with a local VLM")
parser.add_argument("--student", type=str, default=os.path.join("models", "student_multi", "final.pt"))
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=7000)
parser.add_argument("--vlm-device", default="cuda", help="where the VLM runs when loaded in this process")
parser.add_argument("--vlm-url", default="", help="use a vlm.py --serve server, e.g. http://127.0.0.1:8765/ground")
parser.add_argument("--p-position", type=float, default=0.5, help="share of position instructions")
parser.add_argument("--video", action="store_true")
parser.add_argument("--view", choices=["close", "wide"], default="close",
                    help="video scene view: close = near the pipes (as play_vision.py --view close), wide = whole rig")
parser.add_argument("--output", type=str, default="")
parser.add_argument("--hold", type=float, default=1.0)
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
from language_pipeline import LOCATE, PHASE_NAMES, LanguageInsertController  # noqa: E402
from multi_pipe_env import MultiPipeEnv, MultiPipeEnvCfg  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from student_policy import load_student  # noqa: E402
from vision_pipeline import GIVE_UP  # noqa: E402

# scene camera (resolution, lookat, eye) and width of the right column, per --view (same views as play_vision.py)
VIEWS = {
    "close": (((960, 720), (0.09, 0.21, 0.58), (0.62, -0.42, 0.86)), 640),
    "wide": (((720, 960), (0.10, 0.02, 1.04), (0.945, -0.989, 1.340)), 720),
}
VIEW, PANEL_W = VIEWS[args.view]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_env(n, render):
    cfg = MultiPipeEnvCfg()
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.p_position = args.p_position
    cfg.sim.device = args.device
    if render:
        cfg.viewer.resolution, cfg.viewer.lookat, cfg.viewer.eye = VIEW
    return MultiPipeEnv(cfg, render_mode="rgb_array" if render else None)


def make_ground_fn():
    if args.vlm_url:
        from vlm import ground_http

        return lambda img, text: ground_http(img, text, url=args.vlm_url)
    from vlm import QwenGrounder

    g = QwenGrounder(device=args.vlm_device)
    print(f"VLM loaded in {g.load_s:.1f} s on {args.vlm_device}")
    return g.ground


# ---------------------------------------------------------------------------
def overview_panel(env, ctrl, i):
    rgb, _, K, eye, rot = env.overview_camera()
    img = cv2.cvtColor((rgb[i].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    s = PANEL_W / img.shape[1]
    img = cv2.resize(img, (PANEL_W, int(img.shape[0] * s)))
    if ctrl.box[i] is not None:
        x1, y1, x2, y2 = (np.asarray(ctrl.box[i]) * s).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, "VLM", (x1, max(y1 - 8, 90)), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if ctrl.found_by[i]:
        d = ctrl.last_det[i]
        a = np.linspace(0, 2 * np.pi, 41)
        ring = np.stack([d.pipe_xy[0] + C.PLATE_HOLE_RADIUS * np.cos(a), d.pipe_xy[1] + C.PLATE_HOLE_RADIUS * np.sin(a),
                         np.full_like(a, d.z_top)], axis=1)
        pix = (env.overview.project(ring) * s).astype(np.int32)
        cv2.polylines(img, [pix], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (PANEL_W, 78), (18, 20, 26), -1)
    cv2.putText(img, "OVERVIEW CAMERA  +  Qwen3-VL-2B (local)", (14, 26), FONT, 0.6, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f'"{env.scenes[i].instruction}"', (14, 60), FONT, 0.75 if PANEL_W >= 720 else 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _short_answer(text):
    """The box of the VLM answer, e.g. '[277, 291, 420, 505]' (the raw answer is JSON in a code block)."""
    m = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", text or "")
    return re.sub(r"\s+", " ", m.group(0)) if m else ("-" if not text else text.strip()[:30])


def tip_and_info_panel(env, ctrl, i, h, result):
    rgb, *_ = env.tip_camera()
    tip = cv2.cvtColor((rgb[i].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    tip = cv2.resize(tip, (h, h))
    cv2.putText(tip, "TIP CAMERA", (10, 24), FONT, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
    info = np.full((h, PANEL_W - h, 3), (26, 24, 20), dtype=np.uint8)
    compact = h < 300                                     # close view: one "key: value" line each
    lines = [("phase", PHASE_NAMES[ctrl.phase[i]]),
             ("VLM box (0-1000)", _short_answer(ctrl.answer[i])),
             ("VLM time", f"{ctrl.vlm_seconds[i]:.1f} s" if ctrl.answer[i] else "-"),
             ("grounding", {"right": "correct pipe", "wrong": "WRONG pipe", "none": "no box", "": "-"}[ctrl.grounding[i]]),
             ("insertion", "image-based student"),
             ("elevator", f"{float(env.robot.data.joint_pos[i, env._jelev[0]]) * 1000:+.0f} mm")]
    y = 26 if compact else 36
    for k, v in lines:
        if compact:
            cv2.putText(info, f"{k}:", (12, y), FONT, 0.42, (160, 172, 190), 1, cv2.LINE_AA)
            cv2.putText(info, v, (150, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 28
        else:
            cv2.putText(info, k, (12, y), FONT, 0.45, (160, 172, 190), 1, cv2.LINE_AA)
            cv2.putText(info, v, (12, y + 22), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 56
    if result:
        color = (40, 170, 40) if result.startswith("THROUGH") else (40, 40, 200)
        top = h - (44 if compact else 58)
        cv2.rectangle(info, (8, top), (PANEL_W - h - 8, h - 8), color, -1)
        cv2.putText(info, result, (16, h - (20 if compact else 27)), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([tip, info], axis=1)


def frame(env, ctrl, i, ep, n_eps, step, result):
    scene = cv2.cvtColor(np.ascontiguousarray(env.render()), cv2.COLOR_RGB2BGR)
    scene = cv2.resize(scene, VIEW[0])
    overlay = scene.copy()
    cv2.rectangle(overlay, (10, 10), (650, 84), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, scene, 0.28, 0, scene)
    cv2.putText(scene, "Stage 5: instruction -> local VLM -> visuomotor policy", (20, 36), FONT, 0.55,
                (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(scene, f"episode {ep}/{n_eps}  step {step:4d}  t = {step * 0.02:5.2f} s", (20, 64), FONT, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    ov = overview_panel(env, ctrl, i)
    bottom = tip_and_info_panel(env, ctrl, i, VIEW[0][1] - ov.shape[0], result)
    return np.concatenate([scene, np.concatenate([ov, bottom], axis=0)], axis=1)


# ---------------------------------------------------------------------------
def main():
    ground = make_ground_fn()
    n = 1 if args.video else args.num_envs
    env = make_env(n, render=args.video)
    student = load_student(os.path.join(TASK_DIR, args.student) if not os.path.isabs(args.student) else args.student,
                           device=env.device)
    ctrl = LanguageInsertController(env, student, ground)
    env.auto_reset = False
    obs, _ = env.reset()
    ctrl.reset(range(n))
    writer, out = None, ""
    if args.video:
        for _ in range(3):
            env.render()
        out = args.output or os.path.join(TASK_DIR, "videos", f"language_{args.view}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (VIEW[0][0] + PANEL_W, VIEW[0][1]))
    results, steps, t0 = [], np.zeros(n, dtype=int), time.time()
    while len(results) < args.episodes:
        obs, _, _, _, extras = env.step(ctrl.step_actions(obs))
        steps += 1
        outcome = extras["done_info"]["outcome"].cpu().numpy()
        ended = []
        for i in range(n):
            name = OUTCOMES[outcome[i]] if outcome[i] >= 0 else ("not_grounded" if ctrl.phase[i] == GIVE_UP else "")
            if not name:
                continue
            ended.append(i)
            sc = env.scenes[i]
            if len(results) < args.episodes:
                results.append(dict(kind=sc.kind, color=sc.colors[sc.target], instruction=sc.instruction,
                                    grounding=ctrl.grounding[i], outcome=name, vlm_s=float(ctrl.vlm_seconds[i]),
                                    steps=int(steps[i])))
                print(f"[{len(results):3d}] {sc.instruction:45s} grounding {ctrl.grounding[i] or '-':6s} -> "
                      f"{name.upper():12s} ({steps[i]} steps, VLM {ctrl.vlm_seconds[i]:.1f} s) "
                      f"answer {ctrl.answer[i].strip()[:60]!r}")
            if writer is not None:
                text = "THROUGH THE PIPE!" if name == "success" else name.upper().replace("_", " ")
                for _ in range(int(args.hold * 25)):
                    writer.write(frame(env, ctrl, i, len(results), args.episodes, steps[i], text))
            steps[i] = 0
        if writer is not None and 0 not in ended:
            writer.write(frame(env, ctrl, 0, len(results) + 1, args.episodes, steps[0], ""))
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


def report(res, wall):
    print(f"\n=== Stage 5: {len(res)} episodes, {wall:.0f} s ===")
    for kind in ("all", "color", "position"):
        r = [x for x in res if kind == "all" or x["kind"] == kind]
        if not r:
            continue
        g = [x["grounding"] for x in r]
        succ = [x["outcome"] == "success" for x in r]
        right_succ = [x["outcome"] == "success" for x in r if x["grounding"] == "right"]
        print(f"[{kind:8s}] {len(r):3d} eps | grounding right {100 * g.count('right') / len(r):5.1f}%  "
              f"wrong {100 * g.count('wrong') / len(r):4.1f}%  none {100 * g.count('none') / len(r):4.1f}% | "
              f"task success {100 * np.mean(succ):5.1f}% | insertion success when grounded right "
              f"{100 * np.mean(right_succ) if right_succ else float('nan'):5.1f}%")
    fails = {}
    for x in res:
        if x["outcome"] != "success":
            fails[x["outcome"]] = fails.get(x["outcome"], 0) + 1
    print("failures: " + (", ".join(f"{k} {v}" for k, v in sorted(fails.items())) or "none"))
    print(f"VLM: {np.mean([x['vlm_s'] for x in res if x['vlm_s'] > 0]):.2f} s per instruction on average")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    app.close()
