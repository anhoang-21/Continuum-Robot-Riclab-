"""
==============================================================================
record_ui_demo.py - Stage 5 demo video: robot (left) + RICLAB command window (right)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe record_ui_demo.py --episodes 4

1920 x 1080 MP4: on the left the robot in Isaac Sim (the viewer camera of the Isaac window,
eye (0.62, -0.42, 0.86) -> target (0.09, 0.21, 0.58)), on the right the command window of
play_language.py --ui, redrawn with PIL (same layout, sharp at 1080p). Each episode the
instruction is typed character by character, Run is pressed, Qwen3-VL-2B (local, CPU)
picks the pipe and the image-based student threads it. The VLM wait (~6-10 s on the CPU) is
shortened in the video; the measured time is shown in the window.
==============================================================================
"""

import argparse
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

parser = argparse.ArgumentParser(description="Stage 5 demo video with the command window")
parser.add_argument("--student", default=os.path.join("models", "student_multi", "final.pt"))
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--seed", type=int, default=7300)
parser.add_argument("--output", default=os.path.join(TASK_DIR, "videos", "ui_demo.mp4"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import re  # noqa: E402
import subprocess  # noqa: E402

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from language_pipeline import PHASE_NAMES, LanguageInsertController  # noqa: E402
from multi_pipe import PALETTE  # noqa: E402
from multi_pipe_env import MultiPipeEnv, MultiPipeEnvCfg  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from student_policy import load_student  # noqa: E402
from vision_pipeline import GIVE_UP  # noqa: E402
from vlm import QwenGrounder  # noqa: E402

W, H, FPS = 1920, 1080, 25
SCENE_W, UI_W = 1280, 640
EYE, TARGET = (0.62, -0.42, 0.86), (0.09, 0.21, 0.58)   # the viewer camera of the Isaac window
SCENE_H = 720                                             # 16:9 like the Isaac viewport; cameras + pipeline below
FONTS = "C:/Windows/Fonts/"
INK, GREY, LINE, ACCENT, GREEN = (29, 36, 48), (107, 118, 134), (227, 231, 238), (227, 52, 47), (31, 157, 85)


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


F = {"title": font("seguisb.ttf", 30), "sub": font("segoeui.ttf", 17), "cap": font("seguisb.ttf", 14),
     "body": font("segoeui.ttf", 18), "bold": font("seguisb.ttf", 18), "entry": font("segoeui.ttf", 21),
     "btn": font("segoeui.ttf", 17), "run": font("seguisb.ttf", 20), "hist": font("segoeui.ttf", 16)}
LOGO = Image.open(os.path.join(TASK_DIR, "assets", "riclab_logo.png")).convert("RGBA")
LOGO = LOGO.resize((460, int(LOGO.height * 460 / LOGO.width)), Image.LANCZOS)


class UiState:
    def __init__(self):
        self.colors, self.text, self.cursor, self.pressed, self.busy = [], "", True, False, False
        self.state = {"status": "ready", "phase": "-", "VLM box (0-1000)": "-", "chosen pipe": "-", "result": "-"}
        self.history = []


def draw_ui(u):
    """The command window of command_panel.py, 640 x 1080."""
    img = Image.new("RGB", (UI_W, H), "white")
    d = ImageDraw.Draw(img)
    d.line([(0, 0), (0, H)], fill=LINE, width=2)
    x = 34
    img.paste(LOGO, (x - 8, 28), LOGO)
    y = 28 + LOGO.height + 14
    d.text((x, y), "Language-guided insertion", font=F["title"], fill=INK)
    y += 44
    d.text((x, y), "Instruction → Qwen3-VL-2B (local) → image-based policy", font=F["sub"], fill=GREY)
    y += 40

    def rule(yy):
        d.line([(x, yy), (UI_W - x, yy)], fill=LINE, width=1)

    rule(y)
    y += 18
    d.text((x, y), "PIPES ON THE TABLE  (left → right in the overview camera)", font=F["cap"], fill=GREY)
    y += 30
    cx = x
    for c in u.colors:
        rgb = tuple(int(255 * v) for v in PALETTE[c])
        d.ellipse([cx, y, cx + 28, y + 28], fill=rgb, outline=(242, 193, 46), width=4)
        d.text((cx + 36, y + 1), c, font=F["body"], fill=INK)
        cx += 36 + int(d.textlength(c, font=F["body"])) + 26
    y += 50
    rule(y)
    y += 18
    d.text((x, y), "INSTRUCTION", font=F["cap"], fill=GREY)
    y += 28
    ex = UI_W - x - 104
    d.rectangle([x, y, ex, y + 48], fill=(246, 247, 249) if u.busy else "white",
                outline=ACCENT if not u.busy else (180, 186, 196), width=2)
    d.text((x + 12, y + 11), u.text, font=F["entry"], fill=INK)
    if u.cursor and not u.busy:
        tx = x + 13 + d.textlength(u.text, font=F["entry"])
        d.line([(tx, y + 10), (tx, y + 38)], fill=INK, width=2)
    run_bg = (160, 30, 28) if u.pressed else ((236, 150, 148) if u.busy else ACCENT)
    d.rectangle([ex + 12, y, UI_W - x, y + 48], fill=run_bg)
    d.text((ex + 12 + (92 - d.textlength("Run", font=F["run"])) / 2, y + 11), "Run", font=F["run"], fill="white")
    y += 64
    bx = x
    for c in u.colors:
        label = f"{c} pipe"
        wdt = d.textlength(label, font=F["btn"]) + 28
        rgb = tuple(int(255 * v) for v in PALETTE[c])
        d.rectangle([bx, y, bx + wdt, y + 40], fill=rgb)
        d.text((bx + 14, y + 9), label, font=F["btn"], fill=INK if c == "white" else "white")
        bx += wdt + 10
    y += 50
    bx = x
    for label in ("Leftmost", "Rightmost", "New scene"):
        wdt = d.textlength(label, font=F["btn"]) + 28
        d.rectangle([bx, y, bx + wdt, y + 40], fill=(241, 243, 247))
        d.text((bx + 14, y + 9), label, font=F["btn"], fill=INK)
        bx += wdt + 10
    y += 60
    rule(y)
    y += 18
    d.text((x, y), "STATE", font=F["cap"], fill=GREY)
    y += 30
    for key, value in u.state.items():
        d.text((x, y), key, font=F["body"], fill=GREY)
        col = INK
        if key == "result" and value not in ("-", ""):
            col = GREEN if value.startswith("THROUGH") else ACCENT
        if key == "status" and value.startswith("Qwen"):
            col = ACCENT
        d.text((x + 200, y), value, font=F["bold"], fill=col)
        y += 34
    y += 12
    rule(y)
    y += 18
    d.text((x, y), "HISTORY", font=F["cap"], fill=GREY)
    y += 28
    d.rectangle([x, y, UI_W - x, H - 30], fill=(250, 251, 252), outline=LINE)
    for k, line in enumerate(u.history[:10]):
        d.text((x + 10, y + 10 + 30 * k), line, font=F["hist"], fill=INK)
    return img


STEPS = (("LOCATE", "1  Overview camera + Qwen3-VL-2B pick the pipe"),
         ("APPROACH", "2  Move to the pre-insertion pose"),
         ("INSERT", "3  Image-based policy threads the pipe"))


def bottom_row(ov, box, chosen_ring, tip, phase):
    """Overview camera with the VLM box | tip camera | pipeline, 1280 x 360."""
    row = Image.new("RGB", (SCENE_W, H - SCENE_H), (14, 20, 32))
    d = ImageDraw.Draw(row)
    o = Image.fromarray(ov).resize((480, 360), Image.LANCZOS)
    od = ImageDraw.Draw(o)
    s = 480 / ov.shape[1]
    if box is not None:
        x1, y1, x2, y2 = (np.asarray(box) * s).tolist()
        od.rectangle([x1, y1, x2, y2], outline=(230, 40, 40), width=4)
        od.text((x1 + 2, max(y1 - 22, 2)), "VLM", font=F["bold"], fill=(230, 40, 40))
    if chosen_ring is not None:
        od.line([tuple(p) for p in (chosen_ring * s).tolist()], fill=(40, 220, 90), width=3)
    row.paste(o, (0, 0))
    d.rectangle([0, 0, 250, 30], fill=(14, 20, 32))
    d.text((8, 5), "OVERVIEW CAMERA", font=F["cap"], fill=(0, 230, 255))
    t = Image.fromarray(tip).resize((360, 360), Image.LANCZOS)
    row.paste(t, (490, 0))
    d.rectangle([490, 0, 640, 30], fill=(14, 20, 32))
    d.text((498, 5), "TIP CAMERA (Seg15)", font=F["cap"], fill=(0, 230, 255))
    x, y = 872, 26
    d.text((x, y), "PIPELINE", font=F["cap"], fill=(160, 172, 190))
    y += 34
    for key, label in STEPS:
        on = phase == key
        d.rectangle([x - 8, y - 6, SCENE_W - 16, y + 60], fill=(30, 70, 110) if on else (22, 30, 44))
        parts = label.split("  ", 1)
        d.text((x, y), parts[0], font=F["run"], fill=(0, 230, 255) if on else (120, 132, 150))
        words, line, yy = parts[1].split(), "", y + 2
        for wd in words:                                  # wrap in the 370 px column
            if d.textlength(line + " " + wd, font=F["body"]) > 330 and line:
                d.text((x + 28, yy), line, font=F["body"], fill="white" if on else (170, 180, 195))
                line, yy = wd, yy + 24
            else:
                line = (line + " " + wd).strip()
        d.text((x + 28, yy), line, font=F["body"], fill="white" if on else (170, 180, 195))
        y += 80
    return row


def compose(scene_rgb, u, caption, row):
    frame = Image.new("RGB", (W, H), "white")
    scene = Image.fromarray(scene_rgb).resize((SCENE_W, SCENE_H), Image.LANCZOS)
    frame.paste(scene, (0, 0))
    frame.paste(row, (0, SCENE_H))
    d = ImageDraw.Draw(frame, "RGBA")
    d.rectangle([20, 20, 20 + 30 + d.textlength(caption, font=F["bold"]), 64], fill=(18, 20, 26, 200))
    d.text((35, 30), caption, font=F["bold"], fill=(0, 230, 255))
    frame.paste(draw_ui(u), (SCENE_W, 0))
    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)


def short_box(text):
    m = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", text or "")
    return re.sub(r"\s+", " ", m.group(0)) if m else "-"


def instruction_for(sc, k):
    """Alternate colour and position instructions (what a user would type)."""
    order = [sc.colors[j] for j in np.argsort(sc.image_x)]
    kinds = ["color", "left", "color", "right"]
    kind = kinds[k % len(kinds)]
    if kind == "left":
        return "Go through the leftmost pipe."
    if kind == "right":
        return "Insert into the pipe on the far right."
    return f"Go through the {order[(k // 2) % len(order)]} pipe."


def main():
    grounder = QwenGrounder("2b", device="cpu")
    cfg = MultiPipeEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.viewer.resolution, cfg.viewer.eye, cfg.viewer.lookat = (SCENE_W, SCENE_H), EYE, TARGET
    env = MultiPipeEnv(cfg, render_mode="rgb_array")
    student = load_student(os.path.join(TASK_DIR, args.student), device=env.device)
    ctrl = LanguageInsertController(env, student, grounder.ground, follow_choice=True)
    env.auto_reset = False
    zero = torch.zeros(1, 7, device=env.device)

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "22",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    u = UiState()
    caption = "Stage 5 · instruction → local VLM → visuomotor policy"

    view = {"box": None, "ring": None, "phase": ""}

    def cameras():
        rgb_ov, *_ = env.overview_camera()
        rgb_tip, *_ = env.tip_camera()
        return ((rgb_ov[0].cpu().numpy() * 255).astype(np.uint8), (rgb_tip[0].cpu().numpy() * 255).astype(np.uint8))

    def write(scene, n=1):
        ov, tip = cameras()
        row = bottom_row(ov, view["box"], view["ring"], tip, view["phase"])
        fr = compose(scene, u, caption, row).tobytes()
        for _ in range(n):
            ff.stdin.write(fr)

    obs, _ = env.reset()
    for _ in range(3):
        env.render()
    for ep in range(args.episodes):
        if ep > 0:
            obs = env.reset_envs([0])
        ctrl.reset([0])
        for _ in range(10):
            obs, *_ = env.step(zero)
        ctrl.reset([0])
        sc = env.scenes[0]
        # same viewing direction as the Isaac window, centred on the three pipes of this scene
        look = np.array([*sc.pipe_xy.mean(axis=0), TARGET[2] + 0.03])
        eye = look + 0.7 * (np.array(EYE) - np.array(TARGET))            # inside the frame: no post in the way
        env.viewport_camera_controller.update_view_location(eye=tuple(eye), lookat=tuple(look))
        for _ in range(3):
            env.render()
        u.colors = [sc.colors[j] for j in np.argsort(sc.image_x)]
        u.state = {"status": "ready: type an instruction", "phase": "-", "VLM box (0-1000)": "-", "chosen pipe": "-",
                   "result": "-"}
        u.text, u.busy = "", False
        view.update(box=None, ring=None, phase="")
        scene = env.render()
        for k in range(int(1.0 * FPS)):                   # new scene
            u.cursor = (k // 12) % 2 == 0
            write(scene)
        text = instruction_for(sc, ep)
        rng = np.random.default_rng(ep)
        for ch in text:                                  # typing, ~13 characters per second
            u.text += ch
            u.cursor = True
            write(scene, int(rng.integers(1, 3)) + (2 if ch in " .," else 0))
        for k in range(int(0.6 * FPS)):
            u.cursor = (k // 8) % 2 == 0
            write(scene)
        u.pressed = True
        write(scene, 4)
        u.pressed, u.busy = False, True
        sc.instruction = text
        u.state.update(status="Qwen3-VL is thinking…", phase="LOCATE")
        view["phase"] = "LOCATE"
        for _ in range(int(1.2 * FPS)):                  # the VLM wait, shortened in the video
            write(scene)
        step, name = 0, ""
        while not name:
            obs, _, _, _, extras = env.step(ctrl.step_actions(obs))
            step += 1
            o = int(extras["done_info"]["outcome"][0])
            if ctrl.phase[0] == GIVE_UP and o < 0:
                name = "not_grounded"
            elif o >= 0:
                name = OUTCOMES[o]
            chosen = sc.colors[sc.target] if ctrl.grounding[0] == "chosen" else "-"
            u.state.update(status=f"running · t = {step * 0.02:.1f} s", phase=PHASE_NAMES[ctrl.phase[0]],
                           **{"VLM box (0-1000)": short_box(ctrl.answer[0]),
                              "chosen pipe": f"{chosen}  (VLM {ctrl.vlm_seconds[0]:.1f} s on the CPU)" if chosen != "-" else "-"})
            view["phase"] = PHASE_NAMES[ctrl.phase[0]]
            if ctrl.box[0] is not None and view["box"] is None:
                view["box"] = ctrl.box[0]
                dd = ctrl.last_det[0]
                a = np.linspace(0, 2 * np.pi, 41)
                ring = np.stack([dd.pipe_xy[0] + 0.0345 * np.cos(a), dd.pipe_xy[1] + 0.0345 * np.sin(a),
                                 np.full_like(a, dd.z_top)], axis=1)
                view["ring"] = env.overview.project(ring)
            scene = env.render()
            write(scene)
        result = "THROUGH THE PIPE!" if name == "success" else name.upper().replace("_", " ")
        u.state.update(result=result, status="done")
        u.history.insert(0, f"{text}  →  {sc.colors[sc.target]}  →  {result} ({step * 0.02:.1f} s)")
        u.busy = False
        write(scene, int(2.0 * FPS))
        print(f"[{ep + 1}] {text!r} -> {sc.colors[sc.target]} -> {name} ({step} steps)", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"video: {args.output}")
    env.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    app.close()
