"""
==============================================================================
make_showcase_video.py - 1080p project video (title cards + the recorded clips)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe make_showcase_video.py --output videos\\showcase.mp4

Title -> Stage 1 (MuJoCo clip) -> Stages 2-3 (tip camera + PPO clip) -> Stage 4 (image-based
student clip) -> results -> next steps. H.264 through the ffmpeg of imageio-ffmpeg; no
Isaac Sim needed. The clips come from play_vision.py --video (and the MuJoCo demo).
==============================================================================
"""

import argparse
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Project showcase video")
parser.add_argument("--mujoco", default=r"D:\mujoco\Continuum_MuJoCo\task_vertical_pipe\videos\vertical_pipe_both_wide.mp4")
parser.add_argument("--vision", default=os.path.join(TASK_DIR, "videos", "vision_random.mp4"),
                    help="play_vision.py --video --view close (PPO with the camera estimate)")
parser.add_argument("--student", default=os.path.join(TASK_DIR, "videos", "student_random_wide.mp4"),
                    help="play_vision.py --video --student ... (image-based student)")
parser.add_argument("--output", default=os.path.join(TASK_DIR, "videos", "showcase.mp4"))
parser.add_argument("--crf", type=int, default=24, help="x264 quality (lower = better, larger)")
args = parser.parse_args()

W, H, FPS = 1920, 1080, 30
BG = (14, 20, 32)
ACCENT = (53, 195, 243)
WHITE = (240, 244, 250)
GREY = (160, 172, 190)
LINE = (40, 52, 72)
FONTS = "C:/Windows/Fonts/"
GITHUB = "github.com/anhoang-21/Continuum-Robot-Riclab-  (branch isaaclab-vertical-pipe)"
BAR = 110                                   # height of the caption bar under the clips

# success rates, 100 episodes per scene and setting (README: tip camera / image-based student)
RESULTS = [
    ("Setting", "Fixed pipe", "Random pipe"),
    ("PPO with the true pipe pose (privileged)", "100 %", "95 – 99 %"),
    ("PPO with the tip-camera estimate", "100 %", "96 – 99 %"),
    ("Image-based student (camera + joints only) *", "100 %", "100 %"),
]


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


def canvas():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    return img, d


def to_bgr(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def title_card():
    img, d = canvas()
    d.text((160, 330), "Autonomous Pipe Insertion", font=font("seguisb.ttf", 88), fill=WHITE)
    d.text((160, 440), "with a Continuum Robot", font=font("seguisb.ttf", 88), fill=WHITE)
    d.rectangle([160, 580, 400, 586], fill=ACCENT)
    d.text((160, 620), "Reinforcement Learning  ·  Eye-in-Hand Vision  ·  Visuomotor Policy",
           font=font("segoeui.ttf", 40), fill=GREY)
    d.text((160, 680), "MuJoCo  →  NVIDIA Isaac Sim / Isaac Lab", font=font("segoeui.ttf", 40), fill=ACCENT)
    return to_bgr(img)


def section_card(tag, title, bullets):
    img, d = canvas()
    d.text((160, 250), tag, font=font("seguisb.ttf", 40), fill=ACCENT)
    d.text((160, 310), title, font=font("seguisb.ttf", 72), fill=WHITE)
    y = 470
    for b in bullets:
        d.ellipse([168, y + 20, 182, y + 34], fill=ACCENT)
        d.text((210, y), b, font=font("segoeui.ttf", 40), fill=WHITE)
        y += 80
    return to_bgr(img)


def results_card():
    img, d = canvas()
    d.text((160, 150), "RESULTS", font=font("seguisb.ttf", 40), fill=ACCENT)
    d.text((160, 210), "Success rate, 100 episodes per setting, Isaac Sim", font=font("seguisb.ttf", 60), fill=WHITE)
    x = [160, 1150, 1470]
    y = 380
    for k, row in enumerate(RESULTS):
        f = font("seguisb.ttf" if k == 0 else "segoeui.ttf", 40)
        for xi, t in zip(x, row):
            d.text((xi, y), t, font=f, fill=GREY if k == 0 else (ACCENT if xi != x[0] else WHITE))
        y += 90
        d.line([160, y - 20, 1760, y - 20], fill=LINE, width=2)
    notes = ["Pipe-centre estimate: median < 1 mm  ·  all rows hold with 5 mm depth noise",
             "* inserts from a pre-insertion pose (pipe in view), where the teacher also reaches 100 %:",
             "   the student matches its privileged teacher from pixels and joint commands alone"]
    for k, t in enumerate(notes):
        d.text((160, y + 20 + 50 * k), t, font=font("segoeui.ttf", 34), fill=GREY)
    return to_bgr(img)


def next_card():
    img, d = canvas()
    d.text((160, 230), "NEXT", font=font("seguisb.ttf", 40), fill=ACCENT)
    d.text((160, 290), "Towards real-world autonomy", font=font("seguisb.ttf", 72), fill=WHITE)
    items = [("Stage 5", "Language-conditioned target selection  →  Vision-Language-Action"),
             ("", "(several pipes; a VLM grounds the instruction, the visuomotor skill executes it)"),
             ("Sim2Real", "Domain randomization, learned pipe detector, real continuum robot")]
    y = 460
    for tag, t in items:
        if tag:
            d.text((160, y), tag, font=font("seguisb.ttf", 42), fill=ACCENT)
        d.text((380, y), t, font=font("segoeui.ttf", 42 if tag else 36), fill=WHITE if tag else GREY)
        y += 80
    d.rectangle([160, 800, 1760, 802], fill=LINE)
    d.text((160, 830), "Code: " + GITHUB, font=font("consola.ttf", 36), fill=GREY)
    return to_bgr(img)


def caption(tag, text):
    """Caption bar (RGBA) drawn over the bottom of the clip frames."""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle([0, H - BAR, W, H], fill=(10, 14, 22, 215))
    d.rectangle([0, H - BAR, 10, H], fill=ACCENT + (255,))
    ft = font("seguisb.ttf", 34)
    d.text((50, H - 82), tag, font=ft, fill=ACCENT + (255,))
    d.text((50 + d.textlength(tag, font=ft) + 30, H - 82), text, font=font("segoeui.ttf", 34), fill=WHITE + (255,))
    a = np.asarray(ov).astype(np.float32)
    return cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:] / 255.0


def clip(path, speed, overlay):
    """(frame count, generator) of a clip played at `speed`, fitted above the caption bar."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = int(n / src_fps / speed * FPS)
    rgb, alpha = overlay

    def gen():
        last, idx = None, -1
        for k in range(count):
            want = min(int(k / FPS * speed * src_fps), n - 1)
            while idx < want:
                ok, im = cap.read()
                if not ok:
                    break
                idx += 1
                last = im
            h, w = last.shape[:2]
            s = min(W / w, (H - BAR) / h)
            vid = cv2.resize(last, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            fr = np.full((H, W, 3), BG[::-1], dtype=np.uint8)
            y0, x0 = (H - BAR - vid.shape[0]) // 2, (W - vid.shape[1]) // 2
            fr[y0:y0 + vid.shape[0], x0:x0 + vid.shape[1]] = vid
            yield (fr * (1 - alpha) + rgb * alpha).astype(np.uint8)

    return count, gen()


def still(img, seconds):
    n = int(seconds * FPS)
    return n, (img for _ in range(n))


def with_fades(segment, fade=0.5):
    n, frames = segment
    m = int(fade * FPS)
    bg = np.full((H, W, 3), BG[::-1], dtype=np.float32)
    for k, fr in enumerate(frames):
        a = min(1.0, (k + 1) / m, (n - k) / m)
        yield (fr * a + bg * (1 - a)).astype(np.uint8) if a < 1 else fr


def main():
    segments = [
        lambda: still(title_card(), 5),
        lambda: still(section_card("STAGE 1", "Reinforcement learning in MuJoCo", [
            "3-section cable-driven continuum robot + prismatic elevator",
            "PPO policy optimizes the insertion trajectory (3 M steps)",
            "Reward: alignment, progress, smoothness, wall-contact penalty",
            "≈ 99 – 100 % success, fixed and randomly placed pipes"]), 5),
        lambda: clip(args.mujoco, 1.0, caption("STAGE 1 · MuJoCo", "PPO policy bends into an S-curve and threads the pipe")),
        lambda: still(section_card("STAGES 2 – 3", "Isaac Sim + eye-in-hand vision", [
            "Task ported to Isaac Lab (GPU PhysX), MuJoCo policy runs without retraining",
            "120° RGB-D camera on the tip: search → detect pipe mouth → 3D estimate",
            "Robot scans on its own when the pipe is out of view",
            "The PPO policy inserts using only the camera's estimate"]), 5),
        lambda: clip(args.vision, 1.5, caption("STAGES 2 – 3 · Isaac Sim",
                                               "Left: scene   ·   Right: tip camera (green = estimated pipe mouth)")),
        lambda: still(section_card("STAGE 4", "Visuomotor policy learned from pixels", [
            "Student CNN: tip RGB-D image (64 × 64) + joint commands  →  7 actions",
            "No pipe pose at all during insertion",
            "DAgger from the privileged PPO teacher: 614 k steps, 18 min on a laptop GPU",
            "Search puts the pipe in view, the student threads it"]), 5),
        lambda: clip(args.student, 1.5, caption("STAGE 4 · Isaac Sim",
                                                "Image-based student   ·   whole rig, tip camera, prismatic-joint gauge")),
        lambda: still(results_card(), 7),
        lambda: still(next_card(), 7),
    ]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = 0
    for seg in segments:
        for fr in with_fades(seg()):
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
            total += 1
    proc.stdin.close()
    proc.wait()
    size = os.path.getsize(args.output) / 1e6
    print(f"{total} frames, {total / FPS:.1f} s, {size:.1f} MB -> {args.output}")


if __name__ == "__main__":
    main()
