"""
==============================================================================
record_cable_video.py - 1920 x 1080 clips of the cable-driven robot (motors -> lead screws -> cables)
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe record_cable_video.py --mode demo
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe record_cable_video.py --mode ppo --episodes 3
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe record_cable_video.py --mode student --episodes 4

Left: Isaac Sim with the 12 cables drawn through the disc holes and the drive of the CAD over the
arm (tension box, 6 lead screws in their cage, motors on top) brought to life: a nut that runs on
every screw, a flag on the screw coupling that turns with the rotor, the cable pair from the nut
down into the tension box (the static CAD nuts are removed). Right: actuator_viz.draw_actuators - the same modules as a diagram with
rotor angle, rpm, motor torque, nut position and cable tensions. The aluminium frame is hidden
(visual only), it has beams at the height of the arm.

  demo     every motor in turn winds its cable in and out, then an S-curve (scripted motor targets)
  ppo      the PPO policy on the 6 motors inserts the robot (models/ppo_cable_128/best_model.pt)
  student  tip-camera search, pre-insertion pose, the image-based student on the 6 motors inserts
           (models/student_cable/final.pt); bottom row: tip camera + pipeline phase
==============================================================================
"""

import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: DLL order before Kit)
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser(description="Clips of the cable-driven robot")
parser.add_argument("--mode", choices=["demo", "ppo", "student"], default="demo")
parser.add_argument("--policy", default=os.path.join("models", "ppo_cable_128", "best_model.pt"))
parser.add_argument("--student", default=os.path.join("models", "student_cable", "final.pt"))
parser.add_argument("--episodes", type=int, default=3, help="ppo / student: episodes per pipe mode")
parser.add_argument("--pipe-mode", choices=["rig", "random", "both"], default="both")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--pull-mm", type=float, default=15.0, help="demo: how far every motor winds its cable in")
parser.add_argument("--output", default="")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args)).app

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402

import cables as CB  # noqa: E402
import constants as C  # noqa: E402
import motors as MT  # noqa: E402
from actuator_viz import draw_actuators  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from vertical_pipe_env import scene_frame, scene_view_cfg, set_scene_view  # noqa: E402

W, H, FPS = 1920, 1080, 25
PANEL_W = 640
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
# 3/4 view on the whole drive: motors + lead screws in their cage over the arm (CAD), arm, pipe (below)
FULL_VIEW = ((W - PANEL_W, H), (0.11, 0.20, 1.16), (1.14, 1.28, 1.44))
# student: close to the pipe (as play_vision.py --view close), tip camera in the bottom row
CLOSE_VIEW = ((W - PANEL_W, 720), (0.09, 0.21, 0.62), (0.66, -0.36, 0.92))


def hide_frame(env):
    """The aluminium frame has a ring of beams at the height of the arm: hide it (visual only)."""
    from pxr import UsdGeom

    for path in env.scene.env_prim_paths:
        prim = env.sim.stage.GetPrimAtPath(f"{path}/Robot/base_link/base_link")
        for child in prim.GetChildren():
            if "visual" in child.GetName():
                UsdGeom.Imageable(child).MakeInvisible()


def configure(cfg, view):
    cfg.scene.num_envs = 1
    cfg.visual = True
    cfg.show_cables = True
    cfg.show_actuators = True
    cfg.cable_draw_radius = 0.0012
    cfg.seed = args.seed
    cfg.sim.device = args.device
    resolution, lookat, eye = view
    cfg.sim.visualizer_cfgs = [scene_view_cfg(eye, lookat, resolution)]
    return cfg


def open_writer(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def banner(img, lines, x=18, y=None, width=None):
    """Dark caption box (bottom left by default) with a coloured title line and white lines under it."""
    width = width or max(cv2.getTextSize(t, FONT, 0.62 if k == 0 else 0.52, 2 if k == 0 else 1)[0][0]
                         for k, t in enumerate(lines)) + 30
    h = 18 + 30 * len(lines)
    y = img.shape[0] - h - 18 if y is None else y
    over = img.copy()
    cv2.rectangle(over, (x, y), (x + width, y + h), (18, 20, 26), -1)
    cv2.addWeighted(over, 0.78, img, 0.22, 0, img)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x + 14, y + 30 + 30 * k), FONT, 0.62 if k == 0 else 0.52,
                    (0, 230, 255) if k == 0 else (235, 235, 235), 2 if k == 0 else 1, AA)


def panel_caption(panel, lines, y0=850, width=600):
    """Caption lines (wrapped) in the free space under the actuator diagram: first line coloured."""
    y = y0
    for k, text in enumerate(lines):
        scale, th, col = (0.6, 2, (0, 230, 255)) if k == 0 else (0.52, 1, (235, 235, 235))
        words, line = text.split(), ""
        for wd in words + [None]:
            trial = (line + " " + wd).strip() if wd else line
            if wd is None or cv2.getTextSize(trial, FONT, scale, th)[0][0] > width:
                cv2.putText(panel, line, (20, y), FONT, scale, col, th, AA)
                y += 28
                line = wd or ""
            else:
                line = trial
        y += 6


def result_badge(img, text, top=True):
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
    x, y = (img.shape[1] - tw) // 2, (40 + th) if top else img.shape[0] - 50
    color = (40, 170, 40) if text.startswith("THROUGH") else (40, 40, 200)
    cv2.rectangle(img, (x - 22, y - th - 16), (x + tw + 22, y + 16), color, -1)
    cv2.putText(img, text, (x, y), FONT, 1.0, (255, 255, 255), 2, AA)


def scene_image(env, size):
    img = cv2.cvtColor(np.ascontiguousarray(scene_frame(env)), cv2.COLOR_RGB2BGR)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def bend_line(env):
    b = env.bend_meas[0].cpu().numpy()
    return "bend  " + "   ".join(f"S{s + 1} {np.degrees(np.linalg.norm(b[s])):4.1f} deg" for s in range(3))


# ---------------------------------------------------------------------------
def run_demo(out):
    from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg

    cfg = configure(CableVerticalPipeEnvCfg(), FULL_VIEW)
    cfg.pipe_modes = ("rig",)
    cfg.max_episode_steps = 10 ** 6
    env = CableVerticalPipeEnv(cfg)
    hide_frame(env)
    env.auto_reset = False
    env.reset()
    env.set_start_state([0], C.PLATE_HOLE_XY[None], [C.PLATE_TOP_Z + C.PIPE_HEIGHT], np.zeros((1, 3, 2)),
                        [C.ELEV_RANGE[1] - 0.002])
    env.scene.write_data_to_sim()
    rate = env.cable_model.u_rate.cpu().numpy()
    pull = args.pull_mm / 1000.0
    steps = []                                              # (motor targets, caption)
    for m in range(6):
        name, s = CB.MOTOR_NAMES[m], CB.MOTOR_SECTION[m] + 1
        n = int(np.ceil(pull / (0.9 * rate[m])))
        cap = f"{name} turns its lead screw: the nut winds cable {name}a in by {args.pull_mm:.0f} mm -> section {s} bends"
        steps += [(np.eye(6)[m] * -pull * (k + 1) / n, cap) for k in range(n)] + [(np.eye(6)[m] * -pull, cap)] * 25
        steps += [(np.eye(6)[m] * -pull * (n - k - 1) / n, f"{name} turns back: the cable is paid out again")
                  for k in range(n)] + [(np.zeros(6), "")] * 8
    bend = np.array([[0.55, 0.25], [-0.75, -0.30], [0.35, 0.15]])
    u_s = env.cable_model.compensate(torch.as_tensor(bend[None], device=env.device)).cpu().numpy()[0]
    n = int(np.ceil(np.abs(u_s / (0.9 * rate)).max()))
    cap = "all 6 motors together: S-curve (the lab's formula compensates the cable coupling)"
    steps += [(u_s * (k + 1) / n, cap) for k in range(n)] + [(u_s, cap)] * 50
    steps += [(u_s * (n - k - 1) / n, "back to straight") for k in range(n)] + [(np.zeros(6), "back to straight")] * 20
    ff = open_writer(out)
    for _ in range(3):
        scene_frame(env)
    with torch.inference_mode():
        for k, (target, caption) in enumerate(steps):
            a = torch.zeros(1, 7, device=env.device)
            a[0, :6] = torch.clamp((torch.as_tensor(target, device=env.device) - env.motor_cmd[0]) / env.cable_model.u_rate,
                                   -1, 1).float()
            env.step(a)
            if k % 2:                                           # 25 fps video of the 50 Hz simulation
                continue
            scene = scene_image(env, FULL_VIEW[0])
            banner(scene, ["Isaac Sim  |  cable-driven continuum robot"], y=18)
            panel = draw_actuators(env, 0, PANEL_W, H)
            panel_caption(panel, ["6 lead-screw motors pull 12 cables", caption or " ",
                                  bend_line(env) + "   (grey cable = slack)"])
            frame = np.hstack([scene, panel])
            ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()


# ---------------------------------------------------------------------------
def run_ppo(out):
    from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg
    from vision_pipeline import load_actor

    modes = ["rig", "random"] if args.pipe_mode == "both" else [args.pipe_mode]
    actor = load_actor(os.path.join(TASK_DIR, args.policy))
    ff = open_writer(out)
    cfg = configure(CableVerticalPipeEnvCfg(), FULL_VIEW)
    cfg.scene.num_envs = len(modes)
    cfg.scene.env_spacing = 6.0                                  # the other robot out of the wide view
    cfg.pipe_modes = tuple(modes)
    cfg.assist_prob = 0.0
    env = CableVerticalPipeEnv(cfg)
    hide_frame(env)
    env.auto_reset = False
    env.set_eval(True)
    obs, _ = env.reset()
    for _ in range(3):
        scene_frame(env)
    with torch.inference_mode():
        for k, mode in enumerate(modes):
            set_scene_view(env, env_index=k)
            if k > 0:                                           # this env ran unseen meanwhile: start it afresh
                obs = env.reset_envs([k])
            for ep in range(args.episodes):
                step, name = 0, ""
                while not name:
                    td = tensordict.TensorDict({"policy": obs["policy"].cpu()}, batch_size=[env.num_envs])
                    obs, _, _, _, extras = env.step(actor(td).to(env.device))
                    step += 1
                    o = extras["done_info"]["outcome"].cpu().numpy()
                    name = OUTCOMES[o[k]] if o[k] >= 0 else ""
                    info = env.info(k)
                    scene = scene_image(env, FULL_VIEW[0])
                    banner(scene, ["Isaac Sim  |  RL policy on the 6 lead-screw motors"], y=18)
                    if name:
                        result_badge(scene, "THROUGH THE PIPE!" if name == "success" else name.upper())
                    panel = draw_actuators(env, k, PANEL_W, H)
                    panel_caption(panel, ["PPO drives the 6 motors (+ elevator), not the joints",
                                          f"[{mode} pipe] episode {ep + 1}/{args.episodes}   t = {step * 0.02:4.2f} s   "
                                          f"stage {info['stage_name']}",
                                          f"tip offset {info['lat_mm']:5.1f} mm   depth {max(info['depth_mm'], 0):5.1f} mm"
                                          f"   elevator {info['elevator_mm']:+6.1f} mm"])
                    frame = np.hstack([scene, panel])
                    for _ in range(int(1.5 * FPS) if name else (1 if step % 2 == 0 else 0)):
                        ff.stdin.write(frame.tobytes())
                    ended = [i for i in range(env.num_envs) if o[i] >= 0]
                    if ended:
                        obs = env.reset_envs(ended)
                print(f"[{mode}] episode {ep + 1}: {name} after {step} steps", flush=True)
    ff.stdin.close()
    ff.wait()


# ---------------------------------------------------------------------------
def run_student(out):
    from cable_stages import CableVisionPipeEnv, CableVisionPipeEnvCfg
    from student_policy import load_student
    from vision_pipeline import GIVE_UP, PHASE_NAMES, SearchInsertController

    modes = ["rig", "random"] if args.pipe_mode == "both" else [args.pipe_mode]
    cfg = configure(CableVisionPipeEnvCfg(), CLOSE_VIEW)
    cfg.scene.num_envs = len(modes)
    cfg.pipe_modes = tuple(modes)
    cfg.assist_prob = 0.0
    env = CableVisionPipeEnv(cfg)
    env.auto_reset = False
    student = load_student(os.path.join(TASK_DIR, args.student), device=env.device)
    ctrl = SearchInsertController(env, actor=None, student=student)
    obs, _ = env.reset()
    ctrl.reset(range(env.num_envs))
    for _ in range(3):
        scene_frame(env)
    ff = open_writer(out)
    steps_txt = (("LOOK / LIFT / SCAN", "tip camera finds the pipe mouth"),
                 ("APPROACH", "motors bend the arm to a pre-insertion pose"),
                 ("INSERT", "image-based student drives the 6 motors"))
    with torch.inference_mode():
        for k, mode in enumerate(modes):
            set_scene_view(env, env_index=k)
            if k > 0:                                           # this env ran unseen meanwhile: start it afresh
                obs = env.reset_envs([k])
                ctrl.reset([k])
            for ep in range(args.episodes):
                step, name = 0, ""
                while not name:
                    obs, _, _, _, extras = env.step(ctrl.step_actions(obs))
                    step += 1
                    o = extras["done_info"]["outcome"].cpu().numpy()
                    name = OUTCOMES[o[k]] if o[k] >= 0 else ("not_found" if ctrl.phase[k] == GIVE_UP else "")
                    if step % 2 and not name:
                        ended = [i for i in range(env.num_envs) if o[i] >= 0 or ctrl.phase[i] == GIVE_UP]
                        if ended:
                            obs = env.reset_envs(ended)
                            ctrl.reset(ended)
                        continue
                    info = env.info(k)
                    phase = PHASE_NAMES[ctrl.phase[k]]
                    scene = scene_image(env, CLOSE_VIEW[0])
                    banner(scene, ["Tip camera + image-based student on the 6 lead-screw motors",
                                   f"[{mode} pipe] episode {ep + 1}/{args.episodes}   t = {step * 0.02:4.2f} s   phase {phase}",
                                   f"tip offset {info['lat_mm']:5.1f} mm   depth {max(info['depth_mm'], 0):5.1f} mm"], y=18)
                    if name:
                        result_badge(scene, "THROUGH THE PIPE!" if name == "success" else name.upper().replace("_", " "),
                                     top=False)
                    rgb, *_ = env.tip_camera()
                    tip = cv2.cvtColor((rgb[k].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    tip = cv2.resize(tip, (360, 360), interpolation=cv2.INTER_LINEAR)
                    cv2.rectangle(tip, (0, 0), (360, 30), (18, 20, 26), -1)
                    cv2.putText(tip, "TIP CAMERA (all the student sees)", (8, 21), FONT, 0.5, (0, 230, 255), 1, AA)
                    row = np.full((360, W - PANEL_W - 360, 3), (26, 24, 20), np.uint8)
                    cv2.putText(row, "PIPELINE", (24, 36), FONT, 0.55, (160, 172, 190), 1, AA)
                    active = {"LOOK": 0, "LIFT": 0, "SCAN": 0, "RETURN": 1, "REFINE": 1, "APPROACH": 1,
                              "INSERT": 2}.get(phase, -1)
                    for j, (t1, t2) in enumerate(steps_txt):
                        y = 64 + 92 * j
                        cv2.rectangle(row, (18, y), (row.shape[1] - 18, y + 76),
                                      (110, 70, 30) if j == active else (44, 30, 22), -1)
                        cv2.putText(row, f"{j + 1}  {t1}", (32, y + 32), FONT, 0.65,
                                    (255, 230, 0) if j == active else (150, 132, 120), 2, AA)
                        cv2.putText(row, t2, (58, y + 60), FONT, 0.55, (255, 255, 255) if j == active else
                                    (170, 180, 195), 1, AA)
                    left = np.vstack([scene, np.hstack([tip, row])])
                    frame = np.hstack([left, draw_actuators(env, k, PANEL_W, H)])
                    for _ in range(int(1.5 * FPS) if name else 1):
                        ff.stdin.write(frame.tobytes())
                    ended = [i for i in range(env.num_envs) if o[i] >= 0 or ctrl.phase[i] == GIVE_UP]
                    if ended:
                        obs = env.reset_envs(ended)
                        ctrl.reset(ended)
                print(f"[{mode}] episode {ep + 1}: {name} after {step} steps (found by {ctrl.found_by[k] or '-'})",
                      flush=True)
    ff.stdin.close()
    ff.wait()


def main():
    out = args.output or os.path.join(TASK_DIR, "videos", f"cable_{args.mode}.mp4")
    {"demo": run_demo, "ppo": run_ppo, "student": run_student}[args.mode](out)
    print(f"video: {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
