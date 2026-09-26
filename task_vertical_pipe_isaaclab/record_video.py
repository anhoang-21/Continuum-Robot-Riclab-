"""
==============================================================================
record_video.py - Two-panel MP4 in Isaac Sim, in the style of the MuJoCo demo
(demo_vertical_pipe.py --fixed-camera: wide frontal view + close-up of the pipe entrance, HUD)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe record_video.py                      # 12 random pipes
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe record_video.py --pipe-mode rig --episodes 5
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe record_video.py --checkpoint models\\ppo_vpipe\\best_model.pt

Episode k uses the pipe placement of the MuJoCo demo for the same seed (reset seed = --seed + k), so
the scenes match videos/vertical_pipe_random_wide.mp4 of the MuJoCo version.
==============================================================================
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file

import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Record a two-panel MP4 of the policy in Isaac Sim")
parser.add_argument("--checkpoint", type=str, default=os.path.join("models", "mujoco_ppo_vpipe_wide", "model.pt"))
parser.add_argument("--pipe-mode", choices=["rig", "random"], default="random")
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--max-offset", type=float, default=0.17)
parser.add_argument("--slowdown", type=float, default=2.0, help=">1 plays slower than real time (fps = 50 / slowdown)")
parser.add_argument("--hold", type=float, default=1.2, help="seconds to hold the final pose of each episode")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=800)
parser.add_argument("--cam-azimuth", type=float, default=0.0, help="azimuth of the fixed wide camera (deg, MuJoCo convention)")
parser.add_argument("--cam-distance", type=float, default=0.8)
parser.add_argument("--output", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

import constants as C  # noqa: E402
from agent_cfg import make_agent_cfg  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg  # noqa: E402

STAGE_COLORS = {"ALIGN": (0, 200, 255), "INSERT": (255, 200, 0), "EXIT": (80, 255, 80)}   # BGR
FAILURE_TEXT = {"rim_hit": "HIT THE PIPE RIM", "missed_pipe": "MISSED THE PIPE", "unstable": "SIM UNSTABLE",
                "timeout": "TIME OUT"}


def load_actor(path):
    cfg = dict(make_agent_cfg(1, 1)["actor"])
    cfg.pop("class_name")
    cfg["distribution_cfg"] = dict(cfg["distribution_cfg"])
    actor = MLPModel(TensorDict({"policy": torch.zeros(1, 33)}, batch_size=[1]),
                     {"actor": ["policy"], "critic": ["policy"]}, "actor", 7, **cfg)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"])
    return actor.eval()


def mujoco_eye(lookat, distance, azimuth, elevation):
    """Camera position of a MuJoCo free camera (azimuth / elevation in degrees)."""
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(lookat) - distance * forward


class Camera:
    """USD camera + replicator RGB annotator (vertical field of view 45 deg like the MuJoCo free camera)."""

    def __init__(self, stage, path, width, height):
        cam = UsdGeom.Camera.Define(stage, path)
        focal = 24.0
        v_ap = 2.0 * focal * np.tan(np.deg2rad(45.0) / 2.0)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateVerticalApertureAttr(v_ap)
        cam.CreateHorizontalApertureAttr(v_ap * width / height)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        self.path = path
        self.product = rep.create.render_product(path, (width, height))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annotator.attach([self.product])
        self.width, self.height = width, height

    def look(self, eye, target):
        set_camera_view(eye=np.asarray(eye, dtype=float), target=np.asarray(target, dtype=float), camera_prim_path=self.path)

    def image(self):
        data = self.annotator.get_data()
        if data is None or data.size == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return cv2.cvtColor(np.ascontiguousarray(data[:, :, :3]), cv2.COLOR_RGB2BGR)


def close_up(env, azimuth):
    """Near the pipe mouth, turned (as little as possible) so the plate hole is not in front of the lens."""
    pipe_xy = env.pipe_xy[0].cpu().numpy()
    to_hole = C.PLATE_HOLE_XY - pipe_xy
    for delta in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        a = np.deg2rad(azimuth + delta)
        if to_hole @ np.array([np.cos(a), np.sin(a)]) > -0.02:
            azimuth += delta
            break
    lookat = np.array([pipe_xy[0], pipe_xy[1], float(env.z_top[0]) - 0.015])
    return mujoco_eye(lookat, 0.40, azimuth, -25.0), lookat


def draw_hud(frame, env, ep, n_eps, step, successes, finished, panel_w):
    info = env.info(0)
    h = frame.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1 = 360
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (x1, 226), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (x1, 226), (90, 95, 110), 1)

    def put(text, y, color=(230, 230, 230), scale=0.42, thick=1):
        cv2.putText(frame, text, (20, y), font, scale, color, thick, cv2.LINE_AA)

    put("PPO policy (Isaac Sim): continuum robot through a vertical pipe", 30, (0, 230, 255), 0.4)
    put(f"Episode {ep}/{n_eps}   step {step:3d}   t = {step * 0.02:4.2f} s   "
        f"success {successes}/{ep - (0 if finished else 1)}", 52)
    stage = info["stage_name"]
    put(f"Stage: {stage}", 76, STAGE_COLORS.get(stage, (230, 230, 230)), 0.55, 2)
    put(f"Tip offset from pipe axis: {info['lat_mm']:5.1f} mm   tilt {info['tilt_deg']:4.1f} deg", 98)
    bore_mm = float(env.z_top[0] - env.z_success[0]) * 1000.0
    depth = float(np.clip(info["depth_mm"], 0.0, bore_mm))
    put(f"Depth: {max(info['depth_mm'], 0.0):5.1f} / {bore_mm:.0f} mm", 120)
    bx0, bx1 = 190, x1 - 14
    cv2.rectangle(frame, (bx0, 110), (bx1, 122), (80, 80, 80), 1)
    cv2.rectangle(frame, (bx0 + 1, 111), (bx0 + 1 + int((bx1 - bx0 - 2) * depth / bore_mm), 121), (80, 220, 80), -1)
    clear = f"{info['clearance_mm']:4.1f} mm" if np.isfinite(info["clearance_mm"]) else "  --"
    touch_color = (60, 170, 255) if info["inner_contacts"] > 0 else (230, 230, 230)
    put(f"Wall clearance: {clear}   wall-contact steps: {info['contact_steps']}", 142, touch_color)
    put(f"Elevator: {info['elevator_mm']:+6.1f} mm  (motor {info['elevator_motor_deg']:+7.0f} deg)", 164)
    m = info["motor_angles"]
    put("Lead-screw motors (deg):", 188, (170, 200, 240))
    put(f"[{m[0]:+5.0f} {m[1]:+5.0f} {m[2]:+5.0f} {m[3]:+5.0f} {m[4]:+5.0f} {m[5]:+5.0f}]", 210, (170, 200, 240))
    if finished:
        ok = finished == "success"
        text = "THROUGH THE PIPE!" if ok else FAILURE_TEXT.get(finished, finished.upper())
        color = (40, 170, 40) if ok else (40, 40, 200)
        (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
        bx, by = (panel_w - tw) // 2 - 20, h - 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + tw + 40, by + th + 26), color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, text, (bx + 20, by + th + 12), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def make_markers():
    def mat(rgb, opacity=1.0):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, emissive_color=tuple(0.4 * c for c in rgb), opacity=opacity)

    cfg = VisualizationMarkersCfg(prim_path="/Visuals/pipe_guides", markers={
        "tip_free": sim_utils.SphereCfg(radius=0.006, visual_material=mat((0.1, 0.9, 1.0))),
        "tip_contact": sim_utils.SphereCfg(radius=0.006, visual_material=mat((1.0, 0.6, 0.1))),
        "tip_success": sim_utils.SphereCfg(radius=0.006, visual_material=mat((0.2, 1.0, 0.3))),
        "tip_failure": sim_utils.SphereCfg(radius=0.006, visual_material=mat((1.0, 0.15, 0.15))),
        "axis": sim_utils.CylinderCfg(radius=0.0012, height=1.0, visual_material=mat((0.3, 1.0, 1.0), 0.35)),
        "target": sim_utils.CylinderCfg(radius=C.PLATE_HOLE_RADIUS, height=0.0016, visual_material=mat((0.2, 1.0, 0.35), 0.45)),
    })
    return VisualizationMarkers(cfg)


def update_markers(markers, env, finished):
    """Pipe axis, exit target and a status-coloured tip marker (VerticalPipeEnv.add_overlay)."""
    pipe_xy = env.pipe_xy[0].cpu().numpy()
    z_top, z_succ = float(env.z_top[0]), float(env.z_success[0])
    origin = env.scene.env_origins[0].cpu().numpy()
    tip = env._meas["tip"][0].cpu().numpy()
    if finished:
        tip_idx = 2 if finished == "success" else 3
    else:
        tip_idx = 1 if int(env._meas["n_inner"][0]) > 0 else 0
    top = z_top + 0.12
    trans = np.array([tip, [pipe_xy[0], pipe_xy[1], 0.5 * (top + z_succ)], [pipe_xy[0], pipe_xy[1], z_succ]]) + origin
    scales = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, top - z_succ], [1.0, 1.0, 1.0]])
    markers.visualize(translations=torch.as_tensor(trans, dtype=torch.float32),
                      scales=torch.as_tensor(scales, dtype=torch.float32),
                      marker_indices=torch.as_tensor([tip_idx, 4, 5]))


def main():
    actor = load_actor(args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(TASK_DIR, args.checkpoint))
    cfg = VerticalPipeEnvCfg()
    cfg.scene.num_envs = 1
    cfg.pipe_modes = (args.pipe_mode,)
    cfg.random_offset = (0.0, args.max_offset)
    cfg.assist_prob = 0.0
    cfg.visual = True
    cfg.sim.device = args.device
    env = VerticalPipeEnv(cfg)
    env.auto_reset = False

    panel_w = args.width // 2
    stage = env.sim.stage
    # darker backdrop and a floor like the MuJoCo scene (visual only)
    dome = stage.GetPrimAtPath("/World/light")
    if dome.IsValid():
        dome.GetAttribute("inputs:color").Set(Gf.Vec3f(0.30, 0.34, 0.42))
        dome.GetAttribute("inputs:intensity").Set(700.0)
    floor = UsdGeom.Mesh.Define(stage, "/World/VideoFloor")
    s = 4.0
    floor.CreatePointsAttr([Gf.Vec3f(-s, -s, 0.0), Gf.Vec3f(s, -s, 0.0), Gf.Vec3f(s, s, 0.0), Gf.Vec3f(-s, s, 0.0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.21, 0.24)])
    wide = Camera(stage, "/World/VideoCameraWide", panel_w, args.height)
    near = Camera(stage, "/World/VideoCameraNear", panel_w, args.height)
    markers = make_markers()
    origin = env.scene.env_origins[0].cpu().numpy()
    axis_xy = env.kin.axis_xy
    wide_lookat = np.array([axis_xy[0], axis_xy[1], 0.64])
    wide.look(mujoco_eye(wide_lookat, args.cam_distance, args.cam_azimuth, -18.0) + origin, wide_lookat + origin)

    out = args.output or os.path.join(TASK_DIR, "videos", f"isaac_vertical_pipe_{args.pipe_mode}_wide.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fps = 50.0 / args.slowdown
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))

    def grab(ep, step, successes, finished):
        update_markers(markers, env, finished)
        env.sim.render()
        left = draw_hud(wide.image(), env, ep, args.episodes, step, successes, finished, panel_w)
        right = near.image()
        overlay = right.copy()
        cv2.rectangle(overlay, (8, 10), (250, 42), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.72, right, 0.28, 0, right)
        cv2.putText(right, "close-up: pipe entrance", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)
        frame = np.concatenate([left, right], axis=1)
        cv2.line(frame, (panel_w, 0), (panel_w, args.height), (40, 40, 40), 2)
        return frame

    successes = 0
    with torch.inference_mode():
        env.reset()
        for ep in range(1, args.episodes + 1):
            env._rngs[0] = np.random.default_rng(args.seed + ep)     # same scene as the MuJoCo demo episode
            obs = env.reset_envs([0])
            eye, lookat = close_up(env, args.cam_azimuth)
            near.look(eye + origin, lookat + origin)
            if ep == 1:
                for _ in range(20):                                   # let the renderer load the meshes
                    update_markers(markers, env, "")
                    env.sim.render()
            step, finished = 0, ""
            frame = grab(ep, step, successes, finished)
            writer.write(frame)
            while not finished:
                act = actor(TensorDict({"policy": obs["policy"].cpu()}, batch_size=[1])).to(env.device)
                obs, _, _, _, extras = env.step(act)
                step += 1
                outcome = int(extras["done_info"]["outcome"][0])
                if outcome >= 0:
                    finished = OUTCOMES[outcome]
                    successes += int(finished == "success")
                frame = grab(ep, step, successes, finished)
                writer.write(frame)
            for _ in range(int(args.hold * fps)):
                writer.write(frame)
            info = env.info(0)
            print(f"[{args.pipe_mode}] episode {ep}/{args.episodes}: {finished.upper():12s} steps {step:3d} | "
                  f"wall-contact steps {info['contact_steps']:3d} | max depth {info['max_depth_mm']:5.1f} mm | "
                  f"pipe {np.round(env.pipe_xy[0].cpu().numpy(), 3)}")
    writer.release()
    print(f"\nSuccess {successes}/{args.episodes} | video: {os.path.abspath(out)} ({os.path.getsize(out) / 2**20:.1f} MB)")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
