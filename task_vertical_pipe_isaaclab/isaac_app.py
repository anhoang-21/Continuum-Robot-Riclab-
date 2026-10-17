"""
==============================================================================
isaac_app.py - Kit start-up and viewer camera for Isaac Sim 6.2 / Isaac Lab 3.0
==============================================================================
Isaac Lab 3.0 changed what the scripts relied on in Isaac Lab 2.3:

  * the launcher has no --headless / --enable_cameras options any more, and Kit only
    opens a window when the Kit visualizer is requested (--viz kit)
    -> add_app_args() adds both options, launch() requests the Kit visualizer when the
       script shows the scene (window, or offscreen frames for an MP4)
  * env.render() returns None for render_mode="rgb_array" and env.viewer /
    env.viewport_camera_controller are gone (the viewer is the KitVisualizer)
    -> viewer_cfg() / look_at() / follow_env() / grab() below
  * the Kit visualizer copies the physics poses to the renderer and pumps the Kit app loop
    (headless: on demand, in grab()); markers are drawn only when it runs. Scripts with their
    own replicator cameras (record_video.py) call grab() to render a frame of all cameras
==============================================================================
"""

from __future__ import annotations

import argparse

import numpy as np


def add_app_args(parser: argparse.ArgumentParser) -> None:
    """Isaac Lab launcher options (--device, --viz, ...) plus --headless and --enable_cameras."""
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    for flag, text in (("--headless", "no window (Kit runs without a viewport)"),
                       ("--enable_cameras", "render camera sensors / offscreen frames")):
        try:
            parser.add_argument(flag, action="store_true", default=False, help=text)
        except argparse.ArgumentError:          # a later Isaac Lab defines it itself
            pass


def launch(args: argparse.Namespace, view: bool = False):
    """
    Start Kit. view=True: the script shows the scene, in a window (args.headless False) or as
    offscreen frames (args.headless True, needs args.enable_cameras) -> the Kit visualizer is started.
    """
    from isaaclab.app import AppLauncher

    if view and not getattr(args, "visualizer", None):
        args.visualizer = "kit"
    return AppLauncher(args).app


# ---------------------------------------------------------------------------
# Viewer camera (KitVisualizer)
# ---------------------------------------------------------------------------
def viewer_cfg(eye, lookat, resolution=(1280, 720), env_index=0):
    """
    Default Kit visualizer settings (the ViewerCfg of Isaac Lab 2.3): eye / lookat relative to the
    origin of env `env_index`. Used as SimulationCfg.default_visualizer_cfg, so it only applies when a
    Kit visualizer is started (--viz kit / launch(view=True)). None if isaaclab_visualizers is missing.
    """
    try:
        from isaaclab_visualizers.kit import KitVisualizerCfg
    except ImportError:
        return None
    return KitVisualizerCfg(eye=tuple(float(v) for v in eye), lookat=tuple(float(v) for v in lookat),
                            window_width=int(resolution[0]), window_height=int(resolution[1]),
                            origin_type="env", origin_env_index=int(env_index),
                            background_color=None)        # keep the dome light background of the scene


def set_viewer(cfg, eye=None, lookat=None, resolution=None, env_index=None):
    """Change the viewer settings of an env cfg (before the env is created)."""
    sim = cfg.sim
    if sim.default_visualizer_cfg is None:
        sim.default_visualizer_cfg = viewer_cfg((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    v = sim.default_visualizer_cfg
    if v is None:
        return
    if eye is not None:
        v.eye = tuple(float(x) for x in eye)
    if lookat is not None:
        v.lookat = tuple(float(x) for x in lookat)
    if resolution is not None:
        v.window_width, v.window_height = int(resolution[0]), int(resolution[1])
    if env_index is not None:
        v.origin_env_index = int(env_index)


def kit_visualizer(env):
    """The running KitVisualizer of the env's simulation, or None (no window / no --viz kit)."""
    for viz in getattr(env.sim, "visualizers", None) or []:
        if getattr(viz.cfg, "visualizer_type", None) == "kit":
            return viz
    return None


def look_at(env, eye, lookat):
    """Put the viewer camera at world position `eye`, looking at `lookat`."""
    viz = kit_visualizer(env)
    if viz is not None:
        viz.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in lookat))


def look_at_env(env, k, eye, lookat):
    """Viewer camera at `eye` looking at `lookat`, both in the frame of env k (env origin)."""
    origin = env.scene.env_origins[k].detach().cpu().numpy().astype(float)
    look_at(env, np.asarray(eye, dtype=float) + origin, np.asarray(lookat, dtype=float) + origin)


def follow_env(env, k):
    """Viewer eye / lookat of the cfg, relative to the origin of env k (set_view_env_index of Isaac Lab 2.3)."""
    viz = kit_visualizer(env)
    if viz is not None:
        look_at_env(env, k, viz.cfg.eye, viz.cfg.lookat)


def grab(env):
    """RGB frame (H, W, 3) uint8 of the viewer camera (env.render() of Isaac Lab 2.3 with rgb_array)."""
    viz = kit_visualizer(env)
    if viz is None:
        raise RuntimeError("no Kit visualizer: start the script with launch(args, view=True) (--viz kit)")
    return np.ascontiguousarray(viz.render_rgb_array())


def set_camera_look_at(stage, prim_path, eye, target) -> None:
    """Pose a USD camera at `eye` looking at `target`, +z up (isaacsim.core.utils.viewports.set_camera_view)."""
    from pxr import Gf, UsdGeom

    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*[float(v) for v in eye]), Gf.Vec3d(*[float(v) for v in target]),
                                   Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())
