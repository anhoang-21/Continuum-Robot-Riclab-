"""
==============================================================================
app_launch.py - AppLauncher arguments for Isaac Lab 3 (env_isaacsim62)
==============================================================================
Isaac Lab 3 has no --headless / --enable_cameras flags: the Kit window opens only when a
Kit visualizer is requested, and the launcher reads "headless", "enable_cameras" and
"video" from the arguments it is given. The scripts' own --video (MP4 written here)
must not reach it: the launcher's video recorder needs MoviePy.

    app = AppLauncher(launcher_args(args, gui=..., cameras=...)).app

gui=True also needs scene_view_cfg(headless=False) in cfg.sim.visualizer_cfgs (vertical_pipe_env.py).
No Isaac imports here: used before Kit starts.
==============================================================================
"""


def launcher_args(args, gui=False, cameras=True):
    return {**vars(args), "enable_cameras": cameras, "headless": not gui, "video": False,
            "visualizer_intent": {"has_kit_visualizer": gui}}
