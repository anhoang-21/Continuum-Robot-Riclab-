"""
==============================================================================
robust_stages.py - the tip-camera / student / language envs on uncertain robots with the safety filter
==============================================================================
cable_stages.py with robust_env.RobustCableEnv in place of CableVerticalPipeEnv: the same Stage 4 / 5 scenes
and pipelines, on robots drawn from an uncertainty set (or the nominal one), with the safety signals of
robust_env.py and the hook for safety_filter.TensionCBF on every motor step (policy, student or the scripted
search / approach motions in bend mode).

    RobustCableVisionPipeEnv   = RobustCableEnv + VisionPipeEnv    (tip camera)
    RobustCableStudentPipeEnv  = RobustCableEnv + StudentPipeEnv   (student training starts)
    RobustCableMultiPipeEnv    = RobustCableEnv + MultiPipeEnv     (3 pipes, overview camera, language)

`attach_filter(env, source)` puts the CBF under whatever drives the env.
==============================================================================
"""

from __future__ import annotations

from isaaclab.utils import configclass

from multi_pipe_env import MultiPipeEnv, MultiPipeEnvCfg, MultiPipeSceneCfg
from robust_env import RobustCableEnv, RobustCableEnvCfg
from student_env import StudentPipeEnv, StudentPipeEnvCfg
from vision_env import VisionPipeEnv, VisionPipeEnvCfg, VisionPipeSceneCfg


@configclass
class RobustCableVisionPipeEnvCfg(RobustCableEnvCfg, VisionPipeEnvCfg):
    bend_map: str = "model"
    obs_source: str = "vision"
    max_episode_steps: int = 1000
    scene: VisionPipeSceneCfg = VisionPipeSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=False)


class RobustCableVisionPipeEnv(RobustCableEnv, VisionPipeEnv):
    cfg: RobustCableVisionPipeEnvCfg


@configclass
class RobustCableStudentPipeEnvCfg(RobustCableEnvCfg, StudentPipeEnvCfg):
    bend_map: str = "model"
    obs_source: str = "gt"
    max_episode_steps: int = 400
    scene: VisionPipeSceneCfg = VisionPipeSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=False)


class RobustCableStudentPipeEnv(RobustCableEnv, StudentPipeEnv):
    cfg: RobustCableStudentPipeEnvCfg


@configclass
class RobustCableMultiPipeEnvCfg(RobustCableEnvCfg, MultiPipeEnvCfg):
    bend_map: str = "model"
    obs_source: str = "vision"
    pipe_modes: tuple = ("random",)
    max_episode_steps: int = 1000
    visual: bool = True
    scene: MultiPipeSceneCfg = MultiPipeSceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=False)


class RobustCableMultiPipeEnv(RobustCableEnv, MultiPipeEnv):
    cfg: RobustCableMultiPipeEnvCfg


def attach_filter(env, source="loadcell", **kw):
    """The tension CBF on every motor step of env (returns it; env.safety_filter = None removes it)."""
    from safety_filter import TensionCBF

    env.safety_filter = TensionCBF(env, source=source, **kw)
    return env.safety_filter


def episode_safety_now(env, ids):
    """Safety statistics of the running episodes of envs ids (for episodes a pipeline ends, e.g. not grounded)."""
    import numpy as np

    out = []
    for i in ids:
        n = max(int(env.episode_length_buf[i]), 1)
        out.append({"viol_frac": float(env.ep["viol_steps"][i]) / n, "slack_frac": float(env.ep["slack_steps"][i]) / n,
                    "over_frac": float(env.ep["over_steps"][i]) / n, "t_max": float(env.ep["t_max"][i]),
                    "t_min": float(env.ep["t_min"][i]) if np.isfinite(float(env.ep["t_min"][i])) else float("nan"),
                    "f_max": float(env.ep["f_max"][i])})
    return out
