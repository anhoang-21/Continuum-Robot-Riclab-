"""
==============================================================================
cable_stages.py - the tip-camera / student / language envs on the cable-driven robot
==============================================================================
The Stage 4 (image-based student) and Stage 5 (instruction -> local VLM -> student) envs of
the joint-servo robot, with the cable drive of cable_env.py underneath: 6 lead-screw motors,
12 cables, passive elastic backbone, gravity. Built by multiple inheritance, the cable env
first in the MRO:

    CableVisionPipeEnv   = CableVerticalPipeEnv + VisionPipeEnv    (tip camera)
    CableStudentPipeEnv  = CableVerticalPipeEnv + StudentPipeEnv   (student training starts)
    CableMultiPipeEnv    = CableVerticalPipeEnv + MultiPipeEnv     (3 pipes, overview camera)

The policies (PPO teacher, image-based student) command the 6 motors ("motor" actions). The
scripted parts of the pipelines (search, approach to the pre-insertion pose) still think in
bend vectors; for them the env turns the commanded bend into motor targets with the cable
model (bend_map "model": elastic cables, backbone, gravity), like a model-based controller of
the real robot. vision_pipeline.py switches every env between the two (set_drive_mode).

Dataclass fields that exist in both parents are declared again below, so that the Stage 4 / 5
values win (otherwise the cable config's inherited defaults would, e.g. a scene without cameras).
==============================================================================
"""

from __future__ import annotations

from isaaclab.utils import configclass

from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg
from multi_pipe_env import MultiPipeEnv, MultiPipeEnvCfg, MultiPipeSceneCfg
from student_env import StudentPipeEnv, StudentPipeEnvCfg
from vision_env import VisionPipeEnv, VisionPipeEnvCfg, VisionPipeSceneCfg

TEACHER = "models/ppo_cable_128/best_model.pt"         # PPO on the 6 motors (train_vertical_pipe.py --cable)


@configclass
class CableVisionPipeEnvCfg(CableVerticalPipeEnvCfg, VisionPipeEnvCfg):
    bend_map: str = "model"
    obs_source: str = "vision"
    max_episode_steps: int = 1000
    scene: VisionPipeSceneCfg = VisionPipeSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=False)


class CableVisionPipeEnv(CableVerticalPipeEnv, VisionPipeEnv):
    cfg: CableVisionPipeEnvCfg


@configclass
class CableStudentPipeEnvCfg(CableVerticalPipeEnvCfg, StudentPipeEnvCfg):
    bend_map: str = "model"
    obs_source: str = "gt"
    max_episode_steps: int = 400
    scene: VisionPipeSceneCfg = VisionPipeSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=False)


class CableStudentPipeEnv(CableVerticalPipeEnv, StudentPipeEnv):
    cfg: CableStudentPipeEnvCfg


@configclass
class CableMultiPipeEnvCfg(CableVerticalPipeEnvCfg, MultiPipeEnvCfg):
    bend_map: str = "model"
    obs_source: str = "vision"
    pipe_modes: tuple = ("random",)
    max_episode_steps: int = 1000
    visual: bool = True
    scene: MultiPipeSceneCfg = MultiPipeSceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=False)


class CableMultiPipeEnv(CableVerticalPipeEnv, MultiPipeEnv):
    cfg: CableMultiPipeEnvCfg
