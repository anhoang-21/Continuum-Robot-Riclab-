"""
==============================================================================
multi_pipe_env.py - Several pipes, an instruction, an overview camera (Stage 5)
==============================================================================
VisionPipeEnv with two more (kinematic, collidable) pipes and a fixed third-person RGB-D
camera on the rig. Every episode: multi_pipe.MultiPipeSampler places the pipes, gives them
distinct tube colours and picks the target + an instruction; the target is the env's
physical "Pipe" (reward, success, contact classification as in the single-pipe task), the
others are "PipeB" / "PipeC". Touching any pipe outside the target's bore counts as a rim
hit (their contacts are in the same contact view).

  student_starts=True   resets at the student's pre-insertion pose for the target
                        (training of the student with other pipes around and random colours)
  overview=False        no overview camera (student training does not need it)
==============================================================================
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

import constants as C
from multi_pipe import PALETTE, TRAIN_TUBE, MultiPipeSampler, OverviewCamera
from vertical_pipe_env import CONTACT_OFFSET, PIPE_USD, REST_OFFSET
from vision_env import VisionPipeEnv, VisionPipeEnvCfg, VisionPipeSceneCfg

_OV = OverviewCamera()
_OV_FOCAL = 10.0


def _distractor(name):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=PIPE_USD["random"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 3.0, 0.0)),
    )


@configclass
class MultiPipeSceneCfg(VisionPipeSceneCfg):
    pipe_b: RigidObjectCfg = _distractor("PipeB")
    pipe_c: RigidObjectCfg = _distractor("PipeC")
    overview_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCam",
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(float(v) for v in _OV.EYE), rot=_OV.quat(), convention="ros"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_OV_FOCAL,
            horizontal_aperture=2.0 * _OV_FOCAL * math.tan(math.radians(_OV.HFOV_DEG) / 2.0),
            clipping_range=(0.02, 5.0),
        ),
        width=_OV.WIDTH,
        height=_OV.HEIGHT,
        depth_clipping_behavior="zero",
    )


@configclass
class MultiPipeEnvCfg(VisionPipeEnvCfg):
    pipe_modes: tuple = ("random",)
    max_episode_steps: int = 1000
    student_starts: bool = False
    overview: bool = True
    train_tube_prob: float = 0.0          # student training: chance that the target keeps the light-blue tube
    p_position: float = 0.5               # share of position instructions (when the scene allows one)
    visual: bool = True
    scene: MultiPipeSceneCfg = MultiPipeSceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=False)

    def sync(self):
        super().sync()
        if not self.overview:
            self.scene.overview_cam = None


class MultiPipeEnv(VisionPipeEnv):
    cfg: MultiPipeEnvCfg
    CONTACT_BODIES = ("Pipe", "PipeB", "PipeC")

    def __init__(self, cfg: MultiPipeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.msampler = MultiPipeSampler(self.kin, n_pipes=3, pipe_height=cfg.pipe_height,
                                         random_offset=cfg.random_offset, p_position=cfg.p_position)
        self.scenes = [None] * self.num_envs
        self.overview = OverviewCamera()
        self._multi_ready = True

    def _setup_scene(self):
        super()._setup_scene()
        self.pipe_b: RigidObject = self.scene["pipe_b"]
        self.pipe_c: RigidObject = self.scene["pipe_c"]
        self.overview_cam: TiledCamera | None = self.scene["overview_cam"] if self.cfg.overview else None

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        if not getattr(self, "_multi_ready", False):
            return super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        DirectRLEnv._reset_idx(self, env_ids)
        ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        k = len(ids)
        rngs = self._eval_rngs if getattr(self, "_eval", False) else self._rngs
        pipe_xy, z_top, bend, elev = np.zeros((k, 2)), np.zeros(k), np.zeros((k, 3, 2)), np.zeros(k)
        for j, i in enumerate(ids):
            for _ in range(20):
                sc = self.msampler.scene(rngs[i])
                st = self.msampler.student_start(rngs[i], sc) if self.cfg.student_starts \
                    else self.msampler.task_start(rngs[i], sc)
                if st is not None:
                    break
            self.scenes[i] = sc
            pipe_xy[j], z_top[j], bend[j], elev[j] = st.pipe_xy, st.z_top, st.bend, st.elev
        self.set_start_state(ids, pipe_xy, z_top, bend, elev)
        self._place_distractors(ids)
        self._paint(ids, rngs)

    def _place_distractors(self, ids):
        idx = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        origins = self.scene.env_origins[idx]
        for obj, slot in ((self.pipe_b, 0), (self.pipe_c, 1)):
            pose = torch.zeros(len(ids), 7, device=self.device)
            for j, i in enumerate(ids):
                sc = self.scenes[i]
                other = [p for p in range(3) if p != sc.target][slot]
                pose[j, :2] = torch.as_tensor(sc.pipe_xy[other], dtype=torch.float32, device=self.device)
                pose[j, 2] = float(sc.z_top[other])
            pose[:, :3] += origins
            pose[:, 3] = 1.0
            obj.write_root_pose_to_sim(pose, env_ids=idx)

    def _paint(self, ids, rngs):
        """Tube (and lower collar) colours of the three pipes, per env, on the USD stage."""
        from pxr import Gf, UsdGeom, Vt

        stage = self.sim.stage
        for i in ids:
            sc = self.scenes[i]
            others = [p for p in range(3) if p != sc.target]
            colors = {"Pipe": PALETTE[sc.colors[sc.target]], "PipeB": PALETTE[sc.colors[others[0]]],
                      "PipeC": PALETTE[sc.colors[others[1]]]}
            if self.cfg.train_tube_prob > 0.0 and rngs[i].uniform() < self.cfg.train_tube_prob:
                colors["Pipe"] = TRAIN_TUBE
            for body, rgb in colors.items():
                for part in ("pipe_tube", "pipe_collar_bot"):
                    prim = stage.GetPrimAtPath(f"{self.scene.env_prim_paths[i]}/{body}/visuals/{part}")
                    if prim.IsValid():
                        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))

    # ------------------------------------------------------------------
    def overview_camera(self):
        """rgb (N, H, W, 3) [0, 1], depth (N, H, W) m, K (3, 3), cam_pos (3,) and cam_rot (3, 3) in the env frame."""
        out = self.overview_cam.data.output
        rgb = out["rgb"][..., :3].float() / 255.0
        depth = torch.nan_to_num(out["distance_to_image_plane"][..., 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        return rgb, depth, self.overview.K, self.overview.EYE, self.overview.rot
