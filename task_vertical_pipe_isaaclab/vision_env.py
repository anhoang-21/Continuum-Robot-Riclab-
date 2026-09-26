"""
==============================================================================
vision_env.py - Vertical-pipe env with an eye-in-hand camera at the robot tip
==============================================================================
VerticalPipeEnv + a tiled RGB-D camera on Seg15, looking along the tip axis out of
the tip face (like an endoscope camera). Physics, action, reward and termination are
unchanged (the reward still uses the true pipe pose); only the pipe pose that the
policy sees can come from the camera:

  obs_source = "gt"      the 33-dim observation of VerticalPipeEnv (true pipe pose)
  obs_source = "vision"  the same observation, computed with the pipe pose estimated
                         from the tip camera (set_estimate), see perception.py /
                         vision_pipeline.py

The camera pose is computed from the Seg15 link pose and the fixed mounting offset
(what forward kinematics gives on the real robot), not read back from the renderer.
Needs --enable_cameras.
==============================================================================
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

import constants as C
import mdp
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg, VerticalPipeSceneCfg

CAM_WIDTH = CAM_HEIGHT = 160
CAM_FOV_DEG = 120.0                       # wide-angle endoscope camera
CAM_FOCAL = 10.0                          # mm (USD units); only the ratio focal / aperture matters
CAM_OFFSET = C.TIP_OFFSET + 0.0005        # camera centre 0.5 mm outside the tip face, on the tip axis


@configclass
class VisionPipeSceneCfg(VerticalPipeSceneCfg):
    # ROS convention (+z forward, +y down in the image); identity rotation = looking along Seg15 +z,
    # which points out of the tip face (down the robot)
    tip_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Seg15/TipCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, CAM_OFFSET), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM_FOCAL,
            horizontal_aperture=2.0 * CAM_FOCAL * math.tan(math.radians(CAM_FOV_DEG) / 2.0),
            clipping_range=(0.004, 5.0),
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        depth_clipping_behavior="zero",
    )


@configclass
class VisionPipeEnvCfg(VerticalPipeEnvCfg):
    obs_source: str = "vision"            # "gt" | "vision"
    max_episode_steps: int = 1000         # search + return + insertion
    depth_noise_std: float = 0.0          # m, Gaussian noise added to every depth pixel
    rgb_noise_std: float = 0.0            # 0..1 scale, Gaussian noise added to the colour image
    scene: VisionPipeSceneCfg = VisionPipeSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=False)


class VisionPipeEnv(VerticalPipeEnv):
    cfg: VisionPipeEnvCfg

    def __init__(self, cfg: VisionPipeEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.obs_source not in ("gt", "vision"):
            raise ValueError(f"obs_source must be 'gt' or 'vision', got {cfg.obs_source!r}")
        # the estimate buffers are read by the first observation inside VerticalPipeEnv.__init__
        self._est_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        n, dev, f64 = self.num_envs, self.device, torch.float64
        # prior before anything is seen: pipe on the robot axis, top at the rig height
        self.pipe_xy_hat = self.axis_xy.clone().contiguous()
        self.z_top_hat = torch.full((n,), C.PLATE_TOP_Z + C.PIPE_HEIGHT, dtype=f64, device=dev)
        self.est_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._est_ready = True
        self._cam_offset = torch.tensor([0.0, 0.0, CAM_OFFSET], dtype=f64, device=dev)

    def _setup_scene(self):
        super()._setup_scene()
        self.tip_cam: TiledCamera = self.scene["tip_cam"]

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def tip_camera(self):
        """
        Latest tip-camera frame of every env:
          rgb (N, H, W, 3) float in [0, 1], depth (N, H, W) m (0 = nothing within range),
          K (N, 3, 3), cam_pos (N, 3) env frame, cam_rot (N, 3, 3) camera->env (ROS axes).
        """
        out = self.tip_cam.data.output
        rgb = out["rgb"][..., :3].float() / 255.0
        depth = out["distance_to_image_plane"][..., 0].float()
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.rgb_noise_std > 0.0:
            rgb = torch.clamp(rgb + self.cfg.rgb_noise_std * torch.randn_like(rgb), 0.0, 1.0)
        if self.cfg.depth_noise_std > 0.0:
            depth = torch.where(depth > 0.0, depth + self.cfg.depth_noise_std * torch.randn_like(depth), depth)
        f64 = torch.float64
        rot = matrix_from_quat(self.robot.data.body_link_quat_w[:, self._tip_body].to(f64))
        pos = self.robot.data.body_link_pos_w[:, self._tip_body].to(f64) - self.scene.env_origins.to(f64)
        cam_pos = pos + torch.einsum("nij,j->ni", rot, self._cam_offset)
        return rgb, depth, self.tip_cam.data.intrinsic_matrices.to(f64), cam_pos, rot

    # ------------------------------------------------------------------
    # Estimated pipe pose
    # ------------------------------------------------------------------
    def set_estimate(self, ids, pipe_xy, z_top):
        idx = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.pipe_xy_hat[idx] = torch.as_tensor(pipe_xy, dtype=torch.float64, device=self.device).view(-1, 2)
        self.z_top_hat[idx] = torch.as_tensor(z_top, dtype=torch.float64, device=self.device).view(-1)
        self.est_valid[idx] = True

    def clear_prev_action(self, ids):
        """Start of the insertion: the policy was trained with prev_action = 0 at the start pose."""
        idx = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.prev_action[idx] = 0.0
        self.actions[idx] = 0.0

    def _observations_from(self, meas):
        if self.cfg.obs_source == "gt" or not self._est_ready:
            return super()._observations_from(meas)
        z_success_hat = self.z_top_hat - self.bore_length - C.EXIT_MARGIN
        return mdp.observation(meas, self.pipe_xy_hat, self.z_top_hat, z_success_hat, self.bend_cmd, self.elev_cmd,
                               self.axis_xy, self.prev_action)

    def set_start_state(self, ids, pipe_xy, z_top, bend, elev, assisted=None):
        super().set_start_state(ids, pipe_xy, z_top, bend, elev, assisted)
        if self._est_ready:
            idx = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
            self.pipe_xy_hat[idx] = self.axis_xy[idx]
            self.z_top_hat[idx] = C.PLATE_TOP_Z + C.PIPE_HEIGHT
            self.est_valid[idx] = False
