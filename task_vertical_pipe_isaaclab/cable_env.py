"""
==============================================================================
cable_env.py - The vertical-pipe task with the real tendon drive (Isaac Lab)
==============================================================================
Same task, observation, reward, termination and resets as vertical_pipe_env.VerticalPipeEnv, but the
30 bending joints are no longer position-controlled. They are passive (a weak backbone spring +
damping) and the robot is bent only by CABLE TENSION, like the real machine (cable_model.py):

    action (7) = speed of the 6 lead-screw motors (section order, one antagonistic cable pair each)
                 + elevator speed
    motor -> free length of its cable pair -> stretch of the cable path -> one-way tension
    -> forces / torques on every disc (PhysX, once per physics step) -> the joints bend

The cable path is a polyline through the holes of the discs; the length and the tension are
recomputed from the PhysX link poses at every 2 ms physics step (the motors move linearly inside
the 20 ms env step). Nothing about the bend is commanded: the shape of every section comes from the
cable forces, the backbone and the contact with the pipe wall.

The observation is the 33 numbers of the joint env, with the 6 "bend command" entries replaced by the
bend that the motor positions correspond to (first-order model, cable_model.bend_to_motor_matrix),
optionally + 6 cable-pair tension differences (cfg.obs_tension, a load-cell signal on the real robot).
==============================================================================
"""

from __future__ import annotations

import numpy as np
import torch

import constants as C
import mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_quat, matrix_from_quat

import cable_model as CM
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg

TENSION_OBS_SCALE = 20.0       # N, scale of the tension-difference observation
VIZ_COLORS = ((1.0, 0.55, 0.10), (0.25, 0.85, 0.35), (0.25, 0.70, 1.0))     # section 1 / 2 / 3


@configclass
class CableVerticalPipeEnvCfg(VerticalPipeEnvCfg):
    # -- cables (cable_model.py); k, c, preload are guesses to be tuned on the robot
    cable_stiffness: float = 2.0e4         # N/m of the whole cable path (cable + sheath + screw)
    cable_damping: float = 150.0           # N s/m
    cable_preload: float = 5.0             # N in both cables of a pair at rest
    cable_max_tension: float = 300.0       # N, numerical guard
    # -- passive joints: weak spring / damper of the backbone; it only spreads the section bend evenly
    backbone_stiffness: float = 2.0        # N m/rad per joint
    backbone_damping: float = C.JOINT_DAMPING
    backbone_rest_at_start: bool = True    # spring rest angle = start pose (no start transient); False: straight
    obs_tension: bool = False              # +6 observation entries: cable pair tension differences
    # -- drawing (play / video only): the cables as coloured lines, brightness = tension
    draw_cables: bool = False
    draw_cables_max_envs: int = 16
    draw_cables_offset: float = 0.006      # m outside the hole circle, so the lines are not hidden by the CAD housing

    def sync(self):
        super().sync()
        self.scene.robot.actuators["bend"] = ImplicitActuatorCfg(
            joint_names_expr=["Seg.*_[xy]"], stiffness=self.backbone_stiffness, damping=self.backbone_damping,
            effort_limit_sim=C.BEND_FORCE_LIMIT, velocity_limit_sim=1.0e4, armature=C.JOINT_ARMATURE, friction=0.0,
        )
        self.observation_space = mdp.OBS_DIM + (CM.N_MOTORS if self.obs_tension else 0)


class CableVerticalPipeEnv(VerticalPipeEnv):
    cfg: CableVerticalPipeEnvCfg

    _STATE_KEYS = VerticalPipeEnv._STATE_KEYS + ("motor_cmd", "_s_prev", "_bb_target", "tension")

    def __init__(self, cfg: CableVerticalPipeEnvCfg, render_mode: str | None = None, **kwargs):
        self._cable_ready = False          # the base class may write states / observations before the buffers exist
        super().__init__(cfg, render_mode, **kwargs)
        n, dev, f64 = self.num_envs, self.device, torch.float64
        self.routing = CM.CableRouting(self.kin)
        self.cable = CM.CableForceModel(self.routing, dev, cfg.cable_stiffness, cfg.cable_damping, cfg.cable_preload,
                                        cfg.cable_max_tension)
        self._cable_ids = torch.as_tensor(self.robot.find_bodies(CM.BODY_NAMES, preserve_order=True)[0], device=dev)
        self._com_b = self._read_com_offsets()
        self._wrench = torch.zeros(2, n, self.robot.num_bodies, 3, device=dev)
        self._M = torch.as_tensor(CM.bend_to_motor_matrix(), dtype=f64, device=dev)
        self._Minv = torch.linalg.inv(self._M)
        self._rate = torch.as_tensor(CM.motor_rate(), dtype=f64, device=dev)
        self._lim = torch.as_tensor(CM.motor_limit(), dtype=f64, device=dev)
        self.motor_cmd = torch.zeros(n, CM.N_MOTORS, dtype=f64, device=dev)       # lead-screw displacement (m)
        self._s_prev = torch.zeros_like(self.motor_cmd)
        self._bb_target = torch.zeros(n, self.robot.num_joints, device=dev)         # backbone spring rest angles
        self.tension = torch.full((n, CM.N_CABLES), float(cfg.cable_preload), device=dev)
        self._fresh = torch.zeros(n, dtype=torch.bool, device=dev)                  # reset this step: link velocities stale
        self._viz = None
        self._cable_ready = True

    def _read_com_offsets(self):
        """Centre of mass of the cable bodies in their link frames (N, 16, 3)."""
        try:
            com = self.robot.data.body_com_pos_b
        except AttributeError:
            com = self.robot.root_physx_view.get_coms().to(self.device)[..., :3]
        return com[:, self._cable_ids].clone()

    # ------------------------------------------------------------------
    # Motors <-> bend (first order, cable_model.bend_to_motor_matrix)
    # ------------------------------------------------------------------
    def _bend_from_motor(self, s):
        return (s @ self._Minv.T).reshape(-1, 3, 2)

    def _motor_from_bend(self, bend):
        return bend.reshape(-1, CM.N_MOTORS) @ self._M.T

    # ------------------------------------------------------------------
    # Action -> motor commands
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions.to(torch.float64), -1.0, 1.0)
        self._s_prev = self.motor_cmd.clone()
        s = torch.clamp(self.motor_cmd + self.actions[:, :CM.N_MOTORS] * self._rate, -self._lim, self._lim)
        bend = self._bend_from_motor(s)                     # same limit as the joint env: |bend| <= THETA_MAX per section
        theta = torch.linalg.norm(bend, dim=2, keepdim=True)
        bend = bend * torch.clamp(C.THETA_MAX / torch.clamp(theta, min=1e-9), max=1.0)
        self.motor_cmd.copy_(self._motor_from_bend(bend))
        self.bend_cmd.copy_(bend)
        self.elev_cmd.copy_(torch.clamp(self.elev_cmd + self.actions[:, 6] * C.ELEV_RATE, C.ELEV_RANGE[0], C.ELEV_RANGE[1]))
        self._write_targets()

    def _write_targets(self, env_ids=None):
        if not self._cable_ready:
            return super()._write_targets(env_ids)
        self._targets.copy_(self._bb_target)                # bending joints: passive, spring only
        self._targets[:, self._jelev[0]] = self.elev_cmd.float()

    # ------------------------------------------------------------------
    # Physics: cable forces at every physics step
    # ------------------------------------------------------------------
    def _link_state(self):
        view = self.robot.root_physx_view
        ids = self._cable_ids
        tf, vel = view.get_link_transforms()[:, ids], view.get_link_velocities()[:, ids]
        n = tf.shape[0]
        pos = tf[..., :3] - self.scene.env_origins[:, None, :]
        rot = matrix_from_quat(convert_quat(tf[..., 3:7].reshape(-1, 4), to="wxyz")).view(n, len(ids), 3, 3)
        com = pos + torch.einsum("nbij,nbj->nbi", rot, self._com_b)
        return pos, rot, com, vel[..., :3], vel[..., 3:]

    def _apply_cable_wrench(self, s):
        pos, rot, com, v, w = self._link_state()
        if self._fresh.any():                               # just reset: the link velocities are those of the old episode
            v, w = v.clone(), w.clone()
            v[self._fresh] = 0.0
            w[self._fresh] = 0.0
            self._fresh[:] = False
        force, torque, self.tension, _ = self.cable.compute(pos, rot, com, v, w, s.to(torch.float32))
        self._wrench.zero_()
        self._wrench[0][:, self._cable_ids] = force
        self._wrench[1][:, self._cable_ids] = torque
        # world-frame force / torque about the COM of every body, for the next physics step
        self.robot.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self._wrench[0].view(-1, 3), torque_data=self._wrench[1].view(-1, 3), position_data=None,
            indices=self.robot._ALL_INDICES, is_global=True)

    def _simulate(self):
        if not self._cable_ready:
            return super()._simulate()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        n = self.cfg.decimation
        for i in range(n):
            s = self._s_prev + (self.motor_cmd - self._s_prev) * ((i + 1) / n)     # the screws turn at constant speed
            self._apply_cable_wrench(s)
            self._sim_step_counter += 1
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
        self.scene.update(dt=self.step_dt)
        if self.cfg.draw_cables:
            self._draw_cables()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _observations_from(self, meas):
        obs = super()._observations_from(meas)
        if not self._cable_ready or not self.cfg.obs_tension:
            return obs
        diff = (self.tension[:, 0::2] - self.tension[:, 1::2]) / TENSION_OBS_SCALE
        return torch.cat([obs, torch.clamp(diff, -5.0, 5.0)], dim=1)

    # ------------------------------------------------------------------
    # Start state: joints at the sampled pose, motors at the displacement that leaves the cables just
    # as tight as their preload (exact cable geometry of the start pose)
    # ------------------------------------------------------------------
    def _write_state(self, idx):
        if not self._cable_ready:
            return super()._write_state(idx)
        f64 = torch.float64
        self._write_root(idx)
        qpos = self.kin.qpos_from(self.bend_cmd[idx].cpu().numpy(), self.elev_cmd[idx].cpu().numpy())
        self.motor_cmd[idx] = torch.as_tensor(self.routing.motors_of_qpos(qpos), dtype=f64, device=self.device)
        self._s_prev[idx] = self.motor_cmd[idx]
        self.bend_cmd[idx] = self._bend_from_motor(self.motor_cmd[idx])
        self.tension[idx] = float(self.cfg.cable_preload)
        self._fresh[idx] = True

        joint_pos = torch.zeros(len(idx), self.robot.num_joints, device=self.device)
        joint_pos[:, self._q_to_joint] = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=idx)
        self._bb_target[idx] = joint_pos if self.cfg.backbone_rest_at_start else 0.0
        self._write_targets()
        self.robot.set_joint_position_target(self._targets[idx], env_ids=idx)
        pose = torch.zeros(len(idx), 7, device=self.device)
        pose[:, :2] = (self.pipe_xy[idx] + self.scene.env_origins[idx, :2].to(f64)).float()
        pose[:, 2] = (self.z_top[idx] + self.scene.env_origins[idx, 2].to(f64)).float()
        pose[:, 3] = 1.0
        self.pipe.write_root_pose_to_sim(pose, env_ids=idx)
        # the link poses / cable holes must follow the written joint state before the next physics step
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.get_physics_sim_view().update_articulations_kinematic()

    def restore(self, snap):
        super().restore(snap)
        self._fresh[:] = True

    # ------------------------------------------------------------------
    # Info / HUD: the motor angles are the real ones now
    # ------------------------------------------------------------------
    def info(self, i: int) -> dict:
        d = super().info(i)
        s = self.motor_cmd[i].cpu().numpy()
        d["motor_angles"] = -s[[0, 1, 4, 5, 2, 3]] / C.LEAD_SCREW_PITCH * 360.0      # deg, order of constants.MOTOR_ALPHAS
        d["cable_tension"] = self.tension[i].cpu().numpy()                              # N, cable 2m = +, 2m + 1 = -
        return d

    # ------------------------------------------------------------------
    # Drawing: 12 coloured polylines per robot through the cable holes (play / video)
    # ------------------------------------------------------------------
    def _make_viz(self):
        from pxr import Gf, UsdGeom, Vt

        stage = self.sim.stage
        n_env = min(self.num_envs, self.cfg.draw_cables_max_envs)
        routing = CM.CableRouting(self.kin, radial_offset=self.cfg.draw_cables_offset)
        hole = torch.as_tensor(routing.hole, dtype=torch.float32, device=self.device)
        UsdGeom.Xform.Define(stage, "/World/CableViz")
        curves = []
        for e in range(n_env):
            row = []
            for c in range(CM.N_CABLES):
                n_pts = int(routing.anchor[c]) + 1
                curve = UsdGeom.BasisCurves.Define(stage, f"/World/CableViz/env_{e}/cable_{c}")
                curve.CreateTypeAttr("linear")
                curve.CreateCurveVertexCountsAttr(Vt.IntArray([n_pts]))
                curve.CreateWidthsAttr(Vt.FloatArray([0.0016]))
                curve.SetWidthsInterpolation("constant")
                curve.CreatePointsAttr(Vt.Vec3fArray(n_pts))
                curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)]))
                row.append((curve.GetPointsAttr(), curve.GetDisplayColorAttr(), n_pts))
            curves.append(row)
        self._viz = {"hole": hole, "curves": curves, "n_env": n_env, "anchor": routing.anchor,
                     "section": CM.MOTOR_SECTION[routing.cable_motor]}

    def _draw_cables(self):
        from pxr import Gf, Sdf, Vt

        if self._viz is None:
            self._make_viz()
        viz, n_env = self._viz, self._viz["n_env"]
        pos, rot, _, _, _ = self._link_state()
        pts = (pos[:n_env, :, None, :] + torch.einsum("nbij,bcj->nbci", rot[:n_env], viz["hole"])
               + self.scene.env_origins[:n_env, None, None, :]).cpu().numpy()          # (n, 16, 12, 3)
        tension = self.tension[:n_env].cpu().numpy()
        with Sdf.ChangeBlock():
            for e in range(n_env):
                for c in range(CM.N_CABLES):
                    points_attr, color_attr, n_pts = viz["curves"][e][c]
                    points_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts[e, :n_pts, c], dtype=np.float32)))
                    base = VIZ_COLORS[viz["section"][c]]
                    b = 0.25 + 0.75 * min(float(tension[e, c]) / (4.0 * self.cfg.cable_preload), 1.0)   # slack = dark
                    color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(base[0] * b, base[1] * b, base[2] * b)]))
