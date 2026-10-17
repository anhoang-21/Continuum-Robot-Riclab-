"""
==============================================================================
cable_env.py - vertical-pipe task with the robot driven by its 12 cables and 6 motors
==============================================================================
Same task as vertical_pipe_env.py (scene, pipe, reward, termination, observation layout,
start-state sampling), but the robot is actuated like the real one:

  * the 30 bending joints have NO position servo: a passive spring k_j (elastic cardan
    backbone) + damping, target 0;
  * 6 lead-screw motors (motors.py: Ezi-Servo EzM-42M / EzM-56M, 8 mm lead) in closed-loop
    position mode. Every physics step (2 ms) the motor dynamics are integrated against the
    cable load on the nut, with the pull-out torque limit of the motor;
  * 12 cables (cables.py): tension from the nut positions and the arm pose (one-sided, with
    pre-tension and damping), applied as the joint torques -G^T T of cables routed through
    every disc. Cables of sections 2 / 3 load the sections below them (cable coupling);
  * gravity on the arm (the elevator carriage is held by its screw; its motor torque is
    computed from the drive force + the carriage weight);
  * the elevator stays a position-controlled lead-screw stage.

Action (7, in [-1, 1]):
  action_mode = "motor" (default): 6 motor speeds (M1 M2 | M5 M6 | M3 M4, cables.MOTOR_NAMES) x
      cables.MOTOR_RATE per 20 ms step + elevator speed. The policy has to handle the cable
      coupling, elasticity and gravity itself.
  action_mode = "bend": 3 bend-vector rates as in the joint env; the lab's controller formula
      (cables.compensation_matrix) turns them into motor targets. The observation is then the
      joint env's, so a policy trained there runs unchanged on the cable-driven robot.
Observation (33): the joint env's layout; in "motor" mode the 6 bend-command entries hold the
motor-nut positions (encoders) / (d * THETA_MAX) instead.

extras["drive"] (every step): motor torque (N m), nut load (N), speed (rpm), cable tensions (N),
elevator motor torque (N m), motor alarms -> analyze_torque.py.
==============================================================================
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import cables as CB
import constants as C
import mdp
import motors as MT
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg

CABLE_COLORS = ((0.95, 0.40, 0.15), (0.20, 0.80, 0.35), (0.25, 0.55, 1.00), (0.55, 0.55, 0.55))   # s1 s2 s3 slack


@configclass
class CableVerticalPipeEnvCfg(VerticalPipeEnvCfg):
    action_mode: str = "motor"                  # "motor" | "bend"
    # bend-driven envs (action_mode "bend", or envs switched with set_drive_mode): commanded bend -> motor targets
    #   "formula": the lab's controller (thesis eq. 3.32, rigid cables)
    #   "model":   cables.CableModel.motors_for_pose (elastic cables + backbone + gravity; scripted motions)
    bend_map: str = "formula"
    one_sided_cables: bool = True               # False: linear cables that can also push
    gravity: bool = True                        # gravity on the arm
    cable_stiffness: float = CB.CABLE_STIFFNESS
    cable_damping: float = CB.CABLE_DAMPING
    cable_pretension: float = CB.CABLE_PRETENSION
    backbone_stiffness: float = CB.BACKBONE_STIFFNESS
    backbone_damping: float = CB.BACKBONE_DAMPING
    motor_bandwidth_hz: float = 60.0            # closed-loop stepper position loop
    cuda_graph: bool = True                     # replay the per-physics-step drive computation as a CUDA graph
    show_cables: bool = False                   # draw the 12 cables through the disc holes (window / video)
    cable_draw_radius: float = 0.0009           # m, drawn cable radius (the steel wires are ~1 mm thick)
    show_actuators: bool = False                # with show_cables: draw the 6 lead-screw modules above the arm

    def sync(self):
        if self.action_mode not in ("motor", "bend"):
            raise ValueError(f"action_mode must be 'motor' or 'bend', got {self.action_mode!r}")
        if self.bend_map not in ("formula", "model"):
            raise ValueError(f"bend_map must be 'formula' or 'model', got {self.bend_map!r}")
        super().sync()
        bend = self.scene.robot.actuators["bend"]
        bend.stiffness = self.backbone_stiffness          # passive spring towards the straight pose
        bend.damping = self.backbone_damping
        bend.effort_limit_sim = 1.0e3
        self.sim.gravity = (0.0, 0.0, -9.81) if self.gravity else (0.0, 0.0, 0.0)
        self.scene.robot.spawn.rigid_props.disable_gravity = not self.gravity


class CableVerticalPipeEnv(VerticalPipeEnv):
    cfg: CableVerticalPipeEnvCfg
    _STATE_KEYS = VerticalPipeEnv._STATE_KEYS + ("motor_cmd", "_motor_cmd_prev")

    def __init__(self, cfg: CableVerticalPipeEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.sync()
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.geom = CB.CableGeometry(self.kin)
        self.gravity_model = CB.GravityModel(self.geom) if cfg.gravity else None
        kw = dict(one_sided=cfg.one_sided_cables, k_c=cfg.cable_stiffness, t0=cfg.cable_pretension,
                  k_j=cfg.backbone_stiffness, c_c=cfg.cable_damping)
        self.cable_model = CB.CableModel(self.geom, device=dev, **kw)
        self._cable_model_cpu = CB.CableModel(self.geom, device="cpu", gravity=self.gravity_model, **kw)
        self.motors = MT.CableMotorBank(n, dev, bandwidth_hz=cfg.motor_bandwidth_hz)
        # bending joints in cable-chain order (Seg1_x, Seg1_y, Seg2_x, ...) -> PhysX joint ids
        self._jchain = torch.as_tensor(self.robot.find_joints(self.geom.joint_names, preserve_order=True)[0],
                                       dtype=torch.long, device=dev)
        f64 = torch.float64
        self.motor_cmd = torch.zeros(n, CB.N_MOTORS, dtype=f64, device=dev)
        self._motor_cmd_prev = torch.zeros(n, CB.N_MOTORS, dtype=f64, device=dev)
        self.tension = torch.zeros(n, CB.N_CABLES, dtype=f64, device=dev)
        self.elev_torque = torch.zeros(n, dtype=f64, device=dev)
        self.bend_meas = torch.zeros(n, 3, 2, dtype=f64, device=dev)
        self._efforts = torch.zeros(n, self.robot.num_joints, device=dev)
        self._motor_scale = torch.as_tensor(C.CABLE_RADIUS[CB.MOTOR_SECTION] * C.THETA_MAX, dtype=f64, device=dev)
        # static inputs of the per-physics-step drive computation (_cable_math)
        self._g_q = torch.zeros(n, 30, dtype=f64, device=dev)
        self._g_qd = torch.zeros(n, 30, dtype=f64, device=dev)
        self._g_frac = torch.zeros(2, dtype=f64, device=dev)
        # per env: the 7-D action is bend rates (True) or motor speeds (False)
        self.bend_driven = torch.full((n,), cfg.action_mode == "bend", dtype=torch.bool, device=dev)
        self._drive_mix = "bend" if cfg.action_mode == "bend" else "motor"      # "bend" | "motor" | "mixed"
        self._cable_graph = self._capture_cable_graph()
        self._markers = self._make_cable_markers() if cfg.show_cables else None

    @property
    def motor(self):
        """Motor-nut positions (N, 6) in m (encoder readings); + = cable a paid out."""
        return self.motors.x

    # ------------------------------------------------------------------
    # Scene: gravity acts on the arm, not on the elevator carriage (its screw holds it; weight in motors.py)
    # ------------------------------------------------------------------
    def _setup_scene(self):
        super()._setup_scene()
        if not self.cfg.gravity:
            return
        from pxr import PhysxSchema

        stage = self.sim.stage
        for env_path in self.scene.env_prim_paths:
            prim = stage.GetPrimAtPath(f"{env_path}/Robot/base_link/Elevator_Link")
            if not prim.IsValid():
                raise RuntimeError(f"Elevator_Link not found in {env_path}/Robot")
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

    # ------------------------------------------------------------------
    # Cable / motor state
    # ------------------------------------------------------------------
    def joint_q(self):
        """The 30 bending joint angles in cable-chain order (N, 30)."""
        return self.robot.data.joint_pos[:, self._jchain].to(torch.float64)

    def measured_bend(self):
        """Bend vectors (N, 3, 2) of the simulated pose (inverse of kinematics.joint_angles)."""
        q = self.joint_q().view(-1, 3, 5, 2).sum(dim=2)                      # per section: (sum x, sum y)
        return torch.stack([q[..., 1], -q[..., 0]], dim=-1)

    def cable_tension(self):
        """Tensions of the 12 cables (N, 12) in N at the end of the last physics step."""
        return self.tension

    # ------------------------------------------------------------------
    # Actions -> motor targets
    # ------------------------------------------------------------------
    def set_drive_mode(self, env_ids, bend: bool):
        """
        Interpret the action of these envs as bend rates (bend=True: scripted search / approach motions,
        turned into motor targets by cfg.bend_map) or as motor speeds (bend=False: the policies).
        """
        self.bend_driven[torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)] = bend
        n_bend = int(self.bend_driven.sum())
        self._drive_mix = "bend" if n_bend == self.num_envs else ("motor" if n_bend == 0 else "mixed")

    def bend_to_motor(self, bend):
        """Commanded bend vectors (k, 3, 2) -> motor targets (k, 6), with cfg.bend_map."""
        if self.cfg.bend_map == "formula":
            return self.cable_model.compensate(bend)
        q = torch.as_tensor(self.geom.q_bend(bend.cpu().numpy()))
        return self._cable_model_cpu.motors_for_pose(q).to(self.device)

    def command_features(self):
        """What the robot knows of its own actuators (N, 6): motor encoders / (d THETA_MAX), or the bend commands."""
        if self.cfg.action_mode == "bend":
            return self.bend_cmd.reshape(self.num_envs, 6)
        return self.motor / self._motor_scale

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions.to(torch.float64), -1.0, 1.0)
        self._motor_cmd_prev.copy_(self.motor_cmd)
        if self._drive_mix == "bend":
            mdp.update_commands(self.bend_cmd, self.elev_cmd, self.actions)
            self.motor_cmd[:] = self.bend_to_motor(self.bend_cmd)
        elif self._drive_mix == "motor":
            self.motor_cmd += self.actions[:, :6] * self.cable_model.u_rate
            self.elev_cmd.copy_(torch.clamp(self.elev_cmd + self.actions[:, 6] * C.ELEV_RATE, *C.ELEV_RANGE))
        else:
            ids = self.bend_driven.nonzero().squeeze(-1)
            bend, elev = self.bend_cmd[ids].clone(), self.elev_cmd[ids].clone()
            mdp.update_commands(bend, elev, self.actions[ids])
            self.bend_cmd[ids], self.elev_cmd[ids] = bend, elev
            self.motor_cmd[ids] = self.bend_to_motor(bend)
            ids = (~self.bend_driven).nonzero().squeeze(-1)
            self.motor_cmd[ids] += self.actions[ids, :6] * self.cable_model.u_rate
            self.elev_cmd[ids] = torch.clamp(self.elev_cmd[ids] + self.actions[ids, 6] * C.ELEV_RATE, *C.ELEV_RANGE)
        u_max = self.cable_model.u_max
        self.motor_cmd[:] = torch.maximum(torch.minimum(self.motor_cmd, u_max), -u_max)
        self._write_targets()

    def _write_targets(self, env_ids=None):
        self._targets[:, self._jchain] = 0.0                                 # backbone springs: straight pose
        self._targets[:, self._jelev[0]] = self.elev_cmd.float()

    # ------------------------------------------------------------------
    # Physics: motors -> cables -> joint torques, every 2 ms
    # ------------------------------------------------------------------
    def _cable_math(self):
        """
        One physics step of the drive on static buffers (_g_q, _g_qd, _g_frac): motor dynamics against the
        cable load, cable tensions, joint torques into _efforts. Everything is written in place, so on the GPU
        the whole function is captured once as a CUDA graph and replayed (a few hundred small kernels,
        ~15 ms per step on Windows when launched one by one).
        """
        model, motors = self.cable_model, self.motors
        Gq, Gqd = self._g_q @ model.G.T, self._g_qd @ model.G.T
        span = self.motor_cmd - self._motor_cmd_prev
        x0 = self._motor_cmd_prev + span * self._g_frac[0]
        x1 = self._motor_cmd_prev + span * self._g_frac[1]
        motors.step(self.physics_dt, x0, x1, lambda x, v: model.nut_load(model.tension_from(Gq, x, Gqd, v)))
        self.tension.copy_(model.tension_from(Gq, motors.x, Gqd, motors.v))
        self._efforts[:, self._jchain] = model.joint_torque(self.tension).float()

    def _capture_cable_graph(self):
        """Record _cable_math as a CUDA graph (GPU only); the state it mutates is restored afterwards."""
        if not str(self.device).startswith("cuda") or not self.cfg.cuda_graph:
            return None
        m = self.motors
        state = [m.x, m.v, m.integral, m.torque, m.load, m.error, m.alarm, self.tension, self._efforts]
        saved = [t.clone() for t in state]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):                                                   # warm-up (allocator, kernels)
                self._cable_math()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):         # PhysX may use CUDA meanwhile
            self._cable_math()
        for t, s in zip(state, saved):
            t.copy_(s)
        return graph

    def _cable_substep(self, frac0, frac1):
        view = self.robot.root_physx_view
        self._g_q.copy_(view.get_dof_positions()[:, self._jchain])
        self._g_qd.copy_(view.get_dof_velocities()[:, self._jchain])
        self._g_frac[0], self._g_frac[1] = frac0, frac1
        if self._cable_graph is not None:
            self._cable_graph.replay()
        else:
            self._cable_math()
        view.set_dof_actuation_forces(self._efforts, self.robot._ALL_INDICES)

    def step(self, action: torch.Tensor):
        """VerticalPipeEnv.step with the motor / cable model integrated at every physics step."""
        action = action.to(self.device)
        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._apply_action()
        self.scene.write_data_to_sim()
        dec = self.cfg.decimation
        for k in range(dec):
            self._cable_substep(k / dec, (k + 1) / dec)
            self._sim_step_counter += 1
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                if self._markers is not None:
                    self.scene.update(dt=self.physics_dt)
                    self._update_cable_markers()
                self.sim.render()
        self.scene.update(dt=self.step_dt)
        self._update_drive_info()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0 and self.auto_reset:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False):
        # headless video (rgb_array without RTX sensors) renders here, not inside step()
        if self._markers is not None:
            self._update_cable_markers()
        return super().render(recompute)

    def _update_drive_info(self):
        je = self._jelev[0]
        q_e = self.robot.data.joint_pos[:, je].to(torch.float64)
        v_e = self.robot.data.joint_vel[:, je].to(torch.float64)
        drive = C.ELEV_KP * (self.elev_cmd - q_e) - (C.ELEV_KV + C.JOINT_DAMPING) * v_e
        self.elev_torque = MT.elevator_torque(drive, v_e)
        m = self.motors
        self.extras["drive"] = {
            "motor_torque": m.torque.clone(), "nut_load": m.load.clone(),
            "motor_rpm": m.v / MT.CABLE_SCREW.lead * 60.0, "tension": self.tension.clone(),
            "elevator_torque": self.elev_torque.clone(), "elevator_force": drive + MT.ELEVATOR_MOVING_MASS * MT.G,
            "motor_alarm": m.alarm.clone(),
        }

    def _get_dones(self):
        out = super()._get_dones()
        if hasattr(self, "_jchain"):
            self.bend_meas = self.measured_bend()
        return out

    def _observations_from(self, meas):
        if self.cfg.action_mode == "bend" or not hasattr(self, "motors"):        # (parents may observe during init)
            return super()._observations_from(meas)
        pipe_xy, z_top, z_success = self.pipe_xy, self.z_top, self.z_success
        if getattr(self.cfg, "obs_source", "gt") == "vision" and getattr(self, "_est_ready", False):
            # vision_env.VisionPipeEnv: the pipe pose measured by the tip camera
            pipe_xy, z_top = self.pipe_xy_hat, self.z_top_hat
            z_success = z_top - self.bore_length - C.EXIT_MARGIN
        motor_obs = (self.motor / self._motor_scale).view(-1, 3, 2)
        return mdp.observation(meas, pipe_xy, z_top, z_success, motor_obs, self.elev_cmd, self.axis_xy,
                               self.prev_action)

    # ------------------------------------------------------------------
    # Start states: the sampled pose held by the cables, at rest
    # ------------------------------------------------------------------
    def _write_state(self, idx):
        """
        The motor positions that hold the sampled pose best (least squares), or with action_mode "bend" and
        bend_map "formula" the lab controller's motor positions for the commanded bend. Either way the robot
        starts in the static equilibrium of those motor positions (cables, backbone, gravity), at rest.
        bend_cmd keeps the sampled bend (the reference of bend-driven scripted motions).
        """
        self._write_root(idx)
        model = self._cable_model_cpu
        bend = self.bend_cmd[idx].cpu()
        q_des = torch.as_tensor(self.geom.q_bend(bend.numpy()))
        if self.cfg.action_mode == "bend" and self.cfg.bend_map == "formula":
            u = model.compensate(bend)
        else:
            u = model.motors_for_pose(q_des)
        u = torch.maximum(torch.minimum(u, model.u_max), -model.u_max)
        q = model.equilibrium(u, q0=q_des)                                    # the sampled pose is close to it
        u = u.to(self.device)
        self.motors.reset(idx, u)
        self.motor_cmd[idx] = u
        self._motor_cmd_prev[idx] = u
        joint_pos = self.robot.data.default_joint_pos[idx].clone()
        joint_pos[:, self._jchain] = q.to(self.device).float()
        joint_pos[:, self._jelev[0]] = self.elev_cmd[idx].float()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=idx)
        self._write_targets()
        self.robot.set_joint_position_target(self._targets[idx], env_ids=idx)
        q_dev = q.to(self.device)
        self.tension[idx] = self.cable_model.tension(q_dev, u)
        bend_meas = q_dev.view(-1, 3, 5, 2).sum(dim=2)
        self.bend_meas[idx] = torch.stack([bend_meas[..., 1], -bend_meas[..., 0]], dim=-1)
        pose = torch.zeros(len(idx), 7, device=self.device)
        pose[:, :2] = (self.pipe_xy[idx] + self.scene.env_origins[idx, :2].to(torch.float64)).float()
        pose[:, 2] = (self.z_top[idx] + self.scene.env_origins[idx, 2].to(torch.float64)).float()
        pose[:, 3] = 1.0
        self.pipe.write_root_pose_to_sim(pose, env_ids=idx)

    def snapshot(self):
        snap = super().snapshot()
        m = self.motors
        snap["motors"] = {k: getattr(m, k).clone() for k in ("x", "v", "integral", "alarm")}
        snap["tension"] = self.tension.clone()
        return snap

    def restore(self, snap):
        super().restore(snap)
        for k, v in snap["motors"].items():
            getattr(self.motors, k).copy_(v)
        self.tension.copy_(snap["tension"])

    def set_motors(self, env_ids, u):
        """Move the motor targets of some envs to u (m) at once (tests / teleoperation); the motors follow."""
        u = torch.as_tensor(u, dtype=torch.float64, device=self.device)
        self.motor_cmd[env_ids] = u
        self._motor_cmd_prev[env_ids] = u

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def info(self, i: int) -> dict:
        out = super().info(i)
        u = self.motor[i].cpu().numpy()
        bend = self.bend_meas[i].cpu().numpy()
        m = self.motors
        out.update({
            "motor_mm": u * 1000.0,                                          # M1 M2 M5 M6 M3 M4
            "motor_angles": CB.motor_angles_deg(u),                          # controller order M1..M6
            "motor_torque": m.torque[i].cpu().numpy(),                       # N m, M1 M2 M5 M6 M3 M4
            "motor_holding": m.hold.cpu().numpy(),
            "nut_load": m.load[i].cpu().numpy(),
            "motor_alarm": m.alarm[i].cpu().numpy(),
            "cable_tension": self.tension[i].cpu().numpy(),
            "elevator_torque": float(self.elev_torque[i]),
            "bend_deg": np.rad2deg(np.linalg.norm(bend, axis=1)),
            "bend_dir_deg": np.rad2deg(np.arctan2(bend[:, 1], bend[:, 0])) % 360.0,
        })
        return out

    # ------------------------------------------------------------------
    # Cable drawing: one thin cylinder per span between two holes
    # ------------------------------------------------------------------
    def _make_cable_markers(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        body_names = self.robot.body_names
        bodies, local, span_a, span_b, span_cable = [], [], [], [], []
        for c, pts in enumerate(self.geom.paths):
            first = len(bodies)
            for b, p in pts:
                bodies.append(body_names.index(self.kin.body_names[b]))
                local.append(p)
            for k in range(len(pts) - 1):
                span_a.append(first + k)
                span_b.append(first + k + 1)
                span_cable.append(c)
        t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)  # noqa: E731
        self._mk_body, self._mk_local = t(bodies), t(np.array(local), torch.float32)
        self._mk_a, self._mk_b, self._mk_cable = t(span_a), t(span_b), t(span_cable)
        self._mk_section = t(CB.CABLE_SECTION)[self._mk_cable]
        self._mk_root = t([sum(len(p) for p in self.geom.paths[:c]) for c in range(CB.N_CABLES)])  # root hole per cable
        markers = {
            f"cable{k}": sim_utils.CylinderCfg(
                radius=self.cfg.cable_draw_radius, height=1.0, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col, metallic=0.6, roughness=0.35))
            for k, col in enumerate(CABLE_COLORS)
        }
        if self.cfg.show_actuators:
            self._make_actuator_geometry()
            for k, col in enumerate(CABLE_COLORS[:3]):                            # 4-6: nuts of sections 1-3
                markers[f"nut{k}"] = sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="Z",
                                                           visual_material=sim_utils.PreviewSurfaceCfg(
                                                               diffuse_color=col, metallic=0.3, roughness=0.4))
            markers["flag"] = sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.85, 0.1), emissive_color=(0.5, 0.4, 0.0)))  # 7
            markers["screw"] = sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="Z",   # 8: screw, drawn over the CAD one
                                                     visual_material=sim_utils.PreviewSurfaceCfg(
                                                         diffuse_color=(0.18, 0.19, 0.22), metallic=0.9, roughness=0.25))
            self._hide_cad_nuts()
        return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/Cables", markers=markers))

    # The lead-screw drive of the CAD (Base_Arm mesh of the MJCF, on the elevator carriage): tension box on the
    # plate over the arm, six vertical 8 mm lead screws in a cage, the motors on top. Measured from the mesh at
    # the rest pose, relative to the first universal joint (root of the arm): screw axes at 70 mm from the arm
    # axis, from 170 to 468 mm above the root; the mesh has static nuts at 339-374 mm.
    CAD_SCREW_R = 0.0701
    CAD_SCREW_DEG = (0.0, 60.0, 90.0, 150.0, 210.0, 300.0)
    CAD_SCREW_Z = (0.170, 0.468)
    CAD_NUT_Z = (0.333, 0.382)
    NUT_TRAVEL = 0.115        # m of drawn nut travel for the full motor range (the real screws are 400 mm)

    def _make_actuator_geometry(self):
        """
        Screw axes (bottom point) of every motor in the Elevator_Link frame (the drive rides with the elevator);
        every motor gets the CAD screw closest to its motor angle.
        """
        k = self.kin
        xpos, xmat = k.forward(np.zeros(k.nq))
        e, s1 = k.body_index["Elevator_Link"], self.geom.seg_bodies[0]
        R = xmat[e]
        free = list(self.CAD_SCREW_DEG)
        base = np.zeros((CB.N_MOTORS, 3))
        for m in np.argsort([min(abs((np.degrees(a) - s + 180) % 360 - 180) for s in free) for a in CB.MOTOR_ALPHA]):
            a = np.degrees(CB.MOTOR_ALPHA[m])
            s = min(free, key=lambda d: abs((a - d + 180) % 360 - 180))
            free.remove(s)
            off = np.array([self.CAD_SCREW_R * np.cos(np.radians(s)), self.CAD_SCREW_R * np.sin(np.radians(s)),
                            self.CAD_SCREW_Z[0]])
            base[m] = R.T @ (xpos[s1] + off - xpos[e])
        self._screw_axes_world_rest = base
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)  # noqa: E731
        self._act_base = t(base)
        self._act_body = self.robot.body_names.index("Elevator_Link")

    def _hide_cad_nuts(self):
        """Remove the static nuts of the CAD screws from the visual mesh (the drawn nuts move instead)."""
        from pxr import Usd, UsdGeom, Vt

        k = self.kin
        xpos, _ = k.forward(np.zeros(k.nq))
        root = xpos[self.geom.seg_bodies[0]]
        axes = np.array([[self.CAD_SCREW_R * np.cos(np.radians(s)), self.CAD_SCREW_R * np.sin(np.radians(s))]
                         for s in self.CAD_SCREW_DEG]) + root[:2]
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for i, env_path in enumerate(self.scene.env_prim_paths):
            prim = self.sim.stage.GetPrimAtPath(f"{env_path}/Robot/base_link/Elevator_Link/visuals/Base_Arm/Base_Arm")
            if not prim.IsValid():
                continue                                                          # collider-only robot
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            G = cache.GetLocalToWorldTransform(prim)
            M = np.array([[G[r][c] for c in range(4)] for r in range(4)])          # row-vector convention
            w = np.c_[pts, np.ones(len(pts))] @ M
            w = w[:, :3] - self.scene.env_origins[i].cpu().numpy()
            dist = np.min(np.linalg.norm(w[:, None, :2] - axes[None], axis=2), axis=1)
            dz = w[:, 2] - root[2]
            inside = (dz > self.CAD_NUT_Z[0]) & (dz < self.CAD_NUT_Z[1]) & (dist > 0.0062) & (dist < 0.05)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            face = np.repeat(np.arange(len(counts)), counts)
            drop = np.ones(len(counts), dtype=bool)
            np.logical_and.at(drop, face, inside[idx])                           # every vertex in the nut region
            keep = ~drop
            keep_fv = keep[face]
            # per-face-vertex / per-face data (normals, uvs) must shrink with the faces
            attrs = [mesh.GetNormalsAttr()] + [pv.GetAttr() for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars()]
            for attr in attrs:
                val = attr.Get() if attr and attr.HasAuthoredValue() else None
                if val is None:
                    continue
                interp = UsdGeom.Primvar(attr).GetInterpolation() if UsdGeom.Primvar.IsPrimvar(attr) \
                    else mesh.GetNormalsInterpolation()
                mask = keep_fv if interp == UsdGeom.Tokens.faceVarying else keep if interp == UsdGeom.Tokens.uniform \
                    else None
                pv = UsdGeom.Primvar(attr) if UsdGeom.Primvar.IsPrimvar(attr) else None
                if mask is not None and pv is not None and pv.IsIndexed():
                    ind = np.asarray(pv.GetIndices())
                    if len(ind) == len(mask):
                        pv.SetIndices(Vt.IntArray.FromNumpy(ind[mask].astype(np.int32)))
                    continue
                if mask is not None and len(val) == len(mask):
                    attr.Set(type(val).FromNumpy(np.asarray(val)[mask]))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts[keep].astype(np.int32)))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(idx[keep_fv].astype(np.int32)))
            if i == 0:
                print(f"[cable_env] removed {int(drop.sum())} faces of the static CAD nuts", flush=True)

    def _actuator_instances(self, pts):
        """(translations, orientations, scales, indices) of the moving parts: 6 nuts, 6 flags, 12 cables."""
        from isaaclab.utils.math import quat_apply

        n, dev = self.num_envs, self.device
        pos_e = self.robot.data.body_link_pos_w[:, self._act_body]
        quat_e = self.robot.data.body_link_quat_w[:, self._act_body][:, None].expand(n, 6, 4)
        up_w = torch.tensor([0.0, 0.0, 1.0], device=dev)
        base = pos_e[:, None] + quat_apply(quat_e, self._act_base[None].expand(n, -1, -1))    # (N, 6, 3)
        L = self.CAD_SCREW_Z[1] - self.CAD_SCREW_Z[0]
        x = self.motor.float()                                                               # (N, 6)
        rng = torch.as_tensor(CB.MOTOR_RANGE, dtype=torch.float32, device=dev)
        nut = base + up_w * (0.5 * L - self.NUT_TRAVEL * x / rng)[..., None]                  # up = cable a in
        top = base + up_w * L
        angle = x / MT.CABLE_SCREW.lead * 2.0 * math.pi                                      # rotor angle
        flag_dir = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], -1)
        ident = torch.zeros(n, 6, 4, device=dev)
        ident[..., 0] = 1.0
        q_flag = torch.stack([torch.cos(angle / 2), torch.zeros_like(angle), torch.zeros_like(angle),
                              torch.sin(angle / 2)], -1)
        sec = torch.as_tensor(CB.MOTOR_SECTION, device=dev)[None].expand(n, -1)
        parts = [
            (base + up_w * 0.5 * L, ident, torch.tensor([0.0065, 0.0065, L], device=dev), 8),  # screws (dark steel)
            (nut, ident, torch.tensor([0.02, 0.02, 0.026], device=dev), None),                 # nuts (round)
            # flag on the coupling at the top of the screw: turns with the rotor (8 mm of nut travel per turn)
            (top + up_w * 0.008 + 0.013 * flag_dir, q_flag, torch.tensor([0.028, 0.006, 0.006], device=dev), 7),
        ]
        tr, ori, sc, idx = [], [], [], []
        for p, q, s, k in parts:
            tr.append(p)
            ori.append(q)
            sc.append(s.expand(n, 6, 3))
            idx.append(4 + sec if k is None else torch.full((n, 6), k, device=dev, dtype=torch.long))
        # the cable pair runs from the nut down along the screw into the tension box (and on to the arm)
        tangent = torch.stack([-torch.sin(torch.as_tensor(CB.MOTOR_ALPHA, device=dev)),
                               torch.cos(torch.as_tensor(CB.MOTOR_ALPHA, device=dev)),
                               torch.zeros(6, device=dev)], -1).float()                      # (6, 3)
        side = torch.as_tensor(CB.CABLE_SIGN, device=dev).float()[None, :, None]            # a / b on either side
        a = nut[:, CB.CABLE_MOTOR] + 0.009 * side * tangent[CB.CABLE_MOTOR][None]            # (N, 12, 3)
        b = base[:, CB.CABLE_MOTOR] + 0.009 * side * tangent[CB.CABLE_MOTOR][None] - up_w * 0.004
        cab = self._spans(a, b)
        slack = self.tension < 0.05
        cable_idx = torch.where(slack, 3, torch.as_tensor(CB.CABLE_SECTION, device=dev)[None].expand(n, -1))
        tr.append(cab[0])
        ori.append(cab[1])
        sc.append(cab[2])
        idx.append(cable_idx)
        return (torch.cat(tr, 1), torch.cat(ori, 1), torch.cat(sc, 1), torch.cat(idx, 1))

    @staticmethod
    def _spans(a, b):
        """Cylinder (height 1 along z) instances from a to b: translations, orientations, scales."""
        v = b - a
        length = torch.linalg.norm(v, dim=-1).clamp(min=1e-6)
        d = v / length[..., None]
        d = d * torch.where(d[..., 2:3] < 0.0, -1.0, 1.0)                        # a cylinder is symmetric
        q = torch.stack([1.0 + d[..., 2], -d[..., 1], d[..., 0], torch.zeros_like(d[..., 0])], dim=-1)
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
        scale = torch.ones_like(v)
        scale[..., 2] = length
        return 0.5 * (a + b), q, scale

    def _update_cable_markers(self):
        from isaaclab.utils.math import quat_apply

        n = self.num_envs
        pos = self.robot.data.body_link_pos_w[:, self._mk_body]                  # (N, P, 3)
        quat = self.robot.data.body_link_quat_w[:, self._mk_body]
        pts = pos + quat_apply(quat, self._mk_local[None].expand(n, -1, -1))
        tr, ori, scale = self._spans(pts[:, self._mk_a], pts[:, self._mk_b])      # (N, S, .)
        slack = self.tension < 0.05                                              # (N, 12)
        idx = torch.where(slack[:, self._mk_cable], 3, self._mk_section[None].expand(n, -1))
        if self.cfg.show_actuators:
            a_tr, a_ori, a_sc, a_idx = self._actuator_instances(pts)
            tr, ori = torch.cat([tr, a_tr], 1), torch.cat([ori, a_ori], 1)
            scale, idx = torch.cat([scale, a_sc], 1), torch.cat([idx, a_idx], 1)
        self._markers.visualize(translations=tr.reshape(-1, 3), orientations=ori.reshape(-1, 4),
                                scales=scale.reshape(-1, 3), marker_indices=idx.reshape(-1))
