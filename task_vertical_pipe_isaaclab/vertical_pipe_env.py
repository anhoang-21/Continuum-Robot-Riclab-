"""
==============================================================================
vertical_pipe_env.py - Isaac Lab port of the "continuum robot through a vertical
pipe" task (D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\vertical_pipe_env.py)
==============================================================================
Direct-workflow env (DirectRLEnv). One Isaac Lab env instance runs N copies of the
MuJoCo VerticalPipeEnv; env i uses cfg.pipe_modes[i % len(pipe_modes)], so
pipe_modes=("rig", "random") reproduces the "--pipe-mode mixed" split of the
MuJoCo trainer (even envs on the rig pipe, odd envs on a random pipe).

Same as the MuJoCo env:
  * sim dt 0.002 s, decimation 10 (= timestep 0.002 x N_SUBSTEPS 10), zero gravity
  * action (7), observation (33), reward, termination, truncation after 400 steps:
    mdp.py, checked against MuJoCo in tests/compare_with_mujoco.py
  * start-state sampling (pipe placement, assisted starts, 50 attempts):
    reset_logic.py, same numpy draws as the MuJoCo env (env i: default_rng(seed + i))
  * servos kp / kv, joint armature 0.01, damping 0.5, force limit 20 N m, friction 0.3

PhysX replacements (see README): position servos -> implicit joint drives,
mocap pipe -> kinematic rigid body, contact list -> PhysX contact points of the
robot/pipe pairs, mjWARN_BADQACC -> non-finite / exploding joint state,
`ncon == 0` of the start pose -> geometric test in kinematics.py.
==============================================================================
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

import constants as C
import mdp
from kinematics import ContinuumKinematics
from reset_logic import ResetSampler

USD_DIR = os.path.join(C.GENERATED_DIR, "usd")
ROBOT_USD = {
    "physics": os.path.join(USD_DIR, "continuum_physics", "continuum_physics.usd"),
    "visual": os.path.join(USD_DIR, "continuum_visual", "continuum_visual.usd"),
}
PIPE_USD = {mode: os.path.join(USD_DIR, f"pipe_{mode}.usd") for mode in C.PIPE_MODES}
SEG_NAMES = [f"Seg{i}" for i in range(1, 16)]

# PhysX collision margins (MuJoCo: margin 0, soft contacts). Contacts are generated inside
# CONTACT_OFFSET; only points closer than cfg.contact_touch_tolerance count as touching.
CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0


@configclass
class VerticalPipeSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD["physics"],
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, linear_damping=0.0, angular_damping=0.0,
                max_depenetration_velocity=1.0, enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,          # contype=1 / conaffinity=0: segments never touch each other
                solver_position_iteration_count=8, solver_velocity_iteration_count=1,
                sleep_threshold=0.0, stabilization_threshold=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                                                   joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={
            # MuJoCo: force = kp (ctrl - q) - kv qdot, plus passive joint damping 0.5 -> drive damping kv + 0.5
            "bend": ImplicitActuatorCfg(
                joint_names_expr=["Seg.*_[xy]"], stiffness=C.BEND_KP, damping=C.BEND_KV + C.JOINT_DAMPING,
                effort_limit_sim=C.BEND_FORCE_LIMIT, velocity_limit_sim=1.0e4,
                armature=C.JOINT_ARMATURE, friction=0.0,
            ),
            "elevator": ImplicitActuatorCfg(
                joint_names_expr=["Elevator_Joint"], stiffness=C.ELEV_KP, damping=C.ELEV_KV + C.JOINT_DAMPING,
                effort_limit_sim=C.ELEV_FORCE_LIMIT, velocity_limit_sim=1.0e4,
                armature=C.JOINT_ARMATURE, friction=0.0,
            ),
        },
    )
    pipe: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pipe",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[PIPE_USD["rig"], PIPE_USD["random"]],
            random_choice=False,                         # env i gets usd_path[i % len]
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(float(C.PLATE_HOLE_XY[0]), float(C.PLATE_HOLE_XY[1]),
                                                       C.PLATE_TOP_Z + C.PIPE_HEIGHT)),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.8, 0.82, 0.86)))
    sun: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.97, 0.92), angle=2.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.8660254, 0.25, 0.0, 0.4330127)))


@configclass
class VerticalPipeEnvCfg(DirectRLEnvCfg):
    # -- MuJoCo VerticalPipeEnv(...) arguments
    pipe_modes: tuple = ("rig", "random")   # per env, cyclic: ("rig",), ("random",) or ("rig", "random") = mixed
    max_episode_steps: int = 400
    pipe_height: float = C.PIPE_HEIGHT
    random_offset: tuple = (0.0, 0.17)
    assist_prob: float = 0.0
    contact_penalty: float = -C.R_CONTACT
    visual: bool = False                     # CAD meshes (play / video only; physics is identical)
    # -- PhysX-specific. MuJoCo lists a contact only when the geoms penetrate, and its soft contacts
    # let the tip sink ~0.5-1.3 mm into the rim when it is pushed onto it; PhysX contacts are rigid.
    contact_touch_tolerance: float = 1.0e-4  # separation (m) up to which an inner-wall point counts as contact
    rim_touch_tolerance: float = 1.0e-3      # separation (m) up to which a point outside inner_contact_r is a rim hit
    unstable_joint_vel: float = 1.0e3        # |qdot| above this (rad/s, m/s) is treated like mjWARN_BADQACC

    # -- env
    seed: int = 0
    decimation: int = C.N_SUBSTEPS
    episode_length_s: float = 400 * C.SIM_DT * C.N_SUBSTEPS
    action_space: int = mdp.ACT_DIM
    observation_space: int = mdp.OBS_DIM
    state_space: int = 0
    is_finite_horizon: bool = False

    sim: SimulationCfg = SimulationCfg(
        dt=C.SIM_DT,
        render_interval=C.N_SUBSTEPS,
        gravity=(0.0, 0.0, 0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=C.CONTACT_FRICTION, dynamic_friction=C.CONTACT_FRICTION, restitution=0.0),
        physx=PhysxCfg(solver_type=1, enable_stabilization=False),
    )
    scene: VerticalPipeSceneCfg = VerticalPipeSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=False)
    viewer: ViewerCfg = ViewerCfg(eye=(0.56, -0.33, 0.80), lookat=(0.09, 0.21, 0.58), origin_type="env", env_index=0)

    def __post_init__(self):
        super().__post_init__()
        self.sync()

    def sync(self):
        """Propagate the task options into the scene / sim settings (call after changing them)."""
        for mode in self.pipe_modes:
            if mode not in C.PIPE_MODES:
                raise ValueError(f"pipe mode must be one of {C.PIPE_MODES}, got {mode!r}")
        self.episode_length_s = self.max_episode_steps * C.SIM_DT * C.N_SUBSTEPS
        self.scene.pipe.spawn.usd_path = [PIPE_USD[m] for m in self.pipe_modes]
        self.scene.robot.spawn.usd_path = ROBOT_USD["visual" if self.visual else "physics"]


class VerticalPipeEnv(DirectRLEnv):
    cfg: VerticalPipeEnvCfg

    def __init__(self, cfg: VerticalPipeEnvCfg, render_mode: str | None = None, **kwargs):
        if not str(cfg.sim.device).startswith("cuda"):
            # CPU pipeline (Isaac Sim 5.1): the kinematic pipe is moved back to its USD pose at every step and
            # the tensor contact view returns no contact data, so rim hits / wall contacts would be missed.
            raise ValueError("VerticalPipeEnv needs the PhysX GPU pipeline (sim.device='cuda:0')")
        cfg.sync()
        super().__init__(cfg, render_mode, **kwargs)
        n, dev, f64 = self.num_envs, self.device, torch.float64

        # joint / body indices in PhysX order
        self._jx = self.robot.find_joints([f"Seg{i}_x" for i in range(1, 16)], preserve_order=True)[0]
        self._jy = self.robot.find_joints([f"Seg{i}_y" for i in range(1, 16)], preserve_order=True)[0]
        self._jelev = self.robot.find_joints("Elevator_Joint")[0]
        self._seg_ids = self.robot.find_bodies(SEG_NAMES, preserve_order=True)[0]
        self._tip_body = self._seg_ids[14]
        self._make_contact_view()

        # kinematics + start-state samplers (numpy, per env as in the MuJoCo env)
        self.kin = ContinuumKinematics()
        self.env_modes = [cfg.pipe_modes[i % len(cfg.pipe_modes)] for i in range(n)]
        self._samplers = {m: ResetSampler(self.kin, m, cfg.pipe_height, cfg.random_offset) for m in set(self.env_modes)}
        self._rngs = [np.random.default_rng(cfg.seed + i) for i in range(n)]
        self._eval_rngs = [np.random.default_rng(cfg.seed + 1000 + i) for i in range(n)]
        self.assist_prob = float(cfg.assist_prob)

        # MuJoCo qpos order -> PhysX joint order
        self._q_to_joint = torch.zeros(self.kin.nq, dtype=torch.long, device=dev)
        self._q_to_joint[torch.as_tensor(self.kin.qadr_x)] = torch.as_tensor(self._jx, device=dev)
        self._q_to_joint[torch.as_tensor(self.kin.qadr_y)] = torch.as_tensor(self._jy, device=dev)
        self._q_to_joint[self.kin.elev_qadr] = self._jelev[0]

        # episode state (float64 like the numpy original)
        self.bend_cmd = torch.zeros(n, 3, 2, dtype=f64, device=dev)
        self.elev_cmd = torch.zeros(n, dtype=f64, device=dev)
        self.actions = torch.zeros(n, mdp.ACT_DIM, dtype=f64, device=dev)
        self.prev_action = torch.zeros(n, mdp.ACT_DIM, dtype=f64, device=dev)
        self.prev_potential = torch.zeros(n, dtype=f64, device=dev)
        self.pipe_xy = torch.zeros(n, 2, dtype=f64, device=dev)
        self.z_top = torch.zeros(n, dtype=f64, device=dev)
        self.bore_length = torch.tensor([C.bore_length(m, cfg.pipe_height) for m in self.env_modes], dtype=f64, device=dev)
        self.z_exit = torch.zeros(n, dtype=f64, device=dev)
        self.z_success = torch.zeros(n, dtype=f64, device=dev)
        self.contact_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.max_depth = torch.zeros(n, dtype=f64, device=dev)
        self.assisted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.axis_xy = torch.as_tensor(self.kin.axis_xy, dtype=f64, device=dev)[None].expand(n, 2)
        self._targets = torch.zeros(n, self.robot.num_joints, device=dev)
        self._reward = torch.zeros(n, device=dev)
        self._meas = None
        # per-step results kept for info / logging
        self.is_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failure = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_obs = torch.zeros(n, mdp.OBS_DIM, device=dev)
        self._write_root(self.robot._ALL_INDICES)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]
        self.pipe: RigidObject = self.scene["pipe"]
        # The importer's fixed joint (world -> base_link) stores its world anchor at the origin. With
        # replicate_physics=False every env is parsed on its own, so PhysX would snap all robots to the
        # world origin: move each env's anchor to that env's origin before the physics starts.
        from pxr import Gf, UsdPhysics

        stage = self.sim.stage
        origins = self.scene.env_origins.cpu().numpy()
        for i, env_path in enumerate(self.scene.env_prim_paths):
            joint = UsdPhysics.Joint.Get(stage, f"{env_path}/Robot/joints/rootJoint_base_link")
            if not joint:
                raise RuntimeError(f"fixed root joint not found in {env_path}/Robot")
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in origins[i]]))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _make_contact_view(self):
        """
        PhysX contact view: sensors = the 15 segments of every robot, filter = the pipe of the same env
        (one sensor pattern per segment, each paired with the pipe pattern, so every sensor has exactly
        one filter). The env of every sensor row is read from its prim path.
        """
        from isaacsim.core.simulation_manager import SimulationManager

        ns = self.scene.env_regex_ns.replace(".*", "*")
        patterns = [f"{ns}/Robot/base_link/{name}" for name in SEG_NAMES]
        filters = [[f"{ns}/Pipe"] for _ in SEG_NAMES]
        view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
            patterns, filter_patterns=filters, max_contact_data_count=64 * len(SEG_NAMES) * self.num_envs)
        if view.sensor_count != len(SEG_NAMES) * self.num_envs or view.filter_count != 1:
            raise RuntimeError(f"contact view: {view.sensor_count} sensors, {view.filter_count} filters")

        def env_of(path):
            return int(path.split("/env_")[1].split("/")[0])

        row_env = [env_of(p) for p in view.sensor_paths]
        self._contact_row_body = [SEG_NAMES.index(p.rsplit("/", 1)[-1]) for p in view.sensor_paths]
        for p, f in zip(view.sensor_paths, view.filter_paths):
            if env_of(f[0]) != env_of(p):
                raise RuntimeError(f"contact filter of {p} is {f[0]}")
        self._contact_view = view
        self._contact_row_env = torch.as_tensor(row_env, dtype=torch.long, device=self.device)
        self._contact_row_seg = torch.as_tensor(self._contact_row_body, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    def _pipe_contacts(self):
        """Robot-pipe contact points of the last physics step -> (#inner-wall contacts, rim/outside hit)."""
        n = self.num_envs
        _, points, _, separations, counts, starts = self._contact_view.get_contact_data(dt=self.physics_dt)
        counts, starts = counts.view(-1).long(), starts.view(-1).long()
        total = int(counts.sum())
        if total == 0:
            return torch.zeros(n, dtype=torch.long, device=self.device), torch.zeros(n, dtype=torch.bool, device=self.device)
        rows = torch.repeat_interleave(torch.arange(counts.numel(), device=self.device), counts)
        first = torch.cumsum(counts, 0) - counts
        idx = starts[rows] + torch.arange(total, device=self.device) - first[rows]
        pts = points[idx].to(torch.float64)
        sep = separations[idx].view(-1)
        env_ids = self._contact_row_env[rows]
        pts = pts - self.scene.env_origins.to(torch.float64)[env_ids]
        if getattr(self, "debug_contacts", False) and getattr(self, "_in_get_dones", False):
            radius = torch.linalg.norm(pts[:, :2] - self.pipe_xy[env_ids], dim=1)
            self.last_contacts = (env_ids.cpu(), radius.cpu(), sep.cpu(), pts.cpu(),
                                  torch.as_tensor(self._contact_row_body)[rows.cpu()])
        # one contact per (segment, stave) pair, like MuJoCo's convex-convex collision
        seg = self._contact_row_seg[rows]
        stave = mdp.stave_index(pts, self.pipe_xy[env_ids], C.N_STAVES)
        pair_ids = (env_ids * len(SEG_NAMES) + seg) * C.N_STAVES + stave
        return mdp.classify_contacts(pts, env_ids, self.pipe_xy, C.inner_contact_radius(), n,
                                     count_mask=sep <= self.cfg.contact_touch_tolerance,
                                     outer_mask=sep <= self.cfg.rim_touch_tolerance, pair_ids=pair_ids)

    def _measure(self, no_contact_ids=None):
        f64 = torch.float64
        pos = self.robot.data.body_link_pos_w[:, self._seg_ids].to(f64) - self.scene.env_origins.to(f64)[:, None]
        rot = matrix_from_quat(self.robot.data.body_link_quat_w[:, self._tip_body].to(f64))
        n_inner, outer = self._pipe_contacts()
        if no_contact_ids is not None:
            n_inner[no_contact_ids] = 0
            outer[no_contact_ids] = False
        return mdp.measure(pos, rot, self.pipe_xy, self.z_top, self.z_success, n_inner, outer, C.PLATE_HOLE_RADIUS)

    def _observations_from(self, meas):
        return mdp.observation(meas, self.pipe_xy, self.z_top, self.z_success, self.bend_cmd, self.elev_cmd,
                               self.axis_xy, self.prev_action)

    # ------------------------------------------------------------------
    # DirectRLEnv hooks
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """
        DirectRLEnv.step, except that the joint targets are written once per env step and the asset
        buffers are refreshed once after the 10 physics steps (the MuJoCo env also sets ctrl once and
        calls mj_step 10 times). The targets do not change between the substeps, so the physics is the
        same; the per-substep writes / reads only cost time.
        """
        action = action.to(self.device)
        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._apply_action()
        self.scene.write_data_to_sim()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
        self.scene.update(dt=self.step_dt)

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

    # False: finished envs keep their final pose until reset_envs() (viewer / video show the end of an episode)
    auto_reset = True

    def reset_envs(self, env_ids):
        """Reset the given envs now (used with auto_reset=False); returns the new observations of all envs."""
        self._reset_idx(torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device))
        self.scene.write_data_to_sim()
        self.obs_buf = self._get_observations()
        return self.obs_buf

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions.to(torch.float64), -1.0, 1.0)
        mdp.update_commands(self.bend_cmd, self.elev_cmd, self.actions)
        self._write_targets()

    def _write_targets(self, env_ids=None):
        jx, jy = mdp.joint_targets(self.bend_cmd)
        self._targets[:, self._jx] = jx.float()
        self._targets[:, self._jy] = jy.float()
        self._targets[:, self._jelev[0]] = self.elev_cmd.float()

    def _apply_action(self):
        self.robot.set_joint_position_target(self._targets)

    def _get_dones(self):
        self._in_get_dones = True
        meas = self._measure()
        self._in_get_dones = False
        self._meas = meas
        self.max_depth = torch.maximum(self.max_depth, meas["depth"])
        self.contact_steps += (meas["n_inner"] > 0).long()

        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        unstable = (~torch.isfinite(q).all(dim=1)) | (~torch.isfinite(qd).all(dim=1)) \
            | (qd.abs() > self.cfg.unstable_joint_vel).any(dim=1)
        reward, pot, terminated, success, failure = mdp.step_reward(
            meas, self.prev_potential, self.actions, self.prev_action, self.cfg.contact_penalty,
            self.z_success, C.PLATE_HOLE_RADIUS, unstable)
        self.prev_potential = pot
        self.prev_action = self.actions.clone()
        self._reward = reward.float()
        self.is_success, self.failure = success, failure
        truncated = (~terminated) & (self.episode_length_buf >= self.cfg.max_episode_steps)

        # final observation of every env (before any reset) -> PPO bootstraps truncated episodes with it,
        # like SB3 does with info["terminal_observation"]
        terminal_obs = self._observations_from(meas)
        done = terminated | truncated
        outcome = torch.where(success, 0, torch.where(failure > 0, failure, 4))      # 4 = timeout
        self.extras["terminal_obs"] = terminal_obs
        self.extras["done_info"] = {
            "outcome": torch.where(done, outcome, torch.full_like(outcome, -1)),
            "contact_steps": self.contact_steps.clone(),
            "episode_length": self.episode_length_buf.clone(),
            "max_depth": self.max_depth.clone(),
            "min_clearance": meas["clearance"].clone(),
        }
        return terminated, truncated

    def _get_rewards(self):
        return self._reward

    def _get_observations(self):
        if self._meas is None:
            self._meas = self._measure()
        self.last_obs = self._observations_from(self._meas)
        return {"policy": self.last_obs}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        k = len(ids)
        rngs = self._eval_rngs if getattr(self, "_eval", False) else self._rngs
        pipe_xy, z_top, bend, elev, assisted = np.zeros((k, 2)), np.zeros(k), np.zeros((k, 3, 2)), np.zeros(k), np.zeros(k, bool)
        for j, i in enumerate(ids):
            st = self._samplers[self.env_modes[i]].sample(rngs[i], self.assist_prob)
            pipe_xy[j], z_top[j], bend[j], elev[j], assisted[j] = st.pipe_xy, st.z_top, st.bend, st.elev, st.assisted
        self.set_start_state(ids, pipe_xy, z_top, bend, elev, assisted)

    def set_start_state(self, ids, pipe_xy, z_top, bend, elev, assisted=None):
        """Put envs `ids` into a given start state (pipe placement + commanded pose, robot at rest)."""
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=self.device)  # noqa: E731
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        if assisted is None:
            assisted = np.zeros(len(idx), dtype=bool)
        self.episode_length_buf[idx] = 0
        self.pipe_xy[idx], self.z_top[idx] = t(pipe_xy), t(z_top)
        self.z_exit[idx] = self.z_top[idx] - self.bore_length[idx]
        self.z_success[idx] = self.z_exit[idx] - C.EXIT_MARGIN
        self.bend_cmd[idx], self.elev_cmd[idx] = t(bend), t(elev)
        self.assisted[idx] = torch.as_tensor(assisted, device=self.device)
        self.prev_action[idx] = 0.0
        self.actions[idx] = 0.0
        self.contact_steps[idx] = 0
        self.max_depth[idx] = 0.0
        self._write_state(idx)

        # measurement of the start pose (the sampler guarantees no pipe contact, like `ncon == 0`)
        meas = self._measure(no_contact_ids=idx)
        if self._meas is None:
            self._meas = meas
        else:
            for key, val in meas.items():
                self._meas[key][idx] = val[idx]
        self.prev_potential[idx] = mdp.potential(meas)[idx]

    def _write_root(self, idx):
        """
        The importer's fixed root joint anchors the base at the world origin; PhysX keeps a fixed-base
        articulation where its root is written, so every robot is put at its env origin explicitly.
        """
        root = torch.zeros(len(idx), 7, device=self.device)
        root[:, :3] = self.scene.env_origins[idx]
        root[:, 3] = 1.0
        self.robot.write_root_pose_to_sim(root, env_ids=idx)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(idx), 6, device=self.device), env_ids=idx)

    def _write_state(self, idx):
        """Robot joints (at rest, targets = commands) and pipe pose of the given envs."""
        self._write_root(idx)
        qpos = self.kin.qpos_from(self.bend_cmd[idx].cpu().numpy(), self.elev_cmd[idx].cpu().numpy())
        joint_pos = torch.zeros(len(idx), self.robot.num_joints, device=self.device)
        joint_pos[:, self._q_to_joint] = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=idx)
        self._write_targets()
        self.robot.set_joint_position_target(self._targets[idx], env_ids=idx)
        pose = torch.zeros(len(idx), 7, device=self.device)
        pose[:, :2] = (self.pipe_xy[idx] + self.scene.env_origins[idx, :2].to(torch.float64)).float()
        pose[:, 2] = (self.z_top[idx] + self.scene.env_origins[idx, 2].to(torch.float64)).float()
        pose[:, 3] = 1.0
        self.pipe.write_root_pose_to_sim(pose, env_ids=idx)

    # ------------------------------------------------------------------
    # Evaluation support: run evaluation episodes on the same sim, then put
    # every env back exactly where it was (MuJoCo trains with a separate eval env)
    # ------------------------------------------------------------------
    _STATE_KEYS = ("bend_cmd", "elev_cmd", "actions", "prev_action", "prev_potential", "pipe_xy", "z_top", "z_exit",
                   "z_success", "contact_steps", "max_depth", "assisted", "episode_length_buf", "last_obs")

    def snapshot(self):
        snap = {k: getattr(self, k).clone() for k in self._STATE_KEYS}
        snap["joint_pos"] = self.robot.data.joint_pos.clone()
        snap["joint_vel"] = self.robot.data.joint_vel.clone()
        snap["meas"] = {k: v.clone() for k, v in self._meas.items()}
        snap["rngs"] = [copy.deepcopy(r.bit_generator.state) for r in self._rngs]
        snap["assist_prob"] = self.assist_prob
        return snap

    def restore(self, snap):
        for k in self._STATE_KEYS:
            getattr(self, k).copy_(snap[k])
        self._meas = {k: v.clone() for k, v in snap["meas"].items()}
        for r, state in zip(self._rngs, snap["rngs"]):
            r.bit_generator.state = state
        self.assist_prob = snap["assist_prob"]
        idx = self.robot._ALL_INDICES
        self._write_root(idx)
        self.robot.write_joint_state_to_sim(snap["joint_pos"], snap["joint_vel"], env_ids=idx)
        self._write_targets()
        self.robot.set_joint_position_target(self._targets)
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, :2] = (self.pipe_xy + self.scene.env_origins[:, :2].to(torch.float64)).float()
        pose[:, 2] = (self.z_top + self.scene.env_origins[:, 2].to(torch.float64)).float()
        pose[:, 3] = 1.0
        self.pipe.write_root_pose_to_sim(pose, env_ids=idx)
        self.scene.write_data_to_sim()

    def set_eval(self, enabled: bool):
        """Evaluation episodes: no assisted starts, separate random streams (like the MuJoCo eval env)."""
        self._eval = bool(enabled)
        if enabled:
            self._train_assist = self.assist_prob
            self.assist_prob = 0.0
        else:
            self.assist_prob = getattr(self, "_train_assist", self.cfg.assist_prob)

    # ------------------------------------------------------------------
    # Info (same fields as VerticalPipeEnv._make_info, env by env)
    # ------------------------------------------------------------------
    def info(self, i: int) -> dict:
        m = {k: v[i] for k, v in self._meas.items()}
        stage = int(mdp.stage({k: v[i:i + 1] for k, v in self._meas.items()}, self.z_exit[i:i + 1])[0])
        theta = torch.linalg.norm(self.bend_cmd[i], dim=1).cpu().numpy()
        phi = np.mod(np.arctan2(self.bend_cmd[i, :, 1].cpu().numpy(), self.bend_cmd[i, :, 0].cpu().numpy()), 2 * np.pi)
        kappa = theta / C.SECTION_LENGTHS
        elev = float(self.robot.data.joint_pos[i, self._jelev[0]])
        return {
            "is_success": bool(self.is_success[i]),
            "failure": C.FAILURE_NAMES[int(self.failure[i])],
            "stage": stage, "stage_name": C.STAGE_NAMES[stage],
            "lat_mm": float(m["lat"]) * 1000.0, "tilt_deg": float(np.rad2deg(float(m["tilt"]))),
            "depth_mm": float(m["depth"]) * 1000.0, "max_depth_mm": float(self.max_depth[i]) * 1000.0,
            "clearance_mm": float(m["clearance"]) * 1000.0, "inner_contacts": int(m["n_inner"]),
            "contact_steps": int(self.contact_steps[i]), "elevator_mm": elev * 1000.0,
            "motor_angles": C.K_to_Length_3Seg(kappa[0], phi[0], kappa[1], phi[1], kappa[2], phi[2]),
            "elevator_motor_deg": elev / C.LEAD_SCREW_PITCH * 360.0,
            "tip_pos": m["tip"].cpu().numpy(),
        }
