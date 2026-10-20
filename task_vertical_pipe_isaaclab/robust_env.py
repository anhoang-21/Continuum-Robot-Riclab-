"""
==============================================================================
robust_env.py - the cable-driven robot under uncertainty, with safety costs (Stage 6, ND1 / ND3 / ND4)
==============================================================================
CableVerticalPipeEnv (6 lead-screw motors, 12 cables, elastic backbone, gravity) where every env is a
different robot (uncertainty.UncertaintySet, drawn again at every reset):

  * cable stiffness / pre-tension / damping per cable, capstan friction in the cable path and nut
    backlash (uncertain_cables.UncertainCableModel, integrated at every 2 ms physics step like the
    nominal model, also as a CUDA graph);
  * backbone spring and damping per env (PhysX drive gains of the 30 bending joints);
  * motor position-loop bandwidth per motor; a tip payload (force on Seg15 at every physics step);
  * one step of command latency; noise on the observed body positions (mocap) and tensions (load cells).

The start state is the static equilibrium of the env's TRUE parameters at the motor positions that hold the
sampled pose (start_mode "true"), or at the positions the NOMINAL model computes (start_mode "nominal": the
controller believes the nominal robot). Reward / termination are those of the task.

Safety signals (every step):
  extras["cost"] (N, 3): slack   sum_c max(0, T_min - T_r,c) / T_min        (a cable goes slack: backlash)
                         over    sum_c max(0, T_c - T_max) / T_max          (T_c = max of nut / robot side)
                         contact max(0, F_pipe - F_max) / F_max             (normal force of the robot on the pipe)
  extras["safety"]: min / max tension, #slack cables, pipe contact force, motor torque / pull-out ratio.
  extras["done_info"] also gets the episode sums (cost_*), the number of violating steps and the extremes.

Options: obs_tension (12 nut-side tensions, load cells, in the observation: 33 + 12), priv_obs (the
normalised parameters as an extra observation group "priv" for privileged teachers / critics).
==============================================================================
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import warp as wp

from isaaclab.utils import configclass

import cables as CB
import mdp
import motors as MT
import uncertainty as U
from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg
from uncertain_cables import StaticSolver, UncertainCableModel

COST_NAMES = ("slack", "over", "contact")


@configclass
class RobustCableEnvCfg(CableVerticalPipeEnvCfg):
    uncertainty: str = "nominal"                # "nominal" | "train" | "ood"
    uncertainty_json: str = ""                  # identified ranges (sysid.py), on top of `uncertainty`
    start_mode: str = "true"                    # "true" | "nominal": motor positions of the start pose
    t_min: float = 0.5                          # N, a robot-side tension below this counts as slack
    t_max: float = 10.0                         # N, design limit of the cables / discs
    f_max: float = 2.0                          # N, normal force of the robot on the pipe
    obs_tension: bool = False                   # 12 load-cell tensions (nut side) in the observation
    priv_obs: bool = False                      # observation group "priv": normalised parameters

    def sync(self):
        if self.start_mode not in ("true", "nominal"):
            raise ValueError(f"start_mode must be 'true' or 'nominal', got {self.start_mode!r}")
        if self.uncertainty not in U.LEVELS:
            raise ValueError(f"uncertainty must be one of {U.LEVELS}, got {self.uncertainty!r}")
        super().sync()
        self.observation_space = mdp.OBS_DIM + (CB.N_CABLES if self.obs_tension else 0)


class RobustCableEnv(CableVerticalPipeEnv):
    cfg: RobustCableEnvCfg

    def __init__(self, cfg: RobustCableEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)          # captures the nominal drive graph (replaced below)
        n, dev = self.num_envs, self.device
        self.uset = (U.UncertaintySet.from_json(cfg.uncertainty_json, cfg.uncertainty) if cfg.uncertainty_json
                     else U.UncertaintySet.make(cfg.uncertainty))
        self.randomized = cfg.uncertainty != "nominal" or bool(cfg.uncertainty_json)
        self.ucm = UncertainCableModel(self.geom, n, dev)
        self.statics = StaticSolver(self.geom, self.gravity_model)
        self._prm_rngs = [np.random.default_rng(cfg.seed + 5000 + i) for i in range(n)]
        self._prm_eval_rngs = [np.random.default_rng(cfg.seed + 6000 + i) for i in range(n)]
        self.prm = U.nominal_params(n)                       # numpy, the parameters of every env
        f32 = torch.float32
        self.priv = torch.zeros(n, U.PRIV_DIM, dtype=f32, device=dev)
        self.latency = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pos_noise = torch.zeros(n, dtype=torch.float64, device=dev)
        self.tension_noise = torch.zeros(n, dtype=torch.float64, device=dev)
        self._act_buf = torch.zeros(n, mdp.ACT_DIM, dtype=torch.float64, device=dev)
        # per-env motor loop gains (CableMotorBank keeps them per motor)
        m = self.motors
        for k in ("kp", "kd", "ki", "i_max"):
            setattr(m, k, getattr(m, k).expand(n, -1).clone())
        # external force on Seg15 (payload + pushes), world frame, applied at every physics step
        self._payload_force = torch.zeros(n, 3, dtype=f32, device=dev)
        self._push_force = torch.zeros(n, 3, dtype=f32, device=dev)
        self._push_left = np.zeros(n, dtype=np.int64)
        self._wrench_on = False
        # safety bookkeeping
        self.contact_force = torch.zeros(n, dtype=torch.float64, device=dev)
        self.cost = torch.zeros(n, len(COST_NAMES), dtype=torch.float64, device=dev)
        self.ep = {k: torch.zeros(n, dtype=torch.float64, device=dev) for k in (
            "cost_slack", "cost_over", "cost_contact", "viol_steps", "slack_steps", "over_steps",
            "contact_viol_steps", "t_max", "t_min", "f_max")}
        self.ep["t_min"].fill_(np.inf)
        self._cable_graph = self._capture_cable_graph()       # now with the uncertain model

    # ------------------------------------------------------------------
    # Drive: the uncertain cable model at every physics step
    # ------------------------------------------------------------------
    def _cable_math(self):
        if not hasattr(self, "ucm"):                          # graph capture inside CableVerticalPipeEnv.__init__
            return super()._cable_math()
        model, motors = self.ucm, self.motors
        Gq, Gqd = self._g_q @ model.G.T, self._g_qd @ model.G.T
        span = self.motor_cmd - self._motor_cmd_prev
        x0 = self._motor_cmd_prev + span * self._g_frac[0]
        x1 = self._motor_cmd_prev + span * self._g_frac[1]
        model.begin_step(Gq, motors.x)
        motors.step(self.physics_dt, x0, x1, lambda x, v: model.nut_load(Gq, Gqd, x, v))
        self.tension.copy_(model.end_step(Gq, Gqd, motors.x, motors.v))
        self._efforts[:, self._jchain] = model.joint_torque(self.tension).float()

    def _capture_cable_graph(self):
        if not hasattr(self, "ucm"):
            return super()._capture_cable_graph()
        mine = list(self.ucm.state_tensors())
        saved = [t.clone() for t in mine]
        graph = super()._capture_cable_graph()
        for t, s in zip(mine, saved):
            t.copy_(s)
        return graph

    def _cable_substep(self, frac0, frac1):
        # the scene writes the external wrench before the first physics step of an env step only
        if self._wrench_on and frac0 > 0.0:
            self.robot.write_data_to_sim()
        super()._cable_substep(frac0, frac1)

    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.to(self.device).to(torch.float64)
        if self.randomized:
            delayed = torch.where(self.latency[:, None], self._act_buf, a)
            self._act_buf.copy_(a)
            a = delayed
        if self._push_left.any():
            self._push_left = np.maximum(self._push_left - 1, 0)
            done = np.nonzero(self._push_left == 0)[0]
            if len(done):
                ids = torch.as_tensor(done, device=self.device)
                if self._push_force[ids].abs().sum() > 0:
                    self._push_force[ids] = 0.0
                    self._update_wrench(ids)
        super()._pre_physics_step(a)

    # ------------------------------------------------------------------
    # External force on the tip: payload (permanent) and pushes (stress tests)
    # ------------------------------------------------------------------
    def _update_wrench(self, ids):
        force = (self._payload_force[ids] + self._push_force[ids])[:, None, :]
        if not self._wrench_on and float(force.abs().sum()) == 0.0:
            return
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force, body_ids=[self._tip_body], env_ids=ids, is_global=True)
        self._wrench_on = bool((self._payload_force.abs().sum() + self._push_force.abs().sum()) > 0)

    def push(self, env_ids, force, steps: int):
        """Push the tip of envs `env_ids` with `force` (k, 3) N (world frame) for `steps` env steps."""
        ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        self._push_force[ids] = torch.as_tensor(np.asarray(force), dtype=torch.float32, device=self.device).view(-1, 3)
        self._push_left[ids.cpu().numpy()] = int(steps)
        self._update_wrench(ids)

    # ------------------------------------------------------------------
    # Parameters of the envs (drawn at every start state)
    # ------------------------------------------------------------------
    def _draw_params(self, ids):
        rngs = self._prm_eval_rngs if getattr(self, "_eval", False) else self._prm_rngs
        draws = [self.uset.sample(rngs[i], 1) for i in ids]
        return {k: np.concatenate([d[k] for d in draws], axis=0) for k in draws[0]}

    def set_params(self, ids, prm: dict):
        """Make envs `ids` the robots described by prm (uncertainty.UncertaintySet.sample of len(ids) envs)."""
        ids_np = np.asarray(list(ids), dtype=np.int64)
        idx = torch.as_tensor(ids_np, device=self.device)
        for k, v in prm.items():
            self.prm[k][ids_np] = v
        self.ucm.set_params(idx, prm)
        # motor position loops
        m = self.motors
        w = 2.0 * np.pi * torch.as_tensor(prm["motor_bw"], dtype=torch.float64, device=self.device)
        J = m.J[None]
        m.kp[idx], m.kd[idx], m.ki[idx] = J * w * w, 2.0 * 0.8 * J * w, J * w ** 3 / 8.0
        m.i_max[idx] = m.hold[None] / m.ki[idx]
        # backbone springs / damping (PhysX drive gains of the bending joints)
        if self.randomized:
            nj = len(self._jchain)
            k_j = torch.as_tensor(prm["k_j"], dtype=torch.float32, device=self.device)[:, None].expand(-1, nj).contiguous()
            c_j = torch.as_tensor(prm["c_j"], dtype=torch.float32, device=self.device)[:, None].expand(-1, nj).contiguous()
            self.robot.write_joint_stiffness_to_sim_index(stiffness=k_j, joint_ids=self._jchain, env_ids=idx)
            self.robot.write_joint_damping_to_sim_index(damping=c_j, joint_ids=self._jchain, env_ids=idx)
        self.latency[idx] = torch.as_tensor(prm["latency"] > 0, device=self.device)
        self.pos_noise[idx] = torch.as_tensor(prm["pos_noise"], dtype=torch.float64, device=self.device)
        self.tension_noise[idx] = torch.as_tensor(prm["tension_noise"], dtype=torch.float64, device=self.device)
        self.priv[idx] = torch.as_tensor(U.privileged(prm), dtype=torch.float32, device=self.device)
        self._payload_force[idx] = 0.0
        if self.cfg.gravity:
            self._payload_force[idx, 2] = -MT.G * torch.as_tensor(prm["payload"], dtype=torch.float32, device=self.device)
        self._push_force[idx] = 0.0
        self._push_left[ids_np] = 0
        self._update_wrench(idx)

    def set_payload(self, ids, kg):
        """Hang `kg` on the tip of envs ids (system identification, stress tests)."""
        ids = list(ids)
        self.prm["payload"][ids] = kg
        idx = torch.as_tensor(ids, device=self.device)
        self._payload_force[idx] = 0.0
        if self.cfg.gravity:
            self._payload_force[idx, 2] = -MT.G * torch.as_tensor(np.broadcast_to(kg, (len(ids),)), dtype=torch.float32,
                                                                  device=self.device)
        self._update_wrench(idx)

    def park_pipes(self):
        """Move every pipe 3 m below its robot (experiments without the task)."""
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, :3] = self.scene.env_origins + torch.tensor([0.0, 0.0, -3.0], device=self.device)
        pose[:, 6] = 1.0
        self.pipe.write_root_pose_to_sim_index(root_pose=pose)

    def straighten(self, ids=None):
        """Motors of envs ids at 0, the robot at rest in its static equilibrium there, friction elements relaxed."""
        ids = list(range(self.num_envs)) if ids is None else list(ids)
        idx = torch.as_tensor(ids, device=self.device)
        z = torch.zeros(len(ids), 6, device=self.device, dtype=torch.float64)
        self.set_motors(idx, z)
        self.motors.reset(idx, z)
        q = self.statics.equilibrium(z.cpu(), **self.static_params(ids)).to(self.device)
        jp = self.robot.data.joint_pos.torch[idx].clone()
        jp[:, self._jchain] = q.float()
        self.robot.write_joint_state_to_sim_index(position=jp, velocity=torch.zeros_like(jp), env_ids=idx)
        self.ucm.reset_state(idx, q @ self.ucm.G.T, z)

    def static_params(self, ids):
        """(k_c, t0, k_j, payload) of envs ids as CPU float64 tensors (StaticSolver arguments)."""
        ids = np.asarray(list(ids), dtype=np.int64)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
        return dict(k_c=t(self.prm["k_c"][ids]), t0=t(self.prm["t0"][ids]), k_j=t(self.prm["k_j"][ids]),
                    payload=self.prm["payload"][ids] if self.cfg.gravity else None)

    # ------------------------------------------------------------------
    # Start states
    # ------------------------------------------------------------------
    def _write_state(self, idx):
        ids = idx.tolist()
        if self.randomized:
            self.set_params(ids, self._draw_params(ids))
        self._write_root(idx)
        sp = self.static_params(ids)
        bend = self.bend_cmd[idx].cpu()
        q_des = torch.as_tensor(self.geom.q_bend(bend.numpy()))
        nominal = self._cable_model_cpu
        if self.cfg.action_mode == "bend" and self.cfg.bend_map == "formula":
            u = nominal.compensate(bend)
        elif self.cfg.start_mode == "nominal":
            u = nominal.motors_for_pose(q_des)
        else:
            u = self.statics.motors_for_pose(q_des, **sp)
        u = torch.maximum(torch.minimum(u, nominal.u_max), -nominal.u_max)
        q = self.statics.equilibrium(u, q0=q_des, **sp)
        u_dev, q_dev = u.to(self.device), q.to(self.device)
        self.motors.reset(idx, u_dev)
        self.motor_cmd[idx] = u_dev
        self._motor_cmd_prev[idx] = u_dev
        joint_pos = self.robot.data.default_joint_pos.torch[idx].clone()
        joint_pos[:, self._jchain] = q_dev.float()
        joint_pos[:, self._jelev[0]] = self.elev_cmd[idx].float()
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=idx)
        self._write_targets()
        self.robot.actuators.target_command.set_position_index(value=self._targets[idx], env_ids=idx)
        self.ucm.reset_state(idx, q_dev @ self.ucm.G.T, u_dev)
        self.tension[idx] = self.ucm.T_r[idx]
        bend_meas = q_dev.view(-1, 3, 5, 2).sum(dim=2)
        self.bend_meas[idx] = torch.stack([bend_meas[..., 1], -bend_meas[..., 0]], dim=-1)
        pose = torch.zeros(len(idx), 7, device=self.device)
        pose[:, :2] = (self.pipe_xy[idx] + self.scene.env_origins[idx, :2].to(torch.float64)).float()
        pose[:, 2] = (self.z_top[idx] + self.scene.env_origins[idx, 2].to(torch.float64)).float()
        pose[:, 6] = 1.0                                   # identity quaternion (x, y, z, w)
        self.pipe.write_root_pose_to_sim_index(root_pose=pose, env_ids=idx)
        # episode bookkeeping
        self._act_buf[idx] = 0.0
        self.contact_force[idx] = 0.0
        self.cost[idx] = 0.0
        for k, v in self.ep.items():
            v[idx] = np.inf if k == "t_min" else 0.0

    # ------------------------------------------------------------------
    # Observation (sensor noise, load cells, privileged parameters)
    # ------------------------------------------------------------------
    def _observations_from(self, meas):
        if hasattr(self, "pos_noise") and self.randomized:
            sig = self.pos_noise[:, None]
            meas = dict(meas)
            for k in ("tip", "seg13", "seg11", "seg6"):
                meas[k] = meas[k] + sig * torch.randn_like(meas[k])
            d = meas["tip_dir"] + (sig / 0.03) * torch.randn_like(meas["tip_dir"])   # marker cluster ~3 cm
            meas["tip_dir"] = d / torch.linalg.norm(d, dim=1, keepdim=True)
        obs = super()._observations_from(meas)
        if self.cfg.obs_tension and hasattr(self, "ucm"):
            T = self.ucm.T_n + self.tension_noise[:, None] * torch.randn_like(self.ucm.T_n)
            obs = torch.cat([obs, ((T - CB.CABLE_PRETENSION) / CB.CABLE_PRETENSION).float()], dim=1)
        elif self.cfg.obs_tension:
            obs = torch.cat([obs, torch.zeros(obs.shape[0], CB.N_CABLES, device=obs.device)], dim=1)
        return obs

    def _get_observations(self):
        out = super()._get_observations()
        if self.cfg.priv_obs:
            out["priv"] = self.priv
        return out

    # ------------------------------------------------------------------
    # Safety signals
    # ------------------------------------------------------------------
    def _pipe_contact_force(self):
        """Normal force (N) of the robot on its pipe(s) at the last physics step, (N,)."""
        view = self._contact_view
        fn = getattr(view, "get_normal_contact_force_matrix", None) or view.get_contact_force_matrix
        mat = fn(dt=self.physics_dt)
        mat = wp.to_torch(mat) if isinstance(mat, wp.array) else mat
        f = torch.linalg.norm(mat.reshape(view.sensor_count, -1, 3).to(torch.float64), dim=-1).sum(-1)
        out = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        out.index_add_(0, self._contact_row_env, f)
        # PhysX keeps stale buffers in a step without contacts: only envs with fresh contact points count
        touching = (self._meas["n_inner"] > 0) | self._meas["outer_hit"]
        return torch.where(touching, out, torch.zeros_like(out))

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        if not hasattr(self, "ucm"):
            return terminated, truncated
        cfg = self.cfg
        T_r, T_n = self.tension, self.ucm.T_n
        T_hi = torch.maximum(T_r, T_n)
        self.contact_force = self._pipe_contact_force()
        slack = torch.clamp(cfg.t_min - T_r, min=0.0).sum(1) / cfg.t_min
        over = torch.clamp(T_hi - cfg.t_max, min=0.0).sum(1) / cfg.t_max
        contact = torch.clamp(self.contact_force - cfg.f_max, min=0.0) / cfg.f_max
        self.cost = torch.stack([slack, over, contact], dim=1)
        n_slack = (T_r < cfg.t_min).sum(1)
        n_over = (T_hi > cfg.t_max).sum(1)
        ep = self.ep
        ep["cost_slack"] += slack
        ep["cost_over"] += over
        ep["cost_contact"] += contact
        ep["slack_steps"] += (n_slack > 0).double()
        ep["over_steps"] += (n_over > 0).double()
        ep["viol_steps"] += ((n_slack > 0) | (n_over > 0)).double()
        ep["contact_viol_steps"] += (self.contact_force > cfg.f_max).double()
        ep["t_max"] = torch.maximum(ep["t_max"], T_hi.max(1).values)
        ep["t_min"] = torch.minimum(ep["t_min"], T_r.min(1).values)
        ep["f_max"] = torch.maximum(ep["f_max"], self.contact_force)
        self.extras["cost"] = self.cost.float()
        m = self.motors
        self.extras["safety"] = {
            "t_min": T_r.min(1).values, "t_max": T_hi.max(1).values, "n_slack": n_slack, "n_over": n_over,
            "contact_force": self.contact_force.clone(),
            "torque_ratio": (m.torque.abs() / m.pullout(m.v)).max(1).values,
        }
        self.extras["done_info"].update({k: v.clone() for k, v in ep.items()})
        return terminated, truncated

    # ------------------------------------------------------------------
    # Evaluation support (PipeRunner.evaluate): parameters and drive state are part of the snapshot
    # ------------------------------------------------------------------
    def snapshot(self):
        snap = super().snapshot()
        m = self.motors
        snap["robust"] = {
            "ucm": [t.clone() for t in self.ucm.param_tensors() + self.ucm.state_tensors()],
            "gains": [getattr(m, k).clone() for k in ("kp", "kd", "ki", "i_max")],
            "prm": copy.deepcopy(self.prm),
            "rngs": [copy.deepcopy(r.bit_generator.state) for r in self._prm_rngs],
            "act_buf": self._act_buf.clone(), "ep": {k: v.clone() for k, v in self.ep.items()},
        }
        return snap

    def restore(self, snap):
        super().restore(snap)
        r = snap["robust"]
        self.set_params(range(self.num_envs), r["prm"])        # PhysX gains, payload, noise, latency
        for t, s in zip(self.ucm.param_tensors() + self.ucm.state_tensors(), r["ucm"]):
            t.copy_(s)
        for k, s in zip(("kp", "kd", "ki", "i_max"), r["gains"]):
            getattr(self.motors, k).copy_(s)
        for rng, state in zip(self._prm_rngs, r["rngs"]):
            rng.bit_generator.state = state
        self._act_buf.copy_(r["act_buf"])
        for k, v in r["ep"].items():
            self.ep[k].copy_(v)

    def info(self, i: int) -> dict:
        out = super().info(i)
        p = self.prm
        out.update({
            "cable_tension_nut": self.ucm.T_n[i].cpu().numpy(),
            "contact_force": float(self.contact_force[i]),
            "cost": self.cost[i].cpu().numpy(),
            "params": {"k_c": float(p["k_c"][i].mean()), "t0": float(p["t0"][i].mean()), "k_j": float(p["k_j"][i]),
                       "mu_theta": float(p["mu_theta"][i].mean()), "backlash_mm": float(p["backlash"][i].mean() * 1e3),
                       "payload_g": float(p["payload"][i] * 1e3), "latency": int(p["latency"][i])},
        })
        return out
