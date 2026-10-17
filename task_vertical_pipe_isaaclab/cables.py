"""
==============================================================================
cables.py - Wire (tendon) actuation of the three-section continuum robot
==============================================================================
Model of the real robot's cable drive (graduation thesis "Control of Three-Section 3D
Printed Tele-operated Wire-Driven Continuum Robot Arm", ch. 3.5 and 4):

  * 3 sections x 4 cables. Every lead-screw motor drives an antagonistic pair: the cable
    at the motor angle alpha and the one opposite (alpha + 180 deg). Motor angles and
    routing radii are the ones of the lab's controller (constants.MOTOR_ALPHAS /
    CABLE_RADIUS): section 1 = M1 0 deg, M2 90 deg (d 40 mm), section 2 = M5 150 deg,
    M6 240 deg (d 30 mm), section 3 = M3 30 deg, M4 300 deg (d 20 mm).
  * A cable starts at the base plate, runs through a hole of every routing disc
    (Seg1, Seg2, ...) at a constant radius and is anchored on the last disc of its
    section (Seg5 / Seg10 / Seg15). Cables of sections 2 and 3 therefore also run
    through the lower sections: bending section 1 stretches / slackens them
    ("accumulative cable coupling", thesis eq. 3.32).
  * Cables only pull: tension T = k_c * max(0, stretch + T0 / k_c) with a pre-tension T0
    (tension box springs). The cardan backbone is elastic (steel springs around it):
    a joint spring k_j per universal-joint axis, no servo on the bending joints.

Physics: a frictionless cable through the disc holes does virtual work
    tau = -T * dl/dq          (l = cable path length, q = the 30 bending joint angles)
i.e. every joint the cable crosses gets a torque ~ T * d. cable_env.py applies exactly
this as joint torques at every 2 ms physics step (tension from the motor-nut positions of
motors.py). PhysX tendons were tried first and do not give this law: spatial tendons
apply the force only at the root and the leaf attachment (PhysxSchema: "at intermediate
attachments no forces are applied"), and fixed tendons spread the force unevenly over the
chain (measured in PhysX 5 / Isaac Sim 5.1: joint torques falling 5:4:3:2:1 from the base
to the anchor instead of equal, about 1/10 of the tension).

dl/dq is computed here from the routing geometry (hole positions on the Seg bodies,
MJCF kinematics); the exact, non-linear path length through the holes is kept for
checks and for drawing the cables.

Everything in this file is numpy / torch only (no Isaac Sim).
==============================================================================
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch

from constants import BEND_RATE, CABLE_RADIUS, SOURCE_MJCF, THETA_MAX
from kinematics import ContinuumKinematics
from motors import CABLE_SCREW

# ---------------------------------------------------------------------------
# Motors and cables
# ---------------------------------------------------------------------------
# action / observation order of the 6 motors: section by section
MOTOR_NAMES = ("M1", "M2", "M5", "M6", "M3", "M4")
MOTOR_SECTION = np.array([0, 0, 1, 1, 2, 2])
MOTOR_ALPHA = np.deg2rad([0.0, 90.0, 150.0, 240.0, 30.0, 300.0])     # world-frame angle of the pulled cable
# index of every motor in the controller's M1..M6 order (constants.K_to_Length_3Seg)
MOTOR_TO_M16 = np.array([0, 1, 4, 5, 2, 3])

# 12 cables: (motor, sign). sign +1: cable at alpha (shortened by a negative motor command),
# sign -1: the antagonist at alpha + 180 deg (lengthened by the same amount)
CABLE_MOTOR = np.repeat(np.arange(6), 2)
CABLE_SIGN = np.tile([1.0, -1.0], 6)
CABLE_SECTION = MOTOR_SECTION[CABLE_MOTOR]
CABLE_ALPHA = MOTOR_ALPHA[CABLE_MOTOR] + np.where(CABLE_SIGN > 0, 0.0, np.pi)
CABLE_D = CABLE_RADIUS[CABLE_SECTION]
CABLE_NAMES = tuple(f"{MOTOR_NAMES[m]}{'a' if s > 0 else 'b'}" for m, s in zip(CABLE_MOTOR, CABLE_SIGN))
N_CABLES, N_MOTORS = 12, 6

# ---------------------------------------------------------------------------
# Physical parameters (calibration values, not measured on the robot)
# ---------------------------------------------------------------------------
CABLE_STIFFNESS = 3000.0      # N/m, steel wire + Bowden sheath + tensioner in series
CABLE_DAMPING = 5.0           # N s/m, on the tendon length rate
CABLE_PRETENSION = 3.0        # N, tension of every cable in the straight pose (tension-box springs)
BACKBONE_STIFFNESS = 0.8      # N m/rad per universal-joint axis (steel springs around the cardan backbone)
BACKBONE_DAMPING = 1.0        # N m s/rad per axis (joint damping 0.5 of the MJCF + friction in the routing)

# motor travel: a section's own bend needs d * theta, the lower sections add the same again
MOTOR_RANGE = CABLE_RADIUS[MOTOR_SECTION] * THETA_MAX * (MOTOR_SECTION + 1)             # m, +-
# motor speed per 20 ms step: bend its own section at BEND_RATE and follow the lower sections
MOTOR_RATE = CABLE_RADIUS[MOTOR_SECTION] * np.cumsum(BEND_RATE)[MOTOR_SECTION]         # m / step

# routing geometry
ROOT_HOLE_DEPTH = 0.019       # m, base-plate hole above the first universal joint
TIP_HOLE_Z = 0.017            # m, hole of Seg15 below its joint (the other discs: half the joint spacing)


def motor_angles_deg(u):
    """Motor displacements (..., 6) in m -> lead-screw motor angles in the controller's M1..M6 order (deg)."""
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    out[..., MOTOR_TO_M16] = u / CABLE_SCREW.lead * 360.0
    return out


# ---------------------------------------------------------------------------
# Routing geometry
# ---------------------------------------------------------------------------
class CableGeometry:
    """
    Hole positions of the 12 cables and their (linearised) length Jacobian.

    Bending joints are indexed in chain order: j = 2 (i - 1) + {0: Seg<i>_x, 1: Seg<i>_y}, i = 1..15.
    A cable of section s crosses the joints Seg1_x .. Seg<5s>_y (10 s joints).
    """

    def __init__(self, kin: ContinuumKinematics | None = None):
        self.kin = kin or ContinuumKinematics()
        k = self.kin
        self.joint_names = [f"Seg{i}_{a}" for i in range(1, 16) for a in ("x", "y")]
        self.joint_qadr = np.array([k.joint_index[n] for n in self.joint_names])
        self.elev_body = k.body_index["Elevator_Link"]
        self.seg_bodies = [k.body_index[f"Seg{i}"] for i in range(1, 16)]

        xpos, xmat = k.forward(np.zeros(k.nq))
        # hole height below each Seg origin: half-way to the next universal joint
        z_hole = []
        for i in range(15):
            if i < 14:
                gap = xmat[self.seg_bodies[i]].T @ (xpos[self.seg_bodies[i + 1]] - xpos[self.seg_bodies[i]])
                z_hole.append(0.5 * gap[2])
            else:
                z_hole.append(TIP_HOLE_Z)
        self.z_hole = np.array(z_hole)

        # path of every cable: list of (body index, local point); first point = base plate (Elevator_Link)
        self.paths = []
        seg1, elev = self.seg_bodies[0], self.elev_body
        for c in range(N_CABLES):
            world_dir = np.array([np.cos(CABLE_ALPHA[c]), np.sin(CABLE_ALPHA[c]), 0.0]) * CABLE_D[c]
            pts = []
            root_w = xpos[seg1] + xmat[seg1] @ (xmat[seg1].T @ world_dir + np.array([0.0, 0.0, -ROOT_HOLE_DEPTH]))
            pts.append((elev, xmat[elev].T @ (root_w - xpos[elev])))
            for i in range(5 * (CABLE_SECTION[c] + 1)):
                b = self.seg_bodies[i]
                local = xmat[b].T @ world_dir
                local[2] = self.z_hole[i]
                pts.append((b, local))
            self.paths.append(pts)

        self.rest_length = self.lengths(np.zeros(k.nq))
        self.G = self._jacobian()                  # (12, 30) m/rad, d l_c / d q_j at the straight pose
        # the joints each cable crosses (non-zero gearing)
        self.cable_joints = [np.arange(10 * (CABLE_SECTION[c] + 1)) for c in range(N_CABLES)]

    # ------------------------------------------------------------------
    def hole_points(self, xpos, xmat):
        """World positions of all holes: list (per cable) of (..., n_holes, 3)."""
        out = []
        for pts in self.paths:
            out.append(np.stack([xpos[..., b, :] + np.einsum("...ij,j->...i", xmat[..., b, :, :], p)
                                 for b, p in pts], axis=-2))
        return out

    def lengths(self, qpos):
        """Exact path length through the holes, (..., 12), for full qpos (MuJoCo order)."""
        xpos, xmat = self.kin.forward(qpos)
        return np.stack([np.linalg.norm(np.diff(h, axis=-2), axis=-1).sum(-1)
                         for h in self.hole_points(xpos, xmat)], axis=-1)

    def _jacobian(self, eps=1e-6):
        q = np.zeros((2 * 30, self.kin.nq))
        for j, qa in enumerate(self.joint_qadr):
            q[2 * j, qa] = eps
            q[2 * j + 1, qa] = -eps
        lens = self.lengths(q)                                  # (60, 12)
        G = (lens[0::2] - lens[1::2]) / (2.0 * eps)             # (30, 12)
        G = G.T
        G[np.abs(G) < 1e-9] = 0.0
        return G

    def q_bend(self, bend):
        """Bend vectors (..., 3, 2) -> the 30 bending joint angles in chain order."""
        jx, jy = self.kin.joint_angles(bend)
        q = np.empty(jx.shape[:-1] + (30,))
        q[..., 0::2], q[..., 1::2] = jx, jy
        return q

    def lengths_from_joints(self, q30, elev=0.0):
        qpos = np.zeros(np.shape(q30)[:-1] + (self.kin.nq,))
        qpos[..., self.joint_qadr] = q30
        qpos[..., self.kin.elev_qadr] = elev
        return self.lengths(qpos)


# ---------------------------------------------------------------------------
# Controller-side model (thesis eq. 3.32): commanded bend -> motor displacements
# ---------------------------------------------------------------------------
def compensation_matrix():
    """
    (6, 3, 2) matrix C with u = sum C * bend: the lab's accumulative cable-coupling formula
    dL_k = -sum_{j <= s(k)} theta_j d_k cos(phi_j - alpha_k), with theta_j (cos phi_j, sin phi_j)
    = the bend vector of section j (constants.K_to_Length_3Seg, in metres).
    """
    C = np.zeros((N_MOTORS, 3, 2))
    for m in range(N_MOTORS):
        d = CABLE_RADIUS[MOTOR_SECTION[m]]
        for j in range(MOTOR_SECTION[m] + 1):
            C[m, j] = -d * np.array([np.cos(MOTOR_ALPHA[m]), np.sin(MOTOR_ALPHA[m])])
    return C


# ---------------------------------------------------------------------------
# Gravity on the arm (the real robot hangs from the elevator)
# ---------------------------------------------------------------------------
class GravityModel:
    """Generalised gravity torques on the 30 bending joints (numpy), masses / COMs from the MJCF."""

    def __init__(self, geom: CableGeometry, mjcf_path=SOURCE_MJCF, g=9.81):
        self.geom, self.kin, self.g = geom, geom.kin, g
        mass, com = np.zeros(self.kin.nbody), np.zeros((self.kin.nbody, 3))
        for body in ET.parse(mjcf_path).getroot().iter("body"):
            inertial = body.find("inertial")
            if inertial is not None:
                b = self.kin.body_index[body.get("name")]
                mass[b] = float(inertial.get("mass"))
                com[b] = [float(v) for v in inertial.get("pos", "0 0 0").split()]
        arm = [self.kin.body_index[n] for n in self.kin.body_names if n.startswith(("Seg", "Dummy_Link"))]
        self.arm, self.mass, self.com = np.array(arm), mass[arm], com[arm]
        # the arm is a serial chain in body order: joint j moves its own body and every arm body after it
        joint_body = [self.kin.joint_body[qa] for qa in geom.joint_qadr]
        self.joint_body = np.array(joint_body)
        self.joint_first = np.array([list(arm).index(b) for b in joint_body])
        self.joint_axis = self.kin.joint_axis[geom.joint_qadr]                    # (30, 3), body frame
        if np.any(np.diff(self.joint_first) < 0):
            raise ValueError("arm bodies are not in chain order")

    @property
    def total_mass(self):
        return float(self.mass.sum())

    def potential(self, q30):
        qpos = np.zeros(np.shape(q30)[:-1] + (self.kin.nq,))
        qpos[..., self.geom.joint_qadr] = q30
        xpos, xmat = self.kin.forward(qpos)
        z = xpos[..., self.arm, 2] + np.einsum("...bj,bj->...b", xmat[..., self.arm, 2, :], self.com)
        return self.g * (z * self.mass).sum(-1)

    def torque(self, q30):
        """
        -dU/dq (N, 30) with one forward-kinematics pass: tau_j = a_j . sum_{b after j} (c_b - p_j) x F_b
        (a_j, p_j: world axis / position of joint j, c_b: COM of body b, F_b = m_b g).
        """
        q30 = np.atleast_2d(np.asarray(q30, dtype=np.float64))
        qpos = np.zeros(q30.shape[:-1] + (self.kin.nq,))
        qpos[..., self.geom.joint_qadr] = q30
        xpos, xmat = self.kin.forward(qpos)
        com = xpos[:, self.arm] + np.einsum("nbij,bj->nbi", xmat[:, self.arm], self.com)        # (N, B, 3)
        force = np.zeros_like(com)
        force[..., 2] = -self.g * self.mass
        moment = np.cross(com, force)
        suffix_m = np.cumsum(moment[:, ::-1], axis=1)[:, ::-1]                              # sum over b >= k
        suffix_f = np.cumsum(force[:, ::-1], axis=1)[:, ::-1]
        k = self.joint_first
        p = xpos[:, self.joint_body]                                                         # (N, 30, 3)
        axis = np.einsum("njik,jk->nji", xmat[:, self.joint_body], self.joint_axis)
        return np.einsum("nji,nji->nj", axis, suffix_m[:, k] - np.cross(p, suffix_f[:, k]))

    def stiffness(self, q30, eps=1e-6):
        """d tau / dq (N, 30, 30) by central differences of `torque`, in one batched kinematics pass."""
        q30 = np.atleast_2d(np.asarray(q30, dtype=np.float64))
        n = q30.shape[0]
        pts = np.repeat(q30[:, None], 60, axis=1)
        idx = np.arange(30)
        pts[:, 2 * idx, idx] += eps
        pts[:, 2 * idx + 1, idx] -= eps
        tau = self.torque(pts.reshape(-1, 30)).reshape(n, 60, 30)
        return ((tau[:, 0::2] - tau[:, 1::2]) / (2.0 * eps)).transpose(0, 2, 1)

    def torque_fd(self, q30, eps=1e-6):
        """-dU/dq by central differences (check of `torque`)."""
        q30 = np.asarray(q30, dtype=np.float64)
        pts = np.repeat(q30[:, None], 60, axis=1)
        idx = np.arange(30)
        pts[:, 2 * idx, idx] += eps
        pts[:, 2 * idx + 1, idx] -= eps
        u = self.potential(pts)
        return -(u[:, 0::2] - u[:, 1::2]) / (2.0 * eps)


# ---------------------------------------------------------------------------
# Torch model of the cable forces (applied by cable_env.py at every physics step)
# ---------------------------------------------------------------------------
class CableModel:
    """
    Batched cable model on torch. q (N, 30) chain-order joint angles, u (N, 6) motor-nut positions.
      stretch e = G q - S u           (S (12, 6): cable c follows sign_c * u_motor(c))
      tension T = k_c max(0, e + T0 / k_c) + c_c de/dt   (one_sided)   or   T0 + k_c e + c_c de/dt
      joint torque = -G^T T           (+ backbone spring -k_j q from the joint drives, + gravity)
    """

    def __init__(self, geom: CableGeometry, device="cpu", dtype=torch.float64, one_sided=True,
                 k_c=CABLE_STIFFNESS, t0=CABLE_PRETENSION, k_j=BACKBONE_STIFFNESS, c_c=CABLE_DAMPING,
                 gravity: GravityModel | None = None):
        self.device, self.dtype = device, dtype
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        self.G = t(geom.G)                                       # (12, 30)
        S = np.zeros((N_CABLES, N_MOTORS))
        S[np.arange(N_CABLES), CABLE_MOTOR] = CABLE_SIGN
        self.S = t(S)                                            # (12, 6)
        self.C = t(compensation_matrix()).reshape(N_MOTORS, 6)   # u = C @ bend.flatten()
        self.one_sided, self.k_c, self.t0 = one_sided, float(k_c), float(t0)
        self.k_j, self.c_c = float(k_j), float(c_c)
        self.gravity = gravity
        self.u_max = t(MOTOR_RANGE)
        self.u_rate = t(MOTOR_RATE)

    def stretch(self, q, u):
        return q.to(self.dtype) @ self.G.T - u.to(self.dtype) @ self.S.T

    def tension(self, q, u, qd=None, ud=None):
        Gqd = None if qd is None else qd.to(self.dtype) @ self.G.T
        return self.tension_from(q.to(self.dtype) @ self.G.T, u, Gqd, ud)

    def tension_from(self, Gq, u, Gqd=None, ud=None):
        """Tension law with the arm's part of the cable length (G q, G qdot) already computed."""
        e = Gq - u.to(self.dtype) @ self.S.T
        if not self.one_sided:
            T = self.t0 + self.k_c * e
        else:
            T = self.k_c * torch.clamp(e + self.t0 / self.k_c, min=0.0)
        if Gqd is not None:
            rate = Gqd if ud is None else Gqd - ud.to(self.dtype) @ self.S.T
            if self.one_sided:
                T = torch.where(T > 0.0, torch.clamp(T + self.c_c * rate, min=0.0), T)
            else:
                T = T + self.c_c * rate
        return T

    def joint_torque(self, T):
        """Cable tensions (N, 12) -> generalised joint torques (N, 30)."""
        return -T @ self.G

    def nut_load(self, T):
        """Cable tensions (N, 12) -> force of the cables on each motor nut, + = towards paying cable a out (N, 6)."""
        return T @ self.S

    def _gravity(self, q):
        if self.gravity is None:
            return torch.zeros_like(q, dtype=self.dtype)
        return torch.as_tensor(self.gravity.torque(q.detach().cpu().numpy()), dtype=self.dtype, device=self.device)

    def residual(self, q, u):
        """Net joint torque (N, 30) of cables + backbone springs + gravity (zero at an equilibrium)."""
        q = q.to(self.dtype)
        return self.joint_torque(self.tension(q, u)) - self.k_j * q + self._gravity(q)

    def compensate(self, bend):
        """Controller mapping (thesis eq. 3.32): bend vectors (N, 3, 2) -> motor displacements (N, 6)."""
        return bend.to(self.dtype).reshape(-1, 6) @ self.C.T

    def equilibrium(self, u, q0=None, iters=40, tol=1e-8):
        """
        Static pose (N, 30) for motor positions u (N, 6): Newton on
            r(q) = -k_j q - k_c G_a^T (G_a q - S_a u + b0) + tau_gravity(q) = 0
        with the active set of taut cables; the gravity stiffness d tau / dq is taken once at q0.
        """
        u = u.to(self.dtype)
        n = u.shape[0]
        q = torch.zeros(n, 30, dtype=self.dtype, device=self.device) if q0 is None else q0.to(self.dtype).clone()
        eye = torch.eye(30, dtype=self.dtype, device=self.device)
        b0 = self.t0 / self.k_c
        K_g = None
        if self.gravity is not None:
            K_g = torch.as_tensor(self.gravity.stiffness(q.cpu().numpy()), dtype=self.dtype, device=self.device)
        for _ in range(iters):
            e = self.stretch(q, u) + b0
            active = (e > 0.0).to(self.dtype) if self.one_sided else torch.ones_like(e)
            T = self.k_c * e * active                                                # taut cables only
            r = -self.k_j * q - (T[:, None, :] @ self.G[None])[:, 0] + self._gravity(q)
            Ga = self.G[None] * active[:, :, None]                                  # (N, 12, 30)
            H = self.k_j * eye + self.k_c * Ga.transpose(1, 2) @ Ga                 # = -dr/dq without gravity
            if K_g is not None:
                H = H - K_g
            step = torch.linalg.solve(H, r[:, :, None])[:, :, 0]
            q = q + step
            if step.abs().max() < tol:
                break
        return q

    def motors_for_pose(self, q_des, iters=4):
        """Motor positions (N, 6) that hold the pose q_des (N, 30) best (least-squares torque balance)."""
        q_des = q_des.to(self.dtype)
        b0 = self.t0 / self.k_c
        tau_g = self._gravity(q_des)
        u = (self.G @ q_des.T).T @ torch.linalg.pinv(self.S).T                     # rigid-cable guess
        for _ in range(iters):
            e = self.stretch(q_des, u) + b0
            active = (e > 0.0).to(self.dtype) if self.one_sided else torch.ones_like(e)
            # k_j q + k_c G^T A (G q + b0) - k_c G^T A S u = tau_gravity  ->  least squares in u
            GA = self.G.T[None] * active[:, None, :]                               # (N, 30, 12)
            A = self.k_c * GA @ self.S                                             # (N, 30, 6)
            rhs = self.k_j * q_des + self.k_c * (GA @ (q_des @ self.G.T + b0)[:, :, None])[:, :, 0] - tau_g
            u = torch.linalg.lstsq(A, rhs[:, :, None]).solution[:, :, 0]
        return torch.maximum(torch.minimum(u, self.u_max), -self.u_max)
