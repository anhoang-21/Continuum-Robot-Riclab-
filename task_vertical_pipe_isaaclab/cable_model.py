"""
==============================================================================
cable_model.py - Tendon (cable) drive of the continuum robot
==============================================================================
The real robot is a chain of 15 universal joints pulled by 6 lead-screw motors. Every motor
drives an antagonistic PAIR of cables (at angle alpha and alpha + 180 deg): the screw shortens
one cable and pays out the other by the same length. Two motors (alpha, alpha + 90 deg) per
section, cable hole radius 40 / 30 / 20 mm. The cables of section 2 run through the discs of
section 1 and end on Seg10, those of section 3 run through sections 1 and 2 and end on Seg15
(constants.K_to_Length_3Seg is the first-order model of exactly this routing).

Model
  * every cable passes one hole in the disc of Elevator_Link (start) and of every Seg body up to its
    anchor (Seg5 / Seg10 / Seg15); the cable path is the polyline through these holes
  * length L = sum of the straight pieces between consecutive holes (exact, nonlinear)
  * free length  L_free = L0 -/+ s - preload / k     (s = lead-screw displacement of the motor,
    + cable shortened, - cable paid out)
  * one-way spring damper  T = max(0, k (L - L_free) + c dL/dt)   (a cable pulls, never pushes)
  * a piece with tension T pulls its two end discs towards each other with T -> force + torque
    (about the centre of mass) on every body; the sum over the robot is zero, like a real tendon

Motor order (section order): [S1 alpha 0, S1 alpha 90, S2 alpha 150, S2 alpha 240, S3 alpha 30,
S3 alpha 300]; cable index c = 2 m (+ cable) or 2 m + 1 (- cable). The alphas are those of
constants.MOTOR_ALPHAS (which lists the section-3 motors before the section-2 ones).

The maths only uses operations that numpy and torch share, so the same functions run on the GPU
inside the Isaac Lab env and on numpy arrays in tests/check_cable_model.py.
==============================================================================
"""

import numpy as np

try:
    import torch
except ImportError:      # the geometry / tests run on numpy alone
    torch = None

import constants as C

N_MOTORS = 6
N_CABLES = 12
N_BODIES = 16            # Elevator_Link, Seg1 .. Seg15
N_PIECES = 15            # cable pieces between consecutive bodies
BODY_NAMES = ["Elevator_Link"] + [f"Seg{i}" for i in range(1, 16)]

MOTOR_SECTION = np.array([0, 0, 1, 1, 2, 2])
MOTOR_ALPHA = np.array([C.MOTOR_ALPHAS[i] for i in (0, 1, 4, 5, 2, 3)])
MOTOR_RADIUS = np.array([C.CABLE_RADIUS[s] for s in MOTOR_SECTION])
SECTION_ANCHOR = np.array([5, 10, 15])          # body index (= SegN) where the cables of a section end

HOLE_Z = 0.019           # hole height in the Seg frame (middle of the 38 mm link); the first hole is 19 mm above Seg1

# The bend vector (u, v) of a section bends the tip towards the direction (v, u) of the Seg frame
# (kinematics.joint_angles), so the cable at motor angle alpha sits at (r sin alpha, r cos alpha) and
# is shortened by r (u cos alpha + v sin alpha) = r theta cos(phi - alpha), as in K_to_Length_3Seg.


def _xp(x):
    return np if isinstance(x, np.ndarray) else torch


def _norm_sq(d):
    return (d * d).sum(-1)


def bend_to_motor_matrix():
    """M (6, 6): motor displacements (m) = M @ bend (6 = 3 sections x (u, v)), first order."""
    M = np.zeros((N_MOTORS, N_MOTORS))
    for m in range(N_MOTORS):
        for s in range(MOTOR_SECTION[m] + 1):          # the cable passes sections 0 .. own section
            M[m, 2 * s] = MOTOR_RADIUS[m] * np.cos(MOTOR_ALPHA[m])
            M[m, 2 * s + 1] = MOTOR_RADIUS[m] * np.sin(MOTOR_ALPHA[m])
    return M


def motor_rate():
    """Motor displacement per env step for action = 1: the section bend changes by BEND_RATE (as the joint env)."""
    return np.array([C.BEND_RATE[s] * MOTOR_RADIUS[m] for m, s in enumerate(MOTOR_SECTION)])


def motor_limit():
    """|s| bound of every motor: all sections it passes bent to THETA_MAX towards it."""
    return MOTOR_RADIUS * C.THETA_MAX * (MOTOR_SECTION + 1)


class CableRouting:
    """Constant geometry of the 12 cables (numpy float64)."""

    def __init__(self, kin=None, hole_z=HOLE_Z, radial_offset=0.0):
        """radial_offset: holes moved outwards by this much (only to draw the cables outside the CAD housing)."""
        if kin is None:
            from kinematics import ContinuumKinematics
            kin = ContinuumKinematics()
        self.kin = kin
        self.body_ids = np.array([kin.body_index[n] for n in BODY_NAMES])
        xpos, xmat = kin.forward(np.zeros(kin.nq))
        rest_pos, rest_rot = xpos[self.body_ids], xmat[self.body_ids]

        self.cable_motor = np.repeat(np.arange(N_MOTORS), 2)                 # (12,)
        self.sign = np.tile([1.0, -1.0], N_MOTORS)                           # + cable shortened by s, - cable lengthened
        alpha = np.repeat(MOTOR_ALPHA, 2) + np.tile([0.0, np.pi], N_MOTORS)
        radius = np.repeat(MOTOR_RADIUS, 2)
        anchor = SECTION_ANCHOR[np.repeat(MOTOR_SECTION, 2)]                 # (12,)
        self.alpha, self.radius, self.anchor = alpha, radius, anchor
        # piece i (between body i and i + 1) belongs to the cable if i < anchor
        self.mask = (np.arange(N_PIECES)[:, None] < anchor[None, :]).astype(np.float64)   # (15, 12)

        self.hole = np.zeros((N_BODIES, N_CABLES, 3))                        # hole positions in the body frames
        for c in range(N_CABLES):
            x, y = (radius[c] + radial_offset) * np.sin(alpha[c]), (radius[c] + radial_offset) * np.cos(alpha[c])
            self.hole[1:, c] = (x, y, hole_z)
            # the first hole: same disc pattern 'hole_z' above Seg1 (Seg1's rest frame), stored in the Elevator_Link frame
            p_world = rest_pos[1] + rest_rot[1] @ np.array([x, y, -hole_z])
            self.hole[0, c] = rest_rot[0].T @ (p_world - rest_pos[0])
        self.rest_len = path_lengths(self.hole, self.mask, rest_pos, rest_rot)     # (12,)

    def lengths_of_qpos(self, qpos):
        """Cable lengths (..., 12) of MuJoCo-order joint positions (..., nq) (numpy FK)."""
        xpos, xmat = self.kin.forward(qpos)
        return path_lengths(self.hole, self.mask, xpos[..., self.body_ids, :], xmat[..., self.body_ids, :, :])

    def motors_of_qpos(self, qpos):
        """Motor displacements (..., 6) that leave both cables of every pair equally tight at this pose."""
        L = self.lengths_of_qpos(qpos)
        return 0.5 * (L[..., 1::2] - L[..., 0::2])


def path_lengths(hole, mask, pos, rot):
    """
    Cable path lengths. hole (16, 12, 3), mask (15, 12); pos (..., 16, 3), rot (..., 16, 3, 3) = body poses.
    Works on numpy arrays and on torch tensors of the same device / dtype.
    """
    xp = _xp(pos)
    P = pos[..., :, None, :] + xp.einsum("...bij,bcj->...bci", rot, hole)     # (..., 16, 12, 3)
    d = P[..., 1:, :, :] - P[..., :-1, :, :]
    pieces = _norm_sq(d) ** 0.5                                               # (..., 15, 12)
    return (pieces * mask).sum(-2)


class CableForceModel:
    """
    Tension and body wrenches of the 12 cables for a batch of robots (torch, any device).
    All positions are in a common frame (the env frame; forces are returned in that frame).
    """

    def __init__(self, routing, device, stiffness, damping, preload, max_tension, dtype=None):
        dtype = dtype or torch.float32
        kw = dict(device=device, dtype=dtype)
        self.hole = torch.as_tensor(routing.hole, **kw)
        self.mask = torch.as_tensor(routing.mask, **kw)
        self.rest_len = torch.as_tensor(routing.rest_len, **kw)
        self.sign = torch.as_tensor(routing.sign, **kw)
        self.cable_motor = torch.as_tensor(routing.cable_motor, device=device)
        self.k, self.c, self.max_tension = float(stiffness), float(damping), float(max_tension)
        self.stretch0 = float(preload) / float(stiffness)      # preload: stretch of a cable at L = L_free + stretch0
        self.preload = float(preload)

    def hole_points(self, pos, rot):
        return pos[:, :, None, :] + torch.einsum("nbij,bcj->nbci", rot, self.hole)     # (N, 16, 12, 3)

    def compute(self, pos, rot, com, v_com, omega, s):
        """
        pos, com, v_com, omega (N, 16, 3): body frame origin, centre of mass, COM velocity, angular velocity;
        rot (N, 16, 3, 3); s (N, 6) motor displacements.
        -> force (N, 16, 3) at the COM, torque (N, 16, 3) about the COM, tension (N, 12), length (N, 12).
        """
        P = self.hole_points(pos, rot)
        d = P[:, 1:] - P[:, :-1]                                      # from the upper to the lower hole of a piece
        length = torch.sqrt(torch.clamp(_norm_sq(d), min=1e-12))      # (N, 15, 12)
        u = d / length[..., None]
        L = (length * self.mask).sum(1)                               # (N, 12)

        v = v_com[:, :, None, :] + torch.cross(omega[:, :, None, :].expand_as(P), P - com[:, :, None, :], dim=-1)
        dL = ((u * (v[:, 1:] - v[:, :-1])).sum(-1) * self.mask).sum(1)

        free = self.rest_len - self.sign * s[:, self.cable_motor] - self.stretch0
        T = torch.clamp(self.k * (L - free) + self.c * dL, min=0.0, max=self.max_tension)     # (N, 12)

        f_piece = (T[:, None, :] * self.mask)[..., None] * u          # pulls the upper hole down, the lower one up
        F = torch.zeros_like(P)
        F[:, :-1] += f_piece
        F[:, 1:] -= f_piece
        force = F.sum(2)
        torque = torch.cross(P - com[:, :, None, :], F, dim=-1).sum(2)
        return force, torque, T, L
