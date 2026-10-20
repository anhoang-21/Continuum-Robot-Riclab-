"""
==============================================================================
shape_control.py - model-based shape controllers of the cable-driven robot (ND2)
==============================================================================
Shape tracking: the three section bend vectors b (3 x 2) follow a reference b_ref(t), with the motor
positions u (6 lead-screw nuts) as the input. The static map of the robot is b = F(u; theta), theta = cable
stiffness k_c, pre-tension T0, backbone stiffness k_j (+ friction, backlash, payload, gravity).

  formula     u = C b_ref                         the lab's controller (thesis eq. 3.32, rigid cables; the Qt GUI)
  statics     u = F^-1(b_ref; theta_nom)          inverse statics of the nominal model (elastic cables,
                                                  backbone, gravity): StaticSolver.motors_for_pose
  statics+PI  u = F^-1(b_ref; theta_nom) + M^-1 (K_P e + K_I int e),  e = b_ref - b_meas
              M = dF/du at the reference (6 x 6), b_meas from the shape sensor (mocap / IMU, with noise)
  adaptive    statics+PI with theta_hat instead of theta_nom (certainty equivalence), theta_hat adapted online
              every `adapt_every` steps by a damped Gauss-Newton step on the noise-normalised residuals
              r = y_meas - y(u; theta_hat), y = section bends (+ the 12 load-cell tensions for
              "adaptive+loadcell"): theta_hat <- theta_hat + gamma (J^T J + (1 + eps tr J^T J) I)^-1 J^T r,
              J = dy/dtheta by finite differences, then projected to physical ranges.

For a static map that is (locally) linear in theta the damped step is a normalised gradient estimator in the
J^T J metric (Ioannou & Sun, Robust Adaptive Control, ch. 4): bounded, and convergent under persistent
excitation as long as the residual is due to theta; the PI term removes what the parameters cannot explain
(friction offsets, payload). The pre-tension does not move the shape while every cable is taut
(docs/stage6_theory.md, section 2): only the load-cell residual identifies it.

All controllers return motor TARGETS; play_shape.py turns them into rate-limited motor speeds of
robust_env.RobustCableEnv (and the tension CBF can sit underneath).
==============================================================================
"""

from __future__ import annotations

import numpy as np
import torch

import cables as CB
from uncertain_cables import StaticSolver


def bend_to_q(geom, bend):
    return torch.as_tensor(geom.q_bend(np.asarray(bend)))


def q_to_bend(q):
    b = q.view(-1, 3, 5, 2).sum(dim=2)
    return torch.stack([b[..., 1], -b[..., 0]], dim=-1)


class ShapeController:
    name = "controller"

    def __init__(self, geom, statics: StaticSolver, n: int, t0: float = CB.CABLE_PRETENSION):
        self.geom, self.ss, self.n = geom, statics, n
        self.k_c = torch.full((n, 12), CB.CABLE_STIFFNESS, dtype=torch.float64)
        self.t0 = torch.full((n, 12), float(t0), dtype=torch.float64)
        self.k_j = torch.full((n,), CB.BACKBONE_STIFFNESS, dtype=torch.float64)
        self.C = torch.as_tensor(CB.compensation_matrix().reshape(6, 6))

    def reset(self):
        pass

    def feedforward(self, b_ref):
        return self.ss.motors_for_pose(bend_to_q(self.geom, b_ref), self.k_c, self.t0, self.k_j)

    def __call__(self, b_ref, b_meas, u_now):
        raise NotImplementedError


class FormulaController(ShapeController):
    name = "formula"

    def __call__(self, b_ref, b_meas, u_now):
        return torch.as_tensor(b_ref).reshape(-1, 6) @ self.C.T


class StaticsController(ShapeController):
    name = "statics"

    def __call__(self, b_ref, b_meas, u_now):
        return self.feedforward(b_ref)


class StaticsPIController(ShapeController):
    name = "statics+PI"

    def __init__(self, *a, kp=0.3, ki=2.0, dt=0.02, **kw):
        super().__init__(*a, **kw)
        self.kp, self.ki, self.dt = kp, ki, dt
        self.integral = torch.zeros(self.n, 6, dtype=torch.float64)
        self._Minv = None

    def reset(self):
        self.integral.zero_()
        self._Minv = None

    def sensitivity(self, u, eps=2e-5):
        """M = d bend / d u (N, 6, 6) of the controller's model at motor positions u."""
        cols = []
        q0 = self.ss.equilibrium(u, self.k_c, self.t0, self.k_j)
        for m in range(6):
            du = torch.zeros_like(u)
            du[:, m] = eps
            qp = self.ss.equilibrium(u + du, self.k_c, self.t0, self.k_j, q0=q0, iters=8)
            qm = self.ss.equilibrium(u - du, self.k_c, self.t0, self.k_j, q0=q0, iters=8)
            cols.append((q_to_bend(qp) - q_to_bend(qm)).reshape(-1, 6) / (2 * eps))
        return torch.stack(cols, dim=-1)

    def __call__(self, b_ref, b_meas, u_now):
        u_ff = self.feedforward(b_ref)
        if self._Minv is None or self._refresh:
            self._Minv = torch.linalg.pinv(self.sensitivity(u_ff))
            self._refresh = False
        e = (torch.as_tensor(b_ref) - torch.as_tensor(b_meas)).reshape(-1, 6)
        self.integral = torch.clamp(self.integral + e * self.dt, -0.3, 0.3)
        v = self.kp * e + self.ki * self.integral
        return u_ff + (self._Minv @ v[:, :, None])[:, :, 0]

    _refresh = True

    def refresh(self):
        """Re-linearise at the next call (the reference moved far)."""
        self._refresh = True


class AdaptiveController(StaticsPIController):
    """
    statics+PI on the adapted model theta_hat = (log k_c, T0, k_j). Residuals in noise units: the measured
    bends (sigma_b) and, with load cells, the 12 nut-side tensions (sigma_T); the pre-tension is only
    observable through the tensions (a pair's common-mode tension does not move the shape).
    """
    name = "adaptive"

    LO = np.array([np.log(800.0), 0.5, 0.3])
    HI = np.array([np.log(12000.0), 9.0, 2.0])

    def __init__(self, *a, gamma=0.5, adapt_every=10, relin_every=50, sigma_b=0.004, sigma_t=0.1, use_tension=True, **kw):
        super().__init__(*a, **kw)
        self.gamma, self.sigma_b, self.sigma_t, self.use_tension = gamma, sigma_b, sigma_t, use_tension
        self.adapt_every, self.relin_every, self._k = adapt_every, relin_every, 0
        self.theta0 = torch.as_tensor(np.array([np.log(CB.CABLE_STIFFNESS), float(self.t0[0, 0]),
                                                CB.BACKBONE_STIFFNESS]), dtype=torch.float64)
        self.theta = self.theta0.repeat(self.n, 1)
        self.k_c, self.t0, self.k_j = self._apply()
        if use_tension:
            self.name = "adaptive+loadcell"

    def reset(self):
        super().reset()
        self._k = 0
        self.theta = self.theta0.repeat(self.n, 1)
        self.k_c, self.t0, self.k_j = self._apply()

    def _apply(self, theta=None):
        th = self.theta if theta is None else theta
        return (torch.exp(th[:, :1]).expand(-1, 12).clone(), th[:, 1:2].expand(-1, 12).clone(), th[:, 2].clone())

    def _model(self, u, theta):
        """Normalised model outputs (bends / sigma_b, tensions / sigma_T) at motor positions u."""
        k_c, t0, k_j = self._apply(theta)
        q = self.ss.equilibrium(u, k_c, t0, k_j, iters=12)
        out = [q_to_bend(q).reshape(-1, 6) / self.sigma_b]
        if self.use_tension:
            out.append(self.ss.tension(q, u, k_c, t0) / self.sigma_t)
        return torch.cat(out, dim=1)

    def adapt(self, u, b_meas, t_meas=None):
        y_hat = self._model(u, self.theta)
        h = torch.tensor([0.02, 0.05, 0.02], dtype=torch.float64)
        J = []
        for i in range(3):
            th = self.theta.clone()
            th[:, i] += h[i]
            J.append((self._model(u, th) - y_hat) / h[i])
        J = torch.stack(J, dim=-1)                                                # (N, m, 3)
        y = [torch.as_tensor(b_meas).reshape(-1, 6) / self.sigma_b]
        if self.use_tension:
            y.append(torch.as_tensor(t_meas) / self.sigma_t)
        r = torch.cat(y, dim=1) - y_hat
        # normalised gradient step (Gauss-Newton direction scaled by gamma, guarded by the normalisation)
        JTJ = J.transpose(1, 2) @ J
        step = torch.linalg.solve(JTJ + torch.eye(3, dtype=torch.float64) * (1.0 + 1e-3 * JTJ.diagonal(dim1=1, dim2=2).sum(-1))[:, None, None],
                                  (J.transpose(1, 2) @ r[:, :, None]))[:, :, 0]
        self.theta = self.theta + self.gamma * step
        self.theta = torch.minimum(torch.maximum(self.theta, torch.as_tensor(self.LO)), torch.as_tensor(self.HI))
        self.k_c, self.t0, self.k_j = self._apply()
        if self._k % self.relin_every == 0:
            self.refresh()                                   # the PI's inverse sensitivity on the new model

    def __call__(self, b_ref, b_meas, u_now, t_meas=None):
        self._k += 1
        if self._k % self.adapt_every == 0:
            self.adapt(u_now, b_meas, t_meas)
        return super().__call__(b_ref, b_meas, u_now)


CONTROLLERS = {c.name: c for c in (FormulaController, StaticsController, StaticsPIController, AdaptiveController)}


# ---------------------------------------------------------------------------
# Reference trajectories (bend vectors over time)
# ---------------------------------------------------------------------------
def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def reference(kind, t, b_goal):
    """b_ref (N, 3, 2) at time t (s). b_goal (N, 3, 2): the S-curve / step target of every env."""
    if kind == "scurve":            # form the S-curve in 2 s, hold, back to straight from 6 s
        s = min_jerk(t / 2.0) if t < 6.0 else 1.0 - min_jerk((t - 6.0) / 2.0)
        return s * b_goal
    if kind == "circle":            # every section bend at its goal magnitude, direction turning at 0.25 Hz
        mag = np.linalg.norm(b_goal, axis=-1, keepdims=True) * min_jerk(t / 1.0)
        ang = 2 * np.pi * 0.25 * t + np.arange(3)[None, :, None] * 2.1
        return mag * np.concatenate([np.cos(ang), np.sin(ang)], axis=-1)
    if kind == "steps":             # three steps between 0, the goal and its mirror
        return b_goal * (1.0 if (t % 6.0) < 2.0 else (-0.6 if (t % 6.0) < 4.0 else 0.0))
    raise ValueError(kind)
