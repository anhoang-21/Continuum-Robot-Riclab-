"""
==============================================================================
uncertain_cables.py - cable drive with per-robot parameters, capstan friction and nut backlash
==============================================================================
cables.CableModel with what it leaves out (ND1), batched over the envs on torch:

  * every env has its own parameters (uncertainty.UncertaintySet): k_c, T0, cable damping per cable;
  * capstan friction in the cable path (Bowden sheath, disc holes), the element of home_hysteresis.py:

        nut --- k_n --- [friction element, payout p] --- k_r --- robot        1/k_c = 1/k_n + 1/k_r
        T_n = k_n max(0, p - s + T0/k_n)      (nut side: what a load cell on the nut reads)
        T_r = k_r max(0, g - p + T0/k_r)      (robot side: what loads the discs)

    with s = S x the payout at the nut, g = G q the length the arm demands. The element sticks
    (p fixed) while exp(-mu theta) <= T_n / T_r <= exp(+mu theta) and otherwise slides with the ratio
    on the bound lam:  T_r = kappa k_c max(0, g - s + T0/k_c),  T_n = lam T_r,
    kappa = (k_n + k_r) / (k_n + lam k_r)  (a sliding cable obeys the frictionless law scaled by kappa);
  * nut backlash: a play of width b between the screw (encoder, motor x) and the nut (xn), as the play
    operator of home_hysteresis.py: xn = clip(xn, x - b/2, x + b/2).

Dynamics, every 2 ms physics step (cable_env.py integrates the motors in 4 sub-steps against the nut load):
  begin_step: the mode of every cable for this step from the current state: stuck cables whose ratio left
              the band start to slide; sliding cables keep sliding while p keeps moving the same way
              (no stick-slip chatter from one step to the next), else they stick;
  nut_load:   T_n of the step's law at the motor sub-step positions -> force on the nuts (S^T T_n);
  end_step:   commit the play, move p of the sliding cables onto the bound, T_r / T_n of the step.
Damping (Kelvin-Voigt, same time constant c_c / k_c in both springs): sliding c_c kappa (g' - s'),
stuck robot side c_c (k_r / k_c) g', stuck nut side -c_c (k_n / k_c) s'.

With mu theta = 0 every cable slides with lam = 1, kappa = 1, so T_r = T_n is exactly the law of
cables.CableModel.tension_from (tests/check_robust_env.py checks bit-identical trajectories).

StaticSolver: equilibrium / motors-for-pose with per-env k_c, T0, k_j and a tip payload (start states,
safety filter, system identification). Everything runs on torch; the gravity / payload torques on numpy
(cables.GravityModel).
==============================================================================
"""

from __future__ import annotations

import numpy as np
import torch

import cables as CB


def _t(a, device, dtype=torch.float64):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


class UncertainCableModel:
    """Batched cable drive of `num_envs` robots with their own parameters (all buffers updated in place)."""

    def __init__(self, geom: CB.CableGeometry, num_envs: int, device="cpu", dtype=torch.float64):
        self.device, self.dtype, self.n = device, dtype, int(num_envs)
        self.G = _t(geom.G, device, dtype)                                   # (12, 30)
        S = np.zeros((CB.N_CABLES, CB.N_MOTORS))
        S[np.arange(CB.N_CABLES), CB.CABLE_MOTOR] = CB.CABLE_SIGN
        self.S = _t(S, device, dtype)                                        # (12, 6)
        self.C = _t(CB.compensation_matrix(), device, dtype).reshape(CB.N_MOTORS, 6)
        self.u_max = _t(CB.MOTOR_RANGE, device, dtype)
        self.u_rate = _t(CB.MOTOR_RATE, device, dtype)
        n, z12, z6 = self.n, (self.n, CB.N_CABLES), (self.n, CB.N_MOTORS)
        zeros = lambda shape: torch.zeros(shape, dtype=dtype, device=device)  # noqa: E731
        # parameters (per env / cable)
        self.k_c, self.t0, self.c_c = zeros(z12), zeros(z12), zeros(z12)
        self.k_n, self.k_r = zeros(z12), zeros(z12)
        self.lam_hi, self.lam_lo = zeros(z12), zeros(z12)
        self.fric = torch.zeros(z12, dtype=torch.bool, device=device)        # mu theta > 0
        self.half_play = zeros(z6)
        # state
        self.p = zeros(z12)                                                  # payout at the friction element
        self.xn = zeros(z6)                                                  # nut positions (after the play)
        self.mode = zeros(z12)                                               # 0 stick, +1 / -1 slide on lam_hi / lam_lo
        # per-step buffers
        self.lam, self.kappa = zeros(z12), zeros(z12)
        self.T_r, self.T_n = zeros(z12), zeros(z12)
        self.set_params(torch.arange(n, device=device), {
            "k_c": np.full(z12, CB.CABLE_STIFFNESS), "t0": np.full(z12, CB.CABLE_PRETENSION),
            "c_c": np.full(z12, CB.CABLE_DAMPING), "mu_theta": np.zeros(z12), "phi": np.full(z12, 0.5),
            "backlash": np.zeros(z6)})

    # ------------------------------------------------------------------
    def param_tensors(self):
        return (self.k_c, self.t0, self.c_c, self.k_n, self.k_r, self.lam_hi, self.lam_lo, self.fric, self.half_play)

    def state_tensors(self):
        return (self.p, self.xn, self.mode, self.lam, self.kappa, self.T_r, self.T_n)

    def set_params(self, ids, prm: dict):
        """Write the parameters of envs `ids` (prm: uncertainty.UncertaintySet.sample of those envs)."""
        t = lambda k: _t(prm[k], self.device, self.dtype)                    # noqa: E731
        k_c, phi, mu = t("k_c"), torch.clamp(t("phi"), 1e-3, 1.0 - 1e-3), t("mu_theta")
        self.k_c[ids], self.t0[ids], self.c_c[ids] = k_c, t("t0"), t("c_c")
        self.k_n[ids], self.k_r[ids] = k_c / phi, k_c / (1.0 - phi)
        self.lam_hi[ids], self.lam_lo[ids] = torch.exp(mu), torch.exp(-mu)
        self.fric[ids] = mu > 0.0
        self.half_play[ids] = 0.5 * t("backlash")

    def reset_state(self, ids, Gq, x):
        """Robot at rest at joint lengths Gq (k, 12), motors at x (k, 6): nut engaged, no friction preload."""
        Gq, x = Gq.to(self.dtype), x.to(self.dtype)
        self.xn[ids] = x
        s = x @ self.S.T
        k_n, k_r, t0, k_c = self.k_n[ids], self.k_r[ids], self.t0[ids], self.k_c[ids]
        p = (k_r * Gq + k_n * s) / (k_n + k_r)                                  # T_n = T_r
        p = torch.minimum(torch.maximum(p, torch.minimum(Gq + t0 / k_r, s - t0 / k_n)),
                          torch.maximum(Gq + t0 / k_r, s - t0 / k_n))          # slack: both sides at zero
        self.p[ids] = p
        self.mode[ids] = torch.where(self.fric[ids], 0.0, 1.0).to(self.dtype)
        T = k_c * torch.clamp(Gq - s + t0 / k_c, min=0.0)
        self.T_r[ids], self.T_n[ids] = T, T

    # ------------------------------------------------------------------
    def _p_slide(self, lam, g, s):
        """Payout p with T_n = lam T_r (taut), or p kept inside the slack interval."""
        k_n, k_r, t0 = self.k_n, self.k_r, self.t0
        taut = g - s + t0 / self.k_c > 0.0
        p_taut = (lam * (t0 + k_r * g) - t0 + k_n * s) / (k_n + lam * k_r)
        p_slack = torch.minimum(torch.maximum(self.p, g + t0 / k_r), s - t0 / k_n)
        return torch.where(taut, p_taut, p_slack)

    def _payout(self, x):
        xn = torch.minimum(torch.maximum(self.xn, x - self.half_play), x + self.half_play)
        return xn, xn @ self.S.T

    def begin_step(self, Gq, x):
        """Mode of every cable for the coming physics step (state at its start: lengths Gq, motors x)."""
        g = Gq
        _, s = self._payout(x)
        k_n, k_r, t0 = self.k_n, self.k_r, self.t0
        T_n0 = k_n * torch.clamp(self.p - s + t0 / k_n, min=0.0)
        T_r0 = k_r * torch.clamp(g - self.p + t0 / k_r, min=0.0)
        tol = 1e-9 * (1.0 + T_n0 + T_r0)
        hi = T_n0 > self.lam_hi * T_r0 + tol
        lo = T_n0 < self.lam_lo * T_r0 - tol
        keep_hi = (self.mode > 0.0) & (self._p_slide(self.lam_hi, g, s) <= self.p)     # still sliding nut-wards
        keep_lo = (self.mode < 0.0) & (self._p_slide(self.lam_lo, g, s) >= self.p)
        one, zero = torch.ones_like(self.mode), torch.zeros_like(self.mode)
        mode = torch.where(hi, one, torch.where(lo, -one, torch.where(keep_hi, one, torch.where(keep_lo, -one, zero))))
        self.mode.copy_(torch.where(self.fric, mode, one))                     # no friction: always "sliding", lam 1
        self.lam.copy_(torch.where(self.mode > 0.0, self.lam_hi, torch.where(self.mode < 0.0, self.lam_lo, one)))
        self.kappa.copy_((k_n + k_r) / (k_n + self.lam * k_r))

    def _laws(self, g, gd, s, sd):
        """(T_r, T_n) of the current modes at lengths g, payout s and their rates."""
        k_c, t0, c_c, k_n, k_r = self.k_c, self.t0, self.c_c, self.k_n, self.k_r
        # sliding: the frictionless law scaled by kappa (= cables.CableModel.tension_from for kappa = lam = 1)
        T_sl = self.kappa * k_c * torch.clamp(g - s + t0 / k_c, min=0.0)
        T_sl = torch.where(T_sl > 0.0, torch.clamp(T_sl + self.kappa * c_c * (gd - sd), min=0.0), T_sl)
        # stuck: the two springs on either side of the fixed element
        T_rk = k_r * torch.clamp(g - self.p + t0 / k_r, min=0.0)
        T_rk = torch.where(T_rk > 0.0, torch.clamp(T_rk + c_c * (k_r / k_c) * gd, min=0.0), T_rk)
        T_nk = k_n * torch.clamp(self.p - s + t0 / k_n, min=0.0)
        T_nk = torch.where(T_nk > 0.0, torch.clamp(T_nk - c_c * (k_n / k_c) * sd, min=0.0), T_nk)
        stick = self.mode == 0.0
        return torch.where(stick, T_rk, T_sl), torch.where(stick, T_nk, self.lam * T_sl)

    def nut_load(self, Gq, Gqd, x, v):
        """Force of the cables on the 6 nuts (+ = towards paying cable a out) at motor positions x, speeds v."""
        _, s = self._payout(x)
        _, T_n = self._laws(Gq, Gqd, s, v @ self.S.T)
        return T_n @ self.S

    def end_step(self, Gq, Gqd, x, v):
        """Commit the step (play, payout of the sliding cables); returns the robot-side tensions T_r (N, 12)."""
        xn, s = self._payout(x)
        self.xn.copy_(xn)
        self.p.copy_(torch.where(self.mode != 0.0, self._p_slide(self.lam, Gq, s), self.p))
        T_r, T_n = self._laws(Gq, Gqd, s, v @ self.S.T)
        self.T_r.copy_(T_r)
        self.T_n.copy_(T_n)
        return self.T_r

    def joint_torque(self, T):
        return -T @ self.G

    def compensate(self, bend):
        """The lab's controller formula (thesis eq. 3.32): bend (N, 3, 2) -> motor displacements (N, 6)."""
        return bend.to(self.dtype).reshape(-1, 6) @ self.C.T


# ---------------------------------------------------------------------------
# Payload at the tip and the static model with per-env parameters
# ---------------------------------------------------------------------------
class PayloadModel:
    """Joint torques (N, 30) of a point mass at the centre of mass of Seg15 (numpy; payload in kg per sample)."""

    def __init__(self, gravity: CB.GravityModel):
        self.gm = gravity
        self.kin, self.geom = gravity.kin, gravity.geom
        self.tip_body = self.kin.body_index["Seg15"]
        self.tip_com = gravity.com[list(gravity.arm).index(self.tip_body)]

    def torque(self, q30, payload):
        q30 = np.atleast_2d(np.asarray(q30, dtype=np.float64))
        m = np.broadcast_to(np.asarray(payload, dtype=np.float64), q30.shape[:1])
        qpos = np.zeros(q30.shape[:-1] + (self.kin.nq,))
        qpos[..., self.geom.joint_qadr] = q30
        xpos, xmat = self.kin.forward(qpos)
        c = xpos[:, self.tip_body] + np.einsum("nij,j->ni", xmat[:, self.tip_body], self.tip_com)    # (N, 3)
        force = np.zeros_like(c)
        force[:, 2] = -self.gm.g * m
        p = xpos[:, self.gm.joint_body]                                                          # (N, 30, 3)
        axis = np.einsum("njik,jk->nji", xmat[:, self.gm.joint_body], self.gm.joint_axis)
        return np.einsum("nji,nji->nj", axis, np.cross(c[:, None] - p, force[:, None]))


class StaticSolver:
    """
    Static cable model with per-env parameters (torch, float64): k_c, T0 (N, 12), k_j (N,), payload (N,) kg.
    Frictionless (a robot at rest after a slow motion with the friction elements relaxed) - the friction only
    shifts the payout p, which reset_state puts at the frictionless split.
    """

    def __init__(self, geom: CB.CableGeometry, gravity: CB.GravityModel | None, device="cpu"):
        self.device, self.dtype = device, torch.float64
        self.G = _t(geom.G, device)
        S = np.zeros((CB.N_CABLES, CB.N_MOTORS))
        S[np.arange(CB.N_CABLES), CB.CABLE_MOTOR] = CB.CABLE_SIGN
        self.S = _t(S, device)
        self.u_max = _t(CB.MOTOR_RANGE, device)
        self.gravity = gravity
        self.payload = PayloadModel(gravity) if gravity is not None else None

    def ext_torque(self, q, payload=None):
        """Gravity + payload torques on the 30 joints (N, 30)."""
        if self.gravity is None:
            return torch.zeros_like(q, dtype=self.dtype)
        qn = q.detach().cpu().numpy()
        tau = self.gravity.torque(qn)
        if payload is not None and np.any(np.asarray(payload) > 0.0):
            tau = tau + self.payload.torque(qn, payload)
        return _t(tau, self.device)

    def ext_stiffness(self, q, payload=None, eps=1e-6):
        """d (gravity + payload torque) / dq (N, 30, 30), central differences in one batched pass."""
        if self.gravity is None:
            return None
        q = q.detach().cpu().numpy()
        n = q.shape[0]
        pts = np.repeat(q[:, None], 60, axis=1)
        idx = np.arange(30)
        pts[:, 2 * idx, idx] += eps
        pts[:, 2 * idx + 1, idx] -= eps
        pl = None if payload is None else np.repeat(np.asarray(payload, dtype=np.float64)[:, None], 60, axis=1).reshape(-1)
        tau = self.ext_torque(torch.as_tensor(pts.reshape(-1, 30)), pl).cpu().numpy().reshape(n, 60, 30)
        return _t(((tau[:, 0::2] - tau[:, 1::2]) / (2.0 * eps)).transpose(0, 2, 1), self.device)

    def tension(self, q, u, k_c, t0):
        e = q.to(self.dtype) @ self.G.T - u.to(self.dtype) @ self.S.T
        return k_c * torch.clamp(e + t0 / k_c, min=0.0)

    def residual(self, q, u, k_c, t0, k_j, payload=None):
        q = q.to(self.dtype)
        return -self.tension(q, u, k_c, t0) @ self.G - k_j[:, None] * q + self.ext_torque(q, payload)

    def equilibrium(self, u, k_c, t0, k_j, payload=None, q0=None, iters=60, tol=1e-9):
        """Static pose (N, 30) for motor positions u (N, 6); Newton with the active set of taut cables."""
        u = u.to(self.dtype)
        n = u.shape[0]
        q = torch.zeros(n, 30, dtype=self.dtype, device=self.device) if q0 is None else q0.to(self.dtype).clone()
        eye = torch.eye(30, dtype=self.dtype, device=self.device)
        K_e = self.ext_stiffness(q, payload)
        for _ in range(iters):
            e = q @ self.G.T - u @ self.S.T + t0 / k_c
            ka = k_c * (e > 0.0).to(self.dtype)                                     # taut cables
            r = -(ka * e) @ self.G - k_j[:, None] * q + self.ext_torque(q, payload)
            H = k_j[:, None, None] * eye + (self.G.T[None] * ka[:, None, :]) @ self.G[None]
            if K_e is not None:
                H = H - K_e
            step = torch.linalg.solve(H, r[:, :, None])[:, :, 0]
            q = q + step
            if step.abs().max() < tol:
                break
        return q

    def motors_for_pose(self, q_des, k_c, t0, k_j, payload=None, iters=4):
        """Motor positions (N, 6) that hold q_des (N, 30) best (least squares), clipped to the motor range."""
        q_des = q_des.to(self.dtype)
        tau_e = self.ext_torque(q_des, payload)
        b0 = t0 / k_c
        u = (q_des @ self.G.T) @ torch.linalg.pinv(self.S).T                        # rigid-cable guess
        for _ in range(iters):
            e = q_des @ self.G.T - u @ self.S.T + b0
            ka = k_c * (e > 0.0).to(self.dtype)
            GA = self.G.T[None] * ka[:, None, :]                                     # (N, 30, 12)
            A = GA @ self.S                                                          # (N, 30, 6)
            rhs = k_j[:, None] * q_des + (GA @ (q_des @ self.G.T + b0)[:, :, None])[:, :, 0] - tau_e
            u = torch.linalg.lstsq(A, rhs[:, :, None]).solution[:, :, 0]
        return torch.maximum(torch.minimum(u, self.u_max), -self.u_max)

    def stiffness(self, q, u, k_c, t0, k_j, payload=None):
        """H = -d residual / dq (N, 30, 30) at (q, u): the arm's static stiffness with the taut cables."""
        e = q.to(self.dtype) @ self.G.T - u.to(self.dtype) @ self.S.T + t0 / k_c
        ka = k_c * (e > 0.0).to(self.dtype)
        eye = torch.eye(30, dtype=self.dtype, device=self.device)
        H = k_j[:, None, None] * eye + (self.G.T[None] * ka[:, None, :]) @ self.G[None]
        K_e = self.ext_stiffness(q, payload)
        return H if K_e is None else H - K_e
