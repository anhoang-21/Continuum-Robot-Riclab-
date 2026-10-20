"""
==============================================================================
safety_filter.py - robust control-barrier-function filter on the 12 cable tensions (ND4)
==============================================================================
Sits between any controller (PPO, PPO-Lagrangian, RMA, image student, scripted bend motions of the
vision / language pipelines) and the 6 lead-screw motors, every 20 ms:

    a_safe = argmin_a  1/2 |a - a_ref|^2 + rho/2 |s|^2
             s.t.  A_s a + s_lo >= -gamma (T_lo - T_min) + m          (cable c stays taut,  12 rows)
                  -A_s a + s_hi >= -gamma (T_max - T_hi) + m           (cable c not overloaded, 12 rows)
                   lo_a <= a <= hi_a                                   (motor speed and travel, hard)
                   s >= 0

with a = du / u_rate the normalised motor step, A_s = dT/du diag(u_rate) the quasi-static tension
sensitivity of the cable drive, and h = T - T_min, T_max - T the barrier functions of the tension band.
Discrete-time CBF condition: h(k+1) >= (1 - gamma) h(k), i.e. the band is forward invariant (Agrawal &
Sreenath, RSS 2017): if the robot is inside the band it stays inside, if it is outside it is pushed back
at rate gamma. The slack s (penalty rho) keeps the QP feasible when no motor step can satisfy every row
(e.g. the S-curve needs more differential tension than the pre-tension allows); filter.last_slack says so.

Tension sensitivity (frictionless quasi-static model with the nominal / identified parameters):
    T_virt = k_c (G q - S u + T0 / k_c)                       virtual tension (negative = slack depth)
    dq/du  = H^-1 G^T K_a S,   H = k_j I + G^T K_a G           K_a = diag(k_c [taut])  (arm compliance)
    A      = diag(k_c) (G dq/du - S)
             G H^-1 G^T = M (k_j I + K_a M)^-1  with M = G G^T (push-through identity: 12 x 12 solves)
The pose q is what a shape sensor gives: the constant-curvature joints of the measured section bends
(exact for G q, the cables cross every joint of a section with the same lever arm).

Tension estimate (`source`):
    "loadcell": T_lo = T_n e^(-mu_bar) - delta, T_hi = T_n e^(+mu_bar) + delta for taut cables (the capstan
                friction makes the robot side differ from the nut side by up to e^(mu_bar)); a cable the
                load cell reads slack gets min(T_virt, 0) - delta (how far it is from taut)
    "model":    T_virt +- (delta + rel_k |T_virt - T0|)       (no load cells: identified-model error)
    "oracle":   the simulator's true robot / nut side tensions (an upper bound of what a filter can do)
Robust margin m_c = rho_A sum_m |A_cm| u_rate_m (relative error rho_A of the sensitivity, e.g. the
identification's k_c interval) - the tightening of a robust CBF for bounded multiplicative model error.

QP solver: dual projected gradient with FISTA momentum on the 24 general rows (the box is separable, so
a*(lambda) = clip(a_ref + C^T lambda, lo_a, hi_a)), batched over the envs on the GPU, warm-started from
the last step's multipliers. tests/check_safety_filter.py compares it with OSQP.
==============================================================================
"""

from __future__ import annotations

import math

import numpy as np
import torch

import cables as CB


class TensionCBF:
    def __init__(self, env, source: str = "loadcell", gamma: float = 0.2, t_min: float | None = None,
                 t_max: float | None = None, mu_bar: float = 0.3, delta: float = 0.3, rel_k: float = 0.3,
                 rho_A: float = 0.25, rho: float = 1e3, iters: int = 150, k_c: float = CB.CABLE_STIFFNESS,
                 t0: float | None = None, k_j: float = CB.BACKBONE_STIFFNESS):
        if source not in ("loadcell", "model", "oracle"):
            raise ValueError(f"source must be loadcell / model / oracle, got {source!r}")
        self.env, self.source = env, source
        dev, f64 = env.device, torch.float64
        self.device = dev
        self.gamma, self.mu_bar, self.delta, self.rel_k, self.rho_A, self.rho, self.iters = (
            gamma, mu_bar, delta, rel_k, rho_A, rho, iters)
        self.t_min = env.cfg.t_min if t_min is None else t_min
        self.t_max = env.cfg.t_max if t_max is None else t_max
        self.k_c = float(k_c)
        self.t0 = float(env.cfg.cable_pretension if t0 is None else t0)
        self.k_j = float(k_j)
        G = torch.as_tensor(env.geom.G, dtype=f64, device=dev)
        S = torch.zeros(CB.N_CABLES, CB.N_MOTORS, dtype=f64, device=dev)
        S[torch.arange(CB.N_CABLES), torch.as_tensor(CB.CABLE_MOTOR)] = torch.as_tensor(CB.CABLE_SIGN, dtype=f64, device=dev)
        self.G, self.S = G, S
        # arm without cables: H0 = k_j I - K_g (gravity stiffness of the hanging arm at the straight pose: the
        # sensitivity error drops from 17 % to 3 % (95th percentile) against the exact pose-dependent one)
        gm = getattr(env, "gravity_model", None)
        K_g = (torch.as_tensor(gm.stiffness(np.zeros((1, 30)))[0], dtype=f64, device=dev) if gm is not None
               else torch.zeros(30, 30, dtype=f64, device=dev))
        H0 = self.k_j * torch.eye(30, dtype=f64, device=dev) - K_g
        self.P = G @ torch.linalg.solve(H0, G.T)                                 # G H0^-1 G^T (12 x 12)
        self.u_rate = torch.as_tensor(CB.MOTOR_RATE, dtype=f64, device=dev)
        self.u_max = torch.as_tensor(CB.MOTOR_RANGE, dtype=f64, device=dev)
        n = env.num_envs
        self.lam = torch.zeros(n, 2 * CB.N_CABLES, dtype=f64, device=dev)       # warm start
        self.last_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_slack = torch.zeros(n, dtype=f64, device=dev)
        self.last_change = torch.zeros(n, dtype=f64, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.lam.zero_()
        else:
            self.lam[torch.as_tensor(list(env_ids), device=self.device)] = 0.0

    # ------------------------------------------------------------------
    def pose(self):
        """Constant-curvature joint angles (N, 30) of the measured section bends (what a shape sensor gives)."""
        b = self.env.measured_bend().to(torch.float64) / 5.0                     # (N, 3, 2)
        q = torch.empty(b.shape[0], 3, 5, 2, dtype=torch.float64, device=self.device)
        q[..., 0] = -b[:, :, None, 1]
        q[..., 1] = b[:, :, None, 0]
        return q.reshape(-1, 30)

    def sensitivity(self, q, u):
        """Virtual tensions (N, 12) and dT/du (N, 12, 6) of the nominal frictionless model."""
        e = q @ self.G.T - u @ self.S.T + self.t0 / self.k_c
        ka = self.k_c * (e > 0.0).to(torch.float64)                             # (N, 12)
        eye = torch.eye(CB.N_CABLES, dtype=torch.float64, device=self.device)
        # G dq/du = G (H0 + G^T K_a G)^-1 G^T K_a S = P (I + K_a P)^-1 K_a S   (push-through identity)
        X = torch.linalg.solve(eye[None] + ka[:, :, None] * self.P[None], ka[:, :, None] * self.S[None])
        A = self.k_c * (self.P[None] @ X - self.S[None])
        return self.k_c * e, A

    def tension_bounds(self, T_virt):
        env = self.env
        if self.source == "oracle":
            return env.tension.clone(), torch.maximum(env.tension, env.ucm.T_n)
        if self.source == "model":
            err = self.delta + self.rel_k * (T_virt - self.t0).abs()
            return T_virt - err, T_virt + err
        T = env.ucm.T_n + env.tension_noise[:, None] * torch.randn_like(env.ucm.T_n)   # load cells
        taut = T > 0.2
        lo = torch.where(taut, T * math.exp(-self.mu_bar), torch.clamp(T_virt, max=0.0)) - self.delta
        hi = torch.where(taut, T * math.exp(self.mu_bar), torch.clamp(T_virt, min=0.0)) + self.delta
        return lo, hi

    # ------------------------------------------------------------------
    def build(self, u, du_ref):
        """QP data for the current state: C a >= d (N, 24, 6 / N, 24), box, a_ref."""
        q = self.pose()
        T_virt, A = self.sensitivity(q, u)
        lo, hi = self.tension_bounds(T_virt)
        As = A * self.u_rate[None, None, :]
        m = self.rho_A * As.abs().sum(-1)
        C = torch.cat([As, -As], dim=1)
        d = torch.cat([-self.gamma * (lo - self.t_min) + m, -self.gamma * (self.t_max - hi) + m], dim=1)
        a_ref = du_ref / self.u_rate
        a_box = torch.maximum(torch.ones_like(a_ref), a_ref.abs())
        lo_a = torch.maximum(-a_box, (-self.u_max - u) / self.u_rate)
        hi_a = torch.minimum(a_box, (self.u_max - u) / self.u_rate)
        return C, d, a_ref, lo_a, hi_a

    def solve(self, C, d, a_ref, lo_a, hi_a, lam0=None):
        """min 1/2 |a - a_ref|^2 + rho/2 |s|^2  s.t. C a + s >= d, s >= 0, lo_a <= a <= hi_a (batched)."""
        lam = torch.zeros_like(d) if lam0 is None else lam0.clone()
        Ct = C.transpose(1, 2)
        L = torch.linalg.eigvalsh(Ct @ C)[:, -1] + 1.0 / self.rho                # Lipschitz constant (6 x 6)
        eta = (1.0 / L)[:, None]
        y, lam_prev, t = lam.clone(), lam.clone(), 1.0
        for _ in range(self.iters):
            a = torch.minimum(torch.maximum(a_ref + (Ct @ y[:, :, None])[:, :, 0], lo_a), hi_a)
            grad = d - (C @ a[:, :, None])[:, :, 0] - y / self.rho
            lam_new = torch.clamp(y + eta * grad, min=0.0)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = lam_new + ((t - 1.0) / t_new) * (lam_new - lam_prev)
            lam_prev, t = lam_new, t_new
        lam = lam_prev
        a = torch.minimum(torch.maximum(a_ref + (Ct @ lam[:, :, None])[:, :, 0], lo_a), hi_a)
        return a, lam

    @torch.no_grad()
    def __call__(self, du_ref: torch.Tensor) -> torch.Tensor:
        """Filtered motor steps (N, 6) for the reference steps du_ref (N, 6) from the current motor positions."""
        u = self.env.motor.to(torch.float64)
        C, d, a_ref, lo_a, hi_a = self.build(u, du_ref.to(torch.float64))
        a, self.lam = self.solve(C, d, a_ref, lo_a, hi_a, self.lam)
        self.last_slack = (self.lam / self.rho).max(dim=1).values                # N of constraint violation accepted
        self.last_change = (a - a_ref).abs().max(dim=1).values
        self.last_active = self.last_change > 1e-3
        return a * self.u_rate


class FilteredController:
    """A controller whose motor steps go through the env's TensionCBF (the filter itself sits in the env)."""

    def __init__(self, ctrl, cbf: TensionCBF):
        self.ctrl, self.cbf = ctrl, cbf
        self.name = getattr(ctrl, "name", "controller") + "+cbf"
        cbf.env.safety_filter = cbf

    @property
    def last_active(self):
        return self.cbf.last_active

    def reset(self, env_ids=None):
        self.ctrl.reset(env_ids)
        self.cbf.reset(env_ids)

    def __call__(self, obs, env=None):
        if env is not None:
            env.safety_filter = self.cbf                                          # (the same env runs other cells)
        return self.ctrl(obs, env)
