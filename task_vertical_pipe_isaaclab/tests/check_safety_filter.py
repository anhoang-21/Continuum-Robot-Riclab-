"""
==============================================================================
tests/check_safety_filter.py - checks of the tension CBF filter (safety_filter.py)
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe tests\\check_safety_filter.py          (no Isaac Sim needed)

1. QP solver (batched dual FISTA) vs OSQP on random filter problems of the robot: same solution
2. tension sensitivity dT/du of the filter vs finite differences of uncertain_cables.StaticSolver
   (with gravity, which the filter leaves out) at random poses
3. no intervention when the reference step keeps every barrier condition; the rows of the solution hold
   (or the reported slack covers the difference)
The closed-loop effect (violations with / without the filter on uncertain robots) is measured by
benchmark_safety.py ("+cbf" controllers).
==============================================================================
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

import cables as CB  # noqa: E402
from safety_filter import TensionCBF  # noqa: E402
from uncertain_cables import StaticSolver  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def fake_env(n, geom):
    cfg = SimpleNamespace(t_min=0.5, t_max=10.0, cable_pretension=CB.CABLE_PRETENSION)
    return SimpleNamespace(device="cpu", num_envs=n, cfg=cfg, geom=geom, gravity_model=CB.GravityModel(geom))


def random_states(n, rng, geom, scale=0.7):
    """Random motor positions and the static poses of the nominal robot (some cables slack at scale 0.7)."""
    gm = CB.GravityModel(geom)
    ss = StaticSolver(geom, gm)
    lim = scale * CB.MOTOR_RANGE / (CB.MOTOR_SECTION + 1)
    u = torch.as_tensor(rng.uniform(-lim, lim, (n, 6)))
    kc = torch.full((n, 12), CB.CABLE_STIFFNESS, dtype=torch.float64)
    t0 = torch.full((n, 12), CB.CABLE_PRETENSION, dtype=torch.float64)
    kj = torch.full((n,), CB.BACKBONE_STIFFNESS, dtype=torch.float64)
    q = ss.equilibrium(u, kc, t0, kj)
    return ss, u, q, kc, t0, kj


def check_solver(cbf, rng, n=64):
    print("\n1. QP solver vs OSQP")
    import osqp
    import scipy.sparse as sp

    geom = cbf.env.geom
    _, u, q, *_ = random_states(n, rng, geom)
    T_virt, A = cbf.sensitivity(q, u)
    lo, hi = T_virt - 0.3, T_virt + 0.3
    As = A * cbf.u_rate[None, None, :]
    m = cbf.rho_A * As.abs().sum(-1)
    C = torch.cat([As, -As], 1)
    d = torch.cat([-cbf.gamma * (lo - cbf.t_min) + m, -cbf.gamma * (cbf.t_max - hi) + m], 1)
    a_ref = torch.as_tensor(rng.uniform(-1, 1, (n, 6)))
    lo_a, hi_a = -torch.ones(n, 6, dtype=torch.float64), torch.ones(n, 6, dtype=torch.float64)
    cbf.iters = 3000
    a, lam = cbf.solve(C, d, a_ref, lo_a, hi_a)
    worst, worst_obj = 0.0, 0.0
    for i in range(n):
        # variables [a (6), s (24)]: min 1/2 |a - a_ref|^2 + rho/2 |s|^2, C a + s >= d, s >= 0, box
        P = sp.block_diag([sp.eye(6), cbf.rho * sp.eye(24)], format="csc")
        qv = np.concatenate([-a_ref[i].numpy(), np.zeros(24)])
        Ab = sp.vstack([sp.hstack([sp.csc_matrix(C[i].numpy()), sp.eye(24)]),
                        sp.hstack([sp.csc_matrix((24, 6)), sp.eye(24)]),
                        sp.hstack([sp.eye(6), sp.csc_matrix((6, 24))])], format="csc")
        l_ = np.concatenate([d[i].numpy(), np.zeros(24), lo_a[i].numpy()])
        u_ = np.concatenate([np.full(24, np.inf), np.full(24, np.inf), hi_a[i].numpy()])
        prob = osqp.OSQP()
        prob.setup(P, qv, Ab, l_, u_, verbose=False, eps_abs=1e-10, eps_rel=1e-10, max_iter=200000, polishing=True)
        x = prob.solve().x[:6]
        worst = max(worst, float(np.abs(x - a[i].numpy()).max()))
        f = lambda z: 0.5 * np.sum((z - a_ref[i].numpy()) ** 2)  # noqa: E731
        worst_obj = max(worst_obj, abs(f(x) - f(a[i].numpy())))
    check("same solution as OSQP (64 problems)", worst < 1e-3, f"(max |a - a_osqp| {worst:.1e}, objective {worst_obj:.1e})")
    cbf.iters = 150
    a150, _ = cbf.solve(C, d, a_ref, lo_a, hi_a)
    err150 = (a150 - a).abs().max().item()
    dT = (C[:, :12] @ (a150 - a)[:, :, None]).abs().max().item()       # tension change of the difference (N)
    print(f"  with the 150 iterations of a control step (cold start; warm-started in closed loop): "
          f"max |a - a*| {err150:.1e} of a motor step")
    check("150 iterations: tension within 0.1 N of the exact filter step", dT < 0.1, f"(max {dT:.3f} N)")


def check_sensitivity(cbf, rng, n=16):
    print("\n2. tension sensitivity dT/du vs finite differences of the static model (with gravity)")
    ss, u, q, kc, t0, kj = random_states(n, rng, cbf.env.geom)
    _, A = cbf.sensitivity(q, u)
    h = 2e-5
    errs = []
    for m in range(6):
        du = torch.zeros_like(u)
        du[:, m] = h
        qp = ss.equilibrium(u + du, kc, t0, kj, q0=q)
        qm = ss.equilibrium(u - du, kc, t0, kj, q0=q)
        dT = (ss.tension(qp, u + du, kc, t0) - ss.tension(qm, u - du, kc, t0)) / (2 * h)
        errs.append((A[:, :, m] - dT).abs() / (dT.abs().max(dim=1, keepdim=True).values + 1e-9))
    err = torch.stack(errs, -1)
    T_virt, _ = cbf.sensitivity(q, u)
    taut = (T_virt > 0.0)[:, :, None].expand_as(err)                     # slack cables: virtual sensitivity
    err = err[taut]
    print(f"  relative error of dT/du (taut cables): median {err.median().item():.3f}, "
          f"95 % {torch.quantile(err, 0.95).item():.3f}, max {err.max().item():.3f}")
    check("sensitivity within the robust margin (rho_A = 0.25) on 95 % of the entries",
          torch.quantile(err, 0.95).item() < cbf.rho_A)


def check_no_intervention(cbf, rng, n=64):
    print("\n3. a safe reference passes unchanged; the solution satisfies the rows")
    _, u, q, *_ = random_states(n, rng, cbf.env.geom, scale=0.25)          # small bends: every cable taut
    T_virt, A = cbf.sensitivity(q, u)
    lo, hi = T_virt - 0.3, T_virt + 0.3
    As = A * cbf.u_rate[None, None, :]
    m = cbf.rho_A * As.abs().sum(-1)
    C = torch.cat([As, -As], 1)
    d = torch.cat([-cbf.gamma * (lo - cbf.t_min) + m, -cbf.gamma * (cbf.t_max - hi) + m], 1)
    box = torch.ones(n, 6, dtype=torch.float64)
    a_ref = torch.as_tensor(rng.uniform(-0.05, 0.05, (n, 6)))
    cbf.iters = 2000
    ok = ((C @ a_ref[:, :, None])[:, :, 0] >= d).all(1)
    a, lam = cbf.solve(C, d, a_ref, -box, box)
    ch = (a - a_ref).abs().max(1).values
    print(f"  {int(ok.sum())} / {n} states where standing still satisfies every row")
    check("no change of a safe reference", bool((ch[ok] < 1e-6).all()) if ok.any() else True,
          f"(max change {ch[ok].max().item() if ok.any() else 0:.1e})")
    viol = torch.clamp(d - (C @ a[:, :, None])[:, :, 0], min=0.0)
    slack = lam / cbf.rho
    check("rows hold up to the reported slack", bool((viol <= slack + 1e-3).all()),
          f"(max violation {viol.max().item():.2e} N, max slack {slack.max().item():.2e} N)")


def main():
    torch.set_printoptions(precision=3)
    rng = np.random.default_rng(0)
    geom = CB.CableGeometry()
    cbf = TensionCBF(fake_env(64, geom), source="model")
    check_solver(cbf, rng)
    check_sensitivity(cbf, rng)
    check_no_intervention(cbf, rng)
    print("\nsummary:", "all checks passed" if not FAILS else f"FAILED: {FAILS}")


if __name__ == "__main__":
    main()
