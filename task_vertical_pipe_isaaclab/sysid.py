"""
==============================================================================
sysid.py - system identification of the cable drive from quasi-static experiments (ND1)
==============================================================================
    python sysid.py data/sysid/robot_0.npz [--sensors encoders+mocap+loadcell] [--out reports/sysid]
    python sysid.py data/sysid/*.npz --study            # all sensor sets, table + uncertainty_identified.json

Data (one experiment = one robot, the schema the K22 should log once the P0 sensors of
CAI_TIEN_ROBOT_K22_CHO_DE_TAI.md are in place; sysid_experiments.py writes it from Isaac Sim):
    u        (T, 6)    motor-nut positions from the encoders, m (M1 M2 M5 M6 M3 M4)
    payload  (T,)      known mass hung on the tip, kg
    markers  (T, 3, 3) mocap: centres of the discs Seg5 / Seg10 / Seg15 in the robot frame, m
    tension  (T, 12)   load cells at the nuts (nut-side tension), N
    imu      (T, 3, 2) tilt of the discs Seg5 / Seg10 / Seg15 (x, y of the disc axis in the world: what an
                       accelerometer at rest gives; no yaw)
    true_*             (simulated data only) the parameters of the robot, for checking
The samples are quasi-static (slow motion, short hold before every sample).

Identification model (the model class of the twin, uncertain_cables.py, at rest): 30 bending joints in
static balance with the backbone springs, gravity, the payload and 12 one-sided cables with a capstan
friction element (stick / slip, active set as in home_hysteresis.py) and nut backlash, simulated along the
measured encoder trajectory. Parameters (25): cable stiffness per section (3), pre-tension per cable (12),
backbone stiffness (1), friction mu theta per section (3), backlash per motor (6); the compliance split phi
is fixed at 0.5 (a model error when the robot's is not). Output-error fit: Levenberg-Marquardt on the
sensor residuals weighted by the sensor noise, finite-difference Jacobian (all perturbed parameter sets
simulated as one batch), 95 % intervals from (J^T W J)^-1 scaled by the reduced chi^2.

Sensor sets of the study (what each upgrade of the robot buys for ND1):
    encoders+imu                today's K22 (3 MPU6050 tilts)
    encoders+mocap              + motion capture (P0.2)
    encoders+loadcell           + 12 load cells (P0.1)
    encoders+mocap+loadcell     both
numpy only.
==============================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # one thread per fit process (small matrices)

import numpy as np

import cables as CB
from uncertain_cables import PayloadModel

MARKER_BODIES = ("Seg5", "Seg10", "Seg15")
NOISE = {"markers": 0.5e-3, "tension": 0.10, "imu": np.deg2rad(0.5)}      # sensor noise (1 sigma)
SENSOR_SETS = ("encoders+imu", "encoders+mocap", "encoders+loadcell", "encoders+mocap+loadcell")
PHI = 0.5

# parameter vector layout
P_KC, P_T0, P_KJ, P_MU, P_B = slice(0, 3), slice(3, 15), slice(15, 16), slice(16, 19), slice(19, 25)
N_PARAM = 25
PARAM_NAMES = ([f"k_c_s{s + 1}" for s in range(3)] + [f"T0_{n}" for n in CB.CABLE_NAMES] + ["k_j"]
               + [f"mu_theta_s{s + 1}" for s in range(3)] + [f"backlash_{m}" for m in CB.MOTOR_NAMES])
LOWER = np.array([300.0] * 3 + [0.0] * 12 + [0.05] + [0.0] * 3 + [0.0] * 6)
UPPER = np.array([20000.0] * 3 + [20.0] * 12 + [5.0] + [1.0] * 3 + [2e-3] * 6)
STEP = np.array([30.0] * 3 + [0.02] * 12 + [0.005] + [0.005] * 3 + [1e-5] * 6)     # finite-difference steps


def nominal_theta():
    th = np.zeros(N_PARAM)
    th[P_KC] = CB.CABLE_STIFFNESS
    th[P_T0] = CB.CABLE_PRETENSION
    th[P_KJ] = CB.BACKBONE_STIFFNESS
    th[P_MU] = 0.05
    th[P_B] = 0.05e-3
    return th


def true_theta(exp: dict) -> np.ndarray:
    """The robot's per-cable parameters reduced to the model's 25 (geometric mean per section)."""
    th = np.zeros(N_PARAM)
    k_c, mu = np.asarray(exp["true_k_c"]), np.asarray(exp["true_mu_theta"])
    for s in range(3):
        sel = CB.CABLE_SECTION == s
        th[s] = np.exp(np.log(k_c[sel]).mean())
        th[16 + s] = mu[sel].mean()
    th[P_T0] = exp["true_t0"]
    th[P_KJ] = exp["true_k_j"]
    th[P_B] = exp["true_backlash"]
    return th


# ---------------------------------------------------------------------------
# Quasi-static model with capstan friction
# ---------------------------------------------------------------------------
class QuasiStaticModel:
    def __init__(self, geom: CB.CableGeometry | None = None):
        self.geom = geom or CB.CableGeometry()
        self.kin = self.geom.kin
        self.gravity = CB.GravityModel(self.geom)
        self.payload = PayloadModel(self.gravity)
        self.G = self.geom.G                                            # (12, 30)
        S = np.zeros((CB.N_CABLES, CB.N_MOTORS))
        S[np.arange(CB.N_CABLES), CB.CABLE_MOTOR] = CB.CABLE_SIGN
        self.S = S
        self.marker_ids = [self.kin.body_index[b] for b in MARKER_BODIES]

    def _ext(self, q, payload):
        tau = self.gravity.torque(q)
        if np.any(payload > 0):
            tau = tau + self.payload.torque(q, payload)
        return tau

    def _ext_stiffness(self, q, payload, eps=1e-6):
        n = q.shape[0]
        pts = np.repeat(q[:, None], 60, axis=1)
        idx = np.arange(30)
        pts[:, 2 * idx, idx] += eps
        pts[:, 2 * idx + 1, idx] -= eps
        tau = self._ext(pts.reshape(-1, 30), np.repeat(payload, 60)).reshape(n, 60, 30)
        return ((tau[:, 0::2] - tau[:, 1::2]) / (2 * eps)).transpose(0, 2, 1)

    def _solve(self, q, keff, z, k_j, payload, K_e, iters=25, tol=1e-10):
        """Equilibrium with the per-cable laws T = keff max(0, G q - z) (Newton, gravity stiffness K_e fixed)."""
        G = self.G
        eye = np.eye(30)
        for _ in range(iters):
            e = q @ G.T - z
            ka = keff * (e > 0.0)
            r = -(ka * e) @ G - k_j[:, None] * q + self._ext(q, payload)
            H = k_j[:, None, None] * eye + np.einsum("ci,nc,cj->nij", G, ka, G) - K_e
            dq = np.linalg.solve(H, r[..., None])[..., 0]
            q = q + dq
            if np.abs(dq).max() < tol:
                break
        return q

    def simulate(self, theta, u, payload):
        """
        theta (B, 25), u (T, 6), payload (T,) -> q (B, T, 30), T_r, T_n (B, T, 12).
        Starts straight at rest at u[0] with the friction elements relaxed.
        """
        theta = np.atleast_2d(theta)
        B, T = theta.shape[0], u.shape[0]
        sec = CB.CABLE_SECTION
        k_c = theta[:, P_KC][:, sec]
        t0 = theta[:, P_T0]
        k_j = theta[:, 15]
        lam_hi = np.exp(theta[:, P_MU][:, sec])
        lam_lo = 1.0 / lam_hi
        half = 0.5 * theta[:, P_B]
        k_n, k_r = k_c / PHI, k_c / (1.0 - PHI)
        kappa = lambda lam: (k_n + k_r) / (k_n + lam * k_r)  # noqa: E731
        G, S = self.G, self.S

        xn = np.repeat(u[:1], B, axis=0)
        q = np.zeros((B, 30))
        pl = np.full(B, payload[0])
        q = self._solve(q, k_c, xn @ S.T - t0 / k_c, k_j, pl, self._ext_stiffness(q, pl))
        g, s = q @ G.T, xn @ S.T
        p = (k_r * g + k_n * s) / (k_n + k_r)
        Q, TR, TN = np.zeros((B, T, 30)), np.zeros((B, T, 12)), np.zeros((B, T, 12))
        for k in range(T):
            xn = np.clip(xn, u[k] - half, u[k] + half)
            s = xn @ S.T
            pl = np.full(B, payload[k])
            K_e = self._ext_stiffness(q, pl)
            mode = np.zeros((B, 12), dtype=np.int8)
            p0 = p
            for _ in range(10):
                lam = np.where(mode > 0, lam_hi, np.where(mode < 0, lam_lo, 1.0))
                stick = mode == 0
                keff = np.where(stick, k_r, kappa(lam) * k_c)
                z = np.where(stick, p0 - t0 / k_r, s - t0 / k_c)
                q = self._solve(q, keff, z, k_j, pl, K_e)
                g = q @ G.T
                taut = g - s + t0 / k_c > 0.0
                p_slide = np.where(taut, (lam * (t0 + k_r * g) - t0 + k_n * s) / (k_n + lam * k_r),
                                   np.clip(p0, g + t0 / k_r, s - t0 / k_n))
                p_new = np.where(stick, p0, p_slide)
                T_r = keff * np.maximum(g - z, 0.0)
                T_n = k_n * np.maximum(p_new - s + t0 / k_n, 0.0)
                tol = 1e-9 * (1.0 + T_r + T_n)
                new = mode.copy()
                new[stick & (T_n > lam_hi * T_r + tol)] = 1
                new[stick & (T_n < lam_lo * T_r - tol)] = -1
                new[(mode == 1) & (p_new > p0 + 1e-14)] = 0
                new[(mode == -1) & (p_new < p0 - 1e-14)] = 0
                if np.array_equal(new, mode):
                    break
                mode = new
            p = p_new
            Q[:, k], TR[:, k], TN[:, k] = q, T_r, np.where(stick, T_n, lam * T_r)
        return Q, TR, TN

    def outputs(self, Q):
        """Joint angles (..., 30) -> markers (..., 3, 3) and disc tilts (..., 3, 2)."""
        qpos = np.zeros(Q.shape[:-1] + (self.kin.nq,))
        qpos[..., self.geom.joint_qadr] = Q
        xpos, xmat = self.kin.forward(qpos.reshape(-1, self.kin.nq))
        mk = xpos[:, self.marker_ids]
        base = xpos[:, [self.kin.body_index["Seg1"]]]
        tilt = xmat[:, self.marker_ids][:, :, :2, 2]                           # x, y of the disc axis
        shp = Q.shape[:-1]
        return (mk - base).reshape(shp + (3, 3)), tilt.reshape(shp + (3, 2))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def load_experiment(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def measurement_vector(exp, sensors, rng):
    """Noisy measurements of the chosen sensors, flattened, and their noise std."""
    ys, sig = [], []
    if "mocap" in sensors:
        ys.append(exp["markers"].ravel() + rng.normal(0, NOISE["markers"], exp["markers"].size))
        sig.append(np.full(exp["markers"].size, NOISE["markers"]))
    if "loadcell" in sensors:
        ys.append(exp["tension"].ravel() + rng.normal(0, NOISE["tension"], exp["tension"].size))
        sig.append(np.full(exp["tension"].size, NOISE["tension"]))
    if "imu" in sensors:
        ys.append(exp["imu"].ravel() + rng.normal(0, NOISE["imu"], exp["imu"].size))
        sig.append(np.full(exp["imu"].size, NOISE["imu"]))
    return np.concatenate(ys), np.concatenate(sig)


def predict(model, theta, exp, sensors):
    Q, _, TN = model.simulate(theta, exp["u"], exp["payload"])
    mk, tilt = model.outputs(Q)
    B = np.atleast_2d(theta).shape[0]
    parts = []
    if "mocap" in sensors:
        parts.append(mk.reshape(B, -1))
    if "loadcell" in sensors:
        parts.append(TN.reshape(B, -1))
    if "imu" in sensors:
        parts.append(tilt.reshape(B, -1))
    return np.concatenate(parts, axis=1)


def fit(model, exp, sensors, iters=25, seed=0, theta0=None, verbose=True):
    rng = np.random.default_rng(seed)
    y, sig = measurement_vector(exp, sensors, rng)
    theta = nominal_theta() if theta0 is None else theta0.copy()
    mu_lm = 1e-2
    t0 = time.time()

    def jac_and_res(th):
        pts = np.repeat(th[None], N_PARAM + 1, axis=0)
        h = STEP.copy()
        up = th + h > UPPER
        h[up] = -h[up]                                     # one-sided steps at the upper bound
        pts[1 + np.arange(N_PARAM), np.arange(N_PARAM)] += h
        pred = predict(model, pts, exp, sensors)
        r = (pred[0] - y) / sig
        J = ((pred[1:] - pred[0]) / h[:, None]).T / sig[:, None]
        return r, J

    r, J = jac_and_res(theta)
    cost = float(r @ r)
    for it in range(iters):
        JTJ = J.T @ J
        g = J.T @ r
        cost_before = cost
        improved = False
        for _ in range(8):
            A = JTJ + mu_lm * np.diag(np.diag(JTJ) + 1e-9)
            step = -np.linalg.lstsq(A, g, rcond=None)[0]
            cand = np.clip(theta + step, LOWER, UPPER)
            pred = predict(model, cand, exp, sensors)[0]
            r_c = (pred - y) / sig
            c_c = float(r_c @ r_c)
            if c_c < cost:
                theta, cost, improved = cand, c_c, True
                mu_lm = max(mu_lm / 3.0, 1e-7)
                break
            mu_lm *= 4.0
        if verbose:
            print(f"    iter {it + 1:2d}: chi2/N {cost / len(y):.3f}  (lm {mu_lm:.1e}, {time.time() - t0:.0f} s)", flush=True)
        if not improved or mu_lm > 1e6 or cost_before - cost < 1e-6 * cost_before:
            break
        r, J = jac_and_res(theta)
    # covariance (reduced chi^2 scaling) -> 95 % intervals
    dof = max(len(y) - N_PARAM, 1)
    s2 = max(cost / dof, 1.0)
    cov = np.linalg.pinv(J.T @ J) * s2
    ci = 1.96 * np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"theta": theta, "ci95": ci, "chi2_per_n": cost / len(y), "n": len(y), "sensors": sensors,
            "cond": float(np.linalg.cond(J.T @ J)), "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
GROUPS = {"k_c": P_KC, "T0": P_T0, "k_j": P_KJ, "mu_theta": P_MU, "backlash": P_B}


def group_errors(theta, truth, ci):
    """Per parameter group: median relative error, median relative 95 % half-width (backlash, mu: absolute)."""
    out = {}
    for name, sl in GROUPS.items():
        est, tru, c = theta[sl], truth[sl], ci[sl]
        if name in ("mu_theta", "backlash"):
            scale = 1e3 if name == "backlash" else 1.0
            out[name] = {"abs_err": float(np.median(np.abs(est - tru)) * scale),
                         "ci_half": float(np.median(c) * scale), "unit": "mm" if name == "backlash" else "-"}
        else:
            out[name] = {"rel_err": float(np.median(np.abs(est - tru) / np.abs(tru))),
                         "rel_ci_half": float(np.median(c / np.abs(tru)))}
    return out


def identified_ranges(fits: list) -> dict:
    """Uncertainty set for the RL training from the fits with every sensor: estimate +- 2 x the 95 % half-width."""
    k_c, t0, k_j, mu, bl = [], [], [], [], []
    for f in fits:
        th, ci = f["theta"], f["ci95"]
        k_c += [th[P_KC] - 2 * ci[P_KC], th[P_KC] + 2 * ci[P_KC]]
        t0 += [th[P_T0] - 2 * ci[P_T0], th[P_T0] + 2 * ci[P_T0]]
        k_j += [th[P_KJ] - 2 * ci[P_KJ], th[P_KJ] + 2 * ci[P_KJ]]
        mu += [th[P_MU] - 2 * ci[P_MU], th[P_MU] + 2 * ci[P_MU]]
        bl += [th[P_B] - 2 * ci[P_B], th[P_B] + 2 * ci[P_B]]
    rng = lambda xs, lo=0.0: [float(max(lo, np.min(np.concatenate(xs)))), float(np.max(np.concatenate(xs)))]  # noqa: E731
    return {"k_c": rng(k_c, 300.0), "t0": rng(t0, 0.2), "k_j": rng(k_j, 0.05), "mu_theta": rng(mu), "backlash": rng(bl)}


_MODEL = None


def fit_file(path, sensors, iters, seed):
    """One (experiment, sensor set) fit; runs in a worker process."""
    global _MODEL
    if _MODEL is None:
        _MODEL = QuasiStaticModel()
    exp = load_experiment(path)
    f = fit(_MODEL, exp, sensors, iters=iters, seed=seed, verbose=False)
    rec = {"file": os.path.basename(path), "sensors": sensors, "theta": f["theta"].tolist(), "ci95": f["ci95"].tolist(),
           "chi2_per_n": f["chi2_per_n"], "cond": f["cond"], "seconds": f["seconds"], "n": f["n"]}
    if "true_k_c" in exp:
        truth = true_theta(exp)
        rec["truth"] = truth.tolist()
        rec["errors"] = group_errors(f["theta"], truth, f["ci95"])
    return rec


def _fmt(rec):
    if "errors" not in rec:
        return f"chi2/N {rec['chi2_per_n']:.2f}"
    return f"chi2/N {rec['chi2_per_n']:.2f} | " + "  ".join(
        f"{k}: " + (f"err {100 * v['rel_err']:.1f}% (+-{100 * v['rel_ci_half']:.1f}%)" if "rel_err" in v
                    else f"err {v['abs_err']:.3f} (+-{v['ci_half']:.3f}){v['unit']}") for k, v in rec["errors"].items())


def main():
    ap = argparse.ArgumentParser(description="System identification of the cable drive (quasi-static experiments)")
    ap.add_argument("data", nargs="+", help="experiment .npz files (sysid_experiments.py)")
    ap.add_argument("--sensors", type=str, default="encoders+mocap+loadcell")
    ap.add_argument("--study", action="store_true", help="every sensor set of SENSOR_SETS")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "sysid"))
    a = ap.parse_args()
    files = sorted(f for pat in a.data for f in glob.glob(pat))
    os.makedirs(a.out, exist_ok=True)
    sets = SENSOR_SETS if a.study else (a.sensors,)
    jobs = [(path, s) for path in files for s in sets]
    print(f"{len(files)} experiments x {len(sets)} sensor sets = {len(jobs)} fits on {a.workers} processes", flush=True)
    results = {s: [] for s in sets}
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(fit_file, path, s, a.iters, k): (path, s) for k, (path, s) in enumerate(jobs)}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["sensors"]].append(rec)
            print(f"  [{time.time() - t0:5.0f} s] {rec['file']} / {rec['sensors']}: {_fmt(rec)}", flush=True)
            with open(os.path.join(a.out, "sysid_results.json"), "w", encoding="utf-8") as fh:
                json.dump({"param_names": PARAM_NAMES, "noise": {k: float(v) for k, v in NOISE.items()},
                           "results": results}, fh, indent=1)
    # summary over robots
    print("\nmedian over robots (relative error | relative 95 % half-width; mu theta, backlash absolute):")
    for s in sets:
        errs = [r["errors"] for r in results[s] if "errors" in r]
        if not errs:
            continue
        parts = []
        for g in GROUPS:
            if "rel_err" in errs[0][g]:
                parts.append(f"{g} {100 * np.median([e[g]['rel_err'] for e in errs]):5.1f}% | "
                             f"{100 * np.median([e[g]['rel_ci_half'] for e in errs]):6.1f}%")
            else:
                parts.append(f"{g} {np.median([e[g]['abs_err'] for e in errs]):.3f} | "
                             f"{np.median([e[g]['ci_half'] for e in errs]):.3f}{errs[0][g]['unit']}")
        print(f"  {s:<26} " + "   ".join(parts))
    full = "encoders+mocap+loadcell"
    if full in results and results[full]:
        fits = [{"theta": np.array(r["theta"]), "ci95": np.array(r["ci95"])} for r in results[full]]
        rngs = identified_ranges(fits)
        path = os.path.join(a.out, "uncertainty_identified.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"source": f"sysid.py, {len(fits)} robots, sensors {full}", "ranges": rngs}, fh, indent=1)
        print(f"\nidentified uncertainty set -> {path}: {rngs}")


if __name__ == "__main__":
    main()
