"""
==============================================================================
check_cable_model.py - Geometry of the cable routing (numpy only, no Isaac Sim)
==============================================================================
    python tests/check_cable_model.py

1. rest lengths: 5 / 10 / 15 pieces of 36-38 mm, section 2 / 3 cables run through the earlier sections
2. the exact cable shortening (forward kinematics of the MJCF chain) against the first-order model
   of constants.K_to_Length_3Seg / cable_model.bend_to_motor_matrix, for small and maximum bends
3. an antagonistic pair is (almost) symmetric: L+ - L0 = -(L- - L0)
4. bend -> motor -> bend round trip
==============================================================================
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cable_model as CM  # noqa: E402
import constants as C  # noqa: E402
from kinematics import ContinuumKinematics  # noqa: E402


def random_bends(rng, n, theta_max):
    theta = rng.uniform(0.0, theta_max, (n, 3))
    phi = rng.uniform(0.0, 2 * np.pi, (n, 3))
    return np.stack([theta * np.cos(phi), theta * np.sin(phi)], axis=-1)      # (n, 3, 2)


def main():
    kin = ContinuumKinematics()
    routing = CM.CableRouting(kin)
    M = CM.bend_to_motor_matrix()
    ok = True

    # 1. rest geometry
    pieces = routing.rest_len
    print("1. rest cable lengths (mm), cable 0 / 4 / 8 (section 1 / 3 / 2 style):",
          np.round(pieces[[0, 4, 8]] * 1000, 1))
    expect = np.array([0.038 * 5, 0.038 * 5 + 0.036 * 5, 0.038 * 5 + 0.036 * 5 + 0.036 * 5])   # rough (spacing 36-38 mm)
    n_pieces = routing.mask.sum(0)
    assert list(n_pieces[::2]) == [5, 5, 10, 10, 15, 15], n_pieces
    for c in range(12):
        sec = CM.MOTOR_SECTION[routing.cable_motor[c]]
        assert abs(routing.rest_len[c] - expect[sec]) < 0.012, (c, routing.rest_len[c], expect[sec])
    print("   OK: pieces per cable", n_pieces.astype(int).tolist())

    # 2. exact shortening vs first order model
    rng = np.random.default_rng(0)
    print("2. cable shortening: exact FK geometry vs M @ bend (first order = K_to_Length_3Seg)")
    for theta_max in (0.1, 0.4, C.THETA_MAX):
        bend = random_bends(rng, 200, theta_max)
        qpos = kin.qpos_from(bend, np.zeros(len(bend)))
        exact = routing.rest_len - routing.lengths_of_qpos(qpos)              # (n, 12) shortening
        model = np.repeat(bend.reshape(-1, 6) @ M.T, 2, axis=1) * routing.sign      # (n, 12)
        err = np.abs(exact - model)
        scale = np.abs(model).max(1).mean()
        print(f"   theta <= {theta_max:4.2f}: max |error| {err.max() * 1000:6.3f} mm, mean {err.mean() * 1000:6.3f} mm "
              f"(mean cable stroke {scale * 1000:5.1f} mm, worst ratio {err.max() / max(np.abs(model).max(), 1e-9):.3f})")
        if theta_max <= 0.1 and err.max() > 0.05e-3:
            ok = False
            print("   FAIL: first-order error too large at small bends -> hole angle convention wrong")

    # direction check: bend section 1 by (0.3, 0) -> cable at alpha 0 shortens, alpha 180 lengthens
    b = np.zeros((3, 2))
    b[0] = (0.3, 0.0)
    d = routing.rest_len - routing.lengths_of_qpos(kin.qpos_from(b, 0.0))
    print(f"   bend S1 (0.3, 0): cable 0 (alpha 0) shortens {d[0] * 1000:+.2f} mm, cable 1 (alpha 180) {d[1] * 1000:+.2f} mm,"
          f" cable 2 (alpha 90) {d[2] * 1000:+.2f} mm")
    ok &= d[0] > 0 > d[1] and abs(d[2]) < 0.1 * abs(d[0])

    # 3. pair symmetry
    bend = random_bends(rng, 200, C.THETA_MAX)
    qpos = kin.qpos_from(bend, np.zeros(200))
    shorten = routing.rest_len - routing.lengths_of_qpos(qpos)
    asym = np.abs(shorten[:, 0::2] + shorten[:, 1::2])
    print(f"3. pair asymmetry |dL+ + dL-|: max {asym.max() * 1000:.3f} mm (second-order effect at theta = {C.THETA_MAX})")

    # 4. round trips
    s = routing.motors_of_qpos(qpos)
    b_lin = s @ np.linalg.inv(M).T
    print(f"4. motors from exact geometry -> bend by M^-1: max |bend error| {np.abs(b_lin - bend.reshape(-1, 6)).max():.4f} rad")
    b2 = bend.reshape(-1, 6)
    assert np.allclose((b2 @ M.T) @ np.linalg.inv(M).T, b2)
    lim = CM.motor_limit()
    print("   motor rate (mm / step):", np.round(CM.motor_rate() * 1000, 3), " limit (mm):", np.round(lim * 1000, 1))
    print("   max |motor| in the random set (mm):", np.round(np.abs(s).max(0) * 1000, 1))

    # cross-check against the constants.K_to_Length_3Seg (motor angles in deg, lead screw pitch 2 mm/rev)
    th = np.linalg.norm(bend, axis=2)
    ph = np.arctan2(bend[..., 1], bend[..., 0])
    k = th / C.SECTION_LENGTHS
    deg = C.K_to_Length_3Seg(k[0, 0], ph[0, 0], k[0, 1], ph[0, 1], k[0, 2], ph[0, 2])       # order of C.MOTOR_ALPHAS
    ours = (M @ bend[0].reshape(6))[[0, 1, 4, 5, 2, 3]]                                     # -> constants' motor order
    print(f"   K_to_Length_3Seg vs M @ bend (same bend): max diff {np.abs(-deg / 360 * C.LEAD_SCREW_PITCH - ours).max() * 1e6:.3f} um")
    assert np.allclose(-deg / 360 * C.LEAD_SCREW_PITCH, ours, atol=1e-9), "motor convention differs from K_to_Length_3Seg"

    print("\nALL OK" if ok else "\nCHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
