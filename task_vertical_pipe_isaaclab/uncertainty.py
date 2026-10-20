"""
==============================================================================
uncertainty.py - the uncertainty set of the cable-driven robot (ND1)
==============================================================================
The parameters of cables.py / motors.py are estimates, not measurements (README, "Uncertain
parameters"). This file states what is uncertain, how much, and draws one parameter set per env:

  parameter            unit     per    nominal   train range        ood range
  k_c   cable stiffness N/m     cable  3000      2000 - 4500 (*)    1500 - 6000 (*)
  T0    pre-tension     N       cable  3.0       2.0 - 4.0 (*)      1.5 - 5.0 (*)
  asym  pair asymmetry  -       pair   0         0 - 0.2            0 - 0.35    (cable b: T0 (1 - asym))
  k_j   backbone spring N m/rad env    0.8       0.6 - 1.0          0.5 - 1.2
  c_j   backbone damp.  N m s   env    1.0       0.6 - 1.5          0.4 - 2.0
  c_c   cable damping   N s/m   cable  5.0       2 - 8              1 - 10
  mu_theta capstan      -       cable  0         0 - 0.30 (*)       0 - 0.45 (*)   (Bowden sheath + disc holes)
  phi   compliance split -      cable  0.5       0.3 - 0.7          0.2 - 0.8      (nut side of the friction element)
  backlash nut play     m       motor  0         0 - 0.3 mm         0 - 0.6 mm
  motor_bw position loop Hz     motor  60        40 - 80            30 - 90
  payload tip mass      kg      env    0         0 - 0.030          0 - 0.050
  latency command delay steps   env    0         0 / 1 (p 0.3)      0 / 1 (p 0.5)
  pos_noise mocap noise m       -      0         0.5 mm             1.0 mm         (observed body positions)
  tension_noise load cell N     -      0         0.10               0.20           (observed tensions, if any)

(*) env value drawn in the range, then every cable / motor spread around it (log-normal, 10 % train /
20 % ood): the 12 cables of one robot are not identical either.

The friction element is the one of home_hysteresis.py: nut-side spring k_n = k_c / phi, a capstan
element that holds while exp(-mu theta) <= T_nut / T_robot <= exp(mu theta), robot-side spring
k_r = k_c / (1 - phi) (1 / k_c = 1 / k_n + 1 / k_r).

`UncertaintySet.from_json` replaces ranges by identified ones (sysid.py writes that file), so the
randomisation of the RL training is the uncertainty set of the identification (doc section 7).

numpy only (no torch, no Isaac Sim).
==============================================================================
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np

import cables as CB

LEVELS = ("nominal", "train", "ood")

NOMINAL = {
    "k_c": CB.CABLE_STIFFNESS, "t0": CB.CABLE_PRETENSION, "asym": 0.0,
    "k_j": CB.BACKBONE_STIFFNESS, "c_j": CB.BACKBONE_DAMPING, "c_c": CB.CABLE_DAMPING,
    "mu_theta": 0.0, "phi": 0.5, "backlash": 0.0, "motor_bw": 60.0, "payload": 0.0,
    "latency_p": 0.0, "pos_noise": 0.0, "tension_noise": 0.0, "spread": 0.0,
}

# (low, high) per level; scalars that are not ranges (probabilities, noise levels, spread) are (v, v)
RANGES = {
    "train": {
        "k_c": (2000.0, 4500.0), "t0": (2.0, 4.0), "asym": (0.0, 0.2),
        "k_j": (0.6, 1.0), "c_j": (0.6, 1.5), "c_c": (2.0, 8.0),
        "mu_theta": (0.0, 0.30), "phi": (0.3, 0.7), "backlash": (0.0, 0.3e-3), "motor_bw": (40.0, 80.0),
        "payload": (0.0, 0.030), "latency_p": (0.3, 0.3), "pos_noise": (0.5e-3, 0.5e-3),
        "tension_noise": (0.10, 0.10), "spread": (0.10, 0.10),
    },
    "ood": {
        "k_c": (1500.0, 6000.0), "t0": (1.5, 5.0), "asym": (0.0, 0.35),
        "k_j": (0.5, 1.2), "c_j": (0.4, 2.0), "c_c": (1.0, 10.0),
        "mu_theta": (0.0, 0.45), "phi": (0.2, 0.8), "backlash": (0.0, 0.6e-3), "motor_bw": (30.0, 90.0),
        "payload": (0.0, 0.050), "latency_p": (0.5, 0.5), "pos_noise": (1.0e-3, 1.0e-3),
        "tension_noise": (0.20, 0.20), "spread": (0.20, 0.20),
    },
}
LOG_UNIFORM = ("k_c", "c_c")          # drawn uniformly in log space

# privileged vector (teacher / critic input): what each entry is, and its scale
PRIV_NAMES = ("log_k_c", "k_c_spread", "t0", "t0_asym", "k_j", "c_j", "log_c_c", "mu_theta", "phi",
              "backlash_mm", "motor_bw", "payload_g", "latency")
PRIV_DIM = len(PRIV_NAMES)


@dataclass
class UncertaintySet:
    """Ranges of every uncertain parameter at one level (`nominal`: all ranges collapse to the nominal values)."""
    level: str = "nominal"
    ranges: dict = field(default_factory=dict)

    @classmethod
    def make(cls, level: str = "nominal", t0: float = NOMINAL["t0"]) -> "UncertaintySet":
        """The ranges of `level`; the pre-tension ranges scale with the design pre-tension t0 (N)."""
        if level not in LEVELS:
            raise ValueError(f"uncertainty level must be one of {LEVELS}, got {level!r}")
        if level == "nominal":
            out = cls(level, {k: (v, v) for k, v in NOMINAL.items()})
        else:
            out = cls(level, copy.deepcopy(RANGES[level]))
        s = t0 / NOMINAL["t0"]
        out.ranges["t0"] = (out.ranges["t0"][0] * s, out.ranges["t0"][1] * s)
        return out

    @classmethod
    def from_json(cls, path: str, base: str = "train", t0: float = NOMINAL["t0"]) -> "UncertaintySet":
        """Ranges of `base`, with the entries of the file (sysid.py: {"ranges": {name: [lo, hi]}}) replacing them."""
        out = cls.make(base, t0)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k, (lo, hi) in data.get("ranges", {}).items():
            if k not in NOMINAL:
                raise KeyError(f"{path}: unknown parameter {k!r}")
            out.ranges[k] = (float(lo), float(hi))
        out.level = f"identified({base})"
        return out

    def to_dict(self) -> dict:
        return {"level": self.level, "ranges": {k: list(v) for k, v in self.ranges.items()}}

    # ------------------------------------------------------------------
    def _draw(self, rng, name, n):
        lo, hi = self.ranges[name]
        if hi <= lo:
            return np.full(n, lo, dtype=np.float64)
        if name in LOG_UNIFORM:
            return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        return rng.uniform(lo, hi, n)

    def sample(self, rng: np.random.Generator, n: int) -> dict:
        """
        One parameter set per env, as numpy arrays:
          k_c, t0, c_c, mu_theta, phi (n, 12); backlash, motor_bw (n, 6); k_j, c_j, payload (n,);
          latency (n,) int (0 / 1 steps); pos_noise, tension_noise (n,) (std of the observation noise).
        """
        d = {k: self._draw(rng, k, n) for k in NOMINAL}
        spread = d["spread"][:, None]

        def per(x, m):                                   # env value spread over its m cables / motors
            if np.all(spread == 0.0):
                return np.repeat(x[:, None], m, axis=1)
            return x[:, None] * np.exp(spread * rng.standard_normal((n, m)))

        t0 = per(d["t0"], CB.N_CABLES)
        t0[:, 1::2] *= (1.0 - d["asym"])[:, None]       # cable b of every pair: the lower pre-tension
        mu = d["mu_theta"][:, None] * (np.ones((n, CB.N_CABLES)) if np.all(spread == 0.0)
                                      else rng.uniform(0.5, 1.5, (n, CB.N_CABLES)))
        return {
            "k_c": per(d["k_c"], CB.N_CABLES), "t0": t0, "c_c": per(d["c_c"], CB.N_CABLES),
            "mu_theta": mu, "phi": np.repeat(d["phi"][:, None], CB.N_CABLES, axis=1),
            "backlash": np.repeat(d["backlash"][:, None], CB.N_MOTORS, axis=1) * (
                1.0 if np.all(spread == 0.0) else rng.uniform(0.5, 1.5, (n, CB.N_MOTORS))),
            "motor_bw": per(d["motor_bw"], CB.N_MOTORS),
            "k_j": d["k_j"], "c_j": d["c_j"], "payload": d["payload"],
            "latency": (rng.uniform(size=n) < d["latency_p"]).astype(np.int64),
            "pos_noise": d["pos_noise"], "tension_noise": d["tension_noise"],
        }


def nominal_params(n: int, t0: float = NOMINAL["t0"]) -> dict:
    return UncertaintySet.make("nominal", t0).sample(np.random.default_rng(0), n)


def privileged(p: dict) -> np.ndarray:
    """Parameter set -> (n, PRIV_DIM) vector, roughly zero-mean / unit-scale over the train ranges."""
    k_c = np.asarray(p["k_c"])
    t0 = np.asarray(p["t0"])
    cols = [
        np.log(k_c.mean(1) / NOMINAL["k_c"]) / 0.4,
        np.log(k_c).std(1) / 0.1,
        (t0.mean(1) - 3.0) / 1.0,
        (t0[:, 0::2] - t0[:, 1::2]).mean(1) / (0.1 * 3.0),
        (np.asarray(p["k_j"]) - 0.8) / 0.2,
        (np.asarray(p["c_j"]) - 1.0) / 0.4,
        np.log(np.asarray(p["c_c"]).mean(1) / NOMINAL["c_c"]) / 0.5,
        np.asarray(p["mu_theta"]).mean(1) / 0.15,
        (np.asarray(p["phi"]).mean(1) - 0.5) / 0.2,
        np.asarray(p["backlash"]).mean(1) * 1e3 / 0.15,
        (np.asarray(p["motor_bw"]).mean(1) - 60.0) / 20.0,
        np.asarray(p["payload"]) * 1e3 / 15.0,
        np.asarray(p["latency"], dtype=np.float64),
    ]
    return np.stack(cols, axis=1)


def take(p: dict, idx) -> dict:
    """The parameter sets of some envs."""
    return {k: np.asarray(v)[idx] for k, v in p.items()}
