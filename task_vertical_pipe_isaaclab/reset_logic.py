"""
==============================================================================
reset_logic.py - Start-state sampling of the vertical-pipe task (numpy, per env)
==============================================================================
Line-by-line port of VerticalPipeEnv.reset / _place_pipe / _assisted_start from
D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\vertical_pipe_env.py.

The random draws are made in the same order and with the same numpy calls, so a
generator seeded like the MuJoCo env (np.random.default_rng(seed)) produces the
same pipe placement and start pose (tests/compare_with_mujoco.py checks this).
Where MuJoCo calls mj_forward on a start pose, ContinuumKinematics is used; the
`data.ncon == 0` test becomes ContinuumKinematics.pipe_overlap.
==============================================================================
"""

from dataclasses import dataclass

import numpy as np

from constants import ELEV_RANGE, EXIT_MARGIN, PLATE_HOLE_XY, PLATE_TOP_Z, START_GAP, bore_length


@dataclass
class StartState:
    pipe_xy: np.ndarray        # (2,) centre of the pipe, world / env frame
    z_top: float               # height of the pipe mouth
    bend: np.ndarray           # (3, 2) commanded bend vectors
    elev: float                # commanded elevator position
    assisted: bool             # the episode started part-way along the solution
    attempts: int              # sampling attempts used (MuJoCo allows 50)


class ResetSampler:
    def __init__(self, kin, pipe_mode, pipe_height, random_offset):
        if pipe_mode not in ("rig", "random"):
            raise ValueError(f"pipe_mode must be 'rig' or 'random', got {pipe_mode!r}")
        self.kin = kin
        self.pipe_mode = pipe_mode
        self.pipe_height = float(pipe_height)
        self.random_offset = tuple(random_offset)
        self.bore_length = bore_length(pipe_mode, self.pipe_height)

    # ------------------------------------------------------------------
    def place_pipe(self, rng):
        """VerticalPipeEnv._place_pipe -> (pipe_xy, z_top)."""
        if self.pipe_mode == "rig":
            return PLATE_HOLE_XY.copy(), PLATE_TOP_Z + self.pipe_height
        dist = rng.uniform(*self.random_offset)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        pipe_xy = self.kin.axis_xy + dist * np.array([np.cos(angle), np.sin(angle)])
        z_bot = rng.uniform(PLATE_TOP_Z + EXIT_MARGIN + 0.015, PLATE_TOP_Z + EXIT_MARGIN + 0.060)
        return pipe_xy, z_bot + self.pipe_height

    def _pose(self, bend, elev):
        xpos, xmat = self.kin.forward(self.kin.qpos_from(bend, elev))
        return xpos, xmat, self.kin.tip_pos(xpos, xmat)

    def _collides(self, xpos, xmat, pipe_xy, z_top):
        return self.kin.pipe_overlap(xpos, xmat, pipe_xy, z_top, self.bore_length)

    def assisted_start(self, rng, pipe_xy, z_top):
        """VerticalPipeEnv._assisted_start -> (ok, bend, elev)."""
        z_success = z_top - self.bore_length - EXIT_MARGIN
        aligned = self.kin.aligned_bend(pipe_xy)
        if rng.uniform() < 0.6:
            # S-curve partly formed, tip still above the pipe
            bend = rng.uniform(0.3, 1.0) * aligned + rng.normal(0.0, 0.02, size=(3, 2))
            tip_target = z_top + rng.uniform(0.01, 0.05)
        else:
            # Aligned and already inside the bore
            bend = aligned + rng.normal(0.0, 0.005, size=(3, 2))
            tip_target = z_top - rng.uniform(0.0, 0.8) * (z_top - z_success)
        _, _, tip = self._pose(bend, 0.0)
        elev = tip_target - tip[2]
        if not ELEV_RANGE[0] <= elev <= ELEV_RANGE[1]:
            return False, bend, elev
        xpos, xmat, _ = self._pose(bend, elev)
        return not self._collides(xpos, xmat, pipe_xy, z_top), bend, elev

    def sample(self, rng, assist_prob):
        """VerticalPipeEnv.reset (the part that picks the scene and the start pose)."""
        assisted = rng.uniform() < assist_prob
        used_assist = False
        for attempt in range(50):
            pipe_xy, z_top = self.place_pipe(rng)
            if assisted:
                ok, bend, elev = self.assisted_start(rng, pipe_xy, z_top)
                if ok:
                    used_assist = True
                    break
            # Normal start: nearly straight robot, tip above the pipe top
            elev_lo = z_top + START_GAP[0] - self.kin.tip_z0
            elev_hi = min(z_top + START_GAP[1] - self.kin.tip_z0, ELEV_RANGE[1] - 0.002)
            elev = rng.uniform(max(elev_lo, ELEV_RANGE[0]), max(elev_hi, elev_lo))
            theta = rng.uniform(0.0, 0.15, size=3)
            direction = rng.uniform(0.0, 2.0 * np.pi, size=3)
            bend = np.stack([theta * np.cos(direction), theta * np.sin(direction)], axis=1)
            xpos, xmat, tip = self._pose(bend, elev)
            if tip[2] > z_top + 0.015 and not self._collides(xpos, xmat, pipe_xy, z_top):
                break
        return StartState(pipe_xy=np.asarray(pipe_xy, dtype=np.float64), z_top=float(z_top),
                          bend=np.asarray(bend, dtype=np.float64).reshape(3, 2), elev=float(elev),
                          assisted=used_assist, attempts=attempt + 1)
