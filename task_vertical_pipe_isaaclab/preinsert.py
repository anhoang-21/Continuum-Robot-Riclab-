"""
==============================================================================
preinsert.py - Hand-over poses and training starts of the image-based student
==============================================================================
The student policy (student_policy.py) sees only the tip camera and the robot's own
joint commands, not the pipe pose. It takes over once the pipe is in view, at a
"pre-insertion" pose: the S-curve partly formed towards the pipe (a fraction f of the
aligned bend, section 3 straight), the tip 3-5 cm above the mouth, chosen so that the
collar is inside the camera's field of view. These poses are inside the teacher's
training distribution (its assisted starts: 0.3-1.0 x aligned bend, tip 1-5 cm above).

  view_fraction()       share of the collar ring the tip camera sees in a pose (kinematics)
  preinsert_pose()      pre-insertion pose for a (possibly estimated) pipe pose; used at
                        deployment with the camera's estimate (vision_pipeline.py)
  StudentStartSampler   training starts: pre-insertion poses computed from a noisy pipe
                        estimate (as after a real search), plus normal starts of the task
                        whose view already contains the pipe

numpy only (no Isaac Sim), like reset_logic.py.
==============================================================================
"""

import math

import numpy as np

import constants as C
from constants import CAM_FOV_DEG, CAM_OFFSET
from reset_logic import ResetSampler, StartState

VIEW_RING_R = C.PLATE_HOLE_RADIUS + C.PIPE_WALL            # collar ring used for the visibility test
VIEW_MIN_FRACTION = 0.6                                    # share of the ring that has to be in the image
PREINSERT_FRACTIONS = (0.6, 0.7, 0.8, 0.9, 1.0)            # bend fraction f, smallest visible one is used
PREINSERT_HEIGHTS = (0.04, 0.05, 0.03)                     # tip above the mouth (m)
TRAIN_MIN_FRACTION = 0.4                                   # training starts: share of the true ring in view


def view_fraction(kin, bend, elev, pipe_xy, z_top, n=32, margin=0.95):
    """Share of the collar ring (at the mouth) that the tip camera sees in pose (bend, elev)."""
    xpos, xmat = kin.forward(kin.qpos_from(bend, elev))
    rot = xmat[kin.tip_id]
    cam = xpos[kin.tip_id] + CAM_OFFSET * rot[:, 2]
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([pipe_xy[0] + VIEW_RING_R * np.cos(a), pipe_xy[1] + VIEW_RING_R * np.sin(a),
                     np.full(n, z_top)], axis=1)
    pc = (ring - cam) @ rot                                # camera frame (ROS axes = Seg15 axes)
    t = math.tan(math.radians(CAM_FOV_DEG) / 2.0) * margin
    z = np.maximum(pc[:, 2], 1e-9)
    ok = (pc[:, 2] > 0.004) & (np.abs(pc[:, 0] / z) < t) & (np.abs(pc[:, 1] / z) < t)
    return float(ok.mean())


def preinsert_pose(kin, pipe_xy, z_top, bore_length, heights=PREINSERT_HEIGHTS, fractions=PREINSERT_FRACTIONS,
                   obstacles=()):
    """
    (bend (3, 2), elev) of the first pre-insertion pose that sees the pipe and does not touch it, or None.
    obstacles: other pipes [(pipe_xy, z_top, bore_length), ...] the pose must not touch either.
    """
    aligned = kin.aligned_bend(pipe_xy)
    for h in heights:
        for f in fractions:
            bend = f * aligned
            xpos, xmat = kin.forward(kin.qpos_from(bend, 0.0))
            elev = z_top + h - kin.tip_pos(xpos, xmat)[2]
            if not C.ELEV_RANGE[0] <= elev <= C.ELEV_RANGE[1]:
                continue
            if view_fraction(kin, bend, elev, pipe_xy, z_top) < VIEW_MIN_FRACTION:
                continue
            xpos, xmat = kin.forward(kin.qpos_from(bend, elev))
            if kin.pipe_overlap(xpos, xmat, pipe_xy, z_top, bore_length):
                continue
            if any(kin.pipe_overlap(xpos, xmat, o_xy, o_z, o_bore) for o_xy, o_z, o_bore in obstacles):
                continue
            return bend, float(elev)
    return None


class StudentStartSampler:
    """Training starts of the student (numpy, one generator per env like ResetSampler)."""

    def __init__(self, kin, pipe_mode, pipe_height, random_offset, p_normal=0.25, est_sigma_xy=0.006,
                 est_sigma_z=0.003, bend_noise=0.02):
        self.base = ResetSampler(kin, pipe_mode, pipe_height, random_offset)
        self.kin = kin
        self.p_normal = p_normal
        self.est_sigma_xy, self.est_sigma_z, self.bend_noise = est_sigma_xy, est_sigma_z, bend_noise

    def sample(self, rng, assist_prob=0.0):
        kin, base = self.kin, self.base
        for attempt in range(100):
            if rng.uniform() < self.p_normal:
                # a normal start of the task that already has the pipe in view
                st = base.sample(rng, 0.0)
                if view_fraction(kin, st.bend, st.elev, st.pipe_xy, st.z_top) >= VIEW_MIN_FRACTION:
                    return st
                continue
            pipe_xy, z_top = base.place_pipe(rng)
            # the search hands over with an estimate of the pipe pose, not the truth
            sigma = rng.uniform(0.0, self.est_sigma_xy)
            xy_hat = pipe_xy + rng.normal(0.0, sigma, size=2)
            z_hat = z_top + rng.normal(0.0, self.est_sigma_z)
            pose = preinsert_pose(kin, xy_hat, z_hat, base.bore_length)
            if pose is None:
                continue
            bend = pose[0] + rng.normal(0.0, self.bend_noise, size=(3, 2))
            elev = pose[1]
            xpos, xmat = kin.forward(kin.qpos_from(bend, elev))
            tip = kin.tip_pos(xpos, xmat)
            if tip[2] < z_top + 0.01 or kin.pipe_overlap(xpos, xmat, pipe_xy, z_top, base.bore_length):
                continue
            if view_fraction(kin, bend, elev, pipe_xy, z_top) < TRAIN_MIN_FRACTION:
                continue
            return StartState(pipe_xy=np.asarray(pipe_xy, dtype=np.float64), z_top=float(z_top), bend=bend,
                              elev=float(elev), assisted=True, attempts=attempt + 1)
        return base.sample(rng, 0.0)
