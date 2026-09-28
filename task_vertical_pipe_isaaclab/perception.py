"""
==============================================================================
perception.py - Pipe-mouth detector for the tip camera (classical, no learning)
==============================================================================
One RGB-D frame of the eye-in-hand camera + the camera pose -> pipe mouth in the
env frame:

  1. colour mask of the yellow collar around the pipe mouth (the only yellow part
     of the scene; the tube is light blue, the lower collar green)
  2. every collar pixel with a valid depth (5x5 median-filtered against depth noise,
     farther than the robot's reach ignored) -> 3D point (pinhole back-projection,
     then camera -> env frame with the pose from forward kinematics)
  3. z of the mouth: the collar's top face (the highest points); only the points
     on / near that face are used below
  4. centre: least-squares circle of known radius through their xy
     (algebraic fit as start, then Gauss-Newton with outlier trimming), so a
     partly visible collar (pipe at the image edge) still gives the centre
  5. a short arc fits two circles (centre on either side of it): the one with the
     tube (light blue, inside and below the collar) within its radius is kept
  6. quality: #points, RMS residual, angular coverage of the arc

numpy + OpenCV; the env hands in the frames (vision_env.VisionPipeEnv.tip_camera).
==============================================================================
"""

from dataclasses import dataclass

import cv2
import numpy as np

import constants as C

COLLAR_TOP = 0.0006                                        # collar top face above the pipe mouth (convert_mjcf_to_usd)
COLLAR_R = (C.PLATE_HOLE_RADIUS + 0.0005, C.PLATE_HOLE_RADIUS + C.PIPE_WALL + 0.004)
FIT_RADIUS = 0.5 * (COLLAR_R[0] + COLLAR_R[1])             # middle of the collar's top face
TUBE_R = C.PLATE_HOLE_RADIUS + C.PIPE_WALL                # outer radius of the tube below the collar
N_BINS = 24                                                # angular bins for the coverage
TOP_BAND = 0.0025                                          # points this far below the top face are left out


@dataclass
class Detection:
    valid: bool
    pipe_xy: np.ndarray        # (2,) env frame
    z_top: float
    n_points: int
    rms: float                 # m, radial residual of the circle fit
    coverage: float            # fraction of the circle covered by collar points (0..1)

    @staticmethod
    def none(n=0):
        return Detection(False, np.full(2, np.nan), float("nan"), n, float("nan"), 0.0)


def collar_mask(rgb):
    """rgb (H, W, 3) float in [0, 1] -> bool mask of the yellow collar."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (np.minimum(r, g) - b > 0.15) & (r > 0.4) & (g > 0.6 * r) & (g < 1.15 * r)


def tube_mask(rgb):
    """rgb (H, W, 3) -> bool mask of the light-blue tube (the bore below the collar)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > r + 0.08) & (b >= g - 0.02) & (b > 0.5)


def backproject(depth, K, mask):
    """Pixels of `mask` with depth > 0 -> (M, 3) points in the camera frame (ROS axes: x right, y down, z forward)."""
    v, u = np.nonzero(mask & (depth > 0.0))
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def fit_circle_algebraic(xy):
    """Kasa fit -> centre (2,), radius."""
    A = np.column_stack([2.0 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def fit_circle_known_radius(xy, radius, c0, iters=15, trim=2.5):
    """Gauss-Newton on sum (|p - c| - radius)^2, points beyond `trim` x RMS dropped after each pass."""
    c = np.asarray(c0, dtype=np.float64).copy()
    keep = np.ones(len(xy), dtype=bool)
    for _ in range(iters):
        d = xy[keep] - c
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        res = dist - radius
        J = -d / dist[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c += step
        all_res = np.abs(np.linalg.norm(xy - c, axis=1) - radius)
        rms = np.sqrt(np.mean(all_res[keep] ** 2))
        keep = all_res <= max(trim * rms, 0.002)
        if np.linalg.norm(step) < 1e-7:
            break
    all_res = np.linalg.norm(xy[keep] - c, axis=1) - radius
    return c, float(np.sqrt(np.mean(all_res ** 2))), keep


def coverage(xy, c):
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    bins = np.floor((ang + np.pi) / (2.0 * np.pi) * N_BINS).astype(int) % N_BINS
    return len(np.unique(bins)) / N_BINS


class PipeDetector:
    def __init__(self, min_points=30, max_rms=0.004, min_coverage=0.2, max_range=0.5, median=5, tube_check=True):
        """
        max_range (m): farther pixels are ignored (the robot's reach; other rigs in the scene are 2 m away).
        median: size of the median filter applied to the depth image (0 = none) against depth noise.
        tube_check: pick the side of a short arc with the light-blue tube (off for coloured tubes).
        """
        self.min_points, self.max_rms, self.min_coverage = min_points, max_rms, min_coverage
        self.max_range, self.median, self.tube_check = max_range, median, tube_check

    def _in_range(self, depth):
        return np.where(depth < self.max_range, depth, 0.0)

    def count(self, rgb, depth):
        """Number of collar pixels with depth (cheap check used while the camera is moving)."""
        return int((collar_mask(rgb) & (self._in_range(depth) > 0.0)).sum())

    def detect(self, rgb, depth, K, cam_pos, cam_rot):
        """
        rgb (H, W, 3) [0, 1], depth (H, W) m, K (3, 3), cam_pos (3,) env frame,
        cam_rot (3, 3) camera -> env. Returns a Detection.
        """
        depth = self._in_range(depth)
        if self.median:
            depth = cv2.medianBlur(depth.astype(np.float32), self.median)
        pts_c = backproject(depth, K, collar_mask(rgb))
        n = len(pts_c)
        if n < self.min_points:
            return Detection.none(n)
        pts = pts_c @ cam_rot.T + cam_pos
        # the collar's flat top face is the highest part of the pipe: keep the points near it. Seen at a
        # grazing angle, most collar pixels are on its 6.6 mm tall outer wall (radius COLLAR_R[1]), which
        # would pull a circle of radius FIT_RADIUS up to 3.75 mm towards the camera
        z_ref = float(np.percentile(pts[:, 2], 90))
        top = pts[pts[:, 2] > z_ref - TOP_BAND]
        if len(top) < self.min_points:
            return Detection.none(len(top))
        z_top = float(np.median(top[:, 2])) - COLLAR_TOP
        xy = top[:, :2]
        # start points: the algebraic fit, and the centroid pushed one radius to either side of the arc
        # (away from / towards the camera)
        cands = [fit_circle_algebraic(xy)[0]]
        centroid = xy.mean(axis=0)
        out = centroid - cam_pos[:2]
        if np.linalg.norm(out) > 1e-6:
            out = FIT_RADIUS * out / np.linalg.norm(out)
            cands += [centroid + out, centroid - out]
        fits = [fit_circle_known_radius(xy, FIT_RADIUS, c_init) for c_init in cands]

        # tube points (bore wall below the collar): the right centre has them inside the collar
        tube = backproject(depth, K, tube_mask(rgb) if self.tube_check else np.zeros(depth.shape, bool))
        tube = tube @ cam_rot.T + cam_pos
        tube = tube[(tube[:, 2] < z_top) & (tube[:, 2] > z_top - C.PIPE_HEIGHT - 0.01)]

        def score(fit):
            c, rms, _ = fit
            outside = np.mean(np.linalg.norm(tube[:, :2] - c, axis=1) > TUBE_R + 0.002) if len(tube) else 0.0
            return (rms > self.max_rms, round(float(outside), 1), rms)

        c, rms, keep = min(fits, key=score)
        cov = coverage(xy[keep], c)
        valid = (keep.sum() >= self.min_points) and (rms <= self.max_rms) and (cov >= self.min_coverage)
        return Detection(bool(valid), c, z_top, int(keep.sum()), rms, cov)


def detect_all(detector, rgb, depth, K, cam_pos, cam_rot, min_pixels=60):
    """Every pipe mouth in view: one detection per connected blob of collar pixels (valid ones only)."""
    depth = np.where(depth < detector.max_range, depth, 0.0)
    mask = (collar_mask(rgb) & (depth > 0.0)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_pixels:
            continue
        det = detector.detect(rgb, np.where(labels == k, depth, 0.0), K, cam_pos, cam_rot)
        if det.valid:
            found.append(det)
    return found
