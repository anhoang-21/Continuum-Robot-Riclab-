"""
==============================================================================
multi_pipe.py - Scenes with several pipes and a language instruction (numpy only)
==============================================================================
Stage 5 (language-conditioned insertion): N pipes of different tube colours stand in
the robot's reach; an instruction names one of them, by colour ("go through the red
pipe") or by its place in the overview camera image ("the leftmost pipe"). The target
is the env's physical, contact-monitored pipe; the others are obstacles.

  OverviewCamera        fixed third-person camera on the rig (pose, projection)
  MultiPipeSampler      scene (pipe placements, colours, target, instruction) and start
                        states (task start or the student's pre-insertion start) that
                        touch none of the pipes
==============================================================================
"""

import math
from dataclasses import dataclass, field

import numpy as np

import constants as C
from preinsert import TRAIN_MIN_FRACTION, VIEW_MIN_FRACTION, preinsert_pose, view_fraction
from reset_logic import ResetSampler, StartState

# tube colours (the collar at the mouth stays yellow on every pipe: it is the mouth marker)
PALETTE = {
    "red": (0.80, 0.12, 0.10),
    "green": (0.12, 0.62, 0.20),
    "blue": (0.12, 0.30, 0.85),
    "purple": (0.50, 0.18, 0.70),
    "white": (0.92, 0.92, 0.92),
    "black": (0.08, 0.08, 0.09),
}
TRAIN_TUBE = (0.55, 0.78, 1.0)             # the light-blue tube of the single-pipe task
MIN_SEPARATION = 0.13                      # m between pipe axes (outer collar radius 42.5 mm)
POSITION_GAP_PX = 40                       # image x gap to the next pipe for "leftmost" / "rightmost"
DEPTH_GAP = 0.05                           # m, distance-to-camera gap to the next pipe for "closest" / "farthest"

COLOR_TEMPLATES = ("Go through the {c} pipe.", "Insert the robot into the {c} pipe.",
                   "Thread the tip through the {c} one.", "Enter the {c} tube.")
POSITION_TEMPLATES = {
    "left": ("Go through the leftmost pipe.", "Insert into the pipe on the far left."),
    "right": ("Go through the rightmost pipe.", "Insert into the pipe on the far right."),
    "near": ("Go through the pipe closest to the camera.", "Insert into the front pipe."),
    "far": ("Go through the pipe farthest from the camera.", "Insert into the pipe at the back."),
}


class OverviewCamera:
    """Third-person camera fixed to the rig, looking at the pipes from the front and above (ROS axes)."""

    EYE = np.array([0.13, -0.12, 0.90])       # inside the frame, in front of the robot (no beam in the way)
    TARGET = np.array([0.13, 0.19, 0.55])
    WIDTH, HEIGHT = 640, 480
    HFOV_DEG = 75.0

    def __init__(self):
        f = self.TARGET - self.EYE
        f /= np.linalg.norm(f)
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        self.rot = np.stack([right, down, f], axis=1)          # camera -> env
        self.fx = self.WIDTH / 2.0 / math.tan(math.radians(self.HFOV_DEG) / 2.0)
        self.K = np.array([[self.fx, 0.0, self.WIDTH / 2.0], [0.0, self.fx, self.HEIGHT / 2.0], [0.0, 0.0, 1.0]])

    def quat(self):
        """(w, x, y, z) of `rot`."""
        m = self.rot
        w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
        x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
        y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
        z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
        return (w, x, y, z)

    def project(self, p):
        """env-frame points (..., 3) -> pixel coordinates (..., 2)."""
        pc = (np.asarray(p) - self.EYE) @ self.rot
        return np.stack([self.K[0, 0] * pc[..., 0] / pc[..., 2] + self.K[0, 2],
                         self.K[1, 1] * pc[..., 1] / pc[..., 2] + self.K[1, 2]], axis=-1)


@dataclass
class MultiScene:
    pipe_xy: np.ndarray            # (N, 2)
    z_top: np.ndarray              # (N,)
    colors: list                   # N colour names
    target: int
    instruction: str
    kind: str                      # "color" | "position" (left / right / near / far)
    image_x: np.ndarray = field(default=None)   # (N,) pixel x of the mouths in the overview image


class MultiPipeSampler:
    def __init__(self, kin, n_pipes=3, pipe_height=C.PIPE_HEIGHT, random_offset=(0.0, 0.17), p_position=0.5):
        self.kin = kin
        self.n = n_pipes
        self.base = ResetSampler(kin, "random", pipe_height, random_offset)
        self.bore = self.base.bore_length
        self.cam = OverviewCamera()
        self.p_position = p_position

    # ------------------------------------------------------------------
    def scene(self, rng, target_color=None):
        for _ in range(200):
            pipes = [self.base.place_pipe(rng) for _ in range(self.n)]
            xy = np.array([p[0] for p in pipes])
            d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(self.n)
            if d.min() >= MIN_SEPARATION:
                break
        z = np.array([p[1] for p in pipes])
        colors = [str(c) for c in rng.choice(list(PALETTE), size=self.n, replace=False)]
        if target_color is not None and target_color not in colors:
            colors[0] = target_color
        target = int(rng.integers(self.n)) if target_color is None else colors.index(target_color)
        mouths = np.column_stack([xy, z])
        img_x = self.cam.project(mouths)[:, 0]
        dist = np.linalg.norm(mouths - self.cam.EYE, axis=1)
        others = [j for j in range(self.n) if j != target]
        # relations in which the target is clearly the extreme one
        rel = []
        if all(img_x[target] < img_x[j] - POSITION_GAP_PX for j in others):
            rel.append("left")
        if all(img_x[target] > img_x[j] + POSITION_GAP_PX for j in others):
            rel.append("right")
        if all(dist[target] < dist[j] - DEPTH_GAP for j in others):
            rel.append("near")
        if all(dist[target] > dist[j] + DEPTH_GAP for j in others):
            rel.append("far")
        if rel and rng.uniform() < self.p_position:
            kind, where = "position", str(rng.choice(rel))
            text = str(rng.choice(POSITION_TEMPLATES[where]))
        else:
            kind, text = "color", str(rng.choice(COLOR_TEMPLATES)).format(c=colors[target])
        return MultiScene(xy, z, colors, target, text, kind, img_x)

    def obstacles(self, sc, exclude):
        return [(sc.pipe_xy[j], float(sc.z_top[j]), self.bore) for j in range(self.n) if j != exclude]

    def _touches(self, sc, bend, elev):
        xpos, xmat = self.kin.forward(self.kin.qpos_from(bend, elev))
        return any(self.kin.pipe_overlap(xpos, xmat, sc.pipe_xy[j], float(sc.z_top[j]), self.bore)
                   for j in range(self.n))

    def task_start(self, rng, sc):
        """Nearly straight robot, tip 2-7 cm above the highest pipe mouth, touching no pipe."""
        kin = self.kin
        for attempt in range(100):
            z_ref = float(sc.z_top.max())
            elev_lo = z_ref + C.START_GAP[0] - kin.tip_z0
            elev_hi = min(z_ref + C.START_GAP[1] - kin.tip_z0, C.ELEV_RANGE[1] - 0.002)
            elev = rng.uniform(max(elev_lo, C.ELEV_RANGE[0]), max(elev_hi, elev_lo))
            theta = rng.uniform(0.0, 0.15, size=3)
            direction = rng.uniform(0.0, 2.0 * np.pi, size=3)
            bend = np.stack([theta * np.cos(direction), theta * np.sin(direction)], axis=1)
            if not self._touches(sc, bend, elev):
                break
        t = sc.target
        return StartState(pipe_xy=sc.pipe_xy[t].copy(), z_top=float(sc.z_top[t]), bend=bend, elev=float(elev),
                          assisted=False, attempts=attempt + 1)

    def student_start(self, rng, sc, est_sigma_xy=0.006, est_sigma_z=0.003, bend_noise=0.02):
        """Pre-insertion pose for the target from a noisy estimate (as after the overview localisation)."""
        kin, t = self.kin, sc.target
        xy, z = sc.pipe_xy[t], float(sc.z_top[t])
        for attempt in range(100):
            sigma = rng.uniform(0.0, est_sigma_xy)
            xy_hat = xy + rng.normal(0.0, sigma, size=2)
            z_hat = z + rng.normal(0.0, est_sigma_z)
            pose = preinsert_pose(kin, xy_hat, z_hat, self.bore, obstacles=self.obstacles(sc, t))
            if pose is None:
                continue
            bend = pose[0] + rng.normal(0.0, bend_noise, size=(3, 2))
            elev = pose[1]
            xpos, xmat = kin.forward(kin.qpos_from(bend, elev))
            if kin.tip_pos(xpos, xmat)[2] < z + 0.01 or self._touches(sc, bend, elev):
                continue
            if view_fraction(kin, bend, elev, xy, z) < TRAIN_MIN_FRACTION:
                continue
            return StartState(pipe_xy=xy.copy(), z_top=z, bend=bend, elev=float(elev), assisted=True,
                              attempts=attempt + 1)
        return None


__all__ = ["PALETTE", "TRAIN_TUBE", "OverviewCamera", "MultiScene", "MultiPipeSampler", "VIEW_MIN_FRACTION"]
