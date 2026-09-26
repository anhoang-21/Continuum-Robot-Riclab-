"""
==============================================================================
kinematics.py - MuJoCo-equivalent kinematics of the continuum robot (numpy)
==============================================================================
Parses the kinematic tree of the MJCF (body pos/quat, joint type/axis/range) and
evaluates it the same way as mj_kinematics. It is used where the MuJoCo env calls
mj_forward / mj_kinematics on a pose that is *not* simulated:

  * the reset logic (tip height of a start pose, Gauss-Newton "aligned_bend"),
  * the start-pose collision test that replaces `data.ncon == 0` (PhysX has no
    collision query without stepping), see `pipe_overlap`.

During simulation the env reads the body poses from PhysX instead; the tests check
that the two agree (tests/check_isaac_env.py) and that this module agrees with
MuJoCo (tests/compare_with_mujoco.py).
==============================================================================
"""

import xml.etree.ElementTree as ET

import numpy as np

from constants import (
    EXIT_MARGIN,
    N_STAVES,
    PIPE_WALL,
    PLATE_HOLE_RADIUS,
    SEG_COLLIDERS,
    SOURCE_MJCF,
    TIP_OFFSET,
)


# ---------------------------------------------------------------------------
# Small rotation helpers (MuJoCo quaternion convention: w, x, y, z)
# ---------------------------------------------------------------------------
def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def axis_angle_to_mat(axis, angle):
    """Rotation matrices about a fixed unit axis for a batch of angles (..., 3, 3)."""
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    m = np.empty(angle.shape + (3, 3))
    m[..., 0, 0] = t * x * x + c
    m[..., 0, 1] = t * x * y - s * z
    m[..., 0, 2] = t * x * z + s * y
    m[..., 1, 0] = t * x * y + s * z
    m[..., 1, 1] = t * y * y + c
    m[..., 1, 2] = t * y * z - s * x
    m[..., 2, 0] = t * x * z - s * y
    m[..., 2, 1] = t * y * z + s * x
    m[..., 2, 2] = t * z * z + c
    return m


def _vec(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=np.float64)


# ---------------------------------------------------------------------------
# Kinematic tree
# ---------------------------------------------------------------------------
class ContinuumKinematics:
    """Kinematic tree of the MJCF with MuJoCo's body / qpos ordering (world body excluded)."""

    def __init__(self, mjcf_path=SOURCE_MJCF):
        root = ET.parse(mjcf_path).getroot()
        compiler = root.find("compiler")
        if compiler is None or compiler.get("angle", "degree") != "radian":
            raise ValueError("expected <compiler angle='radian'>")
        self.body_names, self.parent, self.body_pos, self.body_quat = [], [], [], []
        self.joint_names, self.joint_type, self.joint_axis, self.joint_range, self.joint_body = [], [], [], [], []
        for body in root.find("worldbody").findall("body"):
            self._parse(body, -1)
        self.body_pos = np.array(self.body_pos)
        self.body_rot = quat_to_mat(np.array(self.body_quat))
        self.joint_axis = np.array(self.joint_axis)
        self.joint_range = np.array(self.joint_range)
        self.nbody, self.nq = len(self.body_names), len(self.joint_names)
        self.body_index = {n: i for i, n in enumerate(self.body_names)}
        self.joint_index = {n: i for i, n in enumerate(self.joint_names)}

        self.seg_ids = np.array([self.body_index[f"Seg{i}"] for i in range(1, 16)])
        self.tip_id = self.seg_ids[14]
        self.qadr_x = np.array([self.joint_index[f"Seg{i}_x"] for i in range(1, 16)])
        self.qadr_y = np.array([self.joint_index[f"Seg{i}_y"] for i in range(1, 16)])
        self.elev_qadr = self.joint_index["Elevator_Joint"]

        # Robot axis and straight-tip height at elevator = 0 (as VerticalPipeEnv._cache_ids)
        xpos, xmat = self.forward(np.zeros(self.nq))
        self.axis_xy = xpos[self.seg_ids[0], :2].copy()
        self.tip_z0 = float(self.tip_pos(xpos, xmat)[2])
        self._collider_points = self._make_collider_points()
        self._aligned_cache = {}

    def _parse(self, elem, parent):
        for key in ("euler", "axisangle", "xyaxes", "zaxis"):
            if elem.get(key) is not None:
                raise ValueError(f"body orientation '{key}' is not supported, use quat")
        idx = len(self.body_names)
        self.body_names.append(elem.get("name"))
        self.parent.append(parent)
        self.body_pos.append(_vec(elem.get("pos"), [0.0, 0.0, 0.0]))
        self.body_quat.append(_vec(elem.get("quat"), [1.0, 0.0, 0.0, 0.0]))
        for joint in elem.findall("joint"):
            jtype = joint.get("type", "hinge")
            if jtype not in ("hinge", "slide"):
                raise ValueError(f"joint type {jtype} not supported")
            if np.any(_vec(joint.get("pos"), [0, 0, 0]) != 0.0):
                raise ValueError("joints must sit at the body origin")
            axis = _vec(joint.get("axis"), [0.0, 0.0, 1.0])
            self.joint_names.append(joint.get("name"))
            self.joint_type.append(jtype)
            self.joint_axis.append(axis / np.linalg.norm(axis))
            self.joint_range.append(_vec(joint.get("range"), [0.0, 0.0]))
            self.joint_body.append(idx)
        for child in elem.findall("body"):
            self._parse(child, idx)

    # ------------------------------------------------------------------
    def forward(self, qpos):
        """mj_kinematics for a batch of qpos (..., nq) -> xpos (..., nbody, 3), xmat (..., nbody, 3, 3)."""
        qpos = np.asarray(qpos, dtype=np.float64)
        batch = qpos.shape[:-1]
        xpos = np.empty(batch + (self.nbody, 3))
        xmat = np.empty(batch + (self.nbody, 3, 3))
        body_joints = {}
        for j, b in enumerate(self.joint_body):
            body_joints.setdefault(b, []).append(j)
        for b in range(self.nbody):
            p = self.parent[b]
            if p < 0:
                pos = np.broadcast_to(self.body_pos[b], batch + (3,)).copy()
                rot = np.broadcast_to(self.body_rot[b], batch + (3, 3)).copy()
            else:
                pos = xpos[..., p, :] + np.einsum("...ij,j->...i", xmat[..., p, :, :], self.body_pos[b])
                rot = xmat[..., p, :, :] @ self.body_rot[b]
            for j in body_joints.get(b, []):
                if self.joint_type[j] == "slide":
                    pos = pos + np.einsum("...ij,j->...i", rot, self.joint_axis[j]) * qpos[..., j, None]
                else:
                    rot = rot @ axis_angle_to_mat(self.joint_axis[j], qpos[..., j])
            xpos[..., b, :] = pos
            xmat[..., b, :, :] = rot
        return xpos, xmat

    def tip_pos(self, xpos, xmat):
        return xpos[..., self.tip_id, :] + TIP_OFFSET * xmat[..., self.tip_id, :, 2]

    def tip_dir(self, xmat):
        return xmat[..., self.tip_id, :, 2]

    @staticmethod
    def joint_angles(bend):
        """Bend vectors (..., 3, 2) -> per-joint angles (x joints, y joints), as VerticalPipeEnv._joint_angles."""
        per_seg = np.asarray(bend, dtype=np.float64) / 5.0
        return np.repeat(-per_seg[..., :, 1], 5, axis=-1), np.repeat(per_seg[..., :, 0], 5, axis=-1)

    def qpos_from(self, bend, elev):
        """Full qpos (MuJoCo order) for bend vectors (..., 3, 2) and elevator heights (...)."""
        bend = np.asarray(bend, dtype=np.float64)
        jx, jy = self.joint_angles(bend)
        qpos = np.zeros(bend.shape[:-2] + (self.nq,))
        qpos[..., self.qadr_x] = jx
        qpos[..., self.qadr_y] = jy
        qpos[..., self.elev_qadr] = elev
        return qpos

    # ------------------------------------------------------------------
    def aligned_bend(self, pipe_xy):
        """
        S-curve that puts the tip on the pipe axis pointing straight down, with section 3
        straight: Gauss-Newton on (u1, v1, u2, v2), same iteration as VerticalPipeEnv.aligned_bend.
        """
        pipe_xy = np.asarray(pipe_xy, dtype=np.float64)
        key = (round(float(pipe_xy[0]), 5), round(float(pipe_xy[1]), 5))
        if key in self._aligned_cache:
            return self._aligned_cache[key].copy()

        def residual(b):
            bend = np.zeros(b.shape[:-1] + (3, 2))
            bend[..., 0, :] = b[..., 0:2]
            bend[..., 1, :] = b[..., 2:4]
            xpos, xmat = self.forward(self.qpos_from(bend, 0.0))
            tip = self.tip_pos(xpos, xmat)
            return np.concatenate([(tip[..., :2] - pipe_xy) / 0.1, self.tip_dir(xmat)[..., :2]], axis=-1)

        b = np.zeros(4)
        for _ in range(25):
            # base point + 4 forward-difference points in one batched FK call
            pts = np.repeat(b[None], 5, axis=0)
            pts[1:] += np.eye(4) * 1e-5
            res = residual(pts)
            r = res[0]
            jac = ((res[1:] - r) / 1e-5).T
            step = np.linalg.lstsq(jac, -r, rcond=None)[0]
            b += step
            if np.linalg.norm(step) < 1e-8:
                break
        bend = np.array([[b[0], b[1]], [b[2], b[3]], [0.0, 0.0]])
        if len(self._aligned_cache) < 64:
            self._aligned_cache[key] = bend.copy()
        return bend

    # ------------------------------------------------------------------
    # Start-pose collision test (replaces `data.ncon == 0` of the MuJoCo reset)
    # ------------------------------------------------------------------
    def _make_collider_points(self, spacing=0.0015):
        """Surface samples of the 15 collision cylinders, in their Seg body frames."""
        pts = []
        for i in range(1, 16):
            radius, z0, z1 = SEG_COLLIDERS[i]
            n_ang = int(np.ceil(2.0 * np.pi * radius / spacing))
            ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
            zs = np.linspace(z0, z1, int(np.ceil((z1 - z0) / spacing)) + 1)
            side = np.stack(np.broadcast_arrays(radius * np.cos(ang)[None], radius * np.sin(ang)[None], zs[:, None]), -1)
            caps = []
            for rr in np.linspace(0.0, radius, int(np.ceil(radius / spacing)) + 1)[:-1]:
                n = max(1, int(np.ceil(2.0 * np.pi * rr / spacing)))
                a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                for zc in (z0, z1):
                    caps.append(np.stack([rr * np.cos(a), rr * np.sin(a), np.full(n, zc)], -1))
            pts.append(np.concatenate([side.reshape(-1, 3)] + caps))
        return pts

    @staticmethod
    def stave_boxes(bore_length, pipe_radius=PLATE_HOLE_RADIUS, wall=PIPE_WALL):
        """Stave boxes in the pipe frame (origin = centre of the top opening): centres, rotations, half sizes."""
        half_width = (pipe_radius + wall) * np.tan(np.pi / N_STAVES)
        ang = 2.0 * np.pi * np.arange(N_STAVES) / N_STAVES
        centres = np.stack([(pipe_radius + 0.5 * wall) * np.cos(ang), (pipe_radius + 0.5 * wall) * np.sin(ang),
                            np.full(N_STAVES, -0.5 * bore_length)], -1)
        rots = axis_angle_to_mat([0.0, 0.0, 1.0], ang)
        half = np.array([0.5 * wall, half_width, 0.5 * bore_length])
        return centres, rots, half

    def pipe_overlap(self, xpos, xmat, pipe_xy, z_top, bore_length, pipe_radius=PLATE_HOLE_RADIUS, wall=PIPE_WALL):
        """
        True if any collision cylinder of the robot overlaps a pipe stave (MuJoCo would report
        ncon > 0). Surface samples of each shape are tested against the other shape's volume.
        """
        centre = np.array([pipe_xy[0], pipe_xy[1], z_top])
        centres, rots, half = self.stave_boxes(bore_length, pipe_radius, wall)
        r_max = (pipe_radius + wall) / np.cos(np.pi / N_STAVES)
        # 1) cylinder samples inside a stave box
        for k, bid in enumerate(self.seg_ids):
            radius = SEG_COLLIDERS[k + 1][0]
            origin_local = xpos[bid] - centre
            # cheap reject: the whole cylinder is far from the pipe wall
            if origin_local[2] > 0.06 or origin_local[2] < -bore_length - 0.06:
                continue
            if np.linalg.norm(origin_local[:2]) > r_max + radius + 0.06:
                continue
            p = self._collider_points[k] @ xmat[bid].T + origin_local
            near = (p[:, 2] <= 0.0) & (p[:, 2] >= -bore_length)
            rad = np.linalg.norm(p[:, :2], axis=1)
            near &= (rad >= pipe_radius - 1e-9) & (rad <= r_max)
            if not near.any():
                continue
            p = p[near]
            local = np.einsum("kji,nkj->nki", rots, p[:, None, :] - centres[None])
            if np.any(np.all(np.abs(local) <= half, axis=-1)):
                return True
        # 2) stave samples inside a cylinder (catches box edges poking into a cylinder)
        stave_pts = self._stave_points(bore_length, pipe_radius, wall) + centre
        for k, bid in enumerate(self.seg_ids):
            radius, z0, z1 = SEG_COLLIDERS[k + 1]
            local = (stave_pts - xpos[bid]) @ xmat[bid]
            inside = (local[:, 2] >= z0) & (local[:, 2] <= z1) & (local[:, 0] ** 2 + local[:, 1] ** 2 <= radius ** 2)
            if inside.any():
                return True
        return False

    def _stave_points(self, bore_length, pipe_radius, wall, spacing=0.0015):
        key = (round(bore_length, 6), pipe_radius, wall)
        cache = getattr(self, "_stave_cache", {})
        if key not in cache:
            centres, rots, half = self.stave_boxes(bore_length, pipe_radius, wall)
            axes = [np.linspace(-h, h, max(2, int(np.ceil(2 * h / spacing)) + 1)) for h in half]
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
            on_surface = np.any(np.isclose(np.abs(grid), half, atol=1e-12), axis=1)
            box_pts = grid[on_surface]
            cache[key] = np.concatenate([box_pts @ rots[i].T + centres[i] for i in range(N_STAVES)])
            self._stave_cache = cache
        return cache[key]


def z_success_of(z_top, bore_length):
    return z_top - bore_length - EXIT_MARGIN
