"""
==============================================================================
mdp.py - Action / observation / reward / termination of the vertical-pipe task
==============================================================================
Batched torch versions of VerticalPipeEnv.step / _measure / _get_obs / _potential
(D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\vertical_pipe_env.py). The functions
only take tensors (body positions, contact points, commands), so the same code runs
inside the Isaac Lab env and in tests/compare_with_mujoco.py, which feeds it MuJoCo
states and checks that observations and rewards match the MuJoCo env.
All task math runs in float64 like the numpy original; observations are float32.
==============================================================================
"""

import math

import torch

from constants import (
    BEND_RATE,
    ELEV_RANGE,
    ELEV_RATE,
    R_FAIL,
    R_SUCCESS,
    R_TIME,
    THETA_MAX,
    TIP_DISC_RADIUS,
    TIP_OFFSET,
    W_HEIGHT,
    W_LAT,
    W_LAT_S3,
    W_SMOOTH,
    W_TILT,
)

OBS_DIM = 33
ACT_DIM = 7
# Seg body indices (0-based in the 15-segment list): tip Seg15, Seg13, Seg11 (start of
# section 3), Seg6 (start of section 2)
TIP, SEG13, SEG11, SEG6 = 14, 12, 10, 5
ELEV_MID = 0.5 * (ELEV_RANGE[0] + ELEV_RANGE[1])
ELEV_HALF = 0.5 * (ELEV_RANGE[1] - ELEV_RANGE[0])

# failure codes (index into constants.FAILURE_NAMES)
FAIL_NONE, FAIL_UNSTABLE, FAIL_RIM, FAIL_MISSED = 0, 1, 2, 3


def update_commands(bend_cmd, elev_cmd, action):
    """
    Integrate one action into the commands, in place.
    bend_cmd (N, 3, 2), elev_cmd (N,), action (N, 7) already clipped to [-1, 1].
    """
    rate = torch.as_tensor(BEND_RATE, dtype=bend_cmd.dtype, device=bend_cmd.device)
    bend_cmd += action[:, :6].to(bend_cmd.dtype).reshape(-1, 3, 2) * rate[None, :, None]
    theta = torch.linalg.norm(bend_cmd, dim=2, keepdim=True)
    bend_cmd *= torch.clamp(THETA_MAX / torch.clamp(theta, min=1e-9), max=1.0)
    elev_cmd.copy_(torch.clamp(elev_cmd + action[:, 6].to(elev_cmd.dtype) * ELEV_RATE, ELEV_RANGE[0], ELEV_RANGE[1]))


def joint_targets(bend_cmd):
    """Bend vectors (N, 3, 2) -> angles of the 15 x-joints and the 15 y-joints (N, 15) each."""
    per_seg = bend_cmd / 5.0
    return (-per_seg[:, :, 1]).repeat_interleave(5, dim=1), per_seg[:, :, 0].repeat_interleave(5, dim=1)


def classify_contacts(points, env_ids, pipe_xy, inner_r, num_envs, count_mask=None, outer_mask=None, pair_ids=None,
                      per_pair=1):
    """
    VerticalPipeEnv._classify_contacts for a flat list of robot-pipe contact points.
    points (M, 3) in the env frame, env_ids (M,) -> (#inner-wall contacts (N,), rim/outside hit (N,)).
    count_mask / outer_mask select which points may count as inner-wall contacts / rim hits
    (MuJoCo: every listed contact penetrates, so both default to all points).
    pair_ids (M,): geom pair of every point. MuJoCo 3.13 returns 1-2 contacts per segment / stave pair
    (1: 69 %, 2: 29 %), PhysX a patch of up to ~30 points, so at most `per_pair` points count per pair
    (1 gives the closest counts in tests/check_isaac_env.py: mean |difference| 0.82 vs 0.93 for 2).
    """
    n_inner = torch.zeros(num_envs, dtype=torch.long, device=pipe_xy.device)
    outer = torch.zeros(num_envs, dtype=torch.bool, device=pipe_xy.device)
    if points.shape[0] == 0:
        return n_inner, outer
    radius = torch.linalg.norm(points[:, :2].to(pipe_xy.dtype) - pipe_xy[env_ids], dim=1)
    inner = radius <= inner_r
    counted = inner if count_mask is None else inner & count_mask
    outside = ~inner if outer_mask is None else (~inner) & outer_mask
    if pair_ids is None:
        n_inner.index_add_(0, env_ids, counted.long())
    elif counted.any():
        pairs, inverse, counts = torch.unique(pair_ids[counted], return_inverse=True, return_counts=True)
        pair_env = torch.zeros_like(pairs)
        pair_env[inverse] = env_ids[counted]
        n_inner.index_add_(0, pair_env, torch.clamp(counts, max=per_pair))
    outer[env_ids[outside]] = True
    return n_inner, outer


def stave_index(points, pipe_xy, n_staves):
    """Index of the pipe stave a contact point belongs to (from its angle around the pipe axis)."""
    d = points[:, :2].to(pipe_xy.dtype) - pipe_xy
    ang = torch.atan2(d[:, 1], d[:, 0])
    return torch.remainder(torch.round(ang / (2.0 * math.pi / n_staves)).long(), n_staves)


def measure(seg_pos, tip_rot, pipe_xy, z_top, z_success, n_inner, outer_hit, pipe_radius):
    """
    VerticalPipeEnv._measure.
    seg_pos (N, 15, 3): Seg1..Seg15 body origins in the env frame; tip_rot (N, 3, 3): Seg15 orientation.
    """
    tip_dir = tip_rot[:, :, 2]
    tip = seg_pos[:, TIP] + TIP_OFFSET * tip_dir
    lat = torch.linalg.norm(tip[:, :2] - pipe_xy, dim=1)
    lat_s3 = torch.linalg.norm(seg_pos[:, SEG11, :2] - pipe_xy, dim=1)
    tilt = torch.arccos(torch.clamp(-tip_dir[:, 2], -1.0, 1.0))
    depth = z_top - tip[:, 2]

    # radial clearance of the section-3 discs already below the rim (inf while nothing is inside)
    s3 = seg_pos[:, SEG11:]
    r = torch.linalg.norm(s3[:, :, :2] - pipe_xy[:, None], dim=2)
    clear = torch.where(s3[:, :, 2] < z_top[:, None], pipe_radius - r - TIP_DISC_RADIUS, torch.full_like(r, math.inf))
    clearance = clear.min(dim=1).values
    clearance = torch.where(depth > 0.0, torch.minimum(clearance, pipe_radius - lat - TIP_DISC_RADIUS), clearance)

    return {
        "tip": tip, "tip_dir": tip_dir, "lat": lat, "lat_s3": lat_s3, "tilt": tilt, "depth": depth,
        "h_left": torch.clamp(tip[:, 2] - z_success, min=0.0),
        "n_inner": n_inner, "outer_hit": outer_hit, "clearance": clearance,
        "seg13": seg_pos[:, SEG13], "seg11": seg_pos[:, SEG11], "seg6": seg_pos[:, SEG6],
    }


def potential(meas):
    return -(W_LAT * meas["lat"] / 0.1 + W_LAT_S3 * meas["lat_s3"] / 0.1
             + W_TILT * meas["tilt"] / 0.5 + W_HEIGHT * meas["h_left"] / 0.1)


def observation(meas, pipe_xy, z_top, z_success, bend_cmd, elev_cmd, axis_xy, prev_action):
    """VerticalPipeEnv._get_obs -> (N, 33) float32."""
    top = torch.cat([pipe_xy, z_top[:, None]], dim=1)

    def rel(p):
        return (p - top) / 0.1

    n = pipe_xy.shape[0]
    obs = torch.cat([
        rel(meas["tip"]),                                          # 3
        meas["tip_dir"],                                           # 3
        rel(meas["seg13"]),                                        # 3  mid section 3 (Seg13)
        rel(meas["seg11"]),                                        # 3  start of section 3 (Seg11)
        rel(meas["seg6"]),                                         # 3  start of section 2 (Seg6)
        bend_cmd.reshape(n, 6),                                    # 6
        ((elev_cmd - ELEV_MID) / ELEV_HALF)[:, None],              # 1
        (pipe_xy - axis_xy) / 0.1,                                 # 2
        ((meas["tip"][:, 2] - z_success) / 0.1)[:, None],          # 1
        (torch.clamp(meas["n_inner"], max=4).to(pipe_xy.dtype) / 4.0)[:, None],  # 1
        prev_action.to(pipe_xy.dtype),                             # 7
    ], dim=1)
    return obs.to(torch.float32)


def outcome(meas, z_success, pipe_radius, unstable):
    """Success / failure flags of VerticalPipeEnv.step: (is_success, failure code)."""
    success = (meas["tip"][:, 2] <= z_success) & (meas["lat"] < pipe_radius)
    missed = (meas["depth"] > 0.003) & (meas["lat"] > pipe_radius)
    failure = torch.full_like(meas["lat"], FAIL_NONE, dtype=torch.long)
    failure = torch.where(missed, torch.full_like(failure, FAIL_MISSED), failure)
    failure = torch.where(meas["outer_hit"], torch.full_like(failure, FAIL_RIM), failure)
    failure = torch.where(unstable, torch.full_like(failure, FAIL_UNSTABLE), failure)
    return success & (failure == FAIL_NONE), failure


def step_reward(meas, prev_potential, action, prev_action, contact_penalty, z_success, pipe_radius, unstable):
    """
    Everything VerticalPipeEnv.step computes after the physics:
    returns reward, potential, terminated, is_success, failure code.
    """
    pot = potential(meas)
    reward = pot - prev_potential
    reward = reward + R_TIME
    reward = reward - contact_penalty * (meas["n_inner"] > 0).to(reward.dtype)
    reward = reward - W_SMOOTH * torch.sum((action.to(reward.dtype) - prev_action.to(reward.dtype)) ** 2, dim=1)

    is_success, failure = outcome(meas, z_success, pipe_radius, unstable)
    failed = failure != FAIL_NONE
    reward = reward + torch.where(is_success, torch.full_like(reward, R_SUCCESS), torch.zeros_like(reward))
    reward = reward + torch.where(failed, torch.full_like(reward, R_FAIL), torch.zeros_like(reward))
    terminated = is_success | failed
    return reward, pot, terminated, is_success, failure


def stage(meas, z_exit):
    """0 = ALIGN, 1 = INSERT, 2 = EXIT (VerticalPipeEnv._stage)."""
    st = torch.where(meas["depth"] > 0.0, torch.ones_like(meas["n_inner"]), torch.zeros_like(meas["n_inner"]))
    return torch.where(meas["tip"][:, 2] <= z_exit, torch.full_like(st, 2), st)
