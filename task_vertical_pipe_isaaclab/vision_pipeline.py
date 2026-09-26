"""
==============================================================================
vision_pipeline.py - Look / search with the tip camera, then insert with the RL policy
==============================================================================
Per-env state machine that drives VisionPipeEnv through the same 7-D action as the
policy (bend-vector rates of the 3 sections + elevator rate):

  LOOK     hold the start pose a few steps, detect the pipe mouth; a well-covered
           collar -> INSERT directly
  LIFT     raise the elevator to the top (wider view), look again
  SCAN     tilt the tip camera (sections 2 + 3 bent the same way) and turn the bend
           direction around the robot axis (conical scan); when the collar count
           peaks, stop, let the servos settle, measure
  RETURN   bend back to the start bend, then lower the elevator to the start height:
           the policy starts from the pose distribution it was trained on, and the
           path back is the path out (collision-free)
  REFINE   at the start pose, measure again if the collar is visible (closer view)
  INSERT   the RL policy, whose observation uses the estimated pipe pose
  GIVE_UP  nothing found after the scan -> "not_found"

The raise / tilt only move the tip up and away from the pipe mouth (the start pose
has the tip above the mouth), so the search itself does not touch the pipe.
==============================================================================
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

import constants as C
from perception import Detection, PipeDetector

LOOK, LIFT, SCAN, RETURN, REFINE, INSERT, GIVE_UP = range(7)
PHASE_NAMES = ("LOOK", "LIFT", "SCAN", "RETURN", "REFINE", "INSERT", "GIVE_UP")

SETTLE_STEPS = 4                   # steps without motion before a measurement (servo lag, render latency)
LIFT_ELEV = C.ELEV_RANGE[1] - 0.002
SCAN_TILT = (0.25, 0.35)           # bend (rad) of sections 2 and 3 while scanning: tip camera tilted ~0.6 rad
SCAN_TURNS = 1.1                   # turns of the bend direction before giving up
GOOD_START_COVERAGE = 0.5          # collar coverage that makes the start-pose view good enough
PEAK_DROP = 0.8                    # stop scanning when the collar count falls below this x its peak
FULL_VIEW_POINTS = 700             # ... or when this many collar pixels are in view
SCAN_COOLDOWN = 15                 # steps of turning (no stop) after a view that gave no valid measurement
WORKSPACE_RADIUS = 0.30            # a pipe mouth farther than this from the robot axis is not ours (m)


def load_actor(path):
    """Deterministic rsl_rl actor with the architecture of agent_cfg.py."""
    from rsl_rl.models import MLPModel

    from agent_cfg import make_agent_cfg

    cfg = dict(make_agent_cfg(1, 1)["actor"])
    cfg.pop("class_name")
    cfg["distribution_cfg"] = dict(cfg["distribution_cfg"])
    actor = MLPModel(TensorDict({"policy": torch.zeros(1, 33)}, batch_size=[1]),
                     {"actor": ["policy"], "critic": ["policy"]}, "actor", 7, **cfg)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"])
    return actor.eval()


class SearchInsertController:
    def __init__(self, env, actor, detector: PipeDetector | None = None, use_vision=True):
        """
        use_vision=False: baseline, the policy gets the true pipe pose and starts right away
        (env.cfg.obs_source must be "gt").
        """
        self.env, self.actor = env, actor
        self.det = detector or PipeDetector()
        self.use_vision = use_vision
        n = env.num_envs
        self.phase = np.full(n, LOOK)
        self.timer = np.zeros(n, dtype=int)
        self.start_bend = np.zeros((n, 3, 2))
        self.start_elev = np.zeros(n)
        self.phi0 = np.zeros(n)
        self.phi = np.zeros(n)
        self.peak = np.zeros(n, dtype=int)
        self.cooldown = np.zeros(n, dtype=int)
        self.search_steps = np.zeros(n, dtype=int)
        self.found_by = [""] * n
        self.last_det: list[Detection] = [Detection.none() for _ in range(n)]
        self.bend_rate = np.asarray(C.BEND_RATE)
        self.reset(range(n))

    # ------------------------------------------------------------------
    def reset(self, ids):
        """Call after the env reset these envs (the start pose is read from the env)."""
        bend = self.env.bend_cmd.cpu().numpy()
        elev = self.env.elev_cmd.cpu().numpy()
        for i in ids:
            self.start_bend[i], self.start_elev[i] = bend[i], elev[i]
            self.phase[i] = LOOK if self.use_vision else INSERT
            self.timer[i] = 0
            self.peak[i] = 0
            self.cooldown[i] = 0
            self.search_steps[i] = 0
            self.found_by[i] = "" if self.use_vision else "gt"
            self.last_det[i] = Detection.none()
            # start the scan in the direction the start pose already leans to (any direction works)
            lean = bend[i].sum(axis=0)
            self.phi0[i] = self.phi[i] = float(np.arctan2(lean[1], lean[0]))
        if not self.use_vision:
            self.env.clear_prev_action(list(ids))

    # ------------------------------------------------------------------
    def _track(self, i, bend_target, elev_target):
        """7-D action driving the commands of env i towards the targets (saturated P control on the commands)."""
        bend = self.env.bend_cmd[i].cpu().numpy()
        elev = float(self.env.elev_cmd[i])
        a = np.zeros(7)
        a[:6] = np.clip((bend_target - bend) / self.bend_rate[:, None], -1.0, 1.0).reshape(6)
        a[6] = np.clip((elev_target - elev) / C.ELEV_RATE, -1.0, 1.0)
        done = np.abs(bend_target - bend).max() < 1e-4 and abs(elev_target - elev) < 1e-5
        return a, done

    def _scan_bend(self, i):
        b = self.start_bend[i].copy()
        d = np.array([np.cos(self.phi[i]), np.sin(self.phi[i])])
        b[1] = SCAN_TILT[0] * d
        b[2] = SCAN_TILT[1] * d
        return b

    def _plausible(self, det):
        """Workspace prior: the mouth is within reach of the robot and between the plate and the tip range."""
        return (det.valid and np.linalg.norm(det.pipe_xy - self.env.kin.axis_xy) < WORKSPACE_RADIUS
                and C.PLATE_TOP_Z < det.z_top < self.env.kin.tip_z0 + C.ELEV_RANGE[1])

    def _accept(self, i, det, how):
        self.last_det[i] = det
        self.env.set_estimate([i], det.pipe_xy[None], [det.z_top])
        self.found_by[i] = how

    # ------------------------------------------------------------------
    def step_actions(self, obs):
        """Actions (N, 7) for the next env step; the images are the ones of the last env step."""
        env, n = self.env, self.env.num_envs
        actions = np.zeros((n, 7))
        self._refresh = False
        need_img = self.use_vision and np.isin(self.phase, (LOOK, LIFT, SCAN, REFINE)).any()
        if need_img:
            rgb, depth, K, cam_pos, cam_rot = (t.cpu().numpy() for t in env.tip_camera())

        def detect(i):
            det = self.det.detect(rgb[i], depth[i], K[i], cam_pos[i], cam_rot[i])
            det.valid = self._plausible(det)
            return det

        for i in range(n):
            ph = self.phase[i]
            if ph != INSERT:
                self.search_steps[i] += 1
            if ph == LOOK:
                self.timer[i] += 1
                if self.timer[i] >= SETTLE_STEPS:
                    det = detect(i)
                    if det.valid and det.coverage >= GOOD_START_COVERAGE:
                        self._accept(i, det, "start")
                        self._start_insert(i)
                    else:
                        if det.valid:
                            self._accept(i, det, "start (partial)")
                        self.phase[i], self.timer[i] = LIFT, 0
            elif ph == LIFT:
                a, done = self._track(i, self.start_bend[i], LIFT_ELEV)
                actions[i] = a
                if done:
                    self.timer[i] += 1
                    if self.timer[i] >= SETTLE_STEPS:
                        det = detect(i)
                        if det.valid and det.coverage >= GOOD_START_COVERAGE:
                            self._accept(i, det, "lift")
                            self.phase[i], self.timer[i] = RETURN, 0
                        else:
                            self.phase[i], self.timer[i], self.peak[i] = SCAN, 0, 0
            elif ph == SCAN:
                if self.phi[i] - self.phi0[i] > SCAN_TURNS * 2.0 * np.pi:
                    if self.found_by[i]:                    # partial view from the start pose: use it
                        self.phase[i], self.timer[i] = RETURN, 0
                    else:
                        self.phase[i] = GIVE_UP
                    continue
                count = self.det.count(rgb[i], depth[i])
                if self.timer[i] > 0:                       # stopped at a view, settling
                    self.timer[i] += 1
                    if self.timer[i] > SETTLE_STEPS:
                        det = detect(i)
                        if det.valid:
                            self._accept(i, det, "scan")
                            self.phase[i], self.timer[i] = RETURN, 0
                        else:                               # keep turning past this view
                            self.timer[i], self.peak[i], self.cooldown[i] = 0, 0, SCAN_COOLDOWN
                    continue
                if self.cooldown[i] > 0:
                    self.cooldown[i] -= 1
                else:
                    self.peak[i] = max(self.peak[i], count)
                    full = count >= FULL_VIEW_POINTS
                    if self.peak[i] >= self.det.min_points and (count < PEAK_DROP * self.peak[i] or full):
                        self.timer[i] = 1                   # stop here and settle
                        continue
                target = self._scan_bend(i)
                a, _ = self._track(i, target, LIFT_ELEV)
                actions[i] = a
                err = np.abs(target - env.bend_cmd[i].cpu().numpy()).max()
                if err < 0.01:                              # on the cone: turn further
                    self.phi[i] += 0.9 * min(self.bend_rate[1] / SCAN_TILT[0], self.bend_rate[2] / SCAN_TILT[1])
            elif ph == RETURN:
                bend = env.bend_cmd[i].cpu().numpy()
                if np.abs(self.start_bend[i] - bend).max() >= 1e-4:
                    a, _ = self._track(i, self.start_bend[i], float(env.elev_cmd[i]))   # bend back first
                else:
                    a, done = self._track(i, self.start_bend[i], self.start_elev[i])    # then down
                    if done:
                        self.phase[i], self.timer[i] = REFINE, 0
                actions[i] = a
            elif ph == REFINE:
                self.timer[i] += 1
                if self.timer[i] >= SETTLE_STEPS:
                    det = detect(i)
                    if det.valid and det.coverage >= max(self.last_det[i].coverage, 0.3):
                        self._accept(i, det, self.found_by[i] + "+refine")
                    self._start_insert(i)

        ins = np.nonzero(self.phase == INSERT)[0]
        act = torch.as_tensor(actions, dtype=torch.float32, device=env.device)
        if len(ins):
            if self._refresh:                               # estimates accepted this step: new observation
                obs = env._get_observations()
                env.obs_buf = obs
            policy_act = self.actor(TensorDict({"policy": obs["policy"].to("cpu")}, batch_size=[n])).to(env.device)
            idx = torch.as_tensor(ins, device=env.device)
            act[idx] = policy_act[idx]
        return act

    def _start_insert(self, i):
        self.phase[i], self.timer[i] = INSERT, 0
        self.env.clear_prev_action([i])
        self._refresh = True

    # ------------------------------------------------------------------
    def estimate_error(self, i):
        """(xy error, z error) of the accepted estimate of env i in m (nan if none)."""
        if not self.found_by[i] or self.found_by[i] == "gt":
            return float("nan"), float("nan")
        d = self.last_det[i]
        return (float(np.linalg.norm(d.pipe_xy - self.env.pipe_xy[i].cpu().numpy())),
                float(d.z_top - float(self.env.z_top[i])))
