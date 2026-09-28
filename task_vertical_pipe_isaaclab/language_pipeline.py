"""
==============================================================================
language_pipeline.py - Instruction -> local VLM -> pipe -> image-based insertion (Stage 5)
==============================================================================
Per-env state machine on MultiPipeEnv (hierarchical vision-language-action):

  LOCATE    hold still until the overview camera shows the new scene, then
            1. the local VLM (vlm.py, Qwen3-VL-2B) reads the instruction and the overview
               image and answers with a box around the pipe it means,
            2. every pipe mouth in the overview image is located in 3D (depth + circle fit),
               the one the box points at is the target, the others are obstacles
  APPROACH  up, bend, down to the pre-insertion pose of the target (clear of the other pipes)
  INSERT    the image-based student (student_policy.py) threads the pipe
  GIVE_UP   the VLM gave no usable box -> "not_grounded"

Grounding is checked against the scene's ground truth: right pipe, wrong pipe, or none.
==============================================================================
"""

from __future__ import annotations

import numpy as np

from perception import PipeDetector, detect_all
from vision_pipeline import GIVE_UP, SETTLE_STEPS, SearchInsertController
from vlm import select_in_box

LOCATE = 8
PHASE_NAMES = ("LOOK", "LIFT", "SCAN", "RETURN", "REFINE", "INSERT", "GIVE_UP", "APPROACH", "LOCATE")


class LanguageInsertController(SearchInsertController):
    def __init__(self, env, student, ground_fn):
        """ground_fn(image uint8 (H, W, 3), instruction) -> (box [x1, y1, x2, y2] or None, answer text, seconds)."""
        self.ground_fn = ground_fn
        self.ov_det = PipeDetector(max_range=1.5, tube_check=False)
        n = env.num_envs
        self.box = [None] * n
        self.answer = [""] * n
        self.vlm_seconds = np.zeros(n)
        self.grounding = [""] * n                          # "right" | "wrong" | "none"
        super().__init__(env, actor=None, student=student)

    def reset(self, ids):
        super().reset(ids)
        for i in ids:
            self.phase[i] = LOCATE
            self.box[i], self.answer[i], self.grounding[i] = None, "", ""
            self.vlm_seconds[i] = 0.0
            self.obstacles[i] = []

    def step_actions(self, obs):
        """LOCATE envs hold still (the base class gives them no action) until the overview frame is fresh."""
        env = self.env
        ready = []
        for i in range(env.num_envs):
            if self.phase[i] == LOCATE:
                self.timer[i] += 1
                if self.timer[i] >= SETTLE_STEPS:
                    ready.append(i)
        if ready:
            rgb, depth, K, eye, rot = env.overview_camera()
            rgb, depth = rgb.cpu().numpy(), depth.cpu().numpy()
            for i in ready:
                self._locate(i, rgb[i], depth[i], K, eye, rot)
        return super().step_actions(obs)

    def _locate(self, i, rgb, depth, K, eye, rot):
        env, sc = self.env, self.env.scenes[i]
        box, text, sec = self.ground_fn((rgb * 255).astype(np.uint8), sc.instruction)
        self.box[i], self.answer[i], self.vlm_seconds[i] = box, text, sec
        if box is None:
            self.grounding[i], self.phase[i] = "none", GIVE_UP
            return
        pipes = detect_all(self.ov_det, rgb, depth, K, eye, rot)
        k = select_in_box(pipes, box, lambda p: self._project(K, eye, rot, p))
        if k is None:
            self.grounding[i], self.phase[i] = "none", GIVE_UP
            return
        target = pipes[k]
        self.obstacles[i] = [(d.pipe_xy, d.z_top, float(env.bore_length[i])) for d in pipes
                             if np.linalg.norm(d.pipe_xy - target.pipe_xy) > 0.05]
        truth = [np.linalg.norm(sc.pipe_xy[j] - target.pipe_xy) for j in range(len(sc.colors))]
        self.grounding[i] = "right" if int(np.argmin(truth)) == sc.target and min(truth) < 0.03 else "wrong"
        self._accept(i, target, "vlm")
        self._found(i)

    @staticmethod
    def _project(K, eye, rot, p):
        pc = (p - eye) @ rot
        return np.array([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])

    def estimate_error(self, i):
        """Error of the located pipe vs the pipe the VLM should have chosen (the target)."""
        if self.grounding[i] not in ("right", "wrong"):
            return float("nan"), float("nan")
        return super().estimate_error(i)
