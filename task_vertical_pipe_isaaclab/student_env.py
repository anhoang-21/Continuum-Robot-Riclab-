"""
==============================================================================
student_env.py - VisionPipeEnv whose resets are the student's training starts
==============================================================================
Resets use preinsert.StudentStartSampler (pre-insertion poses from a noisy pipe
estimate + normal starts with the pipe in view) instead of the task's own start
sampling. The policy observation stays the privileged one (obs_source="gt"): it is
what the teacher reads; the student reads the tip camera (student_policy.py).
==============================================================================
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from preinsert import StudentStartSampler
from vision_env import VisionPipeEnv, VisionPipeEnvCfg


@configclass
class StudentPipeEnvCfg(VisionPipeEnvCfg):
    obs_source: str = "gt"               # the teacher reads the true pose; the student reads the images
    max_episode_steps: int = 400
    p_normal: float = 0.25               # share of normal task starts (pipe already in view)


class StudentPipeEnv(VisionPipeEnv):
    cfg: StudentPipeEnvCfg

    def __init__(self, cfg: StudentPipeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._student_samplers = {m: StudentStartSampler(self.kin, m, cfg.pipe_height, cfg.random_offset,
                                                         p_normal=cfg.p_normal)
                                  for m in set(self.env_modes)}
        self.student_starts = True

    def _reset_idx(self, env_ids):
        if not getattr(self, "student_starts", False):
            return super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # DirectRLEnv bookkeeping only (VerticalPipeEnv._reset_idx would sample the task's own starts)
        DirectRLEnv._reset_idx(self, env_ids)
        ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        k = len(ids)
        rngs = self._eval_rngs if getattr(self, "_eval", False) else self._rngs
        pipe_xy, z_top, bend, elev = np.zeros((k, 2)), np.zeros(k), np.zeros((k, 3, 2)), np.zeros(k)
        for j, i in enumerate(ids):
            st = self._student_samplers[self.env_modes[i]].sample(rngs[i])
            pipe_xy[j], z_top[j], bend[j], elev[j] = st.pipe_xy, st.z_top, st.bend, st.elev
        self.set_start_state(ids, pipe_xy, z_top, bend, elev)
