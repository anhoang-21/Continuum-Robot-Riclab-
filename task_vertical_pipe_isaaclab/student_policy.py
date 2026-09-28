"""
==============================================================================
student_policy.py - Image-based insertion policy (the "student")
==============================================================================
Input: the tip camera (RGB + depth, 64 x 64) and the robot's own commands
(bend vectors 6, elevator 1, previous action 7). No pipe pose. Output: the 7-D
action of the task (bend-vector rates + elevator rate).

Trained by DAgger (train_student.py) to reproduce the privileged PPO policy (the
"teacher", which reads the true pipe pose) from the states the student itself visits.
==============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdp import ELEV_HALF, ELEV_MID

IMG = 64
DEPTH_MAX = 0.30          # m; farther (or nothing) = 1
PROPRIO_DIM = 14
ACT_DIM = 7


def student_inputs(env):
    """
    Student observation of every env of a VisionPipeEnv:
    img (N, 4, IMG, IMG) uint8 (RGB, depth / DEPTH_MAX), proprio (N, 14) float32.
    """
    rgb, depth, *_ = env.tip_camera()
    d = torch.where(depth > 0.0, depth.clamp(max=DEPTH_MAX), torch.full_like(depth, DEPTH_MAX)) / DEPTH_MAX
    x = torch.cat([rgb.permute(0, 3, 1, 2), d[:, None]], dim=1)
    x = F.interpolate(x, size=(IMG, IMG), mode="area")
    img = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    n = env.num_envs
    proprio = torch.cat([env.bend_cmd.reshape(n, 6), ((env.elev_cmd - ELEV_MID) / ELEV_HALF)[:, None],
                         env.prev_action], dim=1).float()
    return img, proprio


def random_shift(img, pad=4):
    """DrQ-style augmentation: pad by replication, crop back at a random offset (per sample)."""
    n, c, h, w = img.shape
    x = F.pad(img, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=img.device)[:h]
    ar = ar[:, None].repeat(1, h)[:, :, None]
    base = torch.cat([ar.transpose(1, 0), ar], dim=2)[None].repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=img.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class StudentPolicy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(                          # 6 x 64 x 64 (RGB-D + pixel coordinates)
            nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.ELU(),      # 32 x 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),     # 16 x 16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(),     # 8 x 8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ELU(),    # 4 x 4
            nn.Flatten(), nn.Linear(128 * 16, hidden), nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + PROPRIO_DIM, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, ACT_DIM),
        )
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, IMG), torch.linspace(-1, 1, IMG), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None], persistent=False)

    def forward(self, img, proprio, augment=False):
        x = img.float() / 255.0
        if augment:
            x = random_shift(x)
        x = torch.cat([(x - 0.5) / 0.5, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        return self.head(torch.cat([self.conv(x), proprio], dim=1))

    def act(self, img, proprio):
        return self.forward(img, proprio).clamp(-1.0, 1.0)


def save_student(policy, path, **meta):
    torch.save({"state_dict": policy.state_dict(), "meta": meta}, path)


def load_student(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = StudentPolicy().to(device)
    policy.load_state_dict(ckpt["state_dict"])
    return policy.eval()
