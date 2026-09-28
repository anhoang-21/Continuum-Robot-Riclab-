"""
==============================================================================
make_grounding_set.py - Overview images + instructions with ground truth (Stage 5)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe make_grounding_set.py --scenes 300

Renders multi-pipe scenes (multi_pipe_env.py, CAD rig in view) and saves, per scene:
data/grounding/<k>.png (overview RGB), <k>_depth.npz (overview depth, float16) and one line
in index.jsonl (instruction, kind, pipe poses, colours, target). eval_grounding.py runs the
VLM on them without Isaac Sim.
==============================================================================
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Overview images with instructions and ground truth")
parser.add_argument("--scenes", type=int, default=300)
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=500)
parser.add_argument("--out", default=os.path.join(TASK_DIR, "data", "grounding"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

from PIL import Image  # noqa: E402

from multi_pipe_env import MultiPipeEnv, MultiPipeEnvCfg  # noqa: E402


def main():
    os.makedirs(args.out, exist_ok=True)
    cfg = MultiPipeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    env = MultiPipeEnv(cfg)
    env.auto_reset = False
    env.reset()
    index = open(os.path.join(args.out, "index.jsonl"), "w", encoding="utf-8")
    k = 0
    zero = torch.zeros(env.num_envs, 7, device=env.device)
    while k < args.scenes:
        for _ in range(3):                                  # render the new scenes
            env.step(zero)
        rgb, depth, *_ = env.overview_camera()
        rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        depth = depth.cpu().numpy().astype(np.float16)
        for i in range(env.num_envs):
            if k >= args.scenes:
                break
            sc = env.scenes[i]
            Image.fromarray(rgb[i]).save(os.path.join(args.out, f"{k:04d}.png"))
            np.savez_compressed(os.path.join(args.out, f"{k:04d}_depth.npz"), depth=depth[i])
            index.write(json.dumps({"id": k, "instruction": sc.instruction, "kind": sc.kind, "target": sc.target,
                                    "colors": sc.colors, "pipe_xy": sc.pipe_xy.tolist(),
                                    "z_top": sc.z_top.tolist()}) + "\n")
            k += 1
        env.reset_envs(range(env.num_envs))
        print(f"{k} scenes", flush=True)
    index.close()
    print(f"-> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
