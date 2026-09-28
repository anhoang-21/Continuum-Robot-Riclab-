"""
==============================================================================
eval_grounding.py - How well the local VLM picks the instructed pipe (Stage 5)
==============================================================================
    D:\Isaacsim\env_isaaclab\Scripts\python.exe eval_grounding.py            # data/grounding, GPU
    D:\Isaacsim\env_isaaclab\Scripts\python.exe eval_grounding.py --device cpu --limit 50

For every scene of make_grounding_set.py: Qwen3-VL (2B or 4B) gets the overview image and the
instruction and answers with a box. Scored:
  box      the box contains the target's mouth (projected with the known camera) and no
           other mouth is nearer to the box centre
  3D       of the pipe mouths located in the image (depth + circle fit), the one the box
           points at is within 3 cm of the target
Results per instruction kind (colour / position) and per colour, answers in results.jsonl.
No Isaac Sim needed.
==============================================================================
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from multi_pipe import OverviewCamera  # noqa: E402
from perception import PipeDetector, detect_all  # noqa: E402
from vlm import PROMPTS, QwenGrounder, select_in_box  # noqa: E402

parser = argparse.ArgumentParser(description="VLM grounding accuracy on the multi-pipe overview images")
parser.add_argument("--data", default=os.path.join(TASK_DIR, "data", "grounding"))
parser.add_argument("--device", default="cuda")
parser.add_argument("--model", default="2b", help="2b | 4b | a Hugging Face id")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--prompt", default="v1", choices=["v1", "v2"])
parser.add_argument("--kind", default="", help="only this instruction kind (color / position)")
parser.add_argument("--tag", default="", help="suffix of the results file")
args = parser.parse_args()


def main():
    cam = OverviewCamera()
    det = PipeDetector(max_range=1.5, tube_check=False)
    rows = [json.loads(line) for line in open(os.path.join(args.data, "index.jsonl"), encoding="utf-8")]
    if args.kind:
        rows = [r for r in rows if r["kind"] == args.kind]
    if args.limit:
        rows = rows[:args.limit]
    grounder = QwenGrounder(args.model, device=args.device, prompt=PROMPTS[args.prompt])
    print(f"model loaded in {grounder.load_s:.1f} s on {args.device}; {len(rows)} scenes", flush=True)
    out = open(os.path.join(args.data, f"results{args.tag}.jsonl"), "w", encoding="utf-8")
    stats = {}
    t0 = time.time()
    for n, r in enumerate(rows):
        rgb = np.array(Image.open(os.path.join(args.data, f"{r['id']:04d}.png")).convert("RGB"))
        depth = np.load(os.path.join(args.data, f"{r['id']:04d}_depth.npz"))["depth"].astype(np.float32)
        box, text, sec = grounder.ground(rgb, r["instruction"])
        xy, z = np.array(r["pipe_xy"]), np.array(r["z_top"])
        mouths = cam.project(np.column_stack([xy, z]))
        ok_box, ok_3d, err = False, False, float("nan")
        if box is not None:
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            nearest = int(np.argmin(np.linalg.norm(mouths - c, axis=1)))
            inside = box[0] <= mouths[r["target"], 0] <= box[2] and box[1] <= mouths[r["target"], 1] <= box[3]
            ok_box = bool(inside and nearest == r["target"])
            pipes = detect_all(det, rgb.astype(np.float32) / 255.0, depth, cam.K, cam.EYE, cam.rot)
            k = select_in_box(pipes, box, cam.project)
            if k is not None:
                err = float(np.linalg.norm(pipes[k].pipe_xy - xy[r["target"]]))
                ok_3d = err < 0.03
        keys = ["all", r["kind"]] + ([f"color:{r['colors'][r['target']]}"] if r["kind"] == "color" else [])
        for key in keys:
            s = stats.setdefault(key, [0, 0, 0])
            s[0] += 1
            s[1] += ok_box
            s[2] += ok_3d
        out.write(json.dumps({**r, "answer": text, "box": None if box is None else box.tolist(), "ok_box": ok_box,
                              "ok_3d": ok_3d, "err_m": err, "seconds": sec}) + "\n")
        if (n + 1) % 25 == 0:
            s = stats["all"]
            print(f"{n + 1:4d}/{len(rows)}  box {100 * s[1] / s[0]:5.1f}%  3D {100 * s[2] / s[0]:5.1f}%  "
                  f"({(time.time() - t0) / (n + 1):.2f} s per scene)", flush=True)
    out.close()
    print(f"\n{'':18s} {'n':>4s} {'box':>7s} {'3D':>7s}")
    for key in sorted(stats, key=lambda k: (k != "all", k.startswith("color:"), k)):
        s = stats[key]
        print(f"{key:18s} {s[0]:4d} {100 * s[1] / s[0]:6.1f}% {100 * s[2] / s[0]:6.1f}%")
    print(f"mean {(time.time() - t0) / len(rows):.2f} s per scene on {args.device}")


if __name__ == "__main__":
    main()
