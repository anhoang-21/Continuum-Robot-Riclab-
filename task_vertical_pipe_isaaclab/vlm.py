"""
==============================================================================
vlm.py - Language grounding with a local vision-language model (Qwen3-VL)
==============================================================================
The instruction ("go through the red pipe", "the pipe on the far left", ...) and the
overview camera image go to Qwen3-VL-2B-Instruct (or 4B; Apache-2.0, runs locally through
transformers, no API / tokens); it answers with a bounding box of the pipe it means.
The box is turned into a 3D pipe estimate with the overview depth image: the collar
pixels inside the box -> perception.PipeDetector (same circle fit as the tip camera).

  QwenGrounder         the model (GPU or CPU), ground(image, instruction) -> box
  parse_boxes()        JSON / number parsing of the answer (Qwen3-VL: 0-1000 coordinates)
  select_in_box()      box -> which of the located pipe mouths it points at
  locate_in_box()      box + RGB-D overview frame -> Detection of the pipe mouth inside it
  serve()              the model behind a small HTTP server (python vlm.py --serve), so a
                       simulation process can ask it without holding the model itself
  ground_http()        client of that server
==============================================================================
"""

from __future__ import annotations

import base64
import io
import json
import re
import os
import time

import numpy as np

MODELS = {"2b": "Qwen/Qwen3-VL-2B-Instruct", "4b": "Qwen/Qwen3-VL-4B-Instruct"}
MODEL_ID = MODELS["2b"]
# downloaded models live under $HF_HOME/hub (default D:/hf_models, see README)
CACHE_DIR = os.path.join(os.environ.get("HF_HOME", "D:/hf_models"), "hub")
PROMPTS = {
    "v1": ("You see several pipes standing on a table; each has a yellow rim at its top opening. "
           "Instruction: \"{instruction}\"\n"
           "Which pipe does the instruction refer to? Output the bounding box of that one pipe as JSON: "
           "[{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"target pipe\"}}]"),
    "v2": ("The image shows three pipes on a table, seen from a camera in front of them; each pipe has a "
           "yellow rim at its top opening and a coloured body. Left and right mean left and right in this "
           "image. The pipe closest to the camera (the front pipe) is the one lowest in the image; the pipe "
           "farthest from the camera (at the back) is the one highest in the image.\n"
           "Instruction: \"{instruction}\"\n"
           "Output the bounding box of the one pipe the instruction refers to as JSON: "
           "[{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"target pipe\"}}]"),
}
PROMPT = PROMPTS["v1"]


def parse_boxes(text, width, height):
    """All [x1, y1, x2, y2] boxes in the answer, in pixels (Qwen3-VL gives 0-1000 relative coordinates)."""
    boxes = []
    for m in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
                         text):
        b = np.array([float(v) for v in m.groups()])
        if b.max() <= 1000.0:
            b = b / 1000.0 * np.array([width, height, width, height])
        x1, x2 = sorted((b[0], b[2]))
        y1, y2 = sorted((b[1], b[3]))
        boxes.append(np.array([x1, y1, x2, y2]))
    return boxes


class QwenGrounder:
    def __init__(self, model_id=MODEL_ID, device="cuda", dtype="bfloat16", prompt=PROMPT):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.time()
        self.device, self.prompt = device, prompt
        model_id = MODELS.get(model_id, model_id)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                                                     cache_dir=CACHE_DIR)
        self.model.to(device).eval()
        self.load_s = time.time() - t0

    def ground(self, image, instruction, max_new_tokens=96):
        """image: (H, W, 3) uint8 RGB -> (box in pixels or None, raw answer, seconds)."""
        import torch
        from PIL import Image

        t0 = time.time()
        pil = Image.fromarray(image)
        messages = [{"role": "user", "content": [{"type": "image", "image": pil},
                                                 {"type": "text", "text": self.prompt.format(instruction=instruction)}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        boxes = parse_boxes(text, pil.width, pil.height)
        return (boxes[0] if boxes else None), text, time.time() - t0


def select_in_box(pipes, box, project):
    """
    The pipe the box points at, out of all pipes located in the image (perception.detect_all):
    the one whose mouth projects inside the box and nearest to its centre (else nearest to the centre).
    project: env-frame point (3,) -> pixel (2,). Returns (index or None).
    """
    if not pipes:
        return None
    c = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    pix = [project(np.array([d.pipe_xy[0], d.pipe_xy[1], d.z_top])) for d in pipes]
    inside = [k for k, p in enumerate(pix) if box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]]
    cand = inside or range(len(pipes))
    return min(cand, key=lambda k: np.linalg.norm(pix[k] - c))


def locate_in_box(detector, rgb, depth, K, cam_pos, cam_rot, box, grow=0.15):
    """Pipe mouth inside the (slightly enlarged) box of the overview image -> perception.Detection."""
    h, w = depth.shape
    x1, y1, x2, y2 = box
    dx, dy = grow * (x2 - x1), grow * (y2 - y1)
    roi = np.zeros((h, w), dtype=bool)
    roi[max(int(y1 - dy), 0):min(int(y2 + dy) + 1, h), max(int(x1 - dx), 0):min(int(x2 + dx) + 1, w)] = True
    return detector.detect(rgb, np.where(roi, depth, 0.0), K, cam_pos, cam_rot)


# ---------------------------------------------------------------------------
# HTTP server / client (the simulation and the model run in separate processes)
# ---------------------------------------------------------------------------
def _encode(image):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ground_http(image, instruction, url="http://127.0.0.1:8765/ground", timeout=300):
    import urllib.request

    body = json.dumps({"image": _encode(image), "instruction": instruction}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ans = json.loads(r.read())
    box = np.array(ans["box"]) if ans["box"] is not None else None
    return box, ans["text"], ans["seconds"]


def serve(port=8765, device="cuda", model=MODEL_ID):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    grounder = QwenGrounder(model, device=device)
    print(f"{grounder.model_id} on {device}, loaded in {grounder.load_s:.1f} s; http://127.0.0.1:{port}/ground", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            image = np.array(Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB"))
            box, text, sec = grounder.ground(image, req["instruction"])
            print(f"[{sec:4.1f} s] {req['instruction']!r} -> {text.strip()!r}", flush=True)
            data = json.dumps({"box": None if box is None else box.tolist(), "text": text, "seconds": sec}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Local VLM grounding server (Qwen3-VL)")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="2b", help="2b | 4b | a Hugging Face id")
    a = p.parse_args()
    if a.serve:
        serve(a.port, a.device, a.model)
