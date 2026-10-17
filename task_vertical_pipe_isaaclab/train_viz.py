"""
==============================================================================
train_viz.py - a small animated view of one robot for TensorBoard during training
==============================================================================
PipeRunner records one env (the last) during its first evaluation episode (every eval, 50k steps by
default) and logs it as an animated GIF under the tag "eval/robot" (TensorBoard -> IMAGES tab; the
step slider picks the evaluation). The GIF is also written to logs/<run>/viz/.

No camera is rendered: the frames are drawn with matplotlib from the link positions (front,
side and top view of the backbone, the pipe, the plate; cable tensions for the cable-driven
robot), so training does not slow down. Drawing runs in a background thread.
==============================================================================
"""

from __future__ import annotations

import io
import os
import threading

import numpy as np

import constants as C

SECTION_COLORS = ("#e8590c", "#2f9e44", "#1c7ed6")
OUTCOME_TEXT = {0: "THROUGH THE PIPE", 1: "UNSTABLE", 2: "RIM HIT", 3: "MISSED PIPE", 4: "TIME OUT"}


class EpisodeRecorder:
    """Link positions of one env over an episode (env frame, numpy)."""

    def __init__(self, env, env_id=0, every=2):
        self.env, self.i, self.every = env, env_id, every
        self.frames, self.outcome, self._k = [], None, 0
        i = env_id
        self.pipe_xy = env.pipe_xy[i].cpu().numpy().copy()
        self.z_top = float(env.z_top[i])
        self.z_success = float(env.z_success[i])
        self.bore = float(env.bore_length[i])
        self.mode = env.env_modes[i]

    def add(self, step):
        self._k += 1
        if (self._k - 1) % self.every:
            return
        env, i = self.env, self.i
        seg = (env.robot.data.body_link_pos_w.torch[i, env._seg_ids] - env.scene.env_origins[i]).cpu().numpy()
        m = env._meas
        frame = {"step": step, "seg": seg, "tip": m["tip"][i].cpu().numpy(), "depth": float(m["depth"][i]),
                 "lat": float(m["lat"][i]), "contacts": int(m["n_inner"][i])}
        if hasattr(env, "tension"):                                    # cable-driven robot
            frame["tension"] = env.tension[i].cpu().numpy()
            frame["torque"] = env.motors.torque[i].cpu().numpy()
        self.frames.append(frame)

    def finish(self, outcome):
        self.outcome = int(outcome)


def render_gif(rec: EpisodeRecorder, title="", dpi=90):
    """Frames -> animated GIF bytes (front / side / top view)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle, Rectangle
    from PIL import Image

    segs = np.stack([f["seg"] for f in rec.frames])                    # (T, 15, 3)
    tips = np.stack([f["tip"] for f in rec.frames])
    pts = np.concatenate([segs.reshape(-1, 3), tips])
    cx, cy = rec.pipe_xy
    xlo, xhi = min(pts[:, 0].min(), cx - C.PLATE_HOLE_RADIUS) - 0.04, max(pts[:, 0].max(), cx + C.PLATE_HOLE_RADIUS) + 0.04
    ylo, yhi = min(pts[:, 1].min(), cy - C.PLATE_HOLE_RADIUS) - 0.04, max(pts[:, 1].max(), cy + C.PLATE_HOLE_RADIUS) + 0.04
    zlo, zhi = min(pts[:, 2].min(), rec.z_success) - 0.03, pts[:, 2].max() + 0.03
    half = 0.5 * max(xhi - xlo, yhi - ylo, zhi - zlo)
    mx, my, mz = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)
    cable = "tension" in rec.frames[0]

    images = []
    for k, f in enumerate(rec.frames):
        fig = Figure(figsize=(10.0 if cable else 7.5, 3.6), dpi=dpi)
        FigureCanvasAgg(fig)
        n_ax = 4 if cable else 3
        w = 0.98 / n_ax
        axes = [fig.add_axes([0.01 + j * w, 0.03, w - 0.012, 0.83]) for j in range(3)]
        if cable:
            axes.append(fig.add_axes([0.01 + 3 * w + 0.03, 0.17, w - 0.045, 0.69]))
        chain = np.vstack([f["seg"], f["tip"][None]])
        for ax, (a, b, name) in zip(axes, ((0, 2, "front (x-z)"), (1, 2, "side (y-z)"), (0, 1, "top (x-y)"))):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=8, pad=2)
            ax.set_aspect("equal")
            ca, cb = (mx, my, mz)[a], (mx, my, mz)[b]
            ax.set_xlim(ca - half, ca + half)
            ax.set_ylim(cb - half, cb + half)
            if b == 2:                                                  # side views: pipe walls, plate, goal line
                c = (cx, cy)[a]
                for s in (-1, 1):
                    ax.add_patch(Rectangle((c + s * C.PLATE_HOLE_RADIUS, rec.z_top - rec.bore), s * C.PIPE_WALL,
                                           rec.bore, color="#74c0fc"))
                ax.axhline(C.PLATE_TOP_Z, color="0.6", lw=0.8)
                ax.axhline(rec.z_success, color="#40c057", lw=0.6, ls="--")
            else:
                ax.add_patch(Circle((cx, cy), C.PLATE_HOLE_RADIUS, fill=False, color="#74c0fc", lw=1.5))
            for s in range(3):
                part = chain[5 * s: 5 * s + 6]
                ax.plot(part[:, a], part[:, b], "-o", color=SECTION_COLORS[s], lw=2.2, ms=2.5)
            ax.plot(f["tip"][a], f["tip"][b], "o", color="k", ms=3)
        if cable:
            ax = axes[3]
            import cables as CB
            t = f["tension"]
            ax.bar(np.arange(12), t, color=[SECTION_COLORS[s] for s in CB.CABLE_SECTION],
                   edgecolor=["k" if v < 0.05 else "none" for v in t])
            ax.set_ylim(0, max(12.0, float(max(fr["tension"].max() for fr in rec.frames)) * 1.1))
            ax.set_xticks(np.arange(12))
            ax.set_xticklabels(CB.CABLE_NAMES, fontsize=5, rotation=90)
            ax.tick_params(axis="y", labelsize=6)
            ax.set_title(f"cable tension [N] | max motor {np.abs(f['torque']).max() * 1000:.0f} mN m", fontsize=8, pad=2)
            ax.set_aspect("auto")
        last = k == len(rec.frames) - 1
        state = f"step {f['step']:3d}  depth {max(f['depth'], 0) * 1000:5.1f} mm  offset {f['lat'] * 1000:5.1f} mm" \
                f"  contacts {f['contacts']}"
        if last and rec.outcome is not None:
            state += f"   -> {OUTCOME_TEXT.get(rec.outcome, '')}"
        fig.text(0.01, 0.93, f"{title} [{rec.mode}]   {state}", fontsize=8,
                 color="#2b8a3e" if last and rec.outcome == 0 else "k")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        images.append(Image.open(buf).convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
    out = io.BytesIO()
    ms = int(20 * rec.every)
    durations = [ms] * (len(images) - 1) + [1500]                      # hold the last frame
    images[0].save(out, format="GIF", save_all=True, append_images=images[1:], duration=durations, loop=0)
    return out.getvalue(), images[0].size


def log_gif(writer, tag, gif, size, step):
    """Animated GIF into TensorBoard (IMAGES tab)."""
    from tensorboard.compat.proto.summary_pb2 import Summary

    image = Summary.Image(height=size[1], width=size[0], colorspace=3, encoded_image_string=gif)
    writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag=tag, image=image)]), step)
    writer.flush()


def render_and_log_async(rec, writer, step, out_dir, title):
    def work():
        try:
            gif, size = render_gif(rec, title)
            if writer is not None:
                log_gif(writer, "eval/robot", gif, size, step)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"eval_{step:08d}.gif"), "wb") as fh:
                fh.write(gif)
        except Exception as exc:                                        # never break training for a drawing
            print(f"[train_viz] {exc!r}", flush=True)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread
