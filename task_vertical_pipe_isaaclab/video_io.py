"""
==============================================================================
video_io.py - MP4 writing / reading through imageio-ffmpeg
==============================================================================
The OpenCV of env_isaacsim62 (opencv-python-headless-noffmpeg) has no video I/O, so
cv2.VideoWriter / cv2.VideoCapture open nothing. These two classes take their place
with the same BGR frames:

  Mp4Writer(path, fps, (w, h))   .write(bgr), .release()           H.264, yuv420p
  VideoReader(path)              .isOpened(), .fps, .frame_count, .read() -> (ok, bgr)

Needs imageio-ffmpeg (pip install imageio-ffmpeg; it brings its own ffmpeg binary).
==============================================================================
"""

from __future__ import annotations

import os

import numpy as np


class Mp4Writer:
    """H.264 MP4; write() takes BGR frames (H, W, 3) uint8 of the size given here."""

    def __init__(self, path, fps, size):
        import imageio_ffmpeg

        self._gen = imageio_ffmpeg.write_frames(path, tuple(int(v) for v in size), fps=float(fps), codec="libx264",
                                                pix_fmt_out="yuv420p")
        self._gen.send(None)

    def write(self, bgr):
        self._gen.send(np.ascontiguousarray(bgr[..., ::-1]))

    def release(self):
        self._gen.close()


class VideoReader:
    """Sequential reader with the parts of cv2.VideoCapture used here; read() returns BGR frames."""

    def __init__(self, path):
        self.fps, self.frame_count, self._gen = 0.0, 0, None
        if not os.path.isfile(path):
            return
        import imageio_ffmpeg

        self.frame_count = imageio_ffmpeg.count_frames_and_secs(path)[0]
        self._gen = imageio_ffmpeg.read_frames(path)
        meta = next(self._gen)
        self.fps = float(meta["fps"])
        self._w, self._h = meta["size"]

    def isOpened(self):  # noqa: N802  (cv2.VideoCapture name)
        return self._gen is not None

    def read(self):
        try:
            raw = next(self._gen)
        except StopIteration:
            return False, None
        return True, np.frombuffer(raw, dtype=np.uint8).reshape(self._h, self._w, 3)[..., ::-1].copy()

    def release(self):
        if self._gen is not None:
            self._gen.close()
            self._gen = None
