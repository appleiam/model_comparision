"""Generate a synthetic ~10 s test clip: the Ultralytics bus.jpg scene slides downward,
so its 4 people walk down across a horizontal line at mid-height (expected: 4 IN with in_side=+1).

Usage: python scripts/make_sample_clip.py [out.mp4]
"""
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import ultralytics

out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "data" / "sample_bus.mp4")
img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # 405x540
W, H, FPS, N = 640, 720, 25, 250
rng = np.random.default_rng(0)
bg = (rng.integers(60, 90, (H, W, 3))).astype(np.uint8)
w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", pix_fmt_in="bgr24", macro_block_size=1)
w.send(None)
ih, iw = img.shape[:2]
for i in range(N):
    frame = bg.copy()
    oy = int(-260 + 520 * i / (N - 1))
    ox = (W - iw) // 2
    y0, y1 = max(0, oy), min(H, oy + ih)
    frame[y0:y1, ox:ox + iw] = img[y0 - oy:y1 - oy]
    w.send(frame.tobytes())
w.close()
print("wrote", out)
