"""Recorded episodes -> mp4 (for slides): the cameras side by side, optionally a depth row underneath.

    python scripts/make_video.py --root data/normal_train_w0 --episodes 0 1            # -> videos/normal_train_w0_ep000000.mp4, ...
    python scripts/make_video.py --root data/normal_train_w0 --episodes 3 --depth      # + colour-mapped depth row
    python scripts/make_video.py --root data/low_contrast_val_w0 --episodes 0 --depth --scale 0.75

Reads the episode PNGs directly (numpy + pillow), so it runs in either env (training or Isaac).
Encoder, first one available: the ffmpeg command (H.264, plays in PowerPoint / Keynote / browsers), PyAV (H.264 if its
FFmpeg has it, else MPEG-4), OpenCV (MPEG-4 Part 2 'mp4v', which some slide software will not play: convert with
`ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4` on any machine that has ffmpeg).
Depth: near = warm, far = cool (turbo colour map), 0 (no reading) = black, range = 1st..99th percentile of the
episode's valid depth per camera.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", required=True, help="recorded dataset directory (has meta.json and episode_*/)")
ap.add_argument("--episodes", type=int, nargs="+", default=[0], help="episode indices (position in sorted order)")
ap.add_argument("--cams", nargs="+", default=None, help="camera order (default: front first, then the rest)")
ap.add_argument("--depth", action="store_true", help="add a colour-mapped depth row")
ap.add_argument("--scale", type=float, default=0.5, help="tile size relative to the stored image (0.5: 640 -> 320)")
ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time at the recording fps)")
ap.add_argument("--no-labels", action="store_true")
ap.add_argument("--out-dir", default="videos")
a = ap.parse_args()


def turbo(x):
    """Polynomial approximation of the turbo colour map; x in [0, 1] -> (..., 3) uint8."""
    x = np.clip(x, 0, 1)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def font(size):
    try:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


class Writer:
    """rgb24 frames -> mp4 with the first available encoder."""

    def __init__(self, path, w, h, fps):
        self.path, self.kind = path, None
        if shutil.which("ffmpeg"):
            self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
                                       "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", path],
                                      stdin=subprocess.PIPE)
            self.kind = "ffmpeg/h264"
            return
        try:
            import av
            from fractions import Fraction
            self.c = av.open(path, "w")
            for codec in ("libx264", "h264", "mpeg4"):
                try:
                    self.s = self.c.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
                    break
                except Exception:
                    continue
            self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
            self.kind, self.av = f"pyav/{self.s.codec_context.name}", av
            return
        except ImportError:
            pass
        import cv2
        self.v = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.kind, self.cv2 = "opencv/mp4v", cv2

    def write(self, frame):
        if self.kind.startswith("ffmpeg"):
            self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self.kind.startswith("pyav"):
            for pkt in self.s.encode(self.av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                self.c.mux(pkt)
        else:
            self.v.write(frame[..., ::-1].copy())

    def close(self):
        if self.kind.startswith("ffmpeg"):
            self.p.stdin.close()
            if self.p.wait() != 0:
                raise SystemExit(f"ffmpeg failed writing {self.path}")
        elif self.kind.startswith("pyav"):
            for pkt in self.s.encode():
                self.c.mux(pkt)
            self.c.close()
        else:
            self.v.release()


meta = json.load(open(os.path.join(a.root, "meta.json")))
cams = a.cams or sorted(meta["cameras"], key=lambda c: (c != "front", c))
dcams = [c for c in cams if c in meta.get("depth_cameras", [])] if a.depth else []
fps = float(meta["fps"]) * a.speed
eps = sorted(d for d in glob.glob(os.path.join(a.root, "episode_*")) if os.path.isdir(d))
os.makedirs(a.out_dir, exist_ok=True)

for i in a.episodes:
    if not 0 <= i < len(eps):
        raise SystemExit(f"episode index {i} out of range: {a.root} has {len(eps)} episodes")
    ep = eps[i]
    frames = {c: sorted(glob.glob(os.path.join(ep, f"rgb.{c}", "*.png"))) for c in cams}
    T = min(len(v) for v in frames.values())
    h0, w0 = np.asarray(Image.open(frames[cams[0]][0])).shape[:2]
    tw, th = int(round(w0 * a.scale / 2)) * 2, int(round(h0 * a.scale / 2)) * 2       # even sizes for yuv420p
    rng = {}
    for c in dcams:                                       # per-camera depth range from a sample of the episode
        files = sorted(glob.glob(os.path.join(ep, f"depth.{c}", "*.png")))
        s = np.concatenate([np.asarray(Image.open(f)).ravel() for f in files[:: max(1, len(files) // 10)]])
        s = s[s > 0].astype(np.float32)
        rng[c] = (np.percentile(s, 1), np.percentile(s, 99)) if s.size else (0.0, 1.0)
    task = open(os.path.join(ep, "task.txt")).read().strip() if os.path.exists(os.path.join(ep, "task.txt")) else ""
    W, H = tw * len(cams), th * (2 if dcams else 1)
    out = os.path.join(a.out_dir, f"{os.path.basename(os.path.normpath(a.root))}_{os.path.basename(ep)}.mp4")
    wr = Writer(out, W, H, fps)
    f_lab, f_task = font(max(12, th // 18)), font(max(12, th // 16))
    for t in range(T):
        canvas = Image.new("RGB", (W, H))
        for k, c in enumerate(cams):
            canvas.paste(Image.open(frames[c][t]).convert("RGB").resize((tw, th), Image.BILINEAR), (k * tw, 0))
            if c in dcams:
                d = np.asarray(Image.open(os.path.join(ep, f"depth.{c}", os.path.basename(frames[c][t])))).astype(np.float32)
                lo, hi = rng[c]
                img = turbo(1.0 - (d - lo) / max(hi - lo, 1.0))
                img[d == 0] = 0
                canvas.paste(Image.fromarray(img).resize((tw, th), Image.NEAREST), (k * tw, th))
        if not a.no_labels:
            dr = ImageDraw.Draw(canvas)
            for k, c in enumerate(cams):
                dr.text((k * tw + 8, 6), c, fill=(255, 255, 255), font=f_lab, stroke_width=2, stroke_fill=(0, 0, 0))
                if c in dcams:
                    dr.text((k * tw + 8, th + 6), f"{c} depth", fill=(255, 255, 255), font=f_lab, stroke_width=2,
                            stroke_fill=(0, 0, 0))
            if task:
                dr.text((8, th - th // 10), task, fill=(255, 255, 255), font=f_task, stroke_width=2, stroke_fill=(0, 0, 0))
        wr.write(np.asarray(canvas))
    wr.close()
    print(f"{out}  ({T} frames, {T / fps:.1f} s, {W}x{H}, {wr.kind})", flush=True)
