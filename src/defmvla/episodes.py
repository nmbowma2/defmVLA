"""On-disk RGB-D episode format + writer. numpy + pillow only, so it runs inside the Isaac Sim python env.

    <root>/meta.json                      fps, cameras, depth_cameras, state_dim, action_dim, depth_unit, episodes
    <root>/episode_000000/low_dim.npz     state (T, S) float32, action (T, A) float32
    <root>/episode_000000/task.txt        language instruction
    <root>/episode_000000/rgb.<cam>/000000.png      uint8 RGB
    <root>/episode_000000/depth.<cam>/000000.png    uint16, millimetres, planar z-depth; 0 = no reading

Depth is lossless 16-bit PNG in millimetres (never video). Feed `distance_to_image_plane` from Isaac (planar
z-depth, what a RealSense reports), in metres; inf / nan / beyond 65.535 m become 0 = invalid, which is the
convention DeFM was trained with.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

DEPTH_SCALE = 1000.0  # stored uint16 = metres * 1000
META = "meta.json"


def depth_m_to_u16(depth_m: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(np.asarray(depth_m, np.float64), nan=0.0, posinf=0.0, neginf=0.0) * DEPTH_SCALE
    d[(d < 0) | (d > 65535)] = 0
    return np.round(d).astype(np.uint16)


def depth_u16_to_m(depth_u16: np.ndarray) -> np.ndarray:
    return np.asarray(depth_u16, np.float32) / DEPTH_SCALE


def center_square(img: np.ndarray, size: Optional[int] = None, depth: bool = False) -> np.ndarray:
    """Centre-crop (H, W[, C]) to its largest square, then resize to size x size (None = keep the crop).

    Use the same call for the RGB and the depth frame of a camera so they stay pixel-aligned. RGB is
    area-averaged when shrinking; depth uses nearest neighbour, so depths are never blended across an edge
    and invalid pixels (0 / inf / nan) stay invalid instead of being averaged into fake distances.
    """
    a = np.asarray(img)
    h, w = a.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    a = a[top:top + s, left:left + s]
    if size and size != s:
        if depth:
            idx = np.minimum(((np.arange(size) + 0.5) * s / size).astype(np.int64), s - 1)
            a = a[idx][:, idx]
        else:
            a = np.asarray(Image.fromarray(a).resize((size, size), Image.BOX if size < s else Image.BILINEAR))
    return np.ascontiguousarray(a)


def episode_dirs(root: str) -> List[str]:
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if d.startswith("episode_") and os.path.isdir(os.path.join(root, d)))


def read_meta(root: str) -> Dict:
    with open(os.path.join(root, META)) as fh:
        return json.load(fh)


class EpisodeWriter:
    """Usage (inside the simulator loop):

        w = EpisodeWriter("data/normal", cameras=["wrist", "front"], depth_cameras=["wrist"], fps=30)
        w.start()
        for step in ...:
            w.add_frame(rgb={"wrist": rgb_w, "front": rgb_f}, depth={"wrist": depth_w_m}, state=s, action=a)
        w.save(task="pick up the mug")          # or w.discard() for a failed demo
    """

    def __init__(self, root: str, cameras: Sequence[str], depth_cameras: Sequence[str], fps: float,
                 square: Optional[int] = None):
        """square=640: centre-crop every RGB and depth frame to a square and resize it to 640 x 640
        (see center_square). Leave it None if the frames are already square."""
        if not set(depth_cameras) <= set(cameras):
            raise ValueError("every depth camera must also be an RGB camera (the tokens are added per camera)")
        self.root, self.cameras, self.depth_cameras, self.fps = root, list(cameras), list(depth_cameras), float(fps)
        self.square = int(square) if square else None
        os.makedirs(root, exist_ok=True)
        self._buf = None

    def start(self):
        self._buf = {"rgb": {c: [] for c in self.cameras}, "depth": {c: [] for c in self.depth_cameras},
                     "state": [], "action": []}

    def add_frame(self, rgb: Dict[str, np.ndarray], depth: Dict[str, np.ndarray], state, action):
        b = self._buf
        for c in self.cameras:
            img = np.asarray(rgb[c])
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"rgb[{c}] must be uint8 (H, W, 3), got {img.dtype} {img.shape}")
            b["rgb"][c].append(center_square(img, self.square) if self.square else img.copy())
        for c in self.depth_cameras:
            d = np.asarray(depth[c]).squeeze()
            if self.square:
                d = center_square(d, self.square, depth=True)
            if d.shape != b["rgb"][c][-1].shape[:2]:
                raise ValueError(f"depth[{c}] {d.shape} must be pixel-aligned with rgb[{c}] {b['rgb'][c][-1].shape[:2]}")
            b["depth"][c].append(depth_m_to_u16(d))
        b["state"].append(np.asarray(state, np.float32).ravel())
        b["action"].append(np.asarray(action, np.float32).ravel())

    def discard(self):
        self._buf = None

    def save(self, task: str) -> str:
        b = self._buf
        if not b or not b["state"]:
            raise ValueError("no frames to save")
        idx = len(episode_dirs(self.root))
        ep = os.path.join(self.root, f"episode_{idx:06d}")
        os.makedirs(ep)
        state, action = np.stack(b["state"]), np.stack(b["action"])
        np.savez(os.path.join(ep, "low_dim.npz"), state=state, action=action)
        with open(os.path.join(ep, "task.txt"), "w") as fh:
            fh.write(task.strip())
        for kind in ("rgb", "depth"):
            for c, frames in b[kind].items():
                d = os.path.join(ep, f"{kind}.{c}")
                os.makedirs(d)
                for t, f in enumerate(frames):
                    Image.fromarray(f).save(os.path.join(d, f"{t:06d}.png"))
        h, w = b["rgb"][self.cameras[0]][0].shape[:2]
        meta = {"fps": self.fps, "cameras": self.cameras, "depth_cameras": self.depth_cameras,
                "state_dim": int(state.shape[1]), "action_dim": int(action.shape[1]), "image_hw": [h, w],
                "depth_unit": "uint16 millimetres, planar z-depth, 0 = invalid", "episodes": idx + 1}
        with open(os.path.join(self.root, META), "w") as fh:
            json.dump(meta, fh, indent=1)
        self._buf = None
        return ep


def write_synthetic(root: str, episodes: int = 4, length: int = 24, hw=(128, 128), state_dim: int = 8,
                    action_dim: int = 7, cameras=("wrist", "front"), depth_cameras=("wrist",), seed: int = 0) -> str:
    """Tiny fake dataset for plumbing tests only (a moving square whose depth and position set the action)."""
    rng = np.random.default_rng(seed)
    w = EpisodeWriter(root, cameras, depth_cameras, fps=10)
    H, W = hw
    for _ in range(episodes):
        w.start()
        x0, y0, z0 = rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6), rng.uniform(0.3, 0.9)
        for t in range(length):
            x, y, z = x0 + 0.2 * t / length, y0 + 0.15 * np.sin(t / 4.0), z0 + 0.3 * np.sin(t / 5.0)
            rgb = {c: np.full((H, W, 3), 40, np.uint8) for c in cameras}
            dep = {c: np.full((H, W), 1.5, np.float32) for c in depth_cameras}
            r0, c0 = int(y * H), int(x * W)
            for c in cameras:
                rgb[c][r0:r0 + H // 6, c0:c0 + W // 6] = (200, 60, 60)
            for c in depth_cameras:
                dep[c][r0:r0 + H // 6, c0:c0 + W // 6] = z
            state = np.concatenate([[x, y, z], rng.normal(0, 0.01, state_dim - 3)])
            action = np.concatenate([[x, y, z], np.zeros(action_dim - 3)])
            w.add_frame(rgb, dep, state, action)
        w.save("move to the red block")
    return root
