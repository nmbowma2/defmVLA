"""torch Dataset over the episode format of defmvla.episodes (lerobot-style sample dicts).

Sample:
    observation.images.<cam>   (3, H, W) float32 in [0, 1]
    observation.depth.<cam>    (1, H, W) float32 metres, 0 = invalid        (only for depth cameras)
    observation.state          (S,) float32
    action                     (horizon, A) float32   actions[t : t + horizon], last action repeated past the end
    action_is_pad              (horizon,) bool        True on the repeated steps (masked out of the loss)
    task                       str

State / action normalisation is NOT done here; it is done by the SmolVLA pre-processor from `fit_stats`.
Depth is never normalised here either: DeFM applies its own metric normalisation inside the model.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np

from .episodes import depth_u16_to_m, episode_dirs, read_meta

RGB_KEY = "observation.images.{}"
DEPTH_KEY = "observation.depth.{}"


def working_arm(state: np.ndarray) -> str:
    """'left' | 'right': the arm that moves in an episode (state = [left q1..q7, gripper, right q1..q7, gripper, ...]);
    the other one stays parked, so its joints are constant."""
    state = np.asarray(state)
    return "left" if np.ptp(state[:, :7], axis=0).sum() >= np.ptp(state[:, 8:15], axis=0).sum() else "right"


def arm_task(task: str, state: np.ndarray) -> str:
    """'<left|right> arm: <task>' (the single-arm skill prompt format of the reference SmolVLA recipe)."""
    return f"{working_arm(state)} arm: {task}"


def load_low_dim(roots: Sequence[str], arm_prompt: bool = False) -> List[Dict]:
    """Per episode: dir, state (T, S), action (T, A), task. Reads no images. arm_prompt: prefix the task with the
    working arm ("left arm: pick up ...") so the language tells the policy which arm to use."""
    eps = []
    for root in roots:
        for d in episode_dirs(root):
            z = np.load(os.path.join(d, "low_dim.npz"))
            with open(os.path.join(d, "task.txt")) as fh:
                task = fh.read().strip()
            state = z["state"].astype(np.float32)
            eps.append({"dir": d, "state": state, "action": z["action"].astype(np.float32),
                        "task": arm_task(task, state) if arm_prompt else task, "root": root})
    if not eps:
        raise FileNotFoundError(f"no episodes under {list(roots)}")
    return eps


def fit_stats(episodes: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """mean / std / min / max of state and action over all rows (std floored so constant dims do not blow up)."""
    out = {}
    for name, key in (("observation.state", "state"), ("action", "action")):
        x = np.concatenate([e[key] for e in episodes]).astype(np.float64)
        out[name] = {"mean": x.mean(0).astype(np.float32), "std": np.maximum(x.std(0), 1e-6).astype(np.float32),
                     "min": x.min(0).astype(np.float32), "max": x.max(0).astype(np.float32)}
    return out


def check_roots(roots: Sequence[str]) -> Dict:
    """All roots must share cameras / dims / image size; returns the common meta."""
    metas = [read_meta(r) for r in roots]
    keys = ("cameras", "depth_cameras", "state_dim", "action_dim", "image_hw", "fps")
    for m, r in zip(metas[1:], roots[1:]):
        diff = [k for k in keys if m[k] != metas[0][k]]
        if diff:
            raise ValueError(f"{r} differs from {roots[0]} in {diff}")
    return metas[0]


class EpisodeDataset:
    """Module-level (picklable) so DataLoader workers work with any start method."""

    def __init__(self, episodes: List[Dict], horizon: int, cameras: Sequence[str], depth_cameras: Sequence[str]):
        self.episodes, self.horizon = episodes, horizon
        self.cameras, self.depth_cameras = list(cameras), list(depth_cameras)
        self.index = [(i, t) for i, e in enumerate(episodes) for t in range(len(e["state"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        import torch
        from PIL import Image

        i, t = self.index[n]
        e, horizon = self.episodes[i], self.horizon
        T = len(e["action"])
        rows = np.minimum(np.arange(t, t + horizon), T - 1)
        s = {"observation.state": torch.from_numpy(e["state"][t]),
             "action": torch.from_numpy(e["action"][rows]),
             "action_is_pad": torch.from_numpy(np.arange(t, t + horizon) > T - 1),
             "task": e["task"]}
        for c in self.cameras:
            img = np.array(Image.open(os.path.join(e["dir"], f"rgb.{c}", f"{t:06d}.png")).convert("RGB"))
            s[RGB_KEY.format(c)] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        for c in self.depth_cameras:
            d = np.asarray(Image.open(os.path.join(e["dir"], f"depth.{c}", f"{t:06d}.png")))
            s[DEPTH_KEY.format(c)] = torch.from_numpy(depth_u16_to_m(d))[None]
        return s


def make_dataset(episodes: List[Dict], horizon: int, cameras: Sequence[str], depth_cameras: Sequence[str]):
    return EpisodeDataset(episodes, horizon, cameras, depth_cameras)
