"""Shared glue for the scripts: config files, pre/post-processors, checkpoints, batch building."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

import numpy as np
import yaml

from .data import DEPTH_KEY, RGB_KEY

CONTRACT_FILE = "depthvla_contract.json"


# ---------------------------------------------------------------- config: YAML + key.sub=value overrides
def load_config(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"override {item!r} is not key.sub=value")
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


# ---------------------------------------------------------------- features / processors
def make_features(cameras: Sequence[str], image_hw: Sequence[int], state_dim: int, action_dim: int):
    """Policy features. Depth is deliberately NOT a policy feature: a VISUAL feature would be embedded by
    SigLIP as one more camera (64 extra tokens). It travels as an extra batch key instead and is consumed by
    the depth branch."""
    from lerobot.configs import FeatureType, PolicyFeature

    h, w = image_hw
    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
              **{RGB_KEY.format(c): PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for c in cameras}}
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return inputs, outputs


def make_processors(config, stats: Dict[str, Dict[str, np.ndarray]]):
    import torch
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    t_stats = {k: {s: torch.as_tensor(np.asarray(v)) for s, v in d.items()} for k, d in stats.items()}
    return make_smolvla_pre_post_processors(config, dataset_stats=t_stats)


def load_processors(checkpoint: str, config, device: str):
    from lerobot.policies import make_pre_post_processors

    return make_pre_post_processors(config, pretrained_path=checkpoint,
                                    preprocessor_overrides={"device_processor": {"device": device}})


def save_checkpoint(out_dir: str, policy, pre, post, contract: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    pre.save_pretrained(out_dir)
    post.save_pretrained(out_dir)
    with open(os.path.join(out_dir, CONTRACT_FILE), "w") as fh:
        json.dump(contract, fh, indent=1)


def read_contract(checkpoint: str) -> Dict[str, Any]:
    with open(os.path.join(checkpoint, CONTRACT_FILE)) as fh:
        return json.load(fh)


def depth_keys(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in batch.items() if k.startswith("observation.depth.")}


def preprocess(pre, batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run the SmolVLA pre-processor (tokenise, normalise state / action, move to device) and carry the depth
    maps through untouched, whatever the processor pipeline does with keys it does not know."""
    import torch

    depth = depth_keys(batch)
    out = pre({k: v for k, v in batch.items() if k not in depth})
    for k, v in depth.items():
        out[k] = torch.as_tensor(v).to(device, non_blocking=True)
    return out


def observation_batch(contract: Dict[str, Any], images: Dict[str, np.ndarray], depth: Dict[str, np.ndarray],
                      state: np.ndarray, task: str) -> Dict[str, Any]:
    """One live observation -> the batch format of training (batch size 1).

    images[cam]: uint8 (H, W, 3).  depth[cam]: float metres (H, W), or uint16 millimetres; 0 / inf / nan = invalid."""
    import torch

    b: Dict[str, Any] = {"observation.state": torch.as_tensor(np.asarray(state, np.float32).reshape(1, -1)),
                         "task": [task]}
    hw = tuple(contract["image_hw"])
    for c in contract["cameras"]:
        img = np.asarray(images[c])
        if img.dtype != np.uint8 or img.shape != (*hw, 3):
            raise ValueError(f"image {c}: expected uint8 {(*hw, 3)}, got {img.dtype} {img.shape}")
        b[RGB_KEY.format(c)] = torch.from_numpy(img.copy()).permute(2, 0, 1).float()[None] / 255.0
    for c in contract["depth_cameras"]:
        d = np.asarray(depth[c]).squeeze()
        if d.shape != hw:
            raise ValueError(f"depth {c}: expected {hw}, got {d.shape}")
        d = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else d.astype(np.float32)
        b[DEPTH_KEY.format(c)] = torch.from_numpy(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0))[None, None]
    return b
