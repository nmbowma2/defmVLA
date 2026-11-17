"""Depth branch: frozen DeFM encoder -> trainable gated MLP -> residual on the RGB image tokens.

    metric depth (B, 1, H, W), metres
      -> DeFM 3-channel metric normalisation (defm.preprocess_depth_batch), resized to `input_size`
      -> frozen DeFM ViT (no grad, always eval)                 (B, C, g, g)      g = input_size / 14
      -> regroup to the VLM's token grid (g_out x g_out)         (B, g_out^2, C')
           "unshuffle": pixel_unshuffle by g / g_out, C' = C * (g / g_out)^2   (keeps the fine detail; this is
                         the same block layout SmolVLM's connector uses to go from 32 x 32 patches to 8 x 8)
           "avg":       average pool, C' = C
      -> LayerNorm -> Linear -> GELU -> Linear -> * gate (per channel, init 0)   (B, g_out^2, vlm_dim)

The output has exactly the shape of one camera's image-token block, so it is ADDED to those tokens: the
prefix length the VLM sees never changes. With gate = 0 the output is exactly zero, so the model starts as
the RGB-only policy and uses depth only as far as the gate opens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

DEFM_PATCH = 14


class DeFMEncoder(nn.Module):
    """Frozen DeFM ViT. Input: metric depth in metres. Output: (B, C, g, g) float32 patch features."""

    def __init__(self, variant: str = "defm_vit_s14", pretrained: bool = True, weights_path: str | None = None,
                 input_size: int = 448):
        super().__init__()
        if "vit" not in variant:
            raise ValueError(f"only the DeFM ViT variants are supported here, got {variant!r}")
        if input_size % DEFM_PATCH:
            raise ValueError(f"defm input size {input_size} is not a multiple of {DEFM_PATCH}")
        from defm.model_factory import create_defm_model

        self.model = create_defm_model(variant, pretrained=pretrained, pretrained_path=weights_path or None)
        self.model.eval().requires_grad_(False)
        self.input_size = input_size
        self.grid = input_size // DEFM_PATCH
        self.out_channels = int(self.model.embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()  # DeFM never leaves eval
        return self

    @torch.no_grad()
    def forward(self, depth_m: Tensor) -> Tensor:
        from defm.utils.utils import preprocess_depth_batch

        with torch.autocast(device_type=depth_m.device.type, enabled=False):
            x = preprocess_depth_batch(depth_m.float(), target_size=self.input_size, patch_size=DEFM_PATCH,
                                       device=depth_m.device)
            feat = self.model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)[0]
        return feat.float()


class DepthFusion(nn.Module):
    """Trainable part: (B, C, g, g) encoder features -> gated residual (B, g_out^2, out_dim)."""

    def __init__(self, in_channels: int, in_grid: int, out_grid: int, out_dim: int, hidden: int = 1024,
                 pool: str = "unshuffle"):
        super().__init__()
        if in_grid % out_grid:
            raise ValueError(
                f"depth feature grid {in_grid} is not a multiple of the VLM image-token grid {out_grid}; "
                f"pick a DeFM input size of {DEFM_PATCH} * k * {out_grid}")
        if pool not in ("unshuffle", "avg"):
            raise ValueError(f"pool must be 'unshuffle' or 'avg', got {pool!r}")
        self.factor = in_grid // out_grid
        self.in_grid, self.out_grid, self.pool = in_grid, out_grid, pool
        c = in_channels * self.factor**2 if pool == "unshuffle" else in_channels
        self.norm = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.gate = nn.Parameter(torch.zeros(out_dim))  # 0 -> the policy is exactly the RGB policy

    @property
    def num_tokens(self) -> int:
        return self.out_grid**2

    def forward(self, feat: Tensor) -> Tensor:
        if feat.shape[-2:] != (self.in_grid, self.in_grid):
            raise ValueError(f"expected a {self.in_grid} x {self.in_grid} feature map, got {tuple(feat.shape)}")
        if self.factor > 1:
            feat = F.pixel_unshuffle(feat, self.factor) if self.pool == "unshuffle" else F.avg_pool2d(feat, self.factor)
        tok = feat.flatten(2).transpose(1, 2)  # (B, g_out^2, C'), row-major like the image tokens
        tok = tok.to(self.fc1.weight.dtype)
        return self.fc2(F.gelu(self.fc1(self.norm(tok)))) * self.gate


def resize_depth_like_rgb(depth: Tensor, width: int, height: int) -> Tensor:
    """Same geometry as SmolVLA's image `resize_with_pad` (aspect-preserving resize, then pad on the left and
    top), for a (B, 1, H, W) metric depth map. Nearest-neighbour, so no depth values are invented at object
    edges or next to invalid (0) pixels; padding is 0 = invalid."""
    cur_h, cur_w = depth.shape[-2:]
    if (cur_h, cur_w) == (height, width):
        return depth
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    out = F.interpolate(depth, size=(new_h, new_w), mode="nearest")
    return F.pad(out, (max(0, width - new_w), 0, max(0, height - new_h), 0), value=0.0)


def token_grid(num_tokens: int) -> int:
    g = math.isqrt(num_tokens)
    if g * g != num_tokens:
        raise ValueError(f"the VLM gives {num_tokens} tokens per image, which is not a square grid")
    return g
