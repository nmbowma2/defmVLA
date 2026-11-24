"""Full-resolution frames: per-encoder resize, wrist depth range, invalid-depth handling (no pretrained weights needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("defm")

from defmvla.fusion import preprocess_valid, resize_depth_like_rgb, resize_rgb_antialias  # noqa: E402


def test_letterbox_geometry_matches_per_encoder():
    img = torch.full((1, 3, 720, 1280), 0.5)
    dep = torch.ones(1, 1, 720, 1280)
    r, d = resize_rgb_antialias(img, 512, 512), resize_depth_like_rgb(dep, 448, 448)
    assert r.shape[-2:] == (512, 512) and d.shape[-2:] == (448, 448)
    assert (r[0, 0, :224] == 0).all() and torch.allclose(r[0, 0, 224:], torch.tensor(0.5))   # 14 of 32 patch rows are padding
    assert (d[0, 0, :196] == 0).all() and (d[0, 0, 196:] == 1).all()                          # the same 14 of 32 for DeFM


def test_preprocess_valid_equals_defm_without_invalids_and_ignores_them_otherwise():
    from defm.utils.utils import preprocess_depth_batch

    d = torch.rand(2, 1, 448, 448) * 0.4 + 0.1
    assert torch.allclose(preprocess_valid(d, 448), preprocess_depth_batch(d, target_size=448, patch_size=14), atol=1e-4)
    holes = d.clone()
    holes[:, :, :100] = 0.0
    keep = preprocess_valid(holes, 448)[:, 2, 100:]
    crop = torch.log1p(holes[:, 0, 100:])
    lo, hi = crop.amin((1, 2), keepdim=True), crop.amax((1, 2), keepdim=True)
    raw = (crop - lo) / (hi - lo)
    slope = (keep[0].max() - keep[0].min())
    assert torch.allclose((keep[0] - keep[0].min()) / slope, raw[0], atol=1e-4)  # full 0..1 range over valid pixels
    for x in (torch.zeros(1, 1, 448, 448), torch.full((1, 1, 448, 448), 0.3)):   # nothing valid / constant
        assert torch.isfinite(preprocess_valid(x, 448)).all()
