"""Wrist depth range and invalid masking inside the real policy (weights)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("defm")
from test_policy import DEPTH_MAP, build  # noqa: E402

pytestmark = pytest.mark.weights
KEY = "observation.depth.front"


def residual(p, depth):
    dev = next(p.parameters()).device
    b = {k: torch.zeros(1, 3, *depth.shape[-2:], device=dev) for k in p.config.image_features}
    b[KEY] = depth.to(dev)
    with torch.no_grad():
        return next(r for r in p.depth_residuals(b) if r is not None)


def test_range_and_mask():
    p = build(native_resize=True, depth_mask_invalid=True, depth_range_m={KEY: [0.07, 0.5]})
    p.eval()
    p.model.depth_fusion.gate.data.fill_(0.5)
    d = torch.full((1, 1, 720, 1280), 0.3)
    d[..., 640:] = 0.6                                   # right half beyond 0.5 m
    d[:, :, :20, :20] = 0.05                             # closer than 0.07 m
    clean = torch.where((d >= 0.07) & (d <= 0.5), d, torch.zeros_like(d))
    a, b = residual(p, d), residual(p, clean)
    assert torch.equal(a, b)
    on = a[0].abs().sum(-1).view(8, 8) > 0
    assert not on[:3].any()                              # 224 of 512 rows are padding: token rows 0-2 hold nothing
    assert on[4:, :4].all() and not on[:, 4:].any()      # valid left half only
    assert (residual(p, torch.full_like(d, 2.0)) == 0).all()
    assert torch.isfinite(a).all()
    q = build()
    assert q.config.depth_range_m == {} and not q.config.native_resize and not q.config.depth_mask_invalid
    with pytest.raises(ValueError):
        build(depth_range_m={"observation.depth.nope": [0.07, 0.5]})
    assert DEPTH_MAP
