"""Offline tests: no weights, no lerobot model. Fusion shapes, zero gate, spatial alignment, episode format."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defmvla.episodes import EpisodeWriter, depth_m_to_u16, depth_u16_to_m, read_meta, write_synthetic  # noqa: E402
from defmvla.fusion import DepthFusion, resize_depth_like_rgb, token_grid  # noqa: E402


@pytest.mark.parametrize("pool", ["unshuffle", "avg"])
def test_fusion_gives_exactly_one_image_token_block(pool):
    f = DepthFusion(in_channels=384, in_grid=32, out_grid=8, out_dim=960, pool=pool)
    out = f(torch.randn(3, 384, 32, 32))
    assert out.shape == (3, 64, 960)  # 64 tokens x 960: the shape of SmolVLA's image tokens for one camera
    assert f.num_tokens == 64 and token_grid(64) == 8


def test_zero_gate_is_exactly_zero_and_gate_gets_gradient():
    f = DepthFusion(384, 32, 8, 960)
    out = f(torch.randn(2, 384, 32, 32))
    assert torch.count_nonzero(out) == 0
    (out + 1.0).pow(2).sum().backward()  # any downstream loss
    assert f.gate.grad.abs().sum() > 0  # the gate can open even though the MLP gets no gradient at step 0
    assert f.fc2.weight.grad.abs().sum() == 0


@pytest.mark.parametrize("pool", ["unshuffle", "avg"])
def test_token_i_only_sees_its_own_4x4_block(pool):
    """Token (r, c) of the 8 x 8 grid must depend only on DeFM patches [4r:4r+4, 4c:4c+4] (row-major), i.e.
    the same image region as SmolVLA's RGB token (r, c)."""
    torch.manual_seed(0)
    f = DepthFusion(16, 32, 8, 32, hidden=64, pool=pool)
    f.gate.data.fill_(1.0)
    x = torch.randn(1, 16, 32, 32)
    base = f(x)
    r, c = 5, 2
    x2 = x.clone()
    x2[:, :, 4 * r:4 * r + 4, 4 * c:4 * c + 4] += 1.0
    changed = (f(x2) - base).abs().sum(-1)[0] > 1e-6
    assert changed.nonzero().flatten().tolist() == [r * 8 + c]


def test_bad_grid_is_rejected():
    with pytest.raises(ValueError):
        DepthFusion(384, in_grid=30, out_grid=8, out_dim=960)
    with pytest.raises(ValueError):
        token_grid(60)


def test_depth_resize_matches_lerobot_rgb_geometry():
    vla_utils = pytest.importorskip("lerobot.policies.common.vla_utils")
    # a marker pixel must land at the same place in the resized RGB and the resized depth
    h, w = 96, 160
    rgb = torch.zeros(1, 3, h, w)
    dep = torch.zeros(1, 1, h, w)
    rgb[:, :, 40:56, 100:116] = 1.0
    dep[:, :, 40:56, 100:116] = 1.0
    r = vla_utils.resize_with_pad(rgb, 512, 512, pad_value=0)
    d = resize_depth_like_rgb(dep, 512, 512)
    assert d.shape == (1, 1, 512, 512) and r.shape == (1, 3, 512, 512)
    rm, dm = r[0, 0] > 0.5, d[0, 0] > 0.5
    iou = (rm & dm).sum() / (rm | dm).sum()
    assert iou > 0.9, f"depth and RGB letterboxing disagree (IoU {iou:.2f})"
    assert d[0, 0, :100].sum() == 0  # padding is on the top (and left): invalid depth


def test_depth_u16_round_trip_and_invalid():
    d = np.array([[0.0, 0.5, 1.2345], [np.inf, np.nan, 70.0]], np.float32)
    u = depth_m_to_u16(d)
    assert u.dtype == np.uint16 and u[1].tolist() == [0, 0, 0]  # inf / nan / beyond range -> invalid
    assert np.allclose(depth_u16_to_m(u)[0], [0.0, 0.5, 1.2345], atol=5e-4)


def test_episode_round_trip(tmp_path):
    from defmvla.data import check_roots, fit_stats, load_low_dim, make_dataset

    root = write_synthetic(str(tmp_path / "ds"), episodes=2, length=6, hw=(64, 64), state_dim=8, action_dim=7)
    meta = read_meta(root)
    assert meta["episodes"] == 2 and meta["image_hw"] == [64, 64] and check_roots([root]) == meta
    eps = load_low_dim([root])
    stats = fit_stats(eps)
    assert stats["action"]["mean"].shape == (7,) and (stats["action"]["std"] > 0).all()
    ds = make_dataset(eps, horizon=4, cameras=meta["cameras"], depth_cameras=meta["depth_cameras"])
    assert len(ds) == 12
    s = ds[5]  # last frame of episode 0: 3 of 4 chunk steps are padding
    assert s["observation.images.wrist"].shape == (3, 64, 64) and s["observation.images.wrist"].max() <= 1.0
    assert s["observation.depth.wrist"].shape == (1, 64, 64) and s["observation.depth.wrist"].dtype == torch.float32
    assert 0.2 < float(s["observation.depth.wrist"].min()) and float(s["observation.depth.wrist"].max()) == 1.5
    assert s["action"].shape == (4, 7) and s["action_is_pad"].tolist() == [False, True, True, True]
    assert "observation.depth.front" not in s


def test_writer_rejects_misaligned_depth(tmp_path):
    w = EpisodeWriter(str(tmp_path / "x"), ["wrist"], ["wrist"], fps=10)
    w.start()
    with pytest.raises(ValueError):
        w.add_frame({"wrist": np.zeros((64, 64, 3), np.uint8)}, {"wrist": np.zeros((32, 32), np.float32)},
                    np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        EpisodeWriter(str(tmp_path / "y"), ["front"], ["wrist"], fps=10)


def test_working_arm_and_arm_prompt():
    from defmvla.data import arm_task, working_arm
    T = 30
    s = np.zeros((T, 16), np.float32)
    s[:, 8:15] = 1.2                                    # right arm parked (constant)
    s[:, :7] = np.linspace(0, 1, T)[:, None]            # left arm moves
    assert working_arm(s) == "left"
    assert arm_task("pick up the cube", s) == "left arm: pick up the cube"
    s2 = s.copy(); s2[:, :7] = 0.3; s2[:, 8:15] = np.linspace(0, 1, T)[:, None]
    assert working_arm(s2) == "right"
