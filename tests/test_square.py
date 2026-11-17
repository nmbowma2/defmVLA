"""center_square: crop + resize used by EpisodeWriter(square=...) and PolicyClient(square=...)."""

import numpy as np
from PIL import Image

from depthvla.episodes import EpisodeWriter, center_square, depth_u16_to_m, episode_dirs, read_meta


def test_center_square_crops_the_middle_and_keeps_rgb_depth_aligned():
    h, w = 720, 1280
    rgb = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), 1.0, np.float32)
    # a marker at the image centre and one outside the central square
    rgb[340:380, 620:660] = 255
    depth[340:380, 620:660] = 0.25
    rgb[:, :100] = 200
    depth[:, :100] = 9.0

    r, d = center_square(rgb, 640), center_square(depth, 640, depth=True)
    assert r.shape == (640, 640, 3) and r.dtype == np.uint8
    assert d.shape == (640, 640) and d.dtype == np.float32
    assert not (r == 200).any() and not (d == 9.0).any()          # the side strip is cropped away
    ys, xs = np.nonzero(d == 0.25)
    assert abs(ys.mean() - 319.5) < 1 and abs(xs.mean() - 319.5) < 1   # centre stays at the centre
    ry, rx = np.nonzero(r[..., 0] > 127)
    assert abs(ry.mean() - ys.mean()) < 1 and abs(rx.mean() - xs.mean()) < 1   # RGB and depth agree


def test_depth_resize_never_blends_or_invents_values():
    depth = np.full((480, 848), 2.0, np.float32)
    depth[:, 424:] = 0.5                     # a depth edge
    depth[100:140, 300:340] = np.inf         # invalid pixels
    depth[200:240, 300:340] = 0.0
    for size in (224, 448, 480, 640):        # down, same, up
        d = center_square(depth, size, depth=True)
        assert d.shape == (size, size)
        assert set(np.unique(d[np.isfinite(d)]).tolist()) <= {0.0, 0.5, 2.0}
        assert np.isinf(d).any()
    assert center_square(depth, None, depth=True).shape == (480, 480)


def test_writer_square_option(tmp_path):
    root = str(tmp_path / "ds")
    w = EpisodeWriter(root, cameras=["wrist"], depth_cameras=["wrist"], fps=30, square=64)
    w.start()
    for t in range(3):
        depth = np.full((72, 128), 0.4, np.float32)
        depth[0, 0] = np.inf
        w.add_frame({"wrist": np.full((72, 128, 3), 90, np.uint8)}, {"wrist": depth}, np.zeros(4), np.zeros(3))
    ep = w.save("task")
    assert read_meta(root)["image_hw"] == [64, 64] and episode_dirs(root) == [ep]
    rgb = np.asarray(Image.open(f"{ep}/rgb.wrist/000000.png"))
    d = depth_u16_to_m(np.asarray(Image.open(f"{ep}/depth.wrist/000000.png")))
    assert rgb.shape == (64, 64, 3) and d.shape == (64, 64)
    assert np.allclose(d, 0.4, atol=1e-3)
