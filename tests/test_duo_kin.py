"""Kinematics of the vendored duo robot (no Isaac needed): IK reaches the target, the approach is straight down, the
fingers straddle an axis-aligned object, and the two arms are mirror images."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from duo_kin import Arm, urdf_path  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(urdf_path("actual")), reason="sim/robot URDF missing")


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("close", ["x", "y"])
def test_ik_reaches_with_vertical_approach_and_aligned_fingers(side, close):
    arm = Arm(side)
    sgn = 1 if side == "left" else -1
    for p in ([0.42, sgn * 0.12, 0.18], [0.40, sgn * 0.16, 0.05], [0.45, sgn * 0.30, 0.13]):
        q, err, cerr = arm.solve(np.array(p), close_along=close)
        assert err < 1e-3 and cerr < 1e-2
        T = arm.tcp.fk(q)
        assert np.allclose(T[:3, 2], [0, 0, -1], atol=0.02)                       # approach straight down
        assert (q >= arm.lim[:, 0] - 1e-9).all() and (q <= arm.lim[:, 1] + 1e-9).all()


def test_arms_are_mirror_images_about_the_xz_plane():
    L, R = Arm("left"), Arm("right")
    q = np.array([0.3, -0.4, 0.2, -2.0, 0.1, 1.7, 0.5])
    pl = L.tcp_pos(q)
    pr = R.tcp_pos(np.array([-q[0], q[1], -q[2], q[3], -q[4], q[5], -q[6]]))        # mirrored joint angles
    assert np.allclose(pr, pl * np.array([1, -1, 1]), atol=1e-6)


def test_upright_variant_differs_from_tilted():
    a, u = Arm("left", variant="actual"), Arm("left", variant="upright")
    q = np.array([0, -0.5, 0, -2.0, 0, 1.5, 0.7])
    assert not np.allclose(a.tcp_pos(q), u.tcp_pos(q), atol=1e-3)
