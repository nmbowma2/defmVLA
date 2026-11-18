"""Pure-numpy FK / IK for the prismatic FR3 duo URDF (rail_link frame). numpy + scipy only.

The URDF is sim/robot/duo_<variant>.urdf (variant = actual: arms tilted +-45 deg about x; upright: not tilted).
Set DUO_URDF to use another file.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

ROBOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot")
DROID_HOME = (0, -0.628, 0, -2.513, 0, 1.885, 0.785)


def urdf_path(variant="actual"):
    return os.environ.get("DUO_URDF") or os.path.join(ROBOT_DIR, f"duo_{variant}.urdf")


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def T_of(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def rot_axis(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
    return T


class Chain:
    """Forward kinematics from `root` to `tip_link` over the revolute joints whose name starts with arm_prefix."""

    def __init__(self, tip_link, urdf=None, root="rail_link", arm_prefix="left_fr3_joint", variant="actual"):
        r = ET.parse(urdf or urdf_path(variant)).getroot()
        by_child = {j.find("child").get("link"): j for j in r.findall("joint")}
        seq, link = [], tip_link
        while link != root:
            j = by_child[link]
            seq.append(j)
            link = j.find("parent").get("link")
        self.seq = seq[::-1]
        self.arm = [j for j in self.seq if j.get("name").startswith(arm_prefix) and j.get("type") == "revolute"]
        self.lim = np.array([[float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))] for j in self.arm])

    def fk(self, q):
        T, qi = np.eye(4), 0
        for j in self.seq:
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz").split()] if o is not None and o.get("xyz") else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy").split()] if o is not None and o.get("rpy") else [0, 0, 0]
            T = T @ T_of(xyz, rpy)
            if j in self.arm:
                T = T @ rot_axis([float(v) for v in j.find("axis").get("xyz").split()], q[qi])
                qi += 1
        return T


class Arm:
    """One arm: TCP chain + hand chain (the hand's +y axis is the finger closing direction)."""

    def __init__(self, side, variant="actual", urdf=None):
        pre = f"{side}_fr3_joint"
        self.side = side
        self.tcp = Chain(f"{side}_fr3_hand_tcp", urdf=urdf, arm_prefix=pre, variant=variant)
        self.hand = Chain(f"{side}_fr3_hand", urdf=urdf, arm_prefix=pre, variant=variant)
        self.lim = self.tcp.lim

    def tcp_pos(self, q):
        return self.tcp.fk(q)[:3, 3]

    def close_axis(self, q):
        return self.hand.fk(q)[:3, 1]

    def solve(self, pos, q0=None, close_along="y", seed=0, restarts=30):
        """Joint angles that put the TCP at `pos` with the approach (tcp z) pointing straight down and the finger
        closing axis horizontal along world `close_along` ('x' or 'y'), so the fingers straddle an axis-aligned
        object. Without the closing-axis term the hand yaw is left to the regulariser and varies with position.
        Returns (q, position error in metres, closing-axis error)."""
        from scipy.optimize import least_squares

        q0 = np.array(DROID_HOME if q0 is None else q0, float)
        want = np.array([1.0, 0, 0]) if close_along == "x" else np.array([0, 1.0, 0])

        def res(q):
            T = self.tcp.fk(q)
            e_pos = (T[:3, 3] - pos) * 20
            e_dir = (T[:3, 2] - np.array([0, 0, -1.0])) * 5
            c = self.close_axis(q)
            e_close = (1.0 - abs(float(np.dot(c, want)))) * 5      # 0 when the closing axis is +-want
            return np.concatenate([e_pos, e_dir, [e_close], (q - q0) * 0.02])

        best = None
        rng = np.random.default_rng(seed)
        lo, hi = self.lim[:, 0] + 0.05, self.lim[:, 1] - 0.05
        for k in range(restarts):
            s = np.clip(q0, lo, hi) if k == 0 else np.clip(q0 + rng.normal(0, 0.6, 7), lo, hi)
            r = least_squares(res, s, bounds=(self.lim[:, 0], self.lim[:, 1]))
            if best is None or r.cost < best.cost:
                best = r
            if best.cost < 1e-8:
                break
        q = best.x
        return q, float(np.linalg.norm(self.tcp_pos(q) - pos)), float(1 - abs(np.dot(self.close_axis(q), want)))
