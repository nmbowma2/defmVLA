"""Scripted pick-and-place demonstrator for DuoSim: IK waypoints for the arm on the cube's side, the other arm parked.

Waypoints (TCP, metres, rail frame; heights above the table top sim.table_z): above cube -> down -> close -> lift ->
above bowl -> lower -> open -> retreat.
Every IK call holds the approach straight down and the finger closing axis horizontal along `close_along`
(so the fingers straddle the cube instead of closing diagonally on it).
"""
import numpy as np


class Demonstrator:
    def __init__(self, sim, grasp_z=0.05, close_along="x", pre_z=0.18, carry_z=0.22, release_z=0.13, hold=20):
        self.sim, self.grasp_z, self.close_along = sim, grasp_z, close_along
        self.pre_z, self.carry_z, self.release_z, self.hold = pre_z, carry_z, release_z, hold

    def plan(self, info):
        """[(q7 target, gripper target, steps)] for the working arm of this episode."""
        arm, (cx, cy), (bx, by) = info["arm"], info["cube_xy"], info["bowl_xy"]
        A, z0 = self.sim.arms[arm], getattr(self.sim, "table_z", 0.0)
        pts = [((cx, cy, z0 + self.pre_z), 0, 30), ((cx, cy, z0 + self.grasp_z), 0, 25), ((cx, cy, z0 + self.grasp_z), 1, self.hold),
               ((cx, cy, z0 + self.carry_z), 1, 25), ((bx, by, z0 + self.carry_z), 1, 35), ((bx, by, z0 + self.release_z), 1, 15),
               ((bx, by, z0 + self.release_z), 0, 15), ((bx, by, z0 + self.carry_z), 0, 20)]
        q_prev, out = self.sim.home_q(arm), []
        for p, g, n in pts:
            q, err, cerr = A.solve(np.array(p), q0=q_prev, close_along=self.close_along)
            if err > 0.01:
                raise RuntimeError(f"IK failed for {arm} at {p}: position error {err:.3f} m")
            out.append((q, g, n))
            q_prev = q
        return out

    def rollout(self, info, on_frame=None):
        """Run the plan; returns success. on_frame(rgb, depth, state, action) is called before every step (the
        recorder streams frames straight into its writer, so no second copy of an episode is held in RAM)."""
        sim, arm = self.sim, info["arm"]
        other = "right" if arm == "left" else "left"
        wps = self.plan(info)
        q_cur, g_cur = sim.home_q(arm), 0.0
        n_steps = 0
        for q_t, g_t, n in wps:
            for i in range(1, n + 1):
                u = i / n
                work = np.concatenate([(1 - u) * q_cur + u * q_t, [g_t if u > 0.3 else g_cur]])
                tgt = sim.pack(work, sim.idle_target(other)) if arm == "left" else sim.pack(sim.idle_target(other), work)
                if on_frame is not None:
                    rgb, dep = sim.images()
                    on_frame(rgb, dep, sim.state(), tgt)
                sim.step(tgt)
                n_steps += 1
            q_cur, g_cur = q_t, g_t
        return sim.success()


class DrawerDemonstrator(Demonstrator):
    """Scripted drawer opening (scenario open_drawer; the drawer is DRAWER_JOINT in duo_env), same rollout / recording interface as Demonstrator.

    Waypoints (TCP, rail frame): in front of the handle -> onto the handle bar -> close -> pull straight back in
    `steps` IK steps -> open -> back off. The approach is horizontal (+x, towards the cabinet) and the fingers close
    vertically around the bar. Each arm grasps the bar `side_off` towards its own side of its centre."""

    def __init__(self, sim, pull=0.25, pre=0.10, inset=0.0, side_off=0.04, hold=15, steps=6):
        self.sim, self.pull, self.pre, self.inset, self.side_off, self.hold, self.steps = sim, pull, pre, inset, side_off, hold, steps

    def plan(self, info):
        arm = info["arm"]
        hx, hy, hz = info["handle"]
        y = hy + (self.side_off if arm == "left" else -self.side_off)
        gx = hx + self.inset
        pts = [((gx - self.pre, y, hz), 0, 40), ((gx, y, hz), 0, 25), ((gx, y, hz), 1, self.hold)]
        pts += [((gx - self.pull * i / self.steps, y, hz), 1, 10) for i in range(1, self.steps + 1)]
        pts += [((gx - self.pull, y, hz), 0, 12), ((gx - self.pull - 0.08, y, hz), 0, 20)]
        A, q_prev, out = self.sim.arms[arm], self.sim.home_q(arm), []
        for p, g, n in pts:
            q, err, cerr = A.solve(np.array(p), q0=q_prev, close_along="z", approach=(1.0, 0.0, 0.0))
            if err > 0.01:
                raise RuntimeError(f"IK failed for {arm} at {p}: position error {err:.3f} m")
            out.append((q, g, n))
            q_prev = q
        return out
