"""Isaac Sim environment for the prismatic FR3 duo (16-DoF bimanual, rail held fixed): table, cube + bowl, three
RGB-D cameras. Import AFTER creating the SimulationApp.

Layout (the same for state and action, so recorded data and rollouts agree):
    [left q1..q7, left gripper, right q1..q7, right gripper]   (+ rail position / target as a 17th entry with with_rail)
    gripper 0 = open .. 1 = closed.  state = measured, action = absolute joint-position target (gripper 0 / 1)

Cameras (RGB + planar depth): left_wrist, right_wrist (D405 on the hands), front (D435). Rig: sim/cameras.yaml.
Conditions: "normal", "low_contrast" (dim light, near-identical grey table / cube / bowl, neutral floor).
Task: pick the cube with the arm on its side (left/right chosen per seed) and put it in the bowl on that side.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from duo_kin import Arm, Chain, urdf_path

HERE = Path(__file__).resolve().parent
CAMS = ["left_wrist", "right_wrist", "front"]
ARM_DOF = ["fr3_joint%d" % i for i in range(1, 8)]
FINGERS = ["fr3_finger_joint1", "fr3_finger_joint2"]
PARK = np.array([0, -0.5, 0, -2.4, 0, 1.9, 0.785])      # neutral pose of the arm that is not working
CUBE_NOM, BOWL_NOM = (0.42, 0.12), (0.45, 0.30)          # left-arm side; the right side mirrors y

CONDITIONS = {
    "normal": dict(dome=900.0, table=(0.62, 0.50, 0.38), cube=(0.85, 0.10, 0.10), bowl=(0.10, 0.25, 0.85), neutral_floor=False),
    "low_contrast": dict(dome=250.0, table=(0.46, 0.46, 0.46), cube=(0.50, 0.49, 0.49), bowl=(0.41, 0.43, 0.46), neutral_floor=True),
}


def usd_path(variant):
    hits = sorted(glob.glob(str(HERE / "robot" / "usd" / variant / "**" / f"duo_{variant}.usd*"), recursive=True))
    hits = [h for h in hits if "payloads" not in h]
    if not hits:
        raise FileNotFoundError(f"no USD for variant '{variant}': run `python sim/build_usd.py` (needs Isaac Sim)")
    return hits[0]


def _focal(hfov_deg, aperture=20.955):
    return aperture / (2 * np.tan(np.radians(hfov_deg) / 2))


def _look_at(eye, target, up=(0, 0, 1.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, eye        # USD camera looks down -Z, up is +Y
    return M


def _add_camera(stage, path, M):
    cam = UsdGeom.Camera.Define(stage, path)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
    return path


SOFT_FINGER_RGB = (0.80, 0.05, 0.05)   # the real soft fingers are red; the URDF importer drops the URDF colour


def _paint_soft_fingers(stage, root):
    """Bind a red material to every soft-finger mesh under root (matched by link or mesh name, since fixed joints
    are merged at import). Returns the number of prims painted; prints a warning when none matched."""
    mat = UsdShade.Material.Define(stage, "/World/Looks/soft_finger_red")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/soft_finger_red/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*SOFT_FINGER_RGB))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    # Bind on the first matching prim of each branch (link, visual or mesh). Instanceable visuals cannot be edited
    # inside, but a binding on the instance prim (strongerThanDescendants) reaches its meshes.
    painted = []
    for p in stage.Traverse():
        path = str(p.GetPath())
        low = path.lower()
        if not low.startswith(root.lower()) or not ("soft_finger" in low or "mount_gripper" in low):
            continue
        if any(path.startswith(q + "/") for q in painted):
            continue
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        if p.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(p).CreateDisplayColorAttr([Gf.Vec3f(*SOFT_FINGER_RGB)])
        painted.append(path)
    if painted:
        print(f"soft fingers painted red: {len(painted)} prims, e.g. {painted[:2]}", flush=True)
    else:
        print("WARNING: no soft-finger prims found to paint red", flush=True)
    return len(painted)


class DuoSim:
    def __init__(self, variant="actual", condition="normal", cameras_yaml=None, with_rail=False):
        self.variant, self.condition, self.with_rail = variant, condition, with_rail
        self.cfg = yaml.safe_load(open(cameras_yaml or HERE / "cameras.yaml"))
        self.fps, self.substeps = int(self.cfg["sim"]["fps"]), int(self.cfg["sim"]["substeps"])
        cond = CONDITIONS[condition]
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / (self.fps * self.substeps), rendering_dt=1 / self.fps)
        stage = self.stage = get_current_stage()
        self.world.scene.add_default_ground_plane()
        FixedCuboid("/World/table", name="table", position=np.array([0.45, 0.0, -0.025]), scale=np.array([1.4, 1.2, 0.05]),
                    color=np.array(cond["table"]))
        if cond["neutral_floor"]:                                  # hide the blue grid floor
            for p in stage.Traverse():
                if str(p.GetPath()).startswith("/World/defaultGroundPlane") and p.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(p).MakeInvisible()
            FixedCuboid("/World/floor", name="floor", position=np.array([0.45, 0.0, -0.11]), scale=np.array([8.0, 8.0, 0.1]),
                        color=np.array(cond["table"]) * 0.9)
        UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(cond["dome"])
        self.cube = DynamicCuboid("/World/cube", name="cube", position=np.array([*CUBE_NOM, 0.025]), scale=np.full(3, 0.05),
                                  color=np.array(cond["cube"]), mass=0.08)
        self.bowl = []                                             # (object, offset from the bowl centre)
        for i, (p, s) in enumerate([((0, 0, 0.005), (0.16, 0.16, 0.01)), ((0.075, 0, 0.03), (0.01, 0.16, 0.05)),
                                    ((-0.075, 0, 0.03), (0.01, 0.16, 0.05)), ((0, 0.075, 0.03), (0.16, 0.01, 0.05)),
                                    ((0, -0.075, 0.03), (0.16, 0.01, 0.05))]):
            o = FixedCuboid(f"/World/bowl_{i}", name=f"bowl_{i}", position=np.array([BOWL_NOM[0], BOWL_NOM[1], 0.0]) + np.array(p),
                            scale=np.array(s), color=np.array(cond["bowl"]))
            self.bowl.append((o, np.array(p, float)))
        add_reference_to_stage(usd_path(variant), "/World/duo")
        prim = stage.GetPrimAtPath("/World/duo")
        prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        self.red_fingers = _paint_soft_fingers(stage, "/World/duo")
        self.robot = self.world.scene.add(SingleArticulation(prim_path="/World/duo/Geometry/rail_link", name="duo"))
        # cameras
        w, f = self.cfg["wrist"], self.cfg["front"]
        paths = {}
        for side in ("left", "right"):
            chain = "/".join(["/World/duo/Geometry/rail_link/base_link"] + [f"{side}_fr3_link{i}" for i in range(1, 8)])
            paths[f"{side}_wrist"] = _add_camera(stage, chain + "/wrist_cam", self._wrist_local(side))
        paths["front"] = _add_camera(stage, "/World/front_cam", _look_at(f["eye"], f["target"]))
        self.cams, self.cam_cfg = {}, {"left_wrist": w, "right_wrist": w, "front": f}
        self.world.reset()
        for n in CAMS:
            c = self.cams[n] = Camera(prim_path=paths[n], name=n, resolution=tuple(self.cam_cfg[n]["resolution"]))
            c.initialize()
            c.set_focal_length(_focal(self.cam_cfg[n]["hfov_deg"]) / 10.0)     # API takes cm-scaled units
            c.set_clipping_range(*self.cam_cfg[n]["clip"])
            c.add_distance_to_image_plane_to_frame()
        names = list(self.robot.dof_names)
        self.names = names
        self.idx = {s: dict(arm=[names.index(f"{s}_{n}") for n in ARM_DOF], fingers=[names.index(f"{s}_{n}") for n in FINGERS])
                    for s in ("left", "right")}
        self.slider = names.index("base_slider_joint")
        kp, kd = np.full(len(names), 800.0), np.full(len(names), 80.0)
        for i, nm in enumerate(names):
            if "finger_joint" in nm:
                kp[i], kd[i] = 2000.0, 100.0
        kp[self.slider], kd[self.slider] = 20000.0, 2000.0
        self.robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
        self.arms = {s: Arm(s, variant=variant) for s in ("left", "right")}
        self._home = {}
        self.state_dim = 16 + int(with_rail)
        for _ in range(10):
            self.world.step(render=True)

    # ---------- geometry helpers
    def _wrist_local(self, side):
        """D405 pose in the link7 frame as a USD camera (-Z forward, +Y up); fingers at the bottom of the frame."""
        c7 = Chain(f"{side}_fr3_link7", arm_prefix=f"{side}_fr3_joint", variant=self.variant)
        cd = Chain(f"{side}_d405", arm_prefix=f"{side}_fr3_joint", variant=self.variant)
        q = np.zeros(7)
        T = np.linalg.inv(c7.fk(q)) @ cd.fk(q)
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]]).T
        cam = np.eye(4)
        cam[:3, :3], cam[:3, 3] = R, self.cfg["wrist"]["offset_xyz"]
        return T @ cam

    def camera_info(self, square):
        """Intrinsics of every camera as rendered and after EpisodeWriter's centre square crop + resize to `square`."""
        out = {}
        for n in CAMS:
            c = self.cam_cfg[n]
            W, H = c["resolution"]
            fx = W / (2 * np.tan(np.radians(c["hfov_deg"]) / 2))
            s = square / min(W, H)
            out[n] = dict(resolution=[W, H], hfov_deg=c["hfov_deg"], fx=fx, fy=fx, cx=W / 2, cy=H / 2, clip=c["clip"],
                          square=dict(size=square, fx=fx * s, fy=fx * s, cx=square / 2, cy=square / 2,
                                      hfov_deg=float(np.degrees(2 * np.arctan(min(W, H) / 2 / fx)))))
        return out

    def home_q(self, side):
        if side not in self._home:
            sgn = 1 if side == "left" else -1
            self._home[side] = self.arms[side].solve(np.array([0.40, sgn * 0.10, 0.20]))[0]
        return self._home[side]

    # ---------- episode
    def reset(self, seed, arm=None):
        """Randomise by seed: working arm (left / right unless given) and cube position. Returns info dict."""
        rng = np.random.default_rng(seed)
        arm = arm or str(rng.choice(["left", "right"]))
        sgn = 1.0 if arm == "left" else -1.0
        cube_xy = np.array([CUBE_NOM[0] + rng.uniform(-0.04, 0.04), sgn * (CUBE_NOM[1] + rng.uniform(-0.04, 0.04))])
        self.bowl_xy = np.array([BOWL_NOM[0], sgn * BOWL_NOM[1]])
        for o, off in self.bowl:
            o.set_world_pose(position=np.array([self.bowl_xy[0], self.bowl_xy[1], 0.0]) + off * np.array([1, sgn, 1]))
        self.cube.set_world_pose(position=np.array([*cube_xy, 0.025]), orientation=np.array([1.0, 0, 0, 0]))
        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))
        other = "right" if arm == "left" else "left"
        q = np.zeros(len(self.names))
        q[self.idx[arm]["arm"]], q[self.idx[other]["arm"]] = self.home_q(arm), PARK
        q[self.idx[arm]["fingers"]] = q[self.idx[other]["fingers"]] = 0.04
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros_like(q))
        self.robot.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(240):
            self.world.step(render=False)
        for _ in range(4):
            self.world.step(render=True)
        return dict(arm=arm, cube_xy=cube_xy, bowl_xy=self.bowl_xy.copy(), seed=int(seed))

    def idle_target(self, arm):
        """Target for the arm that is not working: stay parked, gripper open."""
        return np.concatenate([PARK, [0.0]])

    def pack(self, left8, right8, rail=0.0):
        v = np.concatenate([left8, right8]).astype(np.float32)
        return np.concatenate([v, [rail]]).astype(np.float32) if self.with_rail else v

    # ---------- observe / act
    def state(self):
        q = np.asarray(self.robot.get_joint_positions())
        out = []
        for s in ("left", "right"):
            g = float(np.clip(1.0 - np.mean(q[self.idx[s]["fingers"]]) / 0.04, 0, 1))
            out += list(q[self.idx[s]["arm"]]) + [g]
        if self.with_rail:
            out.append(float(q[self.slider]))
        return np.array(out, np.float32)

    def images(self):
        rgb, dep = {}, {}
        for n, c in self.cams.items():
            rgb[n] = np.asarray(c.get_rgba())[..., :3].astype(np.uint8)
            d = np.asarray(c.get_current_frame()["distance_to_image_plane"]).astype(np.float32).squeeze()
            d[~np.isfinite(d)] = 0.0                                # no reading -> 0 (the dataset's invalid value)
            dep[n] = d
        return rgb, dep

    def step(self, target):
        full = np.asarray(self.robot.get_joint_positions()).copy()
        for k, s in enumerate(("left", "right")):
            t = np.asarray(target[8 * k: 8 * k + 8], float)
            full[self.idx[s]["arm"]] = t[:7]
            full[self.idx[s]["fingers"]] = 0.04 * (1.0 - np.clip(t[7], 0, 1))
        full[self.slider] = float(target[16]) if self.with_rail and len(target) > 16 else 0.0
        self.robot.apply_action(ArticulationAction(joint_positions=full))
        for k in range(self.substeps):
            self.world.step(render=(k == self.substeps - 1))

    def cube_pos(self):
        return np.asarray(self.cube.get_world_pose()[0])

    def success(self):
        p = self.cube_pos()
        return bool(abs(p[0] - self.bowl_xy[0]) < 0.07 and abs(p[1] - self.bowl_xy[1]) < 0.07 and p[2] < 0.07)

    def dump_camera_json(self, root, square):
        os.makedirs(root, exist_ok=True)
        json.dump(dict(variant=self.variant, condition=self.condition, with_rail=self.with_rail, fps=self.fps,
                       rig=self.cfg, intrinsics=self.camera_info(square), state_action_layout=
                       "[left q1..q7, left gripper, right q1..q7, right gripper" + (", rail]" if self.with_rail else "]")),
                  open(os.path.join(root, "cameras.json"), "w"), indent=1)
