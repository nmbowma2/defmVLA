"""Isaac Sim environment for the prismatic FR3 duo (16-DoF bimanual, rail held fixed): table, cube + bowl, three
RGB-D cameras. Import AFTER creating the SimulationApp.

Layout (the same for state and action, so recorded data and rollouts agree):
    [left q1..q7, left gripper, right q1..q7, right gripper]   (+ rail position / target as a 17th entry with with_rail)
    gripper 0 = open .. 1 = closed.  state = measured, action = absolute joint-position target (gripper 0 / 1)

Cameras (RGB + planar depth): left_wrist, right_wrist (D405 on the hands), front (D435). Rig: sim/cameras.yaml.
Conditions: "normal", "low_contrast" (dim light, near-identical grey table / objects, neutral floor).
Scenarios (the arm, left or right, is chosen per seed; the other one stays parked):
    pick_place              pick the opaque cube on the arm's side and put it in the bowl on that side
    pick_place_camo         the same with the cube coloured like the table: hard to see in RGB, plain in depth (a glass
                            cube was tried and dropped: Isaac's real-time renderer draws glass dark, i.e. EASIER in RGB)
    open_drawer             open the bottom drawer of NVIDIA's stock Sektion cabinet (the asset of Isaac Lab's
                            Isaac-Open-Drawer-Franka-v0), loaded from the Isaac asset server; success = drawer open >= 15 cm
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
# Task layout, left-arm side (the right side mirrors y). Placed where the calibrated front (ego) camera sees the table:
# its training crop covers the table from x ~0.40 (bottom edge) outwards; the calibration board lay at x ~0.63.
# Heights are above the table top, which is at scene.table_z in cameras.yaml (the real table, from the calibration).
CUBE_NOM, BOWL_NOM = (0.60, 0.12), (0.62, 0.30)
HOME_NOM = (0.42, 0.32, 0.30)        # working arm's start: TCP x, y (mirrored), height above the table; gripper down. Chosen so the
                                     # arm starts (nearly) out of the front camera's crop and does not hide the cube or bowl
HOME_DRAWER = (0.28, 0.32, 0.30)     # open_drawer start: further back, so the arm starts clear of the cabinet front (x >= 0.38)
TABLE_X = (0.18, 0.86)               # table extent in x: starts in front of the robot base (base box spans x +-0.10) and
                                     # ends where the real table does (x ~0.855, back-projected from the ego photos)
TABLE_X_DRAWER = (0.18, 1.30)        # open_drawer: longer, the cabinet (x up to ~1.1) stands on it (sim only, no real cabinet)
# The real work surface (ego calibration photos): a near-black mat on a white table. Mat extent back-projected from those
# photos through the calibrated ego camera; its edge nearest the robot is out of view and assumed to reach the table start.
MAT_X, MAT_Y = (0.18, 0.78), (-0.29, 0.47)

# cabinet = None keeps the stock cabinet look, a colour paints all of it (handle too). mat = None: no mat (uniform table).
# pick_place_camo gives the cube the colour of the surface it stands on (the mat, else the table).
# normal table / mat albedos are tuned so the rendered ego image matches the real photos (white table ~(176,174,175),
# mat ~(32,38,36) in 8-bit RGB): rendered (177,174,176) and (33,39,36). The mat is matte (roughness 0.95, ior 1.1): with the
# default specular it cannot render darker than ~59, whatever its albedo.
CONDITIONS = {
    "normal": dict(dome=900.0, table=(0.174, 0.163, 0.170), mat=(0.017, 0.020, 0.019), cube=(0.85, 0.10, 0.10), bowl=(0.10, 0.25, 0.85),
                   neutral_floor=False, cabinet=None),
    "low_contrast": dict(dome=250.0, table=(0.46, 0.46, 0.46), mat=None, cube=(0.50, 0.49, 0.49), bowl=(0.41, 0.43, 0.46),
                         neutral_floor=True, cabinet=(0.47, 0.47, 0.47)),
}

SCENARIOS = {
    "pick_place": dict(task="pick up the cube and place it in the bowl"),
    "pick_place_camo": dict(task="pick up the cube and place it in the bowl"),
    "open_drawer": dict(task="open the bottom drawer of the cabinet"),
}
CABINET_USD = "/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"   # relative to the Isaac asset root
CAB_NOM = (0.74, 0.0)            # cabinet origin xy (drawers face the robot, -x). 0.74 +- 0.04 puts the bottom-drawer handle on
                                 # rows ~66-111 of 480 in the calibrated front camera (0.88 put it on the top edge)
CAB_Z = 0.4                      # cabinet origin above the surface it stands on (the table)
# Which drawer: the bottom one. With the calibrated ego camera (sim/cameras.yaml) and the cabinet on the real table
# height, the top-drawer handle (z ~ 0.80) is above the camera's view; the bottom one is in it.
DRAWER_JOINT = "drawer_bottom_joint"
HANDLE_OFF = np.array([-0.356, 0.0, 0.198])   # bottom-drawer handle bar centre from the cabinet origin (drawer closed)
#                      top drawer: joint "drawer_top_joint", handle offset (-0.356, 0, 0.327)
DRAWER_SUCCESS_M = 0.15                        # drawer travel 0..0.40 m


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


def _cv_to_usd(T):
    """4x4 pose of an OpenCV optical frame (x right, y down, z forward) -> the same camera as a USD camera
    (x right, y up, looks down -z)."""
    T = np.array(T, float)
    T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return T


def _add_camera(stage, path, M):
    cam = UsdGeom.Camera.Define(stage, path)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
    return path


def _surface(obj, roughness, ior=None):
    """Set roughness (and ior: specular strength) on an isaacsim object's preview-surface material."""
    sh = obj.get_applied_visual_material().shaders_list[0]
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if ior is not None:
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)


def _paint(stage, root, rgb, name):
    """Bind one colour to everything under root (strongerThanDescendants also reaches instanceable meshes)."""
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


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
    def __init__(self, variant="actual", condition="normal", cameras_yaml=None, with_rail=False, scenario="pick_place"):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {list(SCENARIOS)}")
        self.variant, self.condition, self.with_rail, self.scenario = variant, condition, with_rail, scenario
        self.task = SCENARIOS[scenario]["task"]
        self.cfg = yaml.safe_load(open(cameras_yaml or HERE / "cameras.yaml"))
        self.fps, self.substeps = int(self.cfg["sim"]["fps"]), int(self.cfg["sim"]["substeps"])
        self.table_z = tz = float(self.cfg.get("scene", {}).get("table_z", 0.0))     # table top height (world z)
        cond = CONDITIONS[condition]
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / (self.fps * self.substeps), rendering_dt=1 / self.fps)
        stage = self.stage = get_current_stage()
        self.world.scene.add_default_ground_plane()
        self.table_x = tx = TABLE_X_DRAWER if scenario == "open_drawer" else TABLE_X
        table = FixedCuboid("/World/table", name="table", position=np.array([sum(tx) / 2, 0.0, (tz - 0.05) / 2]),
                    scale=np.array([tx[1] - tx[0], 1.2, tz + 0.05]), color=np.array(cond["table"]))
        _surface(table, roughness=0.8)
        if cond["mat"] is not None:                                # 1 mm mat, its top 0.5 mm above the table top
            mat = FixedCuboid("/World/mat", name="mat", position=np.array([sum(MAT_X) / 2, sum(MAT_Y) / 2, tz]),
                        scale=np.array([MAT_X[1] - MAT_X[0], MAT_Y[1] - MAT_Y[0], 0.001]), color=np.array(cond["mat"]))
            _surface(mat, roughness=0.95, ior=1.1)
        if cond["neutral_floor"]:                                  # hide the blue grid floor
            for p in stage.Traverse():
                if str(p.GetPath()).startswith("/World/defaultGroundPlane") and p.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(p).MakeInvisible()
            FixedCuboid("/World/floor", name="floor", position=np.array([0.45, 0.0, -0.11]), scale=np.array([8.0, 8.0, 0.1]),
                        color=np.array(cond["table"]) * 0.9)
        UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(cond["dome"])
        self.cube, self.bowl, self.cabinet, self.bowl_xy = None, [], None, None
        if scenario == "open_drawer":
            from isaacsim.storage.native import get_assets_root_path
            assets = get_assets_root_path()
            if not assets:
                raise RuntimeError("open_drawer needs the Isaac asset server (stock Sektion cabinet); none is reachable")
            add_reference_to_stage(assets + CABINET_USD, "/World/cabinet")
            xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/cabinet"))
            xf.ClearXformOpOrder()
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(CAB_NOM[0], CAB_NOM[1], tz + CAB_Z))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.0, 0.0, 0.0, 1.0))      # 180 deg about z
            if cond["cabinet"] is not None:
                _paint(stage, "/World/cabinet", cond["cabinet"], "cabinet_paint")
            self.cabinet = self.world.scene.add(SingleArticulation(prim_path="/World/cabinet", name="cabinet"))
        else:
            self.cube = DynamicCuboid("/World/cube", name="cube", position=np.array([*CUBE_NOM, tz + 0.025]), scale=np.full(3, 0.05),
                                      color=np.array(cond["cube"] if scenario != "pick_place_camo" else
                                                     cond["mat"] if cond["mat"] is not None else cond["table"]), mass=0.08)
            if scenario == "pick_place_camo":                        # same surface as what it stands on
                _surface(self.cube, **(dict(roughness=0.95, ior=1.1) if cond["mat"] is not None else dict(roughness=0.8)))
            for i, (p, s) in enumerate([((0, 0, 0.005), (0.16, 0.16, 0.01)), ((0.075, 0, 0.03), (0.01, 0.16, 0.05)),
                                        ((-0.075, 0, 0.03), (0.01, 0.16, 0.05)), ((0, 0.075, 0.03), (0.16, 0.01, 0.05)),
                                        ((0, -0.075, 0.03), (0.16, 0.01, 0.05))]):
                o = FixedCuboid(f"/World/bowl_{i}", name=f"bowl_{i}", position=np.array([BOWL_NOM[0], BOWL_NOM[1], tz]) + np.array(p),
                                scale=np.array(s), color=np.array(cond["bowl"]))
                self.bowl.append((o, np.array(p, float)))             # (object, offset from the bowl centre)
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
        # calibrated pose (sim/calibration_to_cameras.py) when present, else eye / target
        paths["front"] = _add_camera(stage, "/World/front_cam", _cv_to_usd(f["T_world_cam"]) if "T_world_cam" in f
                                     else _look_at(f["eye"], f["target"]))
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
        if self.cabinet is not None:
            # the task drawer: free, damped (stays where it is pulled to); the other drawer and the doors held shut
            cn = list(self.cabinet.dof_names)
            self.drawer = cn.index(DRAWER_JOINT)
            ckp, ckd = np.full(len(cn), 1.0e4), np.full(len(cn), 1.0e3)
            ckp[self.drawer], ckd[self.drawer] = 0.0, 20.0
            self.cabinet.get_articulation_controller().set_gains(kps=ckp, kds=ckd)
        self.arms = {s: Arm(s, variant=variant) for s in ("left", "right")}
        self._home = {}
        self.state_dim = 16 + int(with_rail)
        for _ in range(10):
            self.world.step(render=True)

    # ---------- geometry helpers
    def _wrist_local(self, side):
        """D405 pose in the link7 frame as a USD camera (-Z forward, +Y up); fingers at the bottom of the frame.
        With wrist.T_flange_cam (the hand-eye calibration, flange = fr3_link8) that pose is used; otherwise the URDF
        d405 link + wrist.offset_xyz."""
        c7 = Chain(f"{side}_fr3_link7", arm_prefix=f"{side}_fr3_joint", variant=self.variant)
        q = np.zeros(7)
        if "T_flange_cam" in self.cfg["wrist"]:
            c8 = Chain(f"{side}_fr3_link8", arm_prefix=f"{side}_fr3_joint", variant=self.variant)
            return np.linalg.inv(c7.fk(q)) @ c8.fk(q) @ _cv_to_usd(self.cfg["wrist"]["T_flange_cam"])
        cd = Chain(f"{side}_d405", arm_prefix=f"{side}_fr3_joint", variant=self.variant)
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
            h = HOME_DRAWER if self.scenario == "open_drawer" else HOME_NOM
            self._home[side] = self.arms[side].solve(np.array([h[0], sgn * h[1], self.table_z + h[2]]))[0]
        return self._home[side]

    # ---------- episode
    def reset(self, seed, arm=None):
        """Randomise by seed: working arm (left / right unless given) and the object pose (cube, or cabinet for
        open_drawer). Returns an info dict for the demonstrator."""
        rng = np.random.default_rng(seed)
        arm = arm or str(rng.choice(["left", "right"]))
        sgn = 1.0 if arm == "left" else -1.0
        if self.scenario == "open_drawer":
            cab_xy = np.array([CAB_NOM[0] + rng.uniform(-0.04, 0.04), CAB_NOM[1] + rng.uniform(-0.03, 0.03)])
            self.cabinet.set_world_pose(position=np.array([*cab_xy, self.table_z + CAB_Z]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
            z = np.zeros(len(self.cabinet.dof_names))
            self.cabinet.set_joint_positions(z)
            self.cabinet.set_joint_velocities(z)
            self.cabinet.apply_action(ArticulationAction(joint_positions=z))
            extra = dict(cab_xy=cab_xy, handle=np.array([*cab_xy, self.table_z + CAB_Z]) + HANDLE_OFF)
        else:
            cube_xy = np.array([CUBE_NOM[0] + rng.uniform(-0.04, 0.04), sgn * (CUBE_NOM[1] + rng.uniform(-0.04, 0.04))])
            self.bowl_xy = np.array([BOWL_NOM[0], sgn * BOWL_NOM[1]])
            for o, off in self.bowl:
                o.set_world_pose(position=np.array([self.bowl_xy[0], self.bowl_xy[1], self.table_z]) + off * np.array([1, sgn, 1]))
            self.cube.set_world_pose(position=np.array([*cube_xy, self.table_z + 0.025]), orientation=np.array([1.0, 0, 0, 0]))
            self.cube.set_linear_velocity(np.zeros(3))
            self.cube.set_angular_velocity(np.zeros(3))
            extra = dict(cube_xy=cube_xy, bowl_xy=self.bowl_xy.copy())
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
        return dict(arm=arm, seed=int(seed), **extra)

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

    def drawer_open(self):
        """Task-drawer travel in metres (0 = shut, 0.40 = fully out)."""
        return float(np.asarray(self.cabinet.get_joint_positions())[self.drawer])

    def success(self):
        if self.scenario == "open_drawer":
            return self.drawer_open() >= DRAWER_SUCCESS_M
        p = self.cube_pos()
        return bool(abs(p[0] - self.bowl_xy[0]) < 0.07 and abs(p[1] - self.bowl_xy[1]) < 0.07 and p[2] < self.table_z + 0.07)

    def outcome(self):
        """What to log at the end of an episode (besides success)."""
        if self.scenario == "open_drawer":
            return dict(drawer_open_m=round(self.drawer_open(), 3))
        return dict(cube_final=[round(float(v), 3) for v in self.cube_pos()])

    def dump_camera_json(self, root, square):
        os.makedirs(root, exist_ok=True)
        json.dump(dict(scenario=self.scenario, task=self.task, table_z=self.table_z, layout=dict(
                       cube_nom=CUBE_NOM, bowl_nom=BOWL_NOM, home_nom=HOME_NOM, table_x=self.table_x, mat_x=MAT_X, mat_y=MAT_Y), variant=self.variant, condition=self.condition, with_rail=self.with_rail, fps=self.fps,
                       rig=self.cfg, intrinsics=self.camera_info(square), state_action_layout=
                       "[left q1..q7, left gripper, right q1..q7, right gripper" + (", rail]" if self.with_rail else "]")),
                  open(os.path.join(root, "cameras.json"), "w"), indent=1)
