"""Record RGB-D demonstrations of the prismatic FR3 duo in Isaac Sim into the defmVLA episode format.

    python sim/record_duo.py --out data/normal_train --episodes 50 --seed 0
    python sim/record_duo.py --out data/low_contrast_train --condition low_contrast --episodes 50 --seed 0
    python sim/record_duo.py --out data/drawer_smoke --scenario open_drawer --episodes 2 --preview data/drawer_preview

`python` = the Isaac Sim environment (conda or venv) with `pip install --no-deps -e <defmVLA>` done, and
`python sim/build_usd.py` run once on this machine. See sim/duo_env.py for the state / action layout.

Seeds: attempt k uses seed (--seed + k), which fixes the working arm and cube position, so the same seeds under
"normal" and "low_contrast" give the same task layouts. Keep train and validation seed ranges disjoint
(e.g. train 0.., val 100000..). Every attempt is logged in <out>/recording_log.jsonl; only successes are saved.
The scripted demonstrator moves one arm (left or right, by seed) and parks the other.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=5, help="successful episodes to record")
ap.add_argument("--max-attempts", type=int, default=0, help="0 = 4 x episodes")
ap.add_argument("--seed", type=int, default=0, help="first attempt seed")
ap.add_argument("--variant", default="actual", choices=["actual", "upright"], help="actual = arms tilted 45 deg (the real rig)")
ap.add_argument("--condition", default="normal", choices=["normal", "low_contrast"])
ap.add_argument("--arm", default="random", choices=["random", "left", "right"])
ap.add_argument("--square", type=int, default=640)
ap.add_argument("--scenario", default="pick_place", choices=["pick_place", "pick_place_camo", "open_drawer"])
ap.add_argument("--task", default=None, help="language instruction (default: the scenario's, see sim/duo_env.py SCENARIOS)")
ap.add_argument("--cameras-yaml", default=None)
ap.add_argument("--with-rail", action="store_true", help="add the rail position as a 17th state / action entry (held at 0)")
ap.add_argument("--grasp-z", type=float, default=0.05)
ap.add_argument("--close-along", default="x", choices=["x", "y"])
ap.add_argument("--preview", default=None, help="directory for RGB + depth previews of the first attempt")
a = ap.parse_args()

from isaacsim import SimulationApp

_g = os.environ.get("DUO_GPU")   # pin Isaac (renderer + physics) to one GPU: DUO_GPU=1
GPU = {"active_gpu": int(_g), "physics_gpu": int(_g), "multi_gpu": False} if _g else {}

app = SimulationApp({**GPU, "headless": True, "width": 1280, "height": 720})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from defmvla.episodes import EpisodeWriter
from demonstrator import Demonstrator, DrawerDemonstrator
from duo_env import CAMS, DuoSim

sim = DuoSim(variant=a.variant, condition=a.condition, cameras_yaml=a.cameras_yaml, with_rail=a.with_rail, scenario=a.scenario)
a.task = a.task or sim.task
demo = (DrawerDemonstrator(sim) if a.scenario == "open_drawer"
        else Demonstrator(sim, grasp_z=a.grasp_z, close_along=a.close_along))
w = EpisodeWriter(a.out, cameras=CAMS, depth_cameras=CAMS, fps=sim.fps, square=a.square)
sim.dump_camera_json(a.out, a.square)
log = open(os.path.join(a.out, "recording_log.jsonl"), "a")
saved, tries, max_tries, t0 = 0, 0, (a.max_attempts or 4 * a.episodes), time.time()
while saved < a.episodes and tries < max_tries:
    seed = a.seed + tries
    tries += 1
    info = sim.reset(seed, arm=None if a.arm == "random" else a.arm)
    w.start()
    shots, n = {}, [0]

    def on_frame(rgb, dep, state, action):                         # stream into the writer (square crop + 16-bit depth)
        w.add_frame(rgb=rgb, depth=dep, state=state, action=action)
        if a.preview and tries == 1:
            if n[0] in (0, 90):
                shots[n[0]] = (rgb, dep)
            shots["last"] = (rgb, dep)
        n[0] += 1

    try:
        ok = demo.rollout(info, on_frame=on_frame)
    except RuntimeError as e:                                      # IK failure: counts as a failed attempt
        print(f"attempt {tries} seed {seed}: {e}", flush=True)
        ok = False
    out = sim.outcome()
    print(f"attempt {tries} seed {seed} arm {info['arm']}: {'SUCCESS' if ok else 'fail'} steps={n[0]} {out}", flush=True)
    log.write(json.dumps(dict(seed=seed, arm=info["arm"], success=ok, scenario=a.scenario, condition=a.condition,
                              saved_as=saved if ok else None, **out)) + "\n")
    log.flush()
    if shots:
        from PIL import Image
        os.makedirs(a.preview, exist_ok=True)
        for k, (rgb, dep) in shots.items():
            for c in CAMS:
                Image.fromarray(rgb[c]).save(f"{a.preview}/{k}_{c}_rgb.png")
                np.save(f"{a.preview}/{k}_{c}_depth.npy", dep[c])
    if ok:
        w.save(task=a.task)
        saved += 1
    else:
        w.discard()
print(f"SAVED {saved}/{tries} attempts -> {a.out}  ({time.time() - t0:.0f}s)")
app.close()
