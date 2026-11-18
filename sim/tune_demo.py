"""Measure the scripted demonstrator's success rate (no images saved) for several settings.

    python sim/tune_demo.py --seeds 8 --configs y:0.065 y:0.05 x:0.05      # close_along:grasp_z
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--configs", nargs="+", default=["y:0.05"])
ap.add_argument("--variant", default="actual", choices=["actual", "upright"])
ap.add_argument("--condition", default="normal")
ap.add_argument("--arms", nargs="+", default=["left", "right"])
a = ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 640, "height": 480})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demonstrator import Demonstrator
from duo_env import DuoSim

sim = DuoSim(variant=a.variant, condition=a.condition)
for cfg in a.configs:
    ca, gz = cfg.split(":")
    demo = Demonstrator(sim, grasp_z=float(gz), close_along=ca)
    for arm in a.arms:
        ok, t0, fails = 0, time.time(), []
        for s in range(a.seed0, a.seed0 + a.seeds):
            info = sim.reset(s, arm=arm)
            try:
                success = demo.rollout(info)
            except RuntimeError as e:
                print("   ", e, flush=True)
                success = False
            ok += success
            if not success:
                fails.append((s, [round(float(v), 2) for v in sim.cube_pos()]))
        print(f"RESULT close_along={ca} grasp_z={gz} arm={arm}: {ok}/{a.seeds}  ({time.time() - t0:.0f}s)  fails(seed, cube_end)={fails}", flush=True)
app.close()
