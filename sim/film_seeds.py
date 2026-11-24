"""Film closed-loop rollouts of chosen seeds, successes AND failures, as frames for scripts/make_video.py.

Runs next to a policy server; normally started by scripts/run_film.sh. Settings come from the scored result file. A re-run
may end differently from the scored episode (both outcomes are written to rollouts.json), so label clips by what they show.
"""
import argparse
import json
import os
import shutil
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--results", required=True, help="results/closed_*.json of the model being served")
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=6042)
ap.add_argument("--max-steps", type=int, default=240)
a = ap.parse_args()
R = json.load(open(a.results))
S = R["summary"]
scored = {e["seed"]: e["success"] for e in R["episodes"]}

from isaacsim import SimulationApp

_g = os.environ.get("DUO_GPU")
GPU = {"active_gpu": int(_g), "physics_gpu": int(_g), "multi_gpu": False} if _g else {}
app = SimulationApp({**GPU, "headless": True, "width": 1280, "height": 720})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from defmvla.client import PolicyClient
from duo_env import DuoSim

sim = DuoSim(variant=S["variant"], condition=S["condition"], scenario=S["scenario"], cameras_yaml=S.get("cameras_yaml"))
pc = PolicyClient(port=a.port, host="localhost", square=S.get("square", 640))
info = pc.info()
log = []
for i, seed in enumerate(a.seeds):
    s = sim.reset(seed)  # arm=None as in the evaluation: passing the arm skips a random draw and moves the object
    pc.reset(seed=seed)
    task = f"{s['arm']} arm: {S['task']}" if info.get("arm_prompt") else S["task"]
    ep, t, ok = os.path.join(a.out, f"episode_{i:06d}"), 0, False

    shutil.rmtree(ep, ignore_errors=True)  # frames of an interrupted attempt (may be longer)

    def save(rgb, n):
        for c, im in rgb.items():
            os.makedirs(f"{ep}/rgb.{c}", exist_ok=True)
            Image.fromarray(np.asarray(im)[..., :3]).save(f"{ep}/rgb.{c}/{n:06d}.png")

    rgb, dep = sim.images()
    save(rgb, 0)
    while t < a.max_steps and not ok:
        rgb, dep = sim.images()
        acts = pc.act(images={c: rgb[c] for c in info["cameras"]}, depth={c: dep[c] for c in info["depth_cameras"]},
                      state=sim.state(), task=task)
        for act in acts[: S["n_exec"]]:
            sim.step(act)
            t += 1
            save(sim.images()[0], t)
            ok = bool(sim.success())
            if ok or t >= a.max_steps:
                break
    open(f"{ep}/task.txt", "w").write(task)
    log.append(dict(seed=seed, arm=s["arm"], success=ok, scored_success=scored.get(seed), steps=t, **sim.outcome()))
    print("ROLLOUT", json.dumps(log[-1]), flush=True)
    json.dump(dict(cameras=list(rgb), depth_cameras=[], fps=sim.fps), open(f"{a.out}/meta.json", "w"))
    json.dump(log, open(f"{a.out}/rollouts.json", "w"), indent=1)
app.close()
