"""Closed-loop evaluation: roll a served policy out in Isaac Sim and report the success rate.

    python scripts/serve_policy.py --checkpoint runs/.../checkpoint_001000_deploy --port 6040      # defmVLA env
    python sim/eval_closed_loop.py --port 6040 --episodes 20 --seed0 10000 --out results/rgbd_normal.json  # Isaac env

Every episode is fully fixed by its seed (working arm, cube position, policy noise), so run the SAME --seed0 /
--episodes for every checkpoint and for both --condition normal and low_contrast. Use seeds that no training or
validation recording used. Success = the cube ends inside the bowl on its side. Policy actions are executed in
chunks of the server's n_action_steps, one action per 1/fps step.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=6040)
ap.add_argument("--host", default="localhost")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed0", type=int, default=10000)
ap.add_argument("--max-steps", type=int, default=240, help="policy steps per episode (the demos take ~180)")
ap.add_argument("--variant", default="actual", choices=["actual", "upright"])
ap.add_argument("--condition", default="normal", choices=["normal", "low_contrast"])
ap.add_argument("--arm", default="random", choices=["random", "left", "right"])
ap.add_argument("--square", type=int, default=640, help="same value the training data was recorded with")
ap.add_argument("--scenario", default="pick_place", choices=["pick_place", "pick_place_camo", "open_drawer"])
ap.add_argument("--task", default=None, help="language instruction (default: the scenario's; must match the training data)")
ap.add_argument("--cameras-yaml", default=None)
ap.add_argument("--with-rail", action="store_true")
ap.add_argument("--replan", type=int, default=0, help="execute only the first N actions of each chunk, then re-query (0 = the "
                "checkpoint's n_action_steps). Smaller = closed-loop corrections more often; no retraining needed")
ap.add_argument("--out", default=None, help="write the per-episode results + summary as JSON")
a = ap.parse_args()

from isaacsim import SimulationApp

_g = os.environ.get("DUO_GPU")   # pin Isaac (renderer + physics) to one GPU: DUO_GPU=1
GPU = {"active_gpu": int(_g), "physics_gpu": int(_g), "multi_gpu": False} if _g else {}

app = SimulationApp({**GPU, "headless": True, "width": 1280, "height": 720})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from defmvla.client import PolicyClient
from duo_env import DuoSim

sim = DuoSim(variant=a.variant, condition=a.condition, cameras_yaml=a.cameras_yaml, with_rail=a.with_rail, scenario=a.scenario)
a.task = a.task or sim.task
pc = PolicyClient(port=a.port, host=a.host, square=a.square)
info = pc.info()
assert info["action_dim"] == sim.state_dim, f"policy action_dim {info['action_dim']} != sim {sim.state_dim} (--with-rail?)"
assert abs(info["fps"] - sim.fps) < 1e-6, f"policy trained at {info['fps']} fps, sim runs at {sim.fps}"
n_exec = int(info["n_action_steps"])
if a.replan > 0:
    n_exec = min(n_exec, a.replan)
print("policy:", {k: info[k] for k in ("cameras", "depth_cameras", "use_depth", "horizon", "n_action_steps", "step")}, flush=True)

results = []
for ep in range(a.episodes):
    seed = a.seed0 + ep
    s = sim.reset(seed, arm=None if a.arm == "random" else a.arm)
    pc.reset(seed=seed)
    t, ok, t_ok, lat = 0, False, None, []
    while t < a.max_steps and not ok:
        rgb, dep = sim.images()
        acts = pc.act(images={c: rgb[c] for c in info["cameras"]}, depth={c: dep[c] for c in info["depth_cameras"]},
                      state=sim.state(), task=f"{s['arm']} arm: {a.task}" if info.get("arm_prompt") else a.task)
        lat.append(pc.latency_s)
        for act in acts[:n_exec]:
            sim.step(act)
            t += 1
            if sim.success():
                ok, t_ok = True, t
                break
            if t >= a.max_steps:
                break
    r = dict(seed=seed, arm=s["arm"], success=ok, steps=t, success_step=t_ok, **sim.outcome(),
             mean_latency_s=round(float(np.mean(lat)), 3))
    results.append(r)
    print("EPISODE", json.dumps(r), flush=True)

n = len(results)
summary = dict(scenario=a.scenario, task=a.task, condition=a.condition, variant=a.variant, episodes=n, seed0=a.seed0, successes=sum(r["success"] for r in results),
               success_rate=round(sum(r["success"] for r in results) / max(n, 1), 3),
               by_arm={arm: f"{sum(r['success'] for r in results if r['arm'] == arm)}/{sum(r['arm'] == arm for r in results)}"
                       for arm in ("left", "right")},
               n_exec=n_exec, checkpoint=info.get("checkpoint"), use_depth=info.get("use_depth"), step=info.get("step"))
print("SUMMARY", json.dumps(summary), flush=True)
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(summary=summary, episodes=results), open(a.out, "w"), indent=1)
app.close()
