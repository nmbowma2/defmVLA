#!/usr/bin/env python3
"""Paired (same-seed) comparison of the closed-loop results written by run_pipeline.sh and run_ablation.sh.

  python scripts/compare_rollouts.py open_drawer_vlmlora low_contrast
"""
import json
import os
import sys
from math import comb

PAIRS = [("rgbd", "rgbd_repeat", "identical runs: flips are rollout noise"), ("rgbd", "rgbd_off", "depth on vs off"),
         ("rgbd_repeat", "rgbd_off", "depth on (repeat) vs off"), ("rgbd", "rgb_same", "RGB-D vs RGB, same training step"),
         ("rgb", "rgb_same", "RGB: selected vs RGB-D's step"), ("rgb", "rgbd", "original comparison")]


def mcnemar(a, b):
    """a, b: {seed: bool}. Returns (only a succeeds, only b succeeds, exact two-sided p) over the common seeds."""
    ks = a.keys() & b.keys()
    x, y = sum(a[k] and not b[k] for k in ks), sum(b[k] and not a[k] for k in ks)
    n = x + y
    return x, y, min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n) if n else 1.0


def main():
    exp, conds = sys.argv[1], sys.argv[2:] or ["low_contrast"]
    for c in conds:
        print(f"== {exp} {c}")
        ok = {}
        for m in ("rgb", "rgb_same", "rgbd", "rgbd_repeat", "rgbd_off"):
            f = f"results/closed_{exp}_{m}_{c}.json"
            if not os.path.exists(f):
                print(f"{m:12s} missing")
                continue
            d = json.load(open(f))
            s = d["summary"]
            ok[m] = {e["seed"]: bool(e["success"]) for e in d["episodes"]}
            print(f"{m:12s} {s['successes']:>3d}/{s['episodes']:<3d} {s['by_arm']}  step {s['step']} use_depth {s['use_depth']}")
        for a, b, what in PAIRS:
            if a in ok and b in ok:
                x, y, p = mcnemar(ok[a], ok[b])
                print(f"  {what}: only {a} {x}, only {b} {y}, McNemar p {p:.4f}")


if __name__ == "__main__":
    main()
