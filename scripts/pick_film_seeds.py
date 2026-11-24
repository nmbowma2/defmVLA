#!/usr/bin/env python3
"""Print seeds worth filming, from the paired scores: per arm up to 2 where only RGB-D won, then 1 only-RGB, both-fail, both-win.

  python scripts/pick_film_seeds.py open_drawer_vlmlora low_contrast
"""
import json
import sys

exp, cond = sys.argv[1:3]
A, B = ({e["seed"]: e for e in json.load(open(f"results/closed_{exp}_{m}_{cond}.json"))["episodes"]} for m in ("rgb", "rgbd"))
out = []
for arm in ("right", "left"):
    for want, k in (((False, True), 2), ((True, False), 1), ((False, False), 1), ((True, True), 1)):
        out += [s for s in sorted(A.keys() & B.keys()) if A[s]["arm"] == arm and (A[s]["success"], B[s]["success"]) == want][:k]
print(*out)
