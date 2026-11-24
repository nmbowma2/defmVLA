#!/usr/bin/env python3
"""RGB vs RGB-D stills: one picture per seed and camera, models as rows, time as columns (needs the output of run_film.sh; no GPU).

  python scripts/contact_sheet.py videos/film_open_drawer_vlmlora_low_contrast      # dir holding rgb/ and rgbd/
"""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw

root, T, N = sys.argv[1], 256, 6
logs = {m: json.load(open(f"{root}/{m}/rollouts.json")) for m in ("rgb", "rgbd") if os.path.exists(f"{root}/{m}/rollouts.json")}
cams = json.load(open(f"{root}/{next(iter(logs))}/meta.json"))["cameras"]
os.makedirs(f"{root}/stills", exist_ok=True)
for i in range(max(map(len, logs.values()))):
    for cam in cams:
        have = [m for m in logs if i < len(logs[m])]
        im = Image.new("RGB", (N * T, len(have) * (T + 20)), "white")
        d = ImageDraw.Draw(im)
        for r, m in enumerate(have):
            e = logs[m][i]
            fs = sorted(glob.glob(f"{root}/{m}/episode_{i:06d}/rgb.{cam}/*.png"))
            d.text((4, r * (T + 20) + 4), f"{m}  seed {e['seed']}  {e['arm']} arm  {cam}  {'SUCCESS' if e['success'] else 'FAIL'}"
                   f" (scored {'success' if e['scored_success'] else 'fail'}), {e['steps']} steps", fill="black")
            for c in range(N):
                f = fs[round(c * (len(fs) - 1) / (N - 1))]
                im.paste(Image.open(f).convert("RGB").resize((T, T)), (c * T, r * (T + 20) + 20))
        e = logs[have[0]][i]
        im.save(f"{root}/stills/seed{e['seed']}_{e['arm']}_{cam}.png")
print(f"wrote {root}/stills")
