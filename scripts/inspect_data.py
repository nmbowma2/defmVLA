#!/usr/bin/env python3
"""Look at a recorded dataset before training: RGB + depth grid, depth validity / range per camera, action stats.

  python scripts/inspect_data.py --root data/duo_train --episode 0 --out runs/inspect.png

Depth panels are coloured near = red .. far = blue between --near and --far metres; black = no reading (0).
Numbers are over every `--stride`-th frame of every episode, loaded exactly as training loads them.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from defmvla.data import EpisodeDataset, check_roots, fit_stats, load_low_dim  # noqa: E402
from defmvla.episodes import depth_u16_to_m  # noqa: E402


def colour(d, near, far):
    n = np.clip((d - near) / (far - near), 0, 1)
    x = np.stack([1 - n, 1 - np.abs(2 * n - 1), n], -1)
    x = (255 * x).astype(np.uint8)
    x[d <= 0] = 0
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", nargs="+", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=float, nargs="+", default=[0.0, 0.5, 1.0], help="fractions of the episode to show")
    ap.add_argument("--out", default="inspect.png")
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--panel-height", type=int, default=200)
    a = ap.parse_args()
    from PIL import Image, ImageDraw

    meta = check_roots(a.root)
    cams, dcams = meta["cameras"], meta["depth_cameras"]
    eps = load_low_dim(a.root)
    ds = EpisodeDataset(eps, 1, cams, dcams)
    print(f"{len(eps)} episodes, {len(ds)} frames, fps {meta['fps']}, state {meta['state_dim']}, action {meta['action_dim']}, "
          f"image {meta['image_hw']}, cameras {cams}, depth {dcams}")
    print("episode lengths:", [len(e["state"]) for e in eps])

    # depth statistics over a subsample of frames
    print(f"\n{'camera':<12}{'valid %':>9}{'p1 (m)':>9}{'median':>9}{'p99 (m)':>9}{'max (m)':>9}")
    p99 = {}
    for c in dcams:
        vals, valid, tot = [], 0, 0
        for e in eps:
            for t in range(0, len(e["state"]), a.stride):
                d = depth_u16_to_m(np.asarray(Image.open(os.path.join(e["dir"], f"depth.{c}", f"{t:06d}.png"))))
                v = d > 0
                valid += int(v.sum()); tot += d.size
                if v.any():
                    vals.append(d[v][:: max(1, int(v.sum()) // 5000)])
        x = np.concatenate(vals) if vals else np.zeros(1)
        p99[c] = float(np.percentile(x, 99))
        print(f"{c:<12}{100 * valid / tot:>9.1f}{np.percentile(x, 1):>9.2f}{np.median(x):>9.2f}{np.percentile(x, 99):>9.2f}{x.max():>9.2f}")
    if p99:
        far = max(p99, key=p99.get)
        print(f"\ndepth_max_m (model.depth_max_m, one value for ALL cameras; 0 = off): DeFM's third input channel is a per-image\n"
              f"min-max, so far pixels compress the near range. Farthest p99 is {p99[far]:.2f} m ({far}); the nearest-ranging camera\n"
              f"is {min(p99, key=p99.get)} at {min(p99.values()):.2f} m. If the manipulation happens within ~1-1.5 m, a clamp of that size\n"
              f"keeps the near range resolved. Try both (0 and the clamp) on a short run; it is stored in the checkpoint.")
    for r in a.root:
        cj = os.path.join(r, "cameras.json")
        if os.path.exists(cj):
            import json
            j = json.load(open(cj))
            print(f"\ncameras.json ({r}): variant {j['variant']}, condition {j['condition']}")
            for n, c in j["intrinsics"].items():
                print(f"  {n:<12} rendered {c['resolution'][0]}x{c['resolution'][1]} hfov {c['hfov_deg']:.1f} deg -> "
                      f"after square crop to {c['square']['size']}: hfov {c['square']['hfov_deg']:.1f} deg")
            break

    # low-dim statistics
    st = fit_stats(eps)
    print("\nper-dimension (state | action): mean, std")
    for k in range(meta["action_dim"]):
        print(f"  dim {k:2d}: state {st['observation.state']['mean'][k]:+.3f} {st['observation.state']['std'][k]:.4f} | "
              f"action {st['action']['mean'][k]:+.3f} {st['action']['std'][k]:.4f}"
              + ("   <- constant" if st['action']['std'][k] <= 1e-5 else ""))

    # image grid
    e_i = min(a.episode, len(eps) - 1)
    e = eps[e_i]
    T = len(e["state"])
    H = a.panel_height
    first = [i for i, (ei, _) in enumerate(ds.index) if ei == e_i][0]
    rows = []
    for f in a.frames:
        t = int(round(f * (T - 1)))
        s = ds[first + t]
        row = []
        for c in cams:
            rgb = (s[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            panels = [Image.fromarray(rgb)]
            if c in dcams:
                panels.append(Image.fromarray(colour(s[f"observation.depth.{c}"][0].numpy(), a.near, a.far)))
            for p in panels:
                row.append(np.asarray(p.resize((H, H), Image.NEAREST)))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    img = Image.fromarray(grid)
    d = ImageDraw.Draw(img)
    x = 0
    for c in cams:
        d.text((x + 4, 2), f"{c} rgb", fill=(255, 255, 0)); x += H
        if c in dcams:
            d.text((x + 4, 2), f"{c} depth {a.near}-{a.far} m", fill=(255, 255, 0)); x += H
    for i, f in enumerate(a.frames):
        d.text((4, i * H + H - 12), f"episode {e_i} t={int(round(f * (T - 1)))}/{T - 1}", fill=(255, 255, 0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img.save(a.out)
    print("\nwrote", a.out)
