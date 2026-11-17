#!/usr/bin/env python3
"""Write a tiny fake RGB-D dataset in the episode format, to smoke-test the pipeline end to end before real
simulator data exists. The data is meaningless (a moving square); do not read anything into the losses.

  python scripts/make_synthetic.py --out data/synth_train --episodes 8
  python scripts/make_synthetic.py --out data/synth_val --episodes 2 --seed 1
  python scripts/train.py --config configs/train_rgbd.yaml data.train_roots='[data/synth_train]' \\
      data.val_roots='[data/synth_val]' train.steps=20 train.batch_size=2 train.grad_accum=1 \\
      train.save_every=20 train.val_every=20 train.log_every=5 train.num_workers=0 train.run_dir=runs/smoke
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from depthvla.episodes import write_synthetic  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--size", type=int, default=256, help="square image side")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(write_synthetic(a.out, a.episodes, a.length, (a.size, a.size), seed=a.seed))
