#!/usr/bin/env python3
"""Print the name of the checkpoint to evaluate: the saved one with the lowest validation loss (or the last one).

  python scripts/pick_checkpoint.py runs/pick_place_camo_vlmlora_rgbd            # -> checkpoint_006000
  python scripts/pick_checkpoint.py runs/pick_place_camo_vlmlora_rgbd --rule last
"""

import argparse
import json
import os
import re
import sys


def pick(run_dir: str, rule: str = "best") -> str:
    """Only steps with a saved checkpoint count. "best" needs a validation loss at every one of them."""
    steps = sorted(int(m.group(1)) for d in os.listdir(run_dir) if (m := re.fullmatch(r"checkpoint_(\d{6})", d)))
    if not steps:
        raise SystemExit(f"no checkpoints in {run_dir}")
    if rule == "last":
        return f"checkpoint_{steps[-1]:06d}"
    with open(os.path.join(run_dir, "train_log.jsonl")) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    val = {r["step"]: r["val_loss"] for r in recs if "val_loss" in r}
    missing = [s for s in steps if s not in val]
    if missing:
        raise SystemExit(f"no validation loss for checkpoints {missing} in {run_dir}: make train.save_every a multiple of train.val_every")
    best = min(steps, key=lambda s: val[s])
    print(f"{run_dir}: " + "  ".join(f"{s}:{val[s]}" for s in steps) + f"  -> {best}", file=sys.stderr)
    return f"checkpoint_{best:06d}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--rule", choices=["best", "last"], default="best")
    a = ap.parse_args()
    print(pick(a.run_dir, a.rule))
