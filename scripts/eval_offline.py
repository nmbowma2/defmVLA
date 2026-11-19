#!/usr/bin/env python3
"""Open-loop check of a checkpoint on recorded episodes: predicted action chunks vs the demonstrated ones.
Not a success rate (that needs closed-loop rollouts through scripts/serve_policy.py), but it is fast, runs on
the headless server, and its depth ablation shows how much the policy actually relies on depth.

  python scripts/eval_offline.py --checkpoint runs/rgbd_normal/checkpoint_005000 \\
      --roots normal=data/normal_val low_contrast=data/low_contrast_val --stride 5

Per split it reports the mean absolute action error (un-normalised units, valid steps only) for
  depth      the policy as trained
  no_depth   the same weights with the depth residual switched off (RGB only through the same network)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from defmvla.data import load_low_dim, make_dataset  # noqa: E402
from defmvla.runtime import load_processors, preprocess, read_contract  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--roots", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--stride", type=int, default=5, help="evaluate every n-th frame")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the report here (JSON)")
    a = ap.parse_args()

    import torch

    from defmvla.policy import load_policy

    c = read_contract(a.checkpoint)
    policy = load_policy(a.checkpoint, a.device)
    pre, post = load_processors(a.checkpoint, policy.config, a.device)
    cfg = policy.config
    dev = torch.device(a.device)
    modes = ["depth", "no_depth"] if cfg.use_depth else ["rgb"]
    report = {"checkpoint": os.path.abspath(a.checkpoint), "stride": a.stride, "splits": {}}
    for item in a.roots:
        name, root = item.split("=", 1)
        eps = load_low_dim([root], c.get("arm_prompt", False))
        ds = make_dataset(eps, cfg.chunk_size, c["cameras"], c["depth_cameras"])
        sub = torch.utils.data.Subset(ds, list(range(0, len(ds), a.stride)))
        loader = torch.utils.data.DataLoader(sub, batch_size=a.batch_size, num_workers=a.num_workers)
        err = {m: [] for m in modes}
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        for b in loader:
            gt, pad = b.pop("action").numpy(), b.pop("action_is_pad").numpy()
            bs = gt.shape[0]
            noise = torch.randn((bs, cfg.chunk_size, cfg.max_action_dim), generator=gen, device=dev)
            for m in modes:
                policy.config.use_depth = m == "depth"
                batch = preprocess(pre, dict(b), a.device)
                with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                            enabled=dev.type == "cuda"):
                    pred = post(policy.predict_action_chunk(batch, noise=noise)).float().cpu().numpy()
                err[m].append(np.abs(pred - gt)[~pad])
            policy.config.use_depth = "depth" in modes
        report["splits"][name] = {"root": root, "chunks": len(sub),
                                  **{f"mae_{m}": float(np.concatenate(v).mean()) for m, v in err.items()}}
    print(json.dumps(report, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
