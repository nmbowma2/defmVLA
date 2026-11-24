#!/usr/bin/env python3
"""Turn a training checkpoint into a plain frozen deployment checkpoint: LoRA adapters (action expert and
VLM) are folded into the weights and removed. Outputs are numerically the same policy.

  python scripts/merge_for_deploy.py --checkpoint runs/rgbd_normal/checkpoint_005000
      -> runs/rgbd_normal/checkpoint_005000_deploy   (copy this directory to the workstation)
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from defmvla.runtime import CONTRACT_FILE, read_contract  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="default: <checkpoint>_deploy")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = a.out or a.checkpoint.rstrip("/\\") + "_deploy"

    from defmvla.policy import load_policy

    policy = load_policy(a.checkpoint, a.device)
    had, had_vlm = policy.config.lora_r, policy.config.vlm_lora_r
    policy.merge_lora()
    os.makedirs(out, exist_ok=True)
    policy.save_pretrained(out)
    for f in os.listdir(a.checkpoint):  # processors (normalisation statistics) and anything else that is not weights
        if f not in ("config.json", CONTRACT_FILE) and not f.startswith("model") and os.path.isfile(os.path.join(a.checkpoint, f)):
            shutil.copy2(os.path.join(a.checkpoint, f), os.path.join(out, f))
    with open(os.path.join(out, CONTRACT_FILE), "w") as fh:
        json.dump({**read_contract(a.checkpoint), "lora_r": 0, "vlm_lora_r": 0, "merged_from": os.path.abspath(a.checkpoint),
                   "merged_lora_r": had, "merged_vlm_lora_r": had_vlm}, fh, indent=1)
    print(json.dumps({"out": out, "merged_lora_r": had, "merged_vlm_lora_r": had_vlm,
                      "parameters": sum(p.numel() for p in policy.parameters()),
                      "trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad)}))


if __name__ == "__main__":
    main()
