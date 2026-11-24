#!/usr/bin/env python3
"""Fine-tune SmolVLA (RGB-only baseline or RGB + DeFM depth) on an RGB-D episode dataset.

  python scripts/train.py --config configs/train_rgbd.yaml --dry-run          # config + data check, no GPU
  python scripts/train.py --config configs/train_rgbd.yaml                    # train
  python scripts/train.py --config configs/train_rgbd.yaml train.steps=2000 model.lora_r=32   # overrides
  CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config ...                 # pick a GPU on the server

Frozen: SigLIP, VLM layers, DeFM.  Trained: depth fusion (MLP + gate), the action expert (LoRA adapters by
default, train.mode=expert for all of it) and, with train.train_heads, the state / action / time projections.
Off by default: model.vlm_lora_r > 0 (LoRA on the attention of the last model.vlm_lora_layers VLM layers) and
train.train_connector=true (the VLM's image projector, all of it); both use lr * train.vlm_lr_mult. Checkpoints hold the policy, its pre/post-processors (normalisation statistics of the TRAIN split)
and defmvla_contract.json, so deployment needs nothing else.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from defmvla.data import DEPTH_KEY, RGB_KEY, check_roots, fit_stats, load_low_dim, make_dataset  # noqa: E402
from defmvla.runtime import load_config, make_features, make_processors, preprocess, save_checkpoint  # noqa: E402

DEFAULTS = {
    "data": {"train_roots": [], "val_roots": [], "cameras": None, "depth_cameras": None,
             "arm_prompt": False},   # True: task text "<left|right> arm: ..." (which arm moves is read from the data)
    "model": {"init_from": "lerobot/smolvla_base", "use_depth": True, "horizon": 50, "n_action_steps": 50,
              "defm_variant": "defm_vit_s14", "defm_weights_path": None, "defm_input_size": 448,
              "depth_pool": "unshuffle", "depth_hidden": 1024, "depth_max_m": 0.0, "depth_dropout": 0.25,
              "depth_noise_m": 0.0, "wrist_depth_range": None, "depth_mask_invalid": False, "native_resize": False, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
              "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
              "vlm_lora_r": 0, "vlm_lora_alpha": 32, "vlm_lora_dropout": 0.0, "vlm_lora_layers": 8,
              "vlm_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    "train": {"mode": "lora", "train_heads": True, "train_connector": False, "vlm_lr_mult": 0.5, "steps": 5000, "batch_size": 16, "grad_accum": 2, "lr": 1e-4,
              "depth_lr_mult": 4.0, "decay_lr": 2.5e-6, "warmup_steps": 200, "weight_decay": 1e-10,
              "grad_clip": 10.0, "precision": "bf16", "seed": 0, "num_workers": 8, "log_every": 20,
              "save_every": 1000, "val_every": 500, "val_batches": 16, "device": "cuda", "run_dir": None},
}


def merged(cfg):
    out = {s: dict(d) for s, d in DEFAULTS.items()}
    for s, d in (cfg or {}).items():
        if s not in out:
            raise SystemExit(f"unknown config section {s!r}")
        unknown = set(d or {}) - set(out[s])
        if unknown:
            raise SystemExit(f"unknown keys in {s}: {sorted(unknown)}")
        out[s].update(d or {})
    return out


def lr_scale(step, t):
    """Linear warmup to 1, cosine decay to decay_lr / lr at the last step."""
    if step < t["warmup_steps"]:
        return (step + 1) / t["warmup_steps"]
    p = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
    floor = t["decay_lr"] / t["lr"]
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * min(1.0, p)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate config and data only")
    ap.add_argument("overrides", nargs="*", help="section.key=value")
    a = ap.parse_args(argv)
    cfg = merged(load_config(a.config, a.overrides))
    D, M, T = cfg["data"], cfg["model"], cfg["train"]

    problems = []
    if not D["train_roots"]:
        problems.append("data.train_roots is empty")
    if not T["run_dir"]:
        problems.append("train.run_dir is required")
    if T["mode"] == "lora" and M["lora_r"] <= 0:
        problems.append("train.mode=lora needs model.lora_r > 0")
    if problems:
        raise SystemExit(f"config problems: {problems}")

    meta = check_roots(D["train_roots"] + D["val_roots"])
    cameras = D["cameras"] or meta["cameras"]
    depth_cameras = (D["depth_cameras"] if D["depth_cameras"] is not None else meta["depth_cameras"]) if M["use_depth"] else []
    if not set(cameras) <= set(meta["cameras"]) or not set(depth_cameras) <= set(meta["depth_cameras"]) & set(cameras):
        raise SystemExit(f"cameras {cameras} / depth cameras {depth_cameras} not available in the data "
                         f"({meta['cameras']} / {meta['depth_cameras']}); depth cameras must also be RGB cameras")
    if M["use_depth"] and not depth_cameras:
        raise SystemExit("model.use_depth is true but there is no depth camera")
    train_eps = load_low_dim(D["train_roots"], D["arm_prompt"])
    val_eps = load_low_dim(D["val_roots"], D["arm_prompt"]) if D["val_roots"] else []
    stats = fit_stats(train_eps)
    report = {"config": cfg, "cameras": cameras, "depth_cameras": depth_cameras, "image_hw": meta["image_hw"],
              "state_dim": meta["state_dim"], "action_dim": meta["action_dim"], "fps": meta["fps"],
              "train": {"episodes": len(train_eps), "frames": sum(len(e["state"]) for e in train_eps)},
              "val": {"episodes": len(val_eps), "frames": sum(len(e["state"]) for e in val_eps)},
              "effective_batch": T["batch_size"] * T["grad_accum"],
              "action_std": np.round(stats["action"]["std"], 4).tolist()}
    print(json.dumps(report, indent=1, default=str))
    if meta["image_hw"][0] != meta["image_hw"][1] and M["use_depth"] and not M["depth_mask_invalid"]:
        print("WARNING: non-square images are letterboxed to 512 x 512; the padding counts as depth 0. "
              "Set model.depth_mask_invalid=true (and model.native_resize=true).", flush=True)
    if a.dry_run:
        return 0

    import torch

    from defmvla.policy import build_policy

    torch.manual_seed(T["seed"])
    np.random.seed(T["seed"])
    dev = torch.device(T["device"])
    os.makedirs(T["run_dir"], exist_ok=True)
    with open(os.path.join(T["run_dir"], "train_config.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    inputs, outputs = make_features(cameras, meta["image_hw"], meta["state_dim"], meta["action_dim"])
    depth_range = {DEPTH_KEY.format(c): [float(v) for v in M["wrist_depth_range"]] for c in depth_cameras
                   if "wrist" in c} if M["wrist_depth_range"] else {}
    policy = build_policy(
        M["init_from"], inputs, outputs, device=T["device"],
        lora_r=M["lora_r"] if T["mode"] == "lora" else 0, lora_alpha=M["lora_alpha"],
        lora_dropout=M["lora_dropout"], lora_targets=M["lora_targets"],
        vlm_lora_r=M["vlm_lora_r"], train_connector=T["train_connector"], vlm_lora_alpha=M["vlm_lora_alpha"],
        vlm_lora_dropout=M["vlm_lora_dropout"], vlm_lora_layers=M["vlm_lora_layers"], vlm_lora_targets=M["vlm_lora_targets"],
        use_depth=M["use_depth"], depth_map={RGB_KEY.format(c): DEPTH_KEY.format(c) for c in depth_cameras},
        chunk_size=M["horizon"], n_action_steps=M["n_action_steps"], empty_cameras=0,
        freeze_vision_encoder=True, train_expert_only=True, use_amp=T["precision"] != "fp32",
        defm_variant=M["defm_variant"], defm_weights_path=M["defm_weights_path"],
        defm_input_size=M["defm_input_size"], depth_pool=M["depth_pool"], depth_hidden=M["depth_hidden"],
        depth_max_m=M["depth_max_m"], depth_dropout=M["depth_dropout"], depth_noise_m=M["depth_noise_m"],
        depth_range_m=depth_range, depth_mask_invalid=M["depth_mask_invalid"], native_resize=M["native_resize"])
    counts = policy.configure_trainable(T["mode"], T["train_heads"], T["train_connector"])
    print(json.dumps({"parameters": counts}), flush=True)
    pre, post = make_processors(policy.config, stats)

    contract = {"cameras": cameras, "depth_cameras": depth_cameras, "image_hw": meta["image_hw"],
                "state_dim": meta["state_dim"], "action_dim": meta["action_dim"], "fps": meta["fps"],
                "horizon": M["horizon"], "n_action_steps": M["n_action_steps"], "use_depth": M["use_depth"],
                "depth": "planar z-depth in metres, pixel-aligned with the RGB of the same camera, 0 = invalid",
                "depth_max_m": M["depth_max_m"], "depth_range_m": depth_range, "native_resize": M["native_resize"], "defm_variant": M["defm_variant"], "init_from": M["init_from"],
                "arm_prompt": D["arm_prompt"], "train_roots": D["train_roots"], "train_mode": T["mode"], "lora_r": policy.config.lora_r,
                "vlm_lora_r": policy.config.vlm_lora_r, "vlm_lora_layers": policy.config.vlm_lora_layers, "train_connector": T["train_connector"]}

    sizes = (policy.config.resize_imgs_with_padding[0], M["defm_input_size"]) if M["native_resize"] else (0, 0)
    ds = make_dataset(train_eps, M["horizon"], cameras, depth_cameras, *sizes)
    g = torch.Generator().manual_seed(T["seed"])
    loader = torch.utils.data.DataLoader(ds, batch_size=T["batch_size"], shuffle=True, drop_last=True, generator=g,
                                         num_workers=T["num_workers"], persistent_workers=T["num_workers"] > 0,
                                         pin_memory=dev.type == "cuda")
    val_loader, val_gen = None, torch.Generator()
    if val_eps:
        val_loader = torch.utils.data.DataLoader(make_dataset(val_eps, M["horizon"], cameras, depth_cameras, *sizes),
                                                 batch_size=T["batch_size"], shuffle=True,
                                                 generator=val_gen,
                                                 num_workers=min(2, T["num_workers"]))

    groups = policy.param_groups(T["lr"], T["depth_lr_mult"], T["vlm_lr_mult"])
    params = [p for gp in groups for p in gp["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=T["weight_decay"])
    amp = T["precision"] == "bf16" and dev.type == "cuda"

    def loss_on(batch, **kw):
        batch = preprocess(pre, batch, T["device"])
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            return policy.forward(batch, **kw)[0]

    @torch.no_grad()
    def validate():
        """Mean flow-matching loss on a fixed slice of the val split with fixed noise / time (comparable
        across steps and runs). Train-mode extras (depth dropout / noise, LoRA dropout) are off."""
        policy.eval()
        val_gen.manual_seed(1234)  # the same samples at every validation, so checkpoints can be compared
        gen = torch.Generator(device=dev).manual_seed(0)
        tot, n = 0.0, 0
        for i, b in enumerate(val_loader):
            if i >= T["val_batches"]:
                break
            bs = b["action"].shape[0]
            noise = torch.randn((bs, M["horizon"], policy.config.max_action_dim), generator=gen, device=dev)
            t = torch.rand((bs,), generator=gen, device=dev) * 0.999 + 0.001
            tot += float(loss_on(b, noise=noise, time=t)) * bs
            n += bs
        policy.train()
        return tot / max(n, 1)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    logf = open(os.path.join(T["run_dir"], "train_log.jsonl"), "w")
    policy.train()
    it, t0, step = iter(loader), time.time(), 0
    print(f"{len(ds)} samples, effective batch {T['batch_size'] * T['grad_accum']}", flush=True)
    while step < T["steps"]:
        s = lr_scale(step, T)
        for gp in opt.param_groups:
            gp["lr"] = T["lr"] * gp["lr_mult"] * s
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(T["grad_accum"]):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            loss = loss_on(b)
            (loss / T["grad_accum"]).backward()
            tot += float(loss.detach()) / T["grad_accum"]
        if not math.isfinite(tot):
            raise SystemExit(f"loss is {tot} at step {step}")
        gn = float(torch.nn.utils.clip_grad_norm_(params, T["grad_clip"]))
        opt.step()
        step += 1
        if step % T["log_every"] == 0 or step == 1:
            rec = {"step": step, "loss": round(tot, 5), "grad_norm": round(gn, 4), "lr": opt.param_groups[0]["lr"],
                   "depth_gate": round(policy.gate_abs_mean(), 6),
                   "depth_ratio": round(policy.depth_ratio(), 5), "elapsed_s": round(time.time() - t0, 1)}
            if dev.type == "cuda":
                rec["peak_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            if val_loader is not None and (step % T["val_every"] == 0 or step == T["steps"]):
                rec["val_loss"] = round(validate(), 5)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
        if step % T["save_every"] == 0 or step == T["steps"]:
            out = os.path.join(T["run_dir"], f"checkpoint_{step:06d}")
            save_checkpoint(out, policy, pre, post, {**contract, "step": step})
            print(f"saved {out}", flush=True)
    logf.close()
    print(json.dumps({"done": step, "run_dir": T["run_dir"], "minutes": round((time.time() - t0) / 60, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
