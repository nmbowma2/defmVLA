#!/usr/bin/env python3
"""Serve a frozen checkpoint to a simulator / robot process (which runs in its own python env).

  python scripts/serve_policy.py --checkpoint runs/rgbd_normal/checkpoint_005000_deploy --port 6040
  ... --host 0.0.0.0        # accept connections from another machine (e.g. serve on the GPU box)
  ... --self-test DIR       # one forward pass on frame 0 of an episode directory, print the chunk, exit

Protocol (multiprocessing.connection, one request -> one reply; client: depthvla.client.PolicyClient):
  {"cmd": "info"}                      -> the checkpoint contract (cameras, depth cameras, image size, horizon, ...)
  {"cmd": "reset", "seed": int|None}   -> clears policy state and reseeds the flow-matching noise -> {"ok": true}
  {"cmd": "act", "images": {cam: uint8 (H, W, 3)}, "depth": {cam: float32 metres (H, W) | uint16 mm},
   "state": (S,), "task": str}         -> {"actions": (horizon, A) float32 un-normalised, "latency_s"}

The policy is loaded in eval mode with every parameter frozen (LoRA adapters, if still present, are merged at
load time). The noise generator is reseeded at every reset, so reruns of a seed repeat exactly.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from depthvla.client import DEFAULT_AUTHKEY, DEFAULT_PORT  # noqa: E402
from depthvla.runtime import load_processors, observation_batch, preprocess, read_contract  # noqa: E402


class Runner:
    def __init__(self, checkpoint, device, seed, precision):
        import torch

        from depthvla.policy import load_policy

        self.torch, self.device, self.seed = torch, device, int(seed)
        self.contract = read_contract(checkpoint)
        self.policy = load_policy(checkpoint, device)
        self.policy.merge_lora()
        self.pre, self.post = load_processors(checkpoint, self.policy.config, device)
        self.amp = precision == "bf16" and device.startswith("cuda")
        self.info = {**self.contract, "checkpoint": os.path.abspath(checkpoint), "device": device,
                     "chunk_size": self.policy.config.chunk_size, "seed": self.seed, "precision": precision,
                     "trainable_parameters": sum(p.numel() for p in self.policy.parameters() if p.requires_grad)}
        self.reset()

    def reset(self, seed=None):
        self.policy.reset()
        self.gen = self.torch.Generator(device=self.device).manual_seed(self.seed if seed is None else int(seed))

    def act(self, req):
        torch, cfg = self.torch, self.policy.config
        b = observation_batch(self.contract, req["images"], req.get("depth") or {}, req["state"], req["task"])
        b = preprocess(self.pre, b, self.device)
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=self.gen, device=self.device)
        with torch.inference_mode(), torch.autocast(device_type=torch.device(self.device).type,
                                                    dtype=torch.bfloat16, enabled=self.amp):
            a = self.policy.predict_action_chunk(b, noise=noise)
        return self.post(a).float().cpu().numpy()[0]


def self_test(r, episode_dir):
    from PIL import Image

    c = r.contract
    z = np.load(os.path.join(episode_dir, "low_dim.npz"))
    req = {"images": {cam: np.asarray(Image.open(os.path.join(episode_dir, f"rgb.{cam}", "000000.png")).convert("RGB"))
                      for cam in c["cameras"]},
           "depth": {cam: np.asarray(Image.open(os.path.join(episode_dir, f"depth.{cam}", "000000.png")))
                     for cam in c["depth_cameras"]},
           "state": z["state"][0], "task": open(os.path.join(episode_dir, "task.txt")).read().strip()}
    t0 = time.perf_counter()
    A = r.act(req)
    lat = time.perf_counter() - t0
    r.reset()
    B = r.act(req)
    gt = z["action"][: len(A)]
    print(json.dumps({"self_test": episode_dir, "latency_s": round(lat, 3), "actions_shape": list(A.shape),
                      "finite": bool(np.isfinite(A).all()), "repeatable_after_reset": bool(np.allclose(A, B)),
                      "first_step": np.round(A[0], 4).tolist(), "demo_first_step": np.round(gt[0], 4).tolist(),
                      "mean_abs_err_vs_demo": float(np.abs(A[: len(gt)] - gt).mean())}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="localhost", help="0.0.0.0 to serve other machines")
    ap.add_argument("--authkey", default=DEFAULT_AUTHKEY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", default=None, metavar="EPISODE_DIR")
    a = ap.parse_args()

    r = Runner(a.checkpoint, a.device, a.seed, a.precision)
    print(json.dumps(r.info, default=str), flush=True)
    if a.self_test:
        self_test(r, a.self_test)
        return
    from multiprocessing.connection import Listener

    with Listener((a.host, a.port), authkey=a.authkey.encode()) as lst:
        print(f"listening on {a.host}:{a.port}", flush=True)
        while True:
            try:
                conn = lst.accept()
            except (EOFError, ConnectionResetError, OSError) as e:  # e.g. a port check: keep serving
                print(f"handshake failed ({type(e).__name__}), still listening", flush=True)
                continue
            print("client connected", flush=True)
            try:
                while True:
                    req = conn.recv()
                    try:
                        cmd = req.get("cmd", "act")
                        if cmd == "reset":
                            r.reset(req.get("seed"))
                            conn.send({"ok": True})
                        elif cmd == "info":
                            conn.send(dict(r.info))
                        else:
                            t0 = time.perf_counter()
                            A = r.act(req)
                            conn.send({"actions": A.astype(np.float32), "latency_s": time.perf_counter() - t0})
                    except Exception as e:  # reported to the client; the server keeps running
                        conn.send({"error": repr(e)})
            except (EOFError, ConnectionResetError):
                print("client disconnected", flush=True)


if __name__ == "__main__":
    main()
