# DepthVLA

SmolVLA with a frozen [DeFM](https://github.com/leggedrobotics/defm) depth encoder. Depth tokens are added
element-wise to the RGB image tokens of the same camera; the VLM sees the same number of tokens as stock
SmolVLA. Built on lerobot 0.6.1 (`lerobot/smolvla_base`), following the DeFM fusion of
[entry-VLA](https://github.com/tonyhoVN/entry-VLA) (`Pi0DepthDeFM`, written there for pi0.5) and its SmolVLA
train / serve workflow.

```
RGB  (per camera) ─► SigLIP ─► connector (pixel-shuffle 4x4) ─► 64 tokens x 960 ──┐
                     frozen     frozen                                           (+)─► x sqrt(960) ─┐
depth (same camera) ─► DeFM ViT-S/14 @448 ─► 32x32x384 ─► unshuffle 4x4 ─► 8x8x6144                 │
                       frozen                             ─► LayerNorm ─► MLP ─► x gate(init 0) ──┘  │
                                                            trained                                  ▼
                              [ image tokens | language tokens | state token ] ─► VLM (16 layers, frozen)
                                                                                     │ keys / values
                                                noisy action chunk ─► action expert (LoRA trained) ─► actions
```

| Part | Parameters | State |
|---|---|---|
| SigLIP + connector + VLM text layers | ~350M | frozen |
| DeFM ViT-S/14 | 22M | frozen, always eval, no grad |
| Depth fusion (LayerNorm + MLP + gate) | ~7M | trained, 4x learning rate |
| Action expert | ~100M, held in float32 | LoRA adapters trained, 1.3M at r=16 (`train.mode: expert` trains all of it) |
| State / action / time projections | <1M | trained (`train.train_heads`) |

## The two design questions

**1. Is adding depth tokens onto the SigLIP tokens sound?** The residual is added after the connector, where
each of the 64 tokens stands for one cell of an 8x8 grid over the image. DeFM at 448 px gives 32x32 patches;
grouping them 4x4 (`pixel_unshuffle`, the same block layout SmolVLM's connector uses) gives the same 8x8 grid,
so depth token (r, c) covers the same image region as RGB token (r, c). The fusion output is multiplied by a
gate that starts at 0, so at step 0 the model *is* the RGB policy and depth enters only as far as training
opens the gate. Both of these are tested (`tests/test_policy.py`, `tests/test_fusion_and_data.py`).

**2. Does the projection fit the VLM's token count?** Yes, by construction: no tokens are appended. The
fusion output is `(B, 64, 960)`, exactly one camera's image-token block, and it is summed into that block.
Prefix length, padding masks, attention masks and position ids are identical with and without depth. The
grid sizes are computed from the loaded VLM config (image size / patch size / connector scale), a mismatch
raises at construction, and the hook checks the shape on every forward pass.

The hook wraps SmolVLA's `embed_prefix`: the stock function runs unmodified and the residual is then added to
that camera's token block in the finished prefix (scaled by sqrt(960), as stock scales image tokens). No
upstream code is copied.

## Layout

```
src/depthvla/
  fusion.py     DeFMEncoder (frozen), DepthFusion (trainable), depth letterboxing that matches the RGB one
  policy.py     DepthSmolVLAConfig / DepthSmolVLAPolicy, LoRA add / merge, freezing, build_policy / load_policy
  episodes.py   on-disk RGB-D episode format + EpisodeWriter        (numpy + pillow: runs in the Isaac env)
  data.py       torch Dataset over that format, normalisation statistics
  runtime.py    config files, processors, checkpoints + contract, live-observation batches
  client.py     PolicyClient for the simulator / robot side          (numpy + stdlib: runs in the Isaac env)
scripts/
  train.py  eval_offline.py  merge_for_deploy.py  serve_policy.py  make_synthetic.py
configs/        train_rgb.yaml (baseline)   train_rgbd.yaml
envs/setup_env.sh
tests/
```

## Setup (Ubuntu, same steps on the workstation and the server)

```bash
git clone <this repo> && cd DepthVLA
bash envs/setup_env.sh env        # .venv: torch (cu128), lerobot 0.6.1, peft, defm
bash envs/setup_env.sh weights    # SmolVLA, SmolVLM2 backbone, DeFM -> ~/.cache/huggingface (once, in tmux)
bash envs/setup_env.sh check      # test suite
source .venv/bin/activate
```

In the Isaac Sim python env only the recorder and the client are needed:
`./python.sh -m pip install --no-deps -e /path/to/DepthVLA` (numpy and pillow are already there).

## 1. Record data in Isaac Sim

One camera prim gives both RGB and depth, so they are pixel-aligned. Use `distance_to_image_plane` (planar
z-depth, what a RealSense reports), not `distance_to_camera`.

SmolVLA letterboxes non-square images, and the padding would become invalid depth, so frames are stored
square. `square=640` centre-crops every RGB and depth frame to its largest square and resizes it to 640x640
(`episodes.center_square`): a 1280x720 stream becomes the central 720x720, then 640x640. RGB is
area-averaged; depth uses nearest neighbour, so depths are never blended across an edge and invalid pixels
stay invalid. The model itself resizes RGB to 512 and depth to 448, so storing more than 640 gains nothing.

```python
from depthvla.episodes import EpisodeWriter

cams = ["left_wrist", "right_wrist", "front"]                       # 2 x D405, 1 x D435
w = EpisodeWriter("data/normal_train", cameras=cams, depth_cameras=cams, fps=30, square=640)
w.start()
for step in episode:
    w.add_frame(rgb={c: rgb[c] for c in cams},                     # uint8 (H, W, 3), any aspect ratio
                depth={c: depth_m[c] for c in cams},               # float metres (H, W); inf / nan = invalid
                state=state, action=action)                        # 1-D float vectors
w.save(task="pick up the mug") if success else w.discard()
```

The RGB and depth frame of a camera must have the same aspect ratio and field of view (true for one Isaac
camera prim; on a RealSense, align depth to colour first).

Depth is stored as lossless 16-bit PNG in millimetres (never video). Make one directory per split and
condition: `data/normal_train`, `data/normal_val`, `data/low_contrast_train`, `data/low_contrast_val`.
State and action can be anything consistent (joint positions, TCP pose, ...); they are normalised with
statistics of the training split, stored in the checkpoint.

## 2. Train (headless server)

```bash
python scripts/train.py --config configs/train_rgbd.yaml --dry-run      # data + config check, no GPU
tmux new -s train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/train_rgb.yaml       # RGB-only baseline
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/train_rgbd.yaml      # RGB + DeFM depth
```

Any key can be overridden on the command line (`train.steps=2000 model.lora_r=32
data.train_roots='[data/normal_train,data/low_contrast_train]' train.run_dir=runs/rgbd_mixed_seed0`).
`train_log.jsonl` logs loss, validation loss and `depth_gate` (mean |gate|: 0 = depth unused; it should rise).

The experiment grid is 2 configs x 2 data settings; keep the total number of episodes equal between
"normal" and "normal + low contrast", and use the same seeds:

| | train: normal | train: normal + low contrast |
|---|---|---|
| RGB (`train_rgb.yaml`) | `runs/rgb_normal_seed0` | `runs/rgb_mixed_seed0` |
| RGB-D (`train_rgbd.yaml`) | `runs/rgbd_normal_seed0` | `runs/rgbd_mixed_seed0` |

To warm-start the RGB-D run from the finished RGB run instead of from `smolvla_base`, first merge the RGB
checkpoint (step 4) and set `model.init_from` to the `_deploy` directory.

## 3. Check a checkpoint without the simulator

```bash
python scripts/eval_offline.py --checkpoint runs/rgbd_normal_seed0/checkpoint_005000 \
    --roots normal=data/normal_val low_contrast=data/low_contrast_val
```

Reports the action error on held-out episodes with depth and with the depth residual switched off, per
condition. It is an open-loop number, not a success rate.

## 4. Freeze for deployment and copy to the workstation

```bash
python scripts/merge_for_deploy.py --checkpoint runs/rgbd_normal_seed0/checkpoint_005000
rsync -avP server:DepthVLA/runs/rgbd_normal_seed0/checkpoint_005000_deploy runs/
```

The `_deploy` directory is self-contained: weights (SmolVLA + DeFM + fusion, LoRA folded in), config,
normalisation statistics, and `depthvla_contract.json` (cameras, image size, depth convention).

## 5. Serve and roll out (workstation)

```bash
python scripts/serve_policy.py --checkpoint runs/checkpoint_005000_deploy \
    --self-test data/normal_val/episode_000000        # one forward pass on a recorded frame, then exit
python scripts/serve_policy.py --checkpoint runs/checkpoint_005000_deploy --port 6040
```

In the Isaac Sim process:

```python
from depthvla.client import PolicyClient

pc = PolicyClient(port=6040, square=640)       # same square= as the EpisodeWriter that recorded the data
info = pc.info()
for seed in eval_seeds:
    env.reset(seed=seed); pc.reset(seed=seed)
    while not done:
        actions = pc.act(images={c: rgb[c] for c in cams}, depth={c: depth_m[c] for c in cams},
                         state=state, task=instruction)            # (horizon, action_dim)
        for a in actions[: info["n_action_steps"]]:
            env.step(a)
```

The policy is loaded in eval mode with every parameter frozen. The flow-matching noise is reseeded at every
reset, so a seed reproduces exactly. Run the same seeds for all four checkpoints under normal and
low-contrast rendering.

## What has been run

On a Windows laptop (CPU, lerobot 0.6.1, torch 2.11, transformers 5.5, numpy 2.2), with the real
`smolvla_base`, SmolVLM2 and DeFM ViT-S weights:

- `pytest`: 14 tests pass. Against real weights: prefix length and masks identical with and without depth;
  the residual changes only the 64 tokens of its own camera, by exactly residual x sqrt(960); with gate = 0
  the action chunk equals stock `SmolVLAPolicy` to 1e-4; opening the gate / changing the depth changes the
  actions; gradients reach only fusion, LoRA and heads (none in the VLM or DeFM); save -> load -> merge LoRA
  reproduces the actions to 1e-4.
- End to end on synthetic data: 3 training steps (RGB-D, RGB-only, and RGB-D warm-started from the merged
  RGB checkpoint), merge, server self-test, client/server round trip (same seed repeats, other seed differs),
  offline eval.

Not run: anything on Linux or a GPU (bf16 autocast path, memory use, speed), Isaac Sim, real data, and any
training long enough to say whether depth helps. Batch sizes in the configs are estimates.

## Notes

- **Precision**: `smolvla_base` stores the VLM and most of the action expert in bfloat16, and lerobot keeps
  them that way. Two consequences were handled: the depth residual is added in float32 (in bf16 a small
  residual is rounded away while the gate is near 0), and the action expert is held in float32
  (`expert_float32`, an exact cast) so LoRA updates and the LoRA merge are not rounded. The frozen VLM stays
  bf16; with `precision: bf16` the forward pass runs under autocast as usual.

- **Depth convention**: metres, planar z-depth, pixel-aligned with the RGB of the same camera, 0 = no
  reading. DeFM applies its own metric normalisation inside the model, so there is no depth preprocessing to
  keep in sync between training and deployment. Do not normalise or colour-map depth yourself.
- **`depth_max_m`**: off (0) in the config, so DeFM sees the raw depth of every camera. DeFM's third input
  channel is a per-image min-max, so a far background pixel compresses the near range; if that turns out to
  matter (e.g. for the wrist cameras), set a clamp in metres. It applies to all cameras and is saved in the
  checkpoint.
- **Memory**: the depth residual enters before the frozen VLM layers, so the backward pass runs through
  those layers for the image tokens. Weights there get no gradient, but activations are kept; expect higher
  memory than stock expert-only training. Lower `batch_size` and raise `grad_accum` if needed.
- **Depth dropout** (`depth_dropout: 0.25`): a quarter of the training samples see no depth. It keeps the
  RGB path alive; set it to 0 for a policy that always has depth. Report it as a variable.
- **GPU selection**: `CUDA_VISIBLE_DEVICES=N` with `device: cuda` (lerobot's config does not accept `cuda:N`).
- **Adding DeFM ViT-L**: `model.defm_variant=defm_vit_l14`; the fusion input width follows automatically.
