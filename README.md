# defmVLA

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
src/defmvla/
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

## Setup (Ubuntu, same steps on every machine)

Two environments: **training / serving** (torch, lerobot, defm) and **Isaac Sim** (the simulator, which pins its own
numpy / torch). Existing conda envs are fine: activate one and run the same commands; `envs/setup_env.sh` and
`envs/setup_isaac.sh` install into the *active* env (a new `.venv` / `.venv-isaac` only if none is active).

```bash
git clone <this repo> && cd defmVLA
# training / serving env
bash envs/check_machine.sh train      # what is missing or has the wrong version (GPU, driver, RAM, disk, packages, weights)
bash envs/setup_env.sh env            # torch (cu128), lerobot 0.6.1 (pinned), peft, defm, scipy
bash envs/setup_env.sh weights        # SmolVLA, SmolVLM2 backbone, DeFM -> ~/.cache/huggingface (once)
bash envs/setup_env.sh check          # test suite (23 tests)
# Isaac Sim env (only where you record / evaluate in simulation; needs an RTX-class GPU, NOT A100/H100)
bash envs/check_machine.sh isaac      # needs Isaac Sim 6.x; reports the Vulkan driver, robot USD, ...
bash envs/setup_isaac.sh env          # skip if your env already has Isaac Sim 6.2 + `pip install --no-deps -e .`
bash envs/setup_isaac.sh usd          # builds the robot USD from sim/robot (once per machine)
```

Isaac Sim asks you to accept the NVIDIA Omniverse EULA on first launch; the scripts never accept it for you.
Long jobs: `bash scripts/run_detached.sh NAME COMMAND...` (systemd user unit / tmux, survives the terminal closing, log in
`logs/NAME.log`). Memory budget on a 31 GB machine: 4 loader workers for training, 2 recorder processes at once.

## Simulator (`sim/`): the real robot in Isaac Sim

`sim/robot/` holds the prismatic FR3 duo (URDF, the 22 meshes it needs, the xacro sources; Franka assets are Apache-2.0,
see `LICENSE` / `NOTICE`), so nothing outside this repo is needed. 14 arm joints + 2 gripper commands = 16 DoF; the rail
slider (+-0.5 m) is held at 0 (`--with-rail` makes it a 17th state / action entry). Variant `actual` = arms mounted tilted
+-45 deg about x (the real rig), `upright` = not tilted.

| File | Purpose |
|---|---|
| `sim/duo_env.py` | scene (table, cube, bowl), 3 RGB-D cameras, `normal` / `low_contrast` conditions, per-seed reset (working arm + cube position), state / action layout |
| `sim/cameras.yaml` | **the camera rig: edit to match the physical mounts / intrinsics**; copied into every dataset as `cameras.json` |
| `sim/demonstrator.py`, `sim/duo_kin.py` | scripted IK pick-and-place on either arm (finger closing axis held aligned with the cube: 12/12 success vs ~50% before) |
| `sim/record_duo.py`, `sim/collect_parallel.sh` | record episodes (one or several processes) |
| `sim/eval_closed_loop.py` | roll a served policy out in Isaac, success rate per seed / arm / condition |
| `sim/build_usd.py`, `sim/tune_demo.py` | build the USD; measure demonstrator success |

State and action (16): `[left q1..q7, left gripper, right q1..q7, right gripper]`, joint positions, gripper 0 open .. 1 closed;
action = absolute joint-position target at 15 Hz. Task (both conditions): "pick up the cube and place it in the bowl", with the
arm on the cube's side (left / right by seed) doing it. Seeds fix the layout: use **train 0.., validation 1,000,000..,
closed-loop eval 2,000,000..** and the same seeds under `normal` and `low_contrast`.

```bash
python sim/record_duo.py --out data/normal_train --episodes 50 --seed 0                       # Isaac env
bash sim/collect_parallel.sh data/low_contrast_train 2 50 0 --condition low_contrast            # 2 processes x 50 episodes
python scripts/inspect_data.py --root data/normal_train --out runs/inspect.png                  # look before you train
bash scripts/run_experiment.sh normal "data/normal_train_w0,data/normal_train_w1" data/normal_val 5000   # RGB + RGB-D + offline eval
python scripts/serve_policy.py --checkpoint runs/normal_rgbd/checkpoint_005000_deploy --port 6040          # train env
python sim/eval_closed_loop.py --port 6040 --episodes 20 --seed0 2000000 --out results/rgbd_normal.json    # Isaac env
```

## 1. Record data in Isaac Sim

For the duo robot use `sim/record_duo.py` (above). The generic recorder API, for any other simulator or robot:

One camera prim gives both RGB and depth, so they are pixel-aligned. Use `distance_to_image_plane` (planar
z-depth, what a RealSense reports), not `distance_to_camera`.

SmolVLA letterboxes non-square images, and the padding would become invalid depth, so frames are stored
square. `square=640` centre-crops every RGB and depth frame to its largest square and resizes it to 640x640
(`episodes.center_square`): a 1280x720 stream becomes the central 720x720, then 640x640. RGB is
area-averaged; depth uses nearest neighbour, so depths are never blended across an edge and invalid pixels
stay invalid. The model itself resizes RGB to 512 and depth to 448, so storing more than 640 gains nothing.

```python
from defmvla.episodes import EpisodeWriter

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
`train_log.jsonl` logs loss, validation loss and `depth_gate` (mean |gate|) and `depth_ratio`
(size of the depth residual relative to the RGB tokens). The gate alone does not say whether depth is used: it
multiplies an MLP output of arbitrary size. The depth on / off error of step 3 does.

Backbone adaptation (off in the yaml files, on in `scripts/run_pipeline.sh`): `model.vlm_lora_r=16` puts LoRA on
the attention of the last `model.vlm_lora_layers` VLM layers and `train.train_connector=true` trains the image
projector, both at `lr * train.vlm_lr_mult`.

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
rsync -avP server:defmVLA/runs/rgbd_normal_seed0/checkpoint_005000_deploy runs/
```

The `_deploy` directory is self-contained: weights (SmolVLA + DeFM + fusion, LoRA folded in), config,
normalisation statistics, and `defmvla_contract.json` (cameras, image size, depth convention).

## 5. Serve and roll out (workstation)

```bash
python scripts/serve_policy.py --checkpoint runs/checkpoint_005000_deploy \
    --self-test data/normal_val/episode_000000        # one forward pass on a recorded frame, then exit
python scripts/serve_policy.py --checkpoint runs/checkpoint_005000_deploy --port 6040
```

In the Isaac Sim process:

```python
from defmvla.client import PolicyClient

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

On the Ubuntu 24.04 workstation (RTX 4090 24 GB, torch 2.11+cu128): all 17 tests pass on the GPU; 20 RGB-D
training steps, merge and the serve self-test run end to end in bf16. Measured RGB-D peak memory with 3 RGB + 3
depth cameras at 640x640: 6.0 GB at batch 16, 10.5 GB at batch 32, about 0.5 s per step at batch 16 on a 2-camera
dataset (the desktop session holds ~3 GB of the card). The configs use batch 32, grad_accum 1.

On the workstation with the simulator: 12 + 3 recorded episodes, 1000 steps of RGB-D and RGB-only training (validation loss
0.0679 vs 0.0702, offline action error equal to within 3%: the task is too easy for depth to matter), and the scripted
demonstrator at 12/12. Not run: real data, a closed-loop success rate of a policy that has learned the task, and any
training long enough to say whether depth helps.

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
