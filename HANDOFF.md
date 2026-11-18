# Handoff: moving defmVLA to the GPU workstation

Everything the simulator needs is in this repo (robot, meshes, scripts). Nothing is read from `~/vla_research` or `~/jazzy_ws`.
Tested on the first workstation (RTX 4090, 31 GB RAM, Isaac Sim 6.2, driver 580).

## Day-one runbook (conda envs are fine: activate, then run)

```bash
git pull
# training / serving env
conda activate <train-env>
bash envs/check_machine.sh train          # fix every FAIL line (lerobot must be exactly 0.6.1)
bash envs/setup_env.sh env                # only if something is missing
bash envs/setup_env.sh weights && bash envs/setup_env.sh check        # 23 tests pass
# Isaac env (a machine that can RENDER: RTX-class GPU)
conda activate <isaac-env>
bash envs/check_machine.sh isaac          # needs Isaac Sim 6.x; installs nothing
pip install --no-deps -e .                # defmvla recorder + client into the Isaac env
python sim/build_usd.py                   # robot USD, once per machine
python sim/tune_demo.py --seeds 4 --configs x:0.05                    # expect every episode to succeed
python sim/record_duo.py --out data/smoke --episodes 2 --preview data/smoke_preview     # then look at the images
python scripts/inspect_data.py --root data/smoke --out runs/smoke.png
```

If you accept the Omniverse EULA, `export OMNI_KIT_ACCEPT_EULA=YES` for unattended runs. Nothing here does it for you.

## The twelve pain points: what was done

| # | Pain point | Fix | Status |
|---|---|---|---|
| 1 | Robot lived outside the repo (absolute paths) | `sim/robot/`: both URDF variants, the 22 meshes (27 MB, Franka Apache-2.0 `LICENSE`/`NOTICE`), xacro sources. `sim/build_usd.py` regenerates the USD per machine; the USD now references only this repo | built + loaded + trained against |
| 2 | Separate Isaac env | `envs/setup_isaac.sh` (+ `check_machine.sh isaac`) installs or verifies Isaac Sim 6.2; works in an active conda env; never accepts the EULA | resolve verified with `DRY_RUN=1` (169 packages); full install not repeated |
| 3 | GPU may not render (A100/H100 have no RT cores); Vulkan driver | `envs/check_machine.sh` FAILs for Isaac on those GPUs, checks the Vulkan ICD, driver >= 570, RAM, disk | verified on the 4090 |
| 4 | Uncommitted work, data outside git | commit and push (you); data / runs / logs / USD / results are git-ignored; weights re-download | **yours to run** |
| 5 | No low-contrast condition | `--condition low_contrast` (dim light, near-identical grey table / cube / bowl, neutral floor). Same seed => same layout in both conditions | recorded + inspected |
| 6 | Assumed camera parameters | `sim/cameras.yaml` (poses, resolution, FOV, clip); effective intrinsics and the FOV after the square crop go into `cameras.json` in every dataset and are printed by `inspect_data.py` | **you must set the real values** |
| 7 | Stand-in demonstrator (~50%, one arm only) | Root cause: IK left the gripper yaw free, so fingers closed diagonally. Now the closing axis is constrained: 12/12 (6 left, 6 right) vs 4/12. Either arm works (by seed); the other is parked, so no action dimension is constant | measured |
| 8 | Rail / variant questions | `--with-rail` (17-dim, held at 0), `--variant actual|upright` | **you decide** (below) |
| 9 | Sim depth is perfect; `depth_max_m` | `inspect_data.py` prints the per-camera range and the `depth_max_m` trade-off; `cameras.yaml` `clip` bounds wrist range; `model.depth_noise_m` exists in the training config | decision is yours |
| 10 | Throughput / sizing | `sim/collect_parallel.sh` (disjoint seed blocks per worker, seed convention below). Measured 7 GB RAM per recorder process: 2 workers on 31 GB; RAM/8 in general | measured (3 workers OOM-killed) |
| 11 | Jobs died with the app | `scripts/run_detached.sh` (systemd user unit / tmux / setsid, env passed on, log in `logs/`); `scripts/run_experiment.sh` (RGB + RGB-D + merge + offline eval, one command); `num_workers: 4` in the configs | used for every run here |
| 12 | No closed-loop eval | `sim/eval_closed_loop.py` (seeded episodes, per-arm / condition success, JSON) | harness verified with a 20-step policy (0/2, as it should be); not yet with a trained one |

## Seed convention (keep splits disjoint)
Train 0.. , validation 1,000,000.. , closed-loop eval 2,000,000.. (`collect_parallel.sh` gives worker *w* a block of 100,000).
Use the same seeds under `normal` and `low_contrast`, and the same eval seeds for every checkpoint.

## Decisions only you can make
1. **Real camera rig** -> `sim/cameras.yaml`: front camera pose (the current eye / target is a placeholder), wrist offset, intrinsics,
   resolution. `square=640` crops the wrists to ~62 deg and the front camera to ~43 deg horizontal FOV; if the crop cuts what matters, change the rig or `--square`.
2. **Rail**: held at 0. If the physical rail moves during tasks, record with `--with-rail` and decide the action for it.
3. **Variant**: `actual` (arms tilted +-45 deg about x) is the default; confirm it is the real mounting.
4. **The real task**: the scripted one is a single-arm pick-and-place of a cube into a bowl. For bimanual tasks, replace `sim/demonstrator.py`
   (or record by teleoperation) and keep the 16-vector layout and 15 Hz.
5. **`depth_max_m`** (0 = off): try both on a short run after looking at `inspect_data.py`; **`depth_noise_m`** (e.g. 0.002) to mimic sensor noise.
6. **Not changed, by design**: lerobot is pinned to 0.6.1 (the policy hook wraps its internals). A different version in your conda env will fail `check_machine.sh`.

## Suggested first experiment (4 runs)
```bash
bash sim/collect_parallel.sh data/normal_train 2 50 0 --condition normal
bash sim/collect_parallel.sh data/normal_val 1 15 1000000 --condition normal
bash sim/collect_parallel.sh data/low_contrast_train 2 50 0 --condition low_contrast
bash sim/collect_parallel.sh data/low_contrast_val 1 15 1000000 --condition low_contrast
bash scripts/run_detached.sh normal bash scripts/run_experiment.sh normal "data/normal_train_w0,data/normal_train_w1" "data/normal_val_w0,data/low_contrast_val_w0" 5000
bash scripts/run_detached.sh mixed  bash scripts/run_experiment.sh mixed  "data/normal_train_w0,data/low_contrast_train_w0" "data/normal_val_w0,data/low_contrast_val_w0" 5000
```
(keep the total number of training episodes equal between `normal` and `mixed`). Then serve each `_deploy` checkpoint and run
`sim/eval_closed_loop.py` with `--condition normal` and `--condition low_contrast` and the same eval seeds.

## Leftovers on the first workstation
`data/duo_train`, `data/duo_val`, `runs/duo_rgb`, `runs/duo_rgbd` come from the earlier pipeline (different camera pose, task text and demonstrator).
They are not compatible with the new recorder settings; re-record rather than mixing them in.
