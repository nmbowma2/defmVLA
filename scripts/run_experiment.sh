#!/usr/bin/env bash
# One data setting, both models: train RGB-only and RGB-D with the same seed and recipe, then offline-eval both
# (RGB-D with depth on and off), and merge both for deployment.
#
#   bash scripts/run_experiment.sh TAG TRAIN_ROOTS VAL_ROOTS [STEPS] [extra train.py overrides...]
#   bash scripts/run_experiment.sh normal "data/normal_train_w0,data/normal_train_w1" data/normal_val 5000
#   bash scripts/run_experiment.sh mixed  "data/normal_train_w0,data/low_contrast_train_w0" "data/normal_val,data/low_contrast_val" 5000 train.seed=1
#
# ROOTS are comma-separated directories. Output: runs/TAG_rgb, runs/TAG_rgbd (checkpoints + `_deploy` merges),
# results/TAG_offline_{rgb,rgbd}.json, logs/TAG_*.log, and logs/TAG.done when everything finished.
# Run it detached for long jobs:  bash scripts/run_detached.sh TAG bash scripts/run_experiment.sh TAG ...
# Env: DEFMVLA_PYTHON (default: python), CUDA_VISIBLE_DEVICES, HF_HUB_OFFLINE=1 is set for you (weights must be cached).
set -euo pipefail
TAG=$1; TRAIN=$2; VAL=$3; STEPS=${4:-5000}; shift $(( $# >= 4 ? 4 : $# ))
PY=${DEFMVLA_PYTHON:-python}
export HF_HUB_OFFLINE=1
mkdir -p logs results
rm -f "logs/$TAG.done"
CK=$(printf 'checkpoint_%06d' "$STEPS")
for m in rgb rgbd; do
  "$PY" scripts/train.py --config "configs/train_$m.yaml" "data.train_roots=[$TRAIN]" "data.val_roots=[$VAL]" \
      "train.steps=$STEPS" "train.save_every=$STEPS" "train.run_dir=runs/${TAG}_$m" "$@" > "logs/${TAG}_train_$m.log" 2>&1
  "$PY" scripts/merge_for_deploy.py --checkpoint "runs/${TAG}_$m/$CK" > "logs/${TAG}_merge_$m.log" 2>&1
done
ROOTS=(); i=0; IFS=',' read -ra VR <<< "$VAL"; for r in "${VR[@]}"; do ROOTS+=("val$i=$r"); i=$((i+1)); done
for m in rgb rgbd; do
  "$PY" scripts/eval_offline.py --checkpoint "runs/${TAG}_$m/$CK" --roots "${ROOTS[@]}" --out "results/${TAG}_offline_$m.json" \
      > "logs/${TAG}_eval_$m.log" 2>&1
done
echo done > "logs/$TAG.done"
echo "TAG $TAG finished: runs/${TAG}_rgb runs/${TAG}_rgbd  results/${TAG}_offline_*.json"
