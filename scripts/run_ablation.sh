#!/usr/bin/env bash
# Follow-up rollouts for a finished run_pipeline.sh experiment (no training), same seeds as the pipeline:
#   rgbd_off    the RGB-D checkpoint with depth switched off            -> does depth matter at test time?
#   rgb_same    the RGB checkpoint at the RGB-D model's training step   -> is the gap just the checkpoint?
#   rgbd_repeat the RGB-D checkpoint again, depth on                    -> how many episodes flip between identical runs?
#
#   export OMNI_KIT_ACCEPT_EULA=YES                 # your decision, as for run_pipeline.sh
#   GPU=<n> SCENARIO=open_drawer bash scripts/run_ablation.sh
# Settings: EXP [${SCENARIO}_vlmlora]  CONDS [low_contrast]  RUNS [rgbd_off rgb_same rgbd_repeat]  EVAL_EPS [40]  PORT [6041]
#   ISAAC_ENV [defmvla-isaac]  TRAIN_ENV [defmvla].  Output: results/closed_${EXP}_<run>_<condition>.json and
#   results/ablation_$EXP.txt. Existing result files are kept, so a re-run resumes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ISAAC_ENV=${ISAAC_ENV:-defmvla-isaac}; TRAIN_ENV=${TRAIN_ENV:-defmvla}; GPU=${GPU:-}; SCENARIO=${SCENARIO:-}
EXP=${EXP:-${SCENARIO}_vlmlora}; CONDS=${CONDS:-low_contrast}; RUNS=${RUNS:-rgbd_off rgb_same rgbd_repeat}
EVAL_EPS=${EVAL_EPS:-40}; PORT=${PORT:-6041}
case "$SCENARIO" in pick_place|pick_place_camo|open_drawer) ;; *) echo "set SCENARIO=pick_place|pick_place_camo|open_drawer"; exit 2 ;; esac
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "set GPU=<number>"; exit 2; }
[ "${OMNI_KIT_ACCEPT_EULA:-}" = YES ] || { echo "Set OMNI_KIT_ACCEPT_EULA=YES yourself first (NVIDIA Omniverse EULA)."; exit 2; }
for r in $RUNS; do case $r in rgbd_off|rgb_same|rgbd_repeat) ;; *) echo "unknown run '$r'"; exit 2 ;; esac; done
CK=$(cat "runs/${EXP}_rgbd/selected_checkpoint.txt") || exit 1
[ -d "runs/${EXP}_rgb/$CK" ] || { echo "missing runs/${EXP}_rgb/$CK"; exit 1; }
mkdir -p logs results
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "cannot find conda"; exit 2; }
in_env() { local e=$1; shift; ( conda activate "$e" && "$@" ); }
PID=; trap '[ -n "$PID" ] && kill "$PID" 2>/dev/null' EXIT

run() {
  local name=$1 m=rgbd ck flags=() slog="logs/serve_${EXP}_$1.log" cond out try rc=0 todo=0
  case $name in rgb_same) m=rgb ;; rgbd_off) flags=(--depth-off) ;; esac
  for cond in $CONDS; do [ -s "results/closed_${EXP}_${name}_$cond.json" ] || todo=1; done
  [ $todo = 1 ] || { echo "skip $name (done)"; return 0; }
  ck="runs/${EXP}_$m/${CK}_deploy"
  if [ ! -s "$ck/defmvla_contract.json" ]; then   # the contract is written last, so it marks a finished merge
    in_env "$TRAIN_ENV" env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/merge_for_deploy.py \
        --checkpoint "runs/${EXP}_$m/$CK" || return 1
  fi
  echo "[$(date '+%T')] $name: $ck ${flags[*]}"
  ( conda activate "$TRAIN_ENV" && exec env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/serve_policy.py \
      --checkpoint "$ck" --port "$PORT" "${flags[@]}" ) > "$slog" 2>&1 &
  PID=$!
  for _ in $(seq 1 180); do grep -q listening "$slog" && break; kill -0 "$PID" 2>/dev/null || break; sleep 5; done
  grep -q listening "$slog" || { echo "policy server did not start, see $slog"; kill "$PID" 2>/dev/null; PID=; return 1; }
  for cond in $CONDS; do
    out="results/closed_${EXP}_${name}_$cond.json"
    for try in 1 2 3 4; do   # Isaac sometimes crashes at start-up yet exits 0; the result file is the success test
      [ -s "$out" ] && break
      DUO_GPU=$GPU in_env "$ISAAC_ENV" python sim/eval_closed_loop.py --scenario "$SCENARIO" --port "$PORT" \
          --episodes "$EVAL_EPS" --seed0 2000000 --condition "$cond" --out "$out" >> "logs/ablation_${EXP}_$name.log" 2>&1
      [ -s "$out" ] || sleep 10
    done
    [ -s "$out" ] || { echo "$name/$cond failed, see logs/ablation_${EXP}_$name.log"; rc=1; }
  done
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=
  return $rc
}

fail=0
for r in $RUNS; do run "$r" || fail=1; done
in_env "$TRAIN_ENV" python scripts/compare_rollouts.py "$EXP" $CONDS | tee "results/ablation_$EXP.txt"
exit $fail
