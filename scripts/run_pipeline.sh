#!/usr/bin/env bash
# Unattended end-to-end run:  record (normal RGB-D) -> train RGB-only and RGB-D on NORMAL data only -> offline eval ->
# closed-loop eval of both models under normal AND low_contrast (zero-shot) -> summary.
# Two conda envs, activated per stage: the Isaac env records and evaluates in simulation, the training env trains and serves.
# Resumable: every finished stage writes logs/pipeline/<stage>.done and a re-run skips those. Nothing is deleted or redone
# on its own: to redo a stage remove its .done file (and its data / run directory).
#
#   export OMNI_KIT_ACCEPT_EULA=YES                 # your decision: the Omniverse EULA (the script refuses to start without it)
#   tmux new -s pipe                                # then inside tmux (detach: Ctrl-b d, reattach: tmux attach -t pipe)
#   bash scripts/run_pipeline.sh 2>&1 | tee -a logs/pipeline.log
#   (without tmux:  bash scripts/run_detached.sh pipeline bash scripts/run_pipeline.sh   -- uses the defaults below)
#
# What is trained: ONLY normal-condition data. The low_contrast recordings are a small VALIDATION set (offline diagnostic);
# they never enter training. Low contrast is tested zero-shot in closed loop.
# Settings (environment variables; defaults in brackets):
#   ISAAC_ENV [defmvla-isaac]  TRAIN_ENV [defmvla]  GPU [1: the physical GPU for Isaac AND training]
#   WORKERS [4] x EPS [25] = training episodes (100); VAL_WORKERS [2] x VAL_EPS [10] normal validation episodes (20);
#   LC_VAL_EPS [5] per validation worker of low-contrast validation episodes (10, offline diagnostic only);
#   STEPS [3000] per model, checkpoints every SAVE_EVERY [1000] (an earlier one is the fallback if validation loss turns up);
#   EVAL_EPS [40] closed-loop episodes per (model, condition) [4 runs]; PORT [6040];  DRY_RUN=1 prints the plan only.
# Seeds: train 0.. (worker w: +100000*w), validation 1,000,000.. (low-contrast validation uses the SAME layouts),
# closed-loop eval 2,000,000.. (identical for every model and condition, so the comparisons are paired).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ISAAC_ENV=${ISAAC_ENV:-defmvla-isaac}; TRAIN_ENV=${TRAIN_ENV:-defmvla}; GPU=${GPU:-1}
WORKERS=${WORKERS:-4}; EPS=${EPS:-25}; VAL_WORKERS=${VAL_WORKERS:-2}; VAL_EPS=${VAL_EPS:-10}; LC_VAL_EPS=${LC_VAL_EPS:-5}
STEPS=${STEPS:-3000}; SAVE_EVERY=${SAVE_EVERY:-1000}; EVAL_EPS=${EVAL_EPS:-40}; PORT=${PORT:-6040}
EXP=normal
CK=$(printf 'checkpoint_%06d' "$STEPS")
P=logs/pipeline; mkdir -p "$P" logs results runs data

say() { echo "[$(date '+%F %T')] $*" | tee -a "$P/progress.log"; }

if [ "${DRY_RUN:-0}" != 1 ]; then
  [ "${OMNI_KIT_ACCEPT_EULA:-}" = YES ] || { echo "Set OMNI_KIT_ACCEPT_EULA=YES yourself first (it accepts the NVIDIA Omniverse EULA; this script will not)."; exit 2; }
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "cannot find conda"; exit 2; }
fi

in_env() { local e=$1; shift; ( conda activate "$e" && "$@" ); }

# stage NAME cmd...  : run unless done; stop the whole pipeline on failure (SOFT=1: carry on)
stage() {
  local name=$1; shift
  if [ -e "$P/$name.done" ]; then say "skip   $name (done)"; return 0; fi
  say "start  $name"
  if [ "${DRY_RUN:-0}" = 1 ]; then say "   would run: $*"; return 0; fi
  if "$@" >> "$P/$name.log" 2>&1; then touch "$P/$name.done"; say "done   $name"
  elif [ "${SOFT:-0}" = 1 ]; then touch "$P/$name.done"; say "failed $name (not fatal, see $P/$name.log)"
  else say "FAILED $name  (see $P/$name.log); fix it and re-run this script, it resumes here"; exit 1; fi
}

roots() { # roots PREFIX N  -> data/PREFIX_w0,data/PREFIX_w1,...
  local out="" w; for ((w = 0; w < $2; w++)); do out+="data/$1_w$w,"; done; echo "${out%,}"; }

record() { # record STAGE OUT WORKERS EPISODES SEED0 CONDITION
  local d; for d in "data/$2"_w*; do
    [ -e "$d" ] && { echo "$d already exists but stage $1 is not marked done: remove it or 'touch $P/$1.done'"; return 1; }
  done
  DUO_GPU=$GPU in_env "$ISAAC_ENV" bash sim/collect_parallel.sh "data/$2" "$3" "$4" "$5" --condition "$6"
}

offline_lc() { # offline_lc MODEL : action error on the low-contrast validation episodes (never trained on)
  local m=$1 args=() w
  for ((w = 0; w < VAL_WORKERS; w++)); do args+=("lc$w=data/low_contrast_val_w$w"); done
  in_env "$TRAIN_ENV" env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_offline.py \
      --checkpoint "runs/${EXP}_$m/$CK" --roots "${args[@]}" --out "results/${EXP}_offline_lowcontrast_$m.json"
}

closed_loop() { # closed_loop MODEL : serve it once, roll it out under normal and low_contrast with the same seeds
  local m=$1 ck="runs/${EXP}_${1}/${CK}_deploy" slog="logs/serve_${EXP}_${1}.log" pid cond rc=0
  [ -d "$ck" ] || { echo "missing $ck"; return 1; }
  ( conda activate "$TRAIN_ENV" && exec env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/serve_policy.py \
      --checkpoint "$ck" --port "$PORT" ) > "$slog" 2>&1 &
  pid=$!
  for _ in $(seq 1 180); do grep -q "listening" "$slog" 2>/dev/null && break; kill -0 "$pid" 2>/dev/null || break; sleep 5; done
  grep -q "listening" "$slog" || { echo "policy server did not start, see $slog"; kill "$pid" 2>/dev/null; return 1; }
  for cond in normal low_contrast; do
    DUO_GPU=$GPU in_env "$ISAAC_ENV" python sim/eval_closed_loop.py --port "$PORT" --episodes "$EVAL_EPS" --seed0 2000000 \
        --condition "$cond" --out "results/closed_${EXP}_${m}_${cond}.json" || rc=1
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  return $rc
}

summary() {
  in_env "$TRAIN_ENV" python - "$EXP" <<'EOF'
import json, math, os, sys
exp = sys.argv[1]
def load(m, c):
    f = f"results/closed_{exp}_{m}_{c}.json"
    return json.load(open(f)) if os.path.exists(f) else None
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))
print("Closed-loop success (identical seeds for every row; 95% Wilson interval)")
print(f"{'model (trained on normal only)':32s} {'tested on':14s} {'success':>9s} {'rate':>6s}  {'95% CI':>13s}  by arm")
for m in ("rgb", "rgbd"):
    for c in ("normal", "low_contrast"):
        r = load(m, c)
        if not r: print(f"{m:32s} {c:14s}   missing"); continue
        s = r["summary"]; lo, hi = wilson(s["successes"], s["episodes"])
        tag = "zero-shot" if c == "low_contrast" else ""
        print(f"{m:32s} {c:14s} {s['successes']:>4d}/{s['episodes']:<4d} {s['success_rate']:>6.2f}  [{lo:.2f}, {hi:.2f}]  {s['by_arm']} {tag}")
EOF
}

say "plan: GPU=$GPU  train=normal only, $((WORKERS * EPS)) episodes ($WORKERS workers x $EPS)  val=normal $((VAL_WORKERS * VAL_EPS)) + low_contrast $((VAL_WORKERS * LC_VAL_EPS)) (diagnostic)  steps=$STEPS (ckpt every $SAVE_EVERY)  closed-loop=$EVAL_EPS episodes x {rgb,rgbd} x {normal,low_contrast}"

# 1. record (always RGB-D). Normal training + validation; a small low-contrast VALIDATION set (never trained on)
stage rec_normal_train        record rec_normal_train        normal_train        "$WORKERS"     "$EPS"        0       normal
stage rec_normal_val          record rec_normal_val          normal_val          "$VAL_WORKERS" "$VAL_EPS"    1000000 normal
stage rec_low_contrast_val    record rec_low_contrast_val    low_contrast_val    "$VAL_WORKERS" "$LC_VAL_EPS" 1000000 low_contrast

# 2. look-at-the-data artefacts (never fatal)
SOFT=1 stage inspect_normal       in_env "$TRAIN_ENV" python scripts/inspect_data.py --root data/normal_train_w0       --out runs/inspect_normal.png
SOFT=1 stage inspect_low_contrast in_env "$TRAIN_ENV" python scripts/inspect_data.py --root data/low_contrast_val_w0   --out runs/inspect_low_contrast.png

# 3. train RGB-only and RGB-D on the normal data (same seed / recipe), merge, offline eval on normal validation
stage "train_$EXP" in_env "$TRAIN_ENV" env CUDA_VISIBLE_DEVICES=$GPU bash scripts/run_experiment.sh "$EXP" \
    "$(roots normal_train "$WORKERS")" "$(roots normal_val "$VAL_WORKERS")" "$STEPS" "train.save_every=$SAVE_EVERY"

# 3b. offline action error on the low-contrast validation episodes (depth on / off for RGB-D)
for m in rgb rgbd; do SOFT=1 stage "offline_lc_$m" offline_lc "$m"; done

# 4. closed loop: each model under normal and low_contrast, identical seeds
for m in rgb rgbd; do stage "closed_$m" closed_loop "$m"; done

say "ALL DONE"
[ "${DRY_RUN:-0}" = 1 ] || { summary | tee results/summary.txt; }
