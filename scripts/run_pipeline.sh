#!/usr/bin/env bash
# Unattended end-to-end run:  record (normal RGB-D) -> train RGB-only and RGB-D on NORMAL data only -> offline eval ->
# closed-loop eval of both models under normal AND low_contrast (zero-shot) -> summary.
# Two conda envs, activated per stage: the Isaac env records and evaluates in simulation, the training env trains and serves.
# Resumable: every finished stage writes logs/pipeline/<stage>.done and a re-run skips those. Nothing is deleted or redone
# on its own: to redo a stage remove its .done file (and its data / run directory).
#
#   export OMNI_KIT_ACCEPT_EULA=YES                 # your decision: the Omniverse EULA (the script refuses to start without it)
#   tmux new -s pipe                                # then inside tmux (detach: Ctrl-b d, reattach: tmux attach -t pipe)
#   GPU=<n> SCENARIO=open_drawer bash scripts/run_pipeline.sh 2>&1 | tee -a logs/pipeline.log
#   (without tmux:  GPU=<n> SCENARIO=... bash scripts/run_detached.sh pipeline bash scripts/run_pipeline.sh)
#
# What is trained: ONLY normal-condition data. The low_contrast recordings are a small VALIDATION set (offline diagnostic);
# they never enter training. Low contrast is tested zero-shot in closed loop.
# Settings (environment variables; defaults in brackets):
#   SCENARIO (required) pick_place | pick_place_camo | open_drawer; recordings in data/${SCENARIO}_*
#   RIG [full] full = calibrated 1280x720 frames stored uncropped (wrist depth 7-50 cm, rest invalid), each encoder gets its
#     own letterboxed resize; square = the earlier 640x480 rig cropped to squares. Own data: data/${SCENARIO}_full_*
#   GPU (required) the physical GPU for Isaac AND training;  ISAAC_ENV [defmvla-isaac]  TRAIN_ENV [defmvla]
#   WORKERS [4] x EPS [25] = training episodes (100); VAL_WORKERS [2] x VAL_EPS [10] normal validation episodes (20);
#   LC_VAL_EPS [5] per validation worker of low-contrast validation episodes (10, offline diagnostic only);
#   STEPS [10000] per model, checkpoints every SAVE_EVERY [2000]; CK_RULE [best] evaluate the saved checkpoint with the
#     lowest NORMAL validation loss (last = the final one);
#   EXP [${SCENARIO}_full, or ${SCENARIO}_vlmlora with RIG=square] tag for the trained models (a new tag retrains; recordings are reused: their stages are .done);
#   TRAIN_EXTRA [see below] train.py overrides, the same for the RGB and the RGB-D model
#     (TRAIN_EXTRA="" = the plain configs/train_*.yaml recipe)
#   EVAL_EPS [40] closed-loop episodes per (model, condition) [4 runs]; PORT [6040];  DRY_RUN=1 prints the plan only.
# Seeds: train 0.. (worker w: +100000*w), validation 1,000,000.. (low-contrast validation uses the SAME layouts),
# closed-loop eval 2,000,000.. (identical for every model and condition, so the comparisons are paired).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ISAAC_ENV=${ISAAC_ENV:-defmvla-isaac}; TRAIN_ENV=${TRAIN_ENV:-defmvla}; GPU=${GPU:-}
WORKERS=${WORKERS:-4}; EPS=${EPS:-25}; VAL_WORKERS=${VAL_WORKERS:-2}; VAL_EPS=${VAL_EPS:-10}; LC_VAL_EPS=${LC_VAL_EPS:-5}
STEPS=${STEPS:-10000}; SAVE_EVERY=${SAVE_EVERY:-2000}; EVAL_EPS=${EVAL_EPS:-40}; PORT=${PORT:-6040}
SCENARIO=${SCENARIO:-}                  # REQUIRED: pick_place | pick_place_camo | open_drawer (sim/duo_env.py)
RIG=${RIG:-full}
case "$RIG" in
  full)   EXP=${EXP:-${SCENARIO}_full}; RIG_ARGS=(--cameras-yaml sim/cameras_full.yaml --square 0) ;;
  square) EXP=${EXP:-${SCENARIO}_vlmlora}; RIG_ARGS=() ;;
  *) echo "RIG=full|square"; exit 2 ;;
esac                                    # tag of the trained models: runs/${EXP}_{rgb,rgbd}, results/closed_${EXP}_*
export CK_RULE=${CK_RULE:-best}
# entry-VLA's recipe (whole action expert, 16-step chunks, 8 executed, arm prompt) + backbone adaptation: the connector
# (image projector) trains fully, rank-16 LoRA on the attention of all 16 VLM layers (depth joins before the first one), both at half the lr
TRAIN_EXTRA=${TRAIN_EXTRA-train.mode=expert model.horizon=16 model.n_action_steps=8 data.arm_prompt=true model.vlm_lora_r=16 model.vlm_lora_layers=16 model.vlm_lora_dropout=0.05 train.train_connector=true train.vlm_lr_mult=0.5 train.val_every=1000 train.val_batches=64}
[ "$RIG" = full ] && TRAIN_EXTRA="$TRAIN_EXTRA model.native_resize=true model.depth_mask_invalid=true model.wrist_depth_range=[0.07,0.5]"
case "$SCENARIO" in pick_place|pick_place_camo|open_drawer) ;;
  *) echo "set SCENARIO=pick_place|pick_place_camo|open_drawer (got '$SCENARIO')"; exit 2 ;; esac
D=$SCENARIO; [ "$RIG" = full ] && D=${SCENARIO}_full                         # data prefix: data/${SCENARIO}_{normal_train,normal_val,low_contrast_val}_w*
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "set GPU=<number> (nvidia-smi shows which one is free)"; exit 2; }
selected() { cat "runs/${EXP}_$1/selected_checkpoint.txt"; }   # written by run_experiment.sh
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
  DUO_GPU=$GPU in_env "$ISAAC_ENV" bash sim/collect_parallel.sh "data/$2" "$3" "$4" "$5" --condition "$6" --scenario "$SCENARIO" "${RIG_ARGS[@]}"
}

offline_lc() { # offline_lc MODEL : action error on the low-contrast validation episodes (never trained on)
  local m=$1 args=() w
  for ((w = 0; w < VAL_WORKERS; w++)); do args+=("lc$w=data/${D}_low_contrast_val_w$w"); done
  in_env "$TRAIN_ENV" env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_offline.py \
      --checkpoint "runs/${EXP}_$m/$(selected "$m")" --roots "${args[@]}" --out "results/${EXP}_offline_lowcontrast_$m.json"
}

closed_loop() { # closed_loop MODEL : serve it once, roll it out under normal and low_contrast with the same seeds
  local m=$1 ck="runs/${EXP}_${1}/$(selected "$1")_deploy" slog="logs/serve_${EXP}_${1}.log" pid cond rc=0
  [ -d "$ck" ] || { echo "missing $ck"; return 1; }
  ( conda activate "$TRAIN_ENV" && exec env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/serve_policy.py \
      --checkpoint "$ck" --port "$PORT" ) > "$slog" 2>&1 &
  pid=$!
  for _ in $(seq 1 180); do grep -q "listening" "$slog" 2>/dev/null && break; kill -0 "$pid" 2>/dev/null || break; sleep 5; done
  grep -q "listening" "$slog" || { echo "policy server did not start, see $slog"; kill "$pid" 2>/dev/null; return 1; }
  # Isaac exits 0 even when it crashes at start-up (seen: "ReferenceTime annotator not registered" while creating the
  # cameras, intermittent), so success = the result JSON exists. Up to 4 attempts per (model, condition); the episodes
  # are seeded, so a retry reproduces exactly what the first attempt would have done.
  for cond in normal low_contrast; do
    local out="results/closed_${EXP}_${m}_${cond}.json" try
    [ -s "$out" ] && continue
    for try in 1 2 3 4; do
      DUO_GPU=$GPU in_env "$ISAAC_ENV" python sim/eval_closed_loop.py --scenario "$SCENARIO" --port "$PORT" --episodes "$EVAL_EPS" --seed0 2000000 \
          --condition "$cond" "${RIG_ARGS[@]}" --out "$out"
      [ -s "$out" ] && break
      echo "attempt $try for $m/$cond produced no result file; retrying"; sleep 10
    done
    [ -s "$out" ] || rc=1
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
print("\nDoes the RGB-D model use depth? Action error on held-out demos with depth on / off (same weights; off should be worse)")
for c, f in (("normal", f"results/{exp}_offline_rgbd.json"), ("low_contrast", f"results/{exp}_offline_lowcontrast_rgbd.json")):
    if not os.path.exists(f): print(f"{c:14s}   missing"); continue
    sp = list(json.load(open(f))["splits"].values()); n = sum(s["chunks"] for s in sp)
    on, off = (sum(s[k] * s["chunks"] for s in sp) / n for k in ("mae_depth", "mae_no_depth"))
    print(f"{c:14s} on {on:.5f}  off {off:.5f}  off vs on {100 * (off / on - 1):+.1f} %")
f = f"runs/{exp}_rgbd/train_log.jsonl"
if os.path.exists(f):
    r = [json.loads(x) for x in open(f) if x.strip()][-1]
    print(f"end of training: depth_gate {r['depth_gate']}  depth signal / RGB token size {r.get('depth_ratio', 'not logged')}")
for m in ("rgb", "rgbd"):
    f = f"runs/{exp}_{m}/selected_checkpoint.txt"
    if os.path.exists(f): print(f"evaluated checkpoint {m}: {open(f).read().strip()}")
EOF
}

say "plan: SCENARIO=$SCENARIO EXP=$EXP extra=[$TRAIN_EXTRA] GPU=$GPU  train=normal only, $((WORKERS * EPS)) episodes ($WORKERS workers x $EPS)  val=normal $((VAL_WORKERS * VAL_EPS)) + low_contrast $((VAL_WORKERS * LC_VAL_EPS)) (diagnostic)  steps=$STEPS (ckpt every $SAVE_EVERY)  closed-loop=$EVAL_EPS episodes x {rgb,rgbd} x {normal,low_contrast}"

# 1. record (always RGB-D). Normal training + validation; a small low-contrast VALIDATION set (never trained on)
stage "rec_${D}_normal_train"     record "rec_${D}_normal_train"     "${D}_normal_train"     "$WORKERS"     "$EPS"        0       normal
stage "rec_${D}_normal_val"       record "rec_${D}_normal_val"       "${D}_normal_val"       "$VAL_WORKERS" "$VAL_EPS"    1000000 normal
stage "rec_${D}_low_contrast_val" record "rec_${D}_low_contrast_val" "${D}_low_contrast_val" "$VAL_WORKERS" "$LC_VAL_EPS" 1000000 low_contrast

# 2. look-at-the-data artefacts (never fatal)
SOFT=1 stage "inspect_${D}_normal"       in_env "$TRAIN_ENV" python scripts/inspect_data.py --root "data/${D}_normal_train_w0"     --out "runs/inspect_${D}_normal.png"
SOFT=1 stage "inspect_${D}_low_contrast" in_env "$TRAIN_ENV" python scripts/inspect_data.py --root "data/${D}_low_contrast_val_w0" --out "runs/inspect_${D}_low_contrast.png"

# 3. train RGB-only and RGB-D on the normal data (same seed / recipe), merge, offline eval on normal validation
stage "train_$EXP" in_env "$TRAIN_ENV" env CUDA_VISIBLE_DEVICES=$GPU bash scripts/run_experiment.sh "$EXP" \
    "$(roots "${D}_normal_train" "$WORKERS")" "$(roots "${D}_normal_val" "$VAL_WORKERS")" "$STEPS" "train.save_every=$SAVE_EVERY" $TRAIN_EXTRA

# 3b. offline action error on the low-contrast validation episodes (depth on / off for RGB-D)
for m in rgb rgbd; do SOFT=1 stage "offline_lc_${EXP}_$m" offline_lc "$m"; done

# 4. closed loop: each model under normal and low_contrast, identical seeds
for m in rgb rgbd; do stage "closed_${EXP}_$m" closed_loop "$m"; done

say "ALL DONE"
[ "${DRY_RUN:-0}" = 1 ] || { summary | tee "results/summary_$EXP.txt" results/summary.txt; }
