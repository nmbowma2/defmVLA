#!/usr/bin/env bash
# Film the same seeds with the RGB and the RGB-D model, then make videos and RGB-vs-RGB-D stills (no training).
#   export OMNI_KIT_ACCEPT_EULA=YES                 # your decision
#   GPU=<n> SCENARIO=open_drawer bash scripts/run_film.sh
# Settings: COND [low_contrast]  SEEDS [chosen by pick_film_seeds.py]  EXP [${SCENARIO}_vlmlora]  PORT [6042]
# Output: videos/film_${EXP}_$COND/{rgb,rgbd} (frames), /mp4_rgb, /mp4_rgbd, /stills
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ISAAC_ENV=${ISAAC_ENV:-defmvla-isaac}; TRAIN_ENV=${TRAIN_ENV:-defmvla}; GPU=${GPU:-}; SCENARIO=${SCENARIO:-}
EXP=${EXP:-${SCENARIO}_vlmlora}; COND=${COND:-low_contrast}; PORT=${PORT:-6042}; OUT=videos/film_${EXP}_$COND
[[ "$GPU" =~ ^[0-9]+$ && -n "$SCENARIO" ]] || { echo "set GPU=<number> SCENARIO=<name>"; exit 2; }
[ "${OMNI_KIT_ACCEPT_EULA:-}" = YES ] || { echo "Set OMNI_KIT_ACCEPT_EULA=YES yourself first (NVIDIA Omniverse EULA)."; exit 2; }
mkdir -p logs videos
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "cannot find conda"; exit 2; }
in_env() { local e=$1; shift; ( conda activate "$e" && "$@" ); }
SEEDS=${SEEDS:-$(in_env "$TRAIN_ENV" python scripts/pick_film_seeds.py "$EXP" "$COND")}
[ -n "$SEEDS" ] || { echo "no seeds"; exit 1; }
echo "seeds: $SEEDS"
PID=; trap '[ -n "$PID" ] && kill "$PID" 2>/dev/null' EXIT
for m in rgb rgbd; do
  n=$(wc -w <<< "$SEEDS"); done_n() { local s k=0; for s in $SEEDS; do grep -q "\"seed\": $s," "$OUT/$m/rollouts.json" 2>/dev/null && k=$((k + 1)); done; echo $k; }
  if [ "$(done_n)" != "$n" ]; then
    ck="runs/${EXP}_$m/$(cat "runs/${EXP}_$m/selected_checkpoint.txt")_deploy"; slog="logs/serve_film_${EXP}_$m.log"
    ( conda activate "$TRAIN_ENV" && exec env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU python scripts/serve_policy.py \
        --checkpoint "$ck" --port "$PORT" ) > "$slog" 2>&1 &
    PID=$!
    for _ in $(seq 1 180); do grep -q listening "$slog" && break; kill -0 "$PID" 2>/dev/null || break; sleep 5; done
    grep -q listening "$slog" || { echo "policy server did not start, see $slog"; exit 1; }
    for try in 1 2 3 4; do   # Isaac sometimes crashes at start-up
      # shellcheck disable=SC2086
      DUO_GPU=$GPU in_env "$ISAAC_ENV" python sim/film_seeds.py --results "results/closed_${EXP}_${m}_$COND.json" \
          --seeds $SEEDS --out "$OUT/$m" --port "$PORT" >> "logs/film_${EXP}_$m.log" 2>&1
      [ "$(done_n)" = "$n" ] && break; sleep 10
    done
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=
    [ "$(done_n)" = "$n" ] || { echo "filming $m failed, see logs/film_${EXP}_$m.log"; exit 1; }
  fi
  # shellcheck disable=SC2046
  in_env "$TRAIN_ENV" python scripts/make_video.py --root "$OUT/$m" --episodes $(seq 0 $((n - 1))) --out-dir "$OUT/mp4_$m" || exit 1
done
in_env "$TRAIN_ENV" python scripts/contact_sheet.py "$OUT" && ls "$OUT"
