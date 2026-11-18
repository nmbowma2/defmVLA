#!/usr/bin/env bash
# Record with several Isaac Sim processes at once (one GPU; each process is ~1 min per attempt on its own).
#
#   bash sim/collect_parallel.sh OUT_PREFIX WORKERS EPISODES_PER_WORKER SEED0 [record_duo.py options...]
#   bash sim/collect_parallel.sh data/normal_train 2 50 0          --condition normal
#   bash sim/collect_parallel.sh data/normal_val   1 15 1000000    --condition normal
#   bash sim/collect_parallel.sh data/low_contrast_train 2 50 0    --condition low_contrast     # same seeds => same layouts
#
# Writes OUT_PREFIX_w0 ... OUT_PREFIX_w{N-1}. Worker w takes attempt seeds SEED0 + w*100000 + k, so workers never
# overlap. SEED CONVENTION (keep the splits disjoint): train 0.. , validation 1,000,000.. , closed-loop eval 2,000,000..
# Training reads several roots directly:  data.train_roots='[data/normal_train_w0,data/normal_train_w1,...]'
# Python: ISAAC_PYTHON, else the active python. Worker count: MEASURED ~7 GB of RAM per process (+ ~3 GB of GPU memory),
# so a 31 GB machine can run 2 workers (3 got OOM-killed with a desktop open); RAM_GB / 8 is a good limit. Logs: logs/collect_<name>_w<N>.log
set -uo pipefail
OUT=$1; N=$2; EPS=$3; SEED0=$4; shift 4
PY=${ISAAC_PYTHON:-python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs
name=$(basename "$OUT")
pids=()
for ((w = 0; w < N; w++)); do
  "$PY" "$HERE/record_duo.py" --out "${OUT}_w${w}" --episodes "$EPS" --seed $((SEED0 + w * 100000)) "$@" \
      > "logs/collect_${name}_w${w}.log" 2>&1 < /dev/null &
  pids+=($!)
done
echo "started $N workers (pids ${pids[*]}); logs/collect_${name}_w*.log"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "^SAVED" "logs/collect_${name}"_w*.log
roots=$(for ((w = 0; w < N; w++)); do printf '%s_w%d,' "$OUT" "$w"; done)
echo "data roots: [${roots%,}]"
exit $rc
