#!/usr/bin/env bash
# Run a long command so it survives the terminal / the app / an ssh disconnect closing, with a log file.
#
#   bash scripts/run_detached.sh NAME COMMAND [ARGS...]
#   bash scripts/run_detached.sh rgbd_normal bash scripts/run_experiment.sh normal data/normal_train_w0 data/normal_val 5000
#
# Uses (in order): systemd-run --user (survives everything, and gets its own memory accounting), tmux, setsid.
# The current PATH / conda / venv / CUDA / HF variables are passed on, so an activated conda env keeps working.
# Log: logs/NAME.log   Follow it: tail -f logs/NAME.log   Stop it: systemctl --user stop defmvla-NAME  (or tmux kill-session -t NAME)
set -euo pipefail
NAME=$1; shift
mkdir -p logs
LOG="$PWD/logs/$NAME.log"
if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  ENVS=()
  for v in PATH LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV CUDA_VISIBLE_DEVICES HF_HOME HF_HUB_OFFLINE HF_TOKEN \
           OMNI_KIT_ACCEPT_EULA DEFMVLA_PYTHON ISAAC_PYTHON PYTHONPATH; do
    [ -n "${!v:-}" ] && ENVS+=(--setenv="$v")
  done
  systemd-run --user --unit="defmvla-$NAME" --collect --working-directory="$PWD" "${ENVS[@]}" \
      -p StandardOutput=append:"$LOG" -p StandardError=append:"$LOG" "$@"
  echo "started systemd unit defmvla-$NAME   log: $LOG   stop: systemctl --user stop defmvla-$NAME"
elif command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$NAME" "cd '$PWD' && $* 2>&1 | tee -a '$LOG'"
  echo "started tmux session $NAME   log: $LOG   attach: tmux attach -t $NAME"
else
  setsid nohup "$@" >> "$LOG" 2>&1 < /dev/null &
  echo "started (setsid) pid $!   log: $LOG"
fi
