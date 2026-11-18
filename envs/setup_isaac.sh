#!/usr/bin/env bash
# Isaac Sim 6.2 environment for sim/ (record_duo.py, eval_closed_loop.py). Run from the repo root.
#   bash envs/setup_isaac.sh env      # Isaac Sim 6.2 (pip) + defmvla (recorder / client only) + scipy, pyyaml, opencv
#   bash envs/setup_isaac.sh usd      # build the robot USD from sim/robot (once per machine; the USD holds absolute paths)
#   bash envs/setup_isaac.sh check    # bash envs/check_machine.sh isaac
#
# Environment: an ACTIVE conda env / venv is used as is; otherwise .venv-isaac is created. It is a different env
# from the training one on purpose (Isaac pins its own numpy / torch / warp). Override: ISAAC_PYTHON=/path/to/python.
# If you already have an Isaac env, skip `env`: run `check` in it, then `usd`. sim/ needs Isaac Sim 6.x (the 6.x API:
# isaacsim.core.api, isaacsim.asset.importer.urdf URDFImporter); an older Isaac Sim will not work.
#
# EULA: Isaac Sim asks you to accept the NVIDIA Omniverse EULA on first launch. This script does NOT accept it for
# you; read it, then either answer the prompt or `export OMNI_KIT_ACCEPT_EULA=YES` yourself.
# Torch for the Isaac env comes from TORCH_INDEX (default cu130, which is what was tested with Isaac Sim 6.2 and
# needs a driver >= 580; use https://download.pytorch.org/whl/cu128 on an older driver). DRY_RUN=1 only resolves.
set -euo pipefail

if [ -n "${ISAAC_PYTHON:-}" ]; then PYBIN="$ISAAC_PYTHON"; MANAGED=0
elif [ -n "${VIRTUAL_ENV:-}" ]; then PYBIN="$VIRTUAL_ENV/bin/python"; MANAGED=0
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; then PYBIN="$CONDA_PREFIX/bin/python"; MANAGED=0
else PYBIN=".venv-isaac/bin/python"; MANAGED=1; fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
DRY=""; [ "${DRY_RUN:-0}" = 1 ] && DRY="--dry-run"

case "${1:-}" in
  env)
    command -v uv >/dev/null 2>&1 || { echo "uv is needed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
    [ "$MANAGED" = 1 ] && [ -z "$DRY" ] && uv venv --python 3.12 .venv-isaac
    [ "$MANAGED" = 1 ] && [ -n "$DRY" ] && PYBIN=${PYTHON:-python3}
    echo "installing into: $PYBIN"
    P="uv pip install $DRY --python $PYBIN"
    $P --index-url "$TORCH_INDEX" "torch==2.11.0" "torchvision==0.26.0"
    # Isaac Sim lives on NVIDIA's index; its dependencies on PyPI. unsafe-best-match lets uv take each package from
    # whichever index has the right version (the default only looks at the first index that knows the name).
    $P --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match "isaacsim[all,extscache]==6.2.0.0"
    $P scipy pyyaml pillow opencv-python-headless
    $P --no-deps -e .
    ;;
  usd)
    "$PYBIN" sim/build_usd.py
    ;;
  check)
    DEFMVLA_PYTHON="$PYBIN" bash envs/check_machine.sh isaac
    ;;
  *) echo "usage: $0 env|usd|check   (python: $PYBIN)"; exit 2 ;;
esac
