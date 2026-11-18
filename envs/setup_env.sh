#!/usr/bin/env bash
# Training / serving environment (Ubuntu, NVIDIA GPU). Run from the repo root.
#   bash envs/setup_env.sh env        # install torch (CUDA 12.8 wheels), lerobot 0.6.1, peft, defm into the environment
#   bash envs/setup_env.sh weights    # pre-download SmolVLA, its SmolVLM2 backbone and DeFM (once; then offline)
#   bash envs/setup_env.sh check      # import check + the test suite
#
# Which environment: an ACTIVE conda env (anything but `base`) or venv is used as is. With none active, a new
# .venv is created. Override with DEFMVLA_PYTHON=/path/to/python. Python 3.11 or 3.12. ffmpeg is NOT required.
# Existing envs: `bash envs/check_machine.sh train` tells you what is missing or has the wrong version first;
# `env` then only installs / upgrades what is needed.
set -euo pipefail

pick_python() {
  if [ -n "${DEFMVLA_PYTHON:-}" ]; then PYBIN="$DEFMVLA_PYTHON"; MANAGED=0
  elif [ -n "${VIRTUAL_ENV:-}" ]; then PYBIN="$VIRTUAL_ENV/bin/python"; MANAGED=0
  elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; then PYBIN="$CONDA_PREFIX/bin/python"; MANAGED=0
  else PYBIN=".venv/bin/python"; MANAGED=1; fi
}
pick_python

case "${1:-}" in
  env)
    if [ "$MANAGED" = 1 ]; then
      if command -v uv >/dev/null 2>&1; then uv venv --python 3.12 .venv; else python3 -m venv .venv; fi
    fi
    echo "installing into: $PYBIN"
    if command -v uv >/dev/null 2>&1; then PIP="uv pip install --python $PYBIN"; else PIP="$PYBIN -m pip install"; fi
    # torch first, from the CUDA 12.8 index alone. In one call with everything else this breaks under uv: the
    # PyTorch index carries a stale `requests`, and uv does not fall back to PyPI for a package found on the
    # first index. (Skipped if a CUDA torch >= 2.7 is already there.)
    if ! "$PYBIN" -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if v>=(2,7) and torch.cuda.is_available() else 1)" 2>/dev/null; then
      $PIP --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.12" torchvision
    fi
    $PIP "lerobot[smolvla]==0.6.1" "peft>=0.13" omegaconf pytest pyyaml pillow scipy
    # defm pins numpy<2, lerobot does not: install it without its dependency pins (it only needs torch,
    # torchvision, numpy, omegaconf and huggingface_hub, all present).
    $PIP --no-deps defm
    $PIP --no-deps -e .
    "$PYBIN" -c "import torch, lerobot, peft, defm, defmvla; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lerobot', lerobot.__version__)"
    ;;
  weights)
    "$PYBIN" - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("lerobot/smolvla_base"))
print(snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))
for f in ("config.json", "defm_vit_s14.pth"):
    print(hf_hub_download("leggedrobotics/defm", f))
EOF
    ;;
  check)
    "$PYBIN" -m pytest -q
    ;;
  *) echo "usage: $0 env|weights|check   (python: $PYBIN)"; exit 2 ;;
esac
