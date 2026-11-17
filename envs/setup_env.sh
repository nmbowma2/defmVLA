#!/usr/bin/env bash
# Training / serving environment (Ubuntu, NVIDIA GPU). Run from the repo root, on the workstation AND on the
# server, so both have the same versions.
#   bash envs/setup_env.sh env        # create .venv with torch (CUDA 12.8 wheels), lerobot 0.6.1, peft, defm
#   bash envs/setup_env.sh weights    # pre-download SmolVLA, its SmolVLM2 backbone and DeFM (do this once,
#                                     # inside tmux on the server; afterwards training works offline)
#   bash envs/setup_env.sh check      # import check + the test suite
# Uses uv if it is installed, otherwise python3 -m venv + pip. Needs Python 3.11 or 3.12 and ffmpeg is NOT
# required (the dataset is PNG frames, not video).
set -euo pipefail
PY=${PYTHON:-python3}
VENV=.venv
case "${1:-}" in
  env)
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 "$VENV"
      PIP="uv pip install --python $VENV/bin/python"
    else
      "$PY" -m venv "$VENV"
      "$VENV/bin/python" -m pip install --upgrade pip
      PIP="$VENV/bin/python -m pip install"
    fi
    # torch first, from the CUDA 12.8 index alone (the driver on both machines supports it). Installing it in
    # the same call as everything else breaks under uv: the PyTorch index carries a stale `requests`, and uv
    # does not fall back to PyPI for a package that exists on the first index it finds.
    $PIP --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.12" torchvision
    $PIP "lerobot[smolvla]==0.6.1" "peft>=0.13" omegaconf pytest pyyaml pillow
    # defm pins numpy<2, lerobot does not: install it without its dependency pins (it only needs torch,
    # torchvision, numpy, omegaconf and huggingface_hub, all present).
    $PIP --no-deps defm
    $PIP --no-deps -e .
    "$VENV/bin/python" -c "import torch, lerobot, peft, defm, defmvla; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lerobot', lerobot.__version__)"
    ;;
  weights)
    "$VENV/bin/python" - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("lerobot/smolvla_base"))
print(snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))
for f in ("config.json", "defm_vit_s14.pth"):
    print(hf_hub_download("leggedrobotics/defm", f))
EOF
    ;;
  check)
    "$VENV/bin/python" -m pytest -q
    ;;
  *) echo "usage: $0 env|weights|check"; exit 2 ;;
esac
