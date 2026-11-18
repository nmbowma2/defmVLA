#!/usr/bin/env bash
# Is this machine (and the ACTIVE python environment) ready?   Run from the repo root.
#   bash envs/check_machine.sh train      # in the training / serving env: torch+CUDA, lerobot 0.6.1, peft, defm, defmvla, weights
#   bash envs/check_machine.sh isaac      # in the Isaac Sim env: Isaac Sim 6.x, defmvla, scipy, yaml, robot USD, GPU can render
# Hardware checks (GPU, driver, Vulkan, RAM, disk) always run. Exit status 1 if anything FAILs; WARN is advisory.
ROLE=${1:-train}
PY=${DEFMVLA_PYTHON:-python}
fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

echo "== hardware"
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS=, read -r name mem drv; do
    ok "GPU: $name,$mem MiB, driver $drv"
    major=${drv%%.*}
    [ "$major" -ge 570 ] 2>/dev/null && ok "driver >= 570 (CUDA 12.8 wheels)" || bad "driver $drv < 570: torch cu128 wheels need a newer driver"
    if echo "$name" | grep -qiE 'A100|A30|A800|H100|H200|H800|B100|B200|GB200|V100|P100'; then
      if [ "$ROLE" = isaac ]; then bad "$name has no RT cores: Isaac Sim cannot render camera images on it (training/serving only)"
      else warn "$name has no RT cores: it can train/serve, but record data and run Isaac on an RTX-class GPU"; fi
    fi
  done < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
else bad "nvidia-smi not found"; fi
if [ "$ROLE" = isaac ]; then
  if [ -e /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -e /etc/vulkan/icd.d/nvidia_icd.json ]; then ok "NVIDIA Vulkan ICD present"
  else bad "no NVIDIA Vulkan ICD (/usr/share/vulkan/icd.d/nvidia_icd.json): install the full NVIDIA driver, not a compute-only one"; fi
fi
awk '/MemTotal/{t=$2/1048576} /MemAvailable/{a=$2/1048576} END{printf "  %s    RAM %.0f GB total, %.0f GB available\n", (t>=30?"OK  ":"WARN"), t, a}' /proc/meminfo
echo "        (budget: training loader ~6 GB with 4 workers at batch 32 / 3 cameras @640 (8 workers ~12 GB); each Isaac recorder process ~7 GB)"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 100 ] && ok "disk: ${free_gb} GB free here (an episode is ~260 MB with 3 RGB-D cameras)" || warn "only ${free_gb} GB free here; 100 episodes ~ 26 GB"

echo "== python ($ROLE): $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])' 2>&1)"
$PY - "$ROLE" <<'EOF' || fail=1
import importlib, sys
from importlib import metadata as md
role = sys.argv[1]
bad = 0
def ok(m): print("  OK    " + m)
def warn(m): print("  WARN  " + m)
def fail(m):
    global bad; bad = 1; print("  FAIL  " + m)
def ver(p):
    try: return md.version(p)
    except md.PackageNotFoundError: return None
def need(p, spec=None, hint=""):
    v = ver(p)
    if v is None: fail(f"{p} not installed {hint}"); return None
    if spec and not v.startswith(spec): fail(f"{p} {v}, need {spec}* {hint}"); return v
    ok(f"{p} {v}"); return v
py = sys.version_info
(ok if (3, 11) <= py[:2] <= (3, 12) else lambda m: fail(m + " (need 3.11 or 3.12)"))(f"python {py[0]}.{py[1]}")
if role == "train":
    v = need("torch", hint="(bash envs/setup_env.sh env)")
    if v:
        import torch
        (ok if torch.cuda.is_available() else fail)(f"torch.cuda.is_available() = {torch.cuda.is_available()}  (torch {torch.__version__}, built for CUDA {torch.version.cuda})")
        major = int(v.split(".")[0]); minor = int(v.split(".")[1].split("+")[0])
        if (major, minor) < (2, 7) or (major, minor) >= (2, 12): warn("torch outside the tested range 2.7 to 2.11")
    need("lerobot", "0.6.1", "(pinned: the policy hook wraps lerobot 0.6.1 internals)")
    need("peft"); need("defm"); need("omegaconf"); need("defmvla", hint="(pip install --no-deps -e .)")
    np_v = ver("numpy")
    if np_v: ok(f"numpy {np_v}")
    try:
        from huggingface_hub import try_to_load_from_cache
        for repo, f in (("lerobot/smolvla_base", "config.json"), ("leggedrobotics/defm", "defm_vit_s14.pth")):
            r = try_to_load_from_cache(repo, f)
            (ok if isinstance(r, str) else warn)(f"weights cached: {repo}" if isinstance(r, str) else f"weights not cached: {repo} (bash envs/setup_env.sh weights)")
    except Exception as e:
        warn(f"could not check the weight cache: {e}")
else:
    v = ver("isaacsim")
    if v is None: fail("isaacsim not installed (pip install 'isaacsim[all,extscache]==6.2.0.0', see envs/setup_isaac.sh)")
    elif not v.startswith("6."): fail(f"isaacsim {v}: sim/ is written against the 6.x API (isaacsim.core.api, isaacsim.asset.importer.urdf URDFImporter); older versions will not import")
    else: ok(f"isaacsim {v}")
    need("defmvla", hint="(pip install --no-deps -e <defmVLA>)")
    for m in ("numpy", "PIL", "yaml", "scipy", "cv2"):
        try: importlib.import_module(m); ok(f"import {m}")
        except Exception as e: (fail if m != "cv2" else warn)(f"import {m} failed: {e}")
    import glob, os
    usd = glob.glob(os.path.join("sim", "robot", "usd", "actual", "**", "duo_actual.usd*"), recursive=True)
    (ok if usd else warn)("robot USD built" if usd else "robot USD not built yet: python sim/build_usd.py")
sys.exit(bad)
EOF
[ $? -ne 0 ] && fail=1
echo
[ $fail -eq 0 ] && echo "READY ($ROLE)" || echo "NOT READY ($ROLE): fix the FAIL lines above"
exit $fail
