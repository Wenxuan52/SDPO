#!/usr/bin/env bash
set -euo pipefail

# SDPO H20/H200 environment setup.
# Scope:
#   - Configure Python packages from official sources only.
#   - Install SDPO/verl editable package.
#   - Validate torch CUDA, vLLM, FlashAttention, Ray, Transformers, and verl imports.
#   - Do NOT preprocess data.
#   - Do NOT modify verl/trainer/config/user.yaml.
#
# Run from the SDPO repository root, or set REPO_DIR explicitly:
#   REPO_DIR=/media/damoxing/che-liu-fileset/cxy_worldmodel/SDPO bash setup_h200_env.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
ENV_NAME="${ENV_NAME:-sdpo-h200}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Official package sources.
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# README says vllm==0.8.4 for GH200/Hopper-style cluster runs.
# Override with VLLM_VERSION=... if the repo/run script requires a different one.
VLLM_VERSION="${VLLM_VERSION:-0.8.4}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"

MAX_JOBS="${MAX_JOBS:-8}"
WANDB_MODE="${WANDB_MODE:-offline}"

DATA_ROOT="${DATA_ROOT:-/media/datasets/cheliu21/cxy_worldmodel/sdpo}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"

echo "[setup] repo:          ${REPO_DIR}"
echo "[setup] env:           ${ENV_NAME}"
echo "[setup] pypi:          ${PYPI_INDEX_URL}"
echo "[setup] torch index:   ${TORCH_INDEX_URL}"
echo "[setup] data root:     ${DATA_ROOT}"
echo "[setup] python target: ${PYTHON_VERSION}"

# Force official sources for pip even if the cluster has a global pip.conf mirror.
export PIP_INDEX_URL="${PYPI_INDEX_URL}"
unset PIP_EXTRA_INDEX_URL
export PIP_DISABLE_PIP_VERSION_CHECK=1

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE
export RAY_DEDUP_LOGS=0
export MAX_JOBS

if [ ! -d "${REPO_DIR}" ]; then
  echo "[setup][error] REPO_DIR does not exist: ${REPO_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"

if [ ! -f "requirements.txt" ]; then
  echo "[setup][error] requirements.txt not found. Run from the SDPO repo root or set REPO_DIR." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup][error] nvidia-smi not found; please run on a GPU node." >&2
  exit 1
fi

ARCH="$(uname -m)"
if [ "${ARCH}" != "x86_64" ]; then
  echo "[setup][error] This script is for x86_64 H20/H100/H200 nodes. Detected: ${ARCH}" >&2
  echo "[setup][error] GH200/aarch64 should use the NGC vLLM container route." >&2
  exit 1
fi

echo "[setup] GPU summary:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
  if [ "${CURRENT_ENV}" = "${ENV_NAME}" ]; then
    echo "[setup] already in conda env: ${ENV_NAME}"
  elif conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] activating existing conda env: ${ENV_NAME}"
    conda activate "${ENV_NAME}"
  else
    echo "[setup] creating conda env: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    conda activate "${ENV_NAME}"
  fi
else
  echo "[setup][warn] conda not found; using current Python environment."
fi

echo "[setup] Python executable: $(python -c 'import sys; print(sys.executable)')"
echo "[setup] Python version:    $(python -V)"

mkdir -p "${DATA_ROOT}/checkpoints" "${DATA_ROOT}/output" "${HF_HOME}"

echo "[setup] upgrading base Python packaging tools"
python -m pip install -U pip setuptools wheel packaging \
  -i "${PYPI_INDEX_URL}"

echo "[setup] installing build helpers from official PyPI"
python -m pip install -U ninja cmake \
  -i "${PYPI_INDEX_URL}" || true

echo "[setup] installing vLLM ${VLLM_VERSION} from official PyPI"
python -m pip install "vllm==${VLLM_VERSION}" \
  -i "${PYPI_INDEX_URL}" \
  --extra-index-url "${TORCH_INDEX_URL}"

echo "[setup] installing SDPO requirements from official PyPI"
REQ_FILE="$(mktemp /tmp/sdpo-h200-req.XXXXXX.txt)"
python - <<PY
from pathlib import Path

src = Path("requirements.txt")
dst = Path("${REQ_FILE}")

# Keep the training/runtime dependency set tight. These are optional/dev-only
# for your target SDPO+vLLM rollout run and are common sources of mirror/build
# failures on cluster images.
skip_prefixes = (
    "liger-kernel",
    "pre-commit",
    "vllm",
    "flash-attn",
    "torch",
    "torchvision",
    "torchaudio",
)

out = []
for raw in src.read_text().splitlines():
    s = raw.strip()
    if not s or s.startswith("#"):
        out.append(raw)
        continue
    normalized = s.split("[", 1)[0].split("=", 1)[0].strip()
    if any(s.startswith(prefix) or normalized == prefix for prefix in skip_prefixes):
        out.append(f"# skipped by setup_h200_env.sh: {raw}")
        continue
    out.append(raw)

dst.write_text("\\n".join(out) + "\\n")
print(dst)
PY
python -m pip install -r "${REQ_FILE}" \
  -i "${PYPI_INDEX_URL}" \
  --extra-index-url "${TORCH_INDEX_URL}"

echo "[setup] installing optional runtime helpers"
python -m pip install hf_transfer \
  -i "${PYPI_INDEX_URL}" || true

python -m pip install liger-kernel \
  -i "${PYPI_INDEX_URL}" || true

echo "[setup] installing FlashAttention ${FLASH_ATTN_VERSION}"
python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" \
  --no-build-isolation \
  -i "${PYPI_INDEX_URL}" || {
    echo "[setup][error] flash-attn installation failed."
    echo "[setup][error] Please provide these diagnostics:"
    echo "  python -V"
    echo "  pip config list -v"
    echo "  which nvcc || true"
    echo "  nvcc --version || true"
    echo "  gcc --version || true"
    echo "  python -c 'import torch; print(torch.__version__, torch.version.cuda)'"
    exit 1
  }

echo "[setup] installing SDPO/verl editable package"
python -m pip install -e . --no-deps \
  -i "${PYPI_INDEX_URL}"

echo "[setup] environment validation"
python - <<'PY'
import importlib
import importlib.metadata as md
import os
import sys

print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("HF_HOME:", os.environ.get("HF_HOME"))
print("WANDB_MODE:", os.environ.get("WANDB_MODE"))

for dist_name in [
    "torch",
    "torchvision",
    "transformers",
    "vllm",
    "ray",
    "flash-attn",
    "verl",
    "datasets",
    "peft",
    "tensordict",
]:
    try:
        print(f"{dist_name}: {md.version(dist_name)}")
    except md.PackageNotFoundError:
        print(f"{dist_name}: NOT FOUND")

import torch
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in torch")

for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"gpu {i}: {prop.name}, cc={cap[0]}.{cap[1]}, mem={prop.total_memory / 1024**3:.1f} GiB")

x = torch.randn((512, 512), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("cuda matmul ok:", tuple(y.shape), str(y.dtype))

imports = [
    "vllm",
    "ray",
    "transformers",
    "flash_attn",
    "verl",
]
for name in imports:
    importlib.import_module(name)
    print(f"import ok: {name}")

print("validation ok")
PY

echo "[setup] done"
echo "[setup] activate later with: conda activate ${ENV_NAME}"
