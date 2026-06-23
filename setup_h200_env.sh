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
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# This repo revision imports vLLM V1 internals from:
#   verl/workers/rollout/vllm_rollout/vllm_async_server.py
# In particular, it requires:
#   from vllm.v1.engine.utils import CoreEngineProcManager
# vLLM 0.8.x does not provide that module in A800/H20/H200 x86_64 envs.
# Override with VLLM_VERSION=... only if the validation below still passes.
VLLM_VERSION="${VLLM_VERSION:-0.12.0}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0}"
# flash-attn 2.7.4.post1 can import-fail with torch 2.9.x due to a C++ ABI
# mismatch (undefined c10::Error symbols). vLLM 0.12.0 resolves torch 2.9.x,
# so use the newer FlashAttention line from the repo's stable vLLM image.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.1}"

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
echo "[setup] torch target:  ${TORCH_VERSION}"
echo "[setup] vllm target:   ${VLLM_VERSION}"
echo "[setup] flash target:  ${FLASH_ATTN_VERSION}"

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

echo "[setup] installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
python -m pip install --upgrade --force-reinstall \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo "[setup] installing vLLM ${VLLM_VERSION} from official PyPI"
python -m pip install --upgrade --force-reinstall "vllm==${VLLM_VERSION}" \
  -i "${PYPI_INDEX_URL}" \
  --extra-index-url "${TORCH_INDEX_URL}"

echo "[setup] re-asserting PyTorch ${TORCH_VERSION} after vLLM dependency resolution"
python -m pip install --upgrade --force-reinstall \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo "[setup] installing SDPO requirements from official PyPI"
REQ_FILE="$(mktemp /tmp/sdpo-h200-req.XXXXXX.txt)"
python - <<PY
from pathlib import Path
import re

src = Path("requirements.txt")
dst = Path("${REQ_FILE}")

# Keep the training/runtime dependency set tight. These are optional/dev-only
# for your target SDPO+vLLM rollout run and are common sources of mirror/build
# failures on cluster images. Match exact package names only: torchdata is a
# real verl runtime dependency and must not be skipped just because it starts
# with "torch".
skip_packages = {
    "liger-kernel",
    "pre-commit",
    "vllm",
    "flash-attn",
    "torch",
    "torchvision",
    "torchaudio",
}

out = []
for raw in src.read_text().splitlines():
    s = raw.strip()
    if not s or s.startswith("#"):
        out.append(raw)
        continue
    match = re.match(r"([A-Za-z0-9_.-]+)", s)
    normalized = match.group(1).lower().replace("_", "-") if match else s
    if normalized in skip_packages:
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
# Remove any previously installed incompatible binary extension before
# reinstalling; otherwise Python may keep loading a stale flash_attn_2_cuda .so.
python -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
python -m pip install --upgrade --force-reinstall --no-deps "flash-attn==${FLASH_ATTN_VERSION}" \
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
    "torchdata",
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
    "torchdata",
]
for name in imports:
    importlib.import_module(name)
    print(f"import ok: {name}")

from vllm import LLM, SamplingParams
print("vllm public API import ok:", LLM, SamplingParams)

# Repo-specific compatibility check for the current vLLM async rollout code.
# Do not remove this: `import vllm` can succeed while the actual SDPO vLLM
# rollout server still fails at startup if this private V1 path is absent.
from vllm.v1.engine.utils import CoreEngineProcManager
print("vllm CoreEngineProcManager import ok:", CoreEngineProcManager)

from torchdata.stateful_dataloader import StatefulDataLoader
print("torchdata StatefulDataLoader import ok:", StatefulDataLoader)

import verl.workers.rollout.vllm_rollout.vllm_async_server
print("verl vLLM async server import ok")

print("validation ok")
PY

echo "[setup] done"
echo "[setup] activate later with: conda activate ${ENV_NAME}"
