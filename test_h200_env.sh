#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sdpo-h200}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/workspace/SDPO}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"

echo "== System =="
nvidia-smi || true
nvcc --version || true

echo "== Core imports =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))

import flash_attn
print("flash-attn ok")

import vllm
print("vllm:", vllm.__version__)

from vllm.v1.engine.utils import CoreEngineProcManager
print("CoreEngineProcManager ok")

import datasets
import hydra
import omegaconf
import verl
print("SDPO/verl imports ok")
PY

echo "== Tiny vLLM smoke test =="
python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
    max_model_len=4096,
    trust_remote_code=True,
)

outputs = llm.generate(
    ["Answer briefly: what is DNA?"],
    SamplingParams(max_tokens=64, temperature=0.7),
)

print(outputs[0].outputs[0].text)
print("vLLM Qwen3-8B smoke test ok")
PY

echo "== pip check =="
python -m pip check || true

echo "H200 environment test finished."