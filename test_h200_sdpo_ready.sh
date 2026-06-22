#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sdpo-h200}"
PROJECT_ROOT="${PROJECT_ROOT:-/media/damoxing/che-liu-fileset/cxy_worldmodel/SDPO}"
DATA_ROOT="${DATA_ROOT:-/media/datasets/cheliu21/cxy_worldmodel/sdpo}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
RUN_VLLM_SMOKE="${RUN_VLLM_SMOKE:-0}"

TARGET_SCRIPT="experiments/generalization/run_sdpo_qwen3_8b_biology_reversekl_ema005.sh"
TRAIN_ENTRY="training/verl_training.sh"
BASIC_READY=0
DRY_RUN_READY=0
VLLM_SMOKE_READY=0

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    fail "conda was not found; install conda or make it available on PATH before running this readiness check"
  fi

  conda activate "$ENV_NAME" || fail "could not activate conda environment: $ENV_NAME"
  log "Activated conda environment: $ENV_NAME"
}

check_path_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || fail "$label directory does not exist: $path"
  log "$label exists: $path"
}

check_writable_dir() {
  local path="$1"
  local label="$2"
  check_path_dir "$path" "$label"
  local probe="$path/.sdpo_ready_write_test.$$"
  : > "$probe" || fail "$label is not writable: $path"
  rm -f "$probe"
  log "$label is writable: $path"
}

log "SDPO H200/H20Z readiness check"
log "ENV_NAME=$ENV_NAME"
log "PROJECT_ROOT=$PROJECT_ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "MODEL_NAME=$MODEL_NAME"
log "RUN_VLLM_SMOKE=$RUN_VLLM_SMOKE"

activate_conda

[[ -d "$PROJECT_ROOT" ]] || fail "PROJECT_ROOT does not exist: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"

log "Checking repository files"
[[ -f README.md ]] || fail "README.md not found under PROJECT_ROOT"
[[ -f requirements.txt ]] || fail "requirements.txt not found under PROJECT_ROOT"
[[ -f requirements-gh200.txt ]] || warn "requirements-gh200.txt not found under PROJECT_ROOT"
[[ -f requirements-full.txt ]] || warn "requirements-full.txt not found under PROJECT_ROOT"
[[ -f verl/trainer/config/user.yaml ]] || fail "verl/trainer/config/user.yaml not found"
[[ -f verl/trainer/config/sdpo.yaml ]] || fail "verl/trainer/config/sdpo.yaml not found"
[[ -f "$TRAIN_ENTRY" ]] || fail "training entry not found: $TRAIN_ENTRY"
[[ -f "$TARGET_SCRIPT" ]] || fail "target experiment script not found: $TARGET_SCRIPT"
[[ -f verl/workers/rollout/vllm_rollout/vllm_rollout.py ]] || fail "vLLM rollout implementation not found"
[[ -f verl/workers/rollout/vllm_rollout/vllm_async_server.py ]] || fail "vLLM async server implementation not found"

log "Checking data and output directories"
check_path_dir "$DATA_ROOT/datasets" "DATA_ROOT/datasets"
check_writable_dir "$DATA_ROOT/checkpoints" "DATA_ROOT/checkpoints"
check_writable_dir "$DATA_ROOT/output" "DATA_ROOT/output"

log "Checking GPU visibility"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found on PATH"
nvidia-smi -L || fail "nvidia-smi could not list GPUs"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed 's/^/nvcc: /'
else
  warn "nvcc not found on PATH; continuing because some NGC/vLLM runtime containers do not include a compiler"
fi

log "Checking Python imports and CUDA through public/stable APIs"
python - <<'PY'
import importlib
import torch
print(f"torch={torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
count = torch.cuda.device_count()
print(f"torch CUDA device_count={count}")
if count <= 0:
    raise SystemExit("torch reports zero CUDA devices")
for mod in ["flash_attn", "vllm", "ray", "datasets", "hydra", "omegaconf", "transformers", "verl"]:
    imported = importlib.import_module(mod)
    version = getattr(imported, "__version__", "unknown")
    print(f"import {mod}: ok (version={version})")
from vllm import LLM, SamplingParams
print("import vllm public API LLM/SamplingParams: ok")
PY

log "Searching for vLLM private API dependencies in this repository"
PRIVATE_MATCHES="$(rg -n 'vllm[.]v1[.]engine[.]utils|CoreEngineProcManager' . -g '!test_h200_sdpo_ready.sh' -g '!*.pyc' -g '!*.log' || true)"
if [[ -n "$PRIVATE_MATCHES" ]]; then
  warn "Found references to vLLM private/internal paths. These are not tested here unless training code depends on them at runtime; verify vLLM compatibility if async rollout is used."
  printf '%s\n' "$PRIVATE_MATCHES"
else
  log "No references to vLLM private paths found; no private vLLM API import test is needed."
fi

BASIC_READY=1
printf '\nREADY: basic environment checks passed\n'

log "Running target experiment dry-run without starting training"
if bash "$TARGET_SCRIPT" --dry-run; then
  DRY_RUN_READY=1
  printf '\nREADY: dry-run passed\n'
else
  warn "Target dry-run failed or --dry-run is unsupported. Training was not started. Suggested manual check: bash $TARGET_SCRIPT --dry-run"
fi

if [[ "$RUN_VLLM_SMOKE" == "1" ]]; then
  log "Running optional vLLM smoke test. This may download the model and allocate GPU memory."
  MODEL_NAME="$MODEL_NAME" python - <<'PY'
import os
from vllm import LLM, SamplingParams
model = os.environ["MODEL_NAME"]
llm = LLM(model=model, tensor_parallel_size=1, dtype="bfloat16", max_model_len=2048, trust_remote_code=True)
outputs = llm.generate(["What is biology? Answer briefly."], SamplingParams(max_tokens=16, temperature=0.0))
print(outputs[0].outputs[0].text.strip())
PY
  VLLM_SMOKE_READY=1
  printf '\nREADY: optional vLLM smoke passed\n'
else
  log "Skipping optional vLLM model-load smoke test because RUN_VLLM_SMOKE=0"
  printf 'READY: optional vLLM smoke skipped (set RUN_VLLM_SMOKE=1 to run)\n'
fi

if [[ "$BASIC_READY" -ne 1 ]]; then
  fail "basic environment checks did not complete"
fi
if [[ "$DRY_RUN_READY" -ne 1 ]]; then
  warn "dry-run readiness did not pass; see warning above"
fi
if [[ "$RUN_VLLM_SMOKE" == "1" && "$VLLM_SMOKE_READY" -ne 1 ]]; then
  fail "optional vLLM smoke was requested but did not pass"
fi
