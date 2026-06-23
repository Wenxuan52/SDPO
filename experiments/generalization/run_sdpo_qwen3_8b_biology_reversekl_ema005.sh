#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_sdpo_qwen3_8b_biology_reversekl_ema005.sh [--dry-run]
# Dedicated launch script for:
#   Qwen3-8B + SDPO reverse KL (alpha=1.0) + EMA teacher update rate 0.05
#   on SciKnowEval Biology.
#
# This script submits a Slurm job by default. It assumes the environment was
# prepared by setup_h200_env.sh and only re-exports runtime paths/caches here.

DRY_RUN=false
export USER=${USER:-$(whoami)}
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths / environment. Override these when your cluster uses different mounts.
PROJECT_ROOT="${PROJECT_ROOT:-/media/damoxing/che-liu-fileset/cxy_worldmodel/SDPO}"
DATA_ROOT="${DATA_ROOT:-/media/datasets/cheliu21/cxy_worldmodel/sdpo}"
ENV_NAME="${ENV_NAME:-sdpo-h200}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"

export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# Base settings
CONFIG_NAME="sdpo"
BASE_JOB_NAME="rlvr-biology-rkl"

DATA_PATH="datasets/sciknoweval/biology/"

# Fixed Slurm resources. Defaults target one 8-GPU H20Z/H200 node.
ACCOUNT="${ACCOUNT:-a156}"
NODES="${NODES:-1}"
PARTITION="${PARTITION:-normal}"
TIME="${TIME:-12:00:00}"
SLURM_ENVIRONMENT="${SLURM_ENVIRONMENT:-sdpo}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MEM="${MEM:-460000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-288}"

# Experiment parameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
DONTS_REPROMPT_ON_SELF_SUCCESS="${DONTS_REPROMPT_ON_SELF_SUCCESS:-True}"

# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHA="${ALPHA:-1.0}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# Keep runtime jobs deterministic. setup_h200_env.sh should already have
# installed these deps; set INSTALL_RUNTIME_DEPS=1 only to repair an env.
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-0}"
# auto: use sbatch when available, otherwise run directly in the current shell.
# Set LAUNCH_MODE=sbatch or LAUNCH_MODE=local to force one path.
LAUNCH_MODE="${LAUNCH_MODE:-auto}"

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    mkdir -p "${DATA_ROOT}/output/SDPO" "${DATA_ROOT}/checkpoints" "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

    local setup_cmds="set -euo pipefail; \
if [ -f ${CONDA_SH} ]; then source ${CONDA_SH}; else source \"\$(conda info --base)/etc/profile.d/conda.sh\"; fi; \
conda activate ${ENV_NAME}; \
cd ${PROJECT_ROOT}; \
export PROJECT_ROOT=${PROJECT_ROOT}; \
export DATA_ROOT=${DATA_ROOT}; \
export PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH:-}; \
export HF_HOME=${HF_HOME}; \
export HF_HUB_CACHE=${HF_HUB_CACHE}; \
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}; \
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}; \
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}; \
export WANDB_MODE=${WANDB_MODE}; \
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}; \
if [ ${INSTALL_RUNTIME_DEPS} = 1 ]; then python -m pip install word2number latex2sympy2 math-verify\[antlr4_9_3\]==0.8.0 wandb; fi"

    local run_cmd="bash ${PROJECT_ROOT}/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"
    local wrapped_cmd="srun bash -lc '$setup_cmds; $run_cmd'"
    local local_cmd="bash -lc '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --export=ALL
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --environment="$SLURM_ENVIRONMENT"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="${DATA_ROOT}/output/SDPO/%j.log"
        --error="${DATA_ROOT}/output/SDPO/%j.err"
        --wrap="$wrapped_cmd"
    )

    local effective_launch_mode="$LAUNCH_MODE"
    if [ "$effective_launch_mode" = "auto" ]; then
        if command -v sbatch >/dev/null 2>&1; then
            effective_launch_mode="sbatch"
        else
            effective_launch_mode="local"
        fi
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would run job for: $exp_name"
        echo "PROJECT_ROOT=$PROJECT_ROOT"
        echo "DATA_ROOT=$DATA_ROOT"
        echo "ENV_NAME=$ENV_NAME"
        echo "GPUS_PER_NODE=$GPUS_PER_NODE"
        echo "LAUNCH_MODE=$effective_launch_mode"
        if [ "$effective_launch_mode" = "sbatch" ]; then
            echo "${sbatch_cmd[@]}"
        else
            echo "$local_cmd"
        fi
    elif [ "$effective_launch_mode" = "sbatch" ]; then
        command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found; use LAUNCH_MODE=local to run on this machine" >&2; exit 1; }
        echo "Submitting Slurm job for: $exp_name"
        "${sbatch_cmd[@]}"
    elif [ "$effective_launch_mode" = "local" ]; then
        if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
            export CUDA_VISIBLE_DEVICES
            CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
        fi
        echo "Running local job for: $exp_name"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        bash -lc "$setup_cmds; $run_cmd"
    else
        echo "Unknown LAUNCH_MODE=$LAUNCH_MODE; expected auto, sbatch, or local" >&2
        exit 1
    fi
}

# =============================================================================
# MAIN
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="BIOLOGY-SDPO-Qwen3-8B-reverseKL-alpha${ALPHA}-ema${TEACHER_UPDATE_RATE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"

ARGS="vars.dir=$DATA_ROOT \
vars.log_dir=$DATA_ROOT/output \
vars.ckpt_dir=$DATA_ROOT/checkpoints \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
trainer.n_gpus_per_node=$GPUS_PER_NODE \
data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-biology-reversekl-ema005 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16"

submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
