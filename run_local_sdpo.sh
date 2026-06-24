#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="${CONFIG_NAME:-sdpo}"

# Resolve paths for the local machine. Override these from the environment if the
# repository, dataset root, or model live elsewhere.
export PROJECT_ROOT="${PROJECT_ROOT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
export DATA_ROOT="${DATA_ROOT:-/media/datasets/cheliu21/cxy_worldmodel/sdpo}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/test_full/models/Qwen3-8B}"

# Dataset path relative to DATA_ROOT. user.yaml expands this into:
#   ${DATA_ROOT}/${DATA_PATH}/train.parquet
#   ${DATA_ROOT}/${DATA_PATH}/test.parquet
DATA_PATH="${DATA_PATH:-datasets/sciknoweval/biology}"

# Hugging Face cache paths under DATA_ROOT by default.
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Hyperparameters.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
LAMBDA="${LAMBDA:-0.0}"
CLIP_ADV_HIGH="${CLIP_ADV_HIGH:-null}"
DONTS_REPROMPT_ON_SELF_SUCCESS="${DONTS_REPROMPT_ON_SELF_SUCCESS:-True}"
# Forward KL: 0, Reverse KL: 1, Renyi Forward: 0.25, Renyi Reverse: 0.75, JSD: 0.5
ALPHA="${ALPHA:-0.25}"
RHO="${RHO:-0.95}"
REGULATION_LEVEL="${REGULATION_LEVEL:-0.9}"
SEED="${SEED:-42}"
ENTROPY_COEFF="${ENTROPY_COEFF:-1e-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-800}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${N_GPUS_PER_NODE:-16}}"
export GPUS_PER_NODE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}"

# Allow overriding experiment name suffix.
SUFFIX="${1:-local_renyi}"

# =============================================================================
# SETUP
# =============================================================================

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

mkdir -p \
    "${DATA_ROOT}/output/SDPO" \
    "${DATA_ROOT}/checkpoints" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}"

cd "${PROJECT_ROOT}"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "${MODEL_PATH}" | tr '/' '-')
TIMESTAMP=$(date +%s)
EXP_NAME="seed${SEED}-LOCAL-REGU-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rho${RHO}-regu${REGULATION_LEVEL}-entropy${ENTROPY_COEFF}-mbs32-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}-${TIMESTAMP}"

ARGS="vars.dir=${DATA_ROOT} \
vars.log_dir=${DATA_ROOT}/output \
vars.ckpt_dir=${DATA_ROOT}/checkpoints \
custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py \
data.train_batch_size=${TRAIN_BATCH_SIZE} \
data.seed=${SEED} \
trainer.group_name=SDPO-local \
trainer.seed=${SEED} \
trainer.n_gpus_per_node=${GPUS_PER_NODE} \
trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE} \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.optim.lr=${LR} \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
actor_rollout_ref.actor.self_distillation.rho=${RHO} \
actor_rollout_ref.actor.self_distillation.renyi_regularization=True \
actor_rollout_ref.actor.self_distillation.renyi_regularization_level=${REGULATION_LEVEL} \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.enforce_eager=True"

TRAIN_FILE="${DATA_ROOT}/${DATA_PATH%/}/train.parquet"
VAL_FILE="${DATA_ROOT}/${DATA_PATH%/}/test.parquet"
CKPT_DIR="${DATA_ROOT}/checkpoints/${EXP_NAME}"

cat <<EOM
----------------------------------------------------------------
Starting Local SDPO Training
Experiment: ${EXP_NAME}
Project root: ${PROJECT_ROOT}
Data root: ${DATA_ROOT}
Data path: ${DATA_PATH}
Train file: ${TRAIN_FILE}
Val file: ${VAL_FILE}
Model: ${MODEL_PATH}
Checkpoint dir: ${CKPT_DIR}
GPUs: ${CUDA_VISIBLE_DEVICES}
Total training steps: ${TOTAL_TRAINING_STEPS}
----------------------------------------------------------------
EOM

bash "${PROJECT_ROOT}/training/verl_training.sh" "${EXP_NAME}" "${CONFIG_NAME}" "${DATA_PATH}" ${ARGS}
