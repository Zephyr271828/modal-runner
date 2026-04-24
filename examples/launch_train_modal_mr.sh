#!/bin/bash
# Launch SDAR-MTP training jobs on Modal via modal-runner.
#
# Any env var you set on the `modal-runner run` line is forwarded into the
# container and reaches train_sdar_mtp.sh unchanged, so you can pass
# hyperparameters in the natural shell style:
#
#   LR=1e-3 LOSS_TYPE=kd_diff mtp_init_std=0.2 modal-runner run ./train.sh ...
#
# DATA_PATH / MODEL_PATH / OUTPUT_PATH are special: modal-runner uploads them
# to the Modal volume before the run and pulls OUTPUT_PATH back afterwards.
#
# modal-runner handles:
#   • background detach (no nohup wrapping)
#   • Modal GPU-budget queueing (MAX_MODAL_GPUS)
#   • auto-resume on FunctionTimeoutError / NCCL watchdog / control-plane errors
#   • per-app logs at logs/<name>/<timestamp>.log
#
# Launch with:   bash scripts/launch_train_modal_mr.sh
#
# Prereqs:
#   pip install -e /mnt/weka/home/yucheng/yufeng/modal-runner
#   modal setup    # authenticate once
#
# Image: override with a pre-built image that has torch/flash_attn/etc.

set -euo pipefail

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CUR_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Shared modal-runner flags (apply to every entry) ────────────────────────
DATASET="${DATASET:-sftdatasetv3}"
SEQ_LEN="${SEQ_LEN:-4096}"
GPU_TYPE="${GPU_TYPE:-H100}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_MODAL_GPUS="${MAX_MODAL_GPUS:-64}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MR_IMAGE="${MR_IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
MR_PIP_INSTALL="${MR_PIP_INSTALL:-transformers accelerate deepspeed wandb datasets peft sentencepiece}"

# Paths the user script wants preserved end-to-end.
DATA_PATH_COMMON="${REPO_ROOT}/src/training/sdar/tokenized_cache/${DATASET}/seq_${SEQ_LEN}"
OUTPUT_PATH_COMMON="${REPO_ROOT}/checkpoints"

mr() {
  # One dispatch helper. Positional arg = app name. The rest of the line is
  # read as `VAR=VAL VAR=VAL ...` shell assignments exactly like a normal
  # inline-env invocation of the training script.
  local app_name="$1"; shift
  # shellcheck disable=SC2068  # intentional: forward caller's inline-env tokens
  env \
    DATA_PATH="${DATA_PATH_COMMON}" \
    OUTPUT_PATH="${OUTPUT_PATH_COMMON}" \
    DATASET="${DATASET}" \
    SEQ_LEN="${SEQ_LEN}" \
    $@ \
    modal-runner run "${REPO_ROOT}/scripts/train_sdar_mtp.sh" \
      --name "${app_name}" \
      --num-gpus "${NUM_GPUS}" \
      --gpu-type "${GPU_TYPE}" \
      --repo-dir "${REPO_ROOT}" \
      --image "${MR_IMAGE}" \
      --pip-install "${MR_PIP_INSTALL}" \
      --max-retries "${MAX_RETRIES}" \
      --max-modal-gpus "${MAX_MODAL_GPUS}" \
      --log-dir "${REPO_ROOT}/logs"
}

mkdir -p "${REPO_ROOT}/logs"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRIES — one line per job. Pass whatever env train_sdar_mtp.sh reads.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# [C] kd_diff ablation at 1.7b / 4b / 8b, 3lyr
mr sdar-1_7b-3lyr-kd_diff  \
  MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-3lyr" \
  MODEL_NAME=mtp_sdar-1_7b-chat-b16-3lyr \
  LOSS_TYPE=kd_diff \
  LR=1e-3 \
  REVEAL_MODE=gt \
  GT_TOPK=1 \
  INPUT_PREP_MODE=bd_packed \
  FREEZE_BACKBONE=true \
  FREEZE_LM_HEAD=true \
  mtp_init_std=0.2

mr sdar-4b-3lyr-kd_diff \
  MODEL_PATH="${REPO_ROOT}/src/models/SDAR-4B-Chat-b16-MTP-3lyr" \
  MODEL_NAME=mtp_sdar-4b-chat-b16-3lyr \
  LOSS_TYPE=kd_diff \
  LR=1e-3 \
  REVEAL_MODE=gt \
  GT_TOPK=1 \
  INPUT_PREP_MODE=bd_packed FREEZE_BACKBONE=true FREEZE_LM_HEAD=true \
  mtp_init_std=0.2

mr sdar-8b-3lyr-kd_diff \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff \
   LR=1e-3 \
   REVEAL_MODE=gt \
   GT_TOPK=1 \
   INPUT_PREP_MODE=bd_packed FREEZE_BACKBONE=true FREEZE_LM_HEAD=true \
   mtp_init_std=0.2

# [F] K=1 reveal_gt_1 at 2lyr / 4lyr
mr sdar-1_7b-2lyr-kd_diff_sum \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-2lyr" \
   MODEL_NAME=mtp_sdar-1_7b-chat-b16-2lyr \
   LOSS_TYPE=kd_diff_sum LR=1e-3 \
   REVEAL_MODE=gt GT_TOPK=1 \
   INPUT_PREP_MODE=bd_packed FREEZE_BACKBONE=true FREEZE_LM_HEAD=true \
   mtp_init_std=0.2

mr sdar-1_7b-4lyr-kd_diff_sum \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-4lyr" \
   MODEL_NAME=mtp_sdar-1_7b-chat-b16-4lyr \
   LOSS_TYPE=kd_diff_sum \
   LR=1e-3 \
   REVEAL_MODE=gt GT_TOPK=1 \
   INPUT_PREP_MODE=bd_packed FREEZE_BACKBONE=true FREEZE_LM_HEAD=true \
   mtp_init_std=0.2
