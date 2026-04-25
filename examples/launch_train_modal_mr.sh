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
# Launch with:   bash examples/launch_train_modal_mr.sh
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
GPU_TYPE="${GPU_TYPE:-B200}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_MODAL_GPUS="${MAX_MODAL_GPUS:-50}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MR_IMAGE="${MR_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"
MR_PIP_INSTALL="${MR_PIP_INSTALL:-}"
# Path to a requirements file used by modal-runner to build the image.
# Picked up by modal_app.py via the MR_REQUIREMENTS env var.
export MR_REQUIREMENTS="${MR_REQUIREMENTS:-${REPO_ROOT}/requirements_modal.txt}"

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
    modal-runner run "${REPO_ROOT}/examples/train_sdar_mtp.sh" \
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
# ║  ENTRIES — ordered by priority. Dispatch is sequential (modal-runner     ║
# ║  detaches each), but the Modal GPU-budget queue enforces the order: the  ║
# ║  second/third batches block on `wait_for_slot` until earlier ones finish ║
# ║  (or free enough GPUs).                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Shared hyperparam defaults. Individual entries override LOSS_TYPE / REVEAL_MODE
# / mtp_steps as needed; everything else stays the same.
_DEFAULTS="LR=1e-3 GT_TOPK=1 INPUT_PREP_MODE=bd_packed FREEZE_BACKBONE=true FREEZE_LM_HEAD=true mtp_init_std=0.2"

# ── Priority 1: 1.7b / 4b / 8b, 3lyr, K=2 ───────────────────────────────────
mr sdar-1_7b-3lyr-K2 \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-1_7b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt mtp_steps=2 ${_DEFAULTS}

mr sdar-4b-3lyr-K2 \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-4B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-4b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt mtp_steps=2 ${_DEFAULTS}

mr sdar-8b-3lyr-K2 \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt mtp_steps=2 ${_DEFAULTS}

# ── Priority 2: 8b 3lyr-2h (multihead) ──────────────────────────────────────
mr sdar-8b-3lyr-2h \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr-2h" \
   MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr-2h \
   LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt_multihead ${_DEFAULTS}

# ── Priority 3: 1.7b / 4b / 8b, 3lyr, kd_diff loss ──────────────────────────
mr sdar-1_7b-3lyr-kd_diff \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-1_7b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff REVEAL_MODE=gt ${_DEFAULTS}

mr sdar-4b-3lyr-kd_diff \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-4B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-4b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff REVEAL_MODE=gt ${_DEFAULTS}

mr sdar-8b-3lyr-kd_diff \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr \
   LOSS_TYPE=kd_diff REVEAL_MODE=gt ${_DEFAULTS}

# ── Priority 4: 1.7b num_layers ablation (1 / 2 / 4 / 8) ────────────────────
# mr sdar-1_7b-1lyr \
#    MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-1lyr" \
#    MODEL_NAME=mtp_sdar-1_7b-chat-b16-1lyr \
#    LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt ${_DEFAULTS}

mr sdar-1_7b-2lyr \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-2lyr" \
   MODEL_NAME=mtp_sdar-1_7b-chat-b16-2lyr \
   LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt ${_DEFAULTS}

mr sdar-1_7b-4lyr \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-4lyr" \
   MODEL_NAME=mtp_sdar-1_7b-chat-b16-4lyr \
   LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt ${_DEFAULTS}

# mr sdar-1_7b-8lyr \
#    MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-8lyr" \
#    MODEL_NAME=mtp_sdar-1_7b-chat-b16-8lyr \
#    LOSS_TYPE=kd_diff_sum REVEAL_MODE=gt ${_DEFAULTS}

echo "[launch] all jobs dispatched (running in background)"
echo "[launch] monitor:  modal-runner jobs"
echo "[launch] logs:     tail -f logs/<app_name>/*.log"
