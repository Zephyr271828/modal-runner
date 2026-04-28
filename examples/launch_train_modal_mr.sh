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
# Modal profile from ~/.modal.toml (passed to `modal-runner --user`). Override
# with `MODAL_USER=yucheng bash scripts/launch_train_modal_mr.sh`.
MODAL_USER="${MODAL_USER:-heavyball}"
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
# Repo-local packages installed editable into the image (colon-separated).
export MR_EDITABLE="${MR_EDITABLE:-${REPO_ROOT}/src/training/sdar/llama_factory_sdar}"

# Paths the user script wants preserved end-to-end.
DATA_PATH_COMMON="${REPO_ROOT}/src/training/sdar/tokenized_cache/${DATASET}/seq_${SEQ_LEN}"

mr() {
  # One dispatch helper. Positional arg = app name. The rest of the line is
  # read as `VAR=VAL VAR=VAL ...` shell assignments exactly like a normal
  # inline-env invocation of the training script.
  local app_name="$1"; shift
  # MODEL_NAME is required so we can compute the per-run output_dir.
  local model_name=""
  for kv in "$@"; do
    if [[ "$kv" == MODEL_NAME=* ]]; then model_name="${kv#MODEL_NAME=}"; break; fi
  done
  if [[ -z "$model_name" ]]; then
    echo "[mr] error: MODEL_NAME=... must be set in the inline env for ${app_name}" >&2
    return 1
  fi

  # Dry-run train_sdar_mtp.sh in COMPUTE_OUTPUT_DIR_FILE mode to resolve the
  # full output_dir for this exact config. Modal-runner then uploads / pulls
  # back / mirrors only that subtree, instead of the entire `checkpoints/<MODEL_NAME>`
  # parent (which can contain many sibling runs from other hyperparams).
  local out_dir_file
  out_dir_file="$(mktemp -t mr-out-dir-XXXXXX.txt)"
  local out_path
  if ! env \
        DATASET="${DATASET}" \
        dataset="${DATASET}" \
        SEQ_LEN="${SEQ_LEN}" \
        NUM_GPUS="${NUM_GPUS}" \
        "$@" \
        OUTPUT_PATH="${REPO_ROOT}/checkpoints/${model_name}" \
        CUDA_VISIBLE_DEVICES=0 \
        COMPUTE_OUTPUT_DIR_FILE="${out_dir_file}" \
        bash "${REPO_ROOT}/scripts/train_sdar_mtp.sh" >/dev/null 2>&1
  then
    echo "[mr] error: dry-run to resolve output_dir failed for ${app_name}" >&2
    rm -f "${out_dir_file}"
    return 1
  fi
  out_path="$(cat "${out_dir_file}")"
  rm -f "${out_dir_file}"
  if [[ -z "${out_path}" ]]; then
    echo "[mr] error: dry-run produced empty output_dir for ${app_name}" >&2
    return 1
  fi
  echo "[mr] resolved OUTPUT_PATH=${out_path}"

  # shellcheck disable=SC2068  # intentional: forward caller's inline-env tokens
  env \
    DATA_PATH="${DATA_PATH_COMMON}" \
    OUTPUT_PATH="${out_path}" \
    EXPLICIT_OUTPUT_DIR="${out_path}" \
    DATASET="${DATASET}" \
    dataset="${DATASET}" \
    SEQ_LEN="${SEQ_LEN}" \
    NUM_GPUS="${NUM_GPUS}" \
    $@ \
    modal-runner run "${REPO_ROOT}/scripts/train_sdar_mtp.sh" \
      --name "${app_name}" \
      --user "${MODAL_USER}" \
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
_DEFAULTS="LR=1e-3 gt_topk=1 input_prep_mode=bd_packed freeze_backbone=True freeze_lm_head=True mtp_init_std=0.2 micro_bs=2"

# Active entry: 8b 3lyr, K=2, gt_topk=2.
mr sdar-8b-3lyr-K2-gt_top2 \
   MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr" \
   MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr \
   loss_type=kd_diff_sum reveal_mode=gt mtp_steps=2 ${_DEFAULTS} gt_topk=2

# Additional examples — uncomment to dispatch alongside the entry above.

# mr sdar-1_7b-3lyr-K2 \
#    MODEL_PATH="${REPO_ROOT}/src/models/SDAR-1_7B-Chat-b16-MTP-3lyr" \
#    MODEL_NAME=mtp_sdar-1_7b-chat-b16-3lyr \
#    loss_type=kd_diff_sum reveal_mode=gt mtp_steps=2 ${_DEFAULTS}

# mr sdar-8b-3lyr-2h \
#    MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr-2h" \
#    MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr-2h \
#    loss_type=kd_diff_sum reveal_mode=gt_multihead ${_DEFAULTS}

# mr sdar-8b-3lyr-kd_diff \
#    MODEL_PATH="${REPO_ROOT}/src/models/SDAR-8B-Chat-b16-MTP-3lyr" \
#    MODEL_NAME=mtp_sdar-8b-chat-b16-3lyr \
#    loss_type=kd_diff reveal_mode=gt ${_DEFAULTS}

echo "[launch] all jobs dispatched (running in background)"
echo "[launch] monitor:  modal-runner jobs"
echo "[launch] logs:     tail -f logs/<app_name>/*.log"
