#!/bin/bash

set -euo pipefail

# Make sure all torchrun ranks use the same Python/torch/triton.
# On Modal we run inside a pip-built image with deps in the system Python,
# so skip conda activation. Detect Modal via MODAL_TASK_ID (set by Modal in
# the container).
if [ -z "${MODAL_TASK_ID:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    # The traceback shows training is running under this env (py3.11). Adjust if you use a different one.
    conda activate llamafactory_sdar
fi
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CUR_DIR}/.." && pwd)"
source "${CUR_DIR}/utils.sh"

# Sync modeling_sdar_mtp.py (and siblings) from the canonical
# src/models/SDAR-4B-Chat-MTP/ copy into every src/models/SDAR-*-MTP*/ and
# checkpoints/**/checkpoint-*/ dir before training. LLaMA-Factory autoresume
# loads modeling code from the resumed checkpoint dir via trust_remote_code;
# if that dir holds a stale modeling file, resumes can hit subtle bugs like
# DeepSpeed's "parameter N already reduced" assertion when a forward path's
# edge count no longer matches the persisted reducer topology. Running the
# sync on every training start guarantees resumed code matches source.
if [[ -f "${REPO_ROOT}/misc/sync_modeling_sdar_mtp.sh" ]]; then
  echo "[train_sdar_mtp] syncing modeling files into src/models/ and checkpoints/"
  ( cd "${REPO_ROOT}" && bash misc/sync_modeling_sdar_mtp.sh ) || true
fi

# Case-insensitive boolean check
is_true() {
  local value="${1,,}"  # Convert to lowercase
  [[ "$value" == "true" || "$value" == "yes" || "$value" == "1" ]]
}

export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
export TORCHINDUCTOR_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/torchinductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Distributed timeouts. The actual ProcessGroup watchdog timeout is the
# `timeout` arg to torch.distributed.init_process_group, which defaults to
# 600 s for NCCL. `NCCL_TIMEOUT` does NOT override it (that env var is NCCL's
# own kernel-level timeout, not PyTorch's watchdog). DeepSpeed calls
# init_process_group before transformers can apply ddp_timeout, so
# ddp_timeout=... was effectively a dead letter — every log shows
# `Timeout(ms)=600000`.
#
# Fix: NCCL_PG_TIMEOUT_SEC (read by launcher.py) monkey-patches the default
# before the first init_process_group call, so every PG created in the run
# gets the intended timeout. Default 3600 s (1 h) — long enough to survive
# checkpoint saves, transient Modal network blips, and the occasional
# straggler batch, while still letting us notice a truly hung job.
export NCCL_PG_TIMEOUT_SEC=${NCCL_PG_TIMEOUT_SEC:-3600}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600000}  # NCCL-kernel side; kept for completeness
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

# NCCL FlightRecorder: capture the stack trace of the failed collective when
# the watchdog fires. Without this, every NCCL watchdog line in our logs says
# "Stack trace of the failed collective not found, potentially because
# FlightRecorder is disabled" — so we can't tell what was hanging.
# 2 MB ring buffer is enough for hundreds of recent collectives per rank.
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2097152}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}

# Opt-in NCCL debug output. Off by default (floods logs); set NCCL_DEBUG=WARN
# (or INFO) for one run when diagnosing a new hang pattern.
export NCCL_DEBUG=${NCCL_DEBUG:-}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}

# Opt-in deterministic training. Set DETERMINISTIC=1 to make blue/orange/etc.
# produce bit-identical head-0 weights across reveal_mode branches. Slower by
# ~10-30% depending on ops. The actual torch.use_deterministic_algorithms call
# lives in src/.../llamafactory/launcher.py (must run before any torch op).
export DETERMINISTIC=${DETERMINISTIC:-1}
if [[ "${DETERMINISTIC}" == "1" ]] || [[ "${DETERMINISTIC,,}" == "true" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    echo "[train_sdar_mtp] DETERMINISTIC=1: CUBLAS_WORKSPACE_CONFIG=:4096:8"
fi

PROJ_DIR='/mnt/weka/home/yucheng/yufeng/dflash-sdar/src/training/sdar'
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=${NUM_GPUS:-4}
# Only use get_free_gpus if CUDA_VISIBLE_DEVICES isn't already set by nohup_run.sh
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(get_free_gpus ${NUM_GPUS})
fi
echo "Using GPUs ${CUDA_VISIBLE_DEVICES} for training"

LAYERS=${LAYERS:-3}
MODEL_PATH=${MODEL_PATH:-"$(pwd)/src/models/SDAR-1_7B-Chat-b16-MTP-${LAYERS}lyr"}
MODEL_NAME=${MODEL_NAME:-"mtp_sdar-1_7b-chat-${LAYERS}lyr"}
MODEL_CACHE_NAME=${MODEL_CACHE_NAME:-$(basename "${MODEL_PATH}")}

# MODEL_PATH=${MODEL_PATH:-"$(pwd)/src/models/SDAR-1_7B-Chat-b16-MTP"}
# MODEL_NAME=${MODEL_NAME:-'mtp-sdar-1.7b-chat'}

LR=${LR:-1e-6}
lr_scheduler_type=cosine_with_min_lr
num_epochs=1
global_bs=${global_bs:-16}
micro_bs=1
grad_accum=$((global_bs / micro_bs / NUM_GPUS))
seq_len=4096
block_length=16
# dataset=eagle_chat
freeze_backbone=${freeze_backbone:-True}
freeze_lm_head=${freeze_lm_head:-True}
loss_type=${loss_type:-ce}
input_prep_mode=${input_prep_mode:-infer_matched}
kd_temperature=${kd_temperature:-1.0}
kd_alpha=${kd_alpha:-0.5}
kd_mix_weight=${kd_mix_weight:-0.5}
kd_reverse_kl=${kd_reverse_kl:-False}
mtp_init_std=${mtp_init_std:-0.02}
init_eh_proj=${init_eh_proj:-False}
mtp_proj_lr=${mtp_proj_lr:-}
reveal_mode=${reveal_mode:-gt}
gt_topk=${gt_topk:-1}
gt_random_topk=${gt_random_topk:-1}
gt_curriculum_max=${gt_curriculum_max:-16}
conf_topk=${conf_topk:-1}
conf_threshold=${conf_threshold:-0.9}
mtp_steps=${mtp_steps:-1}
loss_weight_mode=${loss_weight_mode:-uniform}

# Multi-step MTP training reuses the MTP head's weights across K forward paths,
# so in a single backward each MTP parameter has K autograd edges. DeepSpeed
# ZeRO-1/2 fire a per-parameter grad-reduce hook on each edge and assert
# "parameter N has already been reduced" on the second firing. Workarounds:
#   - z0 (pure DDP under the DS engine): one all-reduce per param at end of
#     backward, AFTER autograd sums all K edge contributions into .grad.
#     Fastest option and works because only the MTP head is trained (tiny).
#   - z3: partition-owned grad accumulation, correct but slow.
# Default to z0 when K>=2 (backbone and lm_head are typically frozen); let the
# user override explicitly (e.g., deepspeed=z3 if the trainable slice is too big
# for DDP). If the user already pinned `deepspeed`, respect it.
if [[ "${mtp_steps}" -ge 2 && -z "${deepspeed:-}" ]]; then
  deepspeed=z0
  echo "[train_sdar_mtp] mtp_steps=${mtp_steps} -> forcing deepspeed=z0 (pure DDP; ZeRO-2 incompatible with shared-weight multi-edge backward). Override with deepspeed=z3 if the trainable slice needs sharding."
fi
use_precomputed_states=${use_precomputed_states:-False}
precomputed_states_dir=${precomputed_states_dir:-}
dataset=${dataset:-ultrachat_200k}
max_samples=${MAX_SAMPLES:-}
disable_shuffling=${disable_shuffling:-False}
do_eval=${do_eval:-True}
# dataset=open_r1_math
# dataset=sftdatasetv3


# IMPORTANT: LLaMAFactory will *load* tokenized data from disk if tokenized_path exists,
# and then it will ignore cutoff_len/neat_packing/etc. To make seq_len changes effective,
# use a tokenized_path that depends on seq_len (or remove tokenized_path entirely).
TOKENIZED_PATH=${PROJ_DIR}/tokenized_cache/${dataset}/seq_${seq_len}
if [[ -n "${max_samples}" ]]; then
  TOKENIZED_PATH=${TOKENIZED_PATH}/max_samples_${max_samples}
fi

if is_true "${use_precomputed_states}" && [[ -z "${precomputed_states_dir}" ]]; then
  precomputed_states_dir="$(pwd)/precomputed_states/${MODEL_CACHE_NAME}/${dataset}/seq_${seq_len}"
fi

export TORCH_COMPILE_DISABLE=0
# export TORCHINDUCTOR_DISABLE=1

get_random_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])"
}

# Determine mtp_logit_mix_mode based on loss_type.
# ('sum' losses train A+C≈B, so logits_sum at inference is the matched mode.)
determine_mtp_logit_mix_mode() {
  if [[ "${loss_type}" == *"sum"* ]]; then
    echo "logits_sum"
  else
    echo "none"
  fi
}

RUN_ID_DIR="./wandb/run_ids"
mkdir -p "${RUN_ID_DIR}"

# Persist wandb run id by run_name so reruns resume the same W&B run.
# W&B uses WANDB_RUN_ID + WANDB_RESUME to decide whether to resume.
# SAFE_RUN_NAME=$(echo "${run_name}" | tr '/[:space:]' '__')

export WANDB_RESUME="allow"
export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'
export WANDB_PROJECT="mtp_sdar"

run_exp() {
  # Build loss suffix and output_dir
  loss_suffix="loss_${loss_type}"
  case "${loss_type}" in
    ce)          ;;
    ce_sum)      ;;
    kd)          loss_suffix+="_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    ce_kd)       loss_suffix+="_a_${kd_alpha}_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    logit_mix)   loss_suffix+="_a_${kd_alpha}" ;;
    kd_diff)     loss_suffix+="_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    kd_diff_mix) loss_suffix+="_w_${kd_mix_weight}_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    kd_diff_sum) loss_suffix+="_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    l1)          ;;
    l1_sum)      ;;
    mse)         ;;
    mse_sum)     ;;
  esac

  # Append init std to loss suffix
  loss_suffix+="_init_std_${mtp_init_std}"

  # Append structured init and proj lr if set
  if is_true "${init_eh_proj}"; then
    loss_suffix+="_ehproj"
  fi
  if [[ -n "${mtp_proj_lr}" ]]; then
    loss_suffix+="_plr_${mtp_proj_lr}"
  fi

  # Append reveal_mode to loss suffix. Each mode gets its own parameter
  # appended as "_{value}" so run_name / output_dir stay unique across
  # (mode, knob) combinations.
  loss_suffix+="_reveal_${reveal_mode}"
  case "${reveal_mode}" in
    gt)              loss_suffix+="_${gt_topk}" ;;
    gt_random)
      gt_random_suffix=$(
        sorted=$(
          printf '%s' "${gt_random_topk}" \
            | tr -d '[] ' \
            | tr ',' '\n' \
            | sed '/^$/d' \
            | sort -n
        )
        first=$(echo "$sorted" | head -1)
        last=$(echo "$sorted"  | tail -1)
        count=$(echo "$sorted" | wc -l | tr -d ' ')
        expected=$(( last - first + 1 ))
        # Compact to "first-last" iff values are exactly first,first+1,...,last
        if [[ "$count" -eq "$expected" && "$first" -eq 1 && "$count" -gt 1 ]]; then
          echo "${first}-${last}"
        else
          echo "$sorted" | paste -sd '_' -
        fi
      )
      loss_suffix+="_${gt_random_suffix}"
      ;;
    gt_curriculum)   loss_suffix+="_${gt_curriculum_max}" ;;
    conf_topk)       loss_suffix+="_${conf_topk}" ;;
    conf_threshold)  loss_suffix+="_${conf_threshold}" ;;
  esac

  # Append mtp_steps to the suffix when K >= 2 so K=1 runs keep their historical
  # paths and K>=2 runs get unique checkpoint dirs per K.
  if [[ "${mtp_steps}" -ge 2 ]]; then
    loss_suffix+="_K_${mtp_steps}_lwm_${loss_weight_mode}"
  fi

  _l1=checkpoints/${MODEL_NAME}/ds_${dataset}_seq_${seq_len}_bs_${global_bs}_ep_${num_epochs}_freezebb_${freeze_backbone}_freezelm_${freeze_lm_head}
  _l2=${loss_suffix}_prep_${input_prep_mode}_lr_${LR}_${lr_scheduler_type}
  output_dir=${_l1}/${_l2}
  output_dir=${output_dir//./_}

  run_name=${MODEL_NAME}_ds_${dataset}_seq_${seq_len}_bs_${global_bs}_ep_${num_epochs}_freezebb_${freeze_backbone}_freezelm_${freeze_lm_head}_${_l2}
  run_name=${run_name//./_}

  RUN_ID_FILE="${RUN_ID_DIR}/${run_name}"

  if [[ -s "${RUN_ID_FILE}" ]]; then
    export WANDB_RUN_ID=$(cat "${RUN_ID_FILE}")
  else
    export WANDB_RUN_ID=$(python - <<'PY'
import uuid
try:
    import wandb
    print(wandb.util.generate_id())
except Exception:
    print(uuid.uuid4().hex[:8])
PY
)
    tmpfile="${RUN_ID_FILE}.tmp"
    printf "%s" "${WANDB_RUN_ID}" > "${tmpfile}"
    mv "${tmpfile}" "${RUN_ID_FILE}"
  fi

  echo "[wandb] run_name=${run_name}"
  echo "[wandb] run_id=${WANDB_RUN_ID} (${WANDB_RESUME})"

  torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node ${NUM_GPUS} \
    --master_port $(get_random_port) \
    ${PROJ_DIR}/llama_factory_sdar/src/llamafactory/launcher.py \
    ${PROJ_DIR}/llama_factory_sdar/examples/train_full_sdar/sdar_4b/sdar_4b_math_cot_full.yaml \
    model_name_or_path=${MODEL_PATH} \
    trust_remote_code=true \
    deepspeed=${PROJ_DIR}/llama_factory_sdar/examples/deepspeed/ds_${deepspeed:-z2}_config.json \
    dataset_dir=${PROJ_DIR}/llama_factory_sdar/data \
    dataset=${dataset} \
    tokenized_path=${TOKENIZED_PATH} \
    ${max_samples:+max_samples=${max_samples}} \
    overwrite_cache=true \
    output_dir=${output_dir} \
    overwrite_output_dir=${override:-False} \
    learning_rate=${LR} \
    lr_scheduler_type=${lr_scheduler_type} \
    num_train_epochs=${num_epochs} \
    per_device_train_batch_size=${micro_bs} \
    gradient_accumulation_steps=${grad_accum} \
    cutoff_len=${seq_len} \
    block_length=${block_length} \
    gradient_checkpointing=true \
    logging_steps=${logging_steps:-10} \
    save_steps=${save_steps:-250} \
    save_total_limit=1 \
    ddp_timeout=${DDP_TIMEOUT:-180000000} \
    report_to=wandb \
    run_name=${run_name} \
    input_prep_mode=${input_prep_mode} \
    freeze_backbone=${freeze_backbone} \
    freeze_lm_head=${freeze_lm_head} \
    loss_type=${loss_type} \
    kd_temperature=${kd_temperature} \
    kd_alpha=${kd_alpha} \
    kd_mix_weight=${kd_mix_weight} \
    kd_reverse_kl=${kd_reverse_kl} \
    mtp_init_std=${mtp_init_std} \
    init_eh_proj=${init_eh_proj} \
    ${mtp_proj_lr:+mtp_proj_lr=${mtp_proj_lr}} \
    reveal_mode=${reveal_mode} \
    gt_topk=${gt_topk} \
    gt_random_topk=${gt_random_topk} \
    gt_curriculum_max=${gt_curriculum_max} \
    conf_topk=${conf_topk} \
    conf_threshold=${conf_threshold} \
    mtp_steps=${mtp_steps} \
    loss_weight_mode=${loss_weight_mode} \
    use_precomputed_states=${use_precomputed_states} \
    disable_shuffling=${disable_shuffling:-False} \
    ${precomputed_states_dir:+precomputed_states_dir=${precomputed_states_dir}}

  # Post-training evaluation
  if is_true "${do_eval}"; then
    echo "[$(date)] Training completed. Releasing training GPU claims..."

    # Release training GPU claims so other jobs can use them during eval
    if [[ -n "${NOHUP_JOB_ID:-}" ]] && [[ -n "${NOHUP_QUEUE_DIR:-}" ]]; then
      python3 "${NOHUP_QUEUE_DIR}/gpu_claim.py" release --job-id "${NOHUP_JOB_ID}" 2>/dev/null || true
      echo "[$(date)] GPU claims released."
    fi

    echo "[$(date)] Starting evaluation..."
    mtp_logit_mix_mode=$(determine_mtp_logit_mix_mode)
    export MODEL_PATH="${output_dir}"
    export MTP_LOGIT_MIX_MODE="${mtp_logit_mix_mode}"
    export MTP_VERIFY_MODE="none"
    export REUSE=false
    # When training used confidence-threshold reveal, mirror it at eval time:
    # use the dynamic low-confidence remasking strategy with the same threshold.
    if [[ "${reveal_mode}" == "conf_threshold" ]]; then
      export REMASKING_STRATEGY=low_confidence_dynamic
      export THRESHOLD=${conf_threshold}
    fi
    # When training used gt_multihead, route each sample to its matching MTP head
    # at inference (multihead mode). Use low_confidence_dynamic remasking with
    # threshold 0.9 so the target commits tokens it's confident about.
    if [[ "${reveal_mode}" == "gt_multihead" ]]; then
      export MTP_ROUTER_MODE=multihead
      export REMASKING_STRATEGY=low_confidence_dynamic
      export THRESHOLD=0.9
    fi
    NUM_GPUS=1 MTP_STEPS=1 nohup-queue run "${CUR_DIR}/eval_opencompass_hf.sh"
    NUM_GPUS=1 MTP_STEPS=2 nohup-queue run "${CUR_DIR}/eval_opencompass_hf.sh"
  fi
}

run_exp
exit $?
