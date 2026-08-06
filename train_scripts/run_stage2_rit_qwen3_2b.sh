#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

stage1_model_path=""
timelens_data_root="/workspace/s/lzw/datasets/TimeLens-100K"
output_dir=""
target_size=20000
learning_rate="1e-5"
epochs=1
global_batch_size=128
batch_per_device=1
num_devices=8
deepspeed_config="scripts/zero3.json"
seed=42
report_to="none"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1_model_path) stage1_model_path="$2"; shift 2 ;;
    --timelens_data_root) timelens_data_root="$2"; shift 2 ;;
    --output_dir) output_dir="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    --learning_rate) learning_rate="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${stage1_model_path}" ]]; then
  echo "--stage1_model_path is required."
  exit 1
fi
if [[ ! -f "${stage1_model_path}/config.json" ]]; then
  echo "Stage 1 config not found: ${stage1_model_path}/config.json"
  exit 1
fi
if ! compgen -G "${stage1_model_path}/*.safetensors" > /dev/null \
  && ! compgen -G "${stage1_model_path}/*.bin" > /dev/null; then
  echo "Stage 1 model weights not found: ${stage1_model_path}"
  exit 1
fi
if [[ ! -f "${timelens_data_root}/timelens-100k.jsonl" ]]; then
  echo "TimeLens annotation file not found: ${timelens_data_root}/timelens-100k.jsonl"
  exit 1
fi
if [[ ! -d "${timelens_data_root}/videos" ]]; then
  echo "TimeLens video root not found: ${timelens_data_root}/videos"
  exit 1
fi
if [[ ! -f "${deepspeed_config}" ]]; then
  echo "DeepSpeed config not found: ${deepspeed_config}"
  exit 1
fi
if (( target_size <= 0 )); then
  echo "target_size must be positive."
  exit 1
fi
if (( global_batch_size % (batch_per_device * num_devices) != 0 )); then
  echo "global_batch_size must be divisible by batch_per_device * num_devices."
  exit 1
fi

rit_values="$(python -c '
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    config = json.load(file)

expected_version = "shared_rgb_patch_accumulate_v2"
version = config.get("rit_architecture_version")
if version != expected_version:
    raise SystemExit(
        f"Incompatible RIT architecture: {version!r}; expected {expected_version!r}."
    )

keys = (
    "residual_num_diffs",
    "time_embedding_dim",
    "minimum_tokens_per_block",
    "combined_visual_token_budget",
    "rit_sampling_fps",
    "rit_fps_max_frames",
)
missing = [key for key in keys if key not in config]
if missing:
    raise SystemExit("Missing Stage 1 RIT config fields: " + ", ".join(missing))

print("|".join(str(config[key]) for key in keys))
' "${stage1_model_path}/config.json")"

IFS='|' read -r \
  residual_num_diffs \
  time_embedding_dim \
  min_tokens \
  total_tokens \
  fps \
  fps_max_frames <<< "${rit_values}"

if [[ "${fps_max_frames}" == "None" ]]; then
  max_pseudo_blocks=$((total_tokens / min_tokens))
  max_rgb_blocks=$(((max_pseudo_blocks + 1) / 2))
  fps_max_frames=$((max_rgb_blocks * 2))
fi

gradient_accumulation_steps=$((
  global_batch_size / (batch_per_device * num_devices)
))

if [[ -z "${output_dir}" ]]; then
  if (( target_size % 1000 == 0 )); then
    target_label="$((target_size / 1000))k"
  else
    target_label="${target_size}"
  fi
  output_dir="$(dirname "${stage1_model_path}")/stage2-timelens-${target_label}"
fi

mkdir -p "${output_dir}"

echo "Stage 1 checkpoint : ${stage1_model_path}"
echo "Stage 2 output     : ${output_dir}"
echo "Target size        : ${target_size}"
echo "FPS                : ${fps}"
echo "RGB max frames     : ${fps_max_frames}"
echo "Total tokens       : ${total_tokens}"
echo "Min block tokens   : ${min_tokens}"

deepspeed training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --bits 16 \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger_kernel True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${stage1_model_path}" \
  --processor_path "${stage1_model_path}" \
  --model_id "rit-qwen3-vl-2b-stage2-${target_size}" \
  --conv_type chatml \
  --datasets gemini_refined_data \
  --timelens_data_root "${timelens_data_root}" \
  --target_size "${target_size}" \
  --use_residual_tokens True \
  --residual_num_diffs "${residual_num_diffs}" \
  --time_embedding_dim "${time_embedding_dim}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --lora_enable False \
  --learning_rate "${learning_rate}" \
  --vision_lr "${learning_rate}" \
  --merger_lr "${learning_rate}" \
  --residual_lr "${learning_rate}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${epochs}" \
  --save_only_model True \
  --keep_intermediate_checkpoints True \
  --dataloader_num_workers 4 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "rit-qwen3-vl-2b/stage2-${target_size}"

echo "Stage 2 RIT training completed: ${output_dir}"
