#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

model_path="/path/to/Qwen3-VL-2B-Instruct"
annotation_path="/workspace/s/lzw/datasets/GEB+/train.json"
video_root="/workspace/s/lzw/datasets/GEB+/videos"
model_id="qwen3-vl-2b-gebplus"
min_tokens=64
total_tokens=14336
fps=4
fps_max_frames=""
seed=42

global_batch_size=128
batch_per_device=1
num_devices=8
epochs=1
deepspeed_config="scripts/zero3.json"
output_root="output/Qwen3-VL-2B-GEBPlus/sft"
report_to="none"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --annotation_path) annotation_path="$2"; shift 2 ;;
    --video_root) video_root="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if (( global_batch_size % (batch_per_device * num_devices) != 0 )); then
  echo "global_batch_size must be divisible by batch_per_device * num_devices"
  exit 1
fi
grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi

run_tag="$(date +%Y%m%d-%H%M)"
run_name="sft-${run_tag}_FPS-${fps}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"
echo "Annotation      : ${annotation_path}"
echo "Video root     : ${video_root}"
echo "Output directory: ${output_dir}"

deepspeed training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger_kernel True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${model_path}" \
  --model_id "${model_id}" \
  --conv_type "chatml" \
  --datasets "gebplus" \
  --gebplus_annotation_path "${annotation_path}" \
  --gebplus_video_root "${video_root}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --min_video_len 0 \
  --max_video_len 30 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --learning_rate 1e-5 \
  --merger_lr 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${epochs}" \
  --dataloader_num_workers 4 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${model_id}-sft/${run_name}" \
  --logging_dir wandb \
  --save_only_model True
