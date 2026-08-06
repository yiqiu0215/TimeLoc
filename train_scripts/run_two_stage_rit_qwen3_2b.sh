#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

model_path="/path/to/Qwen3-VL-2B-Instruct"
gebplus_annotation_path="/workspace/s/lzw/datasets/GEB+/train.json"
gebplus_video_root="/workspace/s/lzw/datasets/GEB+/videos"
timelens_data_root="/workspace/s/lzw/datasets/TimeLens-100K"
fps=1
residual_num_diffs=4
time_embedding_dim=128
min_tokens=64
total_tokens=14336
fps_max_frames=""
stage1_learning_rate="1e-5"
stage1_residual_learning_rate="1e-4"
stage2_learning_rate="1e-5"
stage1_epochs=1
stage2_epochs=1
target_size=20000
global_batch_size=128
batch_per_device=1
num_devices=8
deepspeed_config="scripts/zero3.json"
seed=42
output_root="output/RIT-Qwen3VL-2B"
report_to="none"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --gebplus_annotation_path) gebplus_annotation_path="$2"; shift 2 ;;
    --gebplus_video_root) gebplus_video_root="$2"; shift 2 ;;
    --timelens_data_root) timelens_data_root="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --residual_num_diffs) residual_num_diffs="$2"; shift 2 ;;
    --time_embedding_dim) time_embedding_dim="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --stage1_learning_rate) stage1_learning_rate="$2"; shift 2 ;;
    --stage1_residual_learning_rate) stage1_residual_learning_rate="$2"; shift 2 ;;
    --stage2_learning_rate) stage2_learning_rate="$2"; shift 2 ;;
    --stage1_epochs) stage1_epochs="$2"; shift 2 ;;
    --stage2_epochs) stage2_epochs="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ ! -f "${gebplus_annotation_path}" ]]; then
  echo "GEB+ annotation file not found: ${gebplus_annotation_path}"
  exit 1
fi
if [[ ! -d "${gebplus_video_root}" ]]; then
  echo "GEB+ video root not found: ${gebplus_video_root}"
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
  echo "global_batch_size must be divisible by batch_per_device * num_devices"
  exit 1
fi

gradient_accumulation_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
  max_pseudo_blocks=$((total_tokens / min_tokens))
  max_rgb_blocks=$(((max_pseudo_blocks + 1) / 2))
  fps_max_frames=$((max_rgb_blocks * 2))
fi

run_tag="$(date +%Y%m%d-%H%M%S)"
run_root="${output_root}/${run_tag}_FPS-${fps}_TOTAL-${total_tokens}_MIN-${min_tokens}"
stage1_output="${run_root}/stage1-gebplus"
if (( target_size % 1000 == 0 )); then
  target_label="$((target_size / 1000))k"
else
  target_label="${target_size}"
fi
stage2_output="${run_root}/stage2-timelens-${target_label}"
mkdir -p "${run_root}"

echo "Run root          : ${run_root}"
echo "Stage 1 output    : ${stage1_output}"
echo "Stage 2 output    : ${stage2_output}"
echo "Stage 2 samples   : ${target_size}"
echo "RGB max frames    : ${fps_max_frames}"

deepspeed training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --bits 16 \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger_kernel True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${model_path}" \
  --model_id "rit-qwen3-vl-2b-stage1" \
  --conv_type chatml \
  --datasets gebplus \
  --gebplus_annotation_path "${gebplus_annotation_path}" \
  --gebplus_video_root "${gebplus_video_root}" \
  --use_residual_tokens True \
  --residual_num_diffs "${residual_num_diffs}" \
  --time_embedding_dim "${time_embedding_dim}" \
  --remove_unused_columns False \
  --output_dir "${stage1_output}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --min_video_len 0 \
  --max_video_len 30 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --lora_enable False \
  --learning_rate "${stage1_learning_rate}" \
  --vision_lr "${stage1_learning_rate}" \
  --merger_lr "${stage1_learning_rate}" \
  --residual_lr "${stage1_residual_learning_rate}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${stage1_epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${stage1_epochs}" \
  --save_only_model True \
  --keep_intermediate_checkpoints True \
  --dataloader_num_workers 4 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "rit-qwen3-vl-2b/stage1-${run_tag}"

if [[ ! -f "${stage1_output}/config.json" ]]; then
  echo "Stage 1 did not produce config.json; Stage 2 will not start."
  exit 1
fi
if ! compgen -G "${stage1_output}/*.safetensors" > /dev/null \
  && ! compgen -G "${stage1_output}/*.bin" > /dev/null; then
  echo "Stage 1 did not produce readable model weights; Stage 2 will not start."
  exit 1
fi

deepspeed training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --bits 16 \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger_kernel True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${stage1_output}" \
  --processor_path "${stage1_output}" \
  --model_id "rit-qwen3-vl-2b-stage2" \
  --conv_type chatml \
  --datasets gemini_refined_data \
  --timelens_data_root "${timelens_data_root}" \
  --target_size "${target_size}" \
  --use_residual_tokens True \
  --residual_num_diffs "${residual_num_diffs}" \
  --time_embedding_dim "${time_embedding_dim}" \
  --remove_unused_columns False \
  --output_dir "${stage2_output}" \
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
  --learning_rate "${stage2_learning_rate}" \
  --vision_lr "${stage2_learning_rate}" \
  --merger_lr "${stage2_learning_rate}" \
  --residual_lr "${stage2_learning_rate}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${stage2_epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${stage2_epochs}" \
  --save_only_model True \
  --keep_intermediate_checkpoints True \
  --dataloader_num_workers 4 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "rit-qwen3-vl-2b/stage2-${target_label}-${run_tag}"

echo "Two-stage RIT training completed: ${stage2_output}"
