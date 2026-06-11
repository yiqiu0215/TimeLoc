#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

model_path="/root/autodl-tmp/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
processor_path="TencentARC/TimeLens-7B"
datasets="gemini_refined_data"
model_id="timelens-3b"
min_tokens=64
total_tokens=8192
fps=2
fps_max_frames=224
seed=42

global_batch_size=128
batch_per_device=1
num_devices=2
epochs=1
target_size=20000
deepspeed_config="scripts/zero3.json"
output_root="/root/autodl-tmp/output/TimeLens-TTT-3B/sft"
report_to="none"

lact_enable=True
num_lact_heads=4
lact_chunk_size=2648
window_size=2648
use_conv_layer=True
use_momentum=True
use_muon=True
learnable_ttt_scale=True
w0_w2_low_rank=0
use_fused_kernel=False
lact_lr=1e-5
lact_layers="0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26/28/29/30/32/33/34"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --processor_path) processor_path="$2"; shift 2 ;;
    --datasets) datasets="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    --lact_enable) lact_enable="$2"; shift 2 ;;
    --num_lact_heads) num_lact_heads="$2"; shift 2 ;;
    --lact_chunk_size) lact_chunk_size="$2"; shift 2 ;;
    --window_size) window_size="$2"; shift 2 ;;
    --use_conv_layer) use_conv_layer="$2"; shift 2 ;;
    --use_momentum) use_momentum="$2"; shift 2 ;;
    --use_muon) use_muon="$2"; shift 2 ;;
    --learnable_ttt_scale) learnable_ttt_scale="$2"; shift 2 ;;
    --w0_w2_low_rank) w0_w2_low_rank="$2"; shift 2 ;;
    --use_fused_kernel) use_fused_kernel="$2"; shift 2 ;;
    --lact_lr) lact_lr="$2"; shift 2 ;;
    --lact_layers) lact_layers="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_tag="$(date +%Y%m%d-%H%M)"
run_name="sft-lact-${run_tag}_MAXFRAMES-${fps_max_frames}_FPS-${fps}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}_CHUNK-${lact_chunk_size}_WINDOW-${window_size}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

deepspeed training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${model_path}" \
  --processor_path "${processor_path}" \
  --model_id "${model_id}" \
  --conv_type "chatml" \
  --datasets "${datasets}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --target_size "${target_size}" \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --lact_enable "${lact_enable}" \
  --num_lact_heads "${num_lact_heads}" \
  --lact_chunk_size "${lact_chunk_size}" \
  --window_size "${window_size}" \
  --use_conv_layer "${use_conv_layer}" \
  --use_momentum "${use_momentum}" \
  --use_muon "${use_muon}" \
  --learnable_ttt_scale "${learnable_ttt_scale}" \
  --w0_w2_low_rank "${w0_w2_low_rank}" \
  --use_fused_kernel "${use_fused_kernel}" \
  --lact_lr "${lact_lr}" \
  --lact_layers "${lact_layers}" \
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
