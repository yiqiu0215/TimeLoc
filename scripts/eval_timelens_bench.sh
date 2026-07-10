#!/bin/bash

set -e

# Cleanup handler to terminate all child processes on interrupt
cleanup() { pkill -P $$ 2>/dev/null || true; exit 130; }
trap cleanup SIGINT SIGTERM

export PYTHONPATH="./:$PYTHONPATH"

#---------------------------- Datasets ----------------------------#
if [[ -n "$datasets" ]]; then
    # Allow override of datasets from environment variable
    IFS=',' read -ra datasets <<< "$datasets"
else
    # Default datasets
    datasets=(
        "charades-timelens"
        "activitynet-timelens"
        "qvhighlights-timelens"
    )
fi

echo -e "\e[1;36mEvaluating datasets:\e[0m ${datasets[*]}"

#---------------------------- Model Path ----------------------------#
# Use model path from environment variable or default
model_path=${model_path:-"TencentARC/TimeLens-8B"}
processor_path=${processor_path:-""}

#---------------------------- Configuration ----------------------------#
min_tokens=${min_tokens:-64}
total_tokens=${total_tokens:-14336}
FPS=${FPS:-2}
lact_enable=${lact_enable:-False}
num_lact_heads=${num_lact_heads:-4}
lact_chunk_size=${lact_chunk_size:-2648}
window_size=${window_size:-2648}
use_conv_layer=${use_conv_layer:-True}
use_momentum=${use_momentum:-True}
use_muon=${use_muon:-True}
learnable_ttt_scale=${learnable_ttt_scale:-True}
w0_w2_low_rank=${w0_w2_low_rank:-0}
use_fused_kernel=${use_fused_kernel:-False}
lact_layers=${lact_layers:-"0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26"}

# ----------------- Save Path -----------------#
# Prediction Save Path with default or env variable
pred_path=${pred_path:-"./logs"}

# Derive a tag from the model path for run naming
model_tag=$(basename "${model_path%/}")

# Create save path with model_tag and timestamp
pred_path="${pred_path}/${model_tag}_$(date +%Y%m%d_%H%M%S)"

# --------------------- GPU Configuration -----------------#
# If CUDA_VISIBLE_DEVICES is not set, ALL available GPUs will be used
IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
echo -e "\e[1;36mUsing GPUs:\e[0m ${GPULIST[*]}"
CHUNKS=${#GPULIST[@]}

# ----------------- Start Evaluation Loop -----------------#
for dataset in "${datasets[@]}"; do
    echo -e "\n\e[1;33m========================================\e[0m"
    echo -e "\e[1;33mEvaluating Dataset: $dataset\e[0m"
    echo -e "\e[1;33m========================================\e[0m"

    # Set prediction path
    current_pred_path="${pred_path}/${dataset}"
    echo -e "\e[1;32mOutput path:\e[0m $current_pred_path"

    # Run inference for current dataset
    pids=()
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/eval_dataloader.py \
            --dataset $dataset \
            --pred_path $current_pred_path \
            --model_path $model_path \
            --processor_path "$processor_path" \
            --min_tokens $min_tokens \
            --total_tokens $total_tokens \
            --fps $FPS \
            --lact_enable "$lact_enable" \
            --num_lact_heads "$num_lact_heads" \
            --lact_chunk_size "$lact_chunk_size" \
            --window_size "$window_size" \
            --use_conv_layer "$use_conv_layer" \
            --use_momentum "$use_momentum" \
            --use_muon "$use_muon" \
            --learnable_ttt_scale "$learnable_ttt_scale" \
            --w0_w2_low_rank "$w0_w2_low_rank" \
            --use_fused_kernel "$use_fused_kernel" \
            --lact_layers "$lact_layers" \
            --chunk $CHUNKS \
            --index $IDX &
        pids+=($!)
    done

    eval_status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            eval_status=1
        fi
    done
    if [[ "$eval_status" -ne 0 ]]; then
        echo -e "\e[1;31mEvaluation failed for $dataset. Skipping aggregation and metrics.\e[0m"
        exit "$eval_status"
    fi

    # Aggregate results
    shard_files=("${current_pred_path}"_*.jsonl)
    if [[ ! -e "${shard_files[0]}" ]]; then
        echo -e "\e[1;31mNo shard jsonl files found for $dataset: ${current_pred_path}_*.jsonl\e[0m"
        exit 1
    fi
    cat "${shard_files[@]}" > "${current_pred_path}.jsonl" && rm -f "${shard_files[@]}"

    # Compute metrics
    echo -e "\e[1;32mComputing metrics for $dataset\e[0m"
    metric_result=$(python evaluation/compute_metrics.py -f "${current_pred_path}.jsonl")

    echo -e "\e[1;32mCompleted evaluation for $dataset\e[0m"
    echo "$metric_result"
    echo ""
done
