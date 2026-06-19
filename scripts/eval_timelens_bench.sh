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

# --------------- DisTime time modeling (TimeDec + TimeEnc) ----------#
# Set enable_time_dist=1 (or true) to decode timestamps from the time head and
# inject continuous frame-time embeddings (<FRAME_TIME>) instead of text.
# Requires processor_path pointing to the trained checkpoint (resized vocab) and
# a standard Qwen2.5-VL processor base. Default off => original behavior.
enable_time_dist=${enable_time_dist:-""}
time_reg_max=${time_reg_max:-32}
time_head_num_layers=${time_head_num_layers:-3}
time_stamp_token=${time_stamp_token:-"<TIME_STAMP>"}
frame_time_token=${frame_time_token:-"<FRAME_TIME>"}
time_enc_num_layers=${time_enc_num_layers:-3}
time_enc_sigma=${time_enc_sigma:-1.0}

time_head_args=()
case "${enable_time_dist,,}" in
    1|true|yes|on)
        time_head_args=(
            --enable_time_dist
            --time_reg_max "$time_reg_max"
            --time_head_num_layers "$time_head_num_layers"
            --time_stamp_token "$time_stamp_token"
            --frame_time_token "$frame_time_token"
            --time_enc_num_layers "$time_enc_num_layers"
            --time_enc_sigma "$time_enc_sigma"
        )
        echo -e "\e[1;36mDisTime (TimeDec+TimeEnc):\e[0m ENABLED (reg_max=$time_reg_max, layers=$time_head_num_layers, sigma=$time_enc_sigma)"
        ;;
esac

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
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/eval_dataloader.py \
            --dataset $dataset \
            --pred_path $current_pred_path \
            --model_path $model_path \
            --processor_path "$processor_path" \
            --min_tokens $min_tokens \
            --total_tokens $total_tokens \
            --fps $FPS \
            --chunk $CHUNKS \
            --index $IDX \
            "${time_head_args[@]}" &
    done

    wait

    # Aggregate results
    cat "${current_pred_path}"_*.jsonl > "${current_pred_path}.jsonl" && rm -f "${current_pred_path}"_*.jsonl

    # Compute metrics
    echo -e "\e[1;32mComputing metrics for $dataset\e[0m"
    metric_result=$(python evaluation/compute_metrics.py -f "${current_pred_path}.jsonl")

    echo -e "\e[1;32mCompleted evaluation for $dataset\e[0m"
    echo "$metric_result"
    echo ""
done
