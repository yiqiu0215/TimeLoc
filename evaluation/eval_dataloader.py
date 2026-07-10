# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.
# Original code copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import os

import nncore
import torch
from nncore.engine import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time
from training.lact.config import DEFAULT_LACT_LAYERS, LaCTConfig, load_lact_config
from training.lact.generation import generate_with_timelens_lact
from training.lact.loader import load_qwen25_lact_model


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Output prediction path")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Processor checkpoint path. For TimeLens-3B, set to TencentARC/TimeLens-7B.",
    )
    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--lact_enable", type=str2bool, default=False)
    parser.add_argument("--num_lact_heads", type=int, default=4)
    parser.add_argument("--lact_chunk_size", type=int, default=2648)
    parser.add_argument("--window_size", type=int, default=2648)
    parser.add_argument("--use_conv_layer", type=str2bool, default=True)
    parser.add_argument("--use_momentum", type=str2bool, default=True)
    parser.add_argument("--use_muon", type=str2bool, default=True)
    parser.add_argument("--learnable_ttt_scale", type=str2bool, default=True)
    parser.add_argument("--w0_w2_low_rank", type=int, default=0)
    parser.add_argument("--use_fused_kernel", type=str2bool, default=False)
    parser.add_argument("--lact_layers", default=DEFAULT_LACT_LAYERS)

    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Number of chunks to split the dataset for distributed evaluation. Default is 1.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.seed = set_random_seed(args.seed)
    print(f"Setting random seed to {args.seed}")

    pred_path = f"{args.pred_path}_{args.index}.jsonl"

    print(
        f"Dataset: {args.dataset}({args.split}) | Chunk: {args.chunk} | "
        f"Index: {args.index} | Output Path: {pred_path}"
    )

    assert args.device == "auto", (
        'Device should be set to "auto" for multi-GPU evaluation.'
    )

    if args.lact_enable:
        saved_lact_config = load_lact_config(args.model_path)
        if saved_lact_config is not None:
            lact_config = LaCTConfig.from_dict(saved_lact_config)
            print(
                "Using LaCT config from checkpoint: "
                f"{os.path.join(args.model_path, 'lact_config.json')}"
            )
        else:
            lact_config = LaCTConfig.from_args(args)
            print(
                "Warning: lact_config.json not found; using explicit CLI LaCT args."
            )
        model = load_qwen25_lact_model(
            args.model_path,
            lact_config=lact_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device,
        ).eval()

    processor_source = args.processor_path or args.model_path
    args.processor_path = processor_source
    args.format_model_path = processor_source

    # Load processor (model-specific)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        padding_side="left",
        do_resize=False,  # For Video Processing, we do not need to resize the video frames again in the processor
        trust_remote_code=True,
    )
    # Load dataset
    dataset_class = DATASET_DICT[args.dataset]
    annos = dataset_class.load_annos(split=args.split)

    # Sort by video length in descending order
    # 1. balance the video length for each GPU
    # 2. long videos are more likely to cause OOM, so we put them first
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]

    dataset = GroundingDataset(annos, processor, args)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    dumps = []
    for data in nncore.ProgressBar(data_loader):
        inputs = data["inputs"].to("cuda", non_blocking=True)
        anno = data["anno"]

        video_path = anno["video_path"]
        query = anno["query"]
        duration = anno["duration"]
        span = anno["span"]  # ground truth time span

        if args.lact_enable:
            output_ids = generate_with_timelens_lact(
                model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                pixel_values=inputs.get("pixel_values", None),
                pixel_values_videos=inputs.get("pixel_values_videos", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                second_per_grid_ts=inputs.get("second_per_grid_ts", None),
                max_new_tokens=512,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        else:
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=512,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        answers = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answers = answers[0]

        # Parse the answer
        timestamps = extract_time(answers)
        if len(timestamps) != 0:
            print(f"Extracted timestamps: {timestamps}")
        else:
            print("No timestamps extracted, answer might be invalid. Answer:", answers)
            timestamps = [[duration + 10, duration + 20]]

        # Round timestamps to units
        unit = getattr(dataset_class, "UNIT", 1.0)
        timestamps = [
            [
                round(start / unit) * unit,
                round(end / unit) * unit,
            ]
            for start, end in timestamps
        ]

        # Save the inference results
        video_name = os.path.basename(video_path)
        if type(span[0]) is list or type(span[0]) is tuple:
            span = span[0]

        dump = {
            f"{video_name}>>>{query}>>>{span}": {
                "timestamps": timestamps,  # the extracted time span prediction from the model
                "answers": answers,  # the full answer from the model
                "duration": duration,  # save the video duration
            }
        }

        print(
            f"video_path: {video_path}, query: {query}, duration: {duration}, "
            f"answer: {answers}, extracted timestamps: {timestamps}"
        )

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
