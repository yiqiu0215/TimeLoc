# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.
# Original code copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import os

import nncore
import torch
from nncore.engine import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor

from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time
from training.model_loader import get_model_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Output prediction path")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Processor checkpoint path. For TimeLens-3B, set to TencentARC/TimeLens-7B.",
    )
    parser.add_argument("--min_tokens", type=int, default=None)
    parser.add_argument("--total_tokens", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--fps_max_frames", type=int, default=None)
    parser.add_argument("--use_residual_tokens", action="store_true")
    parser.add_argument("--residual_num_diffs", type=int, default=4)

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

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    args.use_residual_tokens = args.use_residual_tokens or bool(
        getattr(config, "use_residual_tokens", False)
    )
    if args.use_residual_tokens:
        args.residual_num_diffs = int(getattr(config, "residual_num_diffs", 4))
        if args.min_tokens is None:
            args.min_tokens = int(getattr(config, "minimum_tokens_per_block", 64))
        if args.total_tokens is None:
            args.total_tokens = int(
                getattr(config, "combined_visual_token_budget", 14336)
            )
        if args.fps is None:
            args.fps = float(getattr(config, "rit_sampling_fps", 1.0))
        if args.fps_max_frames is None:
            args.fps_max_frames = getattr(config, "rit_fps_max_frames", None)
        if args.fps_max_frames is None:
            max_pseudo_blocks = args.total_tokens // args.min_tokens
            args.fps_max_frames = ((max_pseudo_blocks + 1) // 2) * 2
    else:
        args.min_tokens = 16 if args.min_tokens is None else args.min_tokens
        args.total_tokens = 3584 if args.total_tokens is None else args.total_tokens
        args.fps = 2.0 if args.fps is None else args.fps
    model_cls = get_model_class(
        args.model_path, use_residual_tokens=args.use_residual_tokens
    )
    model = model_cls.from_pretrained(
        args.model_path,
        config=config,
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
        num_workers=4,
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
