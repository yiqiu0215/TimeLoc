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
from training.model_loader import is_coarse2refine_checkpoint
from training.modeling.modeling_coarse2refine import (
    Coarse2RefineForConditionalGeneration,
)
from training.modeling.special_tokens import register_coarse2refine_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Output prediction path")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument(
        "--model_id",
        default=None,
        help=(
            "Optional model identifier; use qwen25-vl-3b-coarse2refine for "
            "the Coarse2Refine path."
        ),
    )
    parser.add_argument(
        "--processor_path",
        default=None,
        help=(
            "Processor checkpoint path. Coarse2Refine may reuse "
            "TencentARC/TimeLens-7B; this path is not a model trigger."
        ),
    )
    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    parser.add_argument("--fps", type=int, default=2)

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

    coarse2refine_target = is_coarse2refine_checkpoint(
        args.model_path,
        model_id=args.model_id,
        processor_path=args.processor_path,
    )

    # Load model
    if coarse2refine_target:
        model = Coarse2RefineForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device,
            trust_remote_code=True,
        ).eval()
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
    args.coarse2refine_target = coarse2refine_target

    # Load processor (model-specific)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        padding_side="left",
        do_resize=False,  # For Video Processing, we do not need to resize the video frames again in the processor
        trust_remote_code=True,
    )
    if coarse2refine_target:
        token_spec = register_coarse2refine_tokens(processor.tokenizer)
        config_token_ids = tuple(model.config.time_token_ids)
        if config_token_ids != token_spec.time_token_ids:
            raise ValueError(
                "Processor and Coarse2Refine checkpoint use different time token ids."
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
        status = None
        parse_status = None

        if coarse2refine_target:
            refine_output = model.generate_coarse2refine(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
            )
            generated_ids_trimmed = [
                refine_output.generated_ids[0, refine_output.prompt_length :]
            ]
            answers = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            status = refine_output.statuses[0]
            parse_status = refine_output.parse_statuses[0]
            if status == "ok":
                timestamps = [[
                    float(refine_output.pred_start[0].item()),
                    float(refine_output.pred_end[0].item()),
                ]]
                print(f"Refined timestamps: {timestamps}")
            else:
                timestamps = [[-1, -1]]
                print(f"Coarse2Refine status={status}; returning [-1, -1].")
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
        if status is not None:
            dump[f"{video_name}>>>{query}>>>{span}"]["status"] = status
        if parse_status is not None:
            dump[f"{video_name}>>>{query}>>>{span}"]["parse_status"] = parse_status

        print(
            f"video_path: {video_path}, query: {query}, duration: {duration}, "
            f"answer: {answers}, extracted timestamps: {timestamps}"
        )

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
