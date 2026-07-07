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
from training.model.time_dist_wrapper import attach_time_dist_head, decode_time_spans


def _load_time_head_weights(model, model_path):
    """Load time_dec/time_project weights saved into the main checkpoint shards.

    The head is attached as submodules of the base model, so its parameters are
    serialized with keys ``time_dec.*`` / ``time_project.*`` in the checkpoint.
    ``from_pretrained`` on the base class skips these keys, so we load them here.
    """
    import glob

    prefixes = ("time_dec.", "time_project.", "time_enc.")
    state = {}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        for f in safetensor_files:
            shard = load_file(f)
            for k, v in shard.items():
                if k.startswith(prefixes):
                    state[k] = v
    else:
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
        for f in bin_files:
            shard = torch.load(f, map_location="cpu")
            for k, v in shard.items():
                if k.startswith(prefixes):
                    state[k] = v

    if not state:
        raise FileNotFoundError(
            f"No time head weights (time_dec.*/time_project.*/time_enc.*) found in "
            f"{model_path}. Was the checkpoint trained with enable_time_dist=True?"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    head_missing = [k for k in missing if k.startswith(prefixes)]
    if head_missing:
        raise RuntimeError(f"Missing time head weights after load: {head_missing}")
    print(f"Loaded time head weights: {sorted(state.keys())}")


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
    parser.add_argument(
        "--enable_time_dist",
        action="store_true",
        help="DisTime TimeDec+TimeEnc: decode timestamps from the time head and "
        "inject continuous frame-time embeddings (<TIME_SAMPLE>) instead of text.",
    )
    parser.add_argument("--time_reg_max", type=int, default=32)
    parser.add_argument("--time_head_num_layers", type=int, default=3)
    parser.add_argument("--time_stamp_token", default="<TIME_STAMP>")
    parser.add_argument("--frame_time_token", default="<TIME_SAMPLE>")
    parser.add_argument("--time_enc_num_layers", type=int, default=3)
    parser.add_argument("--time_enc_sigma", type=float, default=1.0)
    parser.add_argument(
        "--time_enc_layout",
        default="prefix",
        choices=["prefix", "interleave"],
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

    # Load model
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

    time_token_id = None
    if args.enable_time_dist:
        tok = processor.tokenizer
        time_token_id = tok.convert_tokens_to_ids(args.time_stamp_token)
        frame_time_token_id = tok.convert_tokens_to_ids(args.frame_time_token)
        if time_token_id is None or time_token_id == tok.unk_token_id:
            raise ValueError(
                f"{args.time_stamp_token!r} not found in tokenizer; ensure the "
                "processor saved with the trained checkpoint is used."
            )
        if frame_time_token_id is None or frame_time_token_id == tok.unk_token_id:
            raise ValueError(
                f"{args.frame_time_token!r} not found in tokenizer; ensure the "
                "processor saved with the trained checkpoint is used."
            )
        attach_time_dist_head(
            model,
            time_token_id=time_token_id,
            reg_max=args.time_reg_max,
            num_layers=args.time_head_num_layers,
            patch_forward=False,  # eval: keep native forward so generate's
            # signature introspection still builds position_ids correctly.
            has_time_enc=True,
            frame_time_token_id=frame_time_token_id,
            time_enc_sigma=args.time_enc_sigma,
            time_enc_num_layers=args.time_enc_num_layers,
        )
        _load_time_head_weights(model, args.model_path)
        model.eval()

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
        frame_times = data.get("frame_times")

        # TimeEnc: stash frame times so the embedding hook injects them during
        # the prefill of generate (consistent with training).
        if args.enable_time_dist and getattr(model, "_has_time_enc", False):
            model._tenc_frame_times = [frame_times] if frame_times is not None else None
            model._tenc_time_gt = None
            model._tenc_duration = [float(duration)]

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=512,
        )

        if args.enable_time_dist and getattr(model, "_has_time_enc", False):
            model._tenc_frame_times = None
            model._tenc_duration = None

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
        if args.enable_time_dist:
            # Decode timestamps from the time head at <TIME_STAMP> positions,
            # over the full generated sequence (prompt + answer). frame_times are
            # re-injected via the embedding hook during this forward.
            vision_kwargs = {
                k: v
                for k, v in inputs.items()
                if k not in ("input_ids", "attention_mask")
            }
            full_attention_mask = torch.ones_like(output_ids)
            timestamps = decode_time_spans(
                model,
                input_ids=output_ids,
                attention_mask=full_attention_mask,
                duration=duration,
                time_token_id=time_token_id,
                frame_times=frame_times,
                **vision_kwargs,
            )
            if len(timestamps) != 0:
                print(f"Time-head timestamps: {timestamps}")
            else:
                print(
                    "No <TIME_STAMP> generated, answer might be invalid. Answer:",
                    answers,
                )
                # Match DisTime: invalid prediction -> [-1, -1] (IoU = 0).
                timestamps = [[-1, -1]]
        else:
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
