"""Analyze TimeDec bin probability distributions on TimeLens-Bench.

The script reruns eval-style inference because existing prediction jsonl files
do not store TimeDec logits/probabilities.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import random
import sys

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import nncore
import torch
import torch.nn.functional as F
from nncore.engine import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.eval_dataloader import _load_time_head_weights
from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from training.model.time_dist_wrapper import attach_time_dist_head


IOU_BUCKETS = [
    ("[0,0.3)", 0.0, 0.3),
    ("[0.3,0.5)", 0.3, 0.5),
    ("[0.5,0.7)", 0.5, 0.7),
    ("[0.7,1.0]", 0.7, 1.0 + 1e-8),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DICT.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_percent",
        type=float,
        default=20.0,
        help="Percentage of the selected dataset chunk to analyze. Default: 20.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    parser.add_argument("--fps", type=int, default=2)

    parser.add_argument("--time_reg_max", type=int, default=32)
    parser.add_argument("--time_head_num_layers", type=int, default=3)
    parser.add_argument("--time_stamp_token", default="<TIME_STAMP>")
    parser.add_argument("--frame_time_token", default="<TIME_SAMPLE>")
    parser.add_argument("--time_enc_num_layers", type=int, default=3)
    parser.add_argument("--time_enc_sigma", type=float, default=1.0)
    parser.add_argument("--time_enc_layout", default="prefix", choices=["prefix", "interleave"])
    return parser.parse_args()


def safe_iou(gt_span, pred_span):
    left = max(float(gt_span[0]), float(pred_span[0]))
    right = min(float(gt_span[1]), float(pred_span[1]))
    union_left = min(float(gt_span[0]), float(pred_span[0]))
    union_right = max(float(gt_span[1]), float(pred_span[1]))
    denom = union_right - union_left
    if denom <= 0:
        return 0.0
    return max(right - left, 0.0) / denom


def norm_span(span):
    if isinstance(span[0], (list, tuple)):
        span = span[0]
    return [float(span[0]), float(span[1])]


def round_span(span, unit):
    return [round(float(t) / unit) * unit for t in span]


@torch.no_grad()
def decode_first_stamp_with_probs(
    model,
    input_ids,
    attention_mask,
    duration,
    time_token_id,
    frame_times=None,
    **vision_kwargs,
):
    reg_max = model._time_reg_max
    if getattr(model, "_has_time_enc", False):
        model._tenc_frame_times = [frame_times] if frame_times is not None else None
        model._tenc_time_gt = None
        model._tenc_duration = [float(duration)]

    forward = getattr(model, "_orig_forward", model.forward)
    outputs = forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
        **vision_kwargs,
    )

    if getattr(model, "_has_time_enc", False):
        model._tenc_frame_times = None
        model._tenc_duration = None

    hidden = outputs.hidden_states[-1]
    time_mask = input_ids[:, 1:] == time_token_id
    time_mask = torch.cat([time_mask, torch.zeros_like(time_mask[:, :1])], dim=1)
    positions = time_mask[0].nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return None

    feats = hidden[0, positions[:1]]
    logits = model.time_dec(feats).float().reshape(2, reg_max + 1)
    probs = F.softmax(logits, dim=-1)
    pred_bin = model.time_project(logits.reshape(1, -1))[0]
    pred_span = (pred_bin / reg_max * float(duration)).tolist()
    return {
        "num_time_stamp": int(positions.numel()),
        "pred_span": [float(pred_span[0]), float(pred_span[1])],
        "start_probs": probs[0].detach().cpu().tolist(),
        "end_probs": probs[1].detach().cpu().tolist(),
    }


def bucket_name(iou):
    for name, lo, hi in IOU_BUCKETS:
        if lo <= iou < hi:
            return name
    return "[0.7,1.0]" if math.isclose(iou, 1.0) else "other"


def prob_columns(reg_max):
    return [f"p_bin_{i}" for i in range(reg_max + 1)]


def write_excel(path, samples, bin_probs, bucket_stats):
    import pandas as pd

    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(samples).to_excel(writer, sheet_name="samples", index=False)
        pd.DataFrame(bin_probs).to_excel(writer, sheet_name="bin_probs", index=False)
        pd.DataFrame(bucket_stats).to_excel(writer, sheet_name="bucket_stats", index=False)


def ensure_output_deps():
    missing = []
    for module in ("pandas", "matplotlib"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if importlib.util.find_spec("openpyxl") is None and importlib.util.find_spec("xlsxwriter") is None:
        missing.append("openpyxl or xlsxwriter")
    if missing:
        raise RuntimeError(
            "Missing packages for Excel/plot output: "
            + ", ".join(missing)
            + ". Please install them in the project environment before running this script."
        )


def plot_distributions(output_dir, reg_max, bin_probs, bucket_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cols = prob_columns(reg_max)
    probs_df = pd.DataFrame(bin_probs)
    stats_df = pd.DataFrame(bucket_stats)
    x = list(range(reg_max + 1))

    valid_df = probs_df[probs_df["side"].isin(["start", "end"])] if "side" in probs_df else pd.DataFrame()
    if not valid_df.empty:
        plt.figure(figsize=(8, 4.5))
        for side in ("start", "end"):
            side_df = valid_df[valid_df["side"] == side]
            if side_df.empty:
                continue
            plt.plot(x, side_df[cols].mean(axis=0), label=side)
        plt.xlabel("bin")
        plt.ylabel("mean probability")
        plt.title("Average TimeDec bin distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avg_all_bins.png"), dpi=200)
        plt.close()

    bucket_names = [name for name, _, _ in IOU_BUCKETS]
    ncols = 2
    nrows = math.ceil(len(bucket_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False)
    for ax, name in zip(axes.flatten(), bucket_names):
        sub = stats_df[stats_df["bucket"] == name]
        for side in ("start", "end"):
            row = sub[sub["side"] == side]
            if row.empty:
                continue
            ax.plot(x, row.iloc[0][cols].astype(float).to_numpy(), label=side)
        n = int(sub["num_samples"].max()) if not sub.empty else 0
        ax.set_title(f"{name} (n={n})")
        ax.set_xlabel("bin")
        ax.set_ylabel("mean probability")
        ax.legend()
    for ax in axes.flatten()[len(bucket_names) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "avg_by_iou_bins.png"), dpi=200)
    plt.close(fig)


def build_bucket_stats(reg_max, sample_rows, prob_rows):
    import pandas as pd

    cols = prob_columns(reg_max)
    samples = pd.DataFrame(sample_rows)
    probs = pd.DataFrame(prob_rows)
    rows = []
    if "valid_stamp" not in samples:
        samples = pd.DataFrame(columns=["sample_id", "valid_stamp", "iou_bucket"])
    if "side" not in probs:
        probs = pd.DataFrame(columns=["sample_id", "side", *cols])

    for bucket, _, _ in IOU_BUCKETS:
        sample_ids = samples.loc[
            (samples["valid_stamp"]) & (samples["iou_bucket"] == bucket),
            "sample_id",
        ].tolist()
        for side in ("start", "end"):
            sub = probs[(probs["sample_id"].isin(sample_ids)) & (probs["side"] == side)]
            row = {"bucket": bucket, "side": side, "num_samples": len(sample_ids)}
            means = sub[cols].mean(axis=0) if not sub.empty else pd.Series([0.0] * len(cols), index=cols)
            row.update({c: float(means[c]) for c in cols})
            rows.append(row)

    valid_ids = samples.loc[samples["valid_stamp"], "sample_id"].tolist()
    for side in ("start", "end"):
        sub = probs[(probs["sample_id"].isin(valid_ids)) & (probs["side"] == side)]
        row = {"bucket": "all", "side": side, "num_samples": len(valid_ids)}
        means = sub[cols].mean(axis=0) if not sub.empty else pd.Series([0.0] * len(cols), index=cols)
        row.update({c: float(means[c]) for c in cols})
        rows.insert(0, row)

    return rows


def main():
    args = parse_args()
    args.enable_time_dist = True
    args.seed = set_random_seed(args.seed)
    args.processor_path = args.processor_path or args.model_path
    args.format_model_path = args.processor_path

    os.makedirs(args.output_dir, exist_ok=True)
    ensure_output_deps()
    if args.device != "auto":
        raise ValueError('Only --device auto is supported, matching eval_dataloader.py.')
    if not 0 < args.sample_percent <= 100:
        raise ValueError("--sample_percent must be in (0, 100].")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    tok = processor.tokenizer
    time_token_id = tok.convert_tokens_to_ids(args.time_stamp_token)
    frame_time_token_id = tok.convert_tokens_to_ids(args.frame_time_token)
    if time_token_id is None or time_token_id == tok.unk_token_id:
        raise ValueError(f"{args.time_stamp_token!r} not found in tokenizer.")
    if frame_time_token_id is None or frame_time_token_id == tok.unk_token_id:
        raise ValueError(f"{args.frame_time_token!r} not found in tokenizer.")

    attach_time_dist_head(
        model,
        time_token_id=time_token_id,
        reg_max=args.time_reg_max,
        num_layers=args.time_head_num_layers,
        patch_forward=False,
        has_time_enc=True,
        frame_time_token_id=frame_time_token_id,
        time_enc_sigma=args.time_enc_sigma,
        time_enc_num_layers=args.time_enc_num_layers,
    )
    _load_time_head_weights(model, args.model_path)
    model.eval()

    dataset_class = DATASET_DICT[args.dataset]
    indexed_annos = list(enumerate(dataset_class.load_annos(split=args.split)))
    indexed_annos.sort(key=lambda x: x[1]["duration"], reverse=True)
    indexed_annos = indexed_annos[args.index :: args.chunk]
    if args.sample_percent < 100:
        num_to_keep = max(1, math.ceil(len(indexed_annos) * args.sample_percent / 100.0))
        rng = random.Random(args.seed + args.index)
        indexed_annos = rng.sample(indexed_annos, num_to_keep)
        indexed_annos.sort(key=lambda x: x[1]["duration"], reverse=True)
    if args.max_samples is not None:
        indexed_annos = indexed_annos[: args.max_samples]

    dataset = GroundingDataset([anno for _, anno in indexed_annos], processor, args)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    sample_rows = []
    prob_rows = []
    cols = prob_columns(args.time_reg_max)
    unit = getattr(dataset_class, "UNIT", 1.0)

    for local_idx, data in enumerate(nncore.ProgressBar(data_loader)):
        original_idx = indexed_annos[local_idx][0]
        sample_id = f"{args.dataset}-{original_idx}"
        inputs = data["inputs"].to("cuda", non_blocking=True)
        anno = data["anno"]
        duration = float(anno["duration"])
        gt_span = norm_span(anno["span"])
        frame_times = data.get("frame_times")

        if getattr(model, "_has_time_enc", False):
            model._tenc_frame_times = [frame_times] if frame_times is not None else None
            model._tenc_time_gt = None
            model._tenc_duration = [duration]

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=args.max_new_tokens,
        )

        if getattr(model, "_has_time_enc", False):
            model._tenc_frame_times = None
            model._tenc_duration = None

        vision_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        decoded = decode_first_stamp_with_probs(
            model,
            input_ids=output_ids,
            attention_mask=torch.ones_like(output_ids),
            duration=duration,
            time_token_id=time_token_id,
            frame_times=frame_times,
            **vision_kwargs,
        )

        valid = decoded is not None
        pred_span = [-1.0, -1.0]
        score = 0.0
        bucket = "invalid"
        num_time_stamp = 0
        if valid:
            pred_span = round_span(decoded["pred_span"], unit)
            score = safe_iou(gt_span, pred_span)
            bucket = bucket_name(score)
            num_time_stamp = decoded["num_time_stamp"]
            for side, probs in (
                ("start", decoded["start_probs"]),
                ("end", decoded["end_probs"]),
            ):
                row = {"sample_id": sample_id, "side": side}
                row.update({c: float(p) for c, p in zip(cols, probs)})
                prob_rows.append(row)

        sample_rows.append(
            {
                "sample_id": sample_id,
                "video_name": os.path.basename(anno["video_path"]),
                "query": anno["query"],
                "duration": duration,
                "gt_span": str(gt_span),
                "pred_span": str(pred_span),
                "iou": float(score),
                "iou_bucket": bucket,
                "valid_stamp": bool(valid),
                "num_time_stamp": int(num_time_stamp),
            }
        )

    bucket_rows = build_bucket_stats(args.time_reg_max, sample_rows, prob_rows)
    prefix = f"{args.dataset}_{args.split}_chunk{args.index}of{args.chunk}"
    excel_path = os.path.join(args.output_dir, f"{prefix}_time_bin_analysis.xlsx")
    write_excel(excel_path, sample_rows, prob_rows, bucket_rows)
    plot_distributions(args.output_dir, args.time_reg_max, prob_rows, bucket_rows)

    print(f"Saved Excel: {excel_path}")
    print(f"Saved plots: {args.output_dir}/avg_all_bins.png, {args.output_dir}/avg_by_iou_bins.png")


if __name__ == "__main__":
    main()
