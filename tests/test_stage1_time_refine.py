import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.data.collator import HybridDataCollator
from training.data.grounding import GroundingDataset
from training.data.time_refine import (
    build_frame_labels,
    build_time_refine_prompt_parts,
    build_vtg_target,
    quantize_time_bins,
)
from training.modeling.special_tokens import (
    is_qwen25_coarse2refine,
    register_coarse2refine_tokens,
)
from training.modeling.configuration_coarse2refine import Coarse2RefineConfig
from training.modeling.modeling_coarse2refine import (
    Coarse2RefineForConditionalGeneration,
)
from training.modeling.special_tokens import RegisteredCoarse2RefineTokens
from training.modeling.candidate_parser import (
    build_candidate_windows,
    parse_time_refine_sequence,
)
from training.modeling.losses import smooth_l1_boundary_loss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from training.modeling.time_token_packer import TimeTokenPacker


class FakeTokenizer:
    pad_token_id = 0
    model_max_length = 128

    def __init__(self):
        self._ids = {"<pad>": self.pad_token_id}

    def add_special_tokens(self, payload):
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self._ids:
                self._ids[token] = max(self._ids.values()) + 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._ids[token]

    def __call__(self, token, add_special_tokens=False):
        return {"input_ids": [self._ids[token]]}


class Stage1TimeRefineTest(unittest.TestCase):
    def test_metadata_prompt_target_and_token_registration(self):
        tokenizer = FakeTokenizer()
        tokens = register_coarse2refine_tokens(tokenizer)
        self.assertEqual(len(tokens.time_token_ids), 301)
        self.assertEqual(
            tokens.bin_for_token_id(tokens.token_id_for_bin(217)),
            217,
        )
        self.assertTrue(
            is_qwen25_coarse2refine(
                "qwen25-vl-3b-coarse2refine",
                "/models/Qwen2.5-VL-3B-Instruct",
                "TencentARC/TimeLens-7B",
            )
        )
        self.assertFalse(
            is_qwen25_coarse2refine(
                "timelens-3b",
                "/models/Qwen2.5-VL-3B-Instruct",
                "TencentARC/TimeLens-7B",
            )
        )
        self.assertEqual(Coarse2RefineConfig.model_type, "coarse2refine")

        timestamps = torch.tensor([0.0, 3.0, 6.0])
        bins = quantize_time_bins(timestamps, duration=6.0)
        labels = build_frame_labels(
            timestamps,
            duration=6.0,
            gt_start=2.9,
            gt_end=3.1,
        ).labels
        self.assertEqual(bins.tolist(), [0, 150, 300])
        self.assertEqual(labels.tolist(), [0, 1, 0])
        target = build_vtg_target(labels, bins)
        self.assertEqual(
            target,
            "<vtg><fg><time_150><fg></vtg>",
        )
        self.assertEqual(
            build_vtg_target(torch.zeros(3, dtype=torch.long), bins),
            "<vtg><fg><fg></vtg>",
        )

        prefix, suffix = build_time_refine_prompt_parts("a person enters")
        self.assertIn("Event query: a person enters", prefix)
        self.assertIn("'a person enters'", suffix)
        self.assertIn("Return only the timestamp tokens", suffix)
        self.assertIn("<vtg><fg><fg></vtg>", suffix)

    def test_collator_pads_frame_metadata(self):
        tokenizer = FakeTokenizer()
        collator = HybridDataCollator(tokenizer)
        batch = [
            {
                "input_ids": torch.tensor([11, 12]),
                "labels": torch.tensor([-100, 12]),
                "frame_bin_ids": torch.tensor([0, 100]),
                "frame_timestamps": torch.tensor([0.0, 2.0]),
                "frame_labels": torch.tensor([0, 1]),
                "frame_valid_mask": torch.tensor([True, True]),
                "gt_start": torch.tensor([1.0]),
                "gt_end": torch.tensor([3.0]),
                "duration": torch.tensor([5.0]),
            },
            {
                "input_ids": torch.tensor([13]),
                "labels": torch.tensor([-100]),
                "frame_bin_ids": torch.tensor([20]),
                "frame_timestamps": torch.tensor([0.5]),
                "frame_labels": torch.tensor([1]),
                "frame_valid_mask": torch.tensor([True]),
                "gt_start": torch.tensor([0.5]),
                "gt_end": torch.tensor([1.5]),
                "duration": torch.tensor([2.0]),
            },
        ]
        output = collator(batch)
        self.assertEqual(tuple(output["frame_bin_ids"].shape), (2, 2))
        self.assertEqual(output["frame_valid_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(tuple(output["gt_start"].shape), (2, 1))
        self.assertEqual(tuple(output["attention_mask"].shape), (2, 2))

    def test_time_token_packer_tracks_blocks_and_time_tokens(self):
        embedding = torch.nn.Embedding(512, 4)
        packer = TimeTokenPacker(
            embedding_layer=embedding,
            vision_start_token_id=1,
            vision_end_token_id=3,
            image_token_id=4,
            video_token_id=2,
            time_token_ids=tuple(range(100, 401)),
            spatial_merge_size=2,
            pad_token_id=0,
        )
        input_ids = torch.tensor(
            [
                [10, 1, 2, 2, 3, 11, 0],
                [0, 0, 10, 1, 2, 3, 11],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1]],
            dtype=torch.long,
        )
        output = packer(
            visual_embeddings=torch.randn(3, 4),
            video_grid_thw=torch.tensor([[2, 2, 2], [1, 2, 2]]),
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_bin_ids=torch.tensor([[10, 20], [30, 0]]),
            frame_timestamps=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            frame_valid_mask=torch.tensor([[True, True], [True, False]]),
        )
        self.assertEqual(tuple(output.packed_inputs_embeds.shape), (2, 10, 4))
        self.assertEqual(output.visual_block_token_ranges, [[(2, 3), (6, 7)], [(2, 3)]])
        self.assertEqual(output.time_token_positions, [[4, 8], [4]])
        self.assertEqual(output.packed_input_ids[0, 4].item(), 110)
        self.assertEqual(output.packed_input_ids[0, 8].item(), 120)
        self.assertEqual(output.packed_attention_mask[1].sum().item(), 6)

    def test_grounding_dataset_builds_time_refine_sample(self):
        class FakeProcessor:
            tokenizer = object()

            def __init__(self):
                self.messages = None

            def apply_chat_template(self, messages, tokenize=False):
                self.messages = messages
                return "<chatml>"

            def __call__(self, **kwargs):
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                    "pixel_values_videos": torch.zeros(4, 2),
                }

        dataset = GroundingDataset.__new__(GroundingDataset)
        dataset.processor = FakeProcessor()
        dataset.model_args = SimpleNamespace(
            processor_path="TencentARC/TimeLens-7B",
            model_name_or_path="/models/Qwen2.5-VL-3B-Instruct",
            model_id="qwen25-vl-3b-coarse2refine",
            conv_type="chatml",
        )
        dataset.data_args = SimpleNamespace(
            min_tokens=16,
            total_tokens=64,
            fps=2.0,
            fps_max_frames=None,
            time_bin_count=301,
            frame_label_coverage_threshold=0.5,
            time_refine_eps=1e-6,
        )
        dataset._format_model_path = "TencentARC/TimeLens-7B"
        dataset.annos = [
            {
                "video_path": "/videos/example.mp4",
                "duration": 2.0,
                "query": "a person enters",
                "span": [0.5, 1.5],
            }
        ]

        videos = [
            (
                torch.zeros(4, 2),
                {"fps": 2.0, "frames_indices": [0, 1, 2, 3]},
            )
        ]
        with patch(
            "training.data.grounding.process_vision_info",
            return_value=(None, videos),
        ), patch(
            "training.data.grounding.preprocess",
            return_value=torch.tensor([-100, -100, -100, 0, 1]),
        ):
            sample = dataset._getitem_coarse2refine_sft(0)

        self.assertEqual(sample["frame_bin_ids"].tolist(), [0, 150])
        self.assertEqual(sample["frame_labels"].tolist(), [0, 1])
        self.assertEqual(sample["frame_valid_mask"].tolist(), [True, True])
        self.assertEqual(sample["gt_start"].tolist(), [0.5])
        self.assertEqual(sample["gt_end"].tolist(), [1.5])
        content = dataset.processor.messages[0]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "video", "text"])
        self.assertEqual(dataset.processor.messages[1]["role"], "assistant")
        self.assertEqual(
            dataset.processor.messages[1]["content"],
            "<vtg><fg><time_150><fg></vtg>",
        )


class FakeVisionCore(torch.nn.Module):
    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return pixel_values_videos

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
    ):
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        )
        return positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1), None


class FakeBaseConfig(PretrainedConfig):
    model_type = "fake_qwen25_vl"

    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=2048,
            hidden_size=8,
            vision_start_token_id=1,
            vision_end_token_id=3,
            image_token_id=4,
            video_token_id=2,
            pad_token_id=0,
            vision_config={"spatial_merge_size": 2},
            **kwargs,
        )


class FakeBaseModel(PreTrainedModel):
    config_class = FakeBaseConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = FakeVisionCore()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.transform = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden = torch.tanh(self.transform(inputs_embeds))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=(hidden,),
        )


class Stage2WrapperTest(unittest.TestCase):
    def test_wrapper_forward_and_backward(self):
        base_model = FakeBaseModel(FakeBaseConfig())
        token_spec = RegisteredCoarse2RefineTokens(
            fg_token_id=500,
            bg_token_id=501,
            vtg_token_id=502,
            vtg_end_token_id=503,
            time_token_ids=tuple(range(1000, 1301)),
        )
        wrapper = Coarse2RefineForConditionalGeneration.from_base_model(
            base_model,
            token_spec,
            base_model_name_or_path="fake-qwen25-vl-3b",
        )
        input_ids = torch.tensor(
            [
                [10, 1, 2, 3, 11, 12, 0],
                [0, 10, 1, 2, 3, 11, 12],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [-100, -100, -100, -100, -100, 7, 8],
                [-100, -100, -100, -100, -100, 9, 10],
            ],
            dtype=torch.long,
        )
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values_videos=torch.randn(3, 8),
            video_grid_thw=torch.tensor([[2, 2, 2], [1, 2, 2]]),
            frame_bin_ids=torch.tensor([[10, 20], [30, 0]]),
            frame_timestamps=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            frame_labels=torch.tensor([[0, 1], [1, 0]]),
            frame_valid_mask=torch.tensor([[True, True], [True, False]]),
            gt_start=torch.tensor([[0.2], [0.0]]),
            gt_end=torch.tensor([[0.8], [0.5]]),
            duration=torch.tensor([[10.0], [10.0]]),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(tuple(output.pred_start.shape), (2,))
        self.assertEqual(tuple(output.pred_end.shape), (2,))
        expected_smooth_l1 = smooth_l1_boundary_loss(
            torch.stack([output.pred_start, output.pred_end], dim=-1),
            torch.tensor([[0.2, 0.8], [0.0, 0.5]]),
        )
        self.assertTrue(torch.allclose(output.smooth_l1_loss, expected_smooth_l1))
        self.assertEqual(wrapper.config.lambda_reg, 0.1)
        output.loss.backward()
        self.assertIsNotNone(wrapper.time_refine_head.time_proj.linear1.weight.grad)
        self.assertIsNotNone(wrapper.base_model.transform.weight.grad)

    def test_candidate_parser_and_wrapper_generation(self):
        token_spec = RegisteredCoarse2RefineTokens(
            fg_token_id=500,
            bg_token_id=501,
            vtg_token_id=502,
            vtg_end_token_id=503,
            time_token_ids=tuple(range(1000, 1301)),
        )
        parsed = parse_time_refine_sequence(
            [999, 502, 500, 1030, 1010, 500, 503],
            token_spec,
            expected_time_bins=[10, 20, 30],
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.labels, (1, 0, 1))
        self.assertEqual(parsed.time_bins, (10, 20, 30))
        self.assertEqual(parsed.foreground_token_positions, (0, 2))
        self.assertEqual(parsed.foreground_token_offsets, (4, 3))
        self.assertEqual(parsed.status, "recovered_out_of_order")

        empty = parse_time_refine_sequence(
            [502, 500, 500, 503],
            token_spec,
            expected_time_bins=[10, 20, 30],
        )
        self.assertTrue(empty.valid)
        self.assertEqual(empty.labels, (0, 0, 0))
        self.assertEqual(empty.status, "no_foreground")

        candidate = build_candidate_windows(parsed.labels)
        self.assertEqual(candidate.selected_run, (0, 2))
        self.assertEqual(candidate.cleaned_labels, (1, 1, 1))
        self.assertEqual(candidate.start_window, (0, 2))

        score_first = build_candidate_windows(
            [1, 0, 1, 0, 0, 1],
            foreground_scores=[0.4, 0.0, 0.4, 0.0, 0.0, 0.8],
            max_background_gap=1,
        )
        self.assertEqual(score_first.selected_run, (5, 5))
        gap_excluded = build_candidate_windows(
            [1, 0, 1, 0, 0, 1],
            foreground_scores=[0.9, 0.0, 0.9, 0.0, 0.0, 0.8],
            max_background_gap=1,
        )
        self.assertEqual(gap_excluded.selected_run, (0, 2))
        longer_score_tie = build_candidate_windows(
            [1, 0, 1, 0, 0, 1],
            foreground_scores=[0.8, 0.0, 0.8, 0.0, 0.0, 0.8],
            max_background_gap=1,
        )
        self.assertEqual(longer_score_tie.selected_run, (0, 2))
        earlier_tie = build_candidate_windows(
            [1, 0, 0, 1],
            foreground_scores=[0.8, 0.0, 0.0, 0.8],
            max_background_gap=0,
        )
        self.assertEqual(earlier_tie.selected_run, (0, 0))

        base_model = FakeBaseModel(FakeBaseConfig())

        def fake_generate(input_ids=None, **kwargs):
            suffix = torch.tensor(
                [[502, 500, 1010, 500, 503]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            return torch.cat([input_ids, suffix], dim=1)

        base_model.generate = fake_generate
        wrapper = Coarse2RefineForConditionalGeneration.from_base_model(
            base_model,
            token_spec,
            base_model_name_or_path="fake-qwen25-vl-3b",
        )
        result = wrapper.generate_coarse2refine(
            input_ids=torch.tensor([[10, 1, 2, 3, 11]], dtype=torch.long),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            pixel_values_videos=torch.randn(2, 8),
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            frame_bin_ids=torch.tensor([[10, 20]]),
            frame_timestamps=torch.tensor([[0.0, 1.0]]),
            frame_valid_mask=torch.tensor([[True, True]]),
            duration=torch.tensor([[1.0]]),
            max_new_tokens=32,
        )
        self.assertEqual(result.statuses, ("ok",))
        self.assertEqual(result.parse_statuses, ("ok",))
        self.assertEqual(result.coarse_labels, ((1, 0),))
        self.assertTrue(torch.isfinite(result.pred_start).all())
        self.assertTrue(torch.isfinite(result.pred_end).all())

        def fake_generate_with_scores(input_ids=None, **kwargs):
            suffix = torch.tensor(
                [[502, 500, 1060, 1010, 500, 503]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            scores = []
            for offset in range(suffix.shape[1]):
                logits = torch.zeros(
                    (1, base_model.config.vocab_size),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                if offset == 2:
                    logits[0, 1060] = 5.0
                elif offset == 3:
                    logits[0, 1010] = 1.0
                scores.append(logits)
            return SimpleNamespace(
                sequences=torch.cat([input_ids, suffix], dim=1),
                scores=tuple(scores),
            )

        base_model.generate = fake_generate_with_scores
        scored_result = wrapper.generate_coarse2refine(
            input_ids=torch.tensor([[10, 1, 2, 3, 11]], dtype=torch.long),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            pixel_values_videos=torch.randn(6, 8),
            video_grid_thw=torch.tensor([[6, 2, 2]]),
            frame_bin_ids=torch.tensor([[10, 20, 30, 40, 50, 60]]),
            frame_timestamps=torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
            frame_valid_mask=torch.tensor([[True, True, True, True, True, True]]),
            duration=torch.tensor([[5.0]]),
            max_new_tokens=32,
        )
        self.assertEqual(scored_result.statuses, ("ok",))
        self.assertEqual(
            scored_result.parse_statuses,
            ("recovered_out_of_order",),
        )
        self.assertEqual(scored_result.coarse_labels, ((1, 0, 0, 0, 0, 1),))
        self.assertEqual(scored_result.candidate_windows[0].selected_run, (5, 5))

    def test_wrapper_checkpoint_save_and_load(self):
        token_spec = RegisteredCoarse2RefineTokens(
            fg_token_id=500,
            bg_token_id=501,
            vtg_token_id=502,
            vtg_end_token_id=503,
            time_token_ids=tuple(range(1000, 1301)),
        )
        wrapper = Coarse2RefineForConditionalGeneration.from_base_model(
            FakeBaseModel(FakeBaseConfig()),
            token_spec,
            base_model_name_or_path="fake-qwen25-vl-3b",
        )
        directory = Path("tests/_stage3_tmp")
        directory.mkdir(exist_ok=True)
        (directory / "base_model").mkdir(exist_ok=True)
        wrapper.save_pretrained(directory)
        self.assertTrue((directory / "base_model").exists())
        with (directory / "config.json").open("r", encoding="utf-8") as reader:
            saved_config = json.load(reader)
        self.assertEqual(saved_config["model_type"], "coarse2refine")
        self.assertEqual(
            saved_config["architectures"], ["Coarse2RefineForConditionalGeneration"]
        )
        loaded = Coarse2RefineForConditionalGeneration.from_pretrained(
            directory,
            base_model=FakeBaseModel(FakeBaseConfig()),
        )
        self.assertEqual(
            loaded.config.time_token_ids,
            list(token_spec.time_token_ids),
        )
        self.assertTrue(
            torch.equal(
                wrapper.time_refine_head.start_scorer.weight,
                loaded.time_refine_head.start_scorer.weight,
            )
        )
        fresh_base = FakeBaseModel(FakeBaseConfig())
        fresh_wrapper = Coarse2RefineForConditionalGeneration.from_base_model(
            fresh_base,
            token_spec,
            base_model_name_or_path="fake-qwen25-vl-3b",
        )
        fresh_wrapper.load_attached_base_checkpoint(directory / "base_model")
        self.assertTrue(
            torch.equal(
                wrapper.base_model.transform.weight,
                fresh_wrapper.base_model.transform.weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
