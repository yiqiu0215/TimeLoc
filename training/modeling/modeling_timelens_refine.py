import warnings
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, PreTrainedModel
from transformers.generation import GenerationMixin

from training.modeling.candidate_parser import (
    CandidateWindows,
    build_candidate_windows,
    parse_time_refine_sequence,
)
from training.modeling.configuration_timelens_refine import TimeLensRefineConfig
from training.modeling.losses import diou_loss_1d, smooth_l1_boundary_loss
from training.modeling.outputs import (
    TimeLensRefineInferenceOutput,
    TimeLensRefineOutput,
)
from training.modeling.refine_windows import build_training_boundary_window
from training.modeling.special_tokens import (
    TIME_BIN_COUNT,
    RegisteredTimeRefineTokens,
)
from training.modeling.time_refine_head import TimeRefineHead
from training.modeling.time_token_packer import TimeTokenPacker


def _first_config_value(configs, name: str, default=None):
    for config in configs:
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _iter_wrapped_models(model):
    queue = [model]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        yield current
        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("model", "base_model"):
            try:
                candidate = getattr(current, name, None)
            except Exception:
                candidate = None
            if candidate is not None and candidate is not current:
                queue.append(candidate)


def _resolve_multimodal_config(model):
    for candidate in _iter_wrapped_models(model):
        config = getattr(candidate, "config", None)
        if config is None:
            continue
        if any(
            hasattr(config, name)
            for name in (
                "vision_config",
                "text_config",
                "vision_start_token_id",
                "image_token_id",
            )
        ):
            return config
    raise ValueError("Could not locate the wrapped Qwen2.5-VL configuration.")


def _resolve_model_with_attr(model, attribute: str):
    for candidate in _iter_wrapped_models(model):
        if hasattr(candidate, attribute):
            return candidate
    return None


class TimeLensRefineForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """HF-compatible wrapper for Qwen2.5-VL-3B-TimeLens SFT refinement."""

    config_class = TimeLensRefineConfig
    base_model_prefix = "_base_model"

    def __init__(self, config: TimeLensRefineConfig, base_model=None):
        super().__init__(config)
        if config.llm_hidden_size is None:
            raise ValueError("TimeLensRefineConfig.llm_hidden_size is required.")
        self._base_model = None
        self.time_refine_head = TimeRefineHead(
            llm_hidden_size=int(config.llm_hidden_size),
            time_hidden_size=config.time_hidden_size,
            num_attention_heads=config.refine_attention_heads,
            ffn_expansion_ratio=config.refine_ffn_expansion_ratio,
            dropout=config.refine_dropout,
            classification_embedding_dim=config.classification_embedding_dim,
            time_embedding_dim=config.time_embedding_dim,
            branch_embedding_dim=config.branch_embedding_dim,
            time_bin_count=config.time_bin_count,
        )
        self.time_token_packer = None
        self._last_refine_window_edge_cases = None
        if base_model is not None:
            self.attach_base_model(base_model)

    @classmethod
    def from_base_model(
        cls,
        base_model,
        token_spec,
        base_model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        base_config = _resolve_multimodal_config(base_model)
        config = TimeLensRefineConfig.from_base_model_config(
            base_config,
            base_model_name_or_path=base_model_name_or_path,
            token_spec=token_spec,
            **kwargs,
        )
        return cls(config, base_model=base_model)

    @property
    def base_model(self):
        return self._base_model

    @property
    def time_proj(self):
        return self.time_refine_head.time_proj

    @property
    def time_refine_transformer(self):
        return self.time_refine_head.transformer

    @property
    def start_scorer(self):
        return self.time_refine_head.start_scorer

    @property
    def end_scorer(self):
        return self.time_refine_head.end_scorer

    @property
    def branch_embedding(self):
        return self.time_refine_head.branch_embedding

    @property
    def time_refine_token_spec(self) -> RegisteredTimeRefineTokens:
        return RegisteredTimeRefineTokens(
            fg_token_id=int(self.config.fg_token_id),
            bg_token_id=int(self.config.bg_token_id),
            vtg_token_id=int(self.config.vtg_token_id),
            vtg_end_token_id=int(self.config.vtg_end_token_id),
            time_token_ids=tuple(int(value) for value in self.config.time_token_ids),
        )

    def attach_base_model(self, base_model):
        self._base_model = base_model
        parameter = next(base_model.parameters())
        if any(value.is_meta for value in self.time_refine_head.parameters()):
            self.time_refine_head.to_empty(device=parameter.device)
        self.time_refine_head.to(dtype=parameter.dtype)
        self._build_time_token_packer()
        return self

    def save_pretrained(
        self,
        save_directory,
        is_main_process: bool = True,
        state_dict=None,
        save_function=torch.save,
        push_to_hub: bool = False,
        max_shard_size="5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token=None,
        **kwargs,
    ):
        """Save the refinement config/head and the attached base model together."""

        if self.base_model is None:
            raise RuntimeError("Cannot save a TimeLensRefine wrapper without a base model.")
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        base_directory = save_directory / "base_model"
        base_directory.mkdir(parents=True, exist_ok=True)

        full_state_dict = state_dict if state_dict is not None else self.state_dict()
        prefix = "_base_model."
        base_state_dict = {
            key[len(prefix) :]: value
            for key, value in full_state_dict.items()
            if key.startswith(prefix)
        }
        head_state_dict = {
            key: value
            for key, value in full_state_dict.items()
            if not key.startswith(prefix)
        }
        base_save_kwargs = {
            "is_main_process": is_main_process,
            "max_shard_size": max_shard_size,
            "safe_serialization": safe_serialization,
            "variant": variant,
            "token": token,
        }
        self.base_model.save_pretrained(
            base_directory,
            state_dict=base_state_dict or None,
            **base_save_kwargs,
        )
        self.config.base_model_subdir = "base_model"
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=head_state_dict or None,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            **kwargs,
        )

    def load_attached_base_checkpoint(self, base_directory) -> None:
        """Load the nested base-model weights into an already attached wrapper."""

        if self.base_model is None:
            raise RuntimeError("Cannot load base weights without an attached model.")
        base_directory = Path(base_directory)
        adapter_config = base_directory / "adapter_config.json"
        if adapter_config.exists():
            if not hasattr(self.base_model, "load_adapter"):
                raise RuntimeError(
                    "The checkpoint contains a PEFT adapter but the attached base model "
                    "is not a PEFT model."
                )
            active_adapter = getattr(self.base_model, "active_adapter", "default")
            if isinstance(active_adapter, (list, tuple)):
                active_adapter = active_adapter[0]
            self.base_model.load_adapter(
                str(base_directory),
                active_adapter,
                is_trainable=True,
            )
            return
        safe_file = base_directory / "model.safetensors"
        bin_file = base_directory / "pytorch_model.bin"
        if safe_file.exists():
            from safetensors.torch import load_file

            base_state_dict = load_file(str(safe_file), device="cpu")
            missing, unexpected = self.base_model.load_state_dict(
                base_state_dict, strict=False
            )
        elif bin_file.exists():
            base_state_dict = torch.load(
                bin_file,
                map_location="cpu",
                weights_only=True,
            )
            missing, unexpected = self.base_model.load_state_dict(
                base_state_dict, strict=False
            )
        else:
            from transformers.modeling_utils import load_sharded_checkpoint

            load_result = load_sharded_checkpoint(
                self.base_model,
                str(base_directory),
                strict=False,
                prefer_safe=True,
            )
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
        if unexpected:
            raise ValueError(
                "Nested base checkpoint contains unexpected keys: "
                f"{unexpected[:8]}"
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a wrapper checkpoint and reconstruct its attached base model."""

        checkpoint = Path(pretrained_model_name_or_path)
        supplied_config = kwargs.get("config")
        if isinstance(supplied_config, TimeLensRefineConfig):
            config = supplied_config
        else:
            config_source = supplied_config or pretrained_model_name_or_path
            config = TimeLensRefineConfig.from_pretrained(config_source)
            kwargs["config"] = config

        base_model = kwargs.pop("base_model", None)
        if base_model is None:
            base_source = None
            subdir = getattr(config, "base_model_subdir", None)
            if subdir:
                candidate = checkpoint / subdir
                if candidate.exists():
                    base_source = candidate
            if base_source is None:
                base_source = getattr(config, "base_model_name_or_path", None)
            if base_source is None:
                raise ValueError(
                    "TimeLensRefine checkpoint does not specify a base model source."
                )

            base_load_keys = {
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
                "revision",
                "torch_dtype",
                "dtype",
                "device_map",
                "max_memory",
                "offload_folder",
                "offload_state_dict",
                "quantization_config",
                "attn_implementation",
                "trust_remote_code",
                "use_safetensors",
                "weights_only",
                "low_cpu_mem_usage",
            }
            base_load_kwargs = {
                key: value for key, value in kwargs.items() if key in base_load_keys
            }
            adapter_config = Path(base_source) / "adapter_config.json"
            if adapter_config.exists():
                base_origin = getattr(config, "base_model_name_or_path", None)
                if not base_origin:
                    raise ValueError(
                        "PEFT TimeLensRefine checkpoint is missing the original base model path."
                    )
                base_model = AutoModelForImageTextToText.from_pretrained(
                    base_origin,
                    **base_load_kwargs,
                )
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise RuntimeError(
                        "Loading a PEFT TimeLensRefine checkpoint requires peft."
                    ) from exc
                base_model = PeftModel.from_pretrained(
                    base_model,
                    str(base_source),
                    is_trainable=False,
                )
            else:
                base_model = AutoModelForImageTextToText.from_pretrained(
                    str(base_source),
                    **base_load_kwargs,
                )
        if checkpoint.exists() and checkpoint.is_dir():
            model = cls(config, *model_args, base_model=base_model)
            state_dict = None
            safe_file = checkpoint / "model.safetensors"
            bin_file = checkpoint / "pytorch_model.bin"
            if safe_file.exists():
                from safetensors.torch import load_file

                state_dict = load_file(str(safe_file), device="cpu")
            elif bin_file.exists():
                state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
            else:
                index_file = checkpoint / "model.safetensors.index.json"
                if not index_file.exists():
                    index_file = checkpoint / "pytorch_model.bin.index.json"
                if index_file.exists():
                    import json

                    with index_file.open("r", encoding="utf-8") as reader:
                        index = json.load(reader)
                    state_dict = {}
                    for shard_name in sorted(set(index["weight_map"].values())):
                        shard_path = checkpoint / shard_name
                        if shard_path.suffix == ".safetensors":
                            from safetensors.torch import load_file

                            state_dict.update(load_file(str(shard_path), device="cpu"))
                        else:
                            state_dict.update(
                                torch.load(
                                    shard_path,
                                    map_location="cpu",
                                    weights_only=True,
                                )
                            )
            if state_dict is None:
                raise FileNotFoundError(
                    f"No model weight file found in TimeLensRefine checkpoint {checkpoint}."
                )
            non_lora_state_file = checkpoint / "non_lora_state_dict.bin"
            if non_lora_state_file.exists():
                state_dict.update(
                    torch.load(
                        non_lora_state_file,
                        map_location="cpu",
                        weights_only=True,
                    )
                )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            unexpected = [key for key in unexpected if not key.startswith("_base_model.")]
            missing = [
                key
                for key in missing
                if not key.startswith("_base_model.")
            ]
            if unexpected or missing:
                raise ValueError(
                    "TimeLensRefine checkpoint state is incompatible: "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            model.eval()
            return model

        kwargs["base_model"] = base_model
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

    def _base_configs(self):
        base_config = _resolve_multimodal_config(self.base_model)
        return [
            base_config,
            getattr(base_config, "text_config", None),
        ]

    def _build_time_token_packer(self):
        if self.base_model is None:
            self.time_token_packer = None
            return
        configs = self._base_configs()
        vision_start_token_id = _first_config_value(
            [self.config, *configs], "vision_start_token_id"
        )
        vision_end_token_id = _first_config_value(
            [self.config, *configs], "vision_end_token_id"
        )
        image_token_id = _first_config_value([self.config, *configs], "image_token_id")
        video_token_id = _first_config_value([self.config, *configs], "video_token_id")
        pad_token_id = _first_config_value(
            [self.config, *configs], "pad_token_id", 0
        )
        missing = {
            name: value
            for name, value in {
                "vision_start_token_id": vision_start_token_id,
                "vision_end_token_id": vision_end_token_id,
                "image_token_id": image_token_id,
                "video_token_id": video_token_id,
            }.items()
            if value is None
        }
        if missing:
            raise ValueError(f"Missing Qwen vision token ids: {missing}.")
        if len(self.config.time_token_ids) != TIME_BIN_COUNT:
            raise ValueError(
                f"Expected {TIME_BIN_COUNT} time token ids, got "
                f"{len(self.config.time_token_ids)}."
            )
        self.time_token_packer = TimeTokenPacker(
            embedding_layer=self.base_model.get_input_embeddings(),
            vision_start_token_id=int(vision_start_token_id),
            vision_end_token_id=int(vision_end_token_id),
            image_token_id=int(image_token_id),
            video_token_id=int(video_token_id),
            time_token_ids=self.config.time_token_ids,
            spatial_merge_size=int(self.config.spatial_merge_size),
            pad_token_id=int(pad_token_id),
        )

    def get_input_embeddings(self):
        if self.base_model is None:
            raise RuntimeError("The base model has not been attached.")
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
        self._build_time_token_packer()

    def get_output_embeddings(self):
        if self.base_model is None:
            raise RuntimeError("The base model has not been attached.")
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, **kwargs):
        resized = self.base_model.resize_token_embeddings(
            new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )
        self._build_time_token_packer()
        return resized

    def enable_input_require_grads(self):
        if hasattr(self.base_model, "enable_input_require_grads"):
            return self.base_model.enable_input_require_grads()
        return super().enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return super().gradient_checkpointing_disable()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Keep legacy generation and expose refinement generation by metadata."""

        refine_keys = {
            "frame_bin_ids",
            "frame_timestamps",
            "frame_valid_mask",
            "duration",
        }
        if refine_keys.intersection(kwargs):
            return self.generate_time_refine(*args, **kwargs)
        
        return self.base_model.generate(*args, **kwargs)

    @torch.no_grad()
    def generate_time_refine(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        frame_bin_ids: torch.Tensor,
        frame_timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        video_grid_thw: torch.Tensor,
        duration: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> TimeLensRefineInferenceOutput:
        """Generate coarse VTG classes, then refine the selected candidate interval."""

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Inference input_ids and attention_mask must be [B, L].")
        batch_size = input_ids.shape[0]
        metadata = {
            "frame_bin_ids": frame_bin_ids,
            "frame_timestamps": frame_timestamps,
            "frame_valid_mask": frame_valid_mask,
        }
        for name, value in metadata.items():
            if value.ndim != 2 or value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} must have shape [B, N] with B={batch_size}, "
                    f"got {tuple(value.shape)}."
                )
        duration = torch.as_tensor(duration, device=input_ids.device, dtype=torch.float32)
        if duration.ndim == 1:
            duration = duration.unsqueeze(-1)
        if duration.shape != (batch_size, 1):
            raise ValueError(
                f"duration must have shape [{batch_size}, 1], got {tuple(duration.shape)}."
            )
        if not torch.isfinite(duration).all() or torch.any(duration <= 0):
            raise ValueError("Inference duration must be finite and positive.")

        packed, packed_image_grid_thw, packed_position_ids = self._prepare_packed_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_bin_ids=frame_bin_ids,
            frame_timestamps=frame_timestamps,
            frame_valid_mask=frame_valid_mask,
            video_grid_thw=video_grid_thw,
            pixel_values_videos=pixel_values_videos,
            visual_embeddings=visual_embeddings,
        )
        base_outputs = self.base_model(
            input_ids=packed.packed_input_ids,
            attention_mask=packed.packed_attention_mask,
            position_ids=packed_position_ids,
            inputs_embeds=packed.packed_inputs_embeds,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=packed_image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )
        if base_outputs.hidden_states is None:
            raise ValueError("Qwen2.5-VL inference did not return hidden states.")

        generation_inputs = dict(
            input_ids=packed.packed_input_ids,
            attention_mask=packed.packed_attention_mask,
            inputs_embeds=packed.packed_inputs_embeds,
            image_grid_thw=packed_image_grid_thw,
            video_grid_thw=None,
            pixel_values=None,
            pixel_values_videos=None,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generation_inputs.update(generation_kwargs)
        generation_inputs.pop("position_ids", None)
        generation_inputs.pop("labels", None)
        generation_result = self.base_model.generate(**generation_inputs)
        if hasattr(generation_result, "sequences"):
            generated_ids = generation_result.sequences
            generation_scores = getattr(generation_result, "scores", None)
        else:
            generated_ids = generation_result
            generation_scores = None
        if generated_ids.ndim != 2 or generated_ids.shape[0] != batch_size:
            raise ValueError(
                "Generation must return token ids with shape [B, L]."
            )

        prompt_length = int(packed.packed_input_ids.shape[1])
        parsed_results = []
        candidates = []
        coarse_labels = []
        coarse_bins = []
        for batch_index in range(batch_size):
            if generated_ids.shape[1] >= prompt_length:
                generated_part = generated_ids[batch_index, prompt_length:].tolist()
            else:
                generated_part = generated_ids[batch_index].tolist()
            valid = frame_valid_mask[batch_index].bool()
            expected_bins = frame_bin_ids[batch_index, valid].long().tolist()
            parsed = parse_time_refine_sequence(
                generated_part,
                self.time_refine_token_spec,
                expected_time_bins=expected_bins,
                expected_length=len(expected_bins),
            )
            parsed_results.append(parsed)
            if not parsed.valid:
                candidates.append(None)
                coarse_labels.append(())
                coarse_bins.append(())
                continue

            foreground_scores = None
            if generation_scores is not None:
                score_values = []
                for offset in parsed.classification_token_offsets:
                    if offset >= len(generation_scores):
                        score_values = []
                        break
                    logits = generation_scores[offset][batch_index].float()
                    score_values.append(
                        float(torch.softmax(logits, dim=-1)[self.config.fg_token_id])
                    )
                if len(score_values) == len(parsed.labels):
                    foreground_scores = score_values
            candidates.append(
                build_candidate_windows(
                    parsed.labels,
                    foreground_scores=foreground_scores,
                    max_background_gap=self.config.inference_max_background_gap,
                    expansion=self.config.inference_candidate_expansion,
                    boundary_radius=self.config.inference_boundary_radius,
                )
            )
            coarse_labels.append(parsed.labels)
            coarse_bins.append(parsed.time_bins)

        pred_start = torch.full(
            (batch_size,), -1.0, dtype=duration.dtype, device=input_ids.device
        )
        pred_end = pred_start.clone()
        active_indices = [
            index
            for index, (parsed, candidate) in enumerate(zip(parsed_results, candidates))
            if parsed.valid and candidate is not None and candidate.has_foreground
        ]
        if active_indices:
            inferred_labels = torch.zeros_like(frame_bin_ids, dtype=torch.long)
            for batch_index, parsed in enumerate(parsed_results):
                if parsed.valid:
                    valid_positions = frame_valid_mask[batch_index].bool().nonzero(
                        as_tuple=True
                    )[0]
                    valid_positions = valid_positions.to(inferred_labels.device)
                    inferred_labels[batch_index, valid_positions] = torch.tensor(
                        parsed.labels,
                        dtype=torch.long,
                        device=inferred_labels.device,
                    )
            groups = self._build_inference_refine_groups(
                base_outputs.hidden_states[-1],
                packed.visual_block_token_ranges,
                frame_bin_ids.to(input_ids.device),
                inferred_labels.to(input_ids.device),
                frame_valid_mask.to(input_ids.device),
                candidates,
            )
            refine_output = self.time_refine_head(
                start_visual_groups=[
                    groups["start_visual_groups"][index]
                    for index in range(len(active_indices))
                ],
                start_classification_groups=groups["start_classification_groups"],
                start_relative_time_groups=groups["start_relative_time_groups"],
                start_absolute_time_groups=groups["start_absolute_time_groups"],
                end_visual_groups=[
                    groups["end_visual_groups"][index]
                    for index in range(len(active_indices))
                ],
                end_classification_groups=groups["end_classification_groups"],
                end_relative_time_groups=groups["end_relative_time_groups"],
                end_absolute_time_groups=groups["end_absolute_time_groups"],
            )
            active_tensor = torch.tensor(
                active_indices, dtype=torch.long, device=input_ids.device
            )
            pred_start[active_tensor] = (
                duration.reshape(-1)[active_tensor]
                * refine_output.pred_start_q
                / float(TIME_BIN_COUNT - 1)
            ).clamp_min(0.0)
            pred_end[active_tensor] = (
                duration.reshape(-1)[active_tensor]
                * refine_output.pred_end_q
                / float(TIME_BIN_COUNT - 1)
            ).clamp_min(0.0)

        statuses = []
        for parsed, candidate in zip(parsed_results, candidates):
            if not parsed.valid:
                statuses.append("invalid_generation")
            elif candidate is None or not candidate.has_foreground:
                statuses.append("no_foreground")
            else:
                statuses.append("ok")
        return TimeLensRefineInferenceOutput(
            pred_start=pred_start,
            pred_end=pred_end,
            generated_ids=generated_ids,
            prompt_length=prompt_length,
            statuses=tuple(statuses),
            coarse_labels=tuple(tuple(values) for values in coarse_labels),
            coarse_time_bins=tuple(tuple(values) for values in coarse_bins),
            candidate_windows=tuple(candidates),
        )

    def _get_video_features(self, pixel_values_videos, video_grid_thw):
        if pixel_values_videos is None or video_grid_thw is None:
            raise ValueError(
                "TimeRefine forward requires pixel_values_videos and video_grid_thw "
                "when visual_embeddings are not supplied."
            )
        core_model = _resolve_model_with_attr(self.base_model, "get_video_features")
        if core_model is not None:
            return core_model.get_video_features(
                pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        visual_owner = _resolve_model_with_attr(self.base_model, "visual")
        visual = getattr(visual_owner, "visual", None)
        if visual is None:
            raise ValueError("Could not locate a Qwen video vision encoder.")
        return visual(pixel_values_videos, grid_thw=video_grid_thw)

    @staticmethod
    def _build_packed_image_grid(video_grid_thw, frame_valid_mask, device):
        grid = torch.as_tensor(video_grid_thw, dtype=torch.long, device=device)
        if grid.ndim == 1:
            grid = grid.reshape(1, 3)
        if grid.ndim != 2 or grid.shape[-1] != 3:
            raise ValueError(f"video_grid_thw must be [B, 3], got {tuple(grid.shape)}.")
        if frame_valid_mask is None:
            counts = grid[:, 0]
        else:
            counts = torch.as_tensor(frame_valid_mask, device=device).bool().sum(dim=1)
        rows = []
        for batch_index, (temporal, height, width) in enumerate(grid.tolist()):
            count = int(counts[batch_index].item())
            if count != int(temporal):
                raise ValueError(
                    "frame_valid_mask count must match video_grid_thw temporal size: "
                    f"sample={batch_index}, mask={count}, grid_t={temporal}."
                )
            rows.append(
                torch.tensor(
                    [[1, int(height), int(width)]] * count,
                    dtype=torch.long,
                    device=device,
                )
            )
        if not rows:
            raise ValueError("At least one packed video is required.")
        return torch.cat(rows, dim=0)

    def _build_qwen_position_ids(
        self,
        packed_input_ids,
        packed_attention_mask,
        packed_image_grid_thw,
    ):
        core_model = _resolve_model_with_attr(self.base_model, "get_rope_index")
        get_rope_index = getattr(core_model, "get_rope_index", None)
        if get_rope_index is None:
            return None
        result = get_rope_index(
            packed_input_ids,
            image_grid_thw=packed_image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=packed_attention_mask,
        )
        if isinstance(result, tuple):
            result = result[0]
        return result

    def _prepare_packed_inputs(
        self,
        input_ids,
        attention_mask,
        frame_bin_ids,
        frame_timestamps,
        frame_valid_mask,
        video_grid_thw,
        pixel_values_videos=None,
        visual_embeddings=None,
        labels=None,
    ):
        if self.time_token_packer is None:
            raise RuntimeError("TimeTokenPacker has not been initialized.")
        if visual_embeddings is None:
            visual_embeddings = self._get_video_features(
                pixel_values_videos,
                video_grid_thw,
            )
        packed = self.time_token_packer(
            visual_embeddings=visual_embeddings,
            video_grid_thw=video_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_bin_ids=frame_bin_ids,
            frame_timestamps=frame_timestamps,
            frame_valid_mask=frame_valid_mask,
            labels=labels,
        )
        packed_image_grid_thw = self._build_packed_image_grid(
            video_grid_thw,
            frame_valid_mask,
            packed.packed_input_ids.device,
        )
        packed_position_ids = self._build_qwen_position_ids(
            packed.packed_input_ids,
            packed.packed_attention_mask,
            packed_image_grid_thw,
        )
        if packed_position_ids is None:
            packed_position_ids = packed.packed_position_ids
        return packed, packed_image_grid_thw, packed_position_ids

    @staticmethod
    def _make_branch_groups(
        blocks,
        window,
        frame_labels,
        frame_bin_ids,
    ):
        indices = list(window.indices)
        selected_bins = frame_bin_ids[indices]
        return {
            "visual": [[blocks[index] for index in indices]],
            "classification": [frame_labels[indices]],
            "relative_time": [selected_bins - selected_bins[0]],
            "absolute_time": [selected_bins],
        }

    def _build_refine_groups(
        self,
        hidden_states,
        visual_block_token_ranges,
        frame_bin_ids,
        frame_timestamps,
        frame_labels,
        frame_valid_mask,
        gt_start,
        gt_end,
    ):
        start_visual_groups = []
        start_classification_groups = []
        start_relative_time_groups = []
        start_absolute_time_groups = []
        end_visual_groups = []
        end_classification_groups = []
        end_relative_time_groups = []
        end_absolute_time_groups = []
        edge_cases = []

        for batch_index, ranges in enumerate(visual_block_token_ranges):
            valid = frame_valid_mask[batch_index].bool()
            valid_positions = valid.nonzero(as_tuple=True)[0]
            timestamps = frame_timestamps[batch_index, valid].float()
            bins = frame_bin_ids[batch_index, valid].long()
            labels = frame_labels[batch_index, valid].long()
            if len(ranges) != int(valid_positions.numel()):
                raise ValueError(
                    "Visual block ranges and frame_valid_mask disagree: "
                    f"ranges={len(ranges)}, valid={int(valid_positions.numel())}."
                )
            blocks = [
                hidden_states[batch_index, start:end]
                for start, end in ranges
            ]
            start_window = build_training_boundary_window(
                timestamps.detach().cpu(),
                float(gt_start[batch_index].item()),
                left_context=self.config.train_start_left_context,
                right_context=self.config.train_start_right_context,
            )
            end_window = build_training_boundary_window(
                timestamps.detach().cpu(),
                float(gt_end[batch_index].item()),
                left_context=self.config.train_end_left_context,
                right_context=self.config.train_end_right_context,
            )
            start = self._make_branch_groups(blocks, start_window, labels, bins)
            end = self._make_branch_groups(blocks, end_window, labels, bins)
            start_visual_groups.extend(start["visual"])
            start_classification_groups.extend(start["classification"])
            start_relative_time_groups.extend(start["relative_time"])
            start_absolute_time_groups.extend(start["absolute_time"])
            end_visual_groups.extend(end["visual"])
            end_classification_groups.extend(end["classification"])
            end_relative_time_groups.extend(end["relative_time"])
            end_absolute_time_groups.extend(end["absolute_time"])
            edge_cases.append([start_window.edge_clipped, end_window.edge_clipped])

        return {
            "start_visual_groups": start_visual_groups,
            "start_classification_groups": start_classification_groups,
            "start_relative_time_groups": start_relative_time_groups,
            "start_absolute_time_groups": start_absolute_time_groups,
            "end_visual_groups": end_visual_groups,
            "end_classification_groups": end_classification_groups,
            "end_relative_time_groups": end_relative_time_groups,
            "end_absolute_time_groups": end_absolute_time_groups,
            "edge_cases": torch.tensor(edge_cases, dtype=torch.bool, device=hidden_states.device),
        }

    @staticmethod
    def _make_inference_branch_groups(blocks, indices, labels, frame_bin_ids):
        indices = list(indices)
        if not indices:
            raise ValueError("Inference refinement windows cannot be empty.")
        selected_bins = frame_bin_ids[indices].long()
        return {
            "visual": [[blocks[index] for index in indices]],
            "classification": [labels[indices].long()],
            "relative_time": [selected_bins - selected_bins[0]],
            "absolute_time": [selected_bins],
        }

    def _build_inference_refine_groups(
        self,
        hidden_states,
        visual_block_token_ranges,
        frame_bin_ids,
        frame_labels,
        frame_valid_mask,
        candidate_windows,
    ):
        active_batch_indices = []
        groups = {
            "start_visual_groups": [],
            "start_classification_groups": [],
            "start_relative_time_groups": [],
            "start_absolute_time_groups": [],
            "end_visual_groups": [],
            "end_classification_groups": [],
            "end_relative_time_groups": [],
            "end_absolute_time_groups": [],
        }
        for batch_index, candidate in enumerate(candidate_windows):
            if candidate is None or not candidate.has_foreground:
                continue
            valid = frame_valid_mask[batch_index].bool()
            labels = frame_labels[batch_index, valid].long()
            bins = frame_bin_ids[batch_index, valid].long()
            ranges = visual_block_token_ranges[batch_index]
            if len(ranges) != int(valid.sum().item()):
                raise ValueError(
                    "Visual block ranges and inference frame_valid_mask disagree."
                )
            blocks = [hidden_states[batch_index, start:end] for start, end in ranges]
            start_indices = range(
                candidate.start_window[0], candidate.start_window[1] + 1
            )
            end_indices = range(candidate.end_window[0], candidate.end_window[1] + 1)
            start = self._make_inference_branch_groups(
                blocks, start_indices, labels, bins
            )
            end = self._make_inference_branch_groups(
                blocks, end_indices, labels, bins
            )
            for branch_name, branch in (("start", start), ("end", end)):
                groups[f"{branch_name}_visual_groups"].extend(branch["visual"])
                groups[f"{branch_name}_classification_groups"].extend(
                    branch["classification"]
                )
                groups[f"{branch_name}_relative_time_groups"].extend(
                    branch["relative_time"]
                )
                groups[f"{branch_name}_absolute_time_groups"].extend(
                    branch["absolute_time"]
                )
            active_batch_indices.append(batch_index)
        groups["active_batch_indices"] = active_batch_indices
        return groups

    def _passthrough_output(self, base_outputs):
        return TimeLensRefineOutput(
            loss=getattr(base_outputs, "loss", None),
            ntp_loss=getattr(base_outputs, "loss", None),
            logits=getattr(base_outputs, "logits", None),
            hidden_states=getattr(base_outputs, "hidden_states", None),
            attentions=getattr(base_outputs, "attentions", None),
            past_key_values=getattr(base_outputs, "past_key_values", None),
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        frame_bin_ids: Optional[torch.Tensor] = None,
        frame_timestamps: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        gt_start: Optional[torch.Tensor] = None,
        gt_end: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TimeLensRefineOutput:
        if self.base_model is None:
            raise RuntimeError("The base model has not been attached.")
        refine_fields = (
            frame_bin_ids,
            frame_timestamps,
            frame_labels,
            frame_valid_mask,
            gt_start,
            gt_end,
            duration,
        )
        if all(value is None for value in refine_fields):
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                cache_position=cache_position,
                **kwargs,
            )
            return self._passthrough_output(base_outputs)
        if any(value is None for value in refine_fields):
            raise ValueError(
                "TimeRefine forward requires all frame metadata and GT fields: "
                "frame_bin_ids, frame_timestamps, frame_labels, frame_valid_mask, "
                "gt_start, gt_end, duration."
            )

        if input_ids is None or attention_mask is None:
            raise ValueError("TimeRefine forward requires input_ids and attention_mask.")
        if labels is None:
            raise ValueError("TimeRefine training requires labels for NTP loss.")
        if video_grid_thw is None:
            raise ValueError("TimeRefine forward requires video_grid_thw.")
        batch_size = input_ids.shape[0]
        for name, value in {
            "frame_bin_ids": frame_bin_ids,
            "frame_timestamps": frame_timestamps,
            "frame_labels": frame_labels,
            "frame_valid_mask": frame_valid_mask,
        }.items():
            if value.ndim != 2 or value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} must have shape [B, N] with B={batch_size}, "
                    f"got {tuple(value.shape)}."
                )
        for name, value in {
            "gt_start": gt_start,
            "gt_end": gt_end,
            "duration": duration,
        }.items():
            if value.ndim != 2 or value.shape != (batch_size, 1):
                raise ValueError(
                    f"{name} must have shape [{batch_size}, 1], got {tuple(value.shape)}."
                )
        duration_flat = duration.reshape(-1).to(input_ids.device, dtype=torch.float32)
        gt_start_flat = gt_start.reshape(-1).to(input_ids.device, dtype=torch.float32)
        gt_end_flat = gt_end.reshape(-1).to(input_ids.device, dtype=torch.float32)
        if not torch.isfinite(torch.stack([duration_flat, gt_start_flat, gt_end_flat])).all():
            raise ValueError("duration and GT boundaries must be finite.")
        if torch.any(duration_flat <= 0) or torch.any(gt_start_flat < 0) or torch.any(
            gt_start_flat > gt_end_flat
        ) or torch.any(gt_end_flat > duration_flat):
            raise ValueError("Training GT must satisfy 0 <= start <= end <= duration.")

        if visual_embeddings is None:
            visual_embeddings = self._get_video_features(
                pixel_values_videos,
                video_grid_thw,
            )
        packed = self.time_token_packer(
            visual_embeddings=visual_embeddings,
            video_grid_thw=video_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_bin_ids=frame_bin_ids,
            frame_timestamps=frame_timestamps,
            frame_valid_mask=frame_valid_mask,
            input_embeddings=inputs_embeds,
            labels=labels,
        )
        packed_image_grid_thw = self._build_packed_image_grid(
            video_grid_thw,
            frame_valid_mask,
            packed.packed_input_ids.device,
        )
        if position_ids is not None:
            if position_ids.shape[-1] != packed.packed_input_ids.shape[-1]:
                raise ValueError(
                    "Explicit position_ids must already match the packed sequence length."
                )
            packed_position_ids = position_ids
        else:
            packed_position_ids = self._build_qwen_position_ids(
                packed.packed_input_ids,
                packed.packed_attention_mask,
                packed_image_grid_thw,
            )
            if packed_position_ids is None:
                packed_position_ids = packed.packed_position_ids

        base_outputs = self.base_model(
            input_ids=packed.packed_input_ids,
            attention_mask=packed.packed_attention_mask,
            position_ids=packed_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=packed.packed_inputs_embeds,
            labels=packed.packed_labels,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=packed_image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            cache_position=cache_position,
            **kwargs,
        )
        if base_outputs.hidden_states is None:
            raise ValueError("Qwen2.5-VL forward did not return hidden states.")
        if base_outputs.loss is None:
            raise ValueError("Qwen2.5-VL forward did not return NTP loss.")

        hidden = base_outputs.hidden_states[-1]
        groups = self._build_refine_groups(
            hidden,
            packed.visual_block_token_ranges,
            frame_bin_ids.to(input_ids.device),
            frame_timestamps.to(input_ids.device),
            frame_labels.to(input_ids.device),
            frame_valid_mask.to(input_ids.device),
            gt_start_flat,
            gt_end_flat,
        )
        refine_output = self.time_refine_head(
            start_visual_groups=groups["start_visual_groups"],
            start_classification_groups=groups["start_classification_groups"],
            start_relative_time_groups=groups["start_relative_time_groups"],
            start_absolute_time_groups=groups["start_absolute_time_groups"],
            end_visual_groups=groups["end_visual_groups"],
            end_classification_groups=groups["end_classification_groups"],
            end_relative_time_groups=groups["end_relative_time_groups"],
            end_absolute_time_groups=groups["end_absolute_time_groups"],
        )
        predicted_span = torch.stack(
            [
                refine_output.pred_start_q / float(TIME_BIN_COUNT - 1),
                refine_output.pred_end_q / float(TIME_BIN_COUNT - 1),
            ],
            dim=-1,
        )
        target_span = torch.stack(
            [gt_start_flat / duration_flat, gt_end_flat / duration_flat],
            dim=-1,
        )
        diou_loss = diou_loss_1d(predicted_span, target_span)
        smooth_l1_loss = smooth_l1_boundary_loss(predicted_span, target_span)
        total_loss = (
            self.config.lambda_ntp * base_outputs.loss
            + self.config.lambda_diou * diou_loss
            + self.config.lambda_reg * smooth_l1_loss
        )
        self._last_refine_window_edge_cases = groups["edge_cases"].detach().cpu()
        if bool(groups["edge_cases"].any().item()):
            warnings.warn(
                "At least one training refinement window was clipped at the sampled "
                "temporal sequence edge.",
                RuntimeWarning,
            )

        pred_start = duration_flat * refine_output.pred_start_q / float(TIME_BIN_COUNT - 1)
        pred_end = duration_flat * refine_output.pred_end_q / float(TIME_BIN_COUNT - 1)
        return TimeLensRefineOutput(
            loss=total_loss,
            ntp_loss=base_outputs.loss,
            diou_loss=diou_loss,
            smooth_l1_loss=smooth_l1_loss,
            logits=base_outputs.logits,
            pred_start=pred_start,
            pred_end=pred_end,
            pred_start_q=refine_output.pred_start_q,
            pred_end_q=refine_output.pred_end_q,
            start_probs=refine_output.start_probs,
            end_probs=refine_output.end_probs,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
            past_key_values=base_outputs.past_key_values,
            window_edge_cases=groups["edge_cases"],
        )
