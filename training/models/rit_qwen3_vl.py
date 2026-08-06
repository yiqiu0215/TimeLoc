import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from transformers.utils import is_torchdynamo_compiling


class ContinuousTimePositionEmbedding(nn.Module):
    def __init__(self, time_embedding_dim: int, hidden_size: int):
        super().__init__()
        if time_embedding_dim <= 0 or time_embedding_dim % 2 != 0:
            raise ValueError("time_embedding_dim must be a positive even integer.")

        exponent = torch.arange(0, time_embedding_dim, 2, dtype=torch.float32)
        exponent = exponent / time_embedding_dim
        self.register_buffer(
            "inv_freq", torch.exp(-math.log(10000.0) * exponent), persistent=False
        )
        self.projection = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, hidden_size),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.to(device=self.inv_freq.device, dtype=torch.float32)
        angles = timestamps[:, None] * self.inv_freq[None, :]
        features = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        features = features.to(dtype=self.projection[0].weight.dtype)
        return self.projection(features)


class RITQwen3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.residual_num_diffs = int(getattr(config, "residual_num_diffs", 4))
        if self.residual_num_diffs <= 0:
            raise ValueError("residual_num_diffs must be positive.")
        self.residual_norm = nn.LayerNorm(config.hidden_size)
        self.residual_gate = nn.Parameter(
            torch.tensor(float(getattr(config, "residual_gate_init", 0.1)))
        )
        self.residual_modality_embedding = nn.Parameter(
            torch.zeros(config.hidden_size)
        )
        self.time_position_embedding = ContinuousTimePositionEmbedding(
            time_embedding_dim=int(getattr(config, "time_embedding_dim", 128)),
            hidden_size=config.hidden_size,
        )

    @staticmethod
    def _split_sizes(grid_thw: torch.Tensor) -> list[int]:
        return [int(value) for value in grid_thw.prod(dim=-1).tolist()]

    def _embed_residuals_with_shared_patch(
        self, packed_patches: torch.Tensor
    ) -> torch.Tensor:
        expected_width = 3 * self.patch_size**2
        if packed_patches.shape[-1] != expected_width:
            raise ValueError(
                f"Unexpected residual patch width: {packed_patches.shape[-1]} "
                f"!= {expected_width}."
            )
        if packed_patches.shape[0] == 0:
            return packed_patches.new_empty(
                (0, self.residual_modality_embedding.numel())
            )

        residual_frame = packed_patches.reshape(
            -1, 3, self.patch_size, self.patch_size
        )
        duplicated_residual = residual_frame.unsqueeze(2).expand(
            -1, -1, 2, -1, -1
        )
        duplicated_residual = duplicated_residual.reshape(
            -1, 3 * 2 * self.patch_size**2
        )
        residual_embeddings = self.patch_embed(duplicated_residual)
        residual_embeddings = self.residual_gate * self.residual_norm(
            residual_embeddings
        )
        return residual_embeddings + self.residual_modality_embedding.to(
            residual_embeddings.dtype
        )

    def _interleave_embeddings(
        self,
        rgb_embeddings: torch.Tensor,
        residual_embeddings: torch.Tensor,
        rgb_grid_thw: torch.Tensor,
        residual_grid_thw: torch.Tensor,
        temporal_midpoints: torch.Tensor,
    ) -> torch.Tensor:
        rgb_chunks = torch.split(rgb_embeddings, self._split_sizes(rgb_grid_thw))
        residual_chunks = torch.split(
            residual_embeddings, self._split_sizes(residual_grid_thw)
        )

        interleaved_videos = []
        midpoint_offset = 0
        for rgb_chunk, residual_chunk, rgb_grid, residual_grid in zip(
            rgb_chunks, residual_chunks, rgb_grid_thw, residual_grid_thw
        ):
            rgb_t, rgb_h, rgb_w = [int(value) for value in rgb_grid.tolist()]
            residual_t, residual_h, residual_w = [
                int(value) for value in residual_grid.tolist()
            ]
            if residual_t != max(rgb_t - 1, 0):
                raise ValueError(
                    f"Residual block count must be K-1, got K={rgb_t}, R={residual_t}."
                )
            if (rgb_h, rgb_w) != (residual_h, residual_w):
                raise ValueError("RGB and residual spatial grids must match.")

            patches_per_block = rgb_h * rgb_w
            rgb_chunk = rgb_chunk.reshape(rgb_t, patches_per_block, -1)
            residual_chunk = residual_chunk.reshape(
                residual_t, patches_per_block, -1
            )
            blocks = []
            for block_index in range(rgb_t):
                blocks.append(rgb_chunk[block_index])
                if block_index < residual_t:
                    blocks.append(residual_chunk[block_index])
            video_embeddings = torch.cat(blocks, dim=0)

            num_midpoints = 2 * rgb_t - 1
            current_midpoints = temporal_midpoints[
                midpoint_offset : midpoint_offset + num_midpoints
            ]
            if current_midpoints.numel() != num_midpoints:
                raise ValueError(
                    "temporal_midpoints does not match the interleaved block count."
                )
            midpoint_offset += num_midpoints
            time_embeddings = self.time_position_embedding(current_midpoints)
            time_embeddings = time_embeddings.repeat_interleave(
                patches_per_block, dim=0
            ).to(video_embeddings.dtype)
            interleaved_videos.append(video_embeddings + time_embeddings)

        if midpoint_offset != temporal_midpoints.numel():
            raise ValueError("Unused temporal midpoints remain after video interleaving.")
        return torch.cat(interleaved_videos, dim=0)

    def forward_interleaved(
        self,
        pixel_values_videos: torch.Tensor,
        pixel_values_residuals: torch.Tensor,
        rgb_video_grid_thw: torch.Tensor,
        residual_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor,
        temporal_midpoints: torch.Tensor,
        **kwargs,
    ):
        rgb_embeddings = self.patch_embed(pixel_values_videos)
        residual_embeddings = self._embed_residuals_with_shared_patch(
            pixel_values_residuals
        )
        hidden_states = self._interleave_embeddings(
            rgb_embeddings,
            residual_embeddings,
            rgb_video_grid_thw,
            residual_grid_thw,
            temporal_midpoints,
        )

        expected_tokens = int(video_grid_thw.prod(dim=-1).sum().item())
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                "Interleaved ViT token count mismatch: "
                f"{hidden_states.shape[0]} != {expected_tokens}."
            )

        hidden_states = hidden_states + self.fast_pos_embed_interpolate(video_grid_thw)
        rotary_pos_emb = self.rot_pos_emb(video_grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        rotary_embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_embedding.cos(), rotary_embedding.sin())

        cu_seqlens = torch.repeat_interleave(
            video_grid_thw[:, 1] * video_grid_thw[:, 2], video_grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=video_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )

        return self.merger(hidden_states), deepstack_feature_lists


class RITQwen3VLModel(Qwen3VLModel):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = RITQwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_interleaved_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        pixel_values_residuals: torch.Tensor,
        rgb_video_grid_thw: torch.Tensor,
        residual_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor,
        temporal_midpoints: torch.Tensor,
    ):
        pixel_values_videos = pixel_values_videos.to(dtype=self.visual.dtype)
        pixel_values_residuals = pixel_values_residuals.to(dtype=self.visual.dtype)
        video_embeds, deepstack_video_embeds = self.visual.forward_interleaved(
            pixel_values_videos=pixel_values_videos,
            pixel_values_residuals=pixel_values_residuals,
            rgb_video_grid_thw=rgb_video_grid_thw,
            residual_grid_thw=residual_grid_thw,
            video_grid_thw=video_grid_thw,
            temporal_midpoints=temporal_midpoints,
        )
        split_sizes = (
            video_grid_thw.prod(dim=-1) // self.visual.spatial_merge_size**2
        ).tolist()
        return torch.split(video_embeds, split_sizes), deepstack_video_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pixel_values_residuals: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rgb_video_grid_thw: Optional[torch.LongTensor] = None,
        residual_grid_thw: Optional[torch.LongTensor] = None,
        temporal_midpoints: Optional[torch.FloatTensor] = None,
        rgb_temporal_midpoints: Optional[torch.FloatTensor] = None,
        residual_temporal_midpoints: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        del rgb_temporal_midpoints, residual_temporal_midpoints
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            required = {
                "pixel_values_residuals": pixel_values_residuals,
                "rgb_video_grid_thw": rgb_video_grid_thw,
                "residual_grid_thw": residual_grid_thw,
                "video_grid_thw": video_grid_thw,
                "temporal_midpoints": temporal_midpoints,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"Missing RIT video inputs: {', '.join(missing)}")
            video_embeds, deepstack_video_embeds = (
                self.get_interleaved_video_features(
                    pixel_values_videos,
                    pixel_values_residuals,
                    rgb_video_grid_thw,
                    residual_grid_thw,
                    video_grid_thw,
                    temporal_midpoints,
                )
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for image_embed, video_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                joint = image_embed.new_zeros(
                    visual_pos_masks.sum(), image_embed.shape[-1]
                )
                joint[image_mask_joint] = image_embed
                joint[video_mask_joint] = video_embed
                deepstack_visual_embeds.append(joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(
                        batch_size // delta.shape[0], dim=0
                    )
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class RITQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        if getattr(config, "use_residual_tokens", False) and getattr(
            config, "rit_architecture_version", None
        ) != "shared_rgb_patch_accumulate_v2":
            raise ValueError(
                "This checkpoint is incompatible with the shared RGB patch embedding architecture."
            )
        defaults = {
            "use_residual_tokens": True,
            "rit_architecture_version": "shared_rgb_patch_accumulate_v2",
            "residual_num_diffs": 4,
            "residual_in_channels": 3,
            "residual_gate_init": 0.1,
            "time_embedding_dim": 128,
            "use_true_midpoint_time_embedding": True,
            "combined_visual_token_budget": 14336,
        }
        for name, value in defaults.items():
            if not hasattr(config, name):
                setattr(config, name, value)
            if not hasattr(config.vision_config, name):
                setattr(config.vision_config, name, getattr(config, name))
        config.architectures = [self.__class__.__name__]

        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = RITQwen3VLModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pixel_values_residuals: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rgb_video_grid_thw: Optional[torch.LongTensor] = None,
        residual_grid_thw: Optional[torch.LongTensor] = None,
        temporal_midpoints: Optional[torch.FloatTensor] = None,
        rgb_temporal_midpoints: Optional[torch.FloatTensor] = None,
        residual_temporal_midpoints: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            pixel_values_residuals=pixel_values_residuals,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rgb_video_grid_thw=rgb_video_grid_thw,
            residual_grid_thw=residual_grid_thw,
            temporal_midpoints=temporal_midpoints,
            rgb_temporal_midpoints=rgb_temporal_midpoints,
            residual_temporal_midpoints=residual_temporal_midpoints,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        pixel_values_residuals=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rgb_video_grid_thw=None,
        residual_grid_thw=None,
        temporal_midpoints=None,
        rgb_temporal_midpoints=None,
        residual_temporal_midpoints=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        residual_inputs = {
            "pixel_values_residuals": pixel_values_residuals,
            "rgb_video_grid_thw": rgb_video_grid_thw,
            "residual_grid_thw": residual_grid_thw,
            "temporal_midpoints": temporal_midpoints,
            "rgb_temporal_midpoints": rgb_temporal_midpoints,
            "residual_temporal_midpoints": residual_temporal_midpoints,
        }
        if cache_position[0] != 0:
            residual_inputs = {name: None for name in residual_inputs}
        model_inputs.update(residual_inputs)
        return model_inputs
