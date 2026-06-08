from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MethodType
from typing import Optional

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Trainer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.utils import is_torchdynamo_compiling

from .causal_swa_lact import LaCTCache, Qwen3VLLaCTSWIGLULayer
from .causal_swa_lact_streaming_chunked import (
    Qwen3VLLaCTSWIGLULayerStreamingChunked,
)

LACT_CONFIG_NAME = "lact_config.json"
STRICT_LACT_LAYERS = "0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26"
STRICT_LACT_LAYER_COUNT = 28

LACT_PARAM_KEYWORDS = (
    "w0",
    "w1",
    "w2",
    "lr_proj",
    "q_scale",
    "q_offset",
    "k_scale",
    "k_offset",
    "ttt_scale_proj",
    "ttt_norm",
    "momentum_proj",
)


@dataclass
class LaCTConfig:
    lact_enable: bool = True
    num_lact_heads: int = 4
    inter_multi: float = 1.0
    lact_chunk_size: int = 2648
    window_size: int = 2648
    qkv_silu: bool = True
    no_v_silu: bool = False
    use_muon: bool = True
    learnable_ttt_scale: bool = True
    w0_w2_low_rank: int = 0
    use_momentum: bool = True
    ttt_prenorm: bool = True
    ttt_nope: bool = False
    use_fused_kernel: bool = False
    fp32_states: bool = True
    use_conv_layer: bool = True
    lact_layers: Optional[str] = STRICT_LACT_LAYERS
    lact_lr: Optional[float] = 1e-5

    @classmethod
    def from_args(cls, args) -> "LaCTConfig":
        values = {}
        for field in fields(cls):
            if hasattr(args, field.name):
                values[field.name] = getattr(args, field.name)
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict) -> "LaCTConfig":
        valid = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in valid})

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_lact_layers(lact_layers: Optional[str]) -> Optional[str]:
    if lact_layers is None:
        return None
    return "/".join(str(int(x.strip())) for x in lact_layers.split("/") if x.strip())


def _parse_lact_layers(lact_layers: Optional[str], num_layers: int) -> Optional[set[int]]:
    normalized = _normalize_lact_layers(lact_layers)
    if normalized is None:
        return None

    if normalized == STRICT_LACT_LAYERS and num_layers != STRICT_LACT_LAYER_COUNT:
        raise ValueError(
            "Strict Spatial-TTT LaCT layer list expects "
            f"{STRICT_LACT_LAYER_COUNT} decoder layers, got {num_layers}."
        )

    indices = {int(x) for x in normalized.split("/") if x}
    invalid = sorted(i for i in indices if i < 0 or i >= num_layers)
    if invalid:
        raise ValueError(
            f"LaCT layer indices out of range for {num_layers} layers: {invalid}."
        )
    return indices


def _get_language_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    raise ValueError("Cannot find Qwen3-VL language model layers.")


def _resolve_checkpoint_for_dispatch(model_path: str) -> str:
    path = Path(model_path)
    if path.is_file():
        return str(path)
    if (path / "model.safetensors.index.json").exists():
        return str(path)
    if (path / "model.safetensors").exists():
        return str(path / "model.safetensors")
    if (path / "pytorch_model.bin.index.json").exists():
        return str(path)
    if (path / "pytorch_model.bin").exists():
        return str(path / "pytorch_model.bin")
    return str(path)


def save_lact_config(output_dir: str | Path, lact_config: LaCTConfig) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / LACT_CONFIG_NAME
    path.write_text(
        json.dumps(lact_config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_lact_config(
    model_path: str | Path,
    lact_config_path: Optional[str | Path] = None,
) -> LaCTConfig:
    config_path = Path(lact_config_path) if lact_config_path else Path(model_path) / LACT_CONFIG_NAME
    values = json.loads(config_path.read_text(encoding="utf-8"))
    return LaCTConfig.from_dict(values)


def is_lact_checkpoint(model_path: str | Path) -> bool:
    return (Path(model_path) / LACT_CONFIG_NAME).exists()


def _load_checkpoint_file(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def load_lact_checkpoint_for_training(
    model: Qwen3VLForConditionalGeneration,
    model_path: str | Path,
) -> Qwen3VLForConditionalGeneration:
    path = Path(model_path)
    checkpoint = Path(_resolve_checkpoint_for_dispatch(str(path)))

    if checkpoint.is_file():
        state_dict = _load_checkpoint_file(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        return model

    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        index_path = checkpoint / "pytorch_model.bin.index.json"

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    else:
        shard_names = sorted(p.name for p in checkpoint.glob("model-*.safetensors"))
        shard_names += sorted(p.name for p in checkpoint.glob("pytorch_model-*.bin"))

    if not shard_names:
        raise FileNotFoundError(f"No model checkpoint files found in {checkpoint}.")

    for shard_name in shard_names:
        state_dict = _load_checkpoint_file(checkpoint / shard_name)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
    return model


def _build_lact_layer(attn_layer, lact_config: LaCTConfig, inference: bool):
    layer_cls = (
        Qwen3VLLaCTSWIGLULayerStreamingChunked
        if inference
        else Qwen3VLLaCTSWIGLULayer
    )
    return layer_cls(
        attn_layer=attn_layer,
        num_lact_heads=lact_config.num_lact_heads,
        inter_multi=lact_config.inter_multi,
        window_size=lact_config.window_size,
        lact_chunk_size=lact_config.lact_chunk_size,
        qkv_silu=lact_config.qkv_silu,
        no_v_silu=lact_config.no_v_silu,
        use_muon=lact_config.use_muon,
        use_momentum=lact_config.use_momentum,
        learnable_ttt_scale=lact_config.learnable_ttt_scale,
        ttt_prenorm=lact_config.ttt_prenorm,
        ttt_nope=lact_config.ttt_nope,
        w0_w2_low_rank=lact_config.w0_w2_low_rank,
        use_fused_kernel=lact_config.use_fused_kernel,
        fp32_states=lact_config.fp32_states,
        use_conv_layer=lact_config.use_conv_layer,
    )


def _wrap_model_with_lact(
    model: Qwen3VLForConditionalGeneration,
    lact_config: LaCTConfig,
    inference: bool,
) -> Qwen3VLForConditionalGeneration:
    layers = _get_language_layers(model)
    lact_layer_indices = _parse_lact_layers(lact_config.lact_layers, len(layers))

    wrapped_count = 0
    for idx, layer in enumerate(layers):
        if lact_layer_indices is not None and idx not in lact_layer_indices:
            continue

        old_attn = layer.self_attn
        model_dtype = next(old_attn.parameters()).dtype
        model_device = next(old_attn.parameters()).device
        lact_layer = _build_lact_layer(old_attn, lact_config, inference=inference)

        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            lact_layer.rotary_emb = model.model.language_model.rotary_emb

        lact_layer = lact_layer.to(dtype=model_dtype, device=model_device)
        layer.self_attn = lact_layer
        wrapped_count += 1

    if wrapped_count == 0:
        raise ValueError("LaCT is enabled but no decoder layer was wrapped.")
    return model


def wrap_model_with_lact_for_training(
    model: Qwen3VLForConditionalGeneration,
    lact_config: LaCTConfig,
) -> Qwen3VLForConditionalGeneration:
    return _wrap_model_with_lact(model, lact_config, inference=False)


def wrap_model_with_lact_for_inference(
    model: Qwen3VLForConditionalGeneration,
    lact_config: LaCTConfig,
) -> Qwen3VLForConditionalGeneration:
    return _wrap_model_with_lact(model, lact_config, inference=True)


def get_lact_param_names(model) -> list[str]:
    return [
        name
        for name, _ in model.named_parameters()
        if any(keyword in name for keyword in LACT_PARAM_KEYWORDS)
    ]


def print_lact_parameters(model) -> None:
    total_params = 0
    trainable_params = 0
    lact_params = 0
    lact_trainable_params = 0
    for name, param in model.named_parameters():
        count = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
        if any(keyword in name for keyword in LACT_PARAM_KEYWORDS):
            lact_params += count
            if param.requires_grad:
                lact_trainable_params += count

    print("=" * 80)
    print("LaCT PARAMETER ANALYSIS")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LaCT parameters: {lact_params:,}")
    print(f"LaCT trainable parameters: {lact_trainable_params:,}")
    print("=" * 80)


def create_lact_optimizer(trainer, lact_lr: float):
    opt_model = trainer.model
    if trainer.optimizer is not None:
        return trainer.optimizer

    decay_parameters = trainer.get_decay_parameter_names(opt_model)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    lact_param_names = set(get_lact_param_names(opt_model))
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in opt_model.named_parameters()
                if n in decay_parameters and n not in lact_param_names and p.requires_grad
            ],
            "weight_decay": trainer.args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in opt_model.named_parameters()
                if n not in decay_parameters and n not in lact_param_names and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in opt_model.named_parameters()
                if n in decay_parameters and n in lact_param_names and p.requires_grad
            ],
            "weight_decay": trainer.args.weight_decay,
            "lr": lact_lr,
        },
        {
            "params": [
                p
                for n, p in opt_model.named_parameters()
                if n not in decay_parameters and n in lact_param_names and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lact_lr,
        },
    ]
    optimizer_grouped_parameters = [
        group for group in optimizer_grouped_parameters if len(group["params"]) > 0
    ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)
    trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return trainer.optimizer


def qwen3vl_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    cache_position=None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
        video_embeds, deepstack_video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw
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
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(
                visual_pos_masks.sum(), img_embed.shape[-1]
            ).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
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
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
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
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    kwargs["video_grid_thw"] = video_grid_thw
    kwargs["video_mask"] = video_mask

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


def patch_qwen3vl_forward(model: Qwen3VLForConditionalGeneration) -> None:
    model.model.forward = MethodType(qwen3vl_forward, model.model)


class SpatialTTTForConditionalGeneration:
    def __init__(self, model: Qwen3VLForConditionalGeneration):
        self.model = model.model
        self.lm_head = model.lm_head
        self._model = model

    def __getattr__(self, name):
        if name in ("model", "lm_head", "_model"):
            return object.__getattribute__(self, name)
        return getattr(self._model, name)

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        self.model = self._model.model
        self.lm_head = self._model.lm_head
        return self

    def eval(self):
        self._model.eval()
        return self

    def train(self, mode=True):
        self._model.train(mode)
        return self

    @torch.no_grad()
    def generate_with_spatial_ttt(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        del pad_token_id
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError("SpatialTTT generation currently supports batch_size=1.")

        device = input_ids.device
        qwen_model = self._model
        inputs_embeds = qwen_model.model.language_model.embed_tokens(input_ids)
        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = qwen_model.model.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = qwen_model.model.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = qwen_model.model.get_placeholder_mask(
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
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        model = self.model.language_model
        rotary_emb = model.rotary_emb
        norm = model.norm
        lm_head = self.lm_head

        hidden_states = inputs_embeds
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw
        )
        self.rope_deltas = rope_deltas
        position_embeddings = rotary_emb(hidden_states, position_ids)

        lact_cache = LaCTCache()
        past_key_values = DynamicCache()
        cache_position = torch.tensor([0], device=device)
        seq_len = hidden_states.shape[1]

        for layer_idx, layer in enumerate(model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if isinstance(
                layer.self_attn,
                (Qwen3VLLaCTSWIGLULayer, Qwen3VLLaCTSWIGLULayerStreamingChunked),
            ):
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    lact_cache=lact_cache,
                    video_mask=video_mask,
                    video_grid_thw=video_grid_thw,
                )
            else:
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    attention_mask=None,
                )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                hidden_states = model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states[:, -1:, :])
        next_token = self._sample_token(logits, temperature, top_k, top_p, do_sample)
        generated_tokens = [next_token]
        embed_tokens = model.embed_tokens

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            hidden_states = embed_tokens(next_token)
            position_ids_step = (
                torch.tensor([seq_len], device=device) + rope_deltas
            ).to(hidden_states.device)
            position_ids_step = position_ids_step.unsqueeze(0).expand(3, -1, -1)
            position_embeddings = rotary_emb(hidden_states, position_ids_step)
            cache_position = torch.tensor([seq_len], device=device)

            for layer in model.layers:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                if isinstance(
                    layer.self_attn,
                    (Qwen3VLLaCTSWIGLULayer, Qwen3VLLaCTSWIGLULayerStreamingChunked),
                ):
                    hidden_states, _ = layer.self_attn(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        lact_cache=lact_cache,
                        video_mask=None,
                        video_grid_thw=video_grid_thw,
                    )
                else:
                    hidden_states, _ = layer.self_attn(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        attention_mask=None,
                    )
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            hidden_states = norm(hidden_states)
            logits = lm_head(hidden_states)
            next_token = self._sample_token(
                logits, temperature, top_k, top_p, do_sample
            )
            generated_tokens.append(next_token)
            seq_len += 1

        return torch.cat([input_ids] + generated_tokens, dim=1)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        logits = logits[:, -1, :]
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def load_lact_model_for_inference(
    model_path: str,
    lact_config_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    attn_implementation: str = "flash_attention_2",
) -> SpatialTTTForConditionalGeneration:
    lact_config = load_lact_config(model_path, lact_config_path)
    config = AutoConfig.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    model = wrap_model_with_lact_for_inference(model, lact_config)
    checkpoint = _resolve_checkpoint_for_dispatch(model_path)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint,
        device_map=device_map,
        dtype=torch_dtype,
    )
    return SpatialTTTForConditionalGeneration(model).eval()


def load_lact_model_for_training(
    model_path: str,
    lact_config: LaCTConfig,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
) -> Qwen3VLForConditionalGeneration:
    config = AutoConfig.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model = Qwen3VLForConditionalGeneration(config)
    model = wrap_model_with_lact_for_training(model, lact_config)
    model = load_lact_checkpoint_for_training(model, model_path)
    return model
