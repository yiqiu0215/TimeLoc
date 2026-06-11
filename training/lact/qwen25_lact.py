from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
    apply_multimodal_rotary_pos_emb,
)
from transformers.utils import is_torchdynamo_compiling

from .config import LACT_PARAM_KEYWORDS, LaCTConfig
from .ttt_operation import (
    block_causal_lact_swiglu,
    l2_norm,
    prenorm_block_causal_lact_swiglu,
    silu_backprop,
    zeropower_via_newtonschulz5,
)


@dataclass
class LaCTLayerState:
    w0: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w0_norm: torch.Tensor
    w1_norm: torch.Tensor
    w2_norm: torch.Tensor
    dw0_momentum: Optional[torch.Tensor] = None
    dw1_momentum: Optional[torch.Tensor] = None
    dw2_momentum: Optional[torch.Tensor] = None
    pending_k: Optional[torch.Tensor] = None
    pending_v: Optional[torch.Tensor] = None
    pending_lr0: Optional[torch.Tensor] = None
    pending_lr1: Optional[torch.Tensor] = None
    pending_lr2: Optional[torch.Tensor] = None
    pending_momentum: Optional[torch.Tensor] = None


class LaCTCache:
    def __init__(self):
        self._layer_states: dict[int, LaCTLayerState] = {}

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._layer_states

    def get_layer_state(self, layer_idx: int) -> LaCTLayerState:
        return self._layer_states[layer_idx]

    def set_layer_state(self, layer_idx: int, state: LaCTLayerState) -> None:
        self._layer_states[layer_idx] = state

    def reset(self) -> None:
        self._layer_states.clear()


@dataclass
class _VideoIndex:
    token_positions: torch.Tensor
    tokens_per_frame: int
    frames_per_video: int
    num_videos: int
    h: int
    w: int


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    return x + math.log(-math.expm1(-x))


def _set_kv_cache_layer(past_key_values, layer_idx: int, key_states, value_states):
    if past_key_values is None:
        return

    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        layer.keys = key_states
        layer.values = value_states
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is None or value_cache is None:
        return
    while len(key_cache) < layer_idx:
        key_cache.append(torch.tensor([]))
        value_cache.append(torch.tensor([]))
    if len(key_cache) == layer_idx:
        key_cache.append(key_states)
        value_cache.append(value_states)
    else:
        key_cache[layer_idx] = key_states
        value_cache[layer_idx] = value_states


def _truncate_kv_cache_layer(
    past_key_values,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if key_states.shape[2] <= window_size:
        return key_states, value_states
    key_states = key_states[:, :, -window_size:, :]
    value_states = value_states[:, :, -window_size:, :]
    _set_kv_cache_layer(past_key_values, layer_idx, key_states, value_states)
    return key_states, value_states


class LowRankFastWeight(nn.Module):
    def __init__(
        self,
        num_heads: int,
        out_features: int,
        in_features: int,
        rank: int = 32,
        init_gain: float = 0.5,
        add_identity: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.init_gain = init_gain
        self.add_identity = add_identity
        self.w_left = nn.Parameter(torch.empty(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.empty(num_heads, rank, in_features))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(
            self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain
        )

    def forward(self) -> torch.Tensor:
        weight = self.w_left @ self.w_right
        if self.add_identity:
            eye = torch.eye(
                self.out_features,
                self.in_features,
                device=weight.device,
                dtype=weight.dtype,
            )
            weight = weight + eye.unsqueeze(0) * 0.5
        return weight


class Qwen2_5VLLaCTSWIGLULayer(nn.Module):
    def __init__(
        self,
        attn_layer: nn.Module,
        num_lact_heads: int,
        inter_multi: float = 1.0,
        window_size: int = 2648,
        lact_chunk_size: int = 2648,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        use_muon: bool = True,
        use_momentum: bool = True,
        learnable_ttt_scale: bool = True,
        ttt_prenorm: bool = True,
        ttt_nope: bool = False,
        w0_w2_low_rank: int = 0,
        use_fused_kernel: bool = False,
        fp32_states: bool = True,
        fw_init_gain: float = 0.5,
        use_conv_layer: bool = True,
    ):
        super().__init__()
        if use_fused_kernel:
            raise NotImplementedError(
                "Qwen2.5-TimeLens LaCT currently supports use_fused_kernel=False only."
            )
        self.attn_layer = attn_layer
        self.config = attn_layer.config
        self.hidden_size = self.config.hidden_size
        self.num_attn_heads = attn_layer.num_heads
        self.num_kv_heads = attn_layer.num_key_value_heads
        self.num_key_value_groups = attn_layer.num_key_value_groups
        self.head_dim = attn_layer.head_dim
        self.layer_idx = attn_layer.layer_idx
        self.scaling = attn_layer.scaling

        self.window_size = window_size
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        if self.hidden_size % self.num_fw_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_lact_heads={self.num_fw_heads}."
            )
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        if self.fw_head_dim < self.head_dim:
            raise ValueError(
                f"fw_head_dim={self.fw_head_dim} must be >= attention head_dim={self.head_dim}."
            )

        self.inter_multi = inter_multi
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.use_muon = use_muon
        self.use_momentum = use_momentum
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fp32_states = fp32_states
        self.use_conv_layer = use_conv_layer

        d_in = self.fw_head_dim
        d_out = self.fw_head_dim
        d_h = int(d_in * inter_multi)

        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(
                self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, fw_init_gain, True
            )
            self.w2 = LowRankFastWeight(
                self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, fw_init_gain, True
            )
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in)
            )
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in)
            )
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, d_out, d_h) / math.sqrt(d_h)
        )

        self.lr_dim = 3 * self.num_fw_heads
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        self.lr_parameterization = "mamba"
        self.base_lr_inv = inv_softplus(0.001)

        self.q_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.q_offset = nn.Parameter(torch.zeros(self.hidden_size))
        self.k_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.k_offset = nn.Parameter(torch.zeros(self.hidden_size))
        self.ttt_norm = nn.RMSNorm(self.fw_head_dim, eps=1e-5, elementwise_affine=True)

        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(self.hidden_size, self.num_fw_heads)
            nn.init.zeros_(self.ttt_scale_proj.weight)
            nn.init.zeros_(self.ttt_scale_proj.bias)

        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        if self.use_conv_layer:
            self.conv_q = nn.Conv3d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=3,
                padding=1,
                groups=self.hidden_size,
                padding_mode="replicate",
                bias=False,
            )
            self.conv_k = nn.Conv3d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=3,
                padding=1,
                groups=self.hidden_size,
                padding_mode="replicate",
                bias=False,
            )
            self.conv_v = nn.Conv3d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=3,
                padding=1,
                groups=self.hidden_size,
                padding_mode="replicate",
                bias=False,
            )
            nn.init.dirac_(self.conv_q.weight, groups=self.hidden_size)
            nn.init.dirac_(self.conv_k.weight, groups=self.hidden_size)
            nn.init.dirac_(self.conv_v.weight, groups=self.hidden_size)

    def _rescale_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q * self.q_scale.view(1, 1, -1) + self.q_offset.view(1, 1, -1)
        k = k * self.k_scale.view(1, 1, -1) + self.k_offset.view(1, 1, -1)
        return q, k

    def _expand_kv_flat(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        if self.num_key_value_groups > 1:
            x = x.unsqueeze(3).expand(
                -1, -1, -1, self.num_key_value_groups, -1
            )
        return x.reshape(batch_size, seq_len, self.hidden_size)

    def _compute_qkv(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.attn_layer.q_proj(hidden_states)
        k = self.attn_layer.k_proj(hidden_states)
        v = self.attn_layer.v_proj(hidden_states)

        query_states = q.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = k.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        value_states = v.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        k_flat = self._expand_kv_flat(k)
        v_expanded = self._expand_kv_flat(v)
        q_flat, k_flat = self._rescale_qk(q, k_flat)
        return query_states, key_states, value_states, q_flat, k_flat, v_expanded

    def _apply_partial_mrope(
        self,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = position_embeddings
        batch_fw_heads = fast_q.shape[0]
        if batch_fw_heads % self.num_fw_heads != 0:
            raise RuntimeError(
                f"Invalid fast head shape {fast_q.shape}; num_fw_heads={self.num_fw_heads}."
            )
        batch_size = batch_fw_heads // self.num_fw_heads
        q = rearrange(fast_q, "(b h) s d -> b h s d", b=batch_size, h=self.num_fw_heads)
        k = rearrange(fast_k, "(b h) s d -> b h s d", b=batch_size, h=self.num_fw_heads)

        q_rope, q_nope = q[..., : self.head_dim], q[..., self.head_dim :]
        k_rope, k_nope = k[..., : self.head_dim], k[..., self.head_dim :]
        q_rope, k_rope = apply_multimodal_rotary_pos_emb(
            q_rope,
            k_rope,
            cos,
            sin,
            self.config.rope_scaling["mrope_section"],
            unsqueeze_dim=1,
        )
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)
        return (
            rearrange(q, "b h s d -> (b h) s d"),
            rearrange(k, "b h s d -> (b h) s d"),
        )

    def _prepare_fast_qkv(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_expanded: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast_q = rearrange(q_flat, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_k = rearrange(k_flat, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_v = rearrange(v_expanded, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)
        if not self.ttt_nope:
            fast_q, fast_k = self._apply_partial_mrope(
                fast_q, fast_k, position_embeddings
            )
        return fast_q, fast_k, fast_v

    def _build_video_index(
        self,
        video_mask: torch.Tensor,
        video_grid_thw: torch.Tensor,
        seq_len: int,
    ) -> _VideoIndex:
        if video_mask.dim() == 3:
            video_mask = video_mask[..., 0]
        if video_mask.dim() != 2 or video_mask.shape[0] != 1:
            raise RuntimeError(
                "LaCT conv currently expects batch_size=1 video_mask with shape [1, seq_len]."
            )
        if video_grid_thw is None or video_grid_thw.numel() == 0:
            raise RuntimeError("video_grid_thw required when use_conv_layer=True.")
        if video_grid_thw.shape[0] != 1:
            raise RuntimeError(
                "LaCT conv first version supports one video per sample. "
                f"Got video_grid_thw shape {tuple(video_grid_thw.shape)}."
            )

        token_positions = torch.nonzero(video_mask[0], as_tuple=False).flatten()
        if token_positions.numel() == 0:
            raise RuntimeError("use_conv_layer=True but no video tokens were found.")

        t, h, w = video_grid_thw[0]
        h = int(h.item()) // 2
        w = int(w.item()) // 2
        frames_per_video = int(t.item())
        tokens_per_frame = h * w
        if tokens_per_frame <= 0:
            raise RuntimeError(f"Invalid video_grid_thw={video_grid_thw.tolist()}.")

        num_video_tokens = int(token_positions.numel())
        expected_tokens = frames_per_video * tokens_per_frame
        if num_video_tokens != expected_tokens:
            raise RuntimeError(
                "video_mask/video_grid_thw mismatch for LaCT conv: "
                f"video tokens={num_video_tokens}, expected={expected_tokens}, "
                f"grid={video_grid_thw.tolist()}, seq_len={seq_len}."
            )

        return _VideoIndex(
            token_positions=token_positions,
            tokens_per_frame=tokens_per_frame,
            frames_per_video=frames_per_video,
            num_videos=1,
            h=h,
            w=w,
        )

    def _apply_full_video_conv(
        self,
        x: torch.Tensor,
        conv_layer: nn.Module,
        video_index: _VideoIndex,
    ) -> torch.Tensor:
        video_tokens = x[0, video_index.token_positions, :]
        video_reshaped = rearrange(
            video_tokens,
            "(n t h w) c -> n c t h w",
            n=video_index.num_videos,
            t=video_index.frames_per_video,
            h=video_index.h,
            w=video_index.w,
        )
        video_conv = conv_layer(video_reshaped)
        x = x.clone()
        x[0, video_index.token_positions, :] = rearrange(
            video_conv, "n c t h w -> (n t h w) c"
        )
        return x

    def _run_ttt_kernel(
        self,
        fw_w0: torch.Tensor,
        fw_w1: torch.Tensor,
        fw_w2: torch.Tensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        fw_lr1: torch.Tensor,
        fw_lr2: torch.Tensor,
        fw_lr3: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fn = prenorm_block_causal_lact_swiglu if self.ttt_prenorm else block_causal_lact_swiglu
        return fn(
            fw_w0,
            fw_w1,
            fw_w2,
            fast_q,
            fast_k,
            fast_v,
            fw_lr1,
            fw_lr2,
            fw_lr3,
            chunk_size=self.lact_chunk_size,
            use_muon=self.use_muon,
            momentum=momentum,
        )

    def _base_fast_weights(self, batch_size: int):
        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(batch_size, 1, 1)
            fw_w2 = self.w2.repeat(batch_size, 1, 1)
        fw_w1 = self.w1.repeat(batch_size, 1, 1)
        if self.fp32_states:
            fw_w0 = fw_w0.float()
            fw_w1 = fw_w1.float()
            fw_w2 = fw_w2.float()
        return fw_w0, fw_w1, fw_w2

    def _compute_ttt_output(
        self,
        hidden_states: torch.Tensor,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_expanded: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        fast_q, fast_k, fast_v = self._prepare_fast_qkv(
            q_flat, k_flat, v_expanded, position_embeddings
        )
        lr = self.lr_proj(hidden_states)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s h -> (b h) s 1", h=self.num_fw_heads
            )
        else:
            momentum = None

        fw_w0, fw_w1, fw_w2 = self._base_fast_weights(batch_size)
        fw_x = self._run_ttt_kernel(
            fw_w0,
            fw_w1,
            fw_w2,
            fast_q,
            fast_k,
            fast_v,
            fw_lr1,
            fw_lr2,
            fw_lr3,
            momentum,
        )

        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s h -> (b h) s 1", h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale
        return rearrange(
            ttt_x_normed,
            "(b h) s d -> b s (h d)",
            h=self.num_fw_heads,
            b=batch_size,
        ).type_as(hidden_states)

    def _init_cache_from_prefill(
        self,
        hidden_states: torch.Tensor,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_expanded: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        lact_cache: LaCTCache,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "LaCT decode cache supports batch_size=1 only."
        fast_q, fast_k, fast_v = self._prepare_fast_qkv(
            q_flat, k_flat, v_expanded, position_embeddings
        )
        lr = self.lr_proj(hidden_states)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s h -> (b h) s 1", h=self.num_fw_heads
            )
        else:
            momentum = None

        fw_w0, fw_w1, fw_w2 = self._base_fast_weights(batch_size)
        w0_norm = fw_w0.norm(dim=2, keepdim=True)
        w1_norm = fw_w1.norm(dim=2, keepdim=True)
        w2_norm = fw_w2.norm(dim=2, keepdim=True)
        dw0_momentum = torch.zeros_like(fw_w0) if momentum is not None else None
        dw1_momentum = torch.zeros_like(fw_w1) if momentum is not None else None
        dw2_momentum = torch.zeros_like(fw_w2) if momentum is not None else None

        q_t = fast_q.transpose(1, 2)
        v_t = fast_v.transpose(1, 2)
        output = torch.zeros_like(v_t)
        if self.fp32_states:
            q_t = q_t.float()
            v_t = v_t.float()
            fast_k = fast_k.float()
            output = output.float()

        e_index = 0
        for s_index in range(0, seq_len - self.lact_chunk_size, self.lact_chunk_size):
            e_index = s_index + self.lact_chunk_size
            ki = fast_k[:, s_index:e_index, :]
            vi = v_t[:, :, s_index:e_index]
            qi = q_t[:, :, s_index:e_index]
            lr0i = fw_lr1[:, s_index:e_index, :]
            lr1i = fw_lr2[:, s_index:e_index, :]
            lr2i = fw_lr3[:, s_index:e_index, :]

            h = torch.bmm(fw_w2, qi)
            gate = F.silu(torch.bmm(fw_w0, qi), inplace=True)
            output[:, :, s_index:e_index] = torch.bmm(fw_w1, gate * h)

            gate_before_act = torch.bmm(fw_w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(fw_w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(fw_w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

            if momentum is not None:
                m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)
                dw0 = dw0 + dw0_momentum * m_i
                dw1 = dw1 + dw1_momentum * m_i
                dw2 = dw2 + dw2_momentum * m_i
                dw0_momentum = dw0
                dw1_momentum = dw1
                dw2_momentum = dw2

            if self.use_muon:
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

            fw_w0 = fw_w0 + dw0
            fw_w1 = fw_w1 + dw1
            fw_w2 = fw_w2 + dw2
            fw_w0 = fw_w0 / (fw_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            fw_w1 = fw_w1 / (fw_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            fw_w2 = fw_w2 / (fw_w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        s_index = e_index
        e_index = seq_len
        remaining_len = e_index - s_index
        if remaining_len > 0:
            qi = q_t[:, :, s_index:e_index]
            h = torch.bmm(fw_w2, qi)
            gate = F.silu(torch.bmm(fw_w0, qi), inplace=True)
            output[:, :, s_index:e_index] = torch.bmm(fw_w1, gate * h)

        state = LaCTLayerState(
            w0=fw_w0,
            w1=fw_w1,
            w2=fw_w2,
            w0_norm=w0_norm,
            w1_norm=w1_norm,
            w2_norm=w2_norm,
            dw0_momentum=dw0_momentum,
            dw1_momentum=dw1_momentum,
            dw2_momentum=dw2_momentum,
            pending_k=fast_k[:, s_index:e_index, :] if remaining_len > 0 else None,
            pending_v=fast_v[:, s_index:e_index, :] if remaining_len > 0 else None,
            pending_lr0=fw_lr1[:, s_index:e_index, :] if remaining_len > 0 else None,
            pending_lr1=fw_lr2[:, s_index:e_index, :] if remaining_len > 0 else None,
            pending_lr2=fw_lr3[:, s_index:e_index, :] if remaining_len > 0 else None,
            pending_momentum=momentum[:, s_index:e_index, :] if momentum is not None and remaining_len > 0 else None,
        )
        lact_cache.set_layer_state(self.layer_idx, state)

        fw_x = output.transpose(1, 2)
        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s h -> (b h) s 1", h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale
        return rearrange(
            ttt_x_normed, "(b h) s d -> b s (h d)", h=self.num_fw_heads
        ).type_as(hidden_states)

    def _compute_ttt_output_decode(
        self,
        hidden_states: torch.Tensor,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_expanded: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        lact_cache: LaCTCache,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1 and seq_len == 1
        state = lact_cache.get_layer_state(self.layer_idx)
        fast_q, fast_k, fast_v = self._prepare_fast_qkv(
            q_flat, k_flat, v_expanded, position_embeddings
        )
        lr = self.lr_proj(hidden_states)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s h -> (b h) s 1", h=self.num_fw_heads
            )
        else:
            momentum = None

        q_t = fast_q.transpose(1, 2)
        if self.fp32_states:
            q_t = q_t.float()
            fast_k = fast_k.float()
            fast_v = fast_v.float()
        h = torch.bmm(state.w2, q_t)
        gate = F.silu(torch.bmm(state.w0, q_t), inplace=True)
        fw_x = torch.bmm(state.w1, gate * h).transpose(1, 2)

        state.pending_k = fast_k if state.pending_k is None else torch.cat([state.pending_k, fast_k], dim=1)
        state.pending_v = fast_v if state.pending_v is None else torch.cat([state.pending_v, fast_v], dim=1)
        state.pending_lr0 = fw_lr1 if state.pending_lr0 is None else torch.cat([state.pending_lr0, fw_lr1], dim=1)
        state.pending_lr1 = fw_lr2 if state.pending_lr1 is None else torch.cat([state.pending_lr1, fw_lr2], dim=1)
        state.pending_lr2 = fw_lr3 if state.pending_lr2 is None else torch.cat([state.pending_lr2, fw_lr3], dim=1)
        if momentum is not None:
            state.pending_momentum = (
                momentum
                if state.pending_momentum is None
                else torch.cat([state.pending_momentum, momentum], dim=1)
            )

        pending_len = state.pending_k.shape[1]
        if pending_len >= self.lact_chunk_size:
            ki = state.pending_k[:, : self.lact_chunk_size, :]
            vi = state.pending_v[:, : self.lact_chunk_size, :].transpose(1, 2)
            lr0i = state.pending_lr0[:, : self.lact_chunk_size, :]
            lr1i = state.pending_lr1[:, : self.lact_chunk_size, :]
            lr2i = state.pending_lr2[:, : self.lact_chunk_size, :]

            gate_before_act = torch.bmm(state.w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(state.w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(state.w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))
            if state.pending_momentum is not None and state.dw0_momentum is not None:
                m_i = state.pending_momentum[:, : self.lact_chunk_size, :].mean(dim=1, keepdim=True)
                dw0 = dw0 + state.dw0_momentum * m_i
                dw1 = dw1 + state.dw1_momentum * m_i
                dw2 = dw2 + state.dw2_momentum * m_i
                state.dw0_momentum = dw0
                state.dw1_momentum = dw1
                state.dw2_momentum = dw2
            if self.use_muon:
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

            state.w0 = state.w0 + dw0
            state.w1 = state.w1 + dw1
            state.w2 = state.w2 + dw2
            state.w0 = state.w0 / (state.w0.norm(dim=2, keepdim=True) + 1e-5) * state.w0_norm
            state.w1 = state.w1 / (state.w1.norm(dim=2, keepdim=True) + 1e-5) * state.w1_norm
            state.w2 = state.w2 / (state.w2.norm(dim=2, keepdim=True) + 1e-5) * state.w2_norm

            for name in ("pending_k", "pending_v", "pending_lr0", "pending_lr1", "pending_lr2", "pending_momentum"):
                value = getattr(state, name)
                if value is not None:
                    value = value[:, self.lact_chunk_size :, :]
                    setattr(state, name, value if value.shape[1] > 0 else None)

        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s h -> (b h) s 1", h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale
        return rearrange(
            ttt_x_normed, "(b h) s d -> b s (h d)", h=self.num_fw_heads
        ).type_as(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lact_cache: Optional[LaCTCache] = None,
        **kwargs,
    ):
        if position_embeddings is None:
            raise RuntimeError("Qwen2.5 LaCT requires position_embeddings from Qwen2.5 text model.")
        batch_size, seq_len, _ = hidden_states.shape
        (
            query_states,
            key_states,
            value_states,
            q_flat,
            k_flat,
            v_expanded,
        ) = self._compute_qkv(hidden_states)

        video_mask = kwargs.get("video_mask", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        if self.use_conv_layer and video_mask is not None:
            if batch_size != 1:
                raise RuntimeError("LaCT conv currently supports batch_size=1 only.")
            video_index = self._build_video_index(video_mask, video_grid_thw, seq_len)
            q_flat = self._apply_full_video_conv(q_flat, self.conv_q, video_index)
            k_flat = self._apply_full_video_conv(k_flat, self.conv_k, video_index)
            v_expanded = self._apply_full_video_conv(v_expanded, self.conv_v, video_index)
        elif self.use_conv_layer and lact_cache is None and video_grid_thw is not None:
            raise RuntimeError("use_conv_layer=True but video_mask was not passed to LaCT layer.")

        if lact_cache is None:
            ttt_output = self._compute_ttt_output(
                hidden_states, q_flat, k_flat, v_expanded, position_embeddings
            )
        elif not lact_cache.has_layer(self.layer_idx):
            ttt_output = self._init_cache_from_prefill(
                hidden_states, q_flat, k_flat, v_expanded, position_embeddings, lact_cache
            )
        else:
            ttt_output = self._compute_ttt_output_decode(
                hidden_states, q_flat, k_flat, v_expanded, position_embeddings, lact_cache
            )

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.config.rope_scaling["mrope_section"],
        )
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            if seq_len == 1:
                key_states, value_states = _truncate_kv_cache_layer(
                    past_key_values,
                    self.layer_idx,
                    key_states,
                    value_states,
                    self.window_size,
                )

        if self.config._attn_implementation == "eager":
            attention_interface = None
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if attention_interface is None:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                eager_attention_forward,
            )

            attention_interface = eager_attention_forward

        attn_kwargs = dict(kwargs)
        attn_kwargs.pop("video_mask", None)
        attn_kwargs.pop("video_grid_thw", None)
        attn_kwargs["sliding_window"] = self.window_size
        attn_output, attn_weights = attention_interface(
            self.attn_layer,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attn_layer.attention_dropout,
            scaling=self.scaling,
            position_ids=position_ids,
            **attn_kwargs,
        )
        if past_key_values is not None:
            _truncate_kv_cache_layer(
                past_key_values,
                self.layer_idx,
                key_states,
                value_states,
                self.window_size,
            )
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        output = self.attn_layer.o_proj(attn_output + ttt_output)
        return output, attn_weights if output_attentions else None


def _get_language_layers(model) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    raise ValueError("Cannot find Qwen2.5 language model layers.")


def _parse_lact_layers(lact_layers: str) -> set[int]:
    return {int(x.strip()) for x in lact_layers.split("/") if x.strip()}


def qwen25_decoder_layer_forward_with_lact(self, *args, **kwargs):
    lact_kwargs = {}
    for key in ("lact_cache", "video_mask", "video_grid_thw"):
        if key in kwargs:
            lact_kwargs[key] = kwargs.pop(key)
    if isinstance(self.self_attn, Qwen2_5VLLaCTSWIGLULayer):
        kwargs.update(lact_kwargs)
    return self._timelens_lact_original_forward(*args, **kwargs)


def _patch_decoder_layer_forward_for_lact(layer) -> None:
    if hasattr(layer, "_timelens_lact_original_forward"):
        return
    layer._timelens_lact_original_forward = layer.forward
    layer.forward = MethodType(qwen25_decoder_layer_forward_with_lact, layer)


def wrap_qwen25_vl_model_with_lact(model, lact_config: LaCTConfig):
    if not lact_config.lact_enable:
        return model

    layers = _get_language_layers(model)
    layer_indices = _parse_lact_layers(lact_config.lact_layers)
    invalid = sorted(i for i in layer_indices if i < 0 or i >= len(layers))
    if invalid:
        raise ValueError(
            f"LaCT layer indices out of range for {len(layers)} decoder layers: {invalid}"
        )

    for idx, layer in enumerate(layers):
        if idx not in layer_indices:
            continue
        old_attn = layer.self_attn
        model_dtype = next(old_attn.parameters()).dtype
        model_device = next(old_attn.parameters()).device
        lact_layer = Qwen2_5VLLaCTSWIGLULayer(
            attn_layer=old_attn,
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
            fw_init_gain=lact_config.fw_init_gain,
            use_conv_layer=lact_config.use_conv_layer,
        )
        layer.self_attn = lact_layer.to(dtype=model_dtype, device=model_device)
    for layer in layers:
        _patch_decoder_layer_forward_for_lact(layer)
    patch_qwen25_vl_forward_for_lact(model)
    return model


def enable_lact_parameters(model) -> None:
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in LACT_PARAM_KEYWORDS):
            param.requires_grad = True


def qwen25vl_forward_with_lact(
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
    rope_deltas=None,
    second_per_grid_ts=None,
    cache_position=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    lact_cache: Optional[LaCTCache] = None,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
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
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        lact_cache=lact_cache,
        video_mask=video_mask,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )
    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def patch_qwen25_vl_forward_for_lact(model) -> None:
    target = model.model if hasattr(model, "model") and hasattr(model.model, "language_model") else model
    target.forward = MethodType(qwen25vl_forward_with_lact, target)
