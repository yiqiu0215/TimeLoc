from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_LACT_LAYERS = "0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26"

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
    "conv_q",
    "conv_k",
    "conv_v",
)


@dataclass
class LaCTConfig:
    lact_enable: bool = False
    num_lact_heads: int = 4
    lact_chunk_size: int = 2648
    window_size: int = 2648
    use_conv_layer: bool = True
    use_momentum: bool = True
    use_muon: bool = True
    learnable_ttt_scale: bool = True
    w0_w2_low_rank: int = 0
    use_fused_kernel: bool = False
    lact_layers: str = DEFAULT_LACT_LAYERS
    qkv_silu: bool = True
    no_v_silu: bool = False
    ttt_prenorm: bool = True
    ttt_nope: bool = False
    fp32_states: bool = True
    inter_multi: float = 1.0
    fw_init_gain: float = 0.5

    @classmethod
    def from_args(cls, args: Any) -> "LaCTConfig":
        return cls(
            lact_enable=bool(getattr(args, "lact_enable", False)),
            num_lact_heads=int(getattr(args, "num_lact_heads", 4)),
            lact_chunk_size=int(getattr(args, "lact_chunk_size", 2648)),
            window_size=int(getattr(args, "window_size", 2648)),
            use_conv_layer=bool(getattr(args, "use_conv_layer", True)),
            use_momentum=bool(getattr(args, "use_momentum", True)),
            use_muon=bool(getattr(args, "use_muon", True)),
            learnable_ttt_scale=bool(getattr(args, "learnable_ttt_scale", True)),
            w0_w2_low_rank=int(getattr(args, "w0_w2_low_rank", 0)),
            use_fused_kernel=bool(getattr(args, "use_fused_kernel", False)),
            lact_layers=getattr(args, "lact_layers", None) or DEFAULT_LACT_LAYERS,
            qkv_silu=bool(getattr(args, "qkv_silu", True)),
            no_v_silu=bool(getattr(args, "no_v_silu", False)),
            ttt_prenorm=bool(getattr(args, "ttt_prenorm", True)),
            ttt_nope=bool(getattr(args, "ttt_nope", False)),
            fp32_states=bool(getattr(args, "fp32_states", True)),
            inter_multi=float(getattr(args, "inter_multi", 1.0)),
            fw_init_gain=float(getattr(args, "fw_init_gain", 0.5)),
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LaCTConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_lact_config(output_dir: str | Path, config: LaCTConfig) -> None:
    path = Path(output_dir) / "lact_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_lact_config(model_path: str | Path) -> Optional[dict[str, Any]]:
    path = Path(model_path) / "lact_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lact_config(model_path: str | Path, expected: LaCTConfig) -> None:
    saved = load_lact_config(model_path)
    if saved is None:
        return

    expected_dict = expected.to_dict()
    keys = (
        "num_lact_heads",
        "lact_chunk_size",
        "window_size",
        "use_conv_layer",
        "use_momentum",
        "use_muon",
        "learnable_ttt_scale",
        "w0_w2_low_rank",
        "use_fused_kernel",
        "lact_layers",
        "qkv_silu",
        "no_v_silu",
        "ttt_prenorm",
        "ttt_nope",
        "fp32_states",
        "inter_multi",
        "fw_init_gain",
    )
    mismatches = [
        f"{key}: saved={saved.get(key)!r}, expected={expected_dict.get(key)!r}"
        for key in keys
        if saved.get(key) != expected_dict.get(key)
    ]
    if mismatches:
        joined = "\n".join(mismatches)
        raise ValueError(
            "LaCT arguments do not match lact_config.json. "
            "Use the same explicit LaCT settings used for training.\n"
            f"{joined}"
        )
