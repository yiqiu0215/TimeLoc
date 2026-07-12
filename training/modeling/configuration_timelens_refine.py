from typing import Optional

from transformers import PretrainedConfig

from training.modeling.special_tokens import TIME_BIN_COUNT


class TimeLensRefineConfig(PretrainedConfig):
    model_type = "timelens_refine"

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        base_model_subdir: Optional[str] = None,
        base_model_config: Optional[dict] = None,
        time_token_ids: Optional[list[int]] = None,
        fg_token_id: Optional[int] = None,
        bg_token_id: Optional[int] = None,
        vtg_token_id: Optional[int] = None,
        vtg_end_token_id: Optional[int] = None,
        vision_start_token_id: Optional[int] = None,
        vision_end_token_id: Optional[int] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        spatial_merge_size: int = 2,
        llm_hidden_size: Optional[int] = None,
        time_hidden_size: int = 512,
        classification_embedding_dim: int = 32,
        time_embedding_dim: int = 32,
        branch_embedding_dim: int = 32,
        refine_attention_heads: int = 8,
        refine_ffn_expansion_ratio: int = 4,
        refine_dropout: float = 0.0,
        lambda_ntp: float = 1.0,
        lambda_diou: float = 1.0,
        lambda_reg: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_subdir = base_model_subdir
        self.base_model_config = base_model_config or {}
        self.time_bin_count = TIME_BIN_COUNT
        self.time_token_ids = list(time_token_ids or [])
        self.fg_token_id = fg_token_id
        self.bg_token_id = bg_token_id
        self.vtg_token_id = vtg_token_id
        self.vtg_end_token_id = vtg_end_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.pad_token_id = pad_token_id
        self.spatial_merge_size = int(spatial_merge_size)
        self.llm_hidden_size = llm_hidden_size
        self.time_hidden_size = int(time_hidden_size)
        self.classification_embedding_dim = int(classification_embedding_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.branch_embedding_dim = int(branch_embedding_dim)
        self.refine_attention_heads = int(refine_attention_heads)
        self.refine_ffn_expansion_ratio = int(refine_ffn_expansion_ratio)
        self.refine_dropout = float(refine_dropout)
        self.lambda_ntp = float(lambda_ntp)
        self.lambda_diou = float(lambda_diou)
        self.lambda_reg = float(lambda_reg)

    @classmethod
    def from_base_model_config(
        cls,
        base_config,
        base_model_name_or_path: Optional[str],
        token_spec,
        **kwargs,
    ):
        base_dict = base_config.to_dict()
        vision_config = getattr(base_config, "vision_config", None)
        llm_hidden_size = getattr(base_config, "hidden_size", None)
        if llm_hidden_size is None:
            text_config = getattr(base_config, "text_config", None)
            llm_hidden_size = getattr(text_config, "hidden_size", None)
        if isinstance(vision_config, dict):
            spatial_merge_size = vision_config.get("spatial_merge_size", 2)
        else:
            spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        return cls(
            base_model_name_or_path=base_model_name_or_path,
            base_model_config=base_dict,
            time_token_ids=list(token_spec.time_token_ids),
            fg_token_id=int(token_spec.fg_token_id),
            bg_token_id=int(token_spec.bg_token_id),
            vtg_token_id=int(token_spec.vtg_token_id),
            vtg_end_token_id=int(token_spec.vtg_end_token_id),
            vision_start_token_id=getattr(base_config, "vision_start_token_id", None),
            vision_end_token_id=getattr(base_config, "vision_end_token_id", None),
            image_token_id=getattr(base_config, "image_token_id", None),
            video_token_id=getattr(base_config, "video_token_id", None),
            pad_token_id=getattr(base_config, "pad_token_id", None),
            spatial_merge_size=spatial_merge_size,
            llm_hidden_size=llm_hidden_size,
            **kwargs,
        )
