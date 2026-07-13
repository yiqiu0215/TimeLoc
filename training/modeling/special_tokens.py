from dataclasses import asdict, dataclass
from typing import Any


TIME_BIN_COUNT = 301
TIME_TOKEN_PREFIX = "<time_"
CLASSIFICATION_TOKENS = ("<fg>", "<bg>", "<vtg>", "</vtg>")


def format_time_token(time_bin: int) -> str:
    if not 0 <= int(time_bin) < TIME_BIN_COUNT:
        raise ValueError(
            f"time_bin must be in [0, {TIME_BIN_COUNT - 1}], got {time_bin}."
        )
    return f"{TIME_TOKEN_PREFIX}{int(time_bin):03d}>"


def get_time_tokens() -> tuple[str, ...]:
    return tuple(format_time_token(i) for i in range(TIME_BIN_COUNT))


@dataclass(frozen=True)
class RegisteredCoarse2RefineTokens:
    fg_token_id: int
    bg_token_id: int
    vtg_token_id: int
    vtg_end_token_id: int
    time_token_ids: tuple[int, ...]

    @property
    def time_token_id_start(self) -> int:
        return self.time_token_ids[0]

    def token_id_for_bin(self, time_bin: int) -> int:
        if not 0 <= int(time_bin) < TIME_BIN_COUNT:
            raise ValueError(
                f"time_bin must be in [0, {TIME_BIN_COUNT - 1}], got {time_bin}."
            )
        return self.time_token_ids[int(time_bin)]

    def bin_for_token_id(self, token_id: int) -> int:
        time_bin = int(token_id) - self.time_token_id_start
        if not 0 <= time_bin < TIME_BIN_COUNT:
            raise ValueError(f"{token_id} is not a registered time token id.")
        if self.time_token_ids[time_bin] != int(token_id):
            raise ValueError(
                "Time token ids are not contiguous; cannot use id-offset decoding."
            )
        return time_bin

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["time_token_ids"] = list(self.time_token_ids)
        data["time_bin_count"] = TIME_BIN_COUNT
        return data


def _extract_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if ids is None:
        raise ValueError("Tokenizer output does not contain input_ids.")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def register_coarse2refine_tokens(tokenizer) -> RegisteredCoarse2RefineTokens:
    """Register the exact token set required by the Coarse2Refine path.

    The design uses ``ID(<time_q>) - ID(<time_000>) == q``.  This invariant is
    checked here instead of being assumed later by the refinement head.
    """

    time_tokens = get_time_tokens()
    all_tokens = list(CLASSIFICATION_TOKENS) + list(time_tokens)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": all_tokens}
    )
    if not isinstance(added, int) or added < 0:
        raise ValueError(f"Unexpected tokenizer.add_special_tokens result: {added!r}")

    token_ids = [
        int(tokenizer.convert_tokens_to_ids(token)) for token in all_tokens
    ]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            "Coarse2Refine special tokens do not have unique token ids."
        )

    time_token_ids = tuple(token_ids[len(CLASSIFICATION_TOKENS) :])
    expected_ids = tuple(
        time_token_ids[0] + i for i in range(TIME_BIN_COUNT)
    )
    if time_token_ids != expected_ids:
        raise ValueError(
            "Time token ids must be contiguous so that id offsets equal time bins: "
            f"first={time_token_ids[0]}, last={time_token_ids[-1]}."
        )

    for token, token_id in zip(all_tokens, token_ids):
        encoded_ids = _extract_token_ids(
            tokenizer(token, add_special_tokens=False)
        )
        if encoded_ids != [token_id]:
            raise ValueError(
                f"{token!r} must tokenize to exactly its registered id {token_id}, "
                f"got {encoded_ids}."
            )

    return RegisteredCoarse2RefineTokens(
        fg_token_id=token_ids[0],
        bg_token_id=token_ids[1],
        vtg_token_id=token_ids[2],
        vtg_end_token_id=token_ids[3],
        time_token_ids=time_token_ids,
    )


def is_qwen25_coarse2refine(
    model_id: str | None,
    model_name_or_path: str | None,
    processor_path: str | None,
) -> bool:
    """Return whether the dedicated Qwen2.5-VL-3B Coarse2Refine path is active.

    The Processor is intentionally not a trigger.  ``TimeLens-7B`` is reused
    only for tokenization/video preprocessing and must not turn an ordinary
    Qwen2.5-VL or TimeLens checkpoint into a Coarse2Refine model.
    """

    model_id_text = (model_id or "").lower()
    model_path_text = (model_name_or_path or "").lower()
    # Keep the argument for API compatibility, but never use the Processor as
    # a model-family trigger.
    _ = processor_path

    def _is_qwen25_3b_target(text: str) -> bool:
        return (
            "coarse2refine" in text
            and ("qwen25" in text or "qwen2.5" in text or "qwen2_5" in text)
            and "3b" in text
        )

    if _is_qwen25_3b_target(model_id_text):
        return True
    return _is_qwen25_3b_target(model_path_text)


# Public compatibility aliases.  New code must use the Coarse2Refine names;
# these aliases do not restore the old TimeLens-based trigger behavior.
RegisteredTimeRefineTokens = RegisteredCoarse2RefineTokens
register_time_refine_tokens = register_coarse2refine_tokens


def is_qwen25_timelens_3b(
    model_id: str | None,
    model_name_or_path: str | None,
    processor_path: str | None,
) -> bool:
    """Deprecated compatibility alias for the Coarse2Refine detector."""

    return is_qwen25_coarse2refine(
        model_id,
        model_name_or_path,
        processor_path,
    )
