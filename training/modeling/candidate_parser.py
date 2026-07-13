from dataclasses import dataclass
import math
from typing import Optional, Sequence

from training.modeling.special_tokens import (
    TIME_BIN_COUNT,
    RegisteredCoarse2RefineTokens,
)


@dataclass(frozen=True)
class ParsedTimeRefineSequence:
    labels: tuple[int, ...]
    time_bins: tuple[int, ...]
    foreground_token_positions: tuple[int, ...]
    foreground_token_offsets: tuple[int, ...]
    valid: bool
    status: str

    @property
    def has_foreground(self) -> bool:
        return self.valid and any(label == 1 for label in self.labels)

    @property
    def classification_token_offsets(self) -> tuple[int, ...]:
        """Compatibility alias for callers using the previous parser field."""

        return self.foreground_token_offsets


@dataclass(frozen=True)
class CandidateWindows:
    foreground_runs: tuple[tuple[int, int], ...]
    merged_runs: tuple[tuple[int, int], ...]
    cleaned_labels: tuple[int, ...]
    selected_run: Optional[tuple[int, int]]
    candidate_start: Optional[int]
    candidate_end: Optional[int]
    start_window: Optional[tuple[int, int]]
    end_window: Optional[tuple[int, int]]

    @property
    def has_foreground(self) -> bool:
        return self.selected_run is not None


def _invalid(status: str) -> ParsedTimeRefineSequence:
    return ParsedTimeRefineSequence(
        labels=(),
        time_bins=(),
        foreground_token_positions=(),
        foreground_token_offsets=(),
        valid=False,
        status=status,
    )


def parse_time_refine_sequence(
    generated_ids: Sequence[int],
    token_spec: RegisteredCoarse2RefineTokens,
    expected_time_bins: Optional[Sequence[int]] = None,
    expected_length: Optional[int] = None,
) -> ParsedTimeRefineSequence:
    """Parse a symmetric foreground set and restore chronological labels."""

    tokens = [int(token) for token in generated_ids]
    try:
        start = tokens.index(int(token_spec.vtg_token_id))
    except ValueError:
        return _invalid("missing_vtg_start")

    try:
        end = tokens.index(int(token_spec.vtg_end_token_id), start + 1)
    except ValueError:
        return _invalid("missing_vtg_end")

    if expected_time_bins is None:
        return _invalid("missing_expected_time_bins")
    expected = [int(value) for value in expected_time_bins]
    if expected_length is None:
        expected_length = len(expected)
    if int(expected_length) <= 0:
        return _invalid("missing_expected_sequence_length")
    expected_length = int(expected_length)
    if len(expected) != expected_length:
        return _invalid("expected_time_bin_count_mismatch")
    if any(value < 0 or value >= TIME_BIN_COUNT for value in expected):
        return _invalid("expected_time_bin_out_of_range")

    body = tokens[start + 1 : end]
    fg_token_id = int(token_spec.fg_token_id)
    if len(body) < 2 or body[0] != fg_token_id:
        return _invalid("missing_foreground_start")
    if body[-1] != fg_token_id:
        return _invalid("missing_foreground_end")
    if sum(token == fg_token_id for token in body) != 2:
        return _invalid("invalid_foreground_delimiter_count")

    time_id_to_bin = {
        int(token_id): index for index, token_id in enumerate(token_spec.time_token_ids)
    }
    selected_bins = []
    selected_offsets = []
    for body_offset, token in enumerate(body[1:-1], start=1):
        if token not in time_id_to_bin:
            if token == int(token_spec.bg_token_id):
                return _invalid("unexpected_background_token")
            return _invalid("unexpected_foreground_body_token")
        selected_bins.append(time_id_to_bin[token])
        selected_offsets.append(start + 1 + body_offset)

    expected_index_by_bin = {
        time_bin: index for index, time_bin in enumerate(expected)
    }
    selected_positions = []
    for time_bin in selected_bins:
        if time_bin not in expected_index_by_bin:
            return _invalid("foreground_time_token_not_in_input")
        position = expected_index_by_bin[time_bin]
        if position in selected_positions:
            return _invalid("duplicate_foreground_time_token")
        selected_positions.append(position)

    recovered_out_of_order = selected_positions != sorted(selected_positions)
    ordered = sorted(zip(selected_positions, selected_offsets))
    foreground_positions = tuple(position for position, _ in ordered)
    foreground_offsets = tuple(offset for _, offset in ordered)
    labels = [0] * expected_length
    for position in foreground_positions:
        labels[position] = 1

    return ParsedTimeRefineSequence(
        labels=tuple(labels),
        time_bins=tuple(expected),
        foreground_token_positions=foreground_positions,
        foreground_token_offsets=foreground_offsets,
        valid=True,
        status=(
            "no_foreground"
            if not foreground_positions
            else "recovered_out_of_order"
            if recovered_out_of_order
            else "ok"
        ),
    )


def _foreground_runs(labels: Sequence[int]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, label in enumerate(labels):
        if int(label) == 1 and start is None:
            start = index
        elif int(label) != 1 and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return runs


def _merge_runs(runs: Sequence[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged = []
    for start, end in runs:
        if not merged or start - merged[-1][1] - 1 > max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    return [tuple(run) for run in merged]


def build_candidate_windows(
    labels: Sequence[int],
    foreground_scores: Optional[Sequence[float]] = None,
    max_background_gap: int = 2,
    expansion: int = 4,
    boundary_radius: int = 4,
) -> CandidateWindows:
    """Clean foreground runs and rank them by score, length, then position."""

    labels = [int(label) for label in labels]
    if not labels:
        raise ValueError("Candidate classification sequence cannot be empty.")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Candidate classifications must contain only 0/1 values.")
    max_background_gap = int(max_background_gap)
    expansion = int(expansion)
    boundary_radius = int(boundary_radius)
    if min(max_background_gap, expansion, boundary_radius) < 0:
        raise ValueError("Candidate cleanup parameters must be non-negative.")
    if foreground_scores is None:
        scores = [float(label) for label in labels]
    else:
        scores = [float(score) for score in foreground_scores]
        if len(scores) != len(labels):
            raise ValueError("foreground_scores must match the classification length.")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("foreground_scores must be finite.")

    runs = _foreground_runs(labels)
    if not runs:
        return CandidateWindows(
            foreground_runs=(),
            merged_runs=(),
            cleaned_labels=tuple(labels),
            selected_run=None,
            candidate_start=None,
            candidate_end=None,
            start_window=None,
            end_window=None,
        )
    merged_runs = _merge_runs(runs, max_background_gap)
    cleaned_labels = labels.copy()
    for start, end in merged_runs:
        cleaned_labels[start : end + 1] = [1] * (end - start + 1)

    def run_key(run):
        start, end = run
        foreground_values = [
            scores[index]
            for index in range(start, end + 1)
            if labels[index] == 1
        ]
        average_score = sum(foreground_values) / len(foreground_values)
        return (average_score, end - start + 1, -start)

    selected = max(merged_runs, key=run_key)
    candidate_start = max(0, selected[0] - expansion)
    candidate_end = min(len(labels) - 1, selected[1] + expansion)
    start_window = (
        max(candidate_start, selected[0] - boundary_radius),
        min(candidate_end, selected[0] + boundary_radius),
    )
    end_window = (
        max(candidate_start, selected[1] - boundary_radius),
        min(candidate_end, selected[1] + boundary_radius),
    )
    return CandidateWindows(
        foreground_runs=tuple(runs),
        merged_runs=tuple(merged_runs),
        cleaned_labels=tuple(cleaned_labels),
        selected_run=selected,
        candidate_start=candidate_start,
        candidate_end=candidate_end,
        start_window=start_window,
        end_window=end_window,
    )
