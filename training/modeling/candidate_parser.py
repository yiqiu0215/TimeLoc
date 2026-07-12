from dataclasses import dataclass
import math
from typing import Optional, Sequence

from training.modeling.special_tokens import TIME_BIN_COUNT, RegisteredTimeRefineTokens


@dataclass(frozen=True)
class ParsedTimeRefineSequence:
    labels: tuple[int, ...]
    time_bins: tuple[int, ...]
    foreground_token_positions: tuple[int, ...]
    classification_token_offsets: tuple[int, ...]
    valid: bool
    status: str

    @property
    def has_foreground(self) -> bool:
        return self.valid and any(label == 1 for label in self.labels)


@dataclass(frozen=True)
class CandidateWindows:
    foreground_runs: tuple[tuple[int, int], ...]
    merged_runs: tuple[tuple[int, int], ...]
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
        classification_token_offsets=(),
        valid=False,
        status=status,
    )


def parse_time_refine_sequence(
    generated_ids: Sequence[int],
    token_spec: RegisteredTimeRefineTokens,
    expected_time_bins: Optional[Sequence[int]] = None,
    expected_length: Optional[int] = None,
) -> ParsedTimeRefineSequence:
    """Parse and validate the constrained ``<vtg>`` classification sequence.

    Ordinary text tokens between structural tokens are treated as formatting
    noise.  Structural tokens, pair count, and timestamp order remain strict;
    this permits the documented cleanup while preventing a misaligned visual
    block sequence from entering refinement.
    """

    tokens = [int(token) for token in generated_ids]
    try:
        start = tokens.index(int(token_spec.vtg_token_id))
    except ValueError:
        return _invalid("missing_vtg_start")

    try:
        end = tokens.index(int(token_spec.vtg_end_token_id), start + 1)
    except ValueError:
        return _invalid("missing_vtg_end")

    body = tokens[start + 1 : end]
    class_ids = {
        int(token_spec.fg_token_id): 1,
        int(token_spec.bg_token_id): 0,
    }
    time_id_to_bin = {
        int(token_id): index for index, token_id in enumerate(token_spec.time_token_ids)
    }
    labels = []
    time_bins = []
    foreground_positions = []
    classification_offsets = []
    cursor = 0
    while cursor < len(body):
        token = body[cursor]
        if token not in class_ids:
            if token in time_id_to_bin:
                return _invalid("orphan_time_token")
            cursor += 1
            continue

        label = class_ids[token]
        time_cursor = cursor + 1
        while time_cursor < len(body) and body[time_cursor] not in time_id_to_bin:
            if body[time_cursor] in class_ids:
                return _invalid("consecutive_classification_tokens")
            time_cursor += 1
        if time_cursor >= len(body):
            return _invalid("classification_without_time_token")

        labels.append(label)
        time_bins.append(time_id_to_bin[body[time_cursor]])
        classification_offsets.append(cursor)
        if label == 1:
            foreground_positions.append(len(labels) - 1)
        cursor = time_cursor + 1

    if expected_length is None and expected_time_bins is not None:
        expected_length = len(expected_time_bins)
    if expected_length is None or int(expected_length) <= 0:
        return _invalid("missing_expected_sequence_length")
    expected_length = int(expected_length)
    if len(labels) != expected_length:
        return _invalid(
            f"classification_count_mismatch:{len(labels)}!={expected_length}"
        )

    if expected_time_bins is not None:
        expected = [int(value) for value in expected_time_bins]
        if len(expected) != expected_length:
            return _invalid("expected_time_bin_count_mismatch")
        if any(value < 0 or value >= TIME_BIN_COUNT for value in expected):
            return _invalid("expected_time_bin_out_of_range")
        if time_bins != expected:
            return _invalid("time_token_order_mismatch")

    return ParsedTimeRefineSequence(
        labels=tuple(labels),
        time_bins=tuple(time_bins),
        foreground_token_positions=tuple(foreground_positions),
        classification_token_offsets=tuple(classification_offsets),
        valid=True,
        status="no_foreground" if not any(labels) else "ok",
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
    """Apply the documented foreground cleanup and overlapping windows."""

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
            selected_run=None,
            candidate_start=None,
            candidate_end=None,
            start_window=None,
            end_window=None,
        )
    merged_runs = _merge_runs(runs, max_background_gap)

    def run_key(run):
        start, end = run
        average_score = sum(scores[start : end + 1]) / (end - start + 1)
        return (end - start + 1, average_score, -start)

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
        selected_run=selected,
        candidate_start=candidate_start,
        candidate_end=candidate_end,
        start_window=start_window,
        end_window=end_window,
    )
