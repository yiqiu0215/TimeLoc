from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundaryWindow:
    indices: tuple[int, ...]
    left_bracket: int
    right_bracket: int
    edge_clipped: bool


def find_bracketing_indices(
    timestamps: torch.Tensor,
    boundary: float,
) -> tuple[int, int, bool]:
    values = torch.as_tensor(timestamps, dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot build a boundary window from an empty timestamp sequence.")
    if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
        raise ValueError("timestamps must be strictly increasing.")
    boundary = float(boundary)
    if boundary <= float(values[0]):
        return 0, 0, boundary < float(values[0])
    if boundary >= float(values[-1]):
        last = values.numel() - 1
        return last, last, boundary > float(values[-1])

    right = int(
        torch.searchsorted(
            values,
            torch.tensor(boundary, dtype=values.dtype, device=values.device),
            right=False,
        ).item()
    )
    if float(values[right]) == boundary:
        return right, right, False
    return right - 1, right, False


def build_training_boundary_window(
    timestamps: torch.Tensor,
    boundary: float,
    left_context: int = 4,
    right_context: int = 4,
) -> BoundaryWindow:
    values = torch.as_tensor(timestamps, dtype=torch.float32).reshape(-1)
    left_bracket, right_bracket, edge_clipped = find_bracketing_indices(
        values, boundary
    )
    n = int(values.numel())
    left_context, right_context = int(left_context), int(right_context)
    if left_context < 0 or right_context < 0:
        raise ValueError("Window context sizes must be non-negative.")
    start = max(0, left_bracket - left_context)
    end = min(n - 1, right_bracket + right_context)
    indices = tuple(range(start, end + 1))
    if not indices:
        raise ValueError("Boundary window construction produced an empty window.")
    if not (float(values[start]) <= float(boundary) <= float(values[end])):
        edge_clipped = True
    return BoundaryWindow(
        indices=indices,
        left_bracket=left_bracket,
        right_bracket=right_bracket,
        edge_clipped=edge_clipped,
    )
