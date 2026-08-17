"""Small public interface for fitting and inspecting linkage maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import SoftMapResult, fit_softmap


@dataclass(frozen=True)
class LinkageData:
    """Probabilistic marker data in offspring-by-marker orientation."""

    probabilities: NDArray[np.float64]
    marker_names: tuple[str, ...]
    reference_positions: NDArray[np.float64] | None = None
    label: str | None = None
    physical_positions: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be a 2D offspring-by-marker matrix")
        if probabilities.shape[1] != len(self.marker_names):
            raise ValueError("marker_names length must match the number of columns")
        if self.reference_positions is not None:
            positions = np.asarray(self.reference_positions, dtype=np.float64)
            if positions.shape != (probabilities.shape[1],):
                raise ValueError("reference_positions must contain one value per marker")
            object.__setattr__(self, "reference_positions", positions)
        if self.physical_positions is not None:
            physical = np.asarray(self.physical_positions, dtype=np.float64)
            if physical.shape != (probabilities.shape[1],):
                raise ValueError("physical_positions must contain one value per marker")
            object.__setattr__(self, "physical_positions", physical)
        object.__setattr__(self, "probabilities", probabilities)

    def shuffled(self, seed: int | None = 1) -> "LinkageData":
        """Return a copy with marker columns in a reproducibly shuffled order."""

        order = np.random.default_rng(seed).permutation(self.probabilities.shape[1])
        reference = (
            self.reference_positions[order]
            if self.reference_positions is not None
            else None
        )
        physical = (
            self.physical_positions[order]
            if self.physical_positions is not None
            else None
        )
        label = f"{self.label}, shuffled input" if self.label else "Shuffled input"
        return LinkageData(
            self.probabilities[:, order],
            tuple(self.marker_names[int(index)] for index in order),
            reference,
            label,
            physical,
        )


@dataclass(frozen=True)
class Map:
    """A fitted map together with the data needed for summaries and plots."""

    data: LinkageData
    result: SoftMapResult

    @property
    def ordered_markers(self) -> list[str]:
        """Marker names in inferred order."""

        return self.result.ordered_names()

    @property
    def framework_markers(self) -> list[str]:
        """Names of markers whose relative order passes the confidence threshold."""

        representatives = self.result.bins.representatives[self.result.framework]
        return [self.data.marker_names[int(marker)] for marker in representatives]

    def summary(self) -> dict[str, int | float | str]:
        """Return a compact, serializable fit summary."""

        return {
            "status": "ok" if self.result.framework.size >= 3 else "limited_support",
            "offspring": int(self.data.probabilities.shape[0]),
            "markers": int(self.data.probabilities.shape[1]),
            "bins": int(self.result.bins.representatives.size),
            "framework_markers": int(self.result.framework.size),
            "confidence": float(self.result.confidence),
        }

    def marker_table(self) -> list[dict[str, str | int | bool | None]]:
        """Return one inspectable result row for every input marker.

        Rows contain the marker name, co-segregation bin, inferred bin rank,
        representative status, optional framework rank, and bootstrap placement
        bounds. The return value uses only built-in Python types, so it can be
        printed directly or passed to a table library such as pandas.
        """

        result = self.result
        rank_by_group = np.empty(
            result.representative_order.size,
            dtype=np.int64,
        )
        rank_by_group[result.representative_order] = np.arange(
            result.representative_order.size
        )
        framework_rank = {
            int(group): rank for rank, group in enumerate(result.framework)
        }
        rows: list[dict[str, str | int | bool | None]] = []
        for marker, name in enumerate(self.data.marker_names):
            group = int(result.bins.membership[marker])
            representative = int(result.bins.representatives[group])
            rows.append({
                "marker": name,
                "bin": group,
                "order_rank": int(rank_by_group[group]),
                "is_representative": marker == representative,
                "framework_rank": framework_rank.get(group),
                "interval_left": int(result.interval_left[group]),
                "interval_right": int(result.interval_right[group]),
            })
        return rows

    def plot(self, path: str | Path | None = None):
        """Create the standard overview figure and optionally save it."""

        from .plotting import plot_map

        return plot_map(self, path=path)

    def plot_marker_order(self, path: str | Path | None = None):
        """Compare the input marker order with the fitted marker order."""

        from .plotting import plot_marker_order

        return plot_marker_order(self, path=path)


def fit(
    data: LinkageData | ArrayLike,
    marker_names: Iterable[str] | None = None,
    *,
    bootstrap: int = 20,
    confidence: float = 0.8,
    seed: int | None = 1,
    bin_threshold: float | None = 0.01,
) -> Map:
    """Fit a confidence-aware linkage map with practical defaults.

    Parameters
    ----------
    data
        A :class:`LinkageData` object or an offspring-by-marker probability matrix.
        Values must lie between zero and one.
    marker_names
        Optional names when ``data`` is an array.
    bootstrap
        Number of bootstrap maps. Use at least 100 for a final analysis.
    confidence
        Minimum pairwise support used to select framework markers.
    seed
        Random seed for reproducible bootstrap results.
    bin_threshold
        Maximum expected disagreement for co-segregating marker bins. Pass ``None``
        to select the threshold automatically.
    """

    if isinstance(data, LinkageData):
        if marker_names is not None:
            raise ValueError("marker_names is already included in LinkageData")
        linkage_data = data
    else:
        probabilities = np.asarray(data, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError("data must be a 2D offspring-by-marker matrix")
        names = (
            tuple(marker_names)
            if marker_names is not None
            else tuple(f"m{i + 1}" for i in range(probabilities.shape[1]))
        )
        linkage_data = LinkageData(probabilities, names)

    result = fit_softmap(
        linkage_data.probabilities,
        linkage_data.marker_names,
        bootstrap_replicates=bootstrap,
        confidence=confidence,
        random_seed=seed,
        bin_threshold=bin_threshold,
        neighbor_count=min(20, linkage_data.probabilities.shape[1]),
    )
    return Map(linkage_data, result)
