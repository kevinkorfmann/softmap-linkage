"""Simple tabular I/O for the SoftMap MVP."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .core import HierarchicalSoftMapResult, SoftMapResult


def read_probability_tsv(path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read marker rows and offspring probabilities, with optional truth metadata."""

    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("probability TSV is empty") from error
        if not header or header[0] != "marker":
            raise ValueError("first column must be named marker")
        probability_start = (
            2 if len(header) > 1 and header[1] == "truth_position" else 1
        )
        if len(header) - probability_start < 2:
            raise ValueError(
                "expected marker followed by at least two offspring columns"
            )
        names: list[str] = []
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"line {line_number} has {len(row)} columns; expected {len(header)}"
                )
            names.append(row[0])
            rows.append([float(value) for value in row[probability_start:]])
    if len(set(names)) != len(names):
        raise ValueError("marker names must be unique")
    return tuple(names), np.asarray(rows, dtype=np.float64).T


def write_result_tsv(result: SoftMapResult, path: str | Path) -> None:
    rep_to_group = {int(marker): group for group, marker in enumerate(result.bins.representatives)}
    representative_rank = np.empty(result.representative_order.size, dtype=np.int64)
    representative_rank[result.representative_order] = np.arange(result.representative_order.size)
    framework_lookup = {int(marker): i for i, marker in enumerate(result.framework)}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "marker",
                "bin",
                "order_rank",
                "is_representative",
                "framework_rank",
                "interval_left",
                "interval_right",
            ]
        )
        for marker, name in enumerate(result.marker_names):
            group = int(result.bins.membership[marker])
            representative = int(result.bins.representatives[group])
            writer.writerow(
                [
                    name,
                    group,
                    int(representative_rank[group]),
                    int(marker == representative),
                    framework_lookup.get(group, ""),
                    int(result.interval_left[group]),
                    int(result.interval_right[group]),
                ]
            )


def write_hierarchical_result_tsv(
    result: HierarchicalSoftMapResult, path: str | Path
) -> None:
    """Write fine-bin membership and supported HMM framework ranks."""

    scaffold_lookup = {
        int(group): rank for rank, group in enumerate(result.scaffold)
    }
    framework_lookup = {
        int(group): rank for rank, group in enumerate(result.framework)
    }
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "marker",
                "fine_bin",
                "is_representative",
                "scaffold_rank",
                "framework_rank",
            ]
        )
        for marker, name in enumerate(result.support.marker_names):
            group = int(result.bins.membership[marker])
            representative = int(result.bins.representatives[group])
            writer.writerow(
                [
                    name,
                    group,
                    int(marker == representative),
                    scaffold_lookup.get(group, ""),
                    framework_lookup.get(group, ""),
                ]
            )
