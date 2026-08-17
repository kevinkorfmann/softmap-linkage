"""Reproduce the complete-F2 Rahnamae benchmark and flagship figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import softmap
from softmap.datasets import CONTEMPORARY_HYBRIDIZATION_URL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=CONTEMPORARY_HYBRIDIZATION_URL,
        help="published CSV URL or a local copy",
    )
    parser.add_argument("--output", type=Path, default=Path("example_output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mappings = []
    metrics = []
    for chromosome in range(1, 9):
        data = softmap.contemporary_hybridization_f2(
            chromosome=chromosome,
            source=args.source,
        )
        mapping = softmap.fit_f2(data, use_physical_scaffold=True)
        mappings.append(mapping)

        inferred = mapping.result.genetic_distances.marker_positions_cm
        published = data.reference_positions
        assert published is not None
        position_correlation = abs(float(np.corrcoef(inferred, published)[0, 1]))
        published_length = float(np.ptp(published))
        inferred_length = float(np.ptp(inferred))
        metrics.append({
            "chromosome": chromosome,
            "markers": len(data.marker_names),
            "position_correlation": position_correlation,
            "inferred_length_cm": inferred_length,
            "published_length_cm": published_length,
            "relative_length_error": (
                inferred_length - published_length
            ) / published_length,
        })

    softmap.plot_physical_output_grid(
        mappings,
        mappings,
        args.output / "physical_softmap_outputs_grid.png",
        use_de_novo_rank=True,
    )
    softmap.plot_physical_output_grid(
        mappings,
        mappings,
        args.output / "physical_softmap_outputs_grid.svg",
        use_de_novo_rank=True,
    )
    softmap.plot_f2_three_stage(
        mappings[2],
        args.output / "rahnamae_chr3_three_stage.png",
    )
    softmap.plot_f2_three_stage(
        mappings[2],
        args.output / "rahnamae_chr3_three_stage.svg",
    )
    positions = softmap.contemporary_map_positions(source=args.source)
    softmap.plot_physical_vs_genetic(
        positions,
        args.output / "physical_vs_genetic_map.png",
    )
    softmap.plot_physical_vs_genetic(
        positions,
        args.output / "physical_vs_genetic_map.svg",
    )
    (args.output / "rahnamae_f2_benchmark.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))

    minimum_correlation = min(row["position_correlation"] for row in metrics)
    maximum_length_error = max(abs(row["relative_length_error"]) for row in metrics)
    if minimum_correlation < 0.995 or maximum_length_error > 0.15:
        raise RuntimeError("Rahnamae benchmark fell below its documented acceptance range")


if __name__ == "__main__":
    main()
