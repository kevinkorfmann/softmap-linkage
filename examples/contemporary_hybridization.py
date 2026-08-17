"""Reproduce the SoftMap plot made from the published Arabis map."""

from pathlib import Path

import softmap

output = Path("example_output")
output.mkdir(exist_ok=True)

data = softmap.contemporary_hybridization(chromosome=1, markers=100).shuffled(seed=11)
mapping = softmap.fit(
    data,
    bootstrap=20,
    confidence=0.8,
    seed=7,
    bin_threshold=0.005,
)
print(mapping.summary())
mapping.plot(output / "contemporary_hybridization.png")
mapping.plot(output / "contemporary_hybridization.svg")
mapping.plot_marker_order(output / "marker_order_before_after.png")
mapping.plot_marker_order(output / "marker_order_before_after.svg")

chromosome_maps = [
    softmap.fit(
        softmap.contemporary_hybridization(
            chromosome=chromosome,
            markers=80,
        ).shuffled(seed=10 + chromosome),
        bootstrap=10,
        confidence=0.8,
        seed=20 + chromosome,
        bin_threshold=0.005,
    )
    for chromosome in range(1, 9)
]
softmap.plot_physical_order_grid(
    chromosome_maps, output / "physical_order_before_after_grid.png"
)
softmap.plot_physical_order_grid(
    chromosome_maps, output / "physical_order_before_after_grid.svg"
)

positions = softmap.contemporary_map_positions()
softmap.plot_physical_vs_genetic(
    positions, output / "physical_vs_genetic_map.png"
)
softmap.plot_physical_vs_genetic(
    positions, output / "physical_vs_genetic_map.svg"
)
