"""Reproduce the SoftMap plot made from the published Arabis map."""

from pathlib import Path

import softmap

output = Path("example_output")
output.mkdir(exist_ok=True)

data = softmap.contemporary_hybridization(chromosome=1, markers=100)
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
