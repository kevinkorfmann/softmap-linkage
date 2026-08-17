"""Rebuild the three published-data figures used in the case-study gallery."""

from pathlib import Path

import numpy as np

import softmap


def order_correlation(mapping: softmap.Map) -> float:
    """Return orientation-free correlation with the published marker order."""

    result = mapping.result
    rank = np.empty(result.representative_order.size, dtype=np.int64)
    rank[result.representative_order] = np.arange(result.representative_order.size)
    assert mapping.data.reference_positions is not None
    reference = mapping.data.reference_positions[result.bins.representatives]
    return abs(float(np.corrcoef(rank, reference)[0, 1]))


output = Path("docs/assets")
cases = (
    (
        "arabis_hybridization",
        softmap.contemporary_hybridization(chromosome=1),
        11,
        7,
        None,
    ),
    ("arabidopsis_ril", softmap.grav2_ril(chromosome=1), 12, 8, 0.005),
    ("mouse_backcross", softmap.hyper_backcross(chromosome=1), 13, 9, 0.005),
)

for filename, data, shuffle_seed, fit_seed, bin_threshold in cases:
    mapping = softmap.fit(
        data.shuffled(seed=shuffle_seed),
        bootstrap=100,
        confidence=0.8,
        seed=fit_seed,
        bin_threshold=bin_threshold,
    )
    mapping.plot(output / f"case_{filename}.png")
    print(filename, mapping.summary(), f"reference_r={order_correlation(mapping):.3f}")
