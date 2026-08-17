"""Small built-in and published datasets for examples."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .api import LinkageData
from .simulate import simulate_backcross

CONTEMPORARY_HYBRIDIZATION_URL = (
    "https://raw.githubusercontent.com/nedarahnama/Contemporary_hybridization/"
    "master/04_genetic_map/map6j_742_2082_gen.csv"
)


def demo(*, offspring: int = 80, markers: int = 60, seed: int = 4) -> LinkageData:
    """Return a small simulated backcross that runs in a few seconds."""

    cross = simulate_backcross(
        n_offspring=offspring,
        n_markers=markers,
        mean_depth=4.0,
        random_seed=seed,
    )
    return LinkageData(
        cross.probabilities,
        tuple(cross.marker_names),
        np.asarray(cross.input_to_truth, dtype=np.float64),
        "Simulated backcross",
    )


def contemporary_hybridization(
    *,
    chromosome: int = 1,
    markers: int = 100,
    source: str | Path = CONTEMPORARY_HYBRIDIZATION_URL,
) -> LinkageData:
    """Load one chromosome from the Rahnamae et al. Arabis map.

    NN and SS calls become probabilities 0.01 and 0.99. NS and missing calls are
    represented as 0.5 because their phase is unresolved in this binary example.
    Markers are sampled evenly across the published map when ``markers`` is smaller
    than the available chromosome marker count.
    """

    if markers < 2:
        raise ValueError("markers must be at least two")
    if str(source).startswith(("http://", "https://")):
        with urlopen(str(source), timeout=60) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")

    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    rows = [row for row in reader if len(row) >= 5 and row[1] == str(chromosome)]
    if not rows:
        raise ValueError(f"chromosome {chromosome} was not found")
    if markers < len(rows):
        indices = np.linspace(0, len(rows) - 1, markers).round().astype(int)
        rows = [rows[int(index)] for index in indices]

    calls = {"NN": 0.01, "SS": 0.99, "NS": 0.5, "-": 0.5, "": 0.5}
    probabilities = np.asarray(
        [[calls.get(value, 0.5) for value in row[3:]] for row in rows],
        dtype=np.float64,
    ).T
    return LinkageData(
        probabilities,
        tuple(row[0] for row in rows),
        np.asarray([float(row[2]) for row in rows]),
        f"Rahnamae et al., chromosome {chromosome}",
    )
