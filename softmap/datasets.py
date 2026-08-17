"""Small built-in and published datasets for examples."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .api import LinkageData
from .simulate import simulate_backcross

CONTEMPORARY_HYBRIDIZATION_URL = (
    "https://raw.githubusercontent.com/nedarahnama/Contemporary_hybridization/"
    "master/04_genetic_map/map6j_742_2082_gen.csv"
)


@lru_cache(maxsize=4)
def _read_url(source: str) -> str:
    with urlopen(source, timeout=60) as response:
        return response.read().decode("utf-8")


def _read_source(source: str | Path) -> str:
    if str(source).startswith(("http://", "https://")):
        return _read_url(str(source))
    return Path(source).read_text(encoding="utf-8")


@dataclass(frozen=True)
class MapPositions:
    """Physical and genetic coordinates for markers on multiple chromosomes."""

    marker_names: tuple[str, ...]
    chromosomes: np.ndarray
    physical_mb: np.ndarray
    genetic_cm: np.ndarray
    label: str | None = None

    def __post_init__(self) -> None:
        marker_count = len(self.marker_names)
        chromosomes = np.asarray(self.chromosomes, dtype=np.int64)
        physical = np.asarray(self.physical_mb, dtype=np.float64)
        genetic = np.asarray(self.genetic_cm, dtype=np.float64)
        if any(values.shape != (marker_count,) for values in (chromosomes, physical, genetic)):
            raise ValueError("coordinate arrays must contain one value per marker")
        if not np.all(np.isfinite(physical)) or not np.all(np.isfinite(genetic)):
            raise ValueError("map coordinates must be finite")
        object.__setattr__(self, "chromosomes", chromosomes)
        object.__setattr__(self, "physical_mb", physical)
        object.__setattr__(self, "genetic_cm", genetic)


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
    text = _read_source(source)

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
    try:
        physical_positions = np.asarray(
            [float(row[0].rsplit("_", 1)[1]) for row in rows]
        )
    except (IndexError, ValueError):
        physical_positions = None
    return LinkageData(
        probabilities,
        tuple(row[0] for row in rows),
        np.asarray([float(row[2]) for row in rows]),
        f"Rahnamae et al. (2026), chromosome {chromosome}",
        physical_positions,
    )


def contemporary_map_positions(
    *, source: str | Path = CONTEMPORARY_HYBRIDIZATION_URL
) -> MapPositions:
    """Load published physical and genetic coordinates for all eight chromosomes."""

    text = _read_source(source)

    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    rows = [row for row in reader if len(row) >= 3]
    if not rows:
        raise ValueError("no map positions were found")
    return MapPositions(
        tuple(row[0] for row in rows),
        np.asarray([int(row[1]) for row in rows]),
        np.asarray([float(row[0].rsplit("_", 1)[1]) for row in rows]),
        np.asarray([float(row[2]) for row in rows]),
        "Rahnamae et al. (2026), physical and genetic map",
    )
