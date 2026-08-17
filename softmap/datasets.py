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

GRAV2_BASE_URL = "https://kbroman.org/qtl2/assets/sampledata/grav2"
HYPER_BASE_URL = (
    "https://raw.githubusercontent.com/kbroman/qtl/main/inst/contrib/bin/test"
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
    markers: int | None = None,
    source: str | Path = CONTEMPORARY_HYBRIDIZATION_URL,
) -> LinkageData:
    """Load one chromosome from the Rahnamae et al. Arabis map.

    NN and SS calls become probabilities 0.01 and 0.99. NS and missing calls are
    represented as 0.5 because their phase is unresolved in this binary example.
    By default all chromosome markers are returned. Markers are sampled evenly
    across the published map when ``markers`` is set below the available count.
    """

    if markers is not None and markers < 2:
        raise ValueError("markers must be at least two")
    text = _read_source(source)

    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    rows = [row for row in reader if len(row) >= 5 and row[1] == str(chromosome)]
    if not rows:
        raise ValueError(f"chromosome {chromosome} was not found")
    if markers is not None and markers < len(rows):
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
        f"Rahnamae et al. (2025), chromosome {chromosome}",
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
        "Rahnamae et al. (2025), physical and genetic map",
    )


def grav2_ril(
    *,
    chromosome: int = 1,
    markers: int | None = None,
    genotype_source: str | Path = f"{GRAV2_BASE_URL}/grav2_geno.csv",
    map_source: str | Path = f"{GRAV2_BASE_URL}/grav2_gmap.csv",
) -> LinkageData:
    """Load one chromosome from the Moore et al. Arabidopsis RIL experiment.

    The R/qtl2 sample files contain two homozygous states, L and C. They become
    probabilities 0.01 and 0.99; missing calls remain uninformative at 0.5.
    Markers may be sampled evenly across the published map for a faster example.
    """

    genotype_rows = list(csv.reader(io.StringIO(_read_source(genotype_source))))
    map_rows = list(csv.DictReader(io.StringIO(_read_source(map_source))))
    if not genotype_rows or len(genotype_rows[0]) < 3:
        raise ValueError("the grav2 genotype table is empty or malformed")
    genotype_index = {name: index for index, name in enumerate(genotype_rows[0])}
    selected = [row for row in map_rows if int(row["chr"]) == chromosome]
    if not selected:
        raise ValueError(f"chromosome {chromosome} was not found")
    if markers is not None:
        if markers < 2:
            raise ValueError("markers must be at least two")
        if markers < len(selected):
            indices = np.linspace(0, len(selected) - 1, markers).round().astype(int)
            selected = [selected[int(index)] for index in indices]

    marker_names = tuple(row["marker"] for row in selected)
    try:
        columns = [genotype_index[name] for name in marker_names]
    except KeyError as error:
        raise ValueError(f"marker {error.args[0]!r} is absent from the genotype table") from error
    calls = {"L": 0.01, "C": 0.99, "-": 0.5, "NA": 0.5, "": 0.5}
    probabilities = np.asarray(
        [[calls.get(row[column], 0.5) for column in columns] for row in genotype_rows[1:]],
        dtype=np.float64,
    )
    return LinkageData(
        probabilities,
        marker_names,
        np.asarray([float(row["pos"]) for row in selected]),
        f"Moore et al. Arabidopsis RILs, chromosome {chromosome}",
    )


def hyper_backcross(
    *,
    chromosome: int | str = 1,
    genotype_source: str | Path = f"{HYPER_BASE_URL}/genohyper.txt",
    map_source: str | Path = f"{HYPER_BASE_URL}/markerposhyper.txt",
    chromosome_source: str | Path = f"{HYPER_BASE_URL}/chridhyper.txt",
) -> LinkageData:
    """Load one chromosome from the Sugiyama et al. mouse backcross.

    The R/qtl test export stores one offspring per row with calls 0, 1, and 9.
    Binary calls become probabilities 0.01 and 0.99; 9 is missing and becomes
    0.5. Published centimorgan coordinates provide an independent order check.
    """

    genotypes = [
        row.split() for row in _read_source(genotype_source).splitlines() if row.strip()
    ]
    map_rows = [
        row.split("\t") for row in _read_source(map_source).splitlines() if row.strip()
    ]
    chromosomes = [
        row.strip() for row in _read_source(chromosome_source).splitlines() if row.strip()
    ]
    if not genotypes or len(map_rows) != len(chromosomes):
        raise ValueError("the hyper backcross source tables are empty or misaligned")
    if any(len(row) != len(map_rows) for row in genotypes):
        raise ValueError("the hyper genotype rows do not match the marker map")
    selected = [index for index, value in enumerate(chromosomes) if value == str(chromosome)]
    if len(selected) < 2:
        raise ValueError(f"chromosome {chromosome} was not found or has fewer than two markers")

    calls = {"0": 0.01, "1": 0.99, "9": 0.5}
    probabilities = np.asarray(
        [[calls.get(row[index], 0.5) for index in selected] for row in genotypes],
        dtype=np.float64,
    )
    return LinkageData(
        probabilities,
        tuple(map_rows[index][0] for index in selected),
        np.asarray([float(map_rows[index][1]) for index in selected]),
        f"Sugiyama et al. mouse backcross, chromosome {chromosome}",
    )
