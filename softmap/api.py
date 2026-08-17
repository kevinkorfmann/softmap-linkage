"""Small public interface for fitting and inspecting linkage maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    LikelihoodMDSEnsembleResult,
    SoftMapResult,
    fit_likelihood_mds_ensemble,
    fit_softmap,
)


_VCF_SUFFIXES = (".vcf", ".vcf.gz", ".bcf")


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


def _diploid_genotype_index(genotype: tuple[int, ...]) -> int | None:
    """Return the VCF Number=G index for a biallelic haploid/diploid call."""

    if len(genotype) == 1:
        return genotype[0] if genotype[0] in (0, 1) else None
    if len(genotype) != 2 or any(allele not in (0, 1) for allele in genotype):
        return None
    first, second = sorted(genotype)
    return {(0, 0): 0, (0, 1): 1, (1, 1): 2}.get((first, second))


def _call_probability(
    call,
    state_zero: tuple[int, ...],
    state_one: tuple[int, ...],
) -> float:
    """Convert a VCF sample call into a probability for the second state."""

    zero_index = _diploid_genotype_index(state_zero)
    one_index = _diploid_genotype_index(state_one)
    for field, scale in (("PL", -0.1), ("GL", 1.0)):
        values = call.get(field)
        if values is None or zero_index is None or one_index is None:
            continue
        if max(zero_index, one_index) >= len(values):
            continue
        zero_value, one_value = values[zero_index], values[one_index]
        if zero_value is None or one_value is None:
            continue
        log_zero = float(zero_value) * scale
        log_one = float(one_value) * scale
        maximum = max(log_zero, log_one)
        zero_likelihood = 10.0 ** (log_zero - maximum)
        one_likelihood = 10.0 ** (log_one - maximum)
        return one_likelihood / (zero_likelihood + one_likelihood)

    genotype = call.get("GT")
    if genotype is None or any(allele is None for allele in genotype):
        return 0.5
    normalized = tuple(sorted(int(allele) for allele in genotype))
    if normalized == tuple(sorted(state_zero)):
        return 0.01
    if normalized == tuple(sorted(state_one)):
        return 0.99
    return 0.5


def read_vcf(
    path: str | Path,
    *,
    chromosome: str | None = None,
    samples: Iterable[str] | None = None,
    parents: tuple[str, str] | None = None,
    cross_design: str = "auto",
) -> LinkageData:
    """Read a VCF, bgzipped VCF, or BCF as binary linkage data.

    The loader accepts biallelic SNPs from one chromosome. ``GT`` hard calls are
    converted to 0.01/0.99 and missing or incompatible calls to 0.5. When ``PL``
    or ``GL`` is present, the corresponding two-state genotype likelihood is
    retained as a probability.

    For already phased binary-cross VCFs, the default ``cross_design="auto"``
    infers the two observed genotype classes at each marker. To orient alleles
    consistently from parental samples, pass ``parents=(state0, state1)`` and set
    ``cross_design`` to ``"backcross"``, ``"ril"``, or ``"doubled_haploid"``.
    In a backcross, the first parent is the recurrent parent. Parent samples are
    excluded from offspring unless explicitly included in ``samples``.
    """

    try:
        import pysam
    except ImportError as error:  # pragma: no cover - declared package dependency
        raise ImportError("VCF/BCF input requires pysam>=0.23") from error

    source = Path(path)
    lower_name = source.name.lower()
    if not any(lower_name.endswith(suffix) for suffix in _VCF_SUFFIXES):
        raise ValueError("VCF input must end in .vcf, .vcf.gz, or .bcf")
    if cross_design not in {"auto", "backcross", "ril", "doubled_haploid"}:
        raise ValueError(
            "cross_design must be auto, backcross, ril, or doubled_haploid"
        )

    with pysam.VariantFile(str(source)) as variants:
        available = tuple(variants.header.samples)
        if parents is not None:
            if len(parents) != 2 or parents[0] == parents[1]:
                raise ValueError("parents must name two distinct VCF samples")
            missing_parents = [parent for parent in parents if parent not in available]
            if missing_parents:
                raise ValueError(f"parent sample not found in VCF: {missing_parents[0]}")
            if cross_design == "auto":
                raise ValueError(
                    "set cross_design when parents are supplied (backcross, ril, "
                    "or doubled_haploid)"
                )
        selected = (
            tuple(samples)
            if samples is not None
            else tuple(
                sample
                for sample in available
                if parents is None or sample not in parents
            )
        )
        if len(selected) < 2:
            raise ValueError("VCF input must contain at least two offspring samples")
        unknown = [sample for sample in selected if sample not in available]
        if unknown:
            raise ValueError(f"sample not found in VCF: {unknown[0]}")
        if len(set(selected)) != len(selected):
            raise ValueError("samples must be unique")

        marker_names: list[str] = []
        positions: list[float] = []
        columns: list[list[float]] = []
        contigs: set[str] = set()
        seen_names: set[str] = set()
        # Iterate sequentially so a plain or unindexed VCF can still be filtered
        # by chromosome. Indexed bgzipped VCF and BCF inputs work the same way.
        for record in variants:
            if chromosome is not None and record.contig != chromosome:
                continue
            if len(record.ref) != 1 or record.alts is None or len(record.alts) != 1:
                continue
            if len(record.alts[0]) != 1:
                continue
            if record.filter.keys() and "PASS" not in record.filter.keys():
                continue

            state_zero: tuple[int, ...] | None = None
            state_one: tuple[int, ...] | None = None
            if parents is not None:
                parent_calls = [record.samples[parent].get("GT") for parent in parents]
                if any(
                    call is None
                    or len(call) not in (1, 2)
                    or any(allele is None for allele in call)
                    for call in parent_calls
                ):
                    continue
                parent_genotypes = [
                    tuple(sorted(int(allele) for allele in call))
                    for call in parent_calls
                    if call is not None
                ]
                if any(len(set(call)) != 1 for call in parent_genotypes):
                    continue
                allele_zero = parent_genotypes[0][0]
                allele_one = parent_genotypes[1][0]
                if allele_zero == allele_one or {allele_zero, allele_one} != {0, 1}:
                    continue
                called_ploidies = {
                    len(genotype)
                    for sample in selected
                    if (genotype := record.samples[sample].get("GT")) is not None
                    and all(allele is not None for allele in genotype)
                }
                offspring_ploidy = (
                    1
                    if cross_design == "doubled_haploid" and called_ploidies == {1}
                    else 2
                )
                if cross_design == "backcross":
                    state_zero = (allele_zero, allele_zero)
                    state_one = tuple(sorted((allele_zero, allele_one)))
                else:
                    state_zero = (allele_zero,) * offspring_ploidy
                    state_one = (allele_one,) * offspring_ploidy
            else:
                observed = {
                    tuple(sorted(int(allele) for allele in genotype))
                    for sample in selected
                    if (genotype := record.samples[sample].get("GT")) is not None
                    and len(genotype) in (1, 2)
                    and all(allele in (0, 1) for allele in genotype)
                }
                if len(observed) != 2:
                    continue
                state_zero, state_one = sorted(
                    observed,
                    key=lambda genotype: (sum(genotype) / len(genotype), genotype),
                )

            assert state_zero is not None and state_one is not None
            marker = (
                record.id
                if record.id not in (None, ".")
                else f"{record.contig}:{record.pos}"
            )
            if marker in seen_names:
                raise ValueError(f"VCF marker names must be unique: {marker}")
            seen_names.add(marker)
            marker_names.append(marker)
            positions.append(float(record.pos))
            contigs.add(record.contig)
            columns.append([
                _call_probability(record.samples[sample], state_zero, state_one)
                for sample in selected
            ])

    if not columns:
        raise ValueError("VCF contains no usable biallelic two-state SNP markers")
    if chromosome is None and len(contigs) != 1:
        raise ValueError(
            "VCF contains multiple chromosomes; pass chromosome= to fit one linkage group"
        )
    label = chromosome if chromosome is not None else next(iter(contigs))
    probabilities = np.asarray(columns, dtype=np.float64).T
    return LinkageData(
        probabilities,
        tuple(marker_names),
        label=label,
        physical_positions=np.asarray(positions, dtype=np.float64),
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

    def plot(
        self,
        path: str | Path | None = None,
        *,
        colors: tuple[str, str, str] = ("#fde0dd", "#fa9fb5", "#c51b8a"),
    ):
        """Create the standard overview figure and optionally save it.

        ``colors`` defines the state-0, uncertain, and state-1 color anchors.
        """

        from .plotting import plot_map

        return plot_map(self, path=path, colors=colors)

    def plot_marker_order(self, path: str | Path | None = None):
        """Compare the input marker order with the fitted marker order."""

        from .plotting import plot_marker_order

        return plot_marker_order(self, path=path)


@dataclass(frozen=True)
class LikelihoodMap:
    """A robust likelihood-MDS order with model-stability rank bands."""

    data: LinkageData
    result: LikelihoodMDSEnsembleResult

    @property
    def ordered_markers(self) -> list[str]:
        return self.result.ordered_names()

    def summary(self) -> dict[str, int | float | str | bool | list[object]]:
        return {
            "method": "SoftMap-LMDS-Ensemble",
            "status": self.result.status,
            "offspring": int(self.data.probabilities.shape[0]),
            "markers": int(self.data.probabilities.shape[1]),
            "candidate_orders": int(self.result.candidate_orders.shape[0]),
            "selected_config": list(self.result.selected_config),
            "stability_mass": self.result.stability_mass,
            "stability_comparable_pair_fraction": (
                self.result.stability_comparable_pair_fraction
            ),
            "unanimous_family_veto_triggered": (
                self.result.unanimous_family_veto_triggered
            ),
        }

    def marker_table(self) -> list[dict[str, str | int]]:
        rank = np.empty(self.result.order.size, dtype=np.int64)
        rank[self.result.order] = np.arange(self.result.order.size)
        return [
            {
                "marker": name,
                "order_rank": int(rank[index]),
                "stability_rank_left": int(self.result.interval_left[index]),
                "stability_rank_right": int(self.result.interval_right[index]),
            }
            for index, name in enumerate(self.data.marker_names)
        ]


def _as_linkage_data(
    data: LinkageData | ArrayLike | str | Path,
    marker_names: Iterable[str] | None,
) -> LinkageData:
    if isinstance(data, LinkageData):
        if marker_names is not None:
            raise ValueError("marker_names is already included in LinkageData")
        return data
    if isinstance(data, (str, Path)):
        if marker_names is not None:
            raise ValueError("marker_names cannot be used with VCF/BCF input")
        return read_vcf(data)
    probabilities = np.asarray(data, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("data must be a 2D offspring-by-marker matrix")
    names = (
        tuple(marker_names)
        if marker_names is not None
        else tuple(f"m{i + 1}" for i in range(probabilities.shape[1]))
    )
    return LinkageData(probabilities, names)


def fit_likelihood(
    data: LinkageData | ArrayLike | str | Path,
    marker_names: Iterable[str] | None = None,
    *,
    stability_mass: float = 0.90,
    maximum_smacof_iterations: int = 500,
) -> LikelihoodMap:
    """Fit SoftMap's robust total order and confidence-first stability bands."""

    linkage_data = _as_linkage_data(data, marker_names)
    result = fit_likelihood_mds_ensemble(
        linkage_data.probabilities,
        linkage_data.marker_names,
        stability_mass=stability_mass,
        maximum_smacof_iterations=maximum_smacof_iterations,
    )
    return LikelihoodMap(linkage_data, result)


def fit(
    data: LinkageData | ArrayLike | str | Path,
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
        A VCF/BCF path, a :class:`LinkageData` object, or an offspring-by-marker
        probability matrix. Values in a matrix must lie between zero and one.
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

    linkage_data = _as_linkage_data(data, marker_names)

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
