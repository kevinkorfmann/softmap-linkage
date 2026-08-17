"""Small public interface for fitting and inspecting linkage maps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    F2MapResult,
    LikelihoodMDSEnsembleResult,
    SoftMapResult,
    fit_f2_likelihood_map,
    fit_scalable_likelihood_mds_ensemble,
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
                raise ValueError(
                    "reference_positions must contain one value per marker"
                )
            object.__setattr__(self, "reference_positions", positions)
        if self.physical_positions is not None:
            physical = np.asarray(self.physical_positions, dtype=np.float64)
            if physical.shape != (probabilities.shape[1],):
                raise ValueError("physical_positions must contain one value per marker")
            object.__setattr__(self, "physical_positions", physical)
        object.__setattr__(self, "probabilities", probabilities)

    def shuffled(self, seed: int | None = 1) -> LinkageData:
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
class F2LinkageData:
    """F2 genotype posteriors in offspring-by-marker-by-(AA, AB, BB) form."""

    probabilities: NDArray[np.float64]
    marker_names: tuple[str, ...]
    reference_positions: NDArray[np.float64] | None = None
    label: str | None = None
    physical_positions: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim != 3 or probabilities.shape[2] != 3:
            raise ValueError("F2 probabilities must have shape (offspring, markers, 3)")
        if probabilities.shape[1] != len(self.marker_names):
            raise ValueError("marker_names length must match the number of markers")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("F2 probabilities must be finite and nonnegative")
        if not np.allclose(np.sum(probabilities, axis=2), 1.0):
            raise ValueError("F2 genotype probabilities must sum to one")
        if self.reference_positions is not None:
            reference = np.asarray(self.reference_positions, dtype=np.float64)
            if reference.shape != (probabilities.shape[1],):
                raise ValueError(
                    "reference_positions must contain one value per marker"
                )
            object.__setattr__(self, "reference_positions", reference)
        if self.physical_positions is not None:
            physical = np.asarray(self.physical_positions, dtype=np.float64)
            if physical.shape != (probabilities.shape[1],):
                raise ValueError("physical_positions must contain one value per marker")
            object.__setattr__(self, "physical_positions", physical)
        object.__setattr__(self, "probabilities", probabilities)

    def shuffled(self, seed: int | None = 1) -> F2LinkageData:
        """Return a copy with marker axes reproducibly shuffled."""

        order = np.random.default_rng(seed).permutation(self.probabilities.shape[1])
        return F2LinkageData(
            self.probabilities[:, order, :],
            tuple(self.marker_names[int(index)] for index in order),
            (
                self.reference_positions[order]
                if self.reference_positions is not None
                else None
            ),
            f"{self.label}, shuffled input" if self.label else "Shuffled input",
            (
                self.physical_positions[order]
                if self.physical_positions is not None
                else None
            ),
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


def _call_f2_probabilities(call, allele_zero: int, allele_one: int) -> np.ndarray:
    """Convert one offspring VCF call to AA/AB/BB posterior probabilities."""

    states = (
        (allele_zero, allele_zero),
        tuple(sorted((allele_zero, allele_one))),
        (allele_one, allele_one),
    )
    indices = tuple(_diploid_genotype_index(state) for state in states)
    prior = np.asarray((0.25, 0.50, 0.25), dtype=np.float64)
    for field, scale in (("PL", -0.1), ("GL", 1.0)):
        values = call.get(field)
        if values is None or any(index is None for index in indices):
            continue
        resolved = tuple(int(index) for index in indices if index is not None)
        if max(resolved) >= len(values):
            continue
        selected = [values[index] for index in resolved]
        if any(value is None for value in selected):
            continue
        logs = np.asarray(selected, dtype=np.float64) * scale
        likelihood = 10.0 ** (logs - np.max(logs))
        posterior = likelihood * prior
        return posterior / posterior.sum()

    genotype = call.get("GT")
    if genotype is None or any(allele is None for allele in genotype):
        return prior.copy()
    normalized = tuple(sorted(int(allele) for allele in genotype))
    if normalized not in states:
        return prior.copy()
    likelihood = np.full(3, 0.005, dtype=np.float64)
    likelihood[states.index(normalized)] = 0.99
    posterior = likelihood * prior
    return posterior / posterior.sum()


def read_vcf(
    path: str | Path,
    *,
    chromosome: str | None = None,
    samples: Iterable[str] | None = None,
    parents: tuple[str, str] | None = None,
    cross_design: str = "auto",
) -> LinkageData | F2LinkageData:
    """Read one chromosome from a VCF, bgzipped VCF, or BCF.

    The loader accepts biallelic SNPs from one chromosome. ``GT`` hard calls are
    converted to 0.01/0.99 and missing or incompatible calls to 0.5. When ``PL``
    or ``GL`` is present, the corresponding two-state genotype likelihood is
    retained as a probability.

    For an F2, ``cross_design="f2"`` retains all three parental genotype
    posteriors (AA, AB, BB); two homozygous parent samples are required. For
    already phased binary-cross VCFs, the default ``cross_design="auto"``
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
    if cross_design not in {"auto", "backcross", "ril", "doubled_haploid", "f2"}:
        raise ValueError(
            "cross_design must be auto, backcross, ril, doubled_haploid, or f2"
        )

    with pysam.VariantFile(str(source)) as variants:
        available = tuple(variants.header.samples)
        if parents is not None:
            if len(parents) != 2 or parents[0] == parents[1]:
                raise ValueError("parents must name two distinct VCF samples")
            missing_parents = [parent for parent in parents if parent not in available]
            if missing_parents:
                raise ValueError(
                    f"parent sample not found in VCF: {missing_parents[0]}"
                )
            if cross_design == "auto":
                raise ValueError(
                    "set cross_design when parents are supplied (backcross, ril, "
                    "doubled_haploid, or f2)"
                )
        if cross_design == "f2" and parents is None:
            raise ValueError("cross_design='f2' requires two homozygous parents")
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
        columns: list[object] = []
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
            if record.filter.keys() and "PASS" not in record.filter:
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
                elif cross_design == "f2":
                    state_zero = (allele_zero, allele_zero)
                    state_one = (allele_one, allele_one)
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
            if cross_design == "f2":
                assert parents is not None
                columns.append(
                    [
                        _call_f2_probabilities(
                            record.samples[sample], state_zero[0], state_one[0]
                        )
                        for sample in selected
                    ]
                )
            else:
                columns.append(
                    [
                        _call_probability(record.samples[sample], state_zero, state_one)
                        for sample in selected
                    ]
                )

    if not columns:
        raise ValueError("VCF contains no usable biallelic SNP markers")
    if chromosome is None and len(contigs) != 1:
        raise ValueError(
            "VCF contains multiple chromosomes; pass chromosome= to fit one linkage group"
        )
    label = chromosome if chromosome is not None else next(iter(contigs))
    if cross_design == "f2":
        probabilities = np.transpose(np.asarray(columns, dtype=np.float64), (1, 0, 2))
        return F2LinkageData(
            probabilities,
            tuple(marker_names),
            label=label,
            physical_positions=np.asarray(positions, dtype=np.float64),
        )
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
            rows.append(
                {
                    "marker": name,
                    "bin": group,
                    "order_rank": int(rank_by_group[group]),
                    "is_representative": marker == representative,
                    "framework_rank": framework_rank.get(group),
                    "interval_left": int(result.interval_left[group]),
                    "interval_right": int(result.interval_right[group]),
                }
            )
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
class F2Map:
    """A complete-information F2 linkage map and its source genotype posteriors."""

    data: F2LinkageData
    result: F2MapResult

    @property
    def ordered_markers(self) -> list[str]:
        return self.result.ordered_names()

    def summary(self) -> dict[str, int | float | str | bool | list[object] | None]:
        distances = self.result.genetic_distances
        return {
            "method": "SoftMap-F2",
            "status": self.result.status,
            "offspring": int(self.data.probabilities.shape[0]),
            "markers": int(self.data.probabilities.shape[1]),
            "ordering_method": self.result.ordering_method,
            "physical_scaffold_used": self.result.physical_scaffold_used,
            "selected_config": list(self.result.selected_config),
            "mean_genotype_certainty": self.result.mean_genotype_certainty,
            "distance_status": distances.status,
            "distance_method": distances.method,
            "map_length_cm": distances.map_length_cm,
            "distance_informative_pair_count": distances.informative_pair_count,
        }

    def marker_table(self) -> list[dict[str, str | int | float | None]]:
        rank = np.empty(self.result.order.size, dtype=np.int64)
        rank[self.result.order] = np.arange(self.result.order.size)
        de_novo_rank = np.empty_like(rank)
        de_novo_rank[self.result.de_novo_order] = np.arange(rank.size)
        positions = self.result.genetic_distances.marker_positions_cm
        return [
            {
                "marker": name,
                "order_rank": int(rank[index]),
                "de_novo_order_rank": int(de_novo_rank[index]),
                "stability_rank_left": int(self.result.interval_left[index]),
                "stability_rank_right": int(self.result.interval_right[index]),
                "genetic_position_cm": (
                    float(positions[index]) if np.isfinite(positions[index]) else None
                ),
            }
            for index, name in enumerate(self.data.marker_names)
        ]

    def plot(self, path: str | Path | None = None):
        """Plot F2 genotype dosage and inferred genetic positions."""

        from .plotting import plot_f2_map

        return plot_f2_map(self, path=path)


@dataclass(frozen=True)
class LikelihoodMap:
    """A robust likelihood-MDS order with model-stability rank bands."""

    data: LinkageData
    result: LikelihoodMDSEnsembleResult

    @property
    def ordered_markers(self) -> list[str]:
        return self.result.ordered_names()

    def summary(
        self,
    ) -> dict[str, int | float | str | bool | list[object] | None]:
        distances = self.result.genetic_distances
        return {
            "method": "SoftMap-LMDS-Ensemble",
            "status": self.result.status,
            "offspring": int(self.data.probabilities.shape[0]),
            "markers": int(self.data.probabilities.shape[1]),
            "candidate_orders": int(self.result.candidate_orders.shape[0]),
            "likelihood_bins": self.result.likelihood_bin_count,
            "binning_method": self.result.binning_method,
            "bin_neighbor_count": self.result.bin_neighbor_count,
            "bin_neighbor_projection_dimensions": (
                self.result.bin_neighbor_projection_dimensions
            ),
            "ordering_method": self.result.ordering_method,
            "landmark_count": self.result.landmark_count,
            "landmark_neighbor_count": self.result.landmark_neighbor_count,
            "landmark_support_exponent": (self.result.landmark_support_exponent),
            "large_scale_rescue_triggered": (self.result.large_scale_rescue_triggered),
            "low_certainty_stability_mass_cap_applied": (
                self.result.low_certainty_stability_mass_cap_applied
            ),
            "posterior_calibration_triggered": (
                self.result.posterior_calibration_triggered
            ),
            "posterior_calibration_temperature": (
                self.result.posterior_calibration_temperature
            ),
            "uncalibrated_mean_genotype_certainty": (
                self.result.uncalibrated_mean_genotype_certainty
            ),
            "uncalibrated_distance_median_absolute_residual_morgan": (
                self.result.uncalibrated_distance_median_absolute_residual_morgan
            ),
            "selected_config": list(self.result.selected_config),
            "selection_method": self.result.selection_method,
            "weighted_objective_support_filter_applied": (
                self.result.weighted_objective_support_filter_applied
            ),
            "penalized_curve_effective_degrees_of_freedom": (
                self.result.penalized_curve_effective_degrees_of_freedom
            ),
            "posterior_refinement_weight": (self.result.posterior_refinement_weight),
            "posterior_refinement_passes_applied": (
                self.result.posterior_refinement_passes_applied
            ),
            "second_refinement_uncertain_pair_threshold": (
                self.result.second_refinement_uncertain_pair_threshold
            ),
            "stability_rank_padding": self.result.stability_rank_padding,
            "minimum_stability_comparable_pair_fraction": (
                self.result.minimum_stability_comparable_pair_fraction
            ),
            "stability_mass": self.result.stability_mass,
            "stability_comparable_pair_fraction": (
                self.result.stability_comparable_pair_fraction
            ),
            "unanimous_family_veto_triggered": (
                self.result.unanimous_family_veto_triggered
            ),
            "distance_status": distances.status if distances is not None else None,
            "distance_method": distances.method if distances is not None else None,
            "map_length_cm": (
                distances.map_length_cm if distances is not None else None
            ),
            "distance_informative_pair_count": (
                distances.informative_pair_count if distances is not None else None
            ),
            "distance_segment_count": (
                distances.segment_count if distances is not None else None
            ),
            "distance_rank_span_weight_exponent": (
                distances.rank_span_weight_exponent if distances is not None else None
            ),
        }

    def marker_table(self) -> list[dict[str, str | int | float | bool | None]]:
        rank = np.empty(self.result.order.size, dtype=np.int64)
        rank[self.result.order] = np.arange(self.result.order.size)
        is_representative = np.zeros(self.result.order.size, dtype=bool)
        is_representative[self.result.bin_representatives] = True
        distances = self.result.genetic_distances
        return [
            {
                "marker": name,
                "order_rank": int(rank[index]),
                "reported_position": int(self.result.reported_positions[index]),
                "likelihood_bin": int(self.result.bin_membership[index]),
                "is_bin_representative": bool(is_representative[index]),
                "stability_rank_left": int(self.result.interval_left[index]),
                "stability_rank_right": int(self.result.interval_right[index]),
                "genetic_position_cm": (
                    float(distances.marker_positions_cm[index])
                    if (
                        distances is not None
                        and np.isfinite(distances.marker_positions_cm[index])
                    )
                    else None
                ),
            }
            for index, name in enumerate(self.data.marker_names)
        ]


def _as_linkage_data(
    data: LinkageData | F2LinkageData | ArrayLike | str | Path,
    marker_names: Iterable[str] | None,
) -> LinkageData | F2LinkageData:
    if isinstance(data, (LinkageData, F2LinkageData)):
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
    data: LinkageData | F2LinkageData | ArrayLike | str | Path,
    marker_names: Iterable[str] | None = None,
    *,
    stability_mass: float = 0.90,
    posterior_refinement_weight: float = 0.75,
    maximum_posterior_refinement_passes: int = 2,
    second_refinement_uncertain_pair_threshold: float = 0.03,
    stability_rank_padding: int = 1,
    minimum_stability_comparable_pair_fraction: float = 0.35,
    maximum_smacof_iterations: int = 500,
    automatic_posterior_calibration: bool = True,
) -> LikelihoodMap | F2Map:
    """Fit SoftMap's robust order and confidence-first stability bands.

    Up to 500 markers use the promoted dense likelihood-MDS ensemble exactly.
    Larger inputs automatically use the promoted likelihood-binned partial-order
    path; co-segregating markers share ``reported_position`` in the marker table.
    """

    linkage_data = _as_linkage_data(data, marker_names)
    if isinstance(linkage_data, F2LinkageData):
        return fit_f2(
            linkage_data,
            stability_mass=stability_mass,
            stability_rank_padding=stability_rank_padding,
            maximum_smacof_iterations=maximum_smacof_iterations,
        )
    result = fit_scalable_likelihood_mds_ensemble(
        linkage_data.probabilities,
        linkage_data.marker_names,
        stability_mass=stability_mass,
        posterior_refinement_weight=posterior_refinement_weight,
        maximum_posterior_refinement_passes=(maximum_posterior_refinement_passes),
        second_refinement_uncertain_pair_threshold=(
            second_refinement_uncertain_pair_threshold
        ),
        stability_rank_padding=stability_rank_padding,
        minimum_stability_comparable_pair_fraction=(
            minimum_stability_comparable_pair_fraction
        ),
        maximum_smacof_iterations=maximum_smacof_iterations,
        automatic_posterior_calibration=automatic_posterior_calibration,
    )
    return LikelihoodMap(linkage_data, result)


def fit_f2(
    data: F2LinkageData | str | Path,
    *,
    chromosome: str | None = None,
    parents: tuple[str, str] | None = None,
    use_physical_scaffold: bool = False,
    stability_mass: float = 0.90,
    stability_rank_padding: int = 1,
    maximum_smacof_iterations: int = 200,
) -> F2Map:
    """Fit a linkage map from complete AA/AB/BB F2 information.

    Set ``use_physical_scaffold=True`` to use VCF/BCF positions as an explicit
    reference-guided order. Recombination fractions and Kosambi centimorgan
    coordinates are always estimated from the F2 genotype information.
    """

    if isinstance(data, (str, Path)):
        if parents is None:
            raise ValueError(
                "fit_f2 with a VCF/BCF path requires parents=(parent0, parent1)"
            )
        linkage_data = read_vcf(
            data,
            chromosome=chromosome,
            parents=parents,
            cross_design="f2",
        )
    else:
        if chromosome is not None or parents is not None:
            raise ValueError("chromosome and parents are only valid with VCF/BCF input")
        linkage_data = data
    if not isinstance(linkage_data, F2LinkageData):
        raise TypeError("fit_f2 requires F2LinkageData or an F2 VCF path")
    result = fit_f2_likelihood_map(
        linkage_data.probabilities,
        linkage_data.marker_names,
        physical_positions=linkage_data.physical_positions,
        use_physical_scaffold=use_physical_scaffold,
        stability_mass=stability_mass,
        stability_rank_padding=stability_rank_padding,
        maximum_smacof_iterations=maximum_smacof_iterations,
    )
    return F2Map(linkage_data, result)


def fit(
    data: LinkageData | F2LinkageData | ArrayLike | str | Path,
    marker_names: Iterable[str] | None = None,
    *,
    bootstrap: int = 20,
    confidence: float = 0.8,
    seed: int | None = 1,
    bin_threshold: float | None = 0.01,
) -> Map | F2Map:
    """Fit a confidence-aware linkage map with practical defaults.

    Parameters
    ----------
    data
        A VCF/BCF path, a :class:`LinkageData` or :class:`F2LinkageData` object,
        or an offspring-by-marker binary probability matrix. F2 VCF paths should
        first be loaded with :func:`read_vcf` and ``cross_design="f2"``.
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
    if isinstance(linkage_data, F2LinkageData):
        return fit_f2(linkage_data)

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
