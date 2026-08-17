"""Simulation and evaluation utilities for the SoftMap go/no-go experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau

from .core import GeneticMapDistances, SoftMapResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SimulatedCross:
    probabilities: FloatArray
    latent_states: IntArray
    true_positions: FloatArray
    marker_names: tuple[str, ...]
    input_to_truth: IntArray
    reference_reads: IntArray | None = None
    alternate_reads: IntArray | None = None
    cross_design: str = "binary_parental_origin"


@dataclass(frozen=True)
class SimulatedF2Cross:
    """Complete tri-state F2 simulation with known marker order."""

    probabilities: FloatArray
    genotypes: IntArray
    true_positions: FloatArray
    marker_names: tuple[str, ...]
    input_to_truth: IntArray
    reference_reads: IntArray | None = None
    alternate_reads: IntArray | None = None


def simulate_f2(
    *,
    n_offspring: int = 300,
    n_markers: int = 80,
    map_length_morgan: float = 1.0,
    missing_probability: float = 0.0,
    mean_depth: float | None = None,
    read_error: float = 0.01,
    random_seed: int | None = None,
    shuffle_markers: bool = True,
) -> SimulatedF2Cross:
    """Simulate two F1 gametes and AA/AB/BB genotype probabilities.

    With the default ``mean_depth=None``, observed genotypes are one-hot and
    missing observations receive the Mendelian 1:2:1 prior. When ``mean_depth``
    is supplied, Poisson read depth and binomial AA/AB/BB allele emissions are
    converted to normalized genotype probabilities under the same prior.
    """

    if n_offspring < 2 or n_markers < 2:
        raise ValueError("n_offspring and n_markers must be at least two")
    if map_length_morgan <= 0.0 or not np.isfinite(map_length_morgan):
        raise ValueError("map_length_morgan must be positive and finite")
    if not 0.0 <= missing_probability <= 1.0:
        raise ValueError("missing_probability must lie in [0, 1]")
    if mean_depth is not None and (not np.isfinite(mean_depth) or mean_depth < 0.0):
        raise ValueError("mean_depth must be finite and nonnegative")
    if not np.isfinite(read_error) or not 0.0 < read_error < 0.5:
        raise ValueError("read_error must lie strictly between zero and 0.5")
    rng = np.random.default_rng(random_seed)
    positions = np.sort(rng.uniform(0.0, map_length_morgan, n_markers))
    gametes = np.empty((n_offspring, 2, n_markers), dtype=np.int64)
    for offspring in range(n_offspring):
        for gamete in range(2):
            start = int(rng.integers(0, 2))
            crossovers = np.sort(
                rng.uniform(0.0, map_length_morgan, rng.poisson(map_length_morgan))
            )
            gametes[offspring, gamete] = start ^ (
                np.searchsorted(crossovers, positions, side="right") % 2
            )
    genotypes = np.sum(gametes, axis=1)
    prior = np.asarray((0.25, 0.50, 0.25), dtype=np.float64)
    reference_reads: IntArray | None = None
    alternate_reads: IntArray | None = None
    if mean_depth is None:
        probabilities = np.zeros((n_offspring, n_markers, 3), dtype=np.float64)
        offspring_index, marker_index = np.indices(genotypes.shape)
        probabilities[offspring_index, marker_index, genotypes] = 1.0
        if missing_probability > 0.0:
            missing = rng.random(genotypes.shape) < missing_probability
            probabilities[missing] = prior
    else:
        depth = rng.poisson(mean_depth, size=genotypes.shape)
        allele_probability = np.asarray(
            (read_error, 0.5, 1.0 - read_error),
            dtype=np.float64,
        )
        alternate_reads = rng.binomial(depth, allele_probability[genotypes])
        reference_reads = depth - alternate_reads
        if missing_probability > 0.0:
            missing = rng.random(genotypes.shape) < missing_probability
            reference_reads[missing] = 0
            alternate_reads[missing] = 0
        log_likelihood = (
            alternate_reads[:, :, None] * np.log(allele_probability)[None, None, :]
            + reference_reads[:, :, None] * np.log1p(-allele_probability)[None, None, :]
            + np.log(prior)[None, None, :]
        )
        log_likelihood -= np.max(log_likelihood, axis=2, keepdims=True)
        probabilities = np.exp(log_likelihood)
        probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    permutation = (
        rng.permutation(n_markers).astype(np.int64)
        if shuffle_markers
        else np.arange(n_markers, dtype=np.int64)
    )
    return SimulatedF2Cross(
        probabilities[:, permutation, :],
        genotypes[:, permutation],
        positions,
        tuple(f"m{truth:06d}" for truth in permutation),
        permutation,
        (reference_reads[:, permutation] if reference_reads is not None else None),
        (alternate_reads[:, permutation] if alternate_reads is not None else None),
    )


def simulate_backcross(
    *,
    n_offspring: int = 200,
    n_markers: int = 10_000,
    map_length_morgan: float = 1.0,
    mean_depth: float = 2.0,
    read_error: float = 0.01,
    contamination: float = 0.0,
    missing_probability: float = 0.0,
    heterozygous_state: bool = False,
    random_seed: int | None = None,
    shuffle_markers: bool = True,
) -> SimulatedCross:
    """Simulate a phased binary cross and read-derived genotype posteriors.

    The default is the original binary parental-origin emission model, appropriate
    for haploid or doubled-haploid-like evidence.  With ``heterozygous_state=True``,
    state zero is ``AA`` and state one is ``AB`` with 50:50 allele sampling, giving
    a biologically literal diploid backcross for raw-read comparators such as
    GUSMap. ``contamination`` mixes each read's allele probability toward 0.5
    without changing the inference model. ``missing_probability`` independently
    drops complete offspring-marker read observations after sequencing; dropped
    observations retain zero reads and posterior probability 0.5.
    """

    if n_offspring < 2 or n_markers < 2:
        raise ValueError("n_offspring and n_markers must be at least two")
    if not np.isfinite(mean_depth) or mean_depth < 0.0:
        raise ValueError("mean_depth must be finite and nonnegative")
    if not np.isfinite(read_error) or not 0.0 < read_error < 0.5:
        raise ValueError("read_error must lie strictly between zero and 0.5")
    if not np.isfinite(contamination) or not 0.0 <= contamination <= 1.0:
        raise ValueError("contamination must lie between zero and one")
    if not np.isfinite(missing_probability) or not 0.0 <= missing_probability <= 1.0:
        raise ValueError("missing_probability must lie between zero and one")
    rng = np.random.default_rng(random_seed)
    positions = np.sort(rng.uniform(0.0, map_length_morgan, n_markers))
    states = np.empty((n_offspring, n_markers), dtype=np.int64)
    for individual in range(n_offspring):
        start = int(rng.integers(0, 2))
        crossovers = np.sort(
            rng.uniform(0.0, map_length_morgan, rng.poisson(map_length_morgan))
        )
        states[individual] = start ^ (
            np.searchsorted(crossovers, positions, side="right") % 2
        )

    depth = rng.poisson(mean_depth, size=states.shape)
    state_one_probability = 0.5 if heterozygous_state else 1.0 - read_error
    allele_probability = np.where(states == 1, state_one_probability, read_error)
    allele_probability = (
        allele_probability * (1.0 - contamination) + 0.5 * contamination
    )
    alternate = rng.binomial(depth, allele_probability)
    reference = depth - alternate
    if missing_probability > 0.0:
        missing = rng.random(states.shape) < missing_probability
        alternate[missing] = 0
        reference[missing] = 0
        depth[missing] = 0
    log_likelihood_0 = alternate * np.log(read_error) + reference * np.log1p(
        -read_error
    )
    log_likelihood_1 = (
        depth * np.log(0.5)
        if heterozygous_state
        else alternate * np.log1p(-read_error) + reference * np.log(read_error)
    )
    maximum = np.maximum(log_likelihood_0, log_likelihood_1)
    likelihood_0 = np.exp(log_likelihood_0 - maximum)
    likelihood_1 = np.exp(log_likelihood_1 - maximum)
    probabilities = likelihood_1 / (likelihood_0 + likelihood_1)
    probabilities[depth == 0] = 0.5

    if shuffle_markers:
        permutation = rng.permutation(n_markers).astype(np.int64)
    else:
        permutation = np.arange(n_markers, dtype=np.int64)
    names = tuple(f"m{truth_index:06d}" for truth_index in permutation)
    return SimulatedCross(
        probabilities[:, permutation],
        states[:, permutation],
        positions,
        names,
        permutation,
        reference[:, permutation],
        alternate[:, permutation],
        "diploid_backcross" if heterozygous_state else "binary_parental_origin",
    )


def evaluate_genetic_map_distances(
    distances: GeneticMapDistances,
    bin_representatives: IntArray,
    cross: SimulatedCross,
) -> dict[str, float | int | str | None]:
    """Evaluate fitted cM coordinates against simulated Haldane-map truth."""

    representatives = np.asarray(bin_representatives, dtype=np.int64)
    marker_count = cross.probabilities.shape[1]
    if representatives.shape != distances.bin_positions_cm.shape:
        raise ValueError("bin representatives must match fitted distance bins")
    if np.any((representatives < 0) | (representatives >= marker_count)):
        raise ValueError("bin representatives contain an invalid marker index")
    true_marker_positions_cm = 100.0 * cross.true_positions[cross.input_to_truth]
    true_span_cm = float(np.ptp(cross.true_positions) * 100.0)
    estimated_length_cm = distances.map_length_cm
    common = {
        "distance_status": distances.status,
        "distance_method": distances.method,
        "distance_candidate_pair_count": distances.candidate_pair_count,
        "distance_informative_pair_count": distances.informative_pair_count,
        "distance_segment_count": distances.segment_count,
        "distance_minimum_pair_recombination": (distances.minimum_pair_recombination),
        "distance_maximum_pair_recombination": (distances.maximum_pair_recombination),
        "distance_composite_rmse_morgan": distances.composite_rmse_morgan,
        "distance_composite_median_absolute_residual_morgan": (
            distances.composite_median_absolute_residual_morgan
        ),
        "distance_composite_p90_absolute_residual_morgan": (
            distances.composite_p90_absolute_residual_morgan
        ),
        "true_map_length_cm": true_span_cm,
        "estimated_map_length_cm": estimated_length_cm,
    }
    if estimated_length_cm is None:
        return {
            **common,
            "map_length_relative_error": None,
            "map_length_inflation": None,
            "adjacent_recombination_mean_absolute_error": None,
            "adjacent_recombination_root_mean_square_error": None,
            "multipoint_adjacent_recombination_mean_absolute_error": None,
            "regularized_adjacent_recombination_mean_absolute_error": None,
            "pairwise_adjacent_recombination_mean_absolute_error": None,
            "representative_position_rmse_cm": None,
            "endpoint_truth_span_fraction": None,
        }

    ordered_representatives = representatives[distances.ordered_bins]
    true_ordered_cm = true_marker_positions_cm[ordered_representatives]
    estimated_ordered_cm = distances.bin_positions_cm[distances.ordered_bins]
    if np.corrcoef(estimated_ordered_cm, true_ordered_cm)[0, 1] < 0.0:
        true_ordered_cm = -true_ordered_cm
    offset = float(np.median(estimated_ordered_cm - true_ordered_cm))
    position_error = estimated_ordered_cm - (true_ordered_cm + offset)
    true_adjacent_morgan = np.abs(np.diff(true_ordered_cm)) / 100.0
    true_adjacent_recombination = 0.5 * (1.0 - np.exp(-2.0 * true_adjacent_morgan))
    adjacent_error = (
        distances.adjacent_local_recombination - true_adjacent_recombination
    )
    multipoint_adjacent_error = (
        distances.adjacent_multipoint_recombination - true_adjacent_recombination
    )
    regularized_adjacent_error = (
        distances.adjacent_recombination - true_adjacent_recombination
    )
    pairwise_adjacent_error = (
        distances.adjacent_pairwise_recombination - true_adjacent_recombination
    )
    endpoint_span = abs(float(true_ordered_cm[-1] - true_ordered_cm[0]))
    return {
        **common,
        "map_length_relative_error": (
            float(estimated_length_cm / true_span_cm - 1.0)
            if true_span_cm > 0.0
            else None
        ),
        "map_length_inflation": (
            float(max(estimated_length_cm / true_span_cm - 1.0, 0.0))
            if true_span_cm > 0.0
            else None
        ),
        "adjacent_recombination_mean_absolute_error": float(
            np.mean(np.abs(adjacent_error))
        ),
        "adjacent_recombination_root_mean_square_error": float(
            np.sqrt(np.mean(adjacent_error * adjacent_error))
        ),
        "multipoint_adjacent_recombination_mean_absolute_error": float(
            np.mean(np.abs(multipoint_adjacent_error))
        ),
        "regularized_adjacent_recombination_mean_absolute_error": float(
            np.mean(np.abs(regularized_adjacent_error))
        ),
        "pairwise_adjacent_recombination_mean_absolute_error": float(
            np.mean(np.abs(pairwise_adjacent_error))
        ),
        "representative_position_rmse_cm": float(
            np.sqrt(np.mean(position_error * position_error))
        ),
        "endpoint_truth_span_fraction": (
            float(endpoint_span / true_span_cm) if true_span_cm > 0.0 else None
        ),
    }


def inversion_fraction(order: IntArray, truth_index: IntArray) -> float:
    """Orientation-invariant fraction of discordant marker pairs."""

    truth = truth_index[order]
    tau = float(kendalltau(np.arange(order.size), truth).statistic)
    tau = abs(tau)
    return (1.0 - tau) / 2.0


def truth_equivalence_membership(latent_states: IntArray) -> IntArray:
    """Assign markers to exact sampled-meiosis co-segregation classes."""

    states = np.asarray(latent_states, dtype=np.int64)
    if states.ndim != 2:
        raise ValueError("latent_states must be an offspring-by-marker matrix")
    if states.shape[1] < 2:
        raise ValueError("at least two markers are required")
    _, membership = np.unique(states.T, axis=0, return_inverse=True)
    return membership.astype(np.int64)


def evaluate_marker_framework(
    marker_order: IntArray,
    cross: SimulatedCross,
) -> dict[str, float | int | None]:
    """Evaluate a marker framework after common truth-equivalence deduplication."""

    markers = np.asarray(marker_order, dtype=np.int64)
    return evaluate_marker_coordinates(
        markers,
        np.arange(markers.size, dtype=np.float64),
        cross,
    )


def evaluate_marker_coordinates(
    marker_indices: IntArray,
    reported_coordinates: FloatArray,
    cross: SimulatedCross,
) -> dict[str, float | int | None]:
    """Evaluate a possibly tied partial order on common truth-equivalence bins.

    Markers in the same sampled-meiosis truth class are deduplicated.  Pairs with
    equal reported coordinates are treated as explicitly unresolved and excluded
    from the inversion denominator rather than being ordered by file position.
    """

    markers = np.asarray(marker_indices, dtype=np.int64)
    reported = np.asarray(reported_coordinates, dtype=np.float64)
    if markers.ndim != 1:
        raise ValueError("marker_indices must be one-dimensional")
    if reported.ndim != 1 or reported.size != markers.size:
        raise ValueError("reported_coordinates must match marker_indices")
    if np.any((markers < 0) | (markers >= cross.probabilities.shape[1])):
        raise ValueError("marker_indices contains an invalid input marker index")
    if np.unique(markers).size != markers.size:
        raise ValueError("marker_indices must be unique")
    if not np.all(np.isfinite(reported)):
        raise ValueError("reported_coordinates contain non-finite values")
    membership = truth_equivalence_membership(cross.latent_states)
    truth_coordinates = np.asarray(
        [
            np.median(cross.input_to_truth[membership == group])
            for group in range(int(membership.max()) + 1)
        ]
    )
    selected_groups = membership[markers]
    groups = np.unique(selected_groups)
    group_reported = np.asarray(
        [np.median(reported[selected_groups == group]) for group in groups]
    )
    group_truth = truth_coordinates[groups]

    discordant = 0
    concordant = 0
    tied = 0
    for left in range(groups.size - 1):
        for right in range(left + 1, groups.size):
            reported_difference = group_reported[right] - group_reported[left]
            if reported_difference == 0.0:
                tied += 1
                continue
            truth_difference = group_truth[right] - group_truth[left]
            if reported_difference * truth_difference < 0.0:
                discordant += 1
            else:
                concordant += 1
    ordered_pairs = discordant + concordant
    error = min(discordant, concordant) / ordered_pairs if ordered_pairs else None
    return {
        "truth_equivalence_bins": int(truth_coordinates.size),
        "framework_truth_bins": int(groups.size),
        "framework_reported_position_bins": int(np.unique(group_reported).size),
        "framework_ordered_truth_bin_pairs": ordered_pairs,
        "framework_tied_truth_bin_pairs": tied,
        "framework_truth_bin_inversion_fraction": error,
    }


def evaluate_marker_intervals(
    marker_bin_membership: IntArray,
    bin_representatives: IntArray,
    framework_bins: IntArray,
    interval_left: IntArray,
    interval_right: IntArray,
    cross: SimulatedCross,
) -> dict[str, float | int | None]:
    """Evaluate reported placement intervals on sampled-meiosis truth classes.

    A method may split one true co-segregation class into several noisy bins.  Such
    a class is counted once, using the union of its reported bin intervals.  If any
    constituent interval is unbounded, the class remains explicitly unbounded;
    this prevents a favorable split bin from hiding unresolved members.
    """

    membership = np.asarray(marker_bin_membership, dtype=np.int64)
    representatives = np.asarray(bin_representatives, dtype=np.int64)
    framework = np.asarray(framework_bins, dtype=np.int64)
    left = np.asarray(interval_left, dtype=np.int64)
    right = np.asarray(interval_right, dtype=np.int64)
    marker_count = cross.probabilities.shape[1]
    bin_count = representatives.size
    if membership.shape != (marker_count,):
        raise ValueError("marker_bin_membership must contain every input marker")
    if representatives.ndim != 1 or bin_count < 2:
        raise ValueError("bin_representatives must contain at least two bins")
    if np.any((membership < 0) | (membership >= bin_count)):
        raise ValueError("marker_bin_membership contains an invalid bin index")
    if np.any((representatives < 0) | (representatives >= marker_count)):
        raise ValueError("bin_representatives contains an invalid marker index")
    if np.unique(representatives).size != bin_count:
        raise ValueError("bin_representatives must be unique")
    if framework.ndim != 1 or framework.size < 2:
        raise ValueError("framework_bins must contain at least two bins")
    if np.any((framework < 0) | (framework >= bin_count)):
        raise ValueError("framework_bins contains an invalid bin index")
    if np.unique(framework).size != framework.size:
        raise ValueError("framework_bins must be unique")
    if left.shape != (bin_count,) or right.shape != (bin_count,):
        raise ValueError("interval arrays must contain every method bin")

    truth_membership = truth_equivalence_membership(cross.latent_states)
    truth_bin_count = int(truth_membership.max()) + 1
    truth_coordinates = np.asarray(
        [
            np.median(cross.input_to_truth[truth_membership == group])
            for group in range(truth_bin_count)
        ]
    )
    representative_truth = truth_membership[representatives]

    unique_framework_truth: list[int] = []
    truth_rank: dict[int, int] = {}
    slot_to_truth_rank = np.empty(framework.size, dtype=np.int64)
    for slot, method_bin in enumerate(framework):
        group = int(representative_truth[int(method_bin)])
        if group not in truth_rank:
            truth_rank[group] = len(unique_framework_truth)
            unique_framework_truth.append(group)
        slot_to_truth_rank[slot] = truth_rank[group]
    framework_lookup = {
        int(method_bin): slot for slot, method_bin in enumerate(framework)
    }

    bounded = 0
    covered = 0
    width_sum = 0
    for truth_group in range(truth_bin_count):
        if truth_group in truth_rank:
            continue
        method_bins = np.unique(membership[truth_membership == truth_group])
        class_bounds: list[tuple[int, int]] = []
        class_unbounded = False
        for method_bin_value in method_bins:
            method_bin = int(method_bin_value)
            if method_bin in framework_lookup:
                rank = int(slot_to_truth_rank[framework_lookup[method_bin]])
                class_bounds.append((rank, rank))
                continue
            low_slot = int(left[method_bin])
            high_slot = int(right[method_bin])
            if low_slot < 0 or high_slot >= framework.size or low_slot > high_slot:
                class_unbounded = True
                break
            low_rank = int(slot_to_truth_rank[low_slot])
            high_rank = int(slot_to_truth_rank[high_slot])
            class_bounds.append((min(low_rank, high_rank), max(low_rank, high_rank)))
        if class_unbounded or not class_bounds:
            continue
        low_rank = min(bound[0] for bound in class_bounds)
        high_rank = max(bound[1] for bound in class_bounds)
        bounded += 1
        width_sum += high_rank - low_rank
        low_truth, high_truth = sorted(
            (
                truth_coordinates[unique_framework_truth[low_rank]],
                truth_coordinates[unique_framework_truth[high_rank]],
            )
        )
        covered += int(low_truth <= truth_coordinates[truth_group] <= high_truth)

    nonframework = truth_bin_count - len(unique_framework_truth)
    return {
        "truth_equivalence_bins": truth_bin_count,
        "framework_truth_bins": len(unique_framework_truth),
        "nonframework_truth_bins": nonframework,
        "bounded_nonframework_truth_bins": bounded,
        "unbounded_nonframework_truth_bins": nonframework - bounded,
        "covered_nonframework_truth_bins": covered,
        "truth_bin_interval_coverage": covered / bounded if bounded else None,
        "truth_bin_interval_width_sum": width_sum,
        "mean_truth_bin_interval_width": width_sum / bounded if bounded else None,
    }


def evaluate_marker_partial_order(
    marker_bin_membership: IntArray,
    bin_representatives: IntArray,
    framework_bins: IntArray,
    interval_left: IntArray,
    interval_right: IntArray,
    cross: SimulatedCross,
) -> dict[str, float | int | None]:
    """Evaluate only pairwise orders logically implied by placement intervals.

    Every sampled-meiosis truth class is represented by the hull of all method-bin
    intervals containing its markers.  A pair is ordered only when those hulls do
    not overlap, so uncertain markers contribute useful relations without being
    forced into an arbitrary total order.  The error is orientation invariant.

    Framework anchors occupy even coordinates and the insertion gaps between
    them occupy odd coordinates.  This preserves strict relations such as
    ``anchor k < marker in gap k < anchor k+1`` while treating overlapping
    placement intervals as unresolved.
    """

    # Reuse the established common-evaluator validation before constructing the
    # more detailed all-marker partial order.
    evaluate_marker_intervals(
        marker_bin_membership,
        bin_representatives,
        framework_bins,
        interval_left,
        interval_right,
        cross,
    )
    membership = np.asarray(marker_bin_membership, dtype=np.int64)
    representatives = np.asarray(bin_representatives, dtype=np.int64)
    framework = np.asarray(framework_bins, dtype=np.int64)
    left = np.asarray(interval_left, dtype=np.int64)
    right = np.asarray(interval_right, dtype=np.int64)
    truth_membership = truth_equivalence_membership(cross.latent_states)
    truth_bin_count = int(truth_membership.max()) + 1
    truth_coordinates = np.asarray(
        [
            np.median(cross.input_to_truth[truth_membership == group])
            for group in range(truth_bin_count)
        ]
    )
    framework_lookup = {
        int(method_bin): slot for slot, method_bin in enumerate(framework)
    }
    minimum_coordinate = -1
    maximum_coordinate = 2 * framework.size - 1
    bounds = np.empty((truth_bin_count, 2), dtype=np.int64)
    for truth_group in range(truth_bin_count):
        class_bounds: list[tuple[int, int]] = []
        method_bins = np.unique(membership[truth_membership == truth_group])
        for method_bin_value in method_bins:
            method_bin = int(method_bin_value)
            if method_bin in framework_lookup:
                coordinate = 2 * framework_lookup[method_bin]
                class_bounds.append((coordinate, coordinate))
                continue
            low_slot = int(left[method_bin])
            high_slot = int(right[method_bin])
            if low_slot < -1 or high_slot > framework.size or low_slot >= high_slot:
                class_bounds.append((minimum_coordinate, maximum_coordinate))
                continue
            low_coordinate = minimum_coordinate if low_slot < 0 else 2 * low_slot + 1
            high_coordinate = (
                maximum_coordinate if high_slot == framework.size else 2 * high_slot - 1
            )
            class_bounds.append((low_coordinate, high_coordinate))
        bounds[truth_group, 0] = min(bound[0] for bound in class_bounds)
        bounds[truth_group, 1] = max(bound[1] for bound in class_bounds)

    concordant = 0
    discordant = 0
    for first in range(truth_bin_count - 1):
        for second in range(first + 1, truth_bin_count):
            if bounds[first, 1] < bounds[second, 0]:
                reported_difference = 1.0
            elif bounds[second, 1] < bounds[first, 0]:
                reported_difference = -1.0
            else:
                continue
            truth_difference = truth_coordinates[second] - truth_coordinates[first]
            if truth_difference == 0.0:
                continue
            if reported_difference * truth_difference > 0.0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    total = truth_bin_count * (truth_bin_count - 1) // 2
    minimum_discordant = min(concordant, discordant)
    representative_truth = truth_membership[representatives]
    framework_truth_groups = representative_truth[framework]
    framework_truth_coordinates = truth_coordinates[framework_truth_groups]
    orientation = (
        1.0
        if framework_truth_coordinates[-1] >= framework_truth_coordinates[0]
        else -1.0
    )
    framework_truth_set = set(map(int, framework_truth_groups))
    nonframework = 0
    informative = 0
    covered = 0
    for truth_group in range(truth_bin_count):
        if truth_group in framework_truth_set:
            continue
        nonframework += 1
        low_coordinate, high_coordinate = map(int, bounds[truth_group])
        informative += int(
            low_coordinate != minimum_coordinate
            or high_coordinate != maximum_coordinate
        )
        value = orientation * truth_coordinates[truth_group]
        is_covered = True
        if low_coordinate != minimum_coordinate:
            lower_slot = (
                low_coordinate // 2
                if low_coordinate % 2 == 0
                else (low_coordinate - 1) // 2
            )
            lower_truth = orientation * framework_truth_coordinates[lower_slot]
            is_covered &= value >= lower_truth
        if high_coordinate != maximum_coordinate:
            upper_slot = (
                high_coordinate // 2
                if high_coordinate % 2 == 0
                else (high_coordinate + 1) // 2
            )
            upper_truth = orientation * framework_truth_coordinates[upper_slot]
            is_covered &= value <= upper_truth
        covered += int(is_covered)
    return {
        "partial_order_truth_bins": truth_bin_count,
        "partial_order_total_truth_bin_pairs": total,
        "partial_order_comparable_truth_bin_pairs": comparable,
        "partial_order_unresolved_truth_bin_pairs": total - comparable,
        "partial_order_comparable_pair_fraction": (
            comparable / total if total else None
        ),
        "partial_order_minimum_discordant_truth_bin_pairs": minimum_discordant,
        "partial_order_truth_bin_inversion_fraction": (
            minimum_discordant / comparable if comparable else None
        ),
        "all_interval_nonframework_truth_bins": nonframework,
        "all_interval_informative_nonframework_truth_bins": informative,
        "all_interval_fully_unresolved_nonframework_truth_bins": (
            nonframework - informative
        ),
        "all_interval_covered_nonframework_truth_bins": covered,
        "all_interval_truth_bin_coverage": (
            covered / nonframework if nonframework else None
        ),
    }


def evaluate_marker_rank_intervals(
    interval_left: IntArray,
    interval_right: IntArray,
    cross: SimulatedCross,
) -> dict[str, float | int | None]:
    """Evaluate bootstrap rank bands as an orientation-free partial order."""

    left = np.asarray(interval_left, dtype=np.int64)
    right = np.asarray(interval_right, dtype=np.int64)
    marker_count = cross.probabilities.shape[1]
    if left.shape != (marker_count,) or right.shape != (marker_count,):
        raise ValueError("rank intervals must contain every input marker")
    if np.any((left < 0) | (right >= marker_count) | (left > right)):
        raise ValueError("rank intervals contain invalid bounds")
    membership = truth_equivalence_membership(cross.latent_states)
    truth_bin_count = int(membership.max()) + 1
    truth_coordinates = np.asarray(
        [
            np.median(cross.input_to_truth[membership == group])
            for group in range(truth_bin_count)
        ]
    )
    bounds = np.asarray(
        [
            (
                int(np.min(left[membership == group])),
                int(np.max(right[membership == group])),
            )
            for group in range(truth_bin_count)
        ],
        dtype=np.int64,
    )
    concordant = 0
    discordant = 0
    for first in range(truth_bin_count - 1):
        for second in range(first + 1, truth_bin_count):
            if bounds[first, 1] < bounds[second, 0]:
                reported_difference = 1.0
            elif bounds[second, 1] < bounds[first, 0]:
                reported_difference = -1.0
            else:
                continue
            truth_difference = truth_coordinates[second] - truth_coordinates[first]
            if reported_difference * truth_difference > 0.0:
                concordant += 1
            elif truth_difference != 0.0:
                discordant += 1
    comparable = concordant + discordant
    total = truth_bin_count * (truth_bin_count - 1) // 2
    reverse = discordant > concordant
    oriented_truth = (
        marker_count - 1 - truth_coordinates if reverse else truth_coordinates
    )
    covered = int(
        np.sum((bounds[:, 0] <= oriented_truth) & (oriented_truth <= bounds[:, 1]))
    )
    return {
        "rank_interval_truth_bins": truth_bin_count,
        "rank_interval_total_truth_bin_pairs": total,
        "rank_interval_comparable_truth_bin_pairs": comparable,
        "rank_interval_comparable_pair_fraction": (
            comparable / total if total else None
        ),
        "rank_interval_minimum_discordant_truth_bin_pairs": min(concordant, discordant),
        "rank_interval_truth_bin_inversion_fraction": (
            min(concordant, discordant) / comparable if comparable else None
        ),
        "rank_interval_covered_truth_bins": covered,
        "rank_interval_truth_bin_coverage": (
            covered / truth_bin_count if truth_bin_count else None
        ),
        "rank_interval_mean_truth_bin_width": float(
            np.mean(bounds[:, 1] - bounds[:, 0])
        ),
    }


def bin_truth_coordinates(
    result: SoftMapResult, input_to_truth: IntArray
) -> FloatArray:
    """Median truth coordinate of every probabilistic co-segregation bin."""

    return np.asarray(
        [
            np.median(input_to_truth[result.bins.membership == group])
            for group in range(result.bins.representatives.size)
        ],
        dtype=np.float64,
    )


def evaluate_result(result: SoftMapResult, cross: SimulatedCross) -> dict[str, float]:
    representative_truth = bin_truth_coordinates(result, cross.input_to_truth)
    representative_error = inversion_fraction(
        result.representative_order, representative_truth
    )
    framework_truth = representative_truth[result.framework]
    framework_error = inversion_fraction(
        np.arange(result.framework.size, dtype=np.int64), framework_truth
    )

    covered = 0
    bounded = 0
    truth_order = representative_truth[result.representative_order]
    orientation = 1 if truth_order[-1] >= truth_order[0] else -1
    framework_truth_oriented = orientation * representative_truth[result.framework]
    framework_set = set(map(int, result.framework))
    for marker in range(representative_truth.size):
        if marker in framework_set:
            continue
        left = int(result.interval_left[marker])
        right = int(result.interval_right[marker])
        if left < 0 or right >= result.framework.size or left > right:
            continue
        bounded += 1
        value = orientation * representative_truth[marker]
        low = min(framework_truth_oriented[left], framework_truth_oriented[right])
        high = max(framework_truth_oriented[left], framework_truth_oriented[right])
        covered += int(low <= value <= high)
    return {
        "markers": float(cross.probabilities.shape[1]),
        "bins": float(result.bins.representatives.size),
        "framework_markers": float(result.framework.size),
        "framework_fraction": float(
            result.framework.size / result.bins.representatives.size
        ),
        "representative_inversion_fraction": representative_error,
        "framework_inversion_fraction": framework_error,
        "bounded_intervals": float(bounded),
        "interval_coverage": float(covered / bounded) if bounded else float("nan"),
    }
