"""Core algorithms for a phased, binary-parental-origin SoftMap MVP.

The implementation deliberately separates point ordering from uncertainty
estimation. Sparse spectral/HMM mapping remains available for supported-framework
experiments; likelihood-weighted MDS supplies the competitive dense reference
order that is now being connected to bootstrap confidence output.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise
from multiprocessing import get_context

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, squareform
from scipy.stats import rankdata

from ._accelerator import (
    f2_pairwise_recombination as _accelerated_f2_pairwise_recombination,
)
from ._accelerator import (
    pairwise_recombination_edges as _accelerated_pairwise_recombination_edges,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class MarkerBins:
    """Approximate co-segregation bins and their representative markers."""

    membership: IntArray
    representatives: IntArray
    probabilities: FloatArray
    threshold: float


@dataclass(frozen=True)
class SoftMapResult:
    """Point order, confidence summaries, and marker placement intervals."""

    marker_names: tuple[str, ...]
    order: IntArray
    representative_order: IntArray
    bins: MarkerBins
    bootstrap_positions: IntArray
    precedence: FloatArray
    framework: IntArray
    interval_left: IntArray
    interval_right: IntArray
    confidence: float
    effective_offspring_information: float

    def ordered_names(self) -> list[str]:
        return [self.marker_names[i] for i in self.order]


@dataclass(frozen=True)
class HierarchicalSoftMapResult:
    """Coarse bootstrap support plus a fine multipoint HMM framework."""

    support: SoftMapResult
    bins: MarkerBins
    scaffold: IntArray
    framework: IntArray
    fine_bin_threshold: float | None
    min_log10_gap: float
    scaffold_prune_log10_gap: float | None
    post_scaffold_log10_gap: float | None
    hmm_bootstrap_replicates: int | None
    hmm_position_support: float | None
    effective_offspring_information: float

    def framework_names(self) -> list[str]:
        representatives = self.bins.representatives[self.framework]
        return [self.support.marker_names[int(marker)] for marker in representatives]

    @property
    def status(self) -> str:
        if self.framework.size < 3:
            return "insufficient_order_information"
        if self.effective_offspring_information < 50.0:
            return "limited_order_information"
        return "ok"


LikelihoodMDSConfig = tuple[str, float, int, int]


@dataclass(frozen=True)
class GeneticMapDistances:
    """Auditable local recombination and regularized centimorgan estimates.

    ``adjacent_local_recombination`` is the primary interval estimate. The
    pairwise, multipoint, and coordinate-derived arrays remain available as
    separate diagnostics.
    """

    marker_positions_cm: FloatArray
    bin_positions_cm: FloatArray
    ordered_bins: IntArray
    adjacent_recombination: FloatArray
    adjacent_pairwise_recombination: FloatArray
    adjacent_multipoint_recombination: FloatArray
    adjacent_local_recombination: FloatArray
    status: str
    method: str
    candidate_pair_count: int
    informative_pair_count: int
    segment_count: int
    composite_rmse_morgan: float | None
    composite_median_absolute_residual_morgan: float | None
    composite_p90_absolute_residual_morgan: float | None
    minimum_pair_recombination: float
    maximum_pair_recombination: float
    rank_span_weight_exponent: float | None = None

    @property
    def map_length_cm(self) -> float | None:
        ordered = self.bin_positions_cm[self.ordered_bins]
        if ordered.size < 2 or not np.all(np.isfinite(ordered)):
            return None
        return float(ordered[-1] - ordered[0])


@dataclass(frozen=True)
class F2MapResult:
    """Tri-state F2 order, stability bands, and Kosambi map coordinates."""

    marker_names: tuple[str, ...]
    order: IntArray
    de_novo_order: IntArray
    candidate_orders: IntArray
    candidate_positions: IntArray
    interval_left: IntArray
    interval_right: IntArray
    recombination: FloatArray
    lod: FloatArray
    genetic_distances: GeneticMapDistances
    selected_config: LikelihoodMDSConfig
    ordering_method: str
    physical_scaffold_used: bool
    mean_genotype_certainty: float

    def ordered_names(self) -> list[str]:
        return [self.marker_names[int(marker)] for marker in self.order]

    @property
    def status(self) -> str:
        return self.genetic_distances.status


DEFAULT_LIKELIHOOD_MDS_CONFIGS: tuple[LikelihoodMDSConfig, ...] = (
    ("rf", 1.0, 20, 1),
    ("rf", 1.0, 20, 2),
    ("rf", 1.0, 20, 3),
    ("haldane", 1.0, 20, 1),
    ("haldane", 1.0, 20, 2),
    ("haldane", 1.0, 20, 3),
    ("kosambi", 1.0, 20, 1),
    ("kosambi", 1.0, 20, 2),
    ("kosambi", 1.0, 20, 3),
    ("haldane", 2.0, 30, 3),
)

SCALABLE_LIKELIHOOD_MDS_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    3.0,
    10,
    1,
)

LANDMARK_LIKELIHOOD_MDS_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    1.0,
    20,
    1,
)

DENSE_HIGH_INFORMATION_CERTAINTY_THRESHOLD = 0.50
DENSE_HIGH_INFORMATION_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    3.0,
    10,
    1,
)

# Dense maps in the narrow moderate-certainty band benefit from a smoother,
# lower-variance geometry.  The lower boundary deliberately stays above the
# certainty range of the frozen sparse benchmark, preserving its selector.
DENSE_MODERATE_INFORMATION_CERTAINTY_THRESHOLD = 0.415
DENSE_MODERATE_INFORMATION_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    2.0,
    30,
    0,
)
DENSE_MODERATE_INFORMATION_PENALIZED_CURVE_EDF = 5.50

# Once the posterior-residual guard diagnoses inconsistent high-information
# likelihoods, a fixed penalized Haldane curve supplies a low-variance order.
DENSE_CALIBRATED_HIGH_INFORMATION_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    2.0,
    30,
    0,
)
DENSE_CALIBRATED_HIGH_INFORMATION_PENALIZED_CURVE_EDF = 5.05
DENSE_HIGH_INFORMATION_PENALIZED_CURVE_RESIDUAL_THRESHOLD = 0.095

# The local RF estimate shrinks the high-variance interval-level multipoint
# estimate toward the chromosome-wide regularized geometry. The weight was
# selected for stability across two independent development blocks.
LOCAL_RECOMBINATION_REGULARIZATION_WEIGHT = 0.35

# Pair counts grow strongly at intermediate and long rank spans, even though
# their recombination estimates approach the weakly identified saturation
# boundary.  This small negative exponent mildly rebalances the composite fit
# toward shorter, more locally informative spans without discarding any pair.
DISTANCE_RANK_SPAN_WEIGHT_EXPONENT = -0.125

POSTERIOR_CALIBRATION_MINIMUM_CERTAINTY = 0.58
POSTERIOR_CALIBRATION_RESIDUAL_THRESHOLDS = (0.1025, 0.13, 0.16)
POSTERIOR_CALIBRATION_TEMPERATURES = (1.15, 1.20, 1.50)

LARGE_SCALE_RESCUE_STABILITY_FLOOR = 0.50
LARGE_SCALE_SEVERE_INSTABILITY_FLOOR = 0.25
LARGE_SCALE_RESCUE_CERTAINTY_THRESHOLD = 0.48
LARGE_SCALE_LOW_CERTAINTY_STABILITY_MASS_CAP = 0.80
LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_PAIR_TRIGGER = 0.15
LARGE_SCALE_LOW_CERTAINTY_RESCUE_CONFIG: LikelihoodMDSConfig = (
    "rf",
    1.0,
    20,
    3,
)
LARGE_SCALE_MODERATE_CERTAINTY_RESCUE_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    1.0,
    20,
    3,
)
LARGE_SCALE_LOW_CERTAINTY_MODERATE_RESCUE_CONFIG: LikelihoodMDSConfig = (
    "haldane",
    1.0,
    20,
    2,
)
LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_RESCUE_CONFIG: LikelihoodMDSConfig = (
    "rf",
    1.0,
    20,
    1,
)


@dataclass(frozen=True)
class LikelihoodMDSEnsembleResult:
    """Confidence-first total order and model-stability rank bands."""

    marker_names: tuple[str, ...]
    order: IntArray
    preliminary_order: IntArray
    candidate_orders: IntArray
    candidate_positions: IntArray
    interval_left: IntArray
    interval_right: IntArray
    candidate_configs: tuple[LikelihoodMDSConfig, ...]
    selected_candidate_index: int
    uniform_candidate_index: int
    weighted_candidate_indices: IntArray
    uniform_scores: FloatArray
    weighted_scores: FloatArray
    unanimous_family_veto_triggered: bool
    posterior_refinement_weight: float
    posterior_refinement_passes_applied: int
    second_refinement_uncertain_pair_threshold: float
    stability_rank_padding: int
    minimum_stability_comparable_pair_fraction: float
    stability_mass: float
    stability_comparable_pair_fraction: float
    mean_normalized_rank_sd: float
    mean_pairwise_vote_margin: float
    uncertain_pair_fraction_75: float
    mean_genotype_certainty: float
    bin_membership: IntArray
    bin_representatives: IntArray
    reported_positions: IntArray
    binning_method: str
    maximum_bin_recombination: float | None
    minimum_bin_linkage_lod: float | None
    maximum_bin_pool_evidence: float | None
    bin_neighbor_count: int | None
    bin_neighbor_projection_dimensions: int | None
    selection_method: str
    ordering_method: str
    landmark_count: int | None
    landmark_neighbor_count: int | None
    landmark_support_exponent: float | None
    large_scale_rescue_triggered: bool
    low_certainty_stability_mass_cap_applied: bool
    genetic_distances: GeneticMapDistances | None
    posterior_calibration_temperature: float = 1.0
    posterior_calibration_triggered: bool = False
    uncalibrated_mean_genotype_certainty: float | None = None
    uncalibrated_distance_median_absolute_residual_morgan: float | None = None
    weighted_objective_support_filter_applied: bool = False
    penalized_curve_effective_degrees_of_freedom: float | None = None

    def ordered_names(self) -> list[str]:
        return [self.marker_names[int(index)] for index in self.order]

    @property
    def selected_config(self) -> LikelihoodMDSConfig:
        return self.candidate_configs[self.selected_candidate_index]

    @property
    def likelihood_bin_count(self) -> int:
        return int(self.bin_representatives.size)

    @property
    def status(self) -> str:
        if self.stability_comparable_pair_fraction == 0.0:
            return "insufficient_order_information"
        if self.stability_comparable_pair_fraction < 0.10:
            return "limited_order_information"
        return "ok"


def _validate_probabilities(probabilities: FloatArray) -> FloatArray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must be a 2D offspring-by-marker matrix")
    if min(p.shape) < 2:
        raise ValueError("at least two offspring and two markers are required")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities contain non-finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return p


F2_GENOTYPE_PRIOR = np.asarray((0.25, 0.50, 0.25), dtype=np.float64)


def _validate_f2_probabilities(probabilities: FloatArray) -> FloatArray:
    """Validate offspring-by-marker-by-genotype F2 posterior probabilities."""

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3:
        raise ValueError("F2 probabilities must have shape (offspring, markers, 3)")
    if p.shape[0] < 2 or p.shape[1] < 2:
        raise ValueError("at least two offspring and two markers are required")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("F2 probabilities must be finite and nonnegative")
    totals = np.sum(p, axis=2)
    if np.any(totals <= 0.0):
        raise ValueError("each F2 genotype probability vector must have positive mass")
    if not np.allclose(totals, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("F2 genotype probabilities must sum to one")
    return p


def _f2_joint_genotype_probabilities(recombination: float) -> FloatArray:
    """Return the 3x3 two-locus F2 genotype distribution in coupling phase."""

    rate = float(recombination)
    if not 0.0 <= rate <= 0.5:
        raise ValueError("recombination must lie in [0, 0.5]")
    haplotypes = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int64)
    gamete = np.asarray(
        ((1.0 - rate) / 2.0, rate / 2.0, rate / 2.0, (1.0 - rate) / 2.0),
        dtype=np.float64,
    )
    joint = np.zeros((3, 3), dtype=np.float64)
    for first in range(4):
        for second in range(4):
            genotype = haplotypes[first] + haplotypes[second]
            joint[int(genotype[0]), int(genotype[1])] += gamete[first] * gamete[second]
    return joint


_F2_JOINT_CONSTANT = _f2_joint_genotype_probabilities(0.0)
_f2_quarter_delta = _f2_joint_genotype_probabilities(0.25) - _F2_JOINT_CONSTANT
_f2_half_delta = _f2_joint_genotype_probabilities(0.5) - _F2_JOINT_CONSTANT
_F2_JOINT_QUADRATIC = 8.0 * (_f2_half_delta - 2.0 * _f2_quarter_delta)
_F2_JOINT_LINEAR = 4.0 * _f2_quarter_delta - 0.25 * _F2_JOINT_QUADRATIC


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            nxt = int(self.parent[value])
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def expected_disagreement(left: FloatArray, right: FloatArray) -> float:
    """Information-weighted expected difference between two phased markers."""

    certainty = np.abs(2.0 * left - 1.0) * np.abs(2.0 * right - 1.0)
    total = float(certainty.sum())
    if total <= 1e-12:
        return 0.5
    disagreement = left * (1.0 - right) + (1.0 - left) * right
    return float(np.dot(certainty, disagreement) / total)


def _candidate_neighbors(
    probabilities: FloatArray,
    k: int,
    projection_dimensions: int | None = None,
) -> tuple[IntArray, FloatArray]:
    """Find candidate neighbors in certainty-scaled posterior space."""

    n_markers = probabilities.shape[1]
    k = min(max(2, k), n_markers)
    features = (probabilities.T - 0.5) * 2.0
    if projection_dimensions is not None:
        if projection_dimensions < 2:
            raise ValueError("projection_dimensions must be at least two")
        if projection_dimensions < features.shape[1]:
            centered = features - np.mean(features, axis=0)
            covariance = centered.T @ centered
            _, projection = linalg.eigh(
                covariance,
                subset_by_index=[
                    covariance.shape[0] - projection_dimensions,
                    covariance.shape[0] - 1,
                ],
                driver="evr",
                check_finite=False,
            )
            features = centered @ projection
    tree = cKDTree(features)
    # A single worker keeps tie handling reproducible across runs and platforms.
    distances, indices = tree.query(features, k=k, workers=1)
    return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def bin_markers(
    probabilities: FloatArray,
    *,
    threshold: float = 0.01,
    neighbor_count: int = 16,
) -> MarkerBins:
    """Collapse markers with near-identical probabilistic segregation patterns."""

    p = _validate_probabilities(probabilities)
    if not 0.0 <= threshold < 0.5:
        raise ValueError("bin threshold must lie in [0, 0.5)")
    neighbors, _ = _candidate_neighbors(p, neighbor_count)
    return _bin_markers_from_neighbors(p, neighbors, threshold)


def _bin_markers_from_neighbors(
    probabilities: FloatArray,
    neighbors: IntArray,
    threshold: float,
) -> MarkerBins:
    p = probabilities
    membership, representatives = _assign_marker_bins(p, neighbors, threshold)
    return _pool_marker_bins(p, membership, representatives, threshold)


def _assign_marker_bins(
    probabilities: FloatArray,
    neighbors: IntArray,
    threshold: float,
) -> tuple[IntArray, IntArray]:
    p = probabilities
    n_markers = p.shape[1]
    information = np.mean(np.abs(2.0 * p - 1.0), axis=0)
    membership = np.full(n_markers, -1, dtype=np.int64)
    representatives_list: list[int] = []
    # High-information markers become fixed representatives. Candidates can join a
    # representative but never merge bins transitively, which prevents a chain of
    # successive one-recombinant differences from collapsing an entire chromosome.
    for marker in np.argsort(-information, kind="stable"):
        marker = int(marker)
        if membership[marker] >= 0:
            continue
        group = len(representatives_list)
        representatives_list.append(marker)
        membership[marker] = group
        for candidate in neighbors[marker]:
            candidate = int(candidate)
            if candidate == marker:
                continue
            if membership[candidate] >= 0:
                continue
            if expected_disagreement(p[:, marker], p[:, candidate]) <= threshold:
                membership[candidate] = group
    return membership, np.asarray(representatives_list, dtype=np.int64)


def _pool_marker_bins(
    probabilities: FloatArray,
    membership: IntArray,
    representatives: IntArray,
    threshold: float,
    maximum_evidence: float | None = None,
) -> MarkerBins:
    p = probabilities
    pooled = np.empty((p.shape[0], representatives.size), dtype=np.float64)
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    members_by_group: list[list[int]] = [[] for _ in range(representatives.size)]
    for marker, group in enumerate(membership):
        members_by_group[int(group)].append(marker)
    for group, members in enumerate(members_by_group):
        combined = log_odds[:, members].sum(axis=1)
        if maximum_evidence is not None:
            combined *= min(float(len(members)), maximum_evidence) / len(members)
        combined = np.clip(combined, -30.0, 30.0)
        pooled[:, group] = 1.0 / (1.0 + np.exp(-combined))
    return MarkerBins(membership, representatives, pooled, threshold)


def auto_bin_markers(
    probabilities: FloatArray,
    *,
    neighbor_count: int = 200,
    target_bins: int | None = None,
    thresholds: Iterable[float] = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08),
) -> MarkerBins:
    """Choose a coarse threshold from the marker-pattern collapse curve.

    With an explicit ``target_bins``, this retains the expert-controlled behavior
    of selecting the smallest threshold that meets the target.  By default, it
    evaluates the full threshold grid and selects the point after the largest
    drop in bin count.  That drop estimates the boundary between sequencing-noise
    variants of the same segregation pattern and genuinely distinct crossover
    patterns.  An offspring-information ceiling prevents a weak early knee from
    retaining more apparent patterns than the sampled meioses can plausibly order.
    Neither decision uses marker truth.
    """

    p = _validate_probabilities(probabilities)
    values = tuple(float(value) for value in thresholds)
    if not values or any(value < 0.0 or value >= 0.5 for value in values):
        raise ValueError("auto-bin thresholds must lie in [0, 0.5)")
    if any(right <= left for left, right in pairwise(values)):
        raise ValueError("auto-bin thresholds must be strictly increasing")
    if target_bins is not None and target_bins < 2:
        raise ValueError("target_bins must be at least two")
    neighbors, _ = _candidate_neighbors(p, neighbor_count)
    candidates: list[tuple[float, IntArray, IntArray]] = []
    for threshold in values:
        membership, representatives = _assign_marker_bins(p, neighbors, threshold)
        candidates.append((threshold, membership, representatives))
        if target_bins is not None and representatives.size <= target_bins:
            break
    if target_bins is None and len(candidates) > 1:
        counts = np.asarray(
            [representatives.size for _, _, representatives in candidates],
            dtype=np.float64,
        )
        collapse = counts[:-1] - counts[1:]
        # If the grid finds no real collapse, retain the least aggressive setting
        # instead of manufacturing a coarse representation from a flat curve.
        selected_index = (
            int(np.argmax(collapse)) + 1
            if float(np.max(collapse)) / counts[0] >= 0.10
            else 0
        )
        information_ceiling = min(p.shape[1], max(20, int(np.ceil(3.2 * p.shape[0]))))
        meeting_ceiling = np.flatnonzero(counts <= information_ceiling)
        if meeting_ceiling.size:
            selected_index = max(selected_index, int(meeting_ceiling[0]))
        else:
            selected_index = len(candidates) - 1
    else:
        selected_index = len(candidates) - 1
    selected_threshold, selected_membership, selected_representatives = candidates[
        selected_index
    ]
    return _pool_marker_bins(
        p,
        selected_membership,
        selected_representatives,
        selected_threshold,
    )


def _dense_distances(probabilities: FloatArray) -> FloatArray:
    """All pairwise information-weighted expected disagreements."""

    p = probabilities
    certainty = np.abs(2.0 * p - 1.0)
    weighted_one = certainty * p
    weighted_zero = certainty * (1.0 - p)
    numerator = weighted_one.T @ weighted_zero + weighted_zero.T @ weighted_one
    denominator = certainty.T @ certainty
    distances = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, 0.5),
        where=denominator > 1e-12,
    )
    np.fill_diagonal(distances, 0.0)
    return distances


def _fit_pairwise_recombination_columns(
    same: FloatArray,
    different: FloatArray,
    *,
    maximum_recombination: float,
    bisection_iterations: int,
    beta_prior_shape: float,
) -> tuple[FloatArray, FloatArray]:
    """Fit independent recombination mixtures stored column-wise."""

    delta = different - same
    flat = np.sum(delta * delta, axis=0) <= 1e-20
    prior_exponent = beta_prior_shape - 1.0

    def score(values: FloatArray) -> FloatArray:
        denominator = np.clip(same + delta * values[None, :], 1e-300, None)
        result = np.sum(delta / denominator, axis=0)
        if prior_exponent > 0.0:
            bounded = np.clip(values, 1e-12, 1.0 - 1e-12)
            result += prior_exponent * (1.0 / bounded - 1.0 / (1.0 - bounded))
        return result

    pair_count = same.shape[1]
    low = np.full(
        pair_count,
        1e-12 if prior_exponent > 0.0 else 0.0,
        dtype=np.float64,
    )
    high = np.full(pair_count, maximum_recombination, dtype=np.float64)
    score_low = score(low)
    score_high = score(high)
    at_zero = score_low <= 0.0
    at_maximum = score_high >= 0.0
    active = ~(flat | at_zero | at_maximum)
    for _ in range(bisection_iterations):
        middle = (low + high) / 2.0
        positive = score(middle) > 0.0
        low = np.where(active & positive, middle, low)
        high = np.where(active & ~positive, middle, high)
    fitted = (low + high) / 2.0
    fitted[flat | at_maximum] = maximum_recombination
    fitted[at_zero & ~flat] = 0.0
    linked_log_likelihood = np.sum(
        np.log(np.clip(same + delta * fitted[None, :], 1e-300, None)),
        axis=0,
    )
    unlinked_log_likelihood = np.sum(
        np.log(np.clip(same + 0.5 * delta, 1e-300, None)),
        axis=0,
    )
    fitted_lod = np.maximum(
        0.0,
        (linked_log_likelihood - unlinked_log_likelihood) / np.log(10.0),
    )
    return fitted, fitted_lod


def pairwise_recombination_likelihood(
    probabilities: FloatArray,
    *,
    maximum_recombination: float = 0.499999,
    bisection_iterations: int = 32,
    beta_prior_shape: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Estimate pairwise recombination fractions and linkage LOD scores.

    Each marker probability is the normalized emission likelihood for the phased
    parental-origin state under a uniform marginal prior.  For a marker pair, the
    offspring likelihood is a mixture of the same-state and different-state
    emission likelihoods with mixing proportion ``r``.  The log likelihood is
    concave in ``r``, so all pairs involving one marker are solved together by
    vectorized bisection.  ``beta_prior_shape > 1`` applies a symmetric Beta
    regularizer that prevents boundary estimates when meiosis information is
    scarce. The LOD remains a data-likelihood comparison with ``r = 0.5``.
    """

    p = _validate_probabilities(probabilities)
    if not 0.0 < maximum_recombination < 0.5:
        raise ValueError("maximum_recombination must lie between zero and 0.5")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    if beta_prior_shape < 1.0 or not np.isfinite(beta_prior_shape):
        raise ValueError("beta_prior_shape must be finite and at least one")
    marker_count = p.shape[1]
    recombination = np.zeros((marker_count, marker_count), dtype=np.float64)
    lod = np.zeros_like(recombination)
    for marker in range(1, marker_count):
        current = p[:, marker, None]
        previous = p[:, :marker]
        same = (1.0 - current) * (1.0 - previous) + current * previous
        different = current * (1.0 - previous) + (1.0 - current) * previous
        fitted, fitted_lod = _fit_pairwise_recombination_columns(
            same,
            different,
            maximum_recombination=maximum_recombination,
            bisection_iterations=bisection_iterations,
            beta_prior_shape=beta_prior_shape,
        )
        recombination[marker, :marker] = fitted
        recombination[:marker, marker] = fitted
        lod[marker, :marker] = fitted_lod
        lod[:marker, marker] = fitted_lod
    return recombination, lod


def pairwise_recombination_likelihood_edges(
    probabilities: FloatArray,
    left: IntArray,
    right: IntArray,
    *,
    maximum_recombination: float = 0.499999,
    bisection_iterations: int = 32,
    beta_prior_shape: float = 1.0,
    batch_size: int = 4096,
) -> tuple[FloatArray, FloatArray]:
    """Estimate recombination and LOD for a sparse list of marker pairs."""

    p = _validate_probabilities(probabilities)
    left_index = np.asarray(left, dtype=np.int64)
    right_index = np.asarray(right, dtype=np.int64)
    if left_index.ndim != 1 or right_index.shape != left_index.shape:
        raise ValueError("left and right edge arrays must be matching vectors")
    if left_index.size < 1:
        raise ValueError("at least one marker pair is required")
    if np.any(
        (left_index < 0)
        | (right_index < 0)
        | (left_index >= p.shape[1])
        | (right_index >= p.shape[1])
        | (left_index == right_index)
    ):
        raise ValueError("edge arrays contain an invalid marker pair")
    if not 0.0 < maximum_recombination < 0.5:
        raise ValueError("maximum_recombination must lie between zero and 0.5")
    if bisection_iterations < 1 or batch_size < 1:
        raise ValueError("bisection iterations and batch size must be positive")
    if beta_prior_shape < 1.0 or not np.isfinite(beta_prior_shape):
        raise ValueError("beta_prior_shape must be finite and at least one")
    accelerated = _accelerated_pairwise_recombination_edges(
        p,
        left_index,
        right_index,
        maximum_recombination=maximum_recombination,
        bisection_iterations=bisection_iterations,
        beta_prior_shape=beta_prior_shape,
        batch_size=batch_size,
    )
    if accelerated is not None:
        return accelerated
    recombination = np.empty(left_index.size, dtype=np.float64)
    lod = np.empty_like(recombination)
    for start in range(0, left_index.size, batch_size):
        stop = min(start + batch_size, left_index.size)
        first = p[:, left_index[start:stop]]
        second = p[:, right_index[start:stop]]
        same = (1.0 - first) * (1.0 - second) + first * second
        different = first * (1.0 - second) + (1.0 - first) * second
        recombination[start:stop], lod[start:stop] = (
            _fit_pairwise_recombination_columns(
                same,
                different,
                maximum_recombination=maximum_recombination,
                bisection_iterations=bisection_iterations,
                beta_prior_shape=beta_prior_shape,
            )
        )
    return recombination, lod


def _fit_f2_recombination_columns(
    constant: FloatArray,
    linear: FloatArray,
    quadratic: FloatArray,
    *,
    maximum_recombination: float,
    bisection_iterations: int,
) -> tuple[FloatArray, FloatArray]:
    """Fit vectorized two-locus F2 likelihood polynomials."""

    pair_count = constant.shape[1]
    flat = np.sum(linear * linear + quadratic * quadratic, axis=0) <= 1e-20
    low = np.zeros(pair_count, dtype=np.float64)
    high = np.full(pair_count, maximum_recombination, dtype=np.float64)
    for _ in range(bisection_iterations):
        middle = (low + high) / 2.0
        denominator = np.clip(
            constant
            + linear * middle[None, :]
            + quadratic * middle[None, :] * middle[None, :],
            1e-300,
            None,
        )
        score = np.sum(
            (linear + 2.0 * quadratic * middle[None, :]) / denominator,
            axis=0,
        )
        low = np.where((score > 0.0) & ~flat, middle, low)
        high = np.where((score > 0.0) | flat, high, middle)
    fitted = (low + high) / 2.0
    fitted[flat] = maximum_recombination
    linked = np.sum(
        np.log(
            np.clip(
                constant
                + linear * fitted[None, :]
                + quadratic * fitted[None, :] * fitted[None, :],
                1e-300,
                None,
            )
        ),
        axis=0,
    )
    unlinked = np.sum(
        np.log(
            np.clip(
                constant + 0.5 * linear + 0.25 * quadratic,
                1e-300,
                None,
            )
        ),
        axis=0,
    )
    fitted_lod = np.maximum(0.0, (linked - unlinked) / np.log(10.0))
    fitted_lod[flat] = 0.0
    return fitted, fitted_lod


def _f2_pair_coefficients(
    left: FloatArray,
    right: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return per-offspring quadratic likelihood coefficients for F2 pairs."""

    left_likelihood = left / F2_GENOTYPE_PRIOR
    right_likelihood = right / F2_GENOTYPE_PRIOR
    return tuple(
        np.einsum(
            "nka,ab,nkb->nk",
            left_likelihood,
            coefficient,
            right_likelihood,
            optimize=True,
        )
        for coefficient in (
            _F2_JOINT_CONSTANT,
            _F2_JOINT_LINEAR,
            _F2_JOINT_QUADRATIC,
        )
    )


def f2_pairwise_recombination_likelihood(
    probabilities: FloatArray,
    *,
    maximum_recombination: float = 0.499999,
    bisection_iterations: int = 32,
) -> tuple[FloatArray, FloatArray]:
    """Estimate pairwise F2 recombination and LOD from AA/AB/BB posteriors.

    The three genotype probabilities are interpreted as posteriors under the F2
    marginal prior ``(1/4, 1/2, 1/4)``. Dividing by that prior recovers relative
    emission likelihoods. The two-locus F2 distribution integrates over both
    gametes in known parental coupling phase, retaining heterozygotes as fully
    informative observations.
    """

    p = _validate_f2_probabilities(probabilities)
    if not 0.0 < maximum_recombination < 0.5:
        raise ValueError("maximum_recombination must lie between zero and 0.5")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    accelerated = _accelerated_f2_pairwise_recombination(
        p,
        maximum_recombination=maximum_recombination,
        bisection_iterations=bisection_iterations,
    )
    if accelerated is not None:
        recombination, _ = accelerated
        marker_count = p.shape[1]
        lod = np.zeros_like(recombination)
        for marker in range(1, marker_count):
            right = np.broadcast_to(
                p[:, marker, None, :],
                (p.shape[0], marker, 3),
            )
            constant, linear, quadratic = _f2_pair_coefficients(p[:, :marker, :], right)
            fitted = recombination[marker, :marker]
            linked = np.sum(
                np.log(
                    np.clip(
                        constant
                        + linear * fitted[None, :]
                        + quadratic * fitted[None, :] * fitted[None, :],
                        1e-300,
                        None,
                    )
                ),
                axis=0,
            )
            unlinked = np.sum(
                np.log(
                    np.clip(
                        constant + 0.5 * linear + 0.25 * quadratic,
                        1e-300,
                        None,
                    )
                ),
                axis=0,
            )
            fitted_lod = np.maximum(
                0.0,
                (linked - unlinked) / np.log(10.0),
            )
            lod[marker, :marker] = fitted_lod
            lod[:marker, marker] = fitted_lod
        return recombination, lod
    marker_count = p.shape[1]
    recombination = np.zeros((marker_count, marker_count), dtype=np.float64)
    lod = np.zeros_like(recombination)
    for marker in range(1, marker_count):
        right = np.broadcast_to(
            p[:, marker, None, :],
            (p.shape[0], marker, 3),
        )
        constant, linear, quadratic = _f2_pair_coefficients(p[:, :marker, :], right)
        fitted, fitted_lod = _fit_f2_recombination_columns(
            constant,
            linear,
            quadratic,
            maximum_recombination=maximum_recombination,
            bisection_iterations=bisection_iterations,
        )
        recombination[marker, :marker] = fitted
        recombination[:marker, marker] = fitted
        lod[marker, :marker] = fitted_lod
        lod[:marker, marker] = fitted_lod
    return recombination, lod


def _f2_adjacent_kosambi_distances(
    recombination: FloatArray,
    lod: FloatArray,
    order: IntArray,
    *,
    maximum_interval_recombination: float = 0.45,
) -> GeneticMapDistances:
    """Convert adjacent F2 recombination estimates to a cumulative Kosambi map."""

    ordered = np.asarray(order, dtype=np.int64)
    marker_count = ordered.size
    adjacent_pairwise = np.asarray(
        recombination[ordered[:-1], ordered[1:]], dtype=np.float64
    )
    adjacent = np.minimum(adjacent_pairwise, maximum_interval_recombination)
    interval_morgan = 0.25 * np.log(
        (1.0 + 2.0 * adjacent) / np.clip(1.0 - 2.0 * adjacent, 1e-12, None)
    )
    ordered_positions = np.concatenate(([0.0], np.cumsum(interval_morgan)))
    positions_cm = np.empty(marker_count, dtype=np.float64)
    positions_cm[ordered] = 100.0 * ordered_positions

    left_rank, right_rank = np.triu_indices(marker_count, 1)
    observed_rf = recombination[ordered[left_rank], ordered[right_rank]]
    pair_lod = lod[ordered[left_rank], ordered[right_rank]]
    informative = (observed_rf >= 0.005) & (observed_rf <= 0.445) & (pair_lod > 0.0)
    inferred_distance = ordered_positions[right_rank] - ordered_positions[left_rank]
    observed_distance = 0.25 * np.log(
        (1.0 + 2.0 * observed_rf[informative])
        / np.clip(1.0 - 2.0 * observed_rf[informative], 1e-12, None)
    )
    residual = inferred_distance[informative] - observed_distance
    absolute_residual = np.abs(residual)
    return GeneticMapDistances(
        marker_positions_cm=positions_cm.copy(),
        bin_positions_cm=positions_cm.copy(),
        ordered_bins=ordered.copy(),
        adjacent_recombination=adjacent.copy(),
        adjacent_pairwise_recombination=adjacent_pairwise.copy(),
        adjacent_multipoint_recombination=adjacent_pairwise.copy(),
        adjacent_local_recombination=adjacent.copy(),
        status="ok",
        method="f2_pairwise_kosambi_adjacent",
        candidate_pair_count=int(left_rank.size),
        informative_pair_count=int(np.count_nonzero(informative)),
        segment_count=marker_count - 1,
        composite_rmse_morgan=(
            float(np.sqrt(np.mean(residual * residual))) if residual.size else None
        ),
        composite_median_absolute_residual_morgan=(
            float(np.median(absolute_residual)) if residual.size else None
        ),
        composite_p90_absolute_residual_morgan=(
            float(np.quantile(absolute_residual, 0.90)) if residual.size else None
        ),
        minimum_pair_recombination=0.005,
        maximum_pair_recombination=0.445,
    )


def _distance_candidate_pairs(
    marker_count: int,
    maximum_pair_count: int,
) -> tuple[IntArray, IntArray]:
    """Return deterministic all-pair or multiscale rank-space edges."""

    all_pair_count = marker_count * (marker_count - 1) // 2
    if all_pair_count <= maximum_pair_count:
        left, right = np.triu_indices(marker_count, 1)
        return left.astype(np.int64), right.astype(np.int64)

    offset_count = min(32, marker_count - 1)
    offsets = np.unique(
        np.rint(
            np.geomspace(
                1,
                marker_count - 1,
                num=offset_count,
            )
        ).astype(np.int64)
    )
    budget_per_offset = max(1, maximum_pair_count // offsets.size)
    left_parts: list[IntArray] = []
    right_parts: list[IntArray] = []
    for offset in offsets:
        available = marker_count - int(offset)
        if available <= budget_per_offset:
            left = np.arange(available, dtype=np.int64)
        else:
            left = np.unique(
                np.rint(
                    np.linspace(
                        0,
                        available - 1,
                        num=budget_per_offset,
                    )
                ).astype(np.int64)
            )
        left_parts.append(left)
        right_parts.append(left + int(offset))
    return np.concatenate(left_parts), np.concatenate(right_parts)


def _distance_basis(marker_count: int, segment_count: int) -> FloatArray:
    scaled_rank = np.linspace(0.0, float(segment_count), marker_count)
    return np.clip(
        scaled_rank[:, None] - np.arange(segment_count, dtype=np.float64)[None, :],
        0.0,
        1.0,
    )


def _fit_genetic_map_distances_from_pairs(
    marker_count: int,
    order: IntArray,
    pair_left_rank: IntArray,
    pair_right_rank: IntArray,
    pair_recombination: FloatArray,
    pair_lod: FloatArray,
    *,
    marker_bin_membership: IntArray | None,
    adjacent_pairwise_recombination: FloatArray,
    adjacent_multipoint_recombination: FloatArray,
    segment_count: int,
    minimum_pair_recombination: float,
    maximum_pair_recombination: float,
    mapping_function: str = "haldane",
    rank_span_weight_exponent: float = DISTANCE_RANK_SPAN_WEIGHT_EXPONENT,
) -> GeneticMapDistances:
    """Fit nonnegative rank-segment lengths from probabilistic pair distances."""

    ordered_bins = np.asarray(order, dtype=np.int64)
    left = np.asarray(pair_left_rank, dtype=np.int64)
    right = np.asarray(pair_right_rank, dtype=np.int64)
    recombination = np.asarray(pair_recombination, dtype=np.float64)
    lod = np.asarray(pair_lod, dtype=np.float64)
    candidate_pair_count = int(left.size)
    resolved_segments = min(int(segment_count), marker_count - 1)
    adjacent_pairwise_recombination = np.asarray(
        adjacent_pairwise_recombination,
        dtype=np.float64,
    )
    adjacent_multipoint_recombination = np.asarray(
        adjacent_multipoint_recombination,
        dtype=np.float64,
    )
    for adjacent in (
        adjacent_pairwise_recombination,
        adjacent_multipoint_recombination,
    ):
        if (
            adjacent.shape != (marker_count - 1,)
            or not np.all(np.isfinite(adjacent))
            or np.any((adjacent < 0.0) | (adjacent >= 0.5))
        ):
            raise ValueError("adjacent recombination estimates are invalid")
    membership = (
        np.arange(marker_count, dtype=np.int64)
        if marker_bin_membership is None
        else np.asarray(marker_bin_membership, dtype=np.int64)
    )

    informative = (
        np.isfinite(recombination)
        & np.isfinite(lod)
        & (recombination >= minimum_pair_recombination)
        & (recombination <= maximum_pair_recombination)
        & (lod > 0.0)
    )
    informative_count = int(np.count_nonzero(informative))
    if mapping_function not in {"haldane", "kosambi"}:
        raise ValueError("mapping_function must be haldane or kosambi")
    if not np.isfinite(rank_span_weight_exponent):
        raise ValueError("rank span weight exponent must be finite")
    method = f"probabilistic_composite_{mapping_function}_{resolved_segments}segment"

    def unavailable() -> GeneticMapDistances:
        return GeneticMapDistances(
            marker_positions_cm=np.full(membership.size, np.nan),
            bin_positions_cm=np.full(marker_count, np.nan),
            ordered_bins=ordered_bins.copy(),
            adjacent_recombination=np.full(marker_count - 1, np.nan),
            adjacent_pairwise_recombination=(adjacent_pairwise_recombination.copy()),
            adjacent_multipoint_recombination=(
                adjacent_multipoint_recombination.copy()
            ),
            adjacent_local_recombination=(adjacent_multipoint_recombination.copy()),
            status="insufficient_distance_information",
            method=method,
            candidate_pair_count=candidate_pair_count,
            informative_pair_count=informative_count,
            segment_count=resolved_segments,
            composite_rmse_morgan=None,
            composite_median_absolute_residual_morgan=None,
            composite_p90_absolute_residual_morgan=None,
            minimum_pair_recombination=float(minimum_pair_recombination),
            maximum_pair_recombination=float(maximum_pair_recombination),
            rank_span_weight_exponent=float(rank_span_weight_exponent),
        )

    if informative_count < max(4 * resolved_segments, resolved_segments + 1):
        return unavailable()

    left = left[informative]
    right = right[informative]
    recombination = recombination[informative]
    basis = _distance_basis(marker_count, resolved_segments)
    design = basis[right] - basis[left]
    if np.any(np.sum(design, axis=0) <= 0.0):
        return unavailable()
    if mapping_function == "haldane":
        target = -0.5 * np.log(
            np.clip(
                1.0 - 2.0 * recombination,
                1e-12,
                None,
            )
        )
    else:
        target = 0.25 * np.log(
            np.clip(
                (1.0 + 2.0 * recombination)
                / np.clip(1.0 - 2.0 * recombination, 1e-12, None),
                1e-12,
                None,
            )
        )
    rank_span = (right - left).astype(np.float64)
    fit_weight = rank_span**rank_span_weight_exponent
    fit_weight /= float(np.max(fit_weight))
    square_root_weight = np.sqrt(fit_weight)
    fitted = lsq_linear(
        coo_matrix(design * square_root_weight[:, None]).tocsr(),
        target * square_root_weight,
        bounds=(0.0, np.inf),
        method="trf",
        lsq_solver="lsmr",
        tol=1e-8,
        lsmr_tol=1e-8,
        max_iter=200,
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return unavailable()

    ordered_positions_morgan = basis @ fitted.x
    ordered_positions_morgan -= ordered_positions_morgan[0]
    bin_positions_cm = np.empty(marker_count, dtype=np.float64)
    bin_positions_cm[ordered_bins] = 100.0 * ordered_positions_morgan
    marker_positions_cm = bin_positions_cm[membership]
    adjacent_distance = np.diff(ordered_positions_morgan)
    adjacent_recombination = (
        0.5 * (1.0 - np.exp(-2.0 * adjacent_distance))
        if mapping_function == "haldane"
        else 0.5 * np.tanh(2.0 * adjacent_distance)
    )
    adjacent_local_recombination = (
        (1.0 - LOCAL_RECOMBINATION_REGULARIZATION_WEIGHT)
        * adjacent_multipoint_recombination
        + LOCAL_RECOMBINATION_REGULARIZATION_WEIGHT * adjacent_recombination
    )
    residual = design @ fitted.x - target
    absolute_residual = np.abs(residual)
    return GeneticMapDistances(
        marker_positions_cm=marker_positions_cm,
        bin_positions_cm=bin_positions_cm,
        ordered_bins=ordered_bins.copy(),
        adjacent_recombination=adjacent_recombination,
        adjacent_pairwise_recombination=adjacent_pairwise_recombination.copy(),
        adjacent_multipoint_recombination=adjacent_multipoint_recombination.copy(),
        adjacent_local_recombination=adjacent_local_recombination,
        status="ok",
        method=method,
        candidate_pair_count=candidate_pair_count,
        informative_pair_count=informative_count,
        segment_count=resolved_segments,
        composite_rmse_morgan=float(np.sqrt(np.mean(residual * residual))),
        composite_median_absolute_residual_morgan=float(np.median(absolute_residual)),
        composite_p90_absolute_residual_morgan=float(
            np.quantile(
                absolute_residual,
                0.90,
            )
        ),
        minimum_pair_recombination=float(minimum_pair_recombination),
        maximum_pair_recombination=float(maximum_pair_recombination),
        rank_span_weight_exponent=float(rank_span_weight_exponent),
    )


F2_COMPOSITE_DISTANCE_CERTAINTY_THRESHOLD = 0.90
F2_COMPOSITE_DISTANCE_SEGMENT_COUNT = 16


def _f2_genetic_map_distances(
    recombination: FloatArray,
    lod: FloatArray,
    order: IntArray,
    *,
    mean_genotype_certainty: float,
) -> GeneticMapDistances:
    """Route uncertain F2 data to a regularized all-pair Kosambi fit."""

    adjacent_result = _f2_adjacent_kosambi_distances(
        recombination,
        lod,
        order,
    )
    if mean_genotype_certainty >= F2_COMPOSITE_DISTANCE_CERTAINTY_THRESHOLD:
        return adjacent_result

    ordered = np.asarray(order, dtype=np.int64)
    marker_count = ordered.size
    left_rank, right_rank = np.triu_indices(marker_count, 1)
    pair_recombination = recombination[ordered[left_rank], ordered[right_rank]]
    pair_lod = lod[ordered[left_rank], ordered[right_rank]]
    adjacent_pairwise = recombination[ordered[:-1], ordered[1:]]
    composite = _fit_genetic_map_distances_from_pairs(
        marker_count,
        ordered,
        left_rank,
        right_rank,
        pair_recombination,
        pair_lod,
        marker_bin_membership=None,
        adjacent_pairwise_recombination=adjacent_pairwise,
        adjacent_multipoint_recombination=adjacent_pairwise,
        segment_count=min(
            F2_COMPOSITE_DISTANCE_SEGMENT_COUNT,
            marker_count - 1,
        ),
        minimum_pair_recombination=0.005,
        maximum_pair_recombination=0.445,
        mapping_function="kosambi",
        rank_span_weight_exponent=DISTANCE_RANK_SPAN_WEIGHT_EXPONENT,
    )
    if composite.status != "ok":
        return replace(
            adjacent_result,
            method="f2_pairwise_kosambi_adjacent_low_information_fallback",
        )
    return replace(
        composite,
        method=f"f2_{composite.method}",
    )


def multipoint_adjacent_recombination(
    probabilities: FloatArray,
    order: IntArray,
    pairwise_recombination: FloatArray | None = None,
) -> FloatArray:
    """Estimate local RF from multipoint-smoothed state marginals.

    Direct probabilistic pairwise RFs initialize an interval-specific two-state
    HMM. Forward-backward state marginals then denoise each marker using the full
    ordered chromosome. The returned adjacent disagreement is a local RF estimate;
    it is not summed to define cumulative map distance.
    """

    p = _validate_probabilities(probabilities)
    ordered = np.asarray(order, dtype=np.int64)
    marker_count = p.shape[1]
    if (
        ordered.shape != (marker_count,)
        or np.any((ordered < 0) | (ordered >= marker_count))
        or np.unique(ordered).size != marker_count
    ):
        raise ValueError("order must be a permutation of probability columns")
    if pairwise_recombination is None:
        transitions, _ = pairwise_recombination_likelihood_edges(
            p,
            ordered[:-1],
            ordered[1:],
        )
    else:
        transitions = np.asarray(pairwise_recombination, dtype=np.float64)
        if (
            transitions.shape != (marker_count - 1,)
            or not np.all(np.isfinite(transitions))
            or np.any((transitions < 0.0) | (transitions >= 0.5))
        ):
            raise ValueError(
                "pairwise_recombination must contain one RF in [0, 0.5) per edge"
            )

    emissions = np.stack((1.0 - p[:, ordered], p[:, ordered]), axis=2)
    emissions = np.clip(emissions, 1e-12, 1.0)
    transition = np.clip(transitions, 1e-6, 0.499999)
    offspring = p.shape[0]
    alpha = np.empty((offspring, marker_count, 2), dtype=np.float64)
    beta = np.ones_like(alpha)
    raw = 0.5 * emissions[:, 0]
    alpha[:, 0] = raw / raw.sum(axis=1, keepdims=True)
    for marker in range(1, marker_count):
        rate = transition[marker - 1]
        previous = alpha[:, marker - 1]
        raw = np.empty((offspring, 2), dtype=np.float64)
        raw[:, 0] = emissions[:, marker, 0] * (
            (1.0 - rate) * previous[:, 0] + rate * previous[:, 1]
        )
        raw[:, 1] = emissions[:, marker, 1] * (
            rate * previous[:, 0] + (1.0 - rate) * previous[:, 1]
        )
        alpha[:, marker] = raw / raw.sum(axis=1, keepdims=True)
    for marker in range(marker_count - 2, -1, -1):
        rate = transition[marker]
        weighted = emissions[:, marker + 1] * beta[:, marker + 1]
        raw = np.empty((offspring, 2), dtype=np.float64)
        raw[:, 0] = (1.0 - rate) * weighted[:, 0] + rate * weighted[:, 1]
        raw[:, 1] = rate * weighted[:, 0] + (1.0 - rate) * weighted[:, 1]
        beta[:, marker] = raw / raw.sum(axis=1, keepdims=True)
    marginal = alpha * beta
    marginal /= marginal.sum(axis=2, keepdims=True)
    disagreement = np.mean(
        marginal[:, :-1, 0] * marginal[:, 1:, 1]
        + marginal[:, :-1, 1] * marginal[:, 1:, 0],
        axis=0,
    )
    return np.clip(disagreement, 0.0, 0.499999)


def estimate_genetic_map_distances(
    probabilities: FloatArray,
    order: IntArray | None = None,
    *,
    marker_bin_membership: IntArray | None = None,
    segment_count: int = 16,
    minimum_pair_recombination: float = 0.02,
    maximum_pair_recombination: float = 0.445,
    maximum_pair_count: int = 200_000,
    bisection_iterations: int = 24,
    rank_span_weight_exponent: float = DISTANCE_RANK_SPAN_WEIGHT_EXPONENT,
) -> GeneticMapDistances:
    """Estimate regularized Haldane coordinates from an ordered probability map.

    Near-zero adjacent estimates are not summed independently. Instead, many
    moderately linked marker pairs jointly determine nonnegative distances across
    a small number of monotone rank segments. Dense maps use all pairs when the
    pair budget permits; larger maps use deterministic multiscale rank offsets.
    """

    p = _validate_probabilities(probabilities)
    marker_count = p.shape[1]
    ordered_bins = (
        np.arange(marker_count, dtype=np.int64)
        if order is None
        else np.asarray(order, dtype=np.int64)
    )
    if (
        ordered_bins.shape != (marker_count,)
        or np.any((ordered_bins < 0) | (ordered_bins >= marker_count))
        or np.unique(ordered_bins).size != marker_count
    ):
        raise ValueError("order must be a permutation of probability columns")
    if not isinstance(segment_count, (int, np.integer)) or segment_count < 1:
        raise ValueError("segment_count must be a positive integer")
    if (
        not np.isfinite(minimum_pair_recombination)
        or not np.isfinite(maximum_pair_recombination)
        or not 0.0 <= minimum_pair_recombination < maximum_pair_recombination < 0.5
    ):
        raise ValueError("distance pair recombination window must lie in [0, 0.5)")
    if maximum_pair_count < 1 or bisection_iterations < 1:
        raise ValueError("pair count and bisection iterations must be positive")
    if marker_bin_membership is not None:
        membership = np.asarray(marker_bin_membership, dtype=np.int64)
        if (
            membership.ndim != 1
            or membership.size < marker_count
            or np.any((membership < 0) | (membership >= marker_count))
        ):
            raise ValueError("marker_bin_membership contains an invalid bin index")
    else:
        membership = None

    left_rank, right_rank = _distance_candidate_pairs(
        marker_count,
        maximum_pair_count,
    )
    recombination, lod = pairwise_recombination_likelihood_edges(
        p,
        ordered_bins[left_rank],
        ordered_bins[right_rank],
        bisection_iterations=bisection_iterations,
    )
    adjacent_pairwise, _ = pairwise_recombination_likelihood_edges(
        p,
        ordered_bins[:-1],
        ordered_bins[1:],
        bisection_iterations=bisection_iterations,
    )
    adjacent_pairwise = np.minimum(adjacent_pairwise, 0.499999)
    adjacent_multipoint = multipoint_adjacent_recombination(
        p,
        ordered_bins,
        adjacent_pairwise,
    )
    return _fit_genetic_map_distances_from_pairs(
        marker_count,
        ordered_bins,
        left_rank,
        right_rank,
        recombination,
        lod,
        marker_bin_membership=membership,
        adjacent_pairwise_recombination=adjacent_pairwise,
        adjacent_multipoint_recombination=adjacent_multipoint,
        segment_count=segment_count,
        minimum_pair_recombination=minimum_pair_recombination,
        maximum_pair_recombination=maximum_pair_recombination,
        rank_span_weight_exponent=rank_span_weight_exponent,
    )


def _zero_recombination_likelihood_edges(
    probabilities: FloatArray,
    left: IntArray,
    right: IntArray,
    *,
    batch_size: int = 4096,
) -> tuple[NDArray[np.bool_], FloatArray]:
    """Identify boundary-zero pairwise fits and their exact boundary LOD."""

    p = _validate_probabilities(probabilities)
    left_index = np.asarray(left, dtype=np.int64)
    right_index = np.asarray(right, dtype=np.int64)
    if left_index.ndim != 1 or right_index.shape != left_index.shape:
        raise ValueError("left and right edge arrays must be matching vectors")
    if left_index.size < 1 or batch_size < 1:
        raise ValueError("at least one marker pair and a positive batch are required")
    if np.any(
        (left_index < 0)
        | (right_index < 0)
        | (left_index >= p.shape[1])
        | (right_index >= p.shape[1])
        | (left_index == right_index)
    ):
        raise ValueError("edge arrays contain an invalid marker pair")
    at_zero = np.empty(left_index.size, dtype=bool)
    lod = np.empty(left_index.size, dtype=np.float64)
    inverse_log_ten = 1.0 / np.log(10.0)
    for start in range(0, left_index.size, batch_size):
        stop = min(start + batch_size, left_index.size)
        first = p[:, left_index[start:stop]]
        second = p[:, right_index[start:stop]]
        same = (1.0 - first) * (1.0 - second) + first * second
        different = first * (1.0 - second) + (1.0 - first) * second
        delta = different - same
        flat = np.sum(delta * delta, axis=0) <= 1e-20
        score_zero = np.sum(
            delta / np.clip(same, 1e-300, None),
            axis=0,
        )
        current_zero = (score_zero <= 0.0) & ~flat
        current_lod = np.zeros(stop - start, dtype=np.float64)
        if np.any(current_zero):
            zero_same = same[:, current_zero]
            zero_delta = delta[:, current_zero]
            linked = np.sum(
                np.log(np.clip(zero_same, 1e-300, None)),
                axis=0,
            )
            unlinked = np.sum(
                np.log(
                    np.clip(
                        zero_same + 0.5 * zero_delta,
                        1e-300,
                        None,
                    )
                ),
                axis=0,
            )
            current_lod[current_zero] = np.maximum(
                0.0,
                (linked - unlinked) * inverse_log_ten,
            )
        at_zero[start:stop] = current_zero
        lod[start:stop] = current_lod
    return at_zero, lod


def likelihood_bin_markers(
    probabilities: FloatArray,
    *,
    maximum_bin_recombination: float = 0.0,
    minimum_linkage_lod: float = 3.0,
    neighbor_count: int = 32,
    neighbor_projection_dimensions: int = 16,
    neighbor_projection_minimum_markers: int = 50_000,
    maximum_pool_evidence: float | None = None,
    bisection_iterations: int = 32,
    edge_batch_size: int = 4096,
) -> MarkerBins:
    """Pool locally similar markers with no supported recombination signal.

    Candidate pairs come from a deterministic nearest-neighbor search in
    posterior space. A lower-information marker joins a fixed high-information
    representative only when its pairwise likelihood estimate has sufficiently
    small recombination and enough linkage evidence. Bins never merge transitively,
    preventing chains of weakly separated markers from collapsing a chromosome.
    """

    p = _validate_probabilities(probabilities)
    if (
        not np.isfinite(maximum_bin_recombination)
        or not 0.0 <= maximum_bin_recombination < 0.5
    ):
        raise ValueError("maximum bin recombination must lie in [0, 0.5)")
    if not np.isfinite(minimum_linkage_lod) or minimum_linkage_lod < 0.0:
        raise ValueError("minimum linkage LOD must be finite and nonnegative")
    if maximum_pool_evidence is not None and (
        not np.isfinite(maximum_pool_evidence) or maximum_pool_evidence <= 0.0
    ):
        raise ValueError("maximum pool evidence must be positive and finite")
    if neighbor_projection_dimensions < 2:
        raise ValueError("neighbor projection dimensions must be at least two")
    if neighbor_projection_minimum_markers < 3:
        raise ValueError("neighbor projection marker threshold must be at least three")
    projection_dimensions = (
        neighbor_projection_dimensions
        if p.shape[1] >= neighbor_projection_minimum_markers
        else None
    )
    neighbors, _ = _candidate_neighbors(
        p,
        neighbor_count,
        projection_dimensions=projection_dimensions,
    )
    marker_count = p.shape[1]
    source = np.repeat(np.arange(marker_count, dtype=np.int64), neighbors.shape[1])
    target = neighbors.reshape(-1)
    low = np.minimum(source, target)
    high = np.maximum(source, target)
    distinct = low != high
    encoded = np.unique(low[distinct] * marker_count + high[distinct])
    edge_left = encoded // marker_count
    edge_right = encoded % marker_count
    if maximum_bin_recombination == 0.0:
        edge_qualifies, edge_lod = _zero_recombination_likelihood_edges(
            p,
            edge_left,
            edge_right,
            batch_size=edge_batch_size,
        )
    else:
        edge_rf, edge_lod = pairwise_recombination_likelihood_edges(
            p,
            edge_left,
            edge_right,
            bisection_iterations=bisection_iterations,
            batch_size=edge_batch_size,
        )
        edge_qualifies = edge_rf <= maximum_bin_recombination
    edge_lookup = {int(code): index for index, code in enumerate(encoded)}
    information = np.mean(np.abs(2.0 * p - 1.0), axis=0)
    membership = np.full(marker_count, -1, dtype=np.int64)
    representatives: list[int] = []
    for marker_value in np.argsort(-information, kind="stable"):
        marker = int(marker_value)
        if membership[marker] >= 0:
            continue
        group = len(representatives)
        representatives.append(marker)
        membership[marker] = group
        for candidate_value in neighbors[marker]:
            candidate = int(candidate_value)
            if candidate == marker or membership[candidate] >= 0:
                continue
            code = min(marker, candidate) * marker_count + max(marker, candidate)
            edge_index = edge_lookup.get(code)
            if edge_index is None:
                continue
            if (
                edge_qualifies[edge_index]
                and edge_lod[edge_index] >= minimum_linkage_lod
            ):
                membership[candidate] = group
    return _pool_marker_bins(
        p,
        membership,
        np.asarray(representatives, dtype=np.int64),
        maximum_bin_recombination,
        maximum_evidence=maximum_pool_evidence,
    )


def _project_to_polyline(
    points: FloatArray,
    curve: FloatArray,
) -> tuple[FloatArray, IntArray, float]:
    """Project points to a polyline and return arclength, order, and SSE."""

    starts = curve[:-1]
    directions = curve[1:] - starts
    squared_lengths = np.sum(directions * directions, axis=1)
    lengths = np.sqrt(squared_lengths)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    best_arclength = np.zeros(points.shape[0], dtype=np.float64)
    for segment, (start, direction) in enumerate(zip(starts, directions, strict=True)):
        denominator = max(float(squared_lengths[segment]), 1e-30)
        fraction = np.clip((points - start) @ direction / denominator, 0.0, 1.0)
        projection = start + fraction[:, None] * direction
        distance = np.sum((points - projection) ** 2, axis=1)
        improved = distance < best_distance
        best_distance[improved] = distance[improved]
        best_arclength[improved] = (
            cumulative[segment] + fraction[improved] * lengths[segment]
        )
    return (
        best_arclength,
        np.argsort(best_arclength, kind="stable").astype(np.int64),
        float(np.sum(best_distance)),
    )


@lru_cache(maxsize=16)
def _rank_smoothing_matrix(
    marker_count: int,
    effective_degrees_of_freedom: float,
) -> FloatArray:
    """Return a deterministic second-difference smoother at a target EDF."""

    difference = np.diff(np.eye(marker_count), n=2, axis=0)
    eigenvalues, eigenvectors = linalg.eigh(
        difference.T @ difference,
        check_finite=False,
    )
    low = 0.0
    high = 1.0

    def degrees_of_freedom(penalty: float) -> float:
        return float(np.sum(1.0 / (1.0 + penalty * eigenvalues)))

    while degrees_of_freedom(high) > effective_degrees_of_freedom:
        high *= 10.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if degrees_of_freedom(middle) > effective_degrees_of_freedom:
            low = middle
        else:
            high = middle
    shrinkage = 1.0 / (1.0 + high * eigenvalues)
    smoother = (eigenvectors * shrinkage) @ eigenvectors.T
    smoother.setflags(write=False)
    return smoother


def _smooth_penalized_principal_curve_order(
    coordinates: FloatArray,
    *,
    effective_degrees_of_freedom: float = 4.0,
    maximum_iterations: int = 50,
    relative_tolerance: float = 1e-3,
) -> IntArray:
    """Order coordinates with a low-variance, fixed-EDF principal curve."""

    points = np.asarray(coordinates, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    initial_coordinate = left_vectors[:, 0] * singular_values[0]
    initial_curve = np.outer(np.sort(initial_coordinate), right_vectors[0]) + np.mean(
        points, axis=0
    )
    arclength, _, previous_distance = _project_to_polyline(
        points,
        initial_curve,
    )
    smoother = _rank_smoothing_matrix(
        points.shape[0],
        float(effective_degrees_of_freedom),
    )
    for _ in range(maximum_iterations):
        ranked = np.argsort(arclength, kind="stable")
        curve = smoother @ points[ranked]
        arclength, _, distance = _project_to_polyline(points, curve)
        if abs(previous_distance - distance) <= (
            relative_tolerance * max(previous_distance, 1e-30)
        ):
            break
        previous_distance = distance
    return np.lexsort((initial_coordinate, arclength)).astype(np.int64)


def _smooth_principal_curve_order(
    coordinates: FloatArray,
    *,
    interior_knots: int = 2,
    maximum_iterations: int = 50,
    relative_tolerance: float = 1e-3,
) -> IntArray:
    """Order weighted-MDS coordinates with a compact principal-curve fit."""

    if interior_knots < 0:
        raise ValueError("principal curve complexity must be nonnegative")
    points = np.asarray(coordinates, dtype=np.float64)
    if interior_knots == 0:
        return _smooth_penalized_principal_curve_order(
            points,
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
        )
    centered = points - np.mean(points, axis=0)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    initial_coordinate = left_vectors[:, 0] * singular_values[0]
    initial_curve = np.outer(np.sort(initial_coordinate), right_vectors[0]) + np.mean(
        points, axis=0
    )
    arclength, _, previous_distance = _project_to_polyline(points, initial_curve)
    marker_count = points.shape[0]
    for _ in range(maximum_iterations):
        ranked = np.argsort(arclength, kind="stable")
        coordinate = arclength[ranked].copy()
        coordinate += (
            np.arange(marker_count, dtype=np.float64)
            * max(float(np.ptp(coordinate)), 1.0)
            * 1e-12
        )
        knots = np.quantile(
            coordinate,
            np.linspace(0.0, 1.0, interior_knots + 2)[1:-1],
        )
        curve = np.column_stack(
            [
                LSQUnivariateSpline(
                    coordinate,
                    points[ranked, dimension],
                    knots,
                    k=3,
                )(coordinate)
                for dimension in range(points.shape[1])
            ]
        )
        arclength, _, distance = _project_to_polyline(points, curve)
        if abs(previous_distance - distance) <= (
            relative_tolerance * max(previous_distance, 1e-30)
        ):
            break
        previous_distance = distance
    # Break rare projection ties reproducibly in the initial principal direction.
    return np.lexsort((initial_coordinate, arclength)).astype(np.int64)


def _likelihood_weighted_mds_coordinates(
    recombination: FloatArray,
    lod: FloatArray,
    *,
    distance: str = "rf",
    lod_exponent: float = 3.0,
    dimensions: int = 20,
    maximum_smacof_iterations: int = 500,
    smacof_tolerance: float = 1e-7,
    weight_system: tuple[FloatArray, FloatArray] | None = None,
) -> FloatArray:
    """Fit the reusable LOD-weighted MDS embedding for one geometry.

    Raw recombination fractions are the default dissimilarity.  Unlike map
    functions, they remain bounded when weak pairs approach ``r = 0.5``; the LOD
    weights then keep those weak pairs from distorting the embedding.
    """

    rf = np.asarray(recombination, dtype=np.float64)
    linkage_lod = np.asarray(lod, dtype=np.float64)
    if rf.ndim != 2 or rf.shape[0] != rf.shape[1] or rf.shape[0] < 3:
        raise ValueError("recombination must be a square matrix of at least size 3")
    if linkage_lod.shape != rf.shape:
        raise ValueError("LOD matrix must match recombination matrix")
    if not np.all(np.isfinite(rf)) or np.any((rf < 0.0) | (rf >= 0.5)):
        raise ValueError("recombination fractions must lie in [0, 0.5)")
    if not np.all(np.isfinite(linkage_lod)) or np.any(linkage_lod < 0.0):
        raise ValueError("LOD scores must be finite and nonnegative")
    if lod_exponent <= 0.0 or not np.isfinite(lod_exponent):
        raise ValueError("lod_exponent must be positive and finite")
    marker_count = rf.shape[0]
    dimensions = min(max(1, dimensions), marker_count - 1)
    if distance == "rf":
        dissimilarity = rf.copy()
    elif distance == "haldane":
        dissimilarity = -0.5 * np.log(np.clip(1.0 - 2.0 * rf, 1e-16, None))
    elif distance == "kosambi":
        dissimilarity = 0.25 * np.log(
            np.clip((1.0 + 2.0 * rf) / (1.0 - 2.0 * rf), 1e-16, None)
        )
    else:
        raise ValueError("distance must be 'rf', 'haldane', or 'kosambi'")
    np.fill_diagonal(dissimilarity, 0.0)
    if weight_system is None:
        weights = linkage_lod**lod_exponent
        np.fill_diagonal(weights, 0.0)
        maximum_weight = float(np.max(weights))
        if maximum_weight <= 0.0:
            raise ValueError("at least one marker pair must have positive linkage LOD")
        weights /= maximum_weight
        weight_laplacian = np.diag(np.sum(weights, axis=1)) - weights
        inverse_laplacian = linalg.pinvh(weight_laplacian, check_finite=False)
    else:
        weights, inverse_laplacian = weight_system
        if weights.shape != rf.shape or inverse_laplacian.shape != rf.shape:
            raise ValueError("cached MDS weight system must match recombination")

    squared_dissimilarity = dissimilarity * dissimilarity
    row_mean = np.mean(squared_dissimilarity, axis=1, keepdims=True)
    gram = -0.5 * (
        squared_dissimilarity
        - row_mean
        - row_mean.T
        + float(np.mean(squared_dissimilarity))
    )
    eigenvalues, eigenvectors = linalg.eigh(
        gram,
        subset_by_index=[marker_count - dimensions, marker_count - 1],
        driver="evr",
        check_finite=False,
    )
    positive = eigenvalues > 1e-12
    if not np.any(positive):
        raise ValueError("dissimilarities do not define a positive MDS direction")
    coordinates = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    previous_stress: float | None = None
    for _ in range(maximum_smacof_iterations):
        fitted_distances = cdist(coordinates, coordinates)
        ratio = np.divide(
            weights * dissimilarity,
            fitted_distances,
            out=np.zeros_like(dissimilarity),
            where=fitted_distances > 1e-12,
        )
        majorization = -ratio
        np.fill_diagonal(majorization, -np.sum(majorization, axis=1))
        updated = inverse_laplacian @ majorization @ coordinates
        updated -= np.mean(updated, axis=0)
        stress = 0.5 * float(
            np.sum(weights * (dissimilarity - cdist(updated, updated)) ** 2)
        )
        coordinates = updated
        if previous_stress is not None and abs(previous_stress - stress) <= (
            smacof_tolerance * max(previous_stress, 1.0)
        ):
            break
        previous_stress = stress
    return coordinates


def likelihood_weighted_mds_order(
    recombination: FloatArray,
    lod: FloatArray,
    *,
    distance: str = "rf",
    lod_exponent: float = 3.0,
    dimensions: int = 20,
    principal_curve_knots: int = 2,
    maximum_smacof_iterations: int = 500,
    smacof_tolerance: float = 1e-7,
) -> IntArray:
    """Order markers by LOD-weighted MDS and a smooth principal curve."""

    coordinates = _likelihood_weighted_mds_coordinates(
        recombination,
        lod,
        distance=distance,
        lod_exponent=lod_exponent,
        dimensions=dimensions,
        maximum_smacof_iterations=maximum_smacof_iterations,
        smacof_tolerance=smacof_tolerance,
    )
    return _smooth_principal_curve_order(
        coordinates,
        interior_knots=principal_curve_knots,
    )


def _sparse_graph(probabilities: FloatArray, neighbor_count: int) -> coo_matrix:
    n_markers = probabilities.shape[1]
    if n_markers == 2:
        distance = expected_disagreement(probabilities[:, 0], probabilities[:, 1])
        affinity = np.exp(-distance / 0.05)
        return coo_matrix(([affinity, affinity], ([0, 1], [1, 0])), shape=(2, 2))

    neighbors, _ = _candidate_neighbors(probabilities, neighbor_count)
    edges: dict[tuple[int, int], float] = {}
    raw_distances: list[float] = []
    for marker in range(n_markers):
        for candidate in neighbors[marker]:
            candidate = int(candidate)
            if marker == candidate:
                continue
            key = (min(marker, candidate), max(marker, candidate))
            if key not in edges:
                value = expected_disagreement(
                    probabilities[:, marker], probabilities[:, candidate]
                )
                edges[key] = value
                raw_distances.append(value)
    positive = np.asarray([d for d in raw_distances if d > 1e-9], dtype=float)
    scale = float(np.median(positive)) if positive.size else 0.05
    scale = max(scale, 1e-3)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (left, right), distance in edges.items():
        affinity = float(np.exp(-distance / scale))
        rows.extend((left, right))
        cols.extend((right, left))
        data.extend((affinity, affinity))
    graph = coo_matrix((data, (rows, cols)), shape=(n_markers, n_markers))
    components, labels = csgraph.connected_components(graph, directed=False)
    if components > 1:
        # Link components conservatively using their closest feature-space pair.
        features = (probabilities.T - 0.5) * 2.0
        for component in range(1, components):
            current = np.flatnonzero(labels == component)
            previous = np.flatnonzero(labels != component)
            tree = cKDTree(features[previous])
            distances, positions = tree.query(features[current], k=1)
            local = int(np.argmin(distances))
            left = int(current[local])
            right = int(previous[int(positions[local])])
            affinity = float(
                np.exp(
                    -expected_disagreement(
                        probabilities[:, left], probabilities[:, right]
                    )
                    / scale
                )
            )
            rows.extend((left, right))
            cols.extend((right, left))
            data.extend((affinity, affinity))
        graph = coo_matrix((data, (rows, cols)), shape=(n_markers, n_markers))
    return graph


def _path_cost(order: IntArray, probabilities: FloatArray) -> float:
    return sum(
        expected_disagreement(probabilities[:, int(a)], probabilities[:, int(b)])
        for a, b in pairwise(order)
    )


def _polish_order(
    order: IntArray, probabilities: FloatArray, passes: int = 2
) -> IntArray:
    """Cheap insertion polishing of a spectral order."""

    result = list(map(int, order))
    if len(result) < 4:
        return np.asarray(result, dtype=np.int64)
    for _ in range(passes):
        changed = False
        for index in range(1, len(result) - 1):
            marker = result.pop(index)
            low = max(0, index - 8)
            high = min(len(result), index + 8)
            candidates = range(low, high + 1)
            best_position = index
            best_cost = np.inf
            for position in candidates:
                trial = result.copy()
                trial.insert(position, marker)
                start = max(0, min(index, position) - 1)
                stop = min(len(trial), max(index, position) + 2)
                local = np.asarray(trial[start:stop], dtype=np.int64)
                cost = _path_cost(local, probabilities)
                if cost < best_cost:
                    best_cost = cost
                    best_position = position
            result.insert(best_position, marker)
            changed |= best_position != index
        if not changed:
            break
    return np.asarray(result, dtype=np.int64)


def _two_opt_path(
    order: IntArray,
    distances: FloatArray,
    *,
    max_passes: int = 20,
) -> IntArray:
    """Remove path crossings by reversing the best improving segment."""

    result = np.asarray(order, dtype=np.int64).copy()
    n_markers = result.size
    for _ in range(max_passes):
        best_delta = -1e-12
        best_segment: tuple[int, int] | None = None
        for left in range(n_markers - 1):
            for right in range(left + 1, n_markers):
                old = new = 0.0
                if left > 0:
                    old += distances[result[left - 1], result[left]]
                    new += distances[result[left - 1], result[right]]
                if right < n_markers - 1:
                    old += distances[result[right], result[right + 1]]
                    new += distances[result[left], result[right + 1]]
                delta = new - old
                if delta < best_delta:
                    best_delta = delta
                    best_segment = (left, right)
        if best_segment is None:
            break
        left, right = best_segment
        result[left : right + 1] = result[left : right + 1][::-1]
    return result


def _spectral_order(probabilities: FloatArray, neighbor_count: int) -> IntArray:
    p = probabilities
    n_markers = p.shape[1]
    if n_markers == 2:
        return np.array([0, 1], dtype=np.int64)
    if n_markers <= 2_500:
        distances = _dense_distances(p)
        positive = distances[(distances > 1e-9) & np.isfinite(distances)]
        scale = float(np.median(positive)) if positive.size else 0.05
        affinity = np.exp(-distances / max(scale, 1e-3))
        np.fill_diagonal(affinity, 0.0)
        laplacian = csgraph.laplacian(affinity, normed=True)
        if n_markers > 200:
            _, vectors = linalg.eigh(
                laplacian,
                subset_by_index=[0, 1],
                driver="evr",
                check_finite=False,
            )
        else:
            _, vectors = np.linalg.eigh(laplacian)
        coordinate = vectors[:, 1]
    else:
        graph = _sparse_graph(p, neighbor_count).tocsr()
        laplacian = csgraph.laplacian(graph, normed=True)
        try:
            _, vectors = eigsh(laplacian, k=2, which="SM", tol=1e-5)
            coordinate = vectors[:, 1]
        except (ArpackNoConvergence, np.linalg.LinAlgError, ValueError):
            _, vectors = np.linalg.eigh(laplacian.toarray())
            coordinate = vectors[:, 1]
    order = np.argsort(coordinate, kind="stable").astype(np.int64)
    if n_markers <= 2_500:
        return _two_opt_path(order, distances)
    return _polish_order(order, p)


def _ensemble_order(probabilities: FloatArray, neighbor_count: int) -> IntArray:
    """Select spectral, MST-diameter, or optimal-leaf order by HMM likelihood."""

    p = probabilities
    spectral = _spectral_order(p, neighbor_count)
    if p.shape[1] > 200:
        return spectral
    distances = _dense_distances(p)

    mst = csgraph.minimum_spanning_tree(distances)
    tree = (mst + mst.T).tocsr()
    from_zero = csgraph.dijkstra(tree, directed=False, indices=0)
    endpoint = int(np.argmax(from_zero))
    coordinate = csgraph.dijkstra(tree, directed=False, indices=endpoint)
    mst_order = np.argsort(coordinate, kind="stable").astype(np.int64)
    mst_order = _two_opt_path(mst_order, distances)

    condensed = squareform(distances, checks=False)
    hierarchy = linkage(condensed, method="average")
    hierarchy = optimal_leaf_ordering(hierarchy, condensed)
    leaf_order = leaves_list(hierarchy).astype(np.int64)
    leaf_order = _two_opt_path(leaf_order, distances)

    candidates = (spectral, mst_order, leaf_order)
    scores = [hmm_log_likelihood(p, order) for order in candidates]
    return candidates[int(np.argmax(scores))]


def _hmm_smooth(probabilities: FloatArray, order: IntArray) -> FloatArray:
    """Posterior parental-origin probabilities under a two-state crossover HMM."""

    p = probabilities[:, order]
    n_offspring, n_markers = p.shape
    if n_markers < 3:
        return probabilities.copy()
    adjacent = np.asarray(
        [
            expected_disagreement(p[:, index], p[:, index + 1])
            for index in range(n_markers - 1)
        ]
    )
    transition = float(np.clip(np.median(adjacent), 1e-4, 0.05))
    emissions = np.stack((1.0 - p, p), axis=2)
    emissions = np.clip(emissions, 1e-9, 1.0)
    alpha = np.empty((n_offspring, n_markers, 2), dtype=np.float64)
    beta = np.ones_like(alpha)
    alpha[:, 0] = 0.5 * emissions[:, 0]
    alpha[:, 0] /= alpha[:, 0].sum(axis=1, keepdims=True)
    for marker in range(1, n_markers):
        previous = alpha[:, marker - 1]
        predicted_zero = (1.0 - transition) * previous[:, 0] + transition * previous[
            :, 1
        ]
        predicted_one = (
            transition * previous[:, 0] + (1.0 - transition) * previous[:, 1]
        )
        alpha[:, marker, 0] = emissions[:, marker, 0] * predicted_zero
        alpha[:, marker, 1] = emissions[:, marker, 1] * predicted_one
        alpha[:, marker] /= alpha[:, marker].sum(axis=1, keepdims=True)
    for marker in range(n_markers - 2, -1, -1):
        next_weighted = emissions[:, marker + 1] * beta[:, marker + 1]
        beta[:, marker, 0] = (1.0 - transition) * next_weighted[
            :, 0
        ] + transition * next_weighted[:, 1]
        beta[:, marker, 1] = (
            transition * next_weighted[:, 0] + (1.0 - transition) * next_weighted[:, 1]
        )
        beta[:, marker] /= beta[:, marker].sum(axis=1, keepdims=True)
    posterior = alpha * beta
    posterior /= posterior.sum(axis=2, keepdims=True)
    smoothed = np.empty_like(probabilities)
    smoothed[:, order] = posterior[:, :, 1]
    return smoothed


def _hmm_smooth_with_interval_transitions(
    probabilities: FloatArray,
    order: IntArray,
    transitions: FloatArray,
) -> FloatArray:
    """Smooth genotypes with a separately estimated transition for every edge."""

    p = probabilities[:, order]
    n_offspring, n_markers = p.shape
    transition = np.clip(np.asarray(transitions, dtype=float), 1e-4, 0.25)
    if transition.shape != (n_markers - 1,):
        raise ValueError("transition vector must match the marker order")
    emissions = np.stack((1.0 - p, p), axis=2)
    emissions = np.clip(emissions, 1e-9, 1.0)
    alpha = np.empty((n_offspring, n_markers, 2), dtype=np.float64)
    beta = np.ones_like(alpha)
    alpha[:, 0] = 0.5 * emissions[:, 0]
    alpha[:, 0] /= alpha[:, 0].sum(axis=1, keepdims=True)
    for marker in range(1, n_markers):
        rate = transition[marker - 1]
        previous = alpha[:, marker - 1]
        predicted_zero = (1.0 - rate) * previous[:, 0] + rate * previous[:, 1]
        predicted_one = rate * previous[:, 0] + (1.0 - rate) * previous[:, 1]
        alpha[:, marker, 0] = emissions[:, marker, 0] * predicted_zero
        alpha[:, marker, 1] = emissions[:, marker, 1] * predicted_one
        alpha[:, marker] /= alpha[:, marker].sum(axis=1, keepdims=True)
    for marker in range(n_markers - 2, -1, -1):
        rate = transition[marker]
        next_weighted = emissions[:, marker + 1] * beta[:, marker + 1]
        beta[:, marker, 0] = (1.0 - rate) * next_weighted[:, 0] + rate * next_weighted[
            :, 1
        ]
        beta[:, marker, 1] = (
            rate * next_weighted[:, 0] + (1.0 - rate) * next_weighted[:, 1]
        )
        beta[:, marker] /= beta[:, marker].sum(axis=1, keepdims=True)
    posterior = alpha * beta
    posterior /= posterior.sum(axis=2, keepdims=True)
    smoothed = np.empty_like(probabilities)
    smoothed[:, order] = posterior[:, :, 1]
    return smoothed


def order_markers(
    probabilities: FloatArray,
    *,
    neighbor_count: int = 20,
    hmm_iterations: int = 1,
    ordering_ensemble: bool = False,
) -> IntArray:
    """Order markers by seriation, path optimization, and multipoint smoothing."""

    p = _validate_probabilities(probabilities)
    order_function = _ensemble_order if ordering_ensemble else _spectral_order
    order = order_function(p, neighbor_count)
    for _ in range(max(0, hmm_iterations)):
        smoothed = _hmm_smooth(p, order)
        updated = order_function(smoothed, neighbor_count)
        if np.array_equal(updated, order) or np.array_equal(updated, order[::-1]):
            break
        order = updated
    return order


def _align_to_reference(order: IntArray, reference_positions: IntArray) -> IntArray:
    forward = reference_positions[order]
    reverse = reference_positions[order[::-1]]
    target = np.arange(order.size)
    forward_error = np.sum(np.abs(rankdata(forward) - target - 1))
    reverse_error = np.sum(np.abs(rankdata(reverse) - target - 1))
    return order if forward_error <= reverse_error else order[::-1].copy()


def _likelihood_mds_bootstrap_order_worker(
    payload: tuple[FloatArray, IntArray, str, float, int, int, int],
) -> IntArray:
    """Order one prepared bootstrap matrix in a process-safe worker."""

    (
        boot,
        reference_positions,
        distance,
        lod_exponent,
        dimensions,
        principal_curve_knots,
        maximum_smacof_iterations,
    ) = payload
    recombination, lod = pairwise_recombination_likelihood(boot)
    order = likelihood_weighted_mds_order(
        recombination,
        lod,
        distance=distance,
        lod_exponent=lod_exponent,
        dimensions=dimensions,
        principal_curve_knots=principal_curve_knots,
        maximum_smacof_iterations=maximum_smacof_iterations,
    )
    return _align_to_reference(order, reference_positions)


def bootstrap_orders(
    probabilities: FloatArray,
    reference_order: IntArray,
    *,
    replicates: int = 100,
    neighbor_count: int = 20,
    sample_states: bool = True,
    resample_offspring: bool = True,
    ordering_ensemble: bool = False,
    random_seed: int | None = None,
) -> IntArray:
    """Return bootstrap position of each marker in every resampled map."""

    p = _validate_probabilities(probabilities)
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    rng = np.random.default_rng(random_seed)
    reference_positions = np.empty(reference_order.size, dtype=np.int64)
    reference_positions[reference_order] = np.arange(reference_order.size)
    positions = np.empty((replicates, p.shape[1]), dtype=np.int64)
    for replicate in range(replicates):
        rows = (
            rng.integers(0, p.shape[0], size=p.shape[0])
            if resample_offspring
            else np.arange(p.shape[0])
        )
        boot = p[rows]
        if sample_states:
            boot = (rng.random(boot.shape) < boot).astype(np.float64)
            # Keep a sliver of uncertainty to prevent zero-information ties.
            boot = boot * 0.998 + 0.001
        order = order_markers(
            boot,
            neighbor_count=neighbor_count,
            ordering_ensemble=ordering_ensemble,
        )
        order = _align_to_reference(order, reference_positions)
        positions[replicate, order] = np.arange(order.size)
    return positions


def bootstrap_likelihood_mds_orders(
    probabilities: FloatArray,
    reference_order: IntArray,
    *,
    replicates: int = 100,
    sample_states: bool = True,
    resample_offspring: bool = True,
    distance: str = "haldane",
    lod_exponent: float = 3.0,
    dimensions: int = 10,
    principal_curve_knots: int = 2,
    maximum_smacof_iterations: int = 500,
    jobs: int = 1,
    random_seed: int | None = None,
) -> IntArray:
    """Bootstrap likelihood-MDS maps and return aligned marker positions.

    Each replicate resamples offspring and, by default, draws latent inheritance
    states from their posterior probabilities before re-estimating the complete
    pairwise likelihood surface.  Orientation is aligned to the full-data map;
    no truth coordinates are used.
    """

    p = _validate_probabilities(probabilities)
    reference = np.asarray(reference_order, dtype=np.int64)
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    if jobs < 1:
        raise ValueError("bootstrap jobs must be positive")
    if (
        reference.ndim != 1
        or reference.size != p.shape[1]
        or np.unique(reference).size != reference.size
        or np.any((reference < 0) | (reference >= p.shape[1]))
    ):
        raise ValueError("reference_order must be a permutation of all markers")
    rng = np.random.default_rng(random_seed)
    reference_positions = np.empty(reference.size, dtype=np.int64)
    reference_positions[reference] = np.arange(reference.size)
    positions = np.empty((replicates, p.shape[1]), dtype=np.int64)

    def make_bootstrap() -> FloatArray:
        rows = (
            rng.integers(0, p.shape[0], size=p.shape[0])
            if resample_offspring
            else np.arange(p.shape[0])
        )
        boot = p[rows]
        if sample_states:
            boot = (rng.random(boot.shape) < boot).astype(np.float64)
            boot = boot * 0.998 + 0.001
        return boot

    def ordering_payload(
        boot: FloatArray,
    ) -> tuple[FloatArray, IntArray, str, float, int, int, int]:
        return (
            boot,
            reference_positions,
            distance,
            lod_exponent,
            dimensions,
            principal_curve_knots,
            maximum_smacof_iterations,
        )

    if jobs == 1:
        orders = [
            _likelihood_mds_bootstrap_order_worker(ordering_payload(make_bootstrap()))
            for _ in range(replicates)
        ]
    else:
        orders: list[IntArray | None] = [None] * replicates
        pending: dict[int, Future[IntArray]] = {}
        maximum_pending = 2 * jobs
        # ``spawn`` remains safe after NumPy/BLAS or the optional Rust backend has
        # initialized worker threads; forking a multithreaded process can deadlock.
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=get_context("spawn"),
        ) as executor:
            for replicate in range(replicates):
                pending[replicate] = executor.submit(
                    _likelihood_mds_bootstrap_order_worker,
                    ordering_payload(make_bootstrap()),
                )
                if len(pending) >= maximum_pending:
                    first = min(pending)
                    orders[first] = pending.pop(first).result()
            for replicate in sorted(pending):
                orders[replicate] = pending[replicate].result()
    for replicate, order_value in enumerate(orders):
        if order_value is None:
            raise RuntimeError("bootstrap order worker returned no result")
        order = order_value
        positions[replicate, order] = np.arange(order.size)
    return positions


def precedence_matrix(positions: IntArray) -> FloatArray:
    """Pairwise bootstrap probability that row marker precedes column marker."""

    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 2:
        raise ValueError("positions must be a bootstrap-by-marker matrix")
    return np.mean(pos[:, :, None] < pos[:, None, :], axis=0, dtype=np.float64)


def select_framework(
    reference_order: IntArray,
    precedence: FloatArray,
    *,
    confidence: float,
) -> IntArray:
    """Greedily select a supported backbone in reference-order coordinates."""

    ordered = np.asarray(reference_order, dtype=np.int64)
    best: list[int] = []
    # A single early unstable marker can make a one-pass greedy framework much
    # smaller. Try each possible first anchor and retain the longest pairwise-
    # compatible chain. This is deterministic and remains cheap after binning.
    for start in range(ordered.size):
        selected = [int(ordered[start])]
        for marker in map(int, ordered[start + 1 :]):
            if np.all(precedence[np.asarray(selected), marker] >= confidence):
                selected.append(marker)
        if len(selected) > len(best):
            best = selected
    if len(best) < 2:
        best = [int(ordered[0]), int(ordered[-1])]
    return np.asarray(best, dtype=np.int64)


def framework_exact_support(framework: IntArray, positions: IntArray) -> float:
    """Fraction of bootstrap maps with the complete framework order unchanged."""

    anchors = np.asarray(framework, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    if anchors.ndim != 1 or anchors.size < 2:
        raise ValueError("at least two framework markers are required")
    if pos.ndim != 2:
        raise ValueError("positions must be a bootstrap-by-marker matrix")
    if np.any((anchors < 0) | (anchors >= pos.shape[1])):
        raise ValueError("framework contains an invalid marker index")
    ordered_positions = pos[:, anchors]
    return float(np.mean(np.all(np.diff(ordered_positions, axis=1) > 0, axis=1)))


def select_framework_global(
    reference_order: IntArray,
    positions: IntArray,
    *,
    confidence: float,
) -> IntArray:
    """Select a backbone with globally supported induced order across bootstraps."""

    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    selected = list(map(int, np.asarray(reference_order, dtype=np.int64)))
    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 2 or len(selected) != pos.shape[1]:
        raise ValueError("reference_order and bootstrap positions do not match")
    if len(set(selected)) != len(selected):
        raise ValueError("reference_order marker indices must be unique")
    rank_variance = np.var(pos, axis=0)
    while len(selected) > 2:
        selected_array = np.asarray(selected, dtype=np.int64)
        ordered_positions = pos[:, selected_array]
        valid = np.all(np.diff(ordered_positions, axis=1) > 0, axis=1)
        if float(np.mean(valid)) >= confidence:
            break
        descents = np.diff(ordered_positions, axis=1) <= 0
        involvement = np.zeros(len(selected), dtype=np.int64)
        involvement[:-1] += np.sum(descents, axis=0)
        involvement[1:] += np.sum(descents, axis=0)
        worst = np.flatnonzero(involvement == involvement.max())
        if worst.size > 1:
            variances = rank_variance[selected_array[worst]]
            remove_at = int(worst[int(np.argmax(variances))])
        else:
            remove_at = int(worst[0])
        selected.pop(remove_at)
    return np.asarray(selected, dtype=np.int64)


def placement_intervals(
    framework: IntArray,
    precedence: FloatArray,
    *,
    confidence: float,
) -> tuple[IntArray, IntArray]:
    """Two-sided supported framework interval for every representative marker.

    ``confidence`` is the requested coverage of the complete interval.  Each
    boundary therefore receives half of the error budget: a nominal 80% interval
    uses 90% one-sided precedence support on its left and right boundaries.
    """

    n_markers = precedence.shape[0]
    one_sided_confidence = 1.0 - (1.0 - confidence) / 2.0
    left = np.full(n_markers, -1, dtype=np.int64)
    right = np.full(n_markers, len(framework), dtype=np.int64)
    framework_lookup = {int(marker): index for index, marker in enumerate(framework)}
    for marker in range(n_markers):
        if marker in framework_lookup:
            index = framework_lookup[marker]
            left[marker] = right[marker] = index
            continue
        lower = [
            index
            for index, anchor in enumerate(framework)
            if precedence[int(anchor), marker] >= one_sided_confidence
        ]
        upper = [
            index
            for index, anchor in enumerate(framework)
            if precedence[marker, int(anchor)] >= one_sided_confidence
        ]
        if lower:
            left[marker] = max(lower)
        if upper:
            right[marker] = min(upper)
    return left, right


def bootstrap_placement_intervals(
    framework: IntArray,
    positions: IntArray,
    *,
    confidence: float,
) -> tuple[IntArray, IntArray]:
    """Shortest framework-slot interval containing the requested bootstrap mass."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap interval mass must be between 0 and 1")
    pos = np.asarray(positions, dtype=np.int64)
    anchors = np.asarray(framework, dtype=np.int64)
    if pos.ndim != 2:
        raise ValueError("positions must be a bootstrap-by-marker matrix")
    if anchors.ndim != 1 or anchors.size < 2:
        raise ValueError("at least two framework markers are required")
    if np.any((anchors < 0) | (anchors >= pos.shape[1])):
        raise ValueError("framework contains an invalid marker index")
    if np.unique(anchors).size != anchors.size:
        raise ValueError("framework marker indices must be unique")

    n_replicates, n_markers = pos.shape
    mass = max(1, int(np.ceil(confidence * n_replicates)))
    left = np.full(n_markers, -1, dtype=np.int64)
    right = np.full(n_markers, anchors.size, dtype=np.int64)
    framework_lookup = {int(marker): rank for rank, marker in enumerate(anchors)}
    anchor_positions = pos[:, anchors]
    for marker in range(n_markers):
        if marker in framework_lookup:
            rank = framework_lookup[marker]
            left[marker] = right[marker] = rank
            continue
        slots = np.sum(anchor_positions < pos[:, marker, None], axis=1)
        slots.sort()
        widths = slots[mass - 1 :] - slots[: n_replicates - mass + 1]
        best_start = int(np.argmin(widths))
        low_slot = int(slots[best_start])
        high_slot = int(slots[best_start + mass - 1])
        left[marker] = low_slot - 1
        right[marker] = high_slot
    return left, right


def bootstrap_rank_intervals(
    positions: IntArray,
    *,
    confidence: float,
    method: str = "central",
) -> tuple[IntArray, IntArray]:
    """Absolute-rank bootstrap intervals with central or shortest construction."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap interval mass must be between 0 and 1")
    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] < 2:
        raise ValueError("positions must be a bootstrap-by-marker matrix")
    marker_count = pos.shape[1]
    if np.any((pos < 0) | (pos >= marker_count)):
        raise ValueError("positions contain an invalid marker rank")
    if any(np.unique(row).size != marker_count for row in pos):
        raise ValueError("each bootstrap row must be a permutation of marker ranks")
    ordered = np.sort(pos, axis=0)
    markers = np.arange(marker_count)
    if method == "central":
        tail = (1.0 - confidence) / 2.0
        low_index = int(np.floor(tail * pos.shape[0]))
        high_index = int(np.ceil((1.0 - tail) * pos.shape[0])) - 1
        left = ordered[low_index]
        right = ordered[high_index]
    elif method == "shortest":
        mass = max(1, int(np.ceil(confidence * pos.shape[0])))
        widths = ordered[mass - 1 :] - ordered[: pos.shape[0] - mass + 1]
        starts = np.argmin(widths, axis=0)
        left = ordered[starts, markers]
        right = ordered[starts + mass - 1, markers]
    else:
        raise ValueError("rank interval method must be 'central' or 'shortest'")
    return left.astype(np.int64), right.astype(np.int64)


def _weighted_vector_correlation(
    first: FloatArray,
    second: FloatArray,
    weights: FloatArray,
) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    first_mean = float(np.sum(weights * first) / total)
    second_mean = float(np.sum(weights * second) / total)
    first_centered = first - first_mean
    second_centered = second - second_mean
    covariance = float(np.sum(weights * first_centered * second_centered))
    denominator = np.sqrt(
        float(np.sum(weights * first_centered**2))
        * float(np.sum(weights * second_centered**2))
    )
    return covariance / denominator if denominator > 0.0 else 0.0


def _likelihood_mds_score_weights(lod: FloatArray) -> tuple[FloatArray, ...]:
    positive = np.maximum(np.asarray(lod, dtype=np.float64), 1e-12)
    observed = positive[lod > 0.0]
    if observed.size == 0:
        observed = positive
    weights: list[FloatArray] = [
        positive,
        positive**2,
        positive**0.25,
        np.sqrt(positive),
        positive**0.75,
        positive**1.5,
    ]
    for quantile in (0.75, 0.90, 0.95):
        cap = float(np.quantile(observed, quantile))
        weights.append(np.minimum(positive, cap) ** 2)
    return tuple(weights)


def _validate_likelihood_mds_configs(
    configs: Iterable[LikelihoodMDSConfig],
) -> tuple[LikelihoodMDSConfig, ...]:
    normalized: list[LikelihoodMDSConfig] = []
    for distance, lod_exponent, dimensions, curve_knots in configs:
        config = (
            str(distance),
            float(lod_exponent),
            int(dimensions),
            int(curve_knots),
        )
        if config[0] not in {"rf", "haldane", "kosambi"}:
            raise ValueError("candidate distance must be rf, haldane, or kosambi")
        if config[1] <= 0.0 or not np.isfinite(config[1]):
            raise ValueError("candidate LOD exponent must be positive and finite")
        if config[2] < 1 or config[3] < 0:
            raise ValueError(
                "candidate dimensions must be positive and curve complexity nonnegative"
            )
        if config not in normalized:
            normalized.append(config)
    if len(normalized) < 2:
        raise ValueError("at least two distinct candidate configurations are required")
    return tuple(normalized)


F2_LIKELIHOOD_MDS_CONFIGS: tuple[LikelihoodMDSConfig, ...] = (
    ("rf", 2.0, 20, 2),
    ("haldane", 2.0, 20, 2),
    ("kosambi", 1.0, 20, 2),
    ("kosambi", 2.0, 20, 2),
    ("kosambi", 3.0, 20, 2),
    ("kosambi", 2.0, 20, 3),
)


def fit_f2_likelihood_map(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    physical_positions: FloatArray | None = None,
    use_physical_scaffold: bool = False,
    candidate_configs: Iterable[LikelihoodMDSConfig] = F2_LIKELIHOOD_MDS_CONFIGS,
    stability_mass: float = 0.90,
    stability_rank_padding: int = 1,
    maximum_smacof_iterations: int = 200,
) -> F2MapResult:
    """Fit a complete-information F2 linkage map.

    Ordering is driven by the exact two-locus three-genotype F2 likelihood. When
    ``use_physical_scaffold`` is enabled, the final rank follows the supplied
    chromosome assembly while the de-novo likelihood order and its model-stability
    bands remain available for audit. This mirrors reference-guided curation used
    in high-quality published maps without feeding published genetic positions to
    the fitter.
    """

    p = _validate_f2_probabilities(probabilities)
    if not 0.0 < stability_mass < 1.0:
        raise ValueError("stability_mass must lie between zero and one")
    if stability_rank_padding < 0:
        raise ValueError("stability_rank_padding must be nonnegative")
    if maximum_smacof_iterations < 1:
        raise ValueError("maximum_smacof_iterations must be positive")
    names = (
        tuple(marker_names)
        if marker_names is not None
        else tuple(f"m{index + 1}" for index in range(p.shape[1]))
    )
    if len(names) != p.shape[1] or len(set(names)) != len(names):
        raise ValueError("marker_names must be unique and match the marker count")
    physical: FloatArray | None = None
    if physical_positions is not None:
        physical = np.asarray(physical_positions, dtype=np.float64)
        if physical.shape != (p.shape[1],) or not np.all(np.isfinite(physical)):
            raise ValueError(
                "physical_positions must be finite with one value per marker"
            )
    if use_physical_scaffold and physical is None:
        raise ValueError("use_physical_scaffold requires physical_positions")

    configs = _validate_likelihood_mds_configs(candidate_configs)
    recombination, lod = f2_pairwise_recombination_likelihood(p)
    orders = np.asarray(
        [
            likelihood_weighted_mds_order(
                recombination,
                lod,
                distance=distance,
                lod_exponent=lod_exponent,
                dimensions=dimensions,
                principal_curve_knots=curve_knots,
                maximum_smacof_iterations=maximum_smacof_iterations,
            )
            for distance, lod_exponent, dimensions, curve_knots in configs
        ],
        dtype=np.int64,
    )
    marker_count = p.shape[1]
    upper = np.triu_indices(marker_count, 1)
    pair_rf = recombination[upper]
    informative = lod[upper] > 0.0
    scores = np.full(len(configs), -np.inf, dtype=np.float64)
    for index, order in enumerate(orders):
        positions = np.empty(marker_count, dtype=np.int64)
        positions[order] = np.arange(marker_count)
        separation = np.abs(positions[upper[0]] - positions[upper[1]])
        if np.count_nonzero(informative) >= 3:
            scores[index] = abs(
                float(np.corrcoef(separation[informative], pair_rf[informative])[0, 1])
            )
    selected_index = int(np.argmax(scores))
    de_novo_order = orders[selected_index].copy()
    order = (
        np.argsort(physical, kind="stable").astype(np.int64)
        if use_physical_scaffold and physical is not None
        else de_novo_order.copy()
    )

    reference_positions = np.empty(marker_count, dtype=np.int64)
    reference_positions[order] = np.arange(marker_count)
    candidate_positions = np.empty_like(orders)
    for index, candidate in enumerate(orders):
        aligned = _align_to_reference(candidate, reference_positions)
        candidate_positions[index, aligned] = np.arange(marker_count)
    interval_left, interval_right = bootstrap_rank_intervals(
        candidate_positions,
        confidence=stability_mass,
        method="central",
    )
    if stability_rank_padding:
        interval_left = np.maximum(interval_left - stability_rank_padding, 0)
        interval_right = np.minimum(
            interval_right + stability_rank_padding, marker_count - 1
        )

    mean_genotype_certainty = float(np.mean(np.max(p, axis=2)))
    distances = _f2_genetic_map_distances(
        recombination,
        lod,
        order,
        mean_genotype_certainty=mean_genotype_certainty,
    )
    return F2MapResult(
        marker_names=names,
        order=order,
        de_novo_order=de_novo_order,
        candidate_orders=orders,
        candidate_positions=candidate_positions,
        interval_left=interval_left,
        interval_right=interval_right,
        recombination=recombination,
        lod=lod,
        genetic_distances=distances,
        selected_config=configs[selected_index],
        ordering_method=(
            "f2_likelihood_mds_physical_scaffold"
            if use_physical_scaffold
            else "f2_likelihood_mds"
        ),
        physical_scaffold_used=bool(use_physical_scaffold),
        mean_genotype_certainty=mean_genotype_certainty,
    )


def fit_likelihood_mds_ensemble(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    candidate_configs: Iterable[LikelihoodMDSConfig] = (DEFAULT_LIKELIHOOD_MDS_CONFIGS),
    selected_config: LikelihoodMDSConfig | None = None,
    stability_mass: float = 0.90,
    posterior_refinement_weight: float = 0.75,
    maximum_posterior_refinement_passes: int = 2,
    second_refinement_uncertain_pair_threshold: float = 0.03,
    stability_rank_padding: int = 1,
    minimum_stability_comparable_pair_fraction: float = 0.35,
    maximum_smacof_iterations: int = 500,
    penalized_curve_effective_degrees_of_freedom: float = 4.0,
    distance_rank_span_weight_exponent: float = (DISTANCE_RANK_SPAN_WEIGHT_EXPONENT),
) -> LikelihoodMDSEnsembleResult:
    """Fit SoftMap's robust likelihood-MDS ensemble.

    Candidate orders are scored by the global correlation between inferred rank
    separation and pairwise recombination fraction. The unweighted score supplies
    the robust default. It is vetoed only when all nine prespecified LOD-weighted
    objectives select the same curve complexity within the very same embedding
    family. The selected geometry is refit after interval-specific HMM smoothing,
    blended conservatively with the raw posteriors. A second refit is allowed only
    when the first-pass ensemble has enough weak pairwise votes to justify it.
    Rank bands summarize first-pass model sensitivity across every candidate and
    are deliberately labelled stability bands rather than nominal confidence
    bounds.
    """

    p = _validate_probabilities(probabilities)
    if not 0.0 < stability_mass < 1.0:
        raise ValueError("stability_mass must lie between zero and one")
    if (
        not np.isfinite(posterior_refinement_weight)
        or not 0.0 <= posterior_refinement_weight <= 1.0
    ):
        raise ValueError("posterior_refinement_weight must lie in [0, 1]")
    if (
        not isinstance(maximum_posterior_refinement_passes, (int, np.integer))
        or maximum_posterior_refinement_passes < 0
    ):
        raise ValueError(
            "maximum_posterior_refinement_passes must be a nonnegative integer"
        )
    if (
        not np.isfinite(second_refinement_uncertain_pair_threshold)
        or not 0.0 <= second_refinement_uncertain_pair_threshold <= 1.0
    ):
        raise ValueError(
            "second_refinement_uncertain_pair_threshold must lie in [0, 1]"
        )
    if (
        not isinstance(stability_rank_padding, (int, np.integer))
        or stability_rank_padding < 0
    ):
        raise ValueError("stability_rank_padding must be a nonnegative integer")
    if (
        not np.isfinite(minimum_stability_comparable_pair_fraction)
        or not 0.0 <= minimum_stability_comparable_pair_fraction <= 1.0
    ):
        raise ValueError(
            "minimum_stability_comparable_pair_fraction must lie in [0, 1]"
        )
    if maximum_smacof_iterations < 1:
        raise ValueError("maximum_smacof_iterations must be positive")
    if not np.isfinite(distance_rank_span_weight_exponent):
        raise ValueError("distance rank span weight exponent must be finite")
    if (
        not np.isfinite(penalized_curve_effective_degrees_of_freedom)
        or penalized_curve_effective_degrees_of_freedom <= 2.0
        or penalized_curve_effective_degrees_of_freedom > p.shape[1]
    ):
        raise ValueError(
            "penalized curve effective degrees of freedom must lie in (2, marker count]"
        )
    requested_configs = tuple(candidate_configs)
    normalized_selected: LikelihoodMDSConfig | None = None
    if selected_config is not None:
        normalized_selected = (
            str(selected_config[0]),
            float(selected_config[1]),
            int(selected_config[2]),
            int(selected_config[3]),
        )
        if normalized_selected not in requested_configs:
            requested_configs = (*requested_configs, normalized_selected)
    configs = _validate_likelihood_mds_configs(requested_configs)
    names = (
        tuple(marker_names)
        if marker_names is not None
        else tuple(f"m{index + 1}" for index in range(p.shape[1]))
    )
    if len(names) != p.shape[1]:
        raise ValueError("marker_names length must match the number of markers")
    if len(set(names)) != len(names):
        raise ValueError("marker_names must be unique")

    recombination, lod = pairwise_recombination_likelihood(p)
    embeddings: dict[tuple[str, float, int], FloatArray] = {}
    weight_systems: dict[float, tuple[FloatArray, FloatArray]] = {}
    candidate_orders: list[IntArray] = []
    for distance, lod_exponent, dimensions, curve_knots in configs:
        geometry = (distance, lod_exponent, dimensions)
        if geometry not in embeddings:
            if lod_exponent not in weight_systems:
                weights = lod**lod_exponent
                np.fill_diagonal(weights, 0.0)
                maximum_weight = float(np.max(weights))
                if maximum_weight <= 0.0:
                    raise ValueError(
                        "at least one marker pair must have positive linkage LOD"
                    )
                weights /= maximum_weight
                weight_laplacian = np.diag(np.sum(weights, axis=1)) - weights
                weight_systems[lod_exponent] = (
                    weights,
                    linalg.pinvh(weight_laplacian, check_finite=False),
                )
            embeddings[geometry] = _likelihood_weighted_mds_coordinates(
                recombination,
                lod,
                distance=distance,
                lod_exponent=lod_exponent,
                dimensions=dimensions,
                maximum_smacof_iterations=maximum_smacof_iterations,
                weight_system=weight_systems[lod_exponent],
            )
        candidate_orders.append(
            _smooth_penalized_principal_curve_order(
                embeddings[geometry],
                effective_degrees_of_freedom=(
                    penalized_curve_effective_degrees_of_freedom
                ),
            )
            if curve_knots == 0
            else _smooth_principal_curve_order(
                embeddings[geometry],
                interior_knots=curve_knots,
            )
        )
    orders = np.asarray(candidate_orders, dtype=np.int64)
    marker_count = p.shape[1]
    upper = np.triu_indices(marker_count, 1)
    pair_rf = recombination[upper]
    pair_lod = lod[upper]
    score_weights = _likelihood_mds_score_weights(pair_lod)
    uniform_scores = np.empty(len(configs), dtype=np.float64)
    weighted_scores = np.empty((len(score_weights), len(configs)), dtype=np.float64)
    for candidate_index, order in enumerate(orders):
        positions = np.empty(marker_count, dtype=np.int64)
        positions[order] = np.arange(marker_count)
        separation = np.abs(positions[upper[0]] - positions[upper[1]]) / max(
            marker_count - 1, 1
        )
        uniform_scores[candidate_index] = float(np.corrcoef(separation, pair_rf)[0, 1])
        for weight_index, weights in enumerate(score_weights):
            weighted_scores[weight_index, candidate_index] = (
                _weighted_vector_correlation(separation, pair_rf, weights)
            )
    uniform_index = int(np.argmax(uniform_scores))
    weighted_indices = np.argmax(weighted_scores, axis=1).astype(np.int64)
    unanimous_index = int(weighted_indices[0])
    unanimous = bool(np.all(weighted_indices == unanimous_index))
    uniform_config = configs[uniform_index]
    unanimous_config = configs[unanimous_index]
    veto = bool(
        unanimous
        and unanimous_index != uniform_index
        and unanimous_config[:3] == uniform_config[:3]
    )
    if normalized_selected is None:
        selected_index = unanimous_index if veto else uniform_index
        selection_method = "global_rf_correlation_with_veto"
    else:
        selected_index = configs.index(normalized_selected)
        veto = False
        selection_method = "fixed_high_information_geometry"

    reference_positions = np.empty(marker_count, dtype=np.int64)
    reference_positions[orders[0]] = np.arange(marker_count)
    candidate_positions = np.empty_like(orders)
    for index, order in enumerate(orders):
        aligned = _align_to_reference(order, reference_positions)
        candidate_positions[index, aligned] = np.arange(marker_count)
    left, right = bootstrap_rank_intervals(
        candidate_positions,
        confidence=stability_mass,
        method="central",
    )
    if stability_rank_padding:
        left = np.maximum(left - stability_rank_padding, 0)
        right = np.minimum(right + stability_rank_padding, marker_count - 1)
    pair_comparable = (right[upper[0]] < left[upper[1]]) | (
        right[upper[1]] < left[upper[0]]
    )
    stability_support_filter_applied = False
    stability_positions = candidate_positions
    if np.mean(pair_comparable) < minimum_stability_comparable_pair_fraction:
        supported_indices = np.unique(
            np.concatenate(
                (
                    np.asarray([uniform_index, selected_index], dtype=np.int64),
                    weighted_indices,
                )
            )
        )
        if 1 < supported_indices.size < len(configs):
            supported_positions = candidate_positions[supported_indices]
            supported_left, supported_right = bootstrap_rank_intervals(
                supported_positions,
                confidence=stability_mass,
                method="central",
            )
            if stability_rank_padding:
                supported_left = np.maximum(
                    supported_left - stability_rank_padding,
                    0,
                )
                supported_right = np.minimum(
                    supported_right + stability_rank_padding,
                    marker_count - 1,
                )
            supported_comparable = (
                supported_right[upper[0]] < supported_left[upper[1]]
            ) | (supported_right[upper[1]] < supported_left[upper[0]])
            if (
                np.mean(supported_comparable)
                >= minimum_stability_comparable_pair_fraction
            ):
                left = supported_left
                right = supported_right
                pair_comparable = supported_comparable
                stability_positions = supported_positions
                stability_support_filter_applied = True
    if np.mean(pair_comparable) < minimum_stability_comparable_pair_fraction:
        left = np.zeros(marker_count, dtype=np.int64)
        right = np.full(marker_count, marker_count - 1, dtype=np.int64)
        pair_comparable = np.zeros(upper[0].size, dtype=bool)
    rank_sd = np.std(stability_positions, axis=0)
    preference_fraction = np.mean(
        candidate_positions[:, :, None] < candidate_positions[:, None, :],
        axis=0,
    )
    vote_margin = np.abs(2.0 * preference_fraction[upper] - 1.0)
    uncertain_pair_fraction_75 = float(np.mean(vote_margin < 0.5))

    preliminary_order = orders[selected_index].copy()
    final_order = preliminary_order.copy()
    refinement_passes = 0
    for pass_index in range(maximum_posterior_refinement_passes):
        if posterior_refinement_weight == 0.0:
            break
        if (
            pass_index > 0
            and uncertain_pair_fraction_75 < second_refinement_uncertain_pair_threshold
        ):
            break
        transition = recombination[final_order[:-1], final_order[1:]]
        smoothed = _hmm_smooth_with_interval_transitions(
            p,
            final_order,
            transition,
        )
        refined_probabilities = (
            1.0 - posterior_refinement_weight
        ) * p + posterior_refinement_weight * smoothed
        refined_recombination, refined_lod = pairwise_recombination_likelihood(
            refined_probabilities
        )
        distance, lod_exponent, dimensions, curve_knots = configs[selected_index]
        refined_order = likelihood_weighted_mds_order(
            refined_recombination,
            refined_lod,
            distance=distance,
            lod_exponent=lod_exponent,
            dimensions=dimensions,
            principal_curve_knots=curve_knots,
            maximum_smacof_iterations=maximum_smacof_iterations,
        )
        current_positions = np.empty(marker_count, dtype=np.int64)
        current_positions[final_order] = np.arange(marker_count)
        final_order = _align_to_reference(refined_order, current_positions)
        refinement_passes += 1
    reported_positions = np.empty(marker_count, dtype=np.int64)
    reported_positions[final_order] = np.arange(marker_count)
    identity = np.arange(marker_count, dtype=np.int64)
    distance_left, distance_right = np.triu_indices(marker_count, 1)
    adjacent_pairwise = recombination[final_order[:-1], final_order[1:]]
    adjacent_pairwise = np.minimum(adjacent_pairwise, 0.499999)
    adjacent_multipoint = multipoint_adjacent_recombination(
        p,
        final_order,
        adjacent_pairwise,
    )
    genetic_distances = _fit_genetic_map_distances_from_pairs(
        marker_count,
        final_order,
        distance_left.astype(np.int64),
        distance_right.astype(np.int64),
        recombination[final_order[distance_left], final_order[distance_right]],
        lod[final_order[distance_left], final_order[distance_right]],
        marker_bin_membership=None,
        adjacent_pairwise_recombination=adjacent_pairwise,
        adjacent_multipoint_recombination=adjacent_multipoint,
        segment_count=16,
        minimum_pair_recombination=0.02,
        maximum_pair_recombination=0.445,
        rank_span_weight_exponent=distance_rank_span_weight_exponent,
    )
    return LikelihoodMDSEnsembleResult(
        marker_names=names,
        order=final_order,
        preliminary_order=preliminary_order,
        candidate_orders=orders,
        candidate_positions=candidate_positions,
        interval_left=left,
        interval_right=right,
        candidate_configs=configs,
        selected_candidate_index=selected_index,
        uniform_candidate_index=uniform_index,
        weighted_candidate_indices=weighted_indices,
        uniform_scores=uniform_scores,
        weighted_scores=weighted_scores,
        unanimous_family_veto_triggered=veto,
        posterior_refinement_weight=float(posterior_refinement_weight),
        posterior_refinement_passes_applied=refinement_passes,
        second_refinement_uncertain_pair_threshold=float(
            second_refinement_uncertain_pair_threshold
        ),
        stability_rank_padding=int(stability_rank_padding),
        minimum_stability_comparable_pair_fraction=float(
            minimum_stability_comparable_pair_fraction
        ),
        stability_mass=stability_mass,
        stability_comparable_pair_fraction=float(np.mean(pair_comparable)),
        mean_normalized_rank_sd=float(np.mean(rank_sd) / max(marker_count - 1, 1)),
        mean_pairwise_vote_margin=float(np.mean(vote_margin)),
        uncertain_pair_fraction_75=uncertain_pair_fraction_75,
        mean_genotype_certainty=float(np.mean(np.abs(2.0 * p - 1.0))),
        bin_membership=identity,
        bin_representatives=identity,
        reported_positions=reported_positions,
        binning_method="none",
        maximum_bin_recombination=None,
        minimum_bin_linkage_lod=None,
        maximum_bin_pool_evidence=None,
        bin_neighbor_count=None,
        bin_neighbor_projection_dimensions=None,
        selection_method=selection_method,
        ordering_method="dense_likelihood_mds",
        landmark_count=None,
        landmark_neighbor_count=None,
        landmark_support_exponent=None,
        large_scale_rescue_triggered=False,
        low_certainty_stability_mass_cap_applied=False,
        genetic_distances=genetic_distances,
        weighted_objective_support_filter_applied=(stability_support_filter_applied),
        penalized_curve_effective_degrees_of_freedom=(
            float(penalized_curve_effective_degrees_of_freedom)
            if configs[selected_index][3] == 0
            else None
        ),
    )


def _interval_comparable_pair_fraction(
    left: IntArray,
    right: IntArray,
) -> float:
    """Fraction of unordered interval pairs whose rank ranges do not overlap."""

    lower = np.asarray(left, dtype=np.int64)
    upper = np.asarray(right, dtype=np.int64)
    if lower.ndim != 1 or upper.shape != lower.shape or lower.size < 2:
        raise ValueError("interval bounds must be matching vectors of size two or more")
    if np.any(lower > upper):
        raise ValueError("interval lower bounds cannot exceed upper bounds")
    sorted_lower = np.sort(lower)
    comparable = int(
        np.sum(lower.size - np.searchsorted(sorted_lower, upper, side="right"))
    )
    return comparable / (lower.size * (lower.size - 1) / 2.0)


def _large_scale_rescue_config(
    stability_comparable_pair_fraction: float,
    mean_genotype_certainty: float,
    uncertain_pair_fraction: float,
) -> LikelihoodMDSConfig | None:
    """Choose the frozen low-stability curve rescue without using map truth."""

    if stability_comparable_pair_fraction < LARGE_SCALE_RESCUE_STABILITY_FLOOR:
        if mean_genotype_certainty >= LARGE_SCALE_RESCUE_CERTAINTY_THRESHOLD:
            return LARGE_SCALE_MODERATE_CERTAINTY_RESCUE_CONFIG
        if stability_comparable_pair_fraction < LARGE_SCALE_SEVERE_INSTABILITY_FLOOR:
            return LARGE_SCALE_LOW_CERTAINTY_RESCUE_CONFIG
        return LARGE_SCALE_LOW_CERTAINTY_MODERATE_RESCUE_CONFIG
    if (
        mean_genotype_certainty < LARGE_SCALE_RESCUE_CERTAINTY_THRESHOLD
        and uncertain_pair_fraction >= LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_PAIR_TRIGGER
    ):
        return LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_RESCUE_CONFIG
    return None


def _large_scale_stability_mass(
    requested_mass: float,
    support_weighting_active: bool,
    mean_genotype_certainty: float,
) -> tuple[float, bool]:
    """Apply the frozen low-certainty stability-mass cap when applicable."""

    apply_cap = (
        support_weighting_active
        and mean_genotype_certainty < LARGE_SCALE_RESCUE_CERTAINTY_THRESHOLD
        and requested_mass > LARGE_SCALE_LOW_CERTAINTY_STABILITY_MASS_CAP
    )
    return (
        LARGE_SCALE_LOW_CERTAINTY_STABILITY_MASS_CAP if apply_cap else requested_mass,
        apply_cap,
    )


def _select_likelihood_landmarks(
    probabilities: FloatArray,
    maximum_landmarks: int,
    support: FloatArray | None = None,
    support_exponent: float = 0.5,
) -> IntArray:
    """Select deterministic farthest landmarks, optionally weighted by support."""

    p = _validate_probabilities(probabilities)
    if maximum_landmarks < 3:
        raise ValueError("maximum_landmarks must be at least three")
    marker_count = p.shape[1]
    if marker_count <= maximum_landmarks:
        return np.arange(marker_count, dtype=np.int64)
    if not np.isfinite(support_exponent) or support_exponent < 0.0:
        raise ValueError("landmark support exponent must be finite and nonnegative")
    if support is None:
        selection_weight = np.ones(marker_count, dtype=np.float64)
    else:
        raw_support = np.asarray(support, dtype=np.float64)
        if (
            raw_support.shape != (marker_count,)
            or not np.all(np.isfinite(raw_support))
            or np.any(raw_support <= 0.0)
        ):
            raise ValueError(
                "landmark support must be a positive finite value per marker"
            )
        selection_weight = raw_support**support_exponent
    features = (p.T - 0.5) * 2.0
    information = np.mean(np.abs(features), axis=1)
    selected = np.empty(maximum_landmarks, dtype=np.int64)
    selected[0] = int(np.argmax(information * selection_weight))
    chosen = np.zeros(marker_count, dtype=bool)
    chosen[selected[0]] = True
    delta = features - features[selected[0]]
    minimum_squared_distance = np.einsum("ij,ij->i", delta, delta)
    for position in range(1, maximum_landmarks):
        score = minimum_squared_distance * (0.5 + information) * selection_weight
        score[chosen] = -np.inf
        marker = int(np.argmax(score))
        selected[position] = marker
        chosen[marker] = True
        delta = features - features[marker]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance,
            squared_distance,
        )
    return selected


def _sampled_candidate_vote_diagnostics(
    candidate_positions: IntArray,
    maximum_pairs: int = 1_000_000,
) -> tuple[float, float]:
    """Return deterministic candidate-vote diagnostics with bounded memory."""

    positions = np.asarray(candidate_positions, dtype=np.int64)
    marker_count = positions.shape[1]
    pair_count = marker_count * (marker_count - 1) // 2
    if pair_count <= maximum_pairs:
        left, right = np.triu_indices(marker_count, 1)
    else:
        rng = np.random.default_rng(0)
        left = rng.integers(0, marker_count, size=maximum_pairs)
        right = rng.integers(0, marker_count, size=maximum_pairs)
        distinct = left != right
        low = np.minimum(left[distinct], right[distinct])
        right = np.maximum(left[distinct], right[distinct])
        left = low
    preference = np.mean(
        positions[:, left] < positions[:, right],
        axis=0,
    )
    margin = np.abs(2.0 * preference - 1.0)
    return float(np.mean(margin)), float(np.mean(margin < 0.5))


def _fit_landmark_likelihood_mds_ensemble(
    probabilities: FloatArray,
    marker_names: tuple[str, ...],
    *,
    candidate_configs: tuple[LikelihoodMDSConfig, ...],
    selected_config: LikelihoodMDSConfig | None,
    maximum_landmarks: int,
    landmark_support: FloatArray | None,
    landmark_support_exponent: float,
    landmark_neighbor_count: int,
    landmark_lod_exponent: float,
    stability_mass: float,
    posterior_refinement_weight: float,
    second_refinement_uncertain_pair_threshold: float,
    stability_rank_padding: int,
    minimum_stability_comparable_pair_fraction: float,
    maximum_smacof_iterations: int,
) -> LikelihoodMDSEnsembleResult:
    """Extend a dense landmark ensemble to all likelihood-bin representatives."""

    p = _validate_probabilities(probabilities)
    if landmark_neighbor_count < 2:
        raise ValueError("landmark_neighbor_count must be at least two")
    if landmark_lod_exponent <= 0.0 or not np.isfinite(landmark_lod_exponent):
        raise ValueError("landmark_lod_exponent must be positive and finite")
    landmarks = _select_likelihood_landmarks(
        p,
        maximum_landmarks,
        support=landmark_support,
        support_exponent=landmark_support_exponent,
    )
    landmark_support = fit_likelihood_mds_ensemble(
        p[:, landmarks],
        tuple(marker_names[int(index)] for index in landmarks),
        candidate_configs=candidate_configs,
        stability_mass=stability_mass,
        posterior_refinement_weight=posterior_refinement_weight,
        maximum_posterior_refinement_passes=0,
        second_refinement_uncertain_pair_threshold=(
            second_refinement_uncertain_pair_threshold
        ),
        stability_rank_padding=stability_rank_padding,
        minimum_stability_comparable_pair_fraction=(
            minimum_stability_comparable_pair_fraction
        ),
        maximum_smacof_iterations=maximum_smacof_iterations,
    )
    marker_count = p.shape[1]
    landmark_mask = np.zeros(marker_count, dtype=bool)
    landmark_mask[landmarks] = True
    remaining = np.flatnonzero(~landmark_mask).astype(np.int64)
    neighbor_count = min(landmark_neighbor_count, landmarks.size)
    landmark_features = (p[:, landmarks].T - 0.5) * 2.0
    remaining_features = (p[:, remaining].T - 0.5) * 2.0
    tree = cKDTree(landmark_features)
    feature_distance, neighbor = tree.query(
        remaining_features,
        k=neighbor_count,
        workers=1,
    )
    neighbor = np.asarray(neighbor, dtype=np.int64)
    feature_distance = np.asarray(feature_distance, dtype=np.float64)
    if neighbor.ndim == 1:
        neighbor = neighbor[:, None]
        feature_distance = feature_distance[:, None]
    edge_left = np.repeat(remaining, neighbor.shape[1])
    edge_right = landmarks[neighbor.reshape(-1)]
    _, edge_lod = pairwise_recombination_likelihood_edges(
        p,
        edge_left,
        edge_right,
    )
    weights = edge_lod.reshape(neighbor.shape) ** landmark_lod_exponent
    row_total = np.sum(weights, axis=1)
    unsupported = row_total <= 1e-30
    if np.any(unsupported):
        fallback = 1.0 / np.maximum(feature_distance[unsupported], 1e-9)
        weights[unsupported] = fallback
        row_total[unsupported] = np.sum(fallback, axis=1)
    weights /= row_total[:, None]

    candidate_count = len(candidate_configs)
    coordinates = np.empty((candidate_count, marker_count), dtype=np.float64)
    coordinates[:, landmarks] = landmark_support.candidate_positions
    neighbor_positions = landmark_support.candidate_positions[:, neighbor]
    coordinates[:, remaining] = np.einsum(
        "nk,cnk->cn",
        weights,
        neighbor_positions,
    )
    raw_orders = np.argsort(coordinates, axis=1, kind="stable").astype(np.int64)
    reference_positions = np.empty(marker_count, dtype=np.int64)
    reference_positions[raw_orders[0]] = np.arange(marker_count)
    candidate_positions = np.empty_like(raw_orders)
    candidate_orders = np.empty_like(raw_orders)
    for index, raw_order in enumerate(raw_orders):
        order = _align_to_reference(raw_order, reference_positions)
        candidate_orders[index] = order
        candidate_positions[index, order] = np.arange(marker_count)

    selected_index = (
        candidate_configs.index(selected_config)
        if selected_config is not None
        else landmark_support.selected_candidate_index
    )
    selected_order = candidate_orders[selected_index]
    left, right = bootstrap_rank_intervals(
        candidate_positions,
        confidence=stability_mass,
        method="central",
    )
    if stability_rank_padding:
        left = np.maximum(left - stability_rank_padding, 0)
        right = np.minimum(right + stability_rank_padding, marker_count - 1)
    comparable_fraction = _interval_comparable_pair_fraction(left, right)
    if comparable_fraction < minimum_stability_comparable_pair_fraction:
        left = np.zeros(marker_count, dtype=np.int64)
        right = np.full(marker_count, marker_count - 1, dtype=np.int64)
        comparable_fraction = 0.0
    mean_vote_margin, uncertain_fraction = _sampled_candidate_vote_diagnostics(
        candidate_positions
    )
    rank_sd = np.std(candidate_positions, axis=0)
    reported_positions = np.empty(marker_count, dtype=np.int64)
    reported_positions[selected_order] = np.arange(marker_count)
    identity = np.arange(marker_count, dtype=np.int64)
    return LikelihoodMDSEnsembleResult(
        marker_names=marker_names,
        order=selected_order,
        preliminary_order=selected_order.copy(),
        candidate_orders=candidate_orders,
        candidate_positions=candidate_positions,
        interval_left=left,
        interval_right=right,
        candidate_configs=candidate_configs,
        selected_candidate_index=selected_index,
        uniform_candidate_index=landmark_support.uniform_candidate_index,
        weighted_candidate_indices=landmark_support.weighted_candidate_indices,
        uniform_scores=landmark_support.uniform_scores,
        weighted_scores=landmark_support.weighted_scores,
        unanimous_family_veto_triggered=(
            landmark_support.unanimous_family_veto_triggered
        ),
        posterior_refinement_weight=float(posterior_refinement_weight),
        posterior_refinement_passes_applied=0,
        second_refinement_uncertain_pair_threshold=float(
            second_refinement_uncertain_pair_threshold
        ),
        stability_rank_padding=int(stability_rank_padding),
        minimum_stability_comparable_pair_fraction=float(
            minimum_stability_comparable_pair_fraction
        ),
        stability_mass=float(stability_mass),
        stability_comparable_pair_fraction=float(comparable_fraction),
        mean_normalized_rank_sd=float(np.mean(rank_sd) / max(marker_count - 1, 1)),
        mean_pairwise_vote_margin=mean_vote_margin,
        uncertain_pair_fraction_75=uncertain_fraction,
        mean_genotype_certainty=float(np.mean(np.abs(2.0 * p - 1.0))),
        bin_membership=identity,
        bin_representatives=identity,
        reported_positions=reported_positions,
        binning_method="none",
        maximum_bin_recombination=None,
        minimum_bin_linkage_lod=None,
        maximum_bin_pool_evidence=None,
        bin_neighbor_count=None,
        bin_neighbor_projection_dimensions=None,
        selection_method=(
            "support_weighted_landmark_likelihood_extension"
            if landmark_support is not None
            else "landmark_likelihood_extension"
        ),
        ordering_method="landmark_likelihood_mds",
        landmark_count=int(landmarks.size),
        landmark_neighbor_count=int(neighbor_count),
        landmark_support_exponent=(
            float(landmark_support_exponent) if landmark_support is not None else None
        ),
        large_scale_rescue_triggered=False,
        low_certainty_stability_mass_cap_applied=False,
        genetic_distances=None,
    )


def _fit_scalable_likelihood_mds_ensemble_once(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    maximum_dense_markers: int = 500,
    maximum_bin_recombination: float = 0.0,
    minimum_bin_linkage_lod: float | None = None,
    bin_neighbor_count: int | None = None,
    bin_neighbor_projection_dimensions: int | None = None,
    bin_neighbor_projection_minimum_markers: int = 50_000,
    large_scale_minimum_markers: int = 50_000,
    maximum_bin_pool_evidence: float | None = None,
    maximum_dense_likelihood_bins: int = 1_000,
    maximum_likelihood_landmarks: int = 256,
    landmark_support_exponent: float = 0.5,
    landmark_support_weighting_minimum_markers: int = 50_000,
    landmark_neighbor_count: int = 32,
    landmark_lod_exponent: float = 4.0,
    landmark_selected_config: LikelihoodMDSConfig | None = (
        LANDMARK_LIKELIHOOD_MDS_CONFIG
    ),
    scalable_selected_config: LikelihoodMDSConfig | None = (
        SCALABLE_LIKELIHOOD_MDS_CONFIG
    ),
    candidate_configs: Iterable[LikelihoodMDSConfig] = (DEFAULT_LIKELIHOOD_MDS_CONFIGS),
    stability_mass: float = 0.90,
    posterior_refinement_weight: float = 0.75,
    maximum_posterior_refinement_passes: int = 2,
    second_refinement_uncertain_pair_threshold: float = 0.03,
    stability_rank_padding: int = 1,
    minimum_stability_comparable_pair_fraction: float = 0.35,
    maximum_smacof_iterations: int = 500,
    dense_selected_config: LikelihoodMDSConfig | None = None,
    dense_selection_method: str | None = None,
    dense_penalized_curve_effective_degrees_of_freedom: float = 4.0,
    dense_distance_rank_span_weight_exponent: float = (
        DISTANCE_RANK_SPAN_WEIGHT_EXPONENT
    ),
) -> LikelihoodMDSEnsembleResult:
    """Fit dense LMDS directly or a likelihood-binned scalable partial order."""

    p = _validate_probabilities(probabilities)
    if maximum_dense_markers < 3:
        raise ValueError("maximum_dense_markers must be at least three")
    if maximum_dense_likelihood_bins < 3:
        raise ValueError("maximum_dense_likelihood_bins must be at least three")
    if maximum_likelihood_landmarks < 10:
        raise ValueError("maximum_likelihood_landmarks must be at least ten")
    if large_scale_minimum_markers < 3:
        raise ValueError("large-scale marker threshold must be at least three")
    if not np.isfinite(landmark_support_exponent) or landmark_support_exponent < 0.0:
        raise ValueError("landmark support exponent must be finite and nonnegative")
    if landmark_support_weighting_minimum_markers < 3:
        raise ValueError(
            "landmark support-weighting marker threshold must be at least three"
        )
    names = (
        tuple(marker_names)
        if marker_names is not None
        else tuple(f"m{index + 1}" for index in range(p.shape[1]))
    )
    if len(names) != p.shape[1] or len(set(names)) != len(names):
        raise ValueError("marker_names must be unique and match the marker count")
    large_scale = p.shape[1] >= large_scale_minimum_markers
    resolved_minimum_bin_linkage_lod = (
        (3.0 if large_scale else 9.0)
        if minimum_bin_linkage_lod is None
        else float(minimum_bin_linkage_lod)
    )
    resolved_bin_neighbor_count = (
        (64 if large_scale else 32)
        if bin_neighbor_count is None
        else int(bin_neighbor_count)
    )
    resolved_bin_neighbor_projection_dimensions = (
        (12 if large_scale else 16)
        if bin_neighbor_projection_dimensions is None
        else int(bin_neighbor_projection_dimensions)
    )
    normalized_candidate_configs = tuple(candidate_configs)
    shared_arguments = {
        "candidate_configs": normalized_candidate_configs,
        "stability_mass": stability_mass,
        "posterior_refinement_weight": posterior_refinement_weight,
        "maximum_posterior_refinement_passes": (maximum_posterior_refinement_passes),
        "second_refinement_uncertain_pair_threshold": (
            second_refinement_uncertain_pair_threshold
        ),
        "stability_rank_padding": stability_rank_padding,
        "minimum_stability_comparable_pair_fraction": (
            minimum_stability_comparable_pair_fraction
        ),
        "maximum_smacof_iterations": maximum_smacof_iterations,
        "penalized_curve_effective_degrees_of_freedom": (
            dense_penalized_curve_effective_degrees_of_freedom
        ),
        "distance_rank_span_weight_exponent": (
            dense_distance_rank_span_weight_exponent
        ),
    }
    if p.shape[1] <= maximum_dense_markers:
        dense_arguments = dict(shared_arguments)
        mean_genotype_certainty = float(np.mean(np.abs(2.0 * p - 1.0)))
        selected_config = dense_selected_config
        selection_method = dense_selection_method
        if selected_config is None and (
            mean_genotype_certainty >= DENSE_HIGH_INFORMATION_CERTAINTY_THRESHOLD
        ):
            selected_config = DENSE_HIGH_INFORMATION_CONFIG
            selection_method = "fixed_high_information_geometry"
        elif selected_config is None and (
            mean_genotype_certainty >= DENSE_MODERATE_INFORMATION_CERTAINTY_THRESHOLD
        ):
            selected_config = DENSE_MODERATE_INFORMATION_CONFIG
            selection_method = "fixed_moderate_information_penalized_curve"
            dense_arguments["penalized_curve_effective_degrees_of_freedom"] = (
                DENSE_MODERATE_INFORMATION_PENALIZED_CURVE_EDF
            )
        if selected_config is not None:
            dense_arguments["selected_config"] = selected_config
            dense_arguments["maximum_posterior_refinement_passes"] = 0
        result = fit_likelihood_mds_ensemble(
            p,
            names,
            **dense_arguments,
        )
        if selection_method is not None:
            result = replace(result, selection_method=selection_method)
        return result

    bins = likelihood_bin_markers(
        p,
        maximum_bin_recombination=maximum_bin_recombination,
        minimum_linkage_lod=resolved_minimum_bin_linkage_lod,
        neighbor_count=resolved_bin_neighbor_count,
        neighbor_projection_dimensions=(resolved_bin_neighbor_projection_dimensions),
        neighbor_projection_minimum_markers=(bin_neighbor_projection_minimum_markers),
        maximum_pool_evidence=maximum_bin_pool_evidence,
    )
    if bins.representatives.size < 3:
        raise ValueError(
            "fewer than three likelihood bins remain; order information is insufficient"
        )
    bin_names = tuple(names[int(index)] for index in bins.representatives)
    scalable_arguments = dict(shared_arguments)
    normalized_selected: LikelihoodMDSConfig | None = None
    if scalable_selected_config is not None:
        normalized_selected = (
            str(scalable_selected_config[0]),
            float(scalable_selected_config[1]),
            int(scalable_selected_config[2]),
            int(scalable_selected_config[3]),
        )
        scalable_configs = normalized_candidate_configs
        if normalized_selected not in scalable_configs:
            scalable_configs = (*scalable_configs, normalized_selected)
        scalable_arguments["candidate_configs"] = scalable_configs
        scalable_arguments["maximum_posterior_refinement_passes"] = 0
    normalized_landmark_selected: LikelihoodMDSConfig | None = None
    if landmark_selected_config is not None:
        normalized_landmark_selected = (
            str(landmark_selected_config[0]),
            float(landmark_selected_config[1]),
            int(landmark_selected_config[2]),
            int(landmark_selected_config[3]),
        )
        if normalized_landmark_selected not in scalable_arguments["candidate_configs"]:
            scalable_arguments["candidate_configs"] = (
                *scalable_arguments["candidate_configs"],
                normalized_landmark_selected,
            )
    using_landmarks = bins.representatives.size > maximum_dense_likelihood_bins
    using_support_weighted_landmarks = (
        using_landmarks and p.shape[1] >= landmark_support_weighting_minimum_markers
    )
    mean_genotype_certainty = float(np.mean(np.abs(2.0 * p - 1.0)))
    effective_stability_mass, stability_mass_cap_applied = _large_scale_stability_mass(
        stability_mass,
        using_support_weighted_landmarks,
        mean_genotype_certainty,
    )
    if using_support_weighted_landmarks:
        for rescue_config in (
            LARGE_SCALE_LOW_CERTAINTY_RESCUE_CONFIG,
            LARGE_SCALE_LOW_CERTAINTY_MODERATE_RESCUE_CONFIG,
            LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_RESCUE_CONFIG,
            LARGE_SCALE_MODERATE_CERTAINTY_RESCUE_CONFIG,
        ):
            if rescue_config not in scalable_arguments["candidate_configs"]:
                scalable_arguments["candidate_configs"] = (
                    *scalable_arguments["candidate_configs"],
                    rescue_config,
                )
    if not using_landmarks:
        support = fit_likelihood_mds_ensemble(
            bins.probabilities,
            bin_names,
            **scalable_arguments,
        )
    else:
        support = _fit_landmark_likelihood_mds_ensemble(
            bins.probabilities,
            bin_names,
            candidate_configs=tuple(scalable_arguments["candidate_configs"]),
            selected_config=normalized_landmark_selected,
            maximum_landmarks=maximum_likelihood_landmarks,
            landmark_support=(
                np.bincount(
                    bins.membership,
                    minlength=bins.representatives.size,
                ).astype(np.float64)
                if using_support_weighted_landmarks
                else None
            ),
            landmark_support_exponent=landmark_support_exponent,
            landmark_neighbor_count=landmark_neighbor_count,
            landmark_lod_exponent=landmark_lod_exponent,
            stability_mass=stability_mass,
            posterior_refinement_weight=posterior_refinement_weight,
            second_refinement_uncertain_pair_threshold=(
                second_refinement_uncertain_pair_threshold
            ),
            stability_rank_padding=stability_rank_padding,
            minimum_stability_comparable_pair_fraction=(
                minimum_stability_comparable_pair_fraction
            ),
            maximum_smacof_iterations=maximum_smacof_iterations,
        )
    active_selected = (
        normalized_landmark_selected if using_landmarks else normalized_selected
    )
    rescue_config: LikelihoodMDSConfig | None = None
    if (
        using_support_weighted_landmarks
        and active_selected == LANDMARK_LIKELIHOOD_MDS_CONFIG
    ):
        rescue_config = _large_scale_rescue_config(
            support.stability_comparable_pair_fraction,
            mean_genotype_certainty,
            support.uncertain_pair_fraction_75,
        )
        if rescue_config is not None:
            active_selected = rescue_config
    if active_selected is None:
        selected_index = support.selected_candidate_index
        selected_bin_order = support.order
        preliminary_bin_order = support.preliminary_order
        selection_method = support.selection_method
        refinement_passes = support.posterior_refinement_passes_applied
        veto = support.unanimous_family_veto_triggered
    else:
        selected_index = support.candidate_configs.index(active_selected)
        selected_bin_order = support.candidate_orders[selected_index]
        preliminary_bin_order = selected_bin_order
        selection_method = (
            "stability_triggered_low_certainty_rf_curve"
            if rescue_config == LARGE_SCALE_LOW_CERTAINTY_RESCUE_CONFIG
            else "stability_triggered_low_certainty_haldane_curve"
            if rescue_config == LARGE_SCALE_LOW_CERTAINTY_MODERATE_RESCUE_CONFIG
            else "uncertain_vote_triggered_low_certainty_rf_curve"
            if rescue_config == LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_RESCUE_CONFIG
            else "stability_triggered_moderate_certainty_haldane_curve"
            if rescue_config == LARGE_SCALE_MODERATE_CERTAINTY_RESCUE_CONFIG
            else "fixed_support_weighted_landmark_geometry"
            if using_support_weighted_landmarks
            else "fixed_landmark_geometry"
            if using_landmarks
            else "fixed_high_information_geometry"
        )
        refinement_passes = 0
        veto = False
    members = tuple(
        np.flatnonzero(bins.membership == group).astype(np.int64)
        for group in range(bins.representatives.size)
    )

    def expand_order(bin_order: IntArray) -> IntArray:
        return np.concatenate(
            [members[int(group)] for group in np.asarray(bin_order)]
        ).astype(np.int64, copy=False)

    final_order = expand_order(selected_bin_order)
    preliminary_order = expand_order(preliminary_bin_order)
    candidate_orders = np.asarray(
        [expand_order(order) for order in support.candidate_orders], dtype=np.int64
    )
    candidate_positions = np.empty_like(candidate_orders)
    for index, positions in enumerate(support.candidate_positions):
        aligned_bin_order = np.argsort(positions, kind="stable")
        aligned_order = expand_order(aligned_bin_order)
        candidate_positions[index, aligned_order] = np.arange(p.shape[1])

    left, right = bootstrap_rank_intervals(
        candidate_positions,
        confidence=effective_stability_mass,
        method="central",
    )
    if stability_rank_padding:
        left = np.maximum(left - stability_rank_padding, 0)
        right = np.minimum(
            right + stability_rank_padding,
            p.shape[1] - 1,
        )
    bin_rank = np.empty(bins.representatives.size, dtype=np.int64)
    bin_rank[selected_bin_order] = np.arange(bins.representatives.size)
    ordered_sizes = np.asarray(
        [members[int(group)].size for group in selected_bin_order], dtype=np.int64
    )
    group_start = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(ordered_sizes[:-1], dtype=np.int64),
        )
    )
    group_end = group_start + ordered_sizes - 1
    marker_group_rank = bin_rank[bins.membership]
    left = np.minimum(left, group_start[marker_group_rank])
    right = np.maximum(right, group_end[marker_group_rank])
    comparable_fraction = _interval_comparable_pair_fraction(left, right)
    if (
        support.stability_comparable_pair_fraction == 0.0
        or comparable_fraction < minimum_stability_comparable_pair_fraction
    ):
        left = np.zeros(p.shape[1], dtype=np.int64)
        right = np.full(p.shape[1], p.shape[1] - 1, dtype=np.int64)
        comparable_fraction = 0.0
    reported_positions = marker_group_rank
    rank_sd = np.std(candidate_positions, axis=0)
    genetic_distances = estimate_genetic_map_distances(
        bins.probabilities,
        selected_bin_order,
        marker_bin_membership=bins.membership,
    )
    return LikelihoodMDSEnsembleResult(
        marker_names=names,
        order=final_order,
        preliminary_order=preliminary_order,
        candidate_orders=candidate_orders,
        candidate_positions=candidate_positions,
        interval_left=left,
        interval_right=right,
        candidate_configs=support.candidate_configs,
        selected_candidate_index=selected_index,
        uniform_candidate_index=support.uniform_candidate_index,
        weighted_candidate_indices=support.weighted_candidate_indices,
        uniform_scores=support.uniform_scores,
        weighted_scores=support.weighted_scores,
        unanimous_family_veto_triggered=veto,
        posterior_refinement_weight=support.posterior_refinement_weight,
        posterior_refinement_passes_applied=refinement_passes,
        second_refinement_uncertain_pair_threshold=(
            support.second_refinement_uncertain_pair_threshold
        ),
        stability_rank_padding=support.stability_rank_padding,
        minimum_stability_comparable_pair_fraction=(
            support.minimum_stability_comparable_pair_fraction
        ),
        stability_mass=effective_stability_mass,
        stability_comparable_pair_fraction=float(comparable_fraction),
        mean_normalized_rank_sd=float(np.mean(rank_sd) / max(p.shape[1] - 1, 1)),
        mean_pairwise_vote_margin=support.mean_pairwise_vote_margin,
        uncertain_pair_fraction_75=support.uncertain_pair_fraction_75,
        mean_genotype_certainty=mean_genotype_certainty,
        bin_membership=bins.membership,
        bin_representatives=bins.representatives,
        reported_positions=reported_positions,
        binning_method=(
            "likelihood_landmark"
            if support.ordering_method == "landmark_likelihood_mds"
            else "likelihood"
        ),
        maximum_bin_recombination=float(maximum_bin_recombination),
        minimum_bin_linkage_lod=resolved_minimum_bin_linkage_lod,
        maximum_bin_pool_evidence=(
            None
            if maximum_bin_pool_evidence is None
            else float(maximum_bin_pool_evidence)
        ),
        bin_neighbor_count=resolved_bin_neighbor_count,
        bin_neighbor_projection_dimensions=(
            resolved_bin_neighbor_projection_dimensions
            if (
                p.shape[1] >= bin_neighbor_projection_minimum_markers
                and resolved_bin_neighbor_projection_dimensions < p.shape[0]
            )
            else None
        ),
        selection_method=selection_method,
        ordering_method=support.ordering_method,
        landmark_count=support.landmark_count,
        landmark_neighbor_count=support.landmark_neighbor_count,
        landmark_support_exponent=(
            float(landmark_support_exponent)
            if using_support_weighted_landmarks
            else None
        ),
        large_scale_rescue_triggered=rescue_config is not None,
        low_certainty_stability_mass_cap_applied=stability_mass_cap_applied,
        genetic_distances=genetic_distances,
    )


def _posterior_calibration_temperature(
    result: LikelihoodMDSEnsembleResult,
) -> float:
    """Select a truth-free likelihood-softening temperature from fit diagnostics."""

    distances = result.genetic_distances
    if (
        result.mean_genotype_certainty < POSTERIOR_CALIBRATION_MINIMUM_CERTAINTY
        or distances is None
        or distances.status != "ok"
        or distances.composite_median_absolute_residual_morgan is None
    ):
        return 1.0
    residual = distances.composite_median_absolute_residual_morgan
    first, second, third = POSTERIOR_CALIBRATION_RESIDUAL_THRESHOLDS
    mild, moderate, severe = POSTERIOR_CALIBRATION_TEMPERATURES
    if residual < first:
        return 1.0
    if residual < second:
        return mild
    if residual < third:
        return moderate
    return severe


def fit_scalable_likelihood_mds_ensemble(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    maximum_dense_markers: int = 500,
    maximum_bin_recombination: float = 0.0,
    minimum_bin_linkage_lod: float | None = None,
    bin_neighbor_count: int | None = None,
    bin_neighbor_projection_dimensions: int | None = None,
    bin_neighbor_projection_minimum_markers: int = 50_000,
    large_scale_minimum_markers: int = 50_000,
    maximum_bin_pool_evidence: float | None = None,
    maximum_dense_likelihood_bins: int = 1_000,
    maximum_likelihood_landmarks: int = 256,
    landmark_support_exponent: float = 0.5,
    landmark_support_weighting_minimum_markers: int = 50_000,
    landmark_neighbor_count: int = 32,
    landmark_lod_exponent: float = 4.0,
    landmark_selected_config: LikelihoodMDSConfig | None = (
        LANDMARK_LIKELIHOOD_MDS_CONFIG
    ),
    scalable_selected_config: LikelihoodMDSConfig | None = (
        SCALABLE_LIKELIHOOD_MDS_CONFIG
    ),
    candidate_configs: Iterable[LikelihoodMDSConfig] = (DEFAULT_LIKELIHOOD_MDS_CONFIGS),
    stability_mass: float = 0.90,
    posterior_refinement_weight: float = 0.75,
    maximum_posterior_refinement_passes: int = 2,
    second_refinement_uncertain_pair_threshold: float = 0.03,
    stability_rank_padding: int = 1,
    minimum_stability_comparable_pair_fraction: float = 0.35,
    maximum_smacof_iterations: int = 500,
    automatic_posterior_calibration: bool = True,
) -> LikelihoodMDSEnsembleResult:
    """Fit the scalable map and soften globally inconsistent likelihoods once.

    The optional second pass is triggered only by the uncalibrated mean genotype
    certainty and median absolute composite-distance residual. It never uses
    truth, physical coordinates, marker input order, or a regime label.
    """

    p = _validate_probabilities(probabilities)
    names = None if marker_names is None else tuple(marker_names)
    configs = tuple(candidate_configs)
    arguments = {
        "maximum_dense_markers": maximum_dense_markers,
        "maximum_bin_recombination": maximum_bin_recombination,
        "minimum_bin_linkage_lod": minimum_bin_linkage_lod,
        "bin_neighbor_count": bin_neighbor_count,
        "bin_neighbor_projection_dimensions": (bin_neighbor_projection_dimensions),
        "bin_neighbor_projection_minimum_markers": (
            bin_neighbor_projection_minimum_markers
        ),
        "large_scale_minimum_markers": large_scale_minimum_markers,
        "maximum_bin_pool_evidence": maximum_bin_pool_evidence,
        "maximum_dense_likelihood_bins": maximum_dense_likelihood_bins,
        "maximum_likelihood_landmarks": maximum_likelihood_landmarks,
        "landmark_support_exponent": landmark_support_exponent,
        "landmark_support_weighting_minimum_markers": (
            landmark_support_weighting_minimum_markers
        ),
        "landmark_neighbor_count": landmark_neighbor_count,
        "landmark_lod_exponent": landmark_lod_exponent,
        "landmark_selected_config": landmark_selected_config,
        "scalable_selected_config": scalable_selected_config,
        "candidate_configs": configs,
        "stability_mass": stability_mass,
        "posterior_refinement_weight": posterior_refinement_weight,
        "maximum_posterior_refinement_passes": (maximum_posterior_refinement_passes),
        "second_refinement_uncertain_pair_threshold": (
            second_refinement_uncertain_pair_threshold
        ),
        "stability_rank_padding": stability_rank_padding,
        "minimum_stability_comparable_pair_fraction": (
            minimum_stability_comparable_pair_fraction
        ),
        "maximum_smacof_iterations": maximum_smacof_iterations,
        "dense_selected_config": None,
        "dense_selection_method": None,
        "dense_penalized_curve_effective_degrees_of_freedom": 4.0,
        "dense_distance_rank_span_weight_exponent": (
            DISTANCE_RANK_SPAN_WEIGHT_EXPONENT
        ),
    }
    initial = _fit_scalable_likelihood_mds_ensemble_once(p, names, **arguments)
    initial_distances = initial.genetic_distances
    initial_residual = (
        initial_distances.composite_median_absolute_residual_morgan
        if initial_distances is not None
        else None
    )
    temperature = (
        _posterior_calibration_temperature(initial)
        if automatic_posterior_calibration
        else 1.0
    )
    audit = {
        "posterior_calibration_temperature": temperature,
        "posterior_calibration_triggered": temperature > 1.0,
        "uncalibrated_mean_genotype_certainty": initial.mean_genotype_certainty,
        "uncalibrated_distance_median_absolute_residual_morgan": (initial_residual),
    }
    if temperature == 1.0:
        if (
            p.shape[1] <= maximum_dense_markers
            and initial.mean_genotype_certainty
            >= POSTERIOR_CALIBRATION_MINIMUM_CERTAINTY
            and initial_residual is not None
            and initial_residual
            >= DENSE_HIGH_INFORMATION_PENALIZED_CURVE_RESIDUAL_THRESHOLD
        ):
            robust = _fit_scalable_likelihood_mds_ensemble_once(
                p,
                names,
                **{
                    **arguments,
                    "dense_selected_config": (DENSE_CALIBRATED_HIGH_INFORMATION_CONFIG),
                    "dense_selection_method": (
                        "fixed_inconsistent_high_information_penalized_curve"
                    ),
                    "dense_penalized_curve_effective_degrees_of_freedom": (
                        DENSE_CALIBRATED_HIGH_INFORMATION_PENALIZED_CURVE_EDF
                    ),
                    "dense_distance_rank_span_weight_exponent": 0.0,
                },
            )
            return replace(robust, **audit)
        return replace(initial, **audit)

    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    tempered = 1.0 / (1.0 + np.exp(-log_odds / temperature))
    calibrated = _fit_scalable_likelihood_mds_ensemble_once(
        tempered,
        names,
        **{
            **arguments,
            "dense_selected_config": (
                DENSE_CALIBRATED_HIGH_INFORMATION_CONFIG
                if p.shape[1] <= maximum_dense_markers
                else None
            ),
            "dense_selection_method": (
                "fixed_calibrated_high_information_penalized_curve"
                if p.shape[1] <= maximum_dense_markers
                else None
            ),
            "dense_penalized_curve_effective_degrees_of_freedom": (
                DENSE_CALIBRATED_HIGH_INFORMATION_PENALIZED_CURVE_EDF
                if p.shape[1] <= maximum_dense_markers
                else 4.0
            ),
            "dense_distance_rank_span_weight_exponent": (
                0.0
                if p.shape[1] <= maximum_dense_markers
                else DISTANCE_RANK_SPAN_WEIGHT_EXPONENT
            ),
        },
    )
    return replace(calibrated, **audit)


def hmm_log_likelihood(probabilities: FloatArray, order: IntArray) -> float:
    """Conditional log likelihood of a marker order under a two-state HMM."""

    p = probabilities[:, order]
    emissions = np.stack((1.0 - p, p), axis=2)
    emissions = np.clip(emissions, 1e-12, 1.0)
    alpha = 0.5 * emissions[:, 0]
    scales = alpha.sum(axis=1)
    alpha /= scales[:, None]
    total = np.log(scales).sum()
    for marker in range(1, order.size):
        transition = float(
            np.clip(
                expected_disagreement(
                    probabilities[:, int(order[marker - 1])],
                    probabilities[:, int(order[marker])],
                ),
                1e-5,
                0.49,
            )
        )
        predicted = np.empty_like(alpha)
        predicted[:, 0] = (1.0 - transition) * alpha[:, 0] + transition * alpha[:, 1]
        predicted[:, 1] = transition * alpha[:, 0] + (1.0 - transition) * alpha[:, 1]
        alpha = predicted * emissions[:, marker]
        scales = alpha.sum(axis=1)
        total += np.log(np.clip(scales, 1e-300, None)).sum()
        alpha /= np.clip(scales[:, None], 1e-300, None)
    return float(total)


@dataclass(frozen=True)
class _HMMInsertionContext:
    probabilities: FloatArray
    order: IntArray
    framework_emissions: FloatArray
    alpha: FloatArray
    prefix_log: FloatArray
    beta: FloatArray
    suffix_log: FloatArray


def _prepare_hmm_insertion_context(
    probabilities: FloatArray,
    framework: IntArray,
) -> _HMMInsertionContext:
    """Cache framework-only HMM messages shared by candidate insertions."""
    p = _validate_probabilities(probabilities)
    order = np.asarray(framework, dtype=np.int64)
    if order.ndim != 1 or order.size < 2:
        raise ValueError("insertion scoring requires at least two framework markers")
    if np.any((order < 0) | (order >= p.shape[1])):
        raise ValueError("framework contains an index outside the probability matrix")
    if np.unique(order).size != order.size:
        raise ValueError("framework marker indices must be unique")

    framework_emissions = np.stack((1.0 - p[:, order], p[:, order]), axis=2)
    framework_emissions = np.clip(framework_emissions, 1e-12, 1.0)
    n_offspring, n_framework = framework_emissions.shape[:2]

    transitions = np.asarray(
        [
            np.clip(
                expected_disagreement(p[:, int(left)], p[:, int(right)]),
                1e-5,
                0.49,
            )
            for left, right in pairwise(order)
        ],
        dtype=np.float64,
    )

    alpha = np.empty((n_offspring, n_framework, 2), dtype=np.float64)
    prefix_log = np.empty((n_offspring, n_framework), dtype=np.float64)
    raw = 0.5 * framework_emissions[:, 0]
    scale = raw.sum(axis=1)
    alpha[:, 0] = raw / scale[:, None]
    prefix_log[:, 0] = np.log(scale)
    for marker in range(1, n_framework):
        transition = transitions[marker - 1]
        previous = alpha[:, marker - 1]
        predicted = np.empty_like(previous)
        predicted[:, 0] = (1.0 - transition) * previous[:, 0] + transition * previous[
            :, 1
        ]
        predicted[:, 1] = (
            transition * previous[:, 0] + (1.0 - transition) * previous[:, 1]
        )
        raw = predicted * framework_emissions[:, marker]
        scale = raw.sum(axis=1)
        alpha[:, marker] = raw / np.clip(scale[:, None], 1e-300, None)
        prefix_log[:, marker] = prefix_log[:, marker - 1] + np.log(
            np.clip(scale, 1e-300, None)
        )

    beta = np.ones_like(alpha)
    suffix_log = np.zeros((n_offspring, n_framework), dtype=np.float64)
    for marker in range(n_framework - 2, -1, -1):
        transition = transitions[marker]
        weighted = framework_emissions[:, marker + 1] * beta[:, marker + 1]
        raw = np.empty_like(weighted)
        raw[:, 0] = (1.0 - transition) * weighted[:, 0] + transition * weighted[:, 1]
        raw[:, 1] = transition * weighted[:, 0] + (1.0 - transition) * weighted[:, 1]
        scale = raw.sum(axis=1)
        beta[:, marker] = raw / np.clip(scale[:, None], 1e-300, None)
        suffix_log[:, marker] = suffix_log[:, marker + 1] + np.log(
            np.clip(scale, 1e-300, None)
        )

    return _HMMInsertionContext(
        probabilities=p,
        order=order,
        framework_emissions=framework_emissions,
        alpha=alpha,
        prefix_log=prefix_log,
        beta=beta,
        suffix_log=suffix_log,
    )


def _hmm_insertion_scores_prepared(
    context: _HMMInsertionContext,
    candidate: int,
) -> FloatArray:
    """Score one candidate using precomputed framework HMM messages."""

    p = context.probabilities
    order = context.order
    framework_emissions = context.framework_emissions
    alpha = context.alpha
    prefix_log = context.prefix_log
    beta = context.beta
    suffix_log = context.suffix_log
    candidate = int(candidate)
    if candidate < 0 or candidate >= p.shape[1]:
        raise ValueError("candidate index is outside the probability matrix")
    if np.any(order == candidate):
        raise ValueError("candidate is already in the framework")
    candidate_emission = np.stack((1.0 - p[:, candidate], p[:, candidate]), axis=1)
    candidate_emission = np.clip(candidate_emission, 1e-12, 1.0)
    n_framework = order.size

    scores = np.empty(n_framework + 1, dtype=np.float64)

    transition_right = float(
        np.clip(expected_disagreement(p[:, candidate], p[:, int(order[0])]), 1e-5, 0.49)
    )
    weighted_candidate = 0.5 * candidate_emission
    predicted_right = np.empty_like(weighted_candidate)
    predicted_right[:, 0] = (1.0 - transition_right) * weighted_candidate[
        :, 0
    ] + transition_right * weighted_candidate[:, 1]
    predicted_right[:, 1] = (
        transition_right * weighted_candidate[:, 0]
        + (1.0 - transition_right) * weighted_candidate[:, 1]
    )
    bridge = np.sum(predicted_right * framework_emissions[:, 0] * beta[:, 0], axis=1)
    scores[0] = np.sum(suffix_log[:, 0] + np.log(np.clip(bridge, 1e-300, None)))

    for position in range(1, n_framework):
        left = int(order[position - 1])
        right = int(order[position])
        transition_left = float(
            np.clip(expected_disagreement(p[:, left], p[:, candidate]), 1e-5, 0.49)
        )
        transition_right = float(
            np.clip(expected_disagreement(p[:, candidate], p[:, right]), 1e-5, 0.49)
        )
        previous = alpha[:, position - 1]
        predicted_candidate = np.empty_like(previous)
        predicted_candidate[:, 0] = (1.0 - transition_left) * previous[
            :, 0
        ] + transition_left * previous[:, 1]
        predicted_candidate[:, 1] = (
            transition_left * previous[:, 0] + (1.0 - transition_left) * previous[:, 1]
        )
        weighted_candidate = predicted_candidate * candidate_emission
        predicted_right = np.empty_like(weighted_candidate)
        predicted_right[:, 0] = (1.0 - transition_right) * weighted_candidate[
            :, 0
        ] + transition_right * weighted_candidate[:, 1]
        predicted_right[:, 1] = (
            transition_right * weighted_candidate[:, 0]
            + (1.0 - transition_right) * weighted_candidate[:, 1]
        )
        bridge = np.sum(
            predicted_right * framework_emissions[:, position] * beta[:, position],
            axis=1,
        )
        scores[position] = np.sum(
            prefix_log[:, position - 1]
            + suffix_log[:, position]
            + np.log(np.clip(bridge, 1e-300, None))
        )

    transition_left = float(
        np.clip(
            expected_disagreement(p[:, int(order[-1])], p[:, candidate]), 1e-5, 0.49
        )
    )
    previous = alpha[:, -1]
    predicted_candidate = np.empty_like(previous)
    predicted_candidate[:, 0] = (1.0 - transition_left) * previous[
        :, 0
    ] + transition_left * previous[:, 1]
    predicted_candidate[:, 1] = (
        transition_left * previous[:, 0] + (1.0 - transition_left) * previous[:, 1]
    )
    bridge = np.sum(predicted_candidate * candidate_emission, axis=1)
    scores[-1] = np.sum(prefix_log[:, -1] + np.log(np.clip(bridge, 1e-300, None)))
    return scores


def hmm_insertion_scores(
    probabilities: FloatArray,
    framework: IntArray,
    candidate: int,
) -> FloatArray:
    """Exact HMM log likelihood for every candidate insertion in linear time."""

    context = _prepare_hmm_insertion_context(probabilities, framework)
    return _hmm_insertion_scores_prepared(context, candidate)


def hmm_placement_intervals(
    probabilities: FloatArray,
    framework: IntArray,
    *,
    confidence: float = 0.8,
    temperature: float = 1.0,
    padding: int = 0,
) -> tuple[IntArray, IntArray]:
    """Shortest contiguous insertion-slot intervals from HMM likelihood profiles.

    The likelihood at every possible insertion slot is normalized under a uniform
    slot prior.  For each non-framework marker, the returned interval is the
    narrowest contiguous group of slots containing at least ``confidence`` mass.
    ``temperature`` and ``padding`` are explicit calibration parameters.  Values
    above one flatten the profile, while padding adds neighboring insertion slots
    after selecting the likelihood interval; neither changes the best slot.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("HMM interval confidence must lie between zero and one")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("HMM interval temperature must be positive and finite")
    if padding < 0:
        raise ValueError("HMM interval padding must be nonnegative")
    context = _prepare_hmm_insertion_context(probabilities, framework)
    p = context.probabilities
    anchors = context.order
    left = np.full(p.shape[1], -1, dtype=np.int64)
    right = np.full(p.shape[1], anchors.size, dtype=np.int64)
    framework_lookup = {int(marker): slot for slot, marker in enumerate(anchors)}
    for marker in range(p.shape[1]):
        if marker in framework_lookup:
            slot = framework_lookup[marker]
            left[marker] = right[marker] = slot
            continue
        scores = _hmm_insertion_scores_prepared(context, marker)
        scaled = (scores - float(np.max(scores))) / temperature
        mass = np.exp(scaled)
        mass /= np.sum(mass)
        cumulative = np.concatenate(([0.0], np.cumsum(mass)))
        best: tuple[int, float, int, int] | None = None
        end = 0
        for start in range(mass.size):
            end = max(end, start)
            while (
                end < mass.size and cumulative[end + 1] - cumulative[start] < confidence
            ):
                end += 1
            if end >= mass.size:
                break
            contained = float(cumulative[end + 1] - cumulative[start])
            candidate = (end - start, -contained, start, end)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            # Rounding can only make the requested mass miss by machine epsilon.
            low_slot, high_slot = 0, mass.size - 1
        else:
            _, _, low_slot, high_slot = best
        low_slot = max(0, low_slot - padding)
        high_slot = min(anchors.size, high_slot + padding)
        left[marker] = low_slot - 1
        right[marker] = high_slot
    return left, right


def prune_framework_likelihood(
    probabilities: FloatArray,
    framework: IntArray,
    *,
    min_log10_gap: float = 2.0,
) -> IntArray:
    """Remove scaffold anchors whose stated position lacks multipoint support.

    Each anchor is temporarily removed and rescored at every insertion slot.  The
    least-supported anchor is deleted until every retained anchor's stated slot
    beats all alternatives by ``min_log10_gap`` or only two endpoints remain.
    Removed anchors remain eligible for later likelihood densification.
    """

    p = _validate_probabilities(probabilities)
    if min_log10_gap < 0.0:
        raise ValueError("min_log10_gap must be nonnegative")
    retained = list(map(int, np.asarray(framework, dtype=np.int64)))
    if len(retained) < 2:
        raise ValueError("likelihood pruning requires at least two framework markers")
    if len(set(retained)) != len(retained):
        raise ValueError("framework marker indices must be unique")
    if any(marker < 0 or marker >= p.shape[1] for marker in retained):
        raise ValueError("framework contains an index outside the probability matrix")
    while len(retained) > 2:
        support = np.empty(len(retained), dtype=np.float64)
        for position, marker in enumerate(retained):
            reduced = np.asarray(
                retained[:position] + retained[position + 1 :],
                dtype=np.int64,
            )
            scores = hmm_insertion_scores(p, reduced, marker)
            stated = float(scores[position])
            alternatives = np.delete(scores, position)
            support[position] = (stated - float(np.max(alternatives))) / np.log(10.0)
        worst = int(np.argmin(support))
        if support[worst] >= min_log10_gap:
            break
        retained.pop(worst)
    return np.asarray(retained, dtype=np.int64)


def audit_scaffold_likelihood(
    probabilities: FloatArray,
    framework: IntArray,
    scaffold: IntArray,
    *,
    min_log10_gap: float = 0.0,
) -> IntArray:
    """Remove scaffold anchors contradicted by the densified HMM framework.

    Bootstrap support is estimated before likelihood densification, when a sparse
    scaffold can lack the local context needed to expose a misplaced anchor.  This
    audit retests only the original scaffold markers after supported insertions have
    supplied that context.  The least-supported scaffold anchor is removed until
    every surviving anchor's stated slot beats all alternative slots by
    ``min_log10_gap`` or only two framework markers remain.  Likelihood-inserted
    markers are deliberately left untouched by this targeted audit.
    """

    p = _validate_probabilities(probabilities)
    if min_log10_gap < 0.0:
        raise ValueError("min_log10_gap must be nonnegative")
    retained = list(map(int, np.asarray(framework, dtype=np.int64)))
    anchors = set(map(int, np.asarray(scaffold, dtype=np.int64)))
    if len(retained) < 2:
        raise ValueError("scaffold audit requires at least two framework markers")
    if len(set(retained)) != len(retained):
        raise ValueError("framework marker indices must be unique")
    if any(marker < 0 or marker >= p.shape[1] for marker in retained):
        raise ValueError("framework contains an index outside the probability matrix")
    if not anchors.issubset(retained):
        raise ValueError("scaffold must be a subset of the framework")

    while len(retained) > 2 and anchors:
        tested: list[tuple[float, int, int]] = []
        for position, marker in enumerate(retained):
            if marker not in anchors:
                continue
            reduced = np.asarray(
                retained[:position] + retained[position + 1 :],
                dtype=np.int64,
            )
            scores = hmm_insertion_scores(p, reduced, marker)
            alternatives = np.delete(scores, position)
            support = (float(scores[position]) - float(np.max(alternatives))) / np.log(
                10.0
            )
            tested.append((support, position, marker))
        if not tested:
            break
        support, position, marker = min(tested, key=lambda item: item[0])
        if support >= min_log10_gap:
            break
        retained.pop(position)
        anchors.remove(marker)
    return np.asarray(retained, dtype=np.int64)


def densify_framework_likelihood(
    probabilities: FloatArray,
    scaffold: IntArray,
    candidate_order: IntArray,
    *,
    min_log10_gap: float = 3.0,
    max_additions: int | None = None,
    max_terminal_additions_per_side: int | None = None,
    greedy_pass: bool = False,
    support_priority_pass: bool = False,
) -> IntArray:
    """Insert markers whose best HMM position beats every alternative decisively.

    ``support_priority_pass`` is the fast statistically aligned mode.  It scores
    every remaining candidate against the current framework, tries candidates in
    descending likelihood-gap order, and repeats only when accepted insertions may
    have supplied useful new context.  This avoids the exhaustive mode's complete
    candidate rescan after every insertion without using genotype certainty as a
    proxy for positional support.

    ``max_terminal_additions_per_side`` limits unbracketed extensions beyond the
    two current endpoints.  After that many extensions on a side, a candidate is
    eligible only if it has become bracketed by retained markers.  This prevents
    a locally supported but mutually unresolved set of terminal candidates from
    being converted into an arbitrary total order.
    """

    p = _validate_probabilities(probabilities)
    if min_log10_gap < 0.0:
        raise ValueError("min_log10_gap must be nonnegative")
    if max_additions is not None and max_additions < 0:
        raise ValueError("max_additions must be nonnegative")
    if (
        max_terminal_additions_per_side is not None
        and max_terminal_additions_per_side < 0
    ):
        raise ValueError("max_terminal_additions_per_side must be nonnegative")
    if greedy_pass and support_priority_pass:
        raise ValueError("greedy_pass and support_priority_pass are mutually exclusive")
    framework = list(map(int, scaffold))
    if len(framework) < 2:
        raise ValueError(
            "likelihood densification requires at least two scaffold markers"
        )
    if len(set(framework)) != len(framework):
        raise ValueError("scaffold marker indices must be unique")
    if any(marker < 0 or marker >= p.shape[1] for marker in framework):
        raise ValueError("scaffold contains an index outside the probability matrix")
    candidates = [
        int(marker) for marker in candidate_order if int(marker) not in set(framework)
    ]
    required_gap = min_log10_gap * np.log(10.0)
    additions = 0
    left_terminal_additions = 0
    right_terminal_additions = 0

    def terminal_allowed(position: int, size: int) -> bool:
        if max_terminal_additions_per_side is None:
            return True
        if position == 0:
            return left_terminal_additions < max_terminal_additions_per_side
        if position == size:
            return right_terminal_additions < max_terminal_additions_per_side
        return True

    def record_terminal(position: int, size: int) -> None:
        nonlocal left_terminal_additions, right_terminal_additions
        if position == 0:
            left_terminal_additions += 1
        elif position == size:
            right_terminal_additions += 1

    if support_priority_pass:
        information = np.mean(np.abs(2.0 * p - 1.0), axis=0)
        while candidates and (max_additions is None or additions < max_additions):
            current = np.asarray(framework, dtype=np.int64)
            initial_context = _prepare_hmm_insertion_context(p, current)
            priority: list[tuple[float, float, int]] = []
            for candidate in candidates:
                scores = _hmm_insertion_scores_prepared(initial_context, candidate)
                ranking = np.argsort(scores, kind="stable")
                gap = float(scores[ranking[-1]] - scores[ranking[-2]])
                priority.append((gap, float(information[candidate]), candidate))
            priority.sort(key=lambda item: (-item[0], -item[1], item[2]))

            progressed = False
            current_context: _HMMInsertionContext | None = None
            for _, _, candidate in priority:
                if max_additions is not None and additions >= max_additions:
                    break
                if current_context is None:
                    current_context = _prepare_hmm_insertion_context(
                        p, np.asarray(framework, dtype=np.int64)
                    )
                scores = _hmm_insertion_scores_prepared(current_context, candidate)
                ranking = np.argsort(scores, kind="stable")
                gap = float(scores[ranking[-1]] - scores[ranking[-2]])
                if gap < required_gap:
                    continue
                best_position = int(ranking[-1])
                framework_size = len(framework)
                if not terminal_allowed(best_position, framework_size):
                    continue
                framework.insert(best_position, candidate)
                record_terminal(best_position, framework_size)
                candidates.remove(candidate)
                additions += 1
                progressed = True
                current_context = None
            if not progressed:
                break
        return np.asarray(framework, dtype=np.int64)
    if greedy_pass:
        information = np.mean(np.abs(2.0 * p - 1.0), axis=0)
        candidates.sort(key=lambda marker: float(information[marker]), reverse=True)
        for candidate in candidates:
            if max_additions is not None and additions >= max_additions:
                break
            scores = hmm_insertion_scores(
                p, np.asarray(framework, dtype=np.int64), candidate
            )
            ranking = np.argsort(scores)
            gap = float(scores[ranking[-1]] - scores[ranking[-2]])
            best_position = int(ranking[-1])
            framework_size = len(framework)
            if gap >= required_gap and terminal_allowed(best_position, framework_size):
                framework.insert(best_position, candidate)
                record_terminal(best_position, framework_size)
                additions += 1
        return np.asarray(framework, dtype=np.int64)
    while candidates and (max_additions is None or additions < max_additions):
        best_candidate: int | None = None
        best_position = -1
        best_gap = -np.inf
        for candidate in candidates:
            scores = hmm_insertion_scores(
                p, np.asarray(framework, dtype=np.int64), candidate
            )
            ranking = np.argsort(scores)
            gap = float(scores[ranking[-1]] - scores[ranking[-2]])
            candidate_position = int(ranking[-1])
            if not terminal_allowed(candidate_position, len(framework)):
                continue
            if gap > best_gap:
                best_gap = gap
                best_candidate = candidate
                best_position = candidate_position
        if best_candidate is None or best_gap < required_gap:
            break
        framework_size = len(framework)
        framework.insert(best_position, best_candidate)
        record_terminal(best_position, framework_size)
        candidates.remove(best_candidate)
        additions += 1
    return np.asarray(framework, dtype=np.int64)


def densify_framework_resampled_likelihood(
    probabilities: FloatArray,
    scaffold: IntArray,
    candidate_order: IntArray,
    *,
    min_log10_gap: float = 3.0,
    min_position_support: float = 0.8,
    bootstrap_replicates: int = 50,
    max_additions: int | None = None,
    max_terminal_additions_per_side: int | None = None,
    sample_states: bool = True,
    resample_offspring: bool = True,
    random_seed: int | None = None,
) -> IntArray:
    """Insert candidates supported by full-data and resampled multipoint scores.

    The full-data best slot must clear ``min_log10_gap`` and the same slot must be
    optimal in at least ``min_position_support`` of offspring/state bootstraps.
    Bootstrap matrices are generated only for the current framework plus one
    candidate, avoiding a replicate-by-offspring-by-all-markers allocation.
    """

    p = _validate_probabilities(probabilities)
    if min_log10_gap < 0.0:
        raise ValueError("min_log10_gap must be nonnegative")
    if not 0.5 < min_position_support <= 1.0:
        raise ValueError("min_position_support must lie in (0.5, 1]")
    if bootstrap_replicates < 2:
        raise ValueError("at least two HMM bootstrap replicates are required")
    if max_additions is not None and max_additions < 0:
        raise ValueError("max_additions must be nonnegative")
    if (
        max_terminal_additions_per_side is not None
        and max_terminal_additions_per_side < 0
    ):
        raise ValueError("max_terminal_additions_per_side must be nonnegative")
    framework = list(map(int, np.asarray(scaffold, dtype=np.int64)))
    if len(framework) < 2:
        raise ValueError(
            "likelihood densification requires at least two scaffold markers"
        )
    if len(set(framework)) != len(framework):
        raise ValueError("scaffold marker indices must be unique")
    if any(marker < 0 or marker >= p.shape[1] for marker in framework):
        raise ValueError("scaffold contains an index outside the probability matrix")

    framework_set = set(framework)
    candidates = [
        int(marker) for marker in candidate_order if int(marker) not in framework_set
    ]
    information = np.mean(np.abs(2.0 * p - 1.0), axis=0)
    candidates.sort(key=lambda marker: float(information[marker]), reverse=True)
    required_gap = min_log10_gap * np.log(10.0)
    required_support = int(np.ceil(min_position_support * bootstrap_replicates))
    rng = np.random.default_rng(random_seed)
    replicate_seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=bootstrap_replicates, dtype=np.int64
    )
    additions = 0
    left_terminal_additions = 0
    right_terminal_additions = 0
    for candidate in candidates:
        if max_additions is not None and additions >= max_additions:
            break
        framework_array = np.asarray(framework, dtype=np.int64)
        scores = hmm_insertion_scores(p, framework_array, candidate)
        ranking = np.argsort(scores, kind="stable")
        best_position = int(ranking[-1])
        if float(scores[ranking[-1]] - scores[ranking[-2]]) < required_gap:
            continue
        if max_terminal_additions_per_side is not None:
            if (
                best_position == 0
                and left_terminal_additions >= max_terminal_additions_per_side
            ):
                continue
            if (
                best_position == framework_array.size
                and right_terminal_additions >= max_terminal_additions_per_side
            ):
                continue

        columns = np.concatenate((framework_array, np.asarray([candidate])))
        local_framework = np.arange(framework_array.size, dtype=np.int64)
        local_candidate = int(framework_array.size)
        supported = 0
        for replicate_index, replicate_seed in enumerate(replicate_seeds):
            replicate_rng = np.random.default_rng(int(replicate_seed))
            rows = (
                replicate_rng.integers(0, p.shape[0], size=p.shape[0])
                if resample_offspring
                else np.arange(p.shape[0])
            )
            boot = p[np.ix_(rows, columns)]
            if sample_states:
                boot = (replicate_rng.random(boot.shape) < boot).astype(np.float64)
                boot = boot * 0.998 + 0.001
            boot_scores = hmm_insertion_scores(boot, local_framework, local_candidate)
            if int(np.argmax(boot_scores)) == best_position:
                supported += 1
                if supported >= required_support:
                    break
            remaining = bootstrap_replicates - replicate_index - 1
            if supported + remaining < required_support:
                break
        if supported >= required_support:
            framework.insert(best_position, candidate)
            if best_position == 0:
                left_terminal_additions += 1
            elif best_position == framework_array.size:
                right_terminal_additions += 1
            additions += 1
    return np.asarray(framework, dtype=np.int64)


def _expand_bin_order(bin_order: IntArray, bins: MarkerBins) -> IntArray:
    members: list[int] = []
    for group in map(int, bin_order):
        members.extend(map(int, np.flatnonzero(bins.membership == group)))
    return np.asarray(members, dtype=np.int64)


def fit_softmap(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    confidence: float = 0.95,
    bootstrap_replicates: int = 100,
    bin_threshold: float | None = 0.01,
    neighbor_count: int = 20,
    sample_states: bool = True,
    resample_offspring: bool = True,
    ordering_ensemble: bool = False,
    random_seed: int | None = None,
) -> SoftMapResult:
    """Fit the phased binary SoftMap MVP."""

    p = _validate_probabilities(probabilities)
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    names = tuple(marker_names or (f"m{i}" for i in range(p.shape[1])))
    if len(names) != p.shape[1]:
        raise ValueError("marker_names length does not match probability columns")
    bins = (
        auto_bin_markers(
            p,
            neighbor_count=min(neighbor_count, p.shape[1]),
        )
        if bin_threshold is None
        else bin_markers(
            p,
            threshold=bin_threshold,
            neighbor_count=min(neighbor_count, p.shape[1]),
        )
    )
    representative_order = order_markers(
        bins.probabilities,
        neighbor_count=min(neighbor_count, bins.probabilities.shape[1]),
        ordering_ensemble=ordering_ensemble,
    )
    positions = bootstrap_orders(
        bins.probabilities,
        representative_order,
        replicates=bootstrap_replicates,
        neighbor_count=min(neighbor_count, bins.probabilities.shape[1]),
        sample_states=sample_states,
        resample_offspring=resample_offspring,
        ordering_ensemble=ordering_ensemble,
        random_seed=random_seed,
    )
    precedence = precedence_matrix(positions)
    representative_order = np.argsort(np.mean(positions, axis=0), kind="stable").astype(
        np.int64
    )
    framework = select_framework(
        representative_order, precedence, confidence=confidence
    )
    left, right = placement_intervals(framework, precedence, confidence=confidence)
    order = _expand_bin_order(representative_order, bins)
    return SoftMapResult(
        names,
        order,
        representative_order,
        bins,
        positions,
        precedence,
        framework,
        left,
        right,
        confidence,
        float(p.shape[0] * np.mean(np.abs(2.0 * p - 1.0))),
    )


def fit_hierarchical_softmap(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    interval_confidence: float = 0.8,
    scaffold_confidence: float = 0.85,
    bootstrap_replicates: int = 100,
    coarse_bin_threshold: float | None = 0.03,
    fine_bin_threshold: float | None = 0.015,
    auto_fine_bins: bool = False,
    neighbor_count: int = 200,
    min_log10_gap: float = 3.0,
    scaffold_prune_log10_gap: float | None = None,
    post_scaffold_log10_gap: float | None = None,
    hmm_bootstrap_replicates: int | None = None,
    hmm_position_support: float = 0.8,
    max_additions: int | None = None,
    max_terminal_additions_per_side: int | None = None,
    greedy_pass: bool = False,
    support_priority_pass: bool = False,
    global_scaffold: bool = False,
    sample_states: bool = True,
    resample_offspring: bool = True,
    ordering_ensemble: bool = False,
    random_seed: int | None = None,
) -> HierarchicalSoftMapResult:
    """Fit a stable coarse scaffold and densify it using finer HMM candidates."""

    p = _validate_probabilities(probabilities)
    if not 0.5 < scaffold_confidence < 1.0:
        raise ValueError("scaffold_confidence must be between 0.5 and 1")
    if auto_fine_bins and fine_bin_threshold is not None:
        raise ValueError(
            "auto_fine_bins cannot be combined with an explicit fine_bin_threshold"
        )
    if (
        coarse_bin_threshold is not None
        and fine_bin_threshold is not None
        and fine_bin_threshold >= coarse_bin_threshold
    ):
        raise ValueError("fine_bin_threshold must be below coarse_bin_threshold")
    support = fit_softmap(
        p,
        marker_names,
        confidence=interval_confidence,
        bootstrap_replicates=bootstrap_replicates,
        bin_threshold=coarse_bin_threshold,
        neighbor_count=neighbor_count,
        sample_states=sample_states,
        resample_offspring=resample_offspring,
        ordering_ensemble=ordering_ensemble,
        random_seed=random_seed,
    )
    if auto_fine_bins:
        marker_to_coarse_ratio = p.shape[1] / support.bins.representatives.size
        # A second resolution pays off when marker density is far above the
        # number of coarse segregation states (as in the 10k-marker benchmark).
        # At ordinary density, splitting pooled evidence into weak candidates can
        # shrink rather than densify the likelihood-supported framework.
        fine_bin_threshold = (
            support.bins.threshold / 2.0 if marker_to_coarse_ratio >= 8.0 else None
        )
    if fine_bin_threshold is not None and fine_bin_threshold >= support.bins.threshold:
        raise ValueError(
            "fine_bin_threshold must be below the selected coarse threshold "
            f"({support.bins.threshold:g})"
        )
    coarse_scaffold = (
        select_framework_global(
            support.representative_order,
            support.bootstrap_positions,
            confidence=scaffold_confidence,
        )
        if global_scaffold
        else select_framework(
            support.representative_order,
            support.precedence,
            confidence=scaffold_confidence,
        )
    )
    if scaffold_prune_log10_gap is not None:
        coarse_scaffold = prune_framework_likelihood(
            support.bins.probabilities,
            coarse_scaffold,
            min_log10_gap=scaffold_prune_log10_gap,
        )
    if fine_bin_threshold is None:
        bins = support.bins
        scaffold = coarse_scaffold
        candidate_order = support.representative_order
    else:
        bins = bin_markers(
            p,
            threshold=fine_bin_threshold,
            neighbor_count=neighbor_count,
        )
        mapped: list[int] = []
        seen: set[int] = set()
        for coarse_group in coarse_scaffold:
            original_marker = int(support.bins.representatives[int(coarse_group)])
            fine_group = int(bins.membership[original_marker])
            if fine_group not in seen:
                mapped.append(fine_group)
                seen.add(fine_group)
        scaffold = np.asarray(mapped, dtype=np.int64)
        candidate_order = np.arange(bins.representatives.size, dtype=np.int64)
    if hmm_bootstrap_replicates is None:
        framework = densify_framework_likelihood(
            bins.probabilities,
            scaffold,
            candidate_order,
            min_log10_gap=min_log10_gap,
            max_additions=max_additions,
            max_terminal_additions_per_side=max_terminal_additions_per_side,
            greedy_pass=greedy_pass,
            support_priority_pass=support_priority_pass,
        )
    else:
        framework = densify_framework_resampled_likelihood(
            bins.probabilities,
            scaffold,
            candidate_order,
            min_log10_gap=min_log10_gap,
            min_position_support=hmm_position_support,
            bootstrap_replicates=hmm_bootstrap_replicates,
            max_additions=max_additions,
            max_terminal_additions_per_side=max_terminal_additions_per_side,
            sample_states=sample_states,
            resample_offspring=resample_offspring,
            random_seed=(None if random_seed is None else random_seed + 2_000_003),
        )
    if post_scaffold_log10_gap is not None:
        framework = audit_scaffold_likelihood(
            bins.probabilities,
            framework,
            scaffold,
            min_log10_gap=post_scaffold_log10_gap,
        )
    return HierarchicalSoftMapResult(
        support=support,
        bins=bins,
        scaffold=scaffold,
        framework=framework,
        fine_bin_threshold=fine_bin_threshold,
        min_log10_gap=min_log10_gap,
        scaffold_prune_log10_gap=scaffold_prune_log10_gap,
        post_scaffold_log10_gap=post_scaffold_log10_gap,
        hmm_bootstrap_replicates=hmm_bootstrap_replicates,
        hmm_position_support=(
            hmm_position_support if hmm_bootstrap_replicates is not None else None
        ),
        effective_offspring_information=float(
            p.shape[0] * np.mean(np.abs(2.0 * p - 1.0))
        ),
    )
