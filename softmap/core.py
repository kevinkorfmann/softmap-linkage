"""Core algorithms for a phased, binary-parental-origin SoftMap MVP.

The implementation deliberately separates point ordering from uncertainty
estimation. Sparse spectral/HMM mapping remains available for supported-framework
experiments; likelihood-weighted MDS supplies the competitive dense reference
order that is now being connected to bootstrap confidence output.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.interpolate import LSQUnivariateSpline
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, squareform
from scipy.stats import rankdata

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
    ("haldane", 3.0, 50, 1),
    ("rf", 2.0, 20, 3),
)


@dataclass(frozen=True)
class LikelihoodMDSEnsembleResult:
    """Confidence-first total order and model-stability rank bands."""

    marker_names: tuple[str, ...]
    order: IntArray
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
    stability_mass: float
    stability_comparable_pair_fraction: float
    mean_normalized_rank_sd: float
    mean_pairwise_vote_margin: float
    uncertain_pair_fraction_75: float
    mean_genotype_certainty: float

    def ordered_names(self) -> list[str]:
        return [self.marker_names[int(index)] for index in self.order]

    @property
    def selected_config(self) -> LikelihoodMDSConfig:
        return self.candidate_configs[self.selected_candidate_index]

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


def _candidate_neighbors(probabilities: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
    """Find candidate neighbors in certainty-scaled posterior space."""

    n_markers = probabilities.shape[1]
    k = min(max(2, k), n_markers)
    features = (probabilities.T - 0.5) * 2.0
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
) -> MarkerBins:
    p = probabilities
    pooled = np.empty((p.shape[0], representatives.size), dtype=np.float64)
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    members_by_group: list[list[int]] = [
        [] for _ in range(representatives.size)
    ]
    for marker, group in enumerate(membership):
        members_by_group[int(group)].append(marker)
    for group, members in enumerate(members_by_group):
        combined = np.clip(log_odds[:, members].sum(axis=1), -30.0, 30.0)
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
    if any(right <= left for left, right in zip(values[:-1], values[1:])):
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
        information_ceiling = min(
            p.shape[1], max(20, int(np.ceil(3.2 * p.shape[0])))
        )
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
    prior_exponent = beta_prior_shape - 1.0
    marker_count = p.shape[1]
    recombination = np.zeros((marker_count, marker_count), dtype=np.float64)
    lod = np.zeros_like(recombination)
    for marker in range(1, marker_count):
        current = p[:, marker, None]
        previous = p[:, :marker]
        same = (1.0 - current) * (1.0 - previous) + current * previous
        different = current * (1.0 - previous) + (1.0 - current) * previous
        delta = different - same
        flat = np.sum(delta * delta, axis=0) <= 1e-20

        def score(values: FloatArray) -> FloatArray:
            denominator = np.clip(
                same + delta * values[None, :], 1e-300, None
            )
            result = np.sum(delta / denominator, axis=0)
            if prior_exponent > 0.0:
                bounded = np.clip(values, 1e-12, 1.0 - 1e-12)
                result += prior_exponent * (
                    1.0 / bounded - 1.0 / (1.0 - bounded)
                )
            return result

        low = np.full(
            marker, 1e-12 if prior_exponent > 0.0 else 0.0,
            dtype=np.float64,
        )
        high = np.full(marker, maximum_recombination, dtype=np.float64)
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
        recombination[marker, :marker] = fitted
        recombination[:marker, marker] = fitted
        lod[marker, :marker] = fitted_lod
        lod[:marker, marker] = fitted_lod
    return recombination, lod


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
    for segment, (start, direction) in enumerate(zip(starts, directions)):
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


def _smooth_principal_curve_order(
    coordinates: FloatArray,
    *,
    interior_knots: int = 2,
    maximum_iterations: int = 50,
    relative_tolerance: float = 1e-3,
) -> IntArray:
    """Order weighted-MDS coordinates with a compact principal-curve fit."""

    if interior_knots < 1:
        raise ValueError("principal curve requires at least one interior knot")
    points = np.asarray(coordinates, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    initial_coordinate = left_vectors[:, 0] * singular_values[0]
    initial_curve = (
        np.outer(np.sort(initial_coordinate), right_vectors[0])
        + np.mean(points, axis=0)
    )
    arclength, order, previous_distance = _project_to_polyline(
        points, initial_curve
    )
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
        curve = np.column_stack([
            LSQUnivariateSpline(
                coordinate,
                points[ranked, dimension],
                knots,
                k=3,
            )(coordinate)
            for dimension in range(points.shape[1])
        ])
        arclength, order, distance = _project_to_polyline(points, curve)
        if abs(previous_distance - distance) <= (
            relative_tolerance * max(previous_distance, 1e-30)
        ):
            break
        previous_distance = distance
    # Break rare projection ties reproducibly in the initial principal direction.
    return np.lexsort((initial_coordinate, arclength)).astype(np.int64)


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
    """Order markers by LOD-weighted MDS and a smooth principal curve.

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
        dissimilarity = 0.25 * np.log(np.clip(
            (1.0 + 2.0 * rf) / (1.0 - 2.0 * rf), 1e-16, None
        ))
    else:
        raise ValueError("distance must be 'rf', 'haldane', or 'kosambi'")
    np.fill_diagonal(dissimilarity, 0.0)
    weights = linkage_lod ** lod_exponent
    np.fill_diagonal(weights, 0.0)
    maximum_weight = float(np.max(weights))
    if maximum_weight <= 0.0:
        raise ValueError("at least one marker pair must have positive linkage LOD")
    weights /= maximum_weight

    centering = np.eye(marker_count) - np.ones((marker_count, marker_count)) / marker_count
    gram = -0.5 * centering @ (dissimilarity * dissimilarity) @ centering
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
    weight_laplacian = np.diag(np.sum(weights, axis=1)) - weights
    inverse_laplacian = linalg.pinvh(weight_laplacian, check_finite=False)
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
        stress = 0.5 * float(np.sum(
            weights * (dissimilarity - cdist(updated, updated)) ** 2
        ))
        coordinates = updated
        if previous_stress is not None and abs(previous_stress - stress) <= (
            smacof_tolerance * max(previous_stress, 1.0)
        ):
            break
        previous_stress = stress
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
            affinity = float(np.exp(-expected_disagreement(
                probabilities[:, left], probabilities[:, right]
            ) / scale))
            rows.extend((left, right))
            cols.extend((right, left))
            data.extend((affinity, affinity))
        graph = coo_matrix((data, (rows, cols)), shape=(n_markers, n_markers))
    return graph


def _path_cost(order: IntArray, probabilities: FloatArray) -> float:
    return sum(
        expected_disagreement(probabilities[:, int(a)], probabilities[:, int(b)])
        for a, b in zip(order[:-1], order[1:])
    )


def _polish_order(order: IntArray, probabilities: FloatArray, passes: int = 2) -> IntArray:
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
        except Exception:
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
    adjacent = np.asarray([
        expected_disagreement(p[:, index], p[:, index + 1])
        for index in range(n_markers - 1)
    ])
    transition = float(np.clip(np.median(adjacent), 1e-4, 0.05))
    emissions = np.stack((1.0 - p, p), axis=2)
    emissions = np.clip(emissions, 1e-9, 1.0)
    alpha = np.empty((n_offspring, n_markers, 2), dtype=np.float64)
    beta = np.ones_like(alpha)
    alpha[:, 0] = 0.5 * emissions[:, 0]
    alpha[:, 0] /= alpha[:, 0].sum(axis=1, keepdims=True)
    for marker in range(1, n_markers):
        previous = alpha[:, marker - 1]
        predicted_zero = (1.0 - transition) * previous[:, 0] + transition * previous[:, 1]
        predicted_one = transition * previous[:, 0] + (1.0 - transition) * previous[:, 1]
        alpha[:, marker, 0] = emissions[:, marker, 0] * predicted_zero
        alpha[:, marker, 1] = emissions[:, marker, 1] * predicted_one
        alpha[:, marker] /= alpha[:, marker].sum(axis=1, keepdims=True)
    for marker in range(n_markers - 2, -1, -1):
        next_weighted = emissions[:, marker + 1] * beta[:, marker + 1]
        beta[:, marker, 0] = (
            (1.0 - transition) * next_weighted[:, 0]
            + transition * next_weighted[:, 1]
        )
        beta[:, marker, 1] = (
            transition * next_weighted[:, 0]
            + (1.0 - transition) * next_weighted[:, 1]
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
            if resample_offspring else np.arange(p.shape[0])
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
            if resample_offspring else np.arange(p.shape[0])
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
            _likelihood_mds_bootstrap_order_worker(
                ordering_payload(make_bootstrap())
            )
            for _ in range(replicates)
        ]
    else:
        orders: list[IntArray | None] = [None] * replicates
        pending: dict[int, Future[IntArray]] = {}
        maximum_pending = 2 * jobs
        with ProcessPoolExecutor(max_workers=jobs) as executor:
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
            index for index, anchor in enumerate(framework)
            if precedence[int(anchor), marker] >= one_sided_confidence
        ]
        upper = [
            index for index, anchor in enumerate(framework)
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
        if config[2] < 1 or config[3] < 1:
            raise ValueError("candidate dimensions and curve knots must be positive")
        if config not in normalized:
            normalized.append(config)
    if len(normalized) < 2:
        raise ValueError("at least two distinct candidate configurations are required")
    return tuple(normalized)


def fit_likelihood_mds_ensemble(
    probabilities: FloatArray,
    marker_names: Iterable[str] | None = None,
    *,
    candidate_configs: Iterable[LikelihoodMDSConfig] = (
        DEFAULT_LIKELIHOOD_MDS_CONFIGS
    ),
    stability_mass: float = 0.90,
    maximum_smacof_iterations: int = 500,
) -> LikelihoodMDSEnsembleResult:
    """Fit SoftMap's robust likelihood-MDS ensemble.

    Candidate orders are scored by the global correlation between inferred rank
    separation and pairwise recombination fraction. The unweighted score supplies
    the robust default. It is vetoed only when all nine prespecified LOD-weighted
    objectives select the same curve complexity within the very same embedding
    family. Rank bands summarize model sensitivity across every candidate and are
    deliberately labelled stability bands rather than nominal confidence bounds.
    """

    p = _validate_probabilities(probabilities)
    if not 0.0 < stability_mass < 1.0:
        raise ValueError("stability_mass must lie between zero and one")
    if maximum_smacof_iterations < 1:
        raise ValueError("maximum_smacof_iterations must be positive")
    configs = _validate_likelihood_mds_configs(candidate_configs)
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
    orders = np.asarray([
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
    ], dtype=np.int64)
    marker_count = p.shape[1]
    upper = np.triu_indices(marker_count, 1)
    pair_rf = recombination[upper]
    pair_lod = lod[upper]
    score_weights = _likelihood_mds_score_weights(pair_lod)
    uniform_scores = np.empty(len(configs), dtype=np.float64)
    weighted_scores = np.empty(
        (len(score_weights), len(configs)), dtype=np.float64
    )
    for candidate_index, order in enumerate(orders):
        positions = np.empty(marker_count, dtype=np.int64)
        positions[order] = np.arange(marker_count)
        separation = np.abs(
            positions[upper[0]] - positions[upper[1]]
        ) / max(marker_count - 1, 1)
        uniform_scores[candidate_index] = float(np.corrcoef(
            separation, pair_rf
        )[0, 1])
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
    selected_index = unanimous_index if veto else uniform_index

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
    pair_comparable = (
        (right[upper[0]] < left[upper[1]])
        | (right[upper[1]] < left[upper[0]])
    )
    rank_sd = np.std(candidate_positions, axis=0)
    preference_fraction = np.mean(
        candidate_positions[:, :, None] < candidate_positions[:, None, :],
        axis=0,
    )
    vote_margin = np.abs(2.0 * preference_fraction[upper] - 1.0)
    return LikelihoodMDSEnsembleResult(
        marker_names=names,
        order=orders[selected_index].copy(),
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
        stability_mass=stability_mass,
        stability_comparable_pair_fraction=float(np.mean(pair_comparable)),
        mean_normalized_rank_sd=float(
            np.mean(rank_sd) / max(marker_count - 1, 1)
        ),
        mean_pairwise_vote_margin=float(np.mean(vote_margin)),
        uncertain_pair_fraction_75=float(np.mean(vote_margin < 0.5)),
        mean_genotype_certainty=float(np.mean(np.abs(2.0 * p - 1.0))),
    )


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
        transition = float(np.clip(expected_disagreement(
            probabilities[:, int(order[marker - 1])],
            probabilities[:, int(order[marker])],
        ), 1e-5, 0.49))
        predicted = np.empty_like(alpha)
        predicted[:, 0] = (
            (1.0 - transition) * alpha[:, 0] + transition * alpha[:, 1]
        )
        predicted[:, 1] = (
            transition * alpha[:, 0] + (1.0 - transition) * alpha[:, 1]
        )
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

    transitions = np.asarray([
        np.clip(
            expected_disagreement(p[:, int(left)], p[:, int(right)]),
            1e-5,
            0.49,
        )
        for left, right in zip(order[:-1], order[1:])
    ], dtype=np.float64)

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
        predicted[:, 0] = (
            (1.0 - transition) * previous[:, 0] + transition * previous[:, 1]
        )
        predicted[:, 1] = (
            transition * previous[:, 0] + (1.0 - transition) * previous[:, 1]
        )
        raw = predicted * framework_emissions[:, marker]
        scale = raw.sum(axis=1)
        alpha[:, marker] = raw / np.clip(scale[:, None], 1e-300, None)
        prefix_log[:, marker] = (
            prefix_log[:, marker - 1] + np.log(np.clip(scale, 1e-300, None))
        )

    beta = np.ones_like(alpha)
    suffix_log = np.zeros((n_offspring, n_framework), dtype=np.float64)
    for marker in range(n_framework - 2, -1, -1):
        transition = transitions[marker]
        weighted = framework_emissions[:, marker + 1] * beta[:, marker + 1]
        raw = np.empty_like(weighted)
        raw[:, 0] = (
            (1.0 - transition) * weighted[:, 0] + transition * weighted[:, 1]
        )
        raw[:, 1] = (
            transition * weighted[:, 0] + (1.0 - transition) * weighted[:, 1]
        )
        scale = raw.sum(axis=1)
        beta[:, marker] = raw / np.clip(scale[:, None], 1e-300, None)
        suffix_log[:, marker] = (
            suffix_log[:, marker + 1] + np.log(np.clip(scale, 1e-300, None))
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
    candidate_emission = np.stack(
        (1.0 - p[:, candidate], p[:, candidate]), axis=1
    )
    candidate_emission = np.clip(candidate_emission, 1e-12, 1.0)
    n_framework = order.size

    scores = np.empty(n_framework + 1, dtype=np.float64)

    transition_right = float(np.clip(expected_disagreement(
        p[:, candidate], p[:, int(order[0])]
    ), 1e-5, 0.49))
    weighted_candidate = 0.5 * candidate_emission
    predicted_right = np.empty_like(weighted_candidate)
    predicted_right[:, 0] = (
        (1.0 - transition_right) * weighted_candidate[:, 0]
        + transition_right * weighted_candidate[:, 1]
    )
    predicted_right[:, 1] = (
        transition_right * weighted_candidate[:, 0]
        + (1.0 - transition_right) * weighted_candidate[:, 1]
    )
    bridge = np.sum(
        predicted_right * framework_emissions[:, 0] * beta[:, 0], axis=1
    )
    scores[0] = np.sum(suffix_log[:, 0] + np.log(np.clip(bridge, 1e-300, None)))

    for position in range(1, n_framework):
        left = int(order[position - 1])
        right = int(order[position])
        transition_left = float(np.clip(expected_disagreement(
            p[:, left], p[:, candidate]
        ), 1e-5, 0.49))
        transition_right = float(np.clip(expected_disagreement(
            p[:, candidate], p[:, right]
        ), 1e-5, 0.49))
        previous = alpha[:, position - 1]
        predicted_candidate = np.empty_like(previous)
        predicted_candidate[:, 0] = (
            (1.0 - transition_left) * previous[:, 0]
            + transition_left * previous[:, 1]
        )
        predicted_candidate[:, 1] = (
            transition_left * previous[:, 0]
            + (1.0 - transition_left) * previous[:, 1]
        )
        weighted_candidate = predicted_candidate * candidate_emission
        predicted_right = np.empty_like(weighted_candidate)
        predicted_right[:, 0] = (
            (1.0 - transition_right) * weighted_candidate[:, 0]
            + transition_right * weighted_candidate[:, 1]
        )
        predicted_right[:, 1] = (
            transition_right * weighted_candidate[:, 0]
            + (1.0 - transition_right) * weighted_candidate[:, 1]
        )
        bridge = np.sum(
            predicted_right
            * framework_emissions[:, position]
            * beta[:, position],
            axis=1,
        )
        scores[position] = np.sum(
            prefix_log[:, position - 1]
            + suffix_log[:, position]
            + np.log(np.clip(bridge, 1e-300, None))
        )

    transition_left = float(np.clip(expected_disagreement(
        p[:, int(order[-1])], p[:, candidate]
    ), 1e-5, 0.49))
    previous = alpha[:, -1]
    predicted_candidate = np.empty_like(previous)
    predicted_candidate[:, 0] = (
        (1.0 - transition_left) * previous[:, 0]
        + transition_left * previous[:, 1]
    )
    predicted_candidate[:, 1] = (
        transition_left * previous[:, 0]
        + (1.0 - transition_left) * previous[:, 1]
    )
    bridge = np.sum(predicted_candidate * candidate_emission, axis=1)
    scores[-1] = np.sum(
        prefix_log[:, -1] + np.log(np.clip(bridge, 1e-300, None))
    )
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
                end < mass.size
                and cumulative[end + 1] - cumulative[start] < confidence
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
            support = (
                float(scores[position]) - float(np.max(alternatives))
            ) / np.log(10.0)
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
        raise ValueError(
            "greedy_pass and support_priority_pass are mutually exclusive"
        )
    framework = list(map(int, scaffold))
    if len(framework) < 2:
        raise ValueError("likelihood densification requires at least two scaffold markers")
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
        while candidates and (
            max_additions is None or additions < max_additions
        ):
            current = np.asarray(framework, dtype=np.int64)
            initial_context = _prepare_hmm_insertion_context(p, current)
            priority: list[tuple[float, float, int]] = []
            for candidate in candidates:
                scores = _hmm_insertion_scores_prepared(
                    initial_context, candidate
                )
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
                scores = _hmm_insertion_scores_prepared(
                    current_context, candidate
                )
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
            if gap >= required_gap and terminal_allowed(
                best_position, framework_size
            ):
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
        raise ValueError("likelihood densification requires at least two scaffold markers")
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
                if resample_offspring else np.arange(p.shape[0])
            )
            boot = p[np.ix_(rows, columns)]
            if sample_states:
                boot = (replicate_rng.random(boot.shape) < boot).astype(np.float64)
                boot = boot * 0.998 + 0.001
            boot_scores = hmm_insertion_scores(
                boot, local_framework, local_candidate
            )
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
        ) if bin_threshold is None else bin_markers(
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
    representative_order = np.argsort(
        np.mean(positions, axis=0), kind="stable"
    ).astype(np.int64)
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
            support.bins.threshold / 2.0
            if marker_to_coarse_ratio >= 8.0
            else None
        )
    if (
        fine_bin_threshold is not None
        and fine_bin_threshold >= support.bins.threshold
    ):
        raise ValueError(
            "fine_bin_threshold must be below the selected coarse threshold "
            f"({support.bins.threshold:g})"
        )
    coarse_scaffold = (
        select_framework_global(
            support.representative_order,
            support.bootstrap_positions,
            confidence=scaffold_confidence,
        ) if global_scaffold else select_framework(
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
            random_seed=(
                None if random_seed is None else random_seed + 2_000_003
            ),
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
