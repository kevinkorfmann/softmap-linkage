from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import softmap

from softmap.core import (
    audit_scaffold_likelihood,
    auto_bin_markers,
    bootstrap_likelihood_mds_orders,
    bootstrap_placement_intervals,
    bootstrap_rank_intervals,
    densify_framework_likelihood,
    densify_framework_resampled_likelihood,
    expected_disagreement,
    fit_hierarchical_softmap,
    fit_softmap,
    framework_exact_support,
    hmm_insertion_scores,
    hmm_log_likelihood,
    hmm_placement_intervals,
    likelihood_weighted_mds_order,
    order_markers,
    pairwise_recombination_likelihood,
    placement_intervals,
    prune_framework_likelihood,
    select_framework_global,
)
from softmap.simulate import (
    bin_truth_coordinates,
    evaluate_marker_framework,
    evaluate_marker_coordinates,
    evaluate_marker_intervals,
    evaluate_marker_partial_order,
    evaluate_marker_rank_intervals,
    evaluate_result,
    simulate_backcross,
    truth_equivalence_membership,
)
from softmap.io import read_probability_tsv
from softmap.cli import main as cli_main


class SoftMapTests(unittest.TestCase):
    def test_small_public_api_returns_summary_and_order(self):
        data = softmap.demo(offspring=30, markers=20, seed=5)
        shuffled = data.shuffled(seed=2)
        self.assertNotEqual(shuffled.marker_names, data.marker_names)
        self.assertEqual(shuffled.probabilities.shape, data.probabilities.shape)
        mapping = softmap.fit(data, bootstrap=3, confidence=0.8, seed=8)
        self.assertEqual(mapping.summary()["markers"], 20)
        self.assertEqual(len(mapping.ordered_markers), 20)

        physical_data = softmap.LinkageData(
            data.probabilities,
            data.marker_names,
            data.reference_positions,
            "Chromosome 1",
            np.linspace(0.0, 10.0, 20),
        ).shuffled(seed=3)
        physical_map = softmap.fit(
            physical_data, bootstrap=3, confidence=0.8, seed=8
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "physical_grid.png"
            figure = softmap.plot_physical_order_grid(
                [physical_map, physical_map], destination
            )
            self.assertTrue(destination.exists())
            self.assertEqual(len(figure.axes), 4)

    def test_published_data_loader_accepts_local_source(self):
        text = (
            "ID,,,a,b,c\n"
            "m1,1,0.0,NN,NS,SS\n"
            "m2,1,1.5,SS,-,NN\n"
            "m3,2,0.0,NN,NN,NN\n"
        )
        with TemporaryDirectory() as directory:
            source = Path(directory) / "map.csv"
            source.write_text(text)
            data = softmap.contemporary_hybridization(
                chromosome=1, markers=2, source=source
            )
        np.testing.assert_allclose(
            data.probabilities,
            np.array([[0.01, 0.99], [0.5, 0.5], [0.99, 0.01]]),
        )
        np.testing.assert_allclose(data.reference_positions, [0.0, 1.5])

    def test_physical_map_loader_and_plot(self):
        text = (
            "ID,,,a,b\n"
            "chr1_1_0.5,1,0.0,NN,SS\n"
            "chr1_1_2.5,1,3.0,SS,NN\n"
            "chr2_2_1.0,2,0.0,NN,SS\n"
            "chr2_2_3.0,2,4.0,SS,NN\n"
        )
        with TemporaryDirectory() as directory:
            source = Path(directory) / "map.csv"
            source.write_text(text)
            positions = softmap.contemporary_map_positions(source=source)
            destination = Path(directory) / "physical_genetic.png"
            figure = softmap.plot_physical_vs_genetic(positions, destination)
            self.assertTrue(destination.exists())
            self.assertEqual(len(figure.axes), 2)
        np.testing.assert_allclose(positions.physical_mb, [0.5, 2.5, 1.0, 3.0])
        np.testing.assert_allclose(positions.genetic_cm, [0.0, 3.0, 0.0, 4.0])

    def test_simulator_retains_read_counts_for_raw_data_comparators(self):
        cross = simulate_backcross(
            n_offspring=12,
            n_markers=18,
            mean_depth=2.0,
            read_error=0.01,
            random_seed=7,
        )
        self.assertIsNotNone(cross.reference_reads)
        self.assertIsNotNone(cross.alternate_reads)
        assert cross.reference_reads is not None
        assert cross.alternate_reads is not None
        self.assertEqual(cross.reference_reads.shape, cross.probabilities.shape)
        self.assertEqual(cross.alternate_reads.shape, cross.probabilities.shape)
        depth = cross.reference_reads + cross.alternate_reads
        self.assertTrue(np.all(cross.probabilities[depth == 0] == 0.5))
        self.assertTrue(np.all(depth >= 0))

    def test_diploid_backcross_posteriors_match_aa_ab_read_model(self):
        read_error = 0.01
        cross = simulate_backcross(
            n_offspring=12,
            n_markers=18,
            mean_depth=3.0,
            read_error=read_error,
            heterozygous_state=True,
            random_seed=8,
        )
        assert cross.reference_reads is not None
        assert cross.alternate_reads is not None
        log_aa = (
            cross.alternate_reads * np.log(read_error)
            + cross.reference_reads * np.log1p(-read_error)
        )
        log_ab = (
            cross.reference_reads + cross.alternate_reads
        ) * np.log(0.5)
        maximum = np.maximum(log_aa, log_ab)
        expected = np.exp(log_ab - maximum) / (
            np.exp(log_aa - maximum) + np.exp(log_ab - maximum)
        )
        np.testing.assert_allclose(cross.probabilities, expected)
        self.assertEqual(cross.cross_design, "diploid_backcross")

    def test_expected_disagreement_respects_uncertainty(self):
        certain_zero = np.array([0.01, 0.01, 0.01])
        certain_one = np.array([0.99, 0.99, 0.99])
        uncertain = np.array([0.5, 0.5, 0.5])
        self.assertLess(expected_disagreement(certain_zero, certain_zero), 0.03)
        self.assertGreater(expected_disagreement(certain_zero, certain_one), 0.97)
        self.assertAlmostEqual(expected_disagreement(certain_zero, uncertain), 0.5)

    def test_small_simulation_recovers_order(self):
        cross = simulate_backcross(
            n_offspring=100,
            n_markers=120,
            mean_depth=8.0,
            random_seed=12,
        )
        result = fit_softmap(
            cross.probabilities,
            cross.marker_names,
            confidence=0.8,
            bootstrap_replicates=8,
            bin_threshold=0.005,
            neighbor_count=12,
            random_seed=9,
        )
        metrics = evaluate_result(result, cross)
        self.assertLess(metrics["representative_inversion_fraction"], 0.15)
        self.assertGreaterEqual(metrics["framework_markers"], 2)
        self.assertGreater(result.effective_offspring_information, 0.0)
        self.assertLessEqual(
            metrics["bounded_intervals"],
            metrics["bins"] - metrics["framework_markers"],
        )

    def test_fixed_seed_is_bitwise_reproducible(self):
        cross = simulate_backcross(
            n_offspring=30,
            n_markers=60,
            mean_depth=2.0,
            random_seed=101,
        )
        arguments = dict(
            probabilities=cross.probabilities,
            marker_names=cross.marker_names,
            confidence=0.8,
            bootstrap_replicates=5,
            bin_threshold=0.01,
            neighbor_count=20,
            random_seed=202,
        )
        first = fit_softmap(**arguments)
        second = fit_softmap(**arguments)
        np.testing.assert_array_equal(first.order, second.order)
        np.testing.assert_array_equal(
            first.representative_order, second.representative_order
        )
        np.testing.assert_array_equal(first.bootstrap_positions, second.bootstrap_positions)
        np.testing.assert_array_equal(first.precedence, second.precedence)
        np.testing.assert_array_equal(first.framework, second.framework)

    def test_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            fit_softmap(np.array([[0.0, 1.2], [0.2, 0.8]]))

    def test_bin_truth_uses_all_members(self):
        probabilities = np.array([
            [0.01, 0.01, 0.99],
            [0.99, 0.99, 0.01],
            [0.01, 0.01, 0.99],
        ])
        result = fit_softmap(
            probabilities,
            confidence=0.8,
            bootstrap_replicates=3,
            bin_threshold=0.03,
            random_seed=3,
        )
        coordinates = bin_truth_coordinates(result, np.array([10, 14, 20]))
        group = int(result.bins.membership[0])
        self.assertEqual(coordinates[group], 12.0)

    def test_common_truth_equivalence_evaluator_deduplicates_markers(self):
        cross = simulate_backcross(
            n_offspring=20,
            n_markers=80,
            mean_depth=2.0,
            random_seed=19,
            shuffle_markers=False,
        )
        membership = truth_equivalence_membership(cross.latent_states)
        metrics = evaluate_marker_framework(np.arange(80), cross)
        self.assertEqual(
            metrics["framework_truth_bins"],
            np.unique(membership).size,
        )
        self.assertAlmostEqual(
            metrics["framework_truth_bin_inversion_fraction"],
            0.0,
        )

    def test_common_evaluator_treats_reported_coordinate_ties_as_unresolved(self):
        cross = simulate_backcross(
            n_offspring=20,
            n_markers=80,
            mean_depth=2.0,
            random_seed=41,
            shuffle_markers=False,
        )
        membership = truth_equivalence_membership(cross.latent_states)
        representatives = np.asarray([
            int(np.flatnonzero(membership == group)[0])
            for group in np.unique(membership)
        ])
        representatives = representatives[np.argsort(
            cross.input_to_truth[representatives]
        )][:4]
        metrics = evaluate_marker_coordinates(
            representatives,
            np.array([0.0, 0.0, 1.0, 2.0]),
            cross,
        )
        self.assertEqual(metrics["framework_truth_bins"], 4)
        self.assertEqual(metrics["framework_reported_position_bins"], 3)
        self.assertEqual(metrics["framework_tied_truth_bin_pairs"], 1)
        self.assertEqual(metrics["framework_ordered_truth_bin_pairs"], 5)
        self.assertAlmostEqual(
            metrics["framework_truth_bin_inversion_fraction"], 0.0
        )

    def test_interval_evaluator_unions_split_truth_classes(self):
        cross = simulate_backcross(
            n_offspring=20,
            n_markers=80,
            mean_depth=2.0,
            random_seed=43,
            shuffle_markers=False,
        )
        marker_count = cross.probabilities.shape[1]
        truth_membership = truth_equivalence_membership(cross.latent_states)
        first = int(np.argmin(cross.input_to_truth))
        last = int(np.argmax(cross.input_to_truth))
        framework = np.array([first, last])
        left = np.zeros(marker_count, dtype=np.int64)
        right = np.ones(marker_count, dtype=np.int64)
        metrics = evaluate_marker_intervals(
            np.arange(marker_count),
            np.arange(marker_count),
            framework,
            left,
            right,
            cross,
        )
        self.assertEqual(
            metrics["bounded_nonframework_truth_bins"],
            metrics["nonframework_truth_bins"],
        )
        self.assertEqual(metrics["truth_bin_interval_coverage"], 1.0)

        interior_group = next(
            int(group) for group in np.unique(truth_membership)
            if group not in truth_membership[framework]
        )
        right[truth_membership == interior_group] = framework.size
        metrics = evaluate_marker_intervals(
            np.arange(marker_count),
            np.arange(marker_count),
            framework,
            left,
            right,
            cross,
        )
        self.assertEqual(metrics["unbounded_nonframework_truth_bins"], 1)

    def test_partial_order_evaluator_orders_only_disjoint_interval_hulls(self):
        cross = simulate_backcross(
            n_offspring=20,
            n_markers=80,
            mean_depth=2.0,
            random_seed=47,
            shuffle_markers=False,
        )
        truth_membership = truth_equivalence_membership(cross.latent_states)
        representatives = np.asarray([
            int(np.flatnonzero(truth_membership == group)[0])
            for group in np.unique(truth_membership)
        ])
        representatives = representatives[np.argsort(
            cross.input_to_truth[representatives]
        )]
        marker_to_bin = np.empty(cross.probabilities.shape[1], dtype=np.int64)
        for marker, truth_group in enumerate(truth_membership):
            marker_to_bin[marker] = int(np.flatnonzero(
                truth_membership[representatives] == truth_group
            )[0])
        framework = np.array([0, representatives.size - 1])
        left = np.zeros(representatives.size, dtype=np.int64)
        right = np.ones(representatives.size, dtype=np.int64)
        left[framework] = np.arange(framework.size)
        right[framework] = np.arange(framework.size)
        metrics = evaluate_marker_partial_order(
            marker_to_bin,
            representatives,
            framework,
            left,
            right,
            cross,
        )
        truth_bins = representatives.size
        self.assertEqual(metrics["partial_order_truth_bins"], truth_bins)
        self.assertEqual(
            metrics["partial_order_comparable_truth_bin_pairs"],
            2 * truth_bins - 3,
        )
        self.assertAlmostEqual(
            metrics["partial_order_truth_bin_inversion_fraction"], 0.0
        )
        self.assertEqual(metrics["all_interval_truth_bin_coverage"], 1.0)
        self.assertEqual(
            metrics["all_interval_informative_nonframework_truth_bins"],
            truth_bins - 2,
        )

        unresolved_bin = 1
        left[unresolved_bin] = -1
        right[unresolved_bin] = framework.size
        unresolved_metrics = evaluate_marker_partial_order(
            marker_to_bin,
            representatives,
            framework,
            left,
            right,
            cross,
        )
        self.assertEqual(
            unresolved_metrics[
                "all_interval_fully_unresolved_nonframework_truth_bins"
            ],
            1,
        )
        self.assertEqual(
            unresolved_metrics["partial_order_comparable_truth_bin_pairs"],
            metrics["partial_order_comparable_truth_bin_pairs"] - 2,
        )

        reversed_left = np.zeros(representatives.size, dtype=np.int64)
        reversed_right = np.ones(representatives.size, dtype=np.int64)
        reversed_left[framework] = np.array([1, 0])
        reversed_right[framework] = np.array([1, 0])
        reversed_metrics = evaluate_marker_partial_order(
            marker_to_bin,
            representatives,
            framework[::-1],
            reversed_left,
            reversed_right,
            cross,
        )
        self.assertEqual(
            reversed_metrics["partial_order_comparable_truth_bin_pairs"],
            metrics["partial_order_comparable_truth_bin_pairs"],
        )
        self.assertAlmostEqual(
            reversed_metrics["partial_order_truth_bin_inversion_fraction"], 0.0
        )

    def test_hmm_densification_recovers_clear_insertions(self):
        rng = np.random.default_rng(1)
        states = np.empty((1000, 6), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=states.shape[0])
        for marker in range(1, states.shape[1]):
            crossovers = rng.random(states.shape[0]) < 0.06
            states[:, marker] = states[:, marker - 1] ^ crossovers
        probabilities = states * 0.98 + 0.01
        framework = densify_framework_likelihood(
            probabilities,
            np.array([0, 5]),
            np.arange(6),
            min_log10_gap=3.0,
        )
        np.testing.assert_array_equal(framework, np.arange(6))

        support_priority = densify_framework_likelihood(
            probabilities,
            np.array([0, 5]),
            np.arange(6),
            min_log10_gap=3.0,
            support_priority_pass=True,
        )
        np.testing.assert_array_equal(support_priority, np.arange(6))

        resampled = densify_framework_resampled_likelihood(
            probabilities,
            np.array([0, 5]),
            np.arange(6),
            min_log10_gap=3.0,
            min_position_support=0.8,
            bootstrap_replicates=20,
            random_seed=5,
        )
        np.testing.assert_array_equal(resampled, np.arange(6))

        ensemble_order = order_markers(
            probabilities,
            ordering_ensemble=True,
        )
        self.assertEqual(set(ensemble_order), set(range(6)))

    def test_hmm_profile_intervals_preserve_supported_gaps(self):
        rng = np.random.default_rng(101)
        states = np.empty((1000, 6), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=states.shape[0])
        for marker in range(1, states.shape[1]):
            crossovers = rng.random(states.shape[0]) < 0.06
            states[:, marker] = states[:, marker - 1] ^ crossovers
        probabilities = states * 0.98 + 0.01
        left, right = hmm_placement_intervals(
            probabilities,
            np.array([0, 5]),
            confidence=0.8,
        )
        np.testing.assert_array_equal(left[[0, 5]], np.array([0, 1]))
        np.testing.assert_array_equal(right[[0, 5]], np.array([0, 1]))
        np.testing.assert_array_equal(left[1:5], np.zeros(4, dtype=np.int64))
        np.testing.assert_array_equal(right[1:5], np.ones(4, dtype=np.int64))
        padded_left, padded_right = hmm_placement_intervals(
            probabilities,
            np.array([0, 5]),
            confidence=0.8,
            padding=1,
        )
        np.testing.assert_array_equal(
            padded_left[1:5], -np.ones(4, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            padded_right[1:5], 2 * np.ones(4, dtype=np.int64)
        )
        with self.assertRaises(ValueError):
            hmm_placement_intervals(
                probabilities,
                np.array([0, 5]),
                temperature=0.0,
            )

    def test_likelihood_weighted_mds_recovers_a_clear_binary_map(self):
        rng = np.random.default_rng(103)
        states = np.empty((800, 12), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=states.shape[0])
        for marker in range(1, states.shape[1]):
            states[:, marker] = states[:, marker - 1] ^ (
                rng.random(states.shape[0]) < 0.04
            )
        probabilities = states * 0.998 + 0.001
        recombination, lod = pairwise_recombination_likelihood(probabilities)
        self.assertTrue(np.all(recombination >= 0.0))
        self.assertTrue(np.all(recombination < 0.5))
        self.assertGreater(float(np.median(lod[np.tril_indices(12, -1)])), 0.0)
        regularized, regularized_lod = pairwise_recombination_likelihood(
            probabilities,
            beta_prior_shape=1.5,
        )
        off_diagonal = np.tril_indices(12, -1)
        self.assertTrue(np.all(regularized[off_diagonal] > 0.0))
        self.assertTrue(np.all(
            regularized[off_diagonal] >= recombination[off_diagonal]
        ))
        self.assertTrue(np.all(
            regularized_lod[off_diagonal] <= lod[off_diagonal] + 1e-12
        ))
        order = likelihood_weighted_mds_order(
            recombination,
            lod,
            dimensions=5,
            maximum_smacof_iterations=200,
        )
        self.assertEqual(set(order), set(range(12)))
        self.assertLess(
            min(
                np.mean(np.abs(order - np.arange(12))),
                np.mean(np.abs(order[::-1] - np.arange(12))),
            ),
            1.0,
        )

    def test_likelihood_mds_bootstrap_is_reproducible_and_aligned(self):
        rng = np.random.default_rng(104)
        states = np.empty((120, 8), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=states.shape[0])
        for marker in range(1, states.shape[1]):
            states[:, marker] = states[:, marker - 1] ^ (
                rng.random(states.shape[0]) < 0.05
            )
        probabilities = states * 0.998 + 0.001
        reference = np.arange(states.shape[1], dtype=np.int64)
        first = bootstrap_likelihood_mds_orders(
            probabilities,
            reference,
            replicates=3,
            dimensions=4,
            maximum_smacof_iterations=100,
            random_seed=105,
        )
        second = bootstrap_likelihood_mds_orders(
            probabilities,
            reference,
            replicates=3,
            dimensions=4,
            maximum_smacof_iterations=100,
            jobs=2,
            random_seed=105,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, states.shape[1]))
        for positions in first:
            np.testing.assert_array_equal(np.sort(positions), reference)
            self.assertLess(
                np.mean(np.abs(positions - reference)),
                2.0,
            )

    def test_bootstrap_rank_intervals_and_evaluator_respect_orientation(self):
        cross = simulate_backcross(
            n_offspring=40,
            n_markers=20,
            mean_depth=8.0,
            random_seed=106,
        )
        truth_ranks = cross.input_to_truth.astype(np.int64)
        positions = np.tile(truth_ranks, (4, 1))
        left, right = bootstrap_rank_intervals(positions, confidence=0.75)
        np.testing.assert_array_equal(left, truth_ranks)
        np.testing.assert_array_equal(right, truth_ranks)
        metrics = evaluate_marker_rank_intervals(left, right, cross)
        self.assertEqual(metrics["rank_interval_truth_bin_coverage"], 1.0)
        self.assertEqual(
            metrics["rank_interval_truth_bin_inversion_fraction"], 0.0
        )
        reverse = cross.probabilities.shape[1] - 1 - truth_ranks
        reverse_metrics = evaluate_marker_rank_intervals(reverse, reverse, cross)
        self.assertEqual(reverse_metrics["rank_interval_truth_bin_coverage"], 1.0)
        self.assertEqual(
            reverse_metrics["rank_interval_truth_bin_inversion_fraction"], 0.0
        )

    def test_terminal_cap_limits_unbracketed_hmm_extensions(self):
        probabilities = np.full((12, 5), 0.25)

        def terminal_scores(context, candidate):
            return np.arange(context.order.size + 1, dtype=np.float64)

        with patch(
            "softmap.core._hmm_insertion_scores_prepared",
            side_effect=terminal_scores,
        ):
            framework = densify_framework_likelihood(
                probabilities,
                np.array([0, 1]),
                np.arange(5),
                min_log10_gap=0,
                max_terminal_additions_per_side=1,
                support_priority_pass=True,
            )
        self.assertEqual(framework.size, 3)
        np.testing.assert_array_equal(framework[:2], np.array([0, 1]))

        with self.assertRaises(ValueError):
            densify_framework_likelihood(
                probabilities,
                np.array([0, 1]),
                np.arange(5),
                max_terminal_additions_per_side=-1,
            )

    def test_likelihood_pruning_removes_an_uninformative_anchor(self):
        rng = np.random.default_rng(2)
        states = np.empty((1000, 6), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=states.shape[0])
        for marker in range(1, states.shape[1]):
            states[:, marker] = states[:, marker - 1] ^ (
                rng.random(states.shape[0]) < 0.06
            )
        probabilities = states * 0.98 + 0.01
        probabilities[:, 2] = 0.5
        pruned = prune_framework_likelihood(
            probabilities,
            np.arange(6),
            min_log10_gap=3.0,
        )
        self.assertNotIn(2, pruned)
        self.assertGreaterEqual(pruned.size, 2)

    def test_post_densification_audit_removes_a_misordered_scaffold_anchor(self):
        rng = np.random.default_rng(91)
        states = np.empty((1500, 6), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=states.shape[0])
        for marker in range(1, states.shape[1]):
            states[:, marker] = states[:, marker - 1] ^ (
                rng.random(states.shape[0]) < 0.05
            )
        probabilities = states * 0.98 + 0.01
        framework = np.array([0, 2, 1, 3, 4, 5])
        scaffold = np.array([0, 2, 1, 5])
        audited = audit_scaffold_likelihood(
            probabilities,
            framework,
            scaffold,
            min_log10_gap=0.0,
        )
        self.assertEqual(audited.size, 5)
        self.assertFalse({1, 2}.issubset(set(audited)))
        self.assertTrue(set(audited).issubset(set(framework)))

        with self.assertRaises(ValueError):
            audit_scaffold_likelihood(
                probabilities,
                framework,
                np.array([0, 99]),
            )

    def test_placement_interval_splits_two_sided_error_budget(self):
        precedence = np.zeros((3, 3), dtype=np.float64)
        precedence[0, 1] = 0.85
        precedence[1, 2] = 0.85
        left, right = placement_intervals(
            np.array([0, 2]), precedence, confidence=0.8
        )
        self.assertEqual(int(left[1]), -1)
        self.assertEqual(int(right[1]), 2)

        precedence[0, 1] = 0.9
        precedence[1, 2] = 0.9
        left, right = placement_intervals(
            np.array([0, 2]), precedence, confidence=0.8
        )
        self.assertEqual(int(left[1]), 0)
        self.assertEqual(int(right[1]), 1)

    def test_bootstrap_slot_interval_uses_shortest_supported_window(self):
        # Marker 1 lies between anchors in three of four maps and before them once.
        positions = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2],
        ])
        left, right = bootstrap_placement_intervals(
            np.array([0, 2]), positions, confidence=0.75
        )
        self.assertEqual((int(left[1]), int(right[1])), (0, 1))

        left, right = bootstrap_placement_intervals(
            np.array([0, 2]), positions, confidence=0.25
        )
        self.assertEqual((int(left[1]), int(right[1])), (-1, 0))
        with self.assertRaises(ValueError):
            bootstrap_placement_intervals(
                np.array([0, 2]), positions, confidence=0.0
            )

    def test_hierarchical_fit_maps_coarse_scaffold_into_fine_bins(self):
        cross = simulate_backcross(
            n_offspring=100,
            n_markers=40,
            mean_depth=5.0,
            random_seed=28,
        )
        result = fit_hierarchical_softmap(
            cross.probabilities,
            cross.marker_names,
            interval_confidence=0.8,
            scaffold_confidence=0.8,
            bootstrap_replicates=3,
            coarse_bin_threshold=0.03,
            fine_bin_threshold=0.005,
            neighbor_count=12,
            min_log10_gap=1.0,
            post_scaffold_log10_gap=0.0,
            max_additions=2,
            support_priority_pass=True,
            random_seed=7,
        )
        self.assertGreaterEqual(result.bins.representatives.size, result.scaffold.size)
        self.assertTrue(set(result.scaffold).issubset(set(result.framework)))
        self.assertEqual(len(result.framework_names()), result.framework.size)
        self.assertEqual(result.post_scaffold_log10_gap, 0.0)
        self.assertIn(
            result.status,
            {
                "ok",
                "limited_order_information",
                "insufficient_order_information",
            },
        )

    def test_hierarchical_auto_binning_preserves_a_finer_candidate_layer(self):
        base = simulate_backcross(
            n_offspring=30,
            n_markers=30,
            mean_depth=2.0,
            random_seed=31,
        )
        probabilities = np.repeat(base.probabilities, 20, axis=1)
        names = tuple(f"m{marker}" for marker in range(probabilities.shape[1]))
        result = fit_hierarchical_softmap(
            probabilities,
            names,
            bootstrap_replicates=3,
            coarse_bin_threshold=None,
            fine_bin_threshold=None,
            auto_fine_bins=True,
            neighbor_count=50,
            min_log10_gap=3.0,
            max_additions=1,
            greedy_pass=True,
            random_seed=8,
        )
        self.assertAlmostEqual(
            result.fine_bin_threshold,
            result.support.bins.threshold / 2.0,
        )
        self.assertGreaterEqual(
            result.bins.representatives.size,
            result.support.bins.representatives.size,
        )

    def test_probability_tsv_does_not_require_simulation_truth(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probabilities.tsv"
            path.write_text(
                "marker\toffspring_1\toffspring_2\n"
                "m1\t0.01\t0.99\n"
                "m2\t0.99\t0.01\n"
            )
            names, probabilities = read_probability_tsv(path)
        self.assertEqual(names, ("m1", "m2"))
        self.assertEqual(probabilities.shape, (2, 2))

    def test_basic_cli_completes_and_reports_information(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "probabilities.tsv"
            output_path = Path(directory) / "map.tsv"
            input_path.write_text(
                "marker\to1\to2\to3\to4\n"
                "m1\t0.01\t0.99\t0.01\t0.99\n"
                "m2\t0.02\t0.98\t0.02\t0.98\n"
                "m3\t0.99\t0.01\t0.99\t0.01\n"
            )
            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "softmap",
                    str(input_path),
                    str(output_path),
                    "--bootstrap",
                    "3",
                ],
            ), redirect_stdout(stdout):
                cli_main()
            summary = json.loads(stdout.getvalue())
            self.assertTrue(output_path.exists())
            self.assertGreater(summary["effective_offspring_information"], 0.0)

    def test_fast_hmm_insertion_scores_match_brute_force(self):
        rng = np.random.default_rng(44)
        probabilities = rng.uniform(0.01, 0.99, size=(25, 7))
        framework = np.array([0, 2, 4, 6])
        candidate = 3
        fast = hmm_insertion_scores(probabilities, framework, candidate)
        brute = []
        for position in range(framework.size + 1):
            trial = list(map(int, framework))
            trial.insert(position, candidate)
            brute.append(
                hmm_log_likelihood(probabilities, np.asarray(trial, dtype=np.int64))
            )
        np.testing.assert_allclose(fast, brute, rtol=1e-12, atol=1e-10)

    def test_hmm_densification_rejects_invalid_controls(self):
        probabilities = np.full((5, 4), 0.25)
        with self.assertRaises(ValueError):
            densify_framework_likelihood(
                probabilities,
                np.array([0, 3]),
                np.arange(4),
                min_log10_gap=-1,
            )
        with self.assertRaises(ValueError):
            densify_framework_likelihood(
                probabilities,
                np.array([0, 0]),
                np.arange(4),
            )
        with self.assertRaises(ValueError):
            densify_framework_likelihood(
                probabilities,
                np.array([0, 3]),
                np.arange(4),
                greedy_pass=True,
                support_priority_pass=True,
            )

    def test_global_framework_meets_complete_order_support(self):
        positions = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 2, 1],
        ])
        framework = select_framework_global(
            np.array([0, 1, 2]), positions, confidence=0.8
        )
        self.assertEqual(framework.size, 2)
        self.assertGreaterEqual(framework_exact_support(framework, positions), 0.8)

    def test_auto_binning_records_a_threshold_and_meets_target(self):
        cross = simulate_backcross(
            n_offspring=50,
            n_markers=120,
            mean_depth=1.0,
            random_seed=32,
        )
        bins = auto_bin_markers(
            cross.probabilities,
            neighbor_count=20,
            target_bins=60,
        )
        self.assertLessEqual(bins.representatives.size, 60)
        self.assertGreaterEqual(bins.threshold, 0.0)

    def test_auto_binning_selects_the_main_pattern_collapse(self):
        cross = simulate_backcross(
            n_offspring=100,
            n_markers=300,
            mean_depth=2.0,
            random_seed=10_000,
        )
        bins = auto_bin_markers(cross.probabilities, neighbor_count=200)
        self.assertEqual(bins.threshold, 0.01)
        self.assertEqual(bins.representatives.size, 131)

    def test_auto_binning_enforces_the_offspring_information_ceiling(self):
        cross = simulate_backcross(
            n_offspring=50,
            n_markers=300,
            mean_depth=1.0,
            random_seed=30_009,
        )
        bins = auto_bin_markers(cross.probabilities, neighbor_count=200)
        self.assertEqual(bins.threshold, 0.015)
        self.assertLessEqual(bins.representatives.size, 160)


if __name__ == "__main__":
    unittest.main()
