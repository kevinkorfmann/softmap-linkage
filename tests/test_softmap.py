import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import softmap
import softmap.core as core_module
from softmap.cli import main as cli_main
from softmap.core import (
    audit_scaffold_likelihood,
    auto_bin_markers,
    bootstrap_likelihood_mds_orders,
    bootstrap_placement_intervals,
    bootstrap_rank_intervals,
    densify_framework_likelihood,
    densify_framework_resampled_likelihood,
    estimate_genetic_map_distances,
    expected_disagreement,
    f2_pairwise_recombination_likelihood,
    fit_f2_likelihood_map,
    fit_hierarchical_softmap,
    fit_likelihood_mds_ensemble,
    fit_scalable_likelihood_mds_ensemble,
    fit_softmap,
    framework_exact_support,
    hmm_insertion_scores,
    hmm_log_likelihood,
    hmm_placement_intervals,
    likelihood_bin_markers,
    likelihood_weighted_mds_order,
    multipoint_adjacent_recombination,
    order_markers,
    pairwise_recombination_likelihood,
    pairwise_recombination_likelihood_edges,
    placement_intervals,
    prune_framework_likelihood,
    select_framework_global,
)
from softmap.io import read_probability_tsv
from softmap.simulate import (
    bin_truth_coordinates,
    evaluate_genetic_map_distances,
    evaluate_marker_coordinates,
    evaluate_marker_framework,
    evaluate_marker_intervals,
    evaluate_marker_partial_order,
    evaluate_marker_rank_intervals,
    evaluate_result,
    simulate_backcross,
    simulate_f2,
    truth_equivalence_membership,
)


class SoftMapTests(unittest.TestCase):
    def test_distance_evaluation_reports_success_and_abstention(self):
        cross = simulate_backcross(
            n_offspring=100,
            n_markers=30,
            mean_depth=2.0,
            random_seed=303,
        )
        order = np.argsort(cross.input_to_truth, kind="stable")
        distances = estimate_genetic_map_distances(cross.probabilities, order)
        metrics = evaluate_genetic_map_distances(
            distances,
            np.arange(30, dtype=np.int64),
            cross,
        )
        self.assertEqual(metrics["distance_status"], "ok")
        self.assertIsNotNone(metrics["representative_position_rmse_cm"])
        self.assertIsNotNone(metrics["adjacent_recombination_mean_absolute_error"])

        unresolved = replace(
            distances,
            bin_positions_cm=np.full(30, np.nan),
        )
        unresolved_metrics = evaluate_genetic_map_distances(
            unresolved,
            np.arange(30, dtype=np.int64),
            cross,
        )
        self.assertIsNone(unresolved_metrics["estimated_map_length_cm"])
        self.assertIsNone(unresolved_metrics["representative_position_rmse_cm"])
        with self.assertRaises(ValueError):
            evaluate_genetic_map_distances(distances, np.arange(29), cross)
        with self.assertRaises(ValueError):
            evaluate_genetic_map_distances(
                distances,
                np.append(np.arange(29), 30),
                cross,
            )

    def test_f2_read_likelihood_simulation_is_normalized_and_reproducible(self):
        arguments = {
            "n_offspring": 60,
            "n_markers": 18,
            "mean_depth": 2.0,
            "read_error": 0.01,
            "missing_probability": 0.10,
            "random_seed": 211,
        }
        first = simulate_f2(**arguments)
        second = simulate_f2(**arguments)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        np.testing.assert_array_equal(first.reference_reads, second.reference_reads)
        np.testing.assert_array_equal(first.alternate_reads, second.alternate_reads)
        np.testing.assert_allclose(np.sum(first.probabilities, axis=2), 1.0)
        self.assertIsNotNone(first.reference_reads)
        self.assertIsNotNone(first.alternate_reads)
        self.assertTrue(np.any(np.max(first.probabilities, axis=2) < 0.999))
        no_reads = (first.reference_reads + first.alternate_reads) == 0
        np.testing.assert_array_equal(
            first.probabilities[no_reads],
            np.tile(np.asarray((0.25, 0.50, 0.25)), (int(np.sum(no_reads)), 1)),
        )

    def test_f2_read_likelihood_simulation_validates_emission_controls(self):
        with self.assertRaises(ValueError):
            simulate_f2(mean_depth=-1.0)
        with self.assertRaises(ValueError):
            simulate_f2(mean_depth=2.0, read_error=0.5)

    def test_low_certainty_f2_uses_regularized_composite_distances(self):
        cross = simulate_f2(
            n_offspring=80,
            n_markers=40,
            mean_depth=2.0,
            read_error=0.01,
            missing_probability=0.05,
            random_seed=210,
        )
        result = fit_f2_likelihood_map(
            cross.probabilities,
            cross.marker_names,
            maximum_smacof_iterations=40,
        )
        self.assertLess(result.mean_genotype_certainty, 0.90)
        self.assertEqual(result.genetic_distances.status, "ok")
        self.assertTrue(
            result.genetic_distances.method.startswith(
                "f2_probabilistic_composite_kosambi_"
            )
        )

    def test_complete_f2_retains_exact_adjacent_distance_route(self):
        cross = simulate_f2(
            n_offspring=80,
            n_markers=20,
            random_seed=209,
        )
        result = fit_f2_likelihood_map(
            cross.probabilities,
            cross.marker_names,
            maximum_smacof_iterations=30,
        )
        self.assertEqual(
            result.genetic_distances.method,
            "f2_pairwise_kosambi_adjacent",
        )

    def test_complete_f2_likelihood_recovers_recombination_and_order(self):
        cross = simulate_f2(
            n_offspring=800,
            n_markers=28,
            map_length_morgan=0.8,
            random_seed=212,
        )
        recombination, lod = f2_pairwise_recombination_likelihood(cross.probabilities)
        truth_morgan = cross.true_positions[cross.input_to_truth]
        expected = 0.5 * (
            1.0 - np.exp(-2.0 * np.abs(truth_morgan[:, None] - truth_morgan[None, :]))
        )
        linked = expected < 0.35
        self.assertLess(
            float(np.mean(np.abs(recombination[linked] - expected[linked]))), 0.025
        )
        self.assertGreater(float(np.max(lod)), 10.0)

        physical = 100.0 * truth_morgan
        result = fit_f2_likelihood_map(
            cross.probabilities,
            cross.marker_names,
            physical_positions=physical,
            maximum_smacof_iterations=80,
        )
        inferred_rank = np.empty(result.order.size, dtype=np.int64)
        inferred_rank[result.order] = np.arange(result.order.size)
        self.assertGreater(
            abs(float(np.corrcoef(inferred_rank, cross.input_to_truth)[0, 1])),
            0.97,
        )
        self.assertEqual(result.genetic_distances.status, "ok")

    def test_f2_public_api_can_use_an_explicit_physical_scaffold(self):
        cross = simulate_f2(
            n_offspring=300,
            n_markers=24,
            random_seed=213,
        )
        physical = 100.0 * cross.true_positions[cross.input_to_truth]
        data = softmap.F2LinkageData(
            cross.probabilities,
            cross.marker_names,
            physical_positions=physical,
        )
        mapping = softmap.fit_f2(
            data,
            use_physical_scaffold=True,
            maximum_smacof_iterations=60,
        )
        self.assertTrue(mapping.summary()["physical_scaffold_used"])
        self.assertEqual(mapping.summary()["distance_status"], "ok")
        ranks = np.asarray([row["order_rank"] for row in mapping.marker_table()])
        np.testing.assert_array_equal(
            np.argsort(ranks), np.argsort(physical, kind="stable")
        )

    def test_composite_distance_recovers_known_markov_map_without_inflation(self):
        rng = np.random.default_rng(141)
        offspring = 800
        markers = 35
        transition = 0.025
        states = np.empty((offspring, markers), dtype=np.int64)
        states[:, 0] = rng.integers(0, 2, size=offspring)
        for marker in range(1, markers):
            states[:, marker] = states[:, marker - 1] ^ (
                rng.random(offspring) < transition
            )
        probabilities = states * 0.998 + 0.001
        distances = estimate_genetic_map_distances(
            probabilities,
            segment_count=8,
        )
        expected_length_cm = -50.0 * (markers - 1) * np.log(1.0 - 2.0 * transition)

        self.assertEqual(distances.status, "ok")
        self.assertEqual(
            core_module.DISTANCE_RANK_SPAN_WEIGHT_EXPONENT,
            -0.125,
        )
        self.assertIsNotNone(distances.map_length_cm)
        self.assertIsNotNone(distances.composite_median_absolute_residual_morgan)
        self.assertIsNotNone(distances.composite_p90_absolute_residual_morgan)
        self.assertLessEqual(
            float(distances.composite_median_absolute_residual_morgan),
            float(distances.composite_p90_absolute_residual_morgan),
        )
        self.assertLess(
            abs(float(distances.map_length_cm) / expected_length_cm - 1.0),
            0.20,
        )
        self.assertTrue(
            np.all(np.diff(distances.bin_positions_cm[distances.ordered_bins]) >= 0.0)
        )
        self.assertTrue(
            np.all(
                (distances.adjacent_recombination >= 0.0)
                & (distances.adjacent_recombination < 0.5)
            )
        )
        self.assertEqual(
            distances.adjacent_pairwise_recombination.shape,
            (markers - 1,),
        )
        self.assertEqual(
            distances.adjacent_multipoint_recombination.shape,
            (markers - 1,),
        )
        self.assertEqual(
            distances.adjacent_local_recombination.shape,
            (markers - 1,),
        )
        np.testing.assert_allclose(
            distances.adjacent_multipoint_recombination,
            multipoint_adjacent_recombination(
                probabilities,
                np.arange(markers, dtype=np.int64),
                distances.adjacent_pairwise_recombination,
            ),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            distances.adjacent_local_recombination,
            0.65 * distances.adjacent_multipoint_recombination
            + 0.35 * distances.adjacent_recombination,
            rtol=1e-15,
            atol=1e-15,
        )

    def test_composite_distance_abstains_without_informative_pair_scale(self):
        rng = np.random.default_rng(142)
        column = rng.choice((0.001, 0.999), size=(100, 1))
        probabilities = np.repeat(column, 20, axis=1)
        distances = estimate_genetic_map_distances(probabilities)

        self.assertEqual(
            distances.status,
            "insufficient_distance_information",
        )
        self.assertIsNone(distances.map_length_cm)
        self.assertTrue(np.all(np.isnan(distances.marker_positions_cm)))

    def test_composite_distance_handles_unlinked_adjacent_pair_boundary(self):
        rng = np.random.default_rng(143)
        probabilities = rng.uniform(0.45, 0.55, size=(100, 20))
        distances = estimate_genetic_map_distances(probabilities)

        self.assertTrue(np.all(distances.adjacent_pairwise_recombination < 0.5))

    def test_automatic_posterior_calibration_is_auditable_and_optional(self):
        cross = simulate_backcross(
            n_offspring=100,
            n_markers=40,
            mean_depth=2.0,
            read_error=0.01,
            contamination=0.10,
            heterozygous_state=True,
            random_seed=777,
        )
        uncalibrated = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            maximum_smacof_iterations=20,
            automatic_posterior_calibration=False,
        )
        calibrated = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            maximum_smacof_iterations=20,
        )

        self.assertFalse(uncalibrated.posterior_calibration_triggered)
        self.assertEqual(
            uncalibrated.selection_method,
            "fixed_inconsistent_high_information_penalized_curve",
        )
        self.assertTrue(calibrated.posterior_calibration_triggered)
        self.assertEqual(calibrated.posterior_calibration_temperature, 1.15)
        self.assertEqual(
            calibrated.selected_config,
            core_module.DENSE_CALIBRATED_HIGH_INFORMATION_CONFIG,
        )
        self.assertEqual(
            calibrated.selection_method,
            "fixed_calibrated_high_information_penalized_curve",
        )
        self.assertEqual(
            calibrated.penalized_curve_effective_degrees_of_freedom,
            core_module.DENSE_CALIBRATED_HIGH_INFORMATION_PENALIZED_CURVE_EDF,
        )
        self.assertEqual(
            calibrated.uncalibrated_mean_genotype_certainty,
            uncalibrated.mean_genotype_certainty,
        )
        assert uncalibrated.genetic_distances is not None
        uncalibrated_residual = (
            uncalibrated.genetic_distances.composite_median_absolute_residual_morgan
        )
        self.assertIsNotNone(
            calibrated.uncalibrated_distance_median_absolute_residual_morgan
        )
        self.assertNotEqual(
            calibrated.uncalibrated_distance_median_absolute_residual_morgan,
            uncalibrated_residual,
        )
        self.assertLess(
            calibrated.mean_genotype_certainty,
            uncalibrated.mean_genotype_certainty,
        )

    def test_supported_stability_fallback_recovers_informative_rank_bands(self):
        cross = simulate_backcross(
            n_offspring=50,
            n_markers=40,
            mean_depth=2.0,
            read_error=0.01,
            contamination=0.10,
            heterozygous_state=True,
            random_seed=760,
        )
        result = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            maximum_smacof_iterations=20,
        )

        self.assertTrue(result.posterior_calibration_triggered)
        self.assertTrue(result.weighted_objective_support_filter_applied)
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(
            result.stability_comparable_pair_fraction,
            result.minimum_stability_comparable_pair_fraction,
        )
        self.assertEqual(
            result.penalized_curve_effective_degrees_of_freedom,
            core_module.DENSE_CALIBRATED_HIGH_INFORMATION_PENALIZED_CURVE_EDF,
        )

    def test_scalable_distance_assigns_exact_ties_within_likelihood_bins(self):
        cross = simulate_backcross(
            n_offspring=200,
            n_markers=15,
            mean_depth=4.0,
            heterozygous_state=True,
            random_seed=143,
        )
        probabilities = np.repeat(cross.probabilities, 3, axis=1)
        result = fit_scalable_likelihood_mds_ensemble(
            probabilities,
            maximum_dense_markers=20,
            minimum_bin_linkage_lod=3.0,
            bin_neighbor_count=20,
            maximum_smacof_iterations=20,
        )
        distances = result.genetic_distances

        self.assertIsNotNone(distances)
        assert distances is not None
        self.assertEqual(distances.status, "ok")
        for group in range(result.likelihood_bin_count):
            members = result.bin_membership == group
            self.assertEqual(
                np.unique(distances.marker_positions_cm[members]).size,
                1,
            )

    def test_likelihood_mds_ensemble_returns_auditable_stability_bands(self):
        cross = simulate_backcross(
            n_offspring=40,
            n_markers=25,
            mean_depth=2.0,
            heterozygous_state=True,
            random_seed=101,
        )
        configs = (
            ("rf", 1.0, 5, 1),
            ("haldane", 1.0, 5, 1),
            ("haldane", 2.0, 8, 2),
        )
        result = fit_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            candidate_configs=configs,
            stability_mass=0.8,
            maximum_smacof_iterations=20,
        )
        np.testing.assert_array_equal(np.sort(result.order), np.arange(25))
        np.testing.assert_array_equal(np.sort(result.preliminary_order), np.arange(25))
        self.assertEqual(result.candidate_orders.shape, (3, 25))
        self.assertEqual(result.candidate_positions.shape, (3, 25))
        self.assertEqual(result.weighted_scores.shape, (9, 3))
        self.assertTrue(np.all(result.interval_left <= result.interval_right))
        self.assertIn(result.selected_config, configs)
        self.assertIn(
            result.status,
            {"ok", "limited_order_information", "insufficient_order_information"},
        )

    def test_likelihood_mds_ensemble_reuses_each_unique_embedding(self):
        cross = simulate_backcross(
            n_offspring=35,
            n_markers=18,
            mean_depth=2.0,
            heterozygous_state=True,
            random_seed=109,
        )
        configs = (
            ("rf", 1.0, 5, 1),
            ("rf", 1.0, 5, 2),
            ("rf", 1.0, 5, 3),
            ("haldane", 1.0, 5, 1),
        )
        recombination, lod = pairwise_recombination_likelihood(cross.probabilities)
        expected = np.asarray(
            [
                likelihood_weighted_mds_order(
                    recombination,
                    lod,
                    distance=distance,
                    lod_exponent=lod_exponent,
                    dimensions=dimensions,
                    principal_curve_knots=curve_knots,
                    maximum_smacof_iterations=20,
                )
                for distance, lod_exponent, dimensions, curve_knots in configs
            ]
        )
        with (
            patch(
                "softmap.core._likelihood_weighted_mds_coordinates",
                wraps=core_module._likelihood_weighted_mds_coordinates,
            ) as embedding_fit,
            patch(
                "softmap.core.linalg.pinvh",
                wraps=core_module.linalg.pinvh,
            ) as inverse_fit,
        ):
            result = fit_likelihood_mds_ensemble(
                cross.probabilities,
                cross.marker_names,
                candidate_configs=configs,
                maximum_posterior_refinement_passes=0,
                maximum_smacof_iterations=20,
            )
        self.assertEqual(embedding_fit.call_count, 2)
        self.assertEqual(inverse_fit.call_count, 1)
        np.testing.assert_array_equal(result.candidate_orders, expected)

    def test_scalable_likelihood_mds_uses_dense_high_information_geometry(self):
        cross = simulate_backcross(
            n_offspring=35,
            n_markers=18,
            mean_depth=2.0,
            heterozygous_state=True,
            random_seed=110,
        )
        arguments = {
            "maximum_posterior_refinement_passes": 0,
            "maximum_smacof_iterations": 20,
        }
        dense = fit_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            candidate_configs=(
                *core_module.DEFAULT_LIKELIHOOD_MDS_CONFIGS,
                core_module.DENSE_HIGH_INFORMATION_CONFIG,
            ),
            selected_config=core_module.DENSE_HIGH_INFORMATION_CONFIG,
            **arguments,
        )
        scalable = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            maximum_dense_markers=20,
            **arguments,
        )
        np.testing.assert_array_equal(scalable.order, dense.order)
        np.testing.assert_array_equal(
            scalable.candidate_positions, dense.candidate_positions
        )
        np.testing.assert_array_equal(scalable.interval_left, dense.interval_left)
        np.testing.assert_array_equal(
            scalable.reported_positions, dense.reported_positions
        )
        self.assertEqual(scalable.binning_method, "none")
        self.assertEqual(
            scalable.selection_method,
            "fixed_high_information_geometry",
        )
        self.assertFalse(scalable.weighted_objective_support_filter_applied)

    def test_likelihood_mds_supports_penalized_principal_curve_candidate(self):
        cross = simulate_backcross(
            n_offspring=35,
            n_markers=18,
            mean_depth=1.5,
            heterozygous_state=True,
            random_seed=145,
        )
        configs = (
            ("haldane", 2.0, 10, 0),
            ("haldane", 2.0, 10, 1),
        )
        result = fit_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            candidate_configs=configs,
            selected_config=configs[0],
            maximum_posterior_refinement_passes=0,
            maximum_smacof_iterations=20,
        )
        self.assertEqual(result.selected_config, configs[0])
        np.testing.assert_array_equal(
            np.sort(result.order),
            np.arange(cross.probabilities.shape[1]),
        )

    def test_scalable_likelihood_mds_preserves_low_certainty_dense_selector(self):
        cross = simulate_backcross(
            n_offspring=35,
            n_markers=18,
            mean_depth=0.75,
            heterozygous_state=True,
            random_seed=144,
        )
        self.assertLess(
            float(np.mean(np.abs(2.0 * cross.probabilities - 1.0))),
            core_module.DENSE_HIGH_INFORMATION_CERTAINTY_THRESHOLD,
        )
        arguments = {
            "maximum_posterior_refinement_passes": 0,
            "maximum_smacof_iterations": 20,
        }
        dense = fit_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            **arguments,
        )
        scalable = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            maximum_dense_markers=20,
            **arguments,
        )
        np.testing.assert_array_equal(scalable.order, dense.order)
        self.assertEqual(
            scalable.selection_method,
            "global_rf_correlation_with_veto",
        )

    def test_scalable_likelihood_mds_uses_moderate_information_geometry(self):
        cross = simulate_backcross(
            n_offspring=35,
            n_markers=18,
            mean_depth=1.2,
            heterozygous_state=True,
            random_seed=144,
        )
        certainty = float(np.mean(np.abs(2.0 * cross.probabilities - 1.0)))
        self.assertGreaterEqual(
            certainty,
            core_module.DENSE_MODERATE_INFORMATION_CERTAINTY_THRESHOLD,
        )
        self.assertLess(
            certainty,
            core_module.DENSE_HIGH_INFORMATION_CERTAINTY_THRESHOLD,
        )
        arguments = {
            "maximum_posterior_refinement_passes": 0,
            "maximum_smacof_iterations": 20,
        }
        dense = fit_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            candidate_configs=(
                *core_module.DEFAULT_LIKELIHOOD_MDS_CONFIGS,
                core_module.DENSE_MODERATE_INFORMATION_CONFIG,
            ),
            selected_config=core_module.DENSE_MODERATE_INFORMATION_CONFIG,
            penalized_curve_effective_degrees_of_freedom=(
                core_module.DENSE_MODERATE_INFORMATION_PENALIZED_CURVE_EDF
            ),
            **arguments,
        )
        scalable = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            maximum_dense_markers=20,
            **arguments,
        )
        np.testing.assert_array_equal(scalable.order, dense.order)
        self.assertEqual(
            scalable.selected_config,
            core_module.DENSE_MODERATE_INFORMATION_CONFIG,
        )
        self.assertEqual(
            scalable.selection_method,
            "fixed_moderate_information_penalized_curve",
        )
        self.assertEqual(
            scalable.penalized_curve_effective_degrees_of_freedom,
            core_module.DENSE_MODERATE_INFORMATION_PENALIZED_CURVE_EDF,
        )

    def test_scalable_likelihood_mds_returns_tied_likelihood_bins(self):
        cross = simulate_backcross(
            n_offspring=80,
            n_markers=15,
            mean_depth=4.0,
            heterozygous_state=True,
            random_seed=112,
        )
        probabilities = np.repeat(cross.probabilities, 3, axis=1)
        names = tuple(f"m{index}" for index in range(probabilities.shape[1]))
        result = fit_scalable_likelihood_mds_ensemble(
            probabilities,
            names,
            maximum_dense_markers=20,
            bin_neighbor_count=15,
            maximum_posterior_refinement_passes=0,
            maximum_smacof_iterations=20,
        )
        np.testing.assert_array_equal(
            np.sort(result.order), np.arange(probabilities.shape[1])
        )
        self.assertEqual(result.binning_method, "likelihood")
        self.assertEqual(result.selected_config, ("haldane", 3.0, 10, 1))
        self.assertEqual(result.selection_method, "fixed_high_information_geometry")
        self.assertEqual(result.minimum_bin_linkage_lod, 9.0)
        self.assertEqual(result.posterior_refinement_passes_applied, 0)
        self.assertLess(result.likelihood_bin_count, probabilities.shape[1])
        self.assertLess(
            np.unique(result.reported_positions).size,
            probabilities.shape[1],
        )
        self.assertTrue(np.all(result.interval_left <= result.interval_right))

    def test_scalable_likelihood_mds_uses_landmarks_above_bin_limit(self):
        cross = simulate_backcross(
            n_offspring=60,
            n_markers=36,
            mean_depth=3.0,
            heterozygous_state=True,
            random_seed=113,
        )
        result = fit_scalable_likelihood_mds_ensemble(
            cross.probabilities,
            cross.marker_names,
            maximum_dense_markers=20,
            maximum_dense_likelihood_bins=8,
            maximum_likelihood_landmarks=12,
            landmark_neighbor_count=4,
            maximum_smacof_iterations=20,
        )
        np.testing.assert_array_equal(
            np.sort(result.order), np.arange(cross.probabilities.shape[1])
        )
        self.assertEqual(result.ordering_method, "landmark_likelihood_mds")
        self.assertEqual(result.binning_method, "likelihood_landmark")
        self.assertEqual(result.landmark_count, 12)
        self.assertEqual(result.landmark_neighbor_count, 4)
        self.assertEqual(result.bin_neighbor_count, 32)
        self.assertIsNone(result.bin_neighbor_projection_dimensions)
        self.assertIsNone(result.landmark_support_exponent)
        self.assertEqual(result.selected_config, ("haldane", 1.0, 20, 1))
        self.assertEqual(result.selection_method, "fixed_landmark_geometry")
        self.assertEqual(result.posterior_refinement_passes_applied, 0)

        with patch(
            "softmap.core.likelihood_bin_markers",
            wraps=likelihood_bin_markers,
        ) as binning:
            weighted = fit_scalable_likelihood_mds_ensemble(
                cross.probabilities,
                cross.marker_names,
                maximum_dense_markers=20,
                maximum_dense_likelihood_bins=8,
                maximum_likelihood_landmarks=12,
                bin_neighbor_projection_minimum_markers=3,
                large_scale_minimum_markers=3,
                landmark_support_weighting_minimum_markers=3,
                landmark_neighbor_count=4,
                maximum_smacof_iterations=20,
            )
        self.assertEqual(binning.call_args.kwargs["minimum_linkage_lod"], 3.0)
        self.assertEqual(binning.call_args.kwargs["neighbor_count"], 64)
        self.assertEqual(
            binning.call_args.kwargs["neighbor_projection_dimensions"],
            12,
        )
        self.assertEqual(
            weighted.selection_method,
            "fixed_support_weighted_landmark_geometry",
        )
        self.assertEqual(weighted.bin_neighbor_count, 64)
        self.assertEqual(weighted.bin_neighbor_projection_dimensions, 12)
        self.assertEqual(weighted.landmark_support_exponent, 0.5)

    def test_support_weighted_landmark_selection_is_validated(self):
        rng = np.random.default_rng(114)
        probabilities = rng.uniform(0.01, 0.99, size=(20, 30))
        support = np.ones(30, dtype=np.float64)
        support[7] = 1e6
        selected = core_module._select_likelihood_landmarks(
            probabilities,
            10,
            support=support,
            support_exponent=0.5,
        )
        self.assertEqual(int(selected[0]), 7)
        self.assertEqual(np.unique(selected).size, 10)
        with self.assertRaisesRegex(ValueError, "support"):
            core_module._select_likelihood_landmarks(
                probabilities,
                10,
                support=np.ones(29),
            )

    def test_large_scale_curve_rescue_uses_stability_and_certainty(self):
        self.assertIsNone(core_module._large_scale_rescue_config(0.5, 0.2, 0.14))
        self.assertEqual(
            core_module._large_scale_rescue_config(0.24, 0.47, 0.0),
            core_module.LARGE_SCALE_LOW_CERTAINTY_RESCUE_CONFIG,
        )
        self.assertEqual(
            core_module._large_scale_rescue_config(0.25, 0.47, 0.0),
            core_module.LARGE_SCALE_LOW_CERTAINTY_MODERATE_RESCUE_CONFIG,
        )
        self.assertEqual(
            core_module._large_scale_rescue_config(0.49, 0.48, 0.0),
            core_module.LARGE_SCALE_MODERATE_CERTAINTY_RESCUE_CONFIG,
        )
        self.assertEqual(
            core_module._large_scale_rescue_config(0.5, 0.47, 0.15),
            core_module.LARGE_SCALE_LOW_CERTAINTY_UNCERTAIN_RESCUE_CONFIG,
        )
        self.assertIsNone(core_module._large_scale_rescue_config(0.5, 0.48, 0.2))

    def test_large_scale_stability_mass_cap_is_low_certainty_only(self):
        self.assertEqual(
            core_module._large_scale_stability_mass(0.9, True, 0.47),
            (0.8, True),
        )
        self.assertEqual(
            core_module._large_scale_stability_mass(0.8, True, 0.47),
            (0.8, False),
        )
        self.assertEqual(
            core_module._large_scale_stability_mass(0.9, True, 0.48),
            (0.9, False),
        )
        self.assertEqual(
            core_module._large_scale_stability_mass(0.9, False, 0.2),
            (0.9, False),
        )

    def test_likelihood_public_api_exposes_selected_order_and_stability(self):
        data = softmap.demo(offspring=30, markers=15, seed=17)
        mapping = softmap.fit_likelihood(
            data,
            stability_mass=0.9,
            maximum_smacof_iterations=20,
        )
        self.assertEqual(len(mapping.ordered_markers), 15)
        self.assertEqual(len(mapping.marker_table()), 15)
        self.assertEqual(mapping.summary()["candidate_orders"], 11)
        self.assertEqual(mapping.summary()["binning_method"], "none")
        self.assertEqual(mapping.summary()["likelihood_bins"], 15)
        self.assertIsNone(mapping.summary()["bin_neighbor_count"])
        self.assertIsNone(mapping.summary()["bin_neighbor_projection_dimensions"])
        self.assertIsNone(mapping.summary()["landmark_support_exponent"])
        self.assertFalse(mapping.summary()["large_scale_rescue_triggered"])
        self.assertFalse(mapping.summary()["low_certainty_stability_mass_cap_applied"])
        self.assertIn(
            "posterior_calibration_triggered",
            mapping.summary(),
        )
        self.assertIn(
            "posterior_calibration_temperature",
            mapping.summary(),
        )
        self.assertIn(
            "uncalibrated_distance_median_absolute_residual_morgan",
            mapping.summary(),
        )
        self.assertEqual(mapping.summary()["posterior_refinement_weight"], 0.75)
        self.assertEqual(mapping.summary()["posterior_refinement_passes_applied"], 0)
        self.assertEqual(
            mapping.summary()["selection_method"],
            "fixed_high_information_geometry",
        )
        self.assertFalse(mapping.summary()["weighted_objective_support_filter_applied"])
        self.assertEqual(
            mapping.summary()["second_refinement_uncertain_pair_threshold"],
            0.03,
        )
        self.assertEqual(mapping.summary()["stability_rank_padding"], 1)
        self.assertEqual(
            mapping.summary()["minimum_stability_comparable_pair_fraction"],
            0.35,
        )
        self.assertIn(
            mapping.summary()["distance_status"],
            {
                "ok",
                "insufficient_distance_information",
            },
        )
        self.assertIn("map_length_cm", mapping.summary())
        self.assertIn("genetic_position_cm", mapping.marker_table()[0])

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
        physical_map = softmap.fit(physical_data, bootstrap=3, confidence=0.8, seed=8)
        physical_distance_map = softmap.fit_likelihood(
            physical_data, maximum_smacof_iterations=50
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "physical_grid.png"
            figure = softmap.plot_physical_order_grid(
                [physical_map, physical_map], destination
            )
            self.assertTrue(destination.exists())
            self.assertEqual(len(figure.axes), 4)
            output_destination = Path(directory) / "physical_outputs.png"
            output_figure = softmap.plot_physical_output_grid(
                [physical_map, physical_map],
                [physical_distance_map, physical_distance_map],
                output_destination,
            )
            self.assertTrue(output_destination.exists())
            self.assertEqual(len(output_figure.axes), 4)

    def test_published_data_loader_accepts_local_source(self):
        text = "ID,,,a,b,c\nm1,1,0.0,NN,NS,SS\nm2,1,1.5,SS,-,NN\nm3,2,0.0,NN,NN,NN\n"
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

    def test_published_f2_loader_retains_heterozygotes(self):
        text = "ID,,,a,b,c\nchr1_1_0.5,1,0.0,NN,NS,SS\nchr1_1_2.5,1,3.0,SS,-,NN\n"
        with TemporaryDirectory() as directory:
            source = Path(directory) / "map.csv"
            source.write_text(text)
            data = softmap.contemporary_hybridization_f2(chromosome=1, source=source)
        self.assertIsInstance(data, softmap.F2LinkageData)
        np.testing.assert_allclose(data.probabilities[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(data.probabilities[1, 0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(data.probabilities[1, 1], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(data.physical_positions, [0.5, 2.5])

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

    def test_grav2_loader_accepts_local_sources(self):
        genotypes = "id,m1,m2,m3\n1,L,C,-\n2,C,L,L\n"
        genetic_map = "marker,chr,pos\nm1,1,0\nm2,1,5\nm3,2,0\n"
        with TemporaryDirectory() as directory:
            genotype_source = Path(directory) / "geno.csv"
            map_source = Path(directory) / "map.csv"
            genotype_source.write_text(genotypes)
            map_source.write_text(genetic_map)
            data = softmap.grav2_ril(
                chromosome=1,
                genotype_source=genotype_source,
                map_source=map_source,
            )
        np.testing.assert_allclose(data.probabilities, [[0.01, 0.99], [0.99, 0.01]])
        np.testing.assert_allclose(data.reference_positions, [0.0, 5.0])

    def test_hyper_loader_accepts_local_sources(self):
        genotypes = "0\t1\t9\n1\t0\t1\n"
        genetic_map = "m1\t0.0\nm2\t4.0\nm3\t0.0\n"
        chromosomes = "1\n1\n2\n"
        with TemporaryDirectory() as directory:
            genotype_source = Path(directory) / "geno.txt"
            map_source = Path(directory) / "map.txt"
            chromosome_source = Path(directory) / "chr.txt"
            genotype_source.write_text(genotypes)
            map_source.write_text(genetic_map)
            chromosome_source.write_text(chromosomes)
            data = softmap.hyper_backcross(
                chromosome=1,
                genotype_source=genotype_source,
                map_source=map_source,
                chromosome_source=chromosome_source,
            )
        np.testing.assert_allclose(data.probabilities, [[0.01, 0.99], [0.99, 0.01]])
        np.testing.assert_allclose(data.reference_positions, [0.0, 4.0])

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

    def test_simulator_applies_reproducible_observation_dropout(self):
        baseline = simulate_backcross(
            n_offspring=40,
            n_markers=80,
            mean_depth=8.0,
            read_error=0.01,
            heterozygous_state=True,
            random_seed=71,
        )
        dropped = simulate_backcross(
            n_offspring=40,
            n_markers=80,
            mean_depth=8.0,
            read_error=0.01,
            missing_probability=0.30,
            heterozygous_state=True,
            random_seed=71,
        )
        repeated = simulate_backcross(
            n_offspring=40,
            n_markers=80,
            mean_depth=8.0,
            read_error=0.01,
            missing_probability=0.30,
            heterozygous_state=True,
            random_seed=71,
        )
        assert baseline.reference_reads is not None
        assert baseline.alternate_reads is not None
        assert dropped.reference_reads is not None
        assert dropped.alternate_reads is not None
        baseline_zero = baseline.reference_reads + baseline.alternate_reads == 0
        dropped_zero = dropped.reference_reads + dropped.alternate_reads == 0
        self.assertGreater(np.mean(dropped_zero), np.mean(baseline_zero) + 0.20)
        self.assertTrue(np.all(dropped.probabilities[dropped_zero] == 0.5))
        np.testing.assert_array_equal(
            dropped.reference_reads,
            repeated.reference_reads,
        )
        np.testing.assert_array_equal(
            dropped.alternate_reads,
            repeated.alternate_reads,
        )

    def test_simulator_rejects_invalid_noise_probabilities(self):
        for argument, value in (
            ("read_error", 0.5),
            ("contamination", 1.1),
            ("missing_probability", -0.1),
        ):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                simulate_backcross(**{argument: value})

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
        log_aa = cross.alternate_reads * np.log(
            read_error
        ) + cross.reference_reads * np.log1p(-read_error)
        log_ab = (cross.reference_reads + cross.alternate_reads) * np.log(0.5)
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
        arguments = {
            "probabilities": cross.probabilities,
            "marker_names": cross.marker_names,
            "confidence": 0.8,
            "bootstrap_replicates": 5,
            "bin_threshold": 0.01,
            "neighbor_count": 20,
            "random_seed": 202,
        }
        first = fit_softmap(**arguments)
        second = fit_softmap(**arguments)
        np.testing.assert_array_equal(first.order, second.order)
        np.testing.assert_array_equal(
            first.representative_order, second.representative_order
        )
        np.testing.assert_array_equal(
            first.bootstrap_positions, second.bootstrap_positions
        )
        np.testing.assert_array_equal(first.precedence, second.precedence)
        np.testing.assert_array_equal(first.framework, second.framework)

    def test_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            fit_softmap(np.array([[0.0, 1.2], [0.2, 0.8]]))

    def test_bin_truth_uses_all_members(self):
        probabilities = np.array(
            [
                [0.01, 0.01, 0.99],
                [0.99, 0.99, 0.01],
                [0.01, 0.01, 0.99],
            ]
        )
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
        representatives = np.asarray(
            [
                int(np.flatnonzero(membership == group)[0])
                for group in np.unique(membership)
            ]
        )
        representatives = representatives[
            np.argsort(cross.input_to_truth[representatives])
        ][:4]
        metrics = evaluate_marker_coordinates(
            representatives,
            np.array([0.0, 0.0, 1.0, 2.0]),
            cross,
        )
        self.assertEqual(metrics["framework_truth_bins"], 4)
        self.assertEqual(metrics["framework_reported_position_bins"], 3)
        self.assertEqual(metrics["framework_tied_truth_bin_pairs"], 1)
        self.assertEqual(metrics["framework_ordered_truth_bin_pairs"], 5)
        self.assertAlmostEqual(metrics["framework_truth_bin_inversion_fraction"], 0.0)

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
            int(group)
            for group in np.unique(truth_membership)
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
        representatives = np.asarray(
            [
                int(np.flatnonzero(truth_membership == group)[0])
                for group in np.unique(truth_membership)
            ]
        )
        representatives = representatives[
            np.argsort(cross.input_to_truth[representatives])
        ]
        marker_to_bin = np.empty(cross.probabilities.shape[1], dtype=np.int64)
        for marker, truth_group in enumerate(truth_membership):
            marker_to_bin[marker] = int(
                np.flatnonzero(truth_membership[representatives] == truth_group)[0]
            )
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
            unresolved_metrics["all_interval_fully_unresolved_nonframework_truth_bins"],
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
        np.testing.assert_array_equal(padded_left[1:5], -np.ones(4, dtype=np.int64))
        np.testing.assert_array_equal(padded_right[1:5], 2 * np.ones(4, dtype=np.int64))
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
        edge_left = np.array([0, 1, 3, 8], dtype=np.int64)
        edge_right = np.array([1, 7, 9, 11], dtype=np.int64)
        edge_rf, edge_lod = pairwise_recombination_likelihood_edges(
            probabilities,
            edge_left,
            edge_right,
            batch_size=2,
        )
        np.testing.assert_array_equal(edge_rf, recombination[edge_left, edge_right])
        np.testing.assert_allclose(
            edge_lod,
            lod[edge_left, edge_right],
            rtol=1e-12,
            atol=1e-12,
        )
        boundary_probabilities = np.column_stack(
            (
                probabilities[:, 0],
                probabilities[:, 0],
                probabilities[:, 1],
                probabilities[:, 1],
            )
        )
        boundary_left = np.array([0, 0, 2], dtype=np.int64)
        boundary_right = np.array([1, 2, 3], dtype=np.int64)
        boundary_rf, boundary_lod = pairwise_recombination_likelihood_edges(
            boundary_probabilities,
            boundary_left,
            boundary_right,
        )
        at_zero, zero_lod = core_module._zero_recombination_likelihood_edges(
            boundary_probabilities,
            boundary_left,
            boundary_right,
            batch_size=2,
        )
        np.testing.assert_array_equal(at_zero, boundary_rf == 0.0)
        np.testing.assert_allclose(
            zero_lod[at_zero],
            boundary_lod[at_zero],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(zero_lod[~at_zero], 0.0)
        self.assertTrue(np.all(recombination >= 0.0))
        self.assertTrue(np.all(recombination < 0.5))
        self.assertGreater(float(np.median(lod[np.tril_indices(12, -1)])), 0.0)
        regularized, regularized_lod = pairwise_recombination_likelihood(
            probabilities,
            beta_prior_shape=1.5,
        )
        off_diagonal = np.tril_indices(12, -1)
        self.assertTrue(np.all(regularized[off_diagonal] > 0.0))
        self.assertTrue(
            np.all(regularized[off_diagonal] >= recombination[off_diagonal])
        )
        self.assertTrue(
            np.all(regularized_lod[off_diagonal] <= lod[off_diagonal] + 1e-12)
        )
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

    def test_likelihood_binning_pools_redundant_probabilistic_markers(self):
        cross = simulate_backcross(
            n_offspring=120,
            n_markers=12,
            mean_depth=4.0,
            heterozygous_state=True,
            random_seed=110,
        )
        probabilities = np.repeat(cross.probabilities, 3, axis=1)
        bins = likelihood_bin_markers(
            probabilities,
            maximum_bin_recombination=0.0,
            minimum_linkage_lod=3.0,
            neighbor_count=12,
            edge_batch_size=7,
        )
        self.assertLessEqual(bins.representatives.size, 12)
        self.assertEqual(bins.membership.size, probabilities.shape[1])
        self.assertEqual(
            bins.probabilities.shape,
            (probabilities.shape[0], bins.representatives.size),
        )

    def test_projected_candidate_neighbors_are_reproducible(self):
        rng = np.random.default_rng(111)
        probabilities = rng.uniform(0.01, 0.99, size=(30, 120))
        first_indices, first_distances = core_module._candidate_neighbors(
            probabilities,
            12,
            projection_dimensions=8,
        )
        second_indices, second_distances = core_module._candidate_neighbors(
            probabilities,
            12,
            projection_dimensions=8,
        )
        np.testing.assert_array_equal(first_indices, second_indices)
        np.testing.assert_allclose(first_distances, second_distances, atol=0.0)
        self.assertEqual(first_indices.shape, (probabilities.shape[1], 12))
        self.assertTrue(np.all(first_indices[:, 0] == np.arange(120)))
        self.assertTrue(np.all(first_distances[:, 0] == 0.0))

    def test_projected_likelihood_binning_pools_exact_duplicates(self):
        cross = simulate_backcross(
            n_offspring=120,
            n_markers=12,
            mean_depth=4.0,
            heterozygous_state=True,
            random_seed=112,
        )
        probabilities = np.repeat(cross.probabilities, 3, axis=1)
        bins = likelihood_bin_markers(
            probabilities,
            maximum_bin_recombination=0.0,
            minimum_linkage_lod=3.0,
            neighbor_count=12,
            neighbor_projection_dimensions=4,
            neighbor_projection_minimum_markers=3,
            edge_batch_size=7,
        )
        self.assertLessEqual(bins.representatives.size, 12)
        for marker in range(12):
            memberships = bins.membership[marker * 3 : marker * 3 + 3]
            self.assertTrue(np.all(memberships == memberships[0]))

    def test_projected_likelihood_binning_validates_projection_controls(self):
        probabilities = np.full((20, 5), 0.5)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            likelihood_bin_markers(
                probabilities,
                neighbor_projection_dimensions=1,
            )
        with self.assertRaisesRegex(ValueError, "threshold"):
            likelihood_bin_markers(
                probabilities,
                neighbor_projection_minimum_markers=2,
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
        self.assertEqual(metrics["rank_interval_truth_bin_inversion_fraction"], 0.0)
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
        left, right = placement_intervals(np.array([0, 2]), precedence, confidence=0.8)
        self.assertEqual(int(left[1]), -1)
        self.assertEqual(int(right[1]), 2)

        precedence[0, 1] = 0.9
        precedence[1, 2] = 0.9
        left, right = placement_intervals(np.array([0, 2]), precedence, confidence=0.8)
        self.assertEqual(int(left[1]), 0)
        self.assertEqual(int(right[1]), 1)

    def test_bootstrap_slot_interval_uses_shortest_supported_window(self):
        # Marker 1 lies between anchors in three of four maps and before them once.
        positions = np.array(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [1, 0, 2],
            ]
        )
        left, right = bootstrap_placement_intervals(
            np.array([0, 2]), positions, confidence=0.75
        )
        self.assertEqual((int(left[1]), int(right[1])), (0, 1))

        left, right = bootstrap_placement_intervals(
            np.array([0, 2]), positions, confidence=0.25
        )
        self.assertEqual((int(left[1]), int(right[1])), (-1, 0))
        with self.assertRaises(ValueError):
            bootstrap_placement_intervals(np.array([0, 2]), positions, confidence=0.0)

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
                "marker\toffspring_1\toffspring_2\nm1\t0.01\t0.99\nm2\t0.99\t0.01\n"
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
            with (
                patch(
                    "sys.argv",
                    [
                        "softmap",
                        str(input_path),
                        str(output_path),
                        "--bootstrap",
                        "3",
                    ],
                ),
                redirect_stdout(stdout),
            ):
                cli_main()
            summary = json.loads(stdout.getvalue())
            self.assertTrue(output_path.exists())
            self.assertGreater(summary["effective_offspring_information"], 0.0)

    def test_likelihood_mds_cli_writes_stability_columns(self):
        cross = simulate_backcross(
            n_offspring=20,
            n_markers=10,
            mean_depth=2.0,
            heterozygous_state=True,
            random_seed=102,
        )
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "probabilities.tsv"
            output_path = Path(directory) / "map.tsv"
            lines = [
                "marker\t"
                + "\t".join(
                    f"o{index + 1}" for index in range(cross.probabilities.shape[0])
                )
            ]
            for marker, name in enumerate(cross.marker_names):
                lines.append(
                    name
                    + "\t"
                    + "\t".join(
                        f"{value:.12g}" for value in cross.probabilities[:, marker]
                    )
                )
            input_path.write_text("\n".join(lines) + "\n")
            stdout = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "softmap",
                        str(input_path),
                        str(output_path),
                        "--likelihood-mds",
                        "--smacof-iterations",
                        "20",
                    ],
                ),
                redirect_stdout(stdout),
            ):
                cli_main()
            summary = json.loads(stdout.getvalue())
            header = output_path.read_text().splitlines()[0]
            self.assertEqual(summary["method"], "SoftMap-LMDS-Ensemble")
            self.assertEqual(summary["binning_method"], "none")
            self.assertIn("reported_position", header)
            self.assertIn("likelihood_bin", header)
            self.assertIn("stability_rank_left", header)
            self.assertIn("genetic_position_cm", header)
            self.assertIn("distance_status", summary)
            self.assertIn("map_length_cm", summary)

    def test_f2_cli_writes_complete_map_columns(self):
        cross = simulate_f2(
            n_offspring=80,
            n_markers=12,
            random_seed=214,
        )
        physical = 1_000_000.0 * cross.true_positions[cross.input_to_truth]
        data = softmap.F2LinkageData(
            cross.probabilities,
            cross.marker_names,
            physical_positions=physical,
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "f2-map.tsv"
            stdout = StringIO()
            with (
                patch("softmap.cli.read_vcf", return_value=data),
                patch(
                    "sys.argv",
                    [
                        "softmap",
                        "family.vcf",
                        str(output_path),
                        "--cross-design",
                        "f2",
                        "--parents",
                        "p0",
                        "p1",
                        "--physical-scaffold",
                        "--smacof-iterations",
                        "20",
                    ],
                ),
                redirect_stdout(stdout),
            ):
                cli_main()
            summary = json.loads(stdout.getvalue())
            header = output_path.read_text().splitlines()[0]
        self.assertEqual(summary["method"], "SoftMap-F2")
        self.assertTrue(summary["physical_scaffold_used"])
        self.assertIn("de_novo_order_rank", header)
        self.assertIn("genetic_position_cm", header)

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
        positions = np.array(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 2, 1],
            ]
        )
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
