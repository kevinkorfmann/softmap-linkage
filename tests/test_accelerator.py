import unittest
from unittest.mock import patch

import numpy as np

import softmap
import softmap._accelerator as accelerator
from softmap import core
from softmap.simulate import simulate_backcross, simulate_f2


class AcceleratorTests(unittest.TestCase):
    def test_backend_status_is_boolean(self):
        self.assertIsInstance(softmap.rust_backend_available(), bool)

    def test_small_inputs_keep_the_numpy_path(self):
        fake_backend = object()
        with patch.object(accelerator, "_rust", fake_backend):
            self.assertIsNone(
                accelerator.pairwise_recombination_edges(
                    np.full((2, 2), 0.5),
                    np.array([0]),
                    np.array([1]),
                    maximum_recombination=0.499999,
                    bisection_iterations=32,
                    beta_prior_shape=1.0,
                    batch_size=4_096,
                )
            )
            self.assertIsNone(
                accelerator.f2_pairwise_recombination(
                    np.full((2, 2, 3), 1.0 / 3.0),
                    maximum_recombination=0.499999,
                    bisection_iterations=32,
                )
            )

    def test_compiled_output_adapter_preserves_numpy_types_and_lod(self):
        class FakeBackend:
            @staticmethod
            def pairwise_recombination_edges(*_args):
                return np.full(4_096, 0.25), np.zeros(4_096)

            @staticmethod
            def f2_pairwise_recombination(*_args):
                return np.zeros((32, 32)), np.zeros((32, 32))

        probabilities = np.full((3, 2), 0.5)
        left = np.zeros(4_096, dtype=np.int64)
        right = np.ones(4_096, dtype=np.int64)
        with patch.object(accelerator, "_rust", FakeBackend()):
            binary = accelerator.pairwise_recombination_edges(
                probabilities,
                left,
                right,
                maximum_recombination=0.499999,
                bisection_iterations=32,
                beta_prior_shape=1.0,
                batch_size=1_024,
            )
            f2 = accelerator.f2_pairwise_recombination(
                np.full((2, 32, 3), 1.0 / 3.0),
                maximum_recombination=0.499999,
                bisection_iterations=32,
            )
        assert binary is not None
        assert f2 is not None
        np.testing.assert_array_equal(binary[0], np.full(4_096, 0.25))
        np.testing.assert_array_equal(binary[1], np.zeros(4_096))
        self.assertEqual(f2[0].shape, (32, 32))

    def test_f2_exact_reduction_is_exercised_without_the_extension(self):
        cross = simulate_f2(
            n_offspring=40,
            n_markers=32,
            mean_depth=2.0,
            random_seed=304,
        )
        with patch.object(
            core,
            "_accelerated_f2_pairwise_recombination",
            return_value=None,
        ):
            reference = core.f2_pairwise_recombination_likelihood(cross.probabilities)
        with patch.object(
            core,
            "_accelerated_f2_pairwise_recombination",
            return_value=(reference[0], np.full_like(reference[1], np.nan)),
        ):
            adapted = core.f2_pairwise_recombination_likelihood(cross.probabilities)
        np.testing.assert_array_equal(adapted[0], reference[0])
        np.testing.assert_array_equal(adapted[1], reference[1])

    @unittest.skipUnless(
        accelerator.rust_backend_available(), "optional Rust extension is not installed"
    )
    def test_compiled_binary_edges_match_numpy(self):
        cross = simulate_backcross(
            n_offspring=60,
            n_markers=120,
            mean_depth=2.0,
            random_seed=301,
        )
        rng = np.random.default_rng(301)
        left = rng.integers(0, 120, 5_000, dtype=np.int64)
        right = (left + rng.integers(1, 120, 5_000, dtype=np.int64)) % 120
        compiled = core.pairwise_recombination_likelihood_edges(
            cross.probabilities, left, right
        )
        with patch.object(
            core,
            "_accelerated_pairwise_recombination_edges",
            return_value=None,
        ):
            reference = core.pairwise_recombination_likelihood_edges(
                cross.probabilities, left, right
            )
        np.testing.assert_array_equal(compiled[0], reference[0])
        np.testing.assert_array_equal(compiled[1], reference[1])

    @unittest.skipUnless(
        accelerator.rust_backend_available(), "optional Rust extension is not installed"
    )
    def test_compiled_f2_matrix_matches_numpy(self):
        cross = simulate_f2(
            n_offspring=80,
            n_markers=48,
            mean_depth=2.0,
            missing_probability=0.1,
            random_seed=302,
        )
        compiled = core.f2_pairwise_recombination_likelihood(cross.probabilities)
        with patch.object(
            core,
            "_accelerated_f2_pairwise_recombination",
            return_value=None,
        ):
            reference = core.f2_pairwise_recombination_likelihood(cross.probabilities)
        np.testing.assert_array_equal(compiled[0], reference[0])
        np.testing.assert_array_equal(compiled[1], reference[1])


if __name__ == "__main__":
    unittest.main()
