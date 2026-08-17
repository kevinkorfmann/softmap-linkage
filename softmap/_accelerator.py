"""Optional compiled numerical kernels with a transparent NumPy fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_DISABLED = os.environ.get("SOFTMAP_DISABLE_RUST", "").lower() in {
    "1",
    "true",
    "yes",
}

try:
    if _DISABLED:
        raise ImportError
    import _softmap_rust as _rust
except ImportError:
    _rust = None


def rust_backend_available() -> bool:
    """Return whether the optional compiled kernels are active."""

    return _rust is not None


def pairwise_recombination_edges(
    probabilities: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    maximum_recombination: float,
    bisection_iterations: int,
    beta_prior_shape: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run the compiled binary-pair kernel when its fixed cost is worthwhile."""

    if _rust is None or left.size < 4_096:
        return None
    result: Any = _rust.pairwise_recombination_edges(
        np.ascontiguousarray(probabilities, dtype=np.float64),
        np.ascontiguousarray(left, dtype=np.int64),
        np.ascontiguousarray(right, dtype=np.int64),
        maximum_recombination,
        bisection_iterations,
        beta_prior_shape,
    )
    recombination = np.asarray(result[0])
    # Keep the final reduction in NumPy so LOD values remain bit-for-bit equal
    # to the reference backend. Tiny platform-libm differences can otherwise
    # perturb exact-threshold decisions on very large maps.
    lod = np.empty_like(recombination)
    for start in range(0, left.size, batch_size):
        stop = min(start + batch_size, left.size)
        first = probabilities[:, left[start:stop]]
        second = probabilities[:, right[start:stop]]
        same = (1.0 - first) * (1.0 - second) + first * second
        different = first * (1.0 - second) + (1.0 - first) * second
        delta = different - same
        fitted = recombination[start:stop]
        linked = np.sum(
            np.log(np.clip(same + delta * fitted[None, :], 1e-300, None)),
            axis=0,
        )
        unlinked = np.sum(
            np.log(np.clip(same + 0.5 * delta, 1e-300, None)),
            axis=0,
        )
        lod[start:stop] = np.maximum(
            0.0,
            (linked - unlinked) / np.log(10.0),
        )
    return recombination, lod


def f2_pairwise_recombination(
    probabilities: np.ndarray,
    *,
    maximum_recombination: float,
    bisection_iterations: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run the compiled dense-F2 kernel for nontrivial marker matrices."""

    if _rust is None or probabilities.shape[1] < 32:
        return None
    result: Any = _rust.f2_pairwise_recombination(
        np.ascontiguousarray(probabilities, dtype=np.float64),
        maximum_recombination,
        bisection_iterations,
    )
    return np.asarray(result[0]), np.asarray(result[1])
