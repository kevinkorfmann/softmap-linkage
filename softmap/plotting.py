"""Accessible overview plots for fitted maps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from .api import Map


def plot_map(mapping: "Map", path: str | Path | None = None) -> "Figure":
    """Plot genotype structure and map agreement or bootstrap uncertainty."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })

    result = mapping.result
    representatives = result.bins.representatives[result.representative_order]
    probability_order = mapping.data.probabilities[:, representatives]
    if probability_order.shape[0] > 160:
        offspring = np.linspace(0, probability_order.shape[0] - 1, 160).round().astype(int)
        probability_order = probability_order[offspring]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    image = axes[0].imshow(
        probability_order,
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes[0].set_title("Probabilities along the inferred map")
    axes[0].set_xlabel("Inferred marker rank")
    axes[0].set_ylabel("Offspring")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.set_label("State 1 probability")

    rank_by_group = np.empty(result.representative_order.size, dtype=np.int64)
    rank_by_group[result.representative_order] = np.arange(result.representative_order.size)
    rep_markers = result.bins.representatives
    inferred = rank_by_group
    reference = mapping.data.reference_positions
    if reference is not None:
        published = reference[rep_markers]
        correlation = float(np.corrcoef(inferred, published)[0, 1])
        if correlation < 0:
            inferred = inferred.max() - inferred
            correlation = -correlation
        axes[1].scatter(
            published,
            inferred,
            s=22,
            color="#0072B2",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )
        fit_line = np.polyfit(published, inferred, 1)
        xline = np.linspace(float(published.min()), float(published.max()), 100)
        axes[1].plot(xline, np.polyval(fit_line, xline), color="#D55E00", linewidth=1.4)
        axes[1].set_title(f"Inferred and reference order  r = {correlation:.2f}")
        axes[1].set_xlabel("Reference map position (cM)")
        axes[1].set_ylabel("Inferred marker rank")
    else:
        boot = result.bootstrap_positions
        lower, middle, upper = np.quantile(boot, [0.025, 0.5, 0.975], axis=0)
        x = rank_by_group
        axes[1].vlines(x, lower, upper, color="#56B4E9", alpha=0.65, linewidth=1)
        axes[1].scatter(x, middle, s=15, color="#0072B2")
        axes[1].plot([0, x.max()], [0, x.max()], color="#666666", linewidth=1)
        axes[1].set_title("Bootstrap rank intervals")
        axes[1].set_xlabel("Inferred marker rank")
        axes[1].set_ylabel("Bootstrap marker rank")

    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, va="bottom", fontweight="normal")

    if mapping.data.label:
        fig.suptitle(mapping.data.label, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig
