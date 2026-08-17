"""Accessible overview plots for fitted maps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from .api import Map
    from .datasets import MapPositions


PHYSICAL_GENETIC_COLORS = ("#fde0dd", "#fa9fb5", "#c51b8a")


def plot_map(
    mapping: "Map",
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> "Figure":
    """Plot genotype structure and map agreement or bootstrap uncertainty."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
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

    cmap = LinearSegmentedColormap.from_list("softmap_probability", colors)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    image = axes[0].imshow(
        probability_order,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
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
        axes[1].plot(xline, np.polyval(fit_line, xline), color="#000000", linewidth=1.4)
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


def plot_physical_vs_genetic(
    positions: "MapPositions",
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> "Figure":
    """Plot physical position against genetic position for every chromosome."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
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
    cmap = LinearSegmentedColormap.from_list("softmap_physical_genetic", colors)
    chromosomes = np.unique(positions.chromosomes)
    columns = 4 if chromosomes.size > 4 else chromosomes.size
    rows = int(np.ceil(chromosomes.size / columns))
    fig, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(10.2, 5.4 if rows == 2 else 3.0 * rows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_array.ravel()
    for axis, chromosome in zip(axes, chromosomes):
        selected = positions.chromosomes == chromosome
        physical = positions.physical_mb[selected]
        genetic = positions.genetic_cm[selected]
        order = np.argsort(physical, kind="stable")
        physical = physical[order]
        genetic = genetic[order]
        scale = np.ptp(physical)
        color_values = (
            (physical - physical.min()) / scale if scale > 0 else np.zeros_like(physical)
        )
        axis.plot(physical, genetic, color=colors[1], alpha=0.34, linewidth=0.65)
        axis.scatter(
            physical,
            genetic,
            c=color_values,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=17,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.25,
            rasterized=True,
        )
        correlation = float(np.corrcoef(physical, genetic)[0, 1])
        axis.set_title(f"Chromosome {int(chromosome)}  r = {correlation:.2f}")
        axis.set_xlabel("Physical position (Mb)")
        axis.set_ylabel("Genetic position (cM)")
        axis.grid(color="#eeeeee", linewidth=0.55)
    for axis in axes[chromosomes.size:]:
        axis.set_visible(False)
    if positions.label:
        fig.suptitle(positions.label, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig


def plot_marker_order(
    mapping: "Map",
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> "Figure":
    """Show the probability matrix before and after marker ordering."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
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
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })
    cmap = LinearSegmentedColormap.from_list("softmap_marker_order", colors)
    probabilities = mapping.data.probabilities
    if probabilities.shape[0] > 180:
        offspring = np.linspace(0, probabilities.shape[0] - 1, 180).round().astype(int)
        probabilities = probabilities[offspring]
    before = probabilities
    after = probabilities[:, mapping.result.order]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.65), sharex=True, sharey=True, constrained_layout=True
    )
    images = []
    for label, title, matrix, axis in zip(
        ("a", "b"),
        ("Before: input marker order", "After: inferred marker order"),
        (before, after),
        axes,
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("Marker rank")
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            va="bottom",
            fontweight="normal",
        )
    axes[0].set_ylabel("Offspring")
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.03, pad=0.025)
    colorbar.set_label("State 1 probability")
    if mapping.data.label:
        fig.suptitle(mapping.data.label, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig


def plot_physical_order_grid(
    mappings: list["Map"] | tuple["Map", ...],
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> "Figure":
    """Plot physical position against marker rank before and after ordering."""

    if not mappings:
        raise ValueError("at least one fitted map is required")
    if len(mappings) > 8:
        raise ValueError("the 4 by 4 layout supports at most eight fitted maps")
    if any(mapping.data.physical_positions is None for mapping in mappings):
        raise ValueError("every fitted map must include physical positions")
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
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
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })
    cmap = LinearSegmentedColormap.from_list("softmap_physical_order", colors)
    rows = int(np.ceil(len(mappings) / 2))
    fig, axes_array = plt.subplots(
        rows,
        4,
        figsize=(8.2, 2.0 * rows + 0.45),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_array.ravel()
    for index, mapping in enumerate(mappings):
        before_axis = axes[index * 2]
        after_axis = axes[index * 2 + 1]
        physical = np.asarray(mapping.data.physical_positions)
        before_rank = np.arange(physical.size, dtype=np.float64)
        after_rank = np.empty(physical.size, dtype=np.float64)
        after_rank[mapping.result.order] = np.arange(physical.size, dtype=np.float64)
        before_r = float(np.corrcoef(physical, before_rank)[0, 1])
        after_r = float(np.corrcoef(physical, after_rank)[0, 1])
        if after_r < 0:
            after_rank = after_rank.max() - after_rank
            after_r = -after_r
        scale = np.ptp(physical)
        color_values = (
            (physical - physical.min()) / scale
            if scale > 0
            else np.zeros_like(physical)
        )
        label = str(index + 1)
        if mapping.data.label and "chromosome " in mapping.data.label:
            label = mapping.data.label.split("chromosome ", 1)[1].split(",", 1)[0]
        for state, axis, ranks, correlation in (
            ("before", before_axis, before_rank, abs(before_r)),
            ("after", after_axis, after_rank, after_r),
        ):
            axis.scatter(
                physical,
                ranks,
                c=color_values,
                cmap=cmap,
                vmin=0,
                vmax=1,
                s=12,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.2,
                rasterized=True,
            )
            axis.set_title(f"Chr {label} · {state} · r = {correlation:.2f}")
            axis.set_xlabel("Physical position (Mb)")
            axis.set_ylabel("Marker rank")
            axis.grid(color="#eeeeee", linewidth=0.45)
        physical_order = np.argsort(physical, kind="stable")
        after_axis.plot(
            physical[physical_order],
            after_rank[physical_order],
            color=colors[1],
            alpha=0.38,
            linewidth=0.6,
            zorder=0,
        )
    for axis in axes[len(mappings) * 2:]:
        axis.set_visible(False)
    fig.suptitle(
        "Rahnamae et al. (2026): physical position vs genetic-map order",
        fontweight="normal",
    )
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig
