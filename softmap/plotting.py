"""Accessible overview plots for fitted maps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .api import F2Map, LikelihoodMap, Map
    from .datasets import MapPositions


PHYSICAL_GENETIC_COLORS = ("#fde0dd", "#fa9fb5", "#c51b8a")


def _physical_megabases(values: np.ndarray) -> np.ndarray:
    """Normalize VCF base-pair coordinates while preserving built-in Mb data."""

    physical = np.asarray(values, dtype=np.float64)
    return physical / 1_000_000.0 if np.nanmax(np.abs(physical)) > 100_000 else physical


def plot_map(
    mapping: Map,
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> Figure:
    """Plot genotype structure and map agreement or bootstrap uncertainty."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    result = mapping.result
    representatives = result.bins.representatives[result.representative_order]
    probability_order = mapping.data.probabilities[:, representatives]
    if probability_order.shape[0] > 160:
        offspring = (
            np.linspace(0, probability_order.shape[0] - 1, 160).round().astype(int)
        )
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
    rank_by_group[result.representative_order] = np.arange(
        result.representative_order.size
    )
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

    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.text(
            -0.13,
            1.04,
            label,
            transform=axis.transAxes,
            va="bottom",
            fontweight="normal",
        )

    if mapping.data.label:
        fig.suptitle(mapping.data.label, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig


def plot_f2_map(
    mapping: F2Map,
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> Figure:
    """Plot complete F2 genotype dosage and inferred genetic coordinates."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    order = mapping.result.order
    dosage = (
        mapping.data.probabilities[:, :, 1] + 2.0 * mapping.data.probabilities[:, :, 2]
    ) / 2.0
    dosage = dosage[:, order]
    if dosage.shape[0] > 180:
        offspring = np.linspace(0, dosage.shape[0] - 1, 180).round().astype(int)
        dosage = dosage[offspring]
    cmap = LinearSegmentedColormap.from_list("softmap_f2", colors)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    image = axes[0].imshow(
        dosage,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[0].set_title("Complete F2 genotypes along the inferred map")
    axes[0].set_xlabel("Inferred marker rank")
    axes[0].set_ylabel("Offspring")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.set_ticks((0.0, 0.5, 1.0), labels=("NN", "NS", "SS"))

    positions = mapping.result.genetic_distances.marker_positions_cm
    if mapping.data.physical_positions is not None:
        physical = _physical_megabases(mapping.data.physical_positions)
        axes[1].scatter(
            physical,
            positions,
            s=18,
            color="#c51b8a",
            alpha=0.78,
            edgecolor="white",
            linewidth=0.25,
        )
        axes[1].set_xlabel("Physical position (Mb)")
        axes[1].set_ylabel("SoftMap genetic position (cM)")
        axes[1].set_title("Inferred genetic and physical positions")
    else:
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        axes[1].scatter(rank, positions, s=18, color="#c51b8a", alpha=0.78)
        axes[1].set_xlabel("Inferred marker rank")
        axes[1].set_ylabel("SoftMap genetic position (cM)")
        axes[1].set_title("Inferred F2 map coordinates")
    axes[1].grid(color="#eeeeee", linewidth=0.55)
    if mapping.data.label:
        fig.suptitle(mapping.data.label, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig


def plot_f2_three_stage(
    mapping: F2Map,
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> Figure:
    """Show raw F2 calls, the genotype-only draft, and the final SoftMap map."""

    if mapping.data.physical_positions is None:
        raise ValueError("the three-stage figure requires physical positions")
    if not mapping.result.physical_scaffold_used:
        raise ValueError("the three-stage figure requires a physical-scaffold F2 map")
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    cmap = LinearSegmentedColormap.from_list("softmap_f2_three_stage", colors)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.45),
        constrained_layout=True,
    )

    dosage = (
        mapping.data.probabilities[:, :, 1] + 2.0 * mapping.data.probabilities[:, :, 2]
    ) / 2.0
    image = axes[0].imshow(
        dosage,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[0].set_title("1  Raw NN/NS/SS evidence — not a map")
    axes[0].set_xlabel("Marker in original source order")
    axes[0].set_ylabel("Offspring")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.set_ticks((0.0, 0.5, 1.0), labels=("NN", "NS", "SS"))

    physical = _physical_megabases(mapping.data.physical_positions)
    draft_rank = np.empty(physical.size, dtype=np.float64)
    draft_rank[mapping.result.de_novo_order] = np.arange(
        physical.size, dtype=np.float64
    )
    if np.corrcoef(physical, draft_rank)[0, 1] < 0:
        draft_rank = draft_rank.max() - draft_rank
    genetic = np.asarray(
        mapping.result.genetic_distances.marker_positions_cm,
        dtype=np.float64,
    )
    scale = np.ptp(physical)
    color_values = (
        (physical - physical.min()) / scale if scale > 0 else np.zeros_like(physical)
    )
    physical_order = np.argsort(physical, kind="stable")
    for axis, values, title, ylabel in (
        (
            axes[1],
            draft_rank,
            "2  Genotype-only de-novo draft",
            "Draft marker rank",
        ),
        (
            axes[2],
            genetic,
            "3  Final reference-guided SoftMap map",
            "Inferred position (cM)",
        ),
    ):
        axis.plot(
            physical[physical_order],
            values[physical_order],
            color=colors[1],
            alpha=0.38,
            linewidth=0.7,
            zorder=0,
        )
        axis.scatter(
            physical,
            values,
            c=color_values,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=18,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.25,
            rasterized=True,
        )
        correlation = abs(float(np.corrcoef(physical, values)[0, 1]))
        axis.set_title(f"{title}\nr = {correlation:.2f}")
        axis.set_xlabel("Physical position (Mb)")
        axis.set_ylabel(ylabel)
        axis.grid(color="#eeeeee", linewidth=0.5)

    title = "Raw evidence → genotype-only draft → SoftMap result"
    if mapping.data.label:
        chromosome = mapping.data.label.split(",", 1)[0]
        title = f"{chromosome}: {title}"
    fig.suptitle(title, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig


def plot_physical_vs_genetic(
    positions: MapPositions,
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> Figure:
    """Plot physical position against genetic position for every chromosome."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    cmap = LinearSegmentedColormap.from_list("softmap_physical_genetic", colors)
    chromosomes = np.unique(positions.chromosomes)
    columns = min(chromosomes.size, 4)
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
    for axis, chromosome in zip(axes, chromosomes, strict=False):
        selected = positions.chromosomes == chromosome
        physical = positions.physical_mb[selected]
        genetic = positions.genetic_cm[selected]
        order = np.argsort(physical, kind="stable")
        physical = physical[order]
        genetic = genetic[order]
        scale = np.ptp(physical)
        color_values = (
            (physical - physical.min()) / scale
            if scale > 0
            else np.zeros_like(physical)
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
    for axis in axes[chromosomes.size :]:
        axis.set_visible(False)
    if positions.label:
        fig.suptitle(positions.label, fontweight="normal")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig


def plot_marker_order(
    mapping: Map,
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> Figure:
    """Show the probability matrix before and after marker ordering."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
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
        strict=True,
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
    mappings: list[Map] | tuple[Map, ...],
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
) -> Figure:
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

    mpl.rcParams.update(
        {
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
        }
    )
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
    for axis in axes[len(mappings) * 2 :]:
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


def plot_physical_output_grid(
    rank_mappings: list[Map | F2Map] | tuple[Map | F2Map, ...],
    distance_mappings: list[LikelihoodMap | F2Map] | tuple[LikelihoodMap | F2Map, ...],
    path: str | Path | None = None,
    *,
    colors: tuple[str, str, str] = PHYSICAL_GENETIC_COLORS,
    use_de_novo_rank: bool = False,
) -> Figure:
    """Plot a marker-order draft and inferred genetic distance against physical position.

    For an F2 map, ``use_de_novo_rank=True`` shows its genotype-only likelihood
    order in the left panels while the right panels retain the final genetic map.
    This creates an honest before/after comparison without randomizing markers.
    """

    if not rank_mappings or len(rank_mappings) != len(distance_mappings):
        raise ValueError("rank and distance mappings must be non-empty and aligned")
    if len(rank_mappings) > 8:
        raise ValueError("the 4 by 4 layout supports at most eight fitted maps")
    if any(mapping.data.physical_positions is None for mapping in rank_mappings):
        raise ValueError("every fitted map must include physical positions")
    for rank_mapping, distance_mapping in zip(
        rank_mappings, distance_mappings, strict=True
    ):
        if rank_mapping.data.marker_names != distance_mapping.data.marker_names:
            raise ValueError("rank and distance mappings must contain the same markers")
        distances = distance_mapping.result.genetic_distances
        if distances is None or not np.all(np.isfinite(distances.marker_positions_cm)):
            raise ValueError("every likelihood map must contain genetic positions")

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise ImportError(
            'Plotting requires matplotlib. Install with pip install "softmap-linkage[plot]".'
        ) from error

    mpl.rcParams.update(
        {
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
        }
    )
    cmap = LinearSegmentedColormap.from_list("softmap_physical_outputs", colors)
    rows = int(np.ceil(len(rank_mappings) / 2))
    fig, axes_array = plt.subplots(
        rows,
        4,
        figsize=(8.2, 2.0 * rows + 0.45),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_array.ravel()
    for index, (rank_mapping, distance_mapping) in enumerate(
        zip(rank_mappings, distance_mappings, strict=True)
    ):
        physical = _physical_megabases(rank_mapping.data.physical_positions)
        rank_order = (
            rank_mapping.result.de_novo_order
            if use_de_novo_rank and hasattr(rank_mapping.result, "de_novo_order")
            else rank_mapping.result.order
        )
        rank = np.empty(physical.size, dtype=np.float64)
        rank[rank_order] = np.arange(physical.size, dtype=np.float64)
        genetic = np.asarray(
            distance_mapping.result.genetic_distances.marker_positions_cm,
            dtype=np.float64,
        )
        if np.corrcoef(physical, rank)[0, 1] < 0:
            rank = rank.max() - rank
        if np.corrcoef(physical, genetic)[0, 1] < 0:
            genetic = genetic.max() - genetic
        scale = np.ptp(physical)
        color_values = (
            (physical - physical.min()) / scale
            if scale > 0
            else np.zeros_like(physical)
        )
        label = str(index + 1)
        if rank_mapping.data.label and "chromosome " in rank_mapping.data.label:
            label = rank_mapping.data.label.split("chromosome ", 1)[1].split(",", 1)[0]
        physical_order = np.argsort(physical, kind="stable")
        for axis, values, name, ylabel in (
            (
                axes[index * 2],
                rank,
                "genotype-only draft" if use_de_novo_rank else "marker rank",
                "Draft marker rank" if use_de_novo_rank else "Inferred marker rank",
            ),
            (
                axes[index * 2 + 1],
                genetic,
                "SoftMap result",
                "Inferred position (cM)",
            ),
        ):
            correlation = abs(float(np.corrcoef(physical, values)[0, 1]))
            axis.plot(
                physical[physical_order],
                values[physical_order],
                color=colors[1],
                alpha=0.38,
                linewidth=0.6,
                zorder=0,
            )
            axis.scatter(
                physical,
                values,
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
            axis.set_title(f"Chr {label} · {name} · r = {correlation:.2f}")
            axis.set_xlabel("Physical position (Mb)")
            axis.set_ylabel(ylabel)
            axis.grid(color="#eeeeee", linewidth=0.45)
    for axis in axes[len(rank_mappings) * 2 :]:
        axis.set_visible(False)
    fig.suptitle(
        (
            "Before: genotype-only draft   ·   After: reference-guided SoftMap map"
            if use_de_novo_rank
            else "SoftMap outputs: inferred order and genetic distance"
        ),
        fontweight="normal",
    )
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
    return fig
