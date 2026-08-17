from pathlib import Path
from tempfile import TemporaryDirectory

import softmap


def test_demo_is_plotable_and_inspectable():
    data = softmap.demo()
    mapping = softmap.fit(data, bootstrap=3)
    rows = mapping.marker_table()

    assert len(rows) == len(data.marker_names)
    assert set(rows[0]) == {
        "marker",
        "bin",
        "order_rank",
        "is_representative",
        "framework_rank",
        "interval_left",
        "interval_right",
    }
    assert {row["marker"] for row in rows} == set(data.marker_names)
    assert sorted(
        row["order_rank"] for row in rows if row["is_representative"]
    ) == list(range(mapping.summary()["bins"]))

    with TemporaryDirectory() as directory:
        destination = Path(directory) / "map.png"
        figure = mapping.plot(destination)
        assert destination.exists()
        assert len(figure.axes) == 3
        probability_cmap = figure.axes[0].images[0].cmap
        from matplotlib.colors import to_hex

        assert to_hex(probability_cmap(0.0)) == "#fde0dd"
        assert to_hex(probability_cmap(1.0)) == "#c51b8a"
        assert to_hex(figure.axes[1].lines[0].get_color()) == "#000000"

        order_destination = Path(directory) / "marker-order.png"
        order_figure = mapping.plot_marker_order(order_destination)
        assert order_destination.exists()
        assert len(order_figure.axes) == 3


def test_complete_f2_result_is_plotable_and_inspectable():
    cross = softmap.simulate_f2(
        n_offspring=80,
        n_markers=12,
        random_seed=215,
    )
    physical = 1_000_000.0 * cross.true_positions[cross.input_to_truth]
    data = softmap.F2LinkageData(
        cross.probabilities,
        cross.marker_names,
        physical_positions=physical,
        label="Simulated F2",
    )
    mapping = softmap.fit_f2(
        data,
        use_physical_scaffold=True,
        maximum_smacof_iterations=30,
    )
    with TemporaryDirectory() as directory:
        destination = Path(directory) / "f2-map.png"
        figure = mapping.plot(destination)
        assert destination.exists()
        assert len(figure.axes) == 3
        assert mapping.summary()["distance_method"] == "f2_pairwise_kosambi_adjacent"
        assert mapping.summary()["physical_scaffold_used"] is True
        assert mapping.marker_table()[0]["genetic_position_cm"] is not None
        comparison = softmap.plot_physical_output_grid(
            [mapping],
            [mapping],
            Path(directory) / "f2-before-after.png",
            use_de_novo_rank=True,
        )
        assert "Before: genotype-only draft" in comparison._suptitle.get_text()
        assert "genotype-only draft" in comparison.axes[0].get_title()
        assert "SoftMap result" in comparison.axes[1].get_title()
        stages = softmap.plot_f2_three_stage(
            mapping,
            Path(directory) / "f2-three-stage.png",
        )
        assert "not a map" in stages.axes[0].get_title()
        assert "Genotype-only" in stages.axes[1].get_title()
        assert "Final reference-guided" in stages.axes[2].get_title()
