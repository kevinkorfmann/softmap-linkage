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
    assert sorted(row["order_rank"] for row in rows if row["is_representative"]) == list(
        range(mapping.summary()["bins"])
    )

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
