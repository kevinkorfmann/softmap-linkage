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
