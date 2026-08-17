"""Fit, plot, and inspect a complete SoftMap example."""

from pprint import pprint

import softmap

data = softmap.demo()
mapping = softmap.fit(data)
figure = mapping.plot("map.png")

print("Fit summary")
pprint(mapping.summary())
print("\nFirst five ordered markers")
pprint(mapping.ordered_markers[:5])
print("\nFirst five marker records")
pprint(mapping.marker_table()[:5])
print("\nPlot panels")
pprint([axis.get_title() for axis in figure.axes if axis.get_title()])
