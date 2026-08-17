"""Minimal SoftMap example."""

import softmap

data = softmap.demo()
mapping = softmap.fit(data, bootstrap=20, seed=7)
print(mapping.summary())
mapping.plot("softmap_example.png")
