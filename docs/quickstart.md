# Quick start

This page is the shortest path to a result. If this is your first analysis, use the
[step-by-step guide](guide.md) for environment setup, input validation, output
interpretation, final-analysis settings, and troubleshooting.

## Install

```bash
pip install "softmap-linkage[plot]"
```

## Run the included example

```python
import softmap

data = softmap.demo(offspring=80, markers=60, seed=4)
mapping = softmap.fit(data, bootstrap=20, confidence=0.8, seed=7)

print(mapping.summary())
mapping.plot("softmap_example.png")
```

The result summary contains the marker count, co-segregation bin count, number of
supported framework markers, and confidence threshold.

For final analyses, increase `bootstrap` to at least 100 and assess stability across
reasonable seeds and bin thresholds.

## Use an array

If the data are already in NumPy, no container is required:

```python
import softmap

mapping = softmap.fit(
    probabilities,
    marker_names=["m1", "m2", "m3"],
    bootstrap=100,
)
```

Rows are offspring, columns are markers, and every value is the probability of
parental-origin state 1.
