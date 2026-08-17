# Plotting

Every fitted map has a `plot` method:

```python
mapping.plot("map.png")
mapping.plot("map.svg")
```

The first panel displays state probabilities along the inferred map. The second
panel compares inferred order with reference positions when available; otherwise it
shows 95% bootstrap rank intervals. The palette is color-vision friendly, and SVG
is recommended for scalable line and text output.

## Contemporary hybridization example

The published example is loaded directly from the source repository:

```python
import softmap

data = softmap.contemporary_hybridization(chromosome=1, markers=100)
mapping = softmap.fit(
    data,
    bootstrap=20,
    confidence=0.8,
    seed=7,
    bin_threshold=0.005,
)
mapping.plot("contemporary_hybridization.svg")
```

The source cross is F2-like and includes NN, NS, and SS calls. For this binary
software demonstration, NN becomes 0.01, SS becomes 0.99, and NS or missing calls
become 0.5. The reference positions are the published centimorgan coordinates.
This conversion tests loading, fitting, uncertainty summaries, and plotting, but it
does not perform full F2 phase inference and should not be treated as a revised
biological map.

Source: [Rahnamae et al., Contemporary_hybridization](https://github.com/nedarahnama/Contemporary_hybridization)
