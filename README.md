# SoftMap

SoftMap builds confidence-aware linkage maps from probabilistic inheritance states.
It keeps uncertain calls uncertain, groups co-segregating markers, and reports a
supported framework instead of forcing every marker into a precise order.

[Documentation](https://kevinkorfmann.github.io/softmap-linkage/)

## Quick start

Install the package with plotting support:

```bash
pip install "softmap-linkage[plot]"
```

Fit and plot a small example:

```python
import softmap

data = softmap.demo()
mapping = softmap.fit(data, bootstrap=20, seed=7)

print(mapping.summary())
mapping.plot("softmap_example.png")
```

Your own data should be an offspring-by-marker NumPy array containing probabilities
between zero and one:

```python
mapping = softmap.fit(probabilities, marker_names)
```

Use at least 100 bootstrap replicates for a final analysis. SoftMap currently
supports one linkage group from phased binary parental-origin data, such as a
doubled-haploid or backcross design. It does not infer linkage groups or phase
general F2 and full-sib data.

## Published-data example

The repository includes a reproducible example using chromosome 1 from the
[Rahnamae et al. contemporary hybridization dataset](https://github.com/nedarahnama/Contemporary_hybridization):

```bash
python examples/contemporary_hybridization.py
```

The example downloads the source table directly, samples 100 markers, fits the map,
and writes linkage-order and physical-versus-genetic figures. Heterozygous and missing F2 calls are represented as
uninformative probabilities, so this is a software demonstration rather than a
replacement analysis of that cross.

## Documentation

The full guide includes installation, input formats, plotting, and the API reference.
It can be previewed locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Development

```bash
pip install -e ".[test]"
pytest -q
```

SoftMap is research software. Inspect the support summaries and validate the model
assumptions for your cross before biological interpretation.
