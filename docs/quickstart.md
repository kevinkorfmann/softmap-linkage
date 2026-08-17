# Quick start

This page is the shortest path to a result. If this is your first analysis, use the
[step-by-step guide](guide.md) for environment setup, input validation, output
interpretation, final-analysis settings, and troubleshooting.

## Install

```bash
python -m pip install "softmap-linkage[plot] @ git+https://github.com/kevinkorfmann/softmap-linkage.git"
```

## Run your VCF or BCF

For a parent-oriented backcross, list the recurrent parent first:

```python
import softmap

data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("recurrent_parent", "donor_parent"),
    cross_design="backcross",
)

mapping = softmap.fit(data, bootstrap=100, confidence=0.8, seed=7)

print(mapping.summary())
print(mapping.ordered_markers[:10])
print(mapping.framework_markers[:10])
print(mapping.marker_table()[:3])

mapping.plot("map.png")
```

The input must contain one linkage group of passing biallelic SNPs. Use
`cross_design="ril"` or `"doubled_haploid"` for those designs. See the
[API reference](api.md#variant-file-input) for sample selection, conversion rules,
and every output field.

## Run the included example

```python
from pprint import pprint

import softmap

data = softmap.demo()
mapping = softmap.fit(data)
figure = mapping.plot("map.png")

pprint(mapping.summary())
pprint(mapping.ordered_markers[:5])
pprint(mapping.marker_table()[:5])
```

Open `map.png` to see probability blocks along the inferred map and the inferred
order against the demo's reference positions. The summary contains the marker and
bin counts, framework size, and confidence threshold. `marker_table()` exposes the
bin, order rank, framework rank, and framework-relative placement interval for
every marker.

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
