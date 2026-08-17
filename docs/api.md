# Python API

The public API has three stages:

1. Load a VCF/BCF or construct an offspring-by-marker probability matrix.
2. Fit an order with `softmap.fit()` or `softmap.fit_likelihood()`.
3. Inspect the returned summary, marker table, ordered names, and plots.

SoftMap fits **one known linkage group at a time**. It assumes that every retained
marker describes the same two phased parental-origin states. It does not discover
linkage groups, fit a general unphased F2/full-sib model, or estimate centimorgan
distances.

## Which function should I use?

| Starting point | Recommended call | Result |
| --- | --- | --- |
| VCF, `.vcf.gz`, or BCF with parents | `read_vcf(...)`, then `fit(data)` | Parent-oriented confidence-aware map |
| VCF/BCF already encoded with consistent binary state orientation | `fit("input.vcf.gz")` | Confidence-aware map using loader defaults |
| NumPy probability matrix | `fit(probabilities, marker_names)` | Confidence-aware map |
| Need a robust complete order and model-sensitivity bands | `fit_likelihood(data)` | Likelihood-MDS ensemble map |

Calling `fit()` directly with a variant-file path uses `read_vcf()` defaults. Use
`read_vcf()` explicitly when you need to select a chromosome, select offspring, or
provide parents and a cross design.

## Variant-file input

### `read_vcf()`

```python
data = softmap.read_vcf(
    path,
    chromosome=None,
    samples=None,
    parents=None,
    cross_design="auto",
)
```

Accepted files are plain VCF (`.vcf`), bgzip-compressed VCF (`.vcf.gz`), and BCF
(`.bcf`). An index is not required because the loader can scan the file
sequentially.

#### Arguments

| Argument | Type | Meaning |
| --- | --- | --- |
| `path` | `str` or `pathlib.Path` | Input `.vcf`, `.vcf.gz`, or `.bcf` file. |
| `chromosome` | `str` or `None` | Exact VCF contig name to retain, such as `"chr1"`. If omitted, the usable markers must belong to one contig. |
| `samples` | iterable of sample names or `None` | Offspring columns to load, in the requested row order. If omitted, all non-parent samples are used. At least two unique samples are required. |
| `parents` | two sample names or `None` | `(state0_parent, state1_parent)`. Both must be distinct, homozygous, informative VCF samples at a retained marker. They are excluded from the offspring by default. |
| `cross_design` | `str` | One of `"auto"`, `"backcross"`, `"ril"`, or `"doubled_haploid"`. A non-auto value is required when `parents` is supplied. |

For a backcross, the first parent must be the recurrent parent. The recurrent-parent
homozygote becomes state 0 and the recurrent/donor heterozygote becomes state 1.
For an RIL or doubled-haploid cross, the first and second parental homozygotes become
states 0 and 1 respectively.

When `parents=None`, the loader keeps markers having exactly two observed offspring
genotype classes and orders those classes by alternate-allele dosage. This is only
appropriate when REF/ALT coding already gives a consistent binary-state orientation
across markers. Supplying parents is safer when REF/ALT orientation can change
relative to parental origin.

#### Records retained

The loader retains a record only when all applicable conditions hold:

- it is a single-nucleotide, biallelic record with one REF and one ALT allele;
- its `FILTER` is `PASS` or empty;
- it is on the requested chromosome, when one was requested;
- it defines two usable inheritance states;
- with parents, both parent calls are present, homozygous, and different;
- without parents, exactly two supported offspring `GT` classes are observed.

Multiallelic variants, indels, failed filters, monomorphic records, records with
three observed genotype classes in automatic mode, and parent-uninformative records
are skipped. If no usable markers remain, the loader raises `ValueError`.

#### Genotype conversion

Each output number is the probability that an offspring carries parental state 1:

| Available VCF field | Conversion |
| --- | --- |
| `PL` | First choice. The two relevant phred-scaled genotype likelihoods are normalized to a state-1 probability. |
| `GL` | Used when `PL` is absent. The two relevant log10 likelihoods are normalized. |
| `GT` only | A compatible state-0 call becomes `0.01`; a compatible state-1 call becomes `0.99`. |
| No usable `PL`/`GL` and a missing or incompatible `GT` | Becomes `0.5`, meaning no information about the binary state. |

`PL` and `GL` preserve uncertainty instead of forcing a hard call. A value of 0.5
does **not** generally mean biologically heterozygous; it means that the binary
parental state is unknown to the mapping model.

#### Return value

`read_vcf()` returns `LinkageData` with:

- `probabilities`: offspring rows by retained-marker columns;
- `marker_names`: VCF `ID`, or `CHROM:POS` when `ID` is missing;
- `physical_positions`: one-based VCF `POS` coordinates in base pairs;
- `label`: the selected or detected chromosome name;
- `reference_positions`: `None` because a VCF does not supply known genetic-map
  positions.

Duplicate marker names cause an error rather than silently overwriting a marker.

#### Parent-oriented example

```python
import softmap

data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    samples=["offspring_01", "offspring_02", "offspring_03"],
    parents=("recurrent_parent", "donor_parent"),
    cross_design="backcross",
)

print(data.probabilities.shape)   # (3, number_of_retained_markers)
print(data.marker_names[:3])
print(data.physical_positions[:3])
```

## Probability-matrix input

### Shape and values

A matrix is arranged as **offspring in rows and markers in columns**:

```python
import numpy as np
import softmap

probabilities = np.array([
    [0.01, 0.04, 0.95],
    [0.99, 0.92, 0.08],
    [0.50, 0.87, 0.12],
])

mapping = softmap.fit(
    probabilities,
    marker_names=["m1", "m2", "m3"],
)
```

The matrix must be two-dimensional, finite, and entirely within `[0, 1]`. At least
two offspring and two markers are required. The number of marker names must equal
the number of columns. If names are omitted, `fit()` generates `m1`, `m2`, and so
on.

### `LinkageData`

Use `LinkageData` when you want to keep names and optional metadata together:

```python
data = softmap.LinkageData(
    probabilities=probabilities,
    marker_names=("m1", "m2", "m3"),
    reference_positions=np.array([0.0, 4.2, 9.8]),
    label="Linkage group 1",
    physical_positions=np.array([102_400, 880_120, 1_430_055]),
)
```

| Field | Meaning |
| --- | --- |
| `probabilities` | Required floating-point offspring-by-marker matrix. |
| `marker_names` | Required marker-name tuple with one entry per matrix column. Names should be unique. |
| `reference_positions` | Optional known reference genetic positions, with one value per marker. Used for comparison/plotting, not to fit the order. |
| `label` | Optional human-readable linkage-group or dataset label. |
| `physical_positions` | Optional physical coordinates, with one value per marker. Used for plotting, not to fit the order. |

`data.shuffled(seed=1)` returns a new `LinkageData` with marker columns and their
metadata permuted together. It is useful for demonstrations and input-order
sensitivity checks; it does not shuffle offspring.

## Confidence-aware fitting

### `fit()`

```python
mapping = softmap.fit(
    data,
    marker_names=None,
    bootstrap=20,
    confidence=0.8,
    seed=1,
    bin_threshold=0.01,
)
```

#### Arguments

| Argument | Type/default | Meaning |
| --- | --- | --- |
| `data` | required | A VCF/BCF path, `LinkageData`, or offspring-by-marker array. |
| `marker_names` | `None` | Names used only when `data` is an array. Do not pass this when `data` is `LinkageData` or a path. |
| `bootstrap` | `20` | Number of resampled maps used to measure ordering support. Use at least 100 for a final analysis; more replicates improve stability but cost more time. |
| `confidence` | `0.8` | Minimum pairwise bootstrap precedence support for framework anchors. It must be greater than 0.5 and less than 1. |
| `seed` | `1` | Random seed for reproducible resampling. Use `None` for a non-fixed seed. |
| `bin_threshold` | `0.01` | Maximum expected disagreement for merging approximately co-segregating markers. Smaller values create stricter, usually more numerous bins. Pass `None` for automatic selection. |

The function first groups nearly indistinguishable markers into bins, orders one
representative per bin, bootstraps offspring information, and selects a framework
whose pairwise order reaches `confidence`. It then expands the bin order so every
input marker appears in the returned complete order.

#### Return value: `Map`

`fit()` returns a `Map`, not a bare list. The main members are:

| Member | Type | Meaning |
| --- | --- | --- |
| `mapping.data` | `LinkageData` | The exact normalized input used for fitting. |
| `mapping.result` | `SoftMapResult` | Low-level arrays and diagnostics for advanced use. |
| `mapping.ordered_markers` | `list[str]` | Every input marker in inferred order, including all members of co-segregation bins. |
| `mapping.framework_markers` | `list[str]` | Supported representative markers in framework order. This is the more defensible biological backbone. |
| `mapping.summary()` | `dict` | JSON-serializable run summary. |
| `mapping.marker_table()` | `list[dict]` | One JSON-serializable row per input marker. |
| `mapping.plot(path=None, colors=...)` | Matplotlib `Figure` | Probability-block overview. Saves it when `path` is supplied. The three colors represent state 0, uncertain values, and state 1. |
| `mapping.plot_marker_order(path=None)` | Matplotlib `Figure` | Input-order versus inferred-order comparison. |

Map direction is arbitrary. A complete reversal represents the same linkage map.

The default probability colors are `("#fde0dd", "#fa9fb5", "#c51b8a")`, matching
the Rahnamae before/after marker-order figures. Pass another three-color tuple to
`mapping.plot(..., colors=...)` to override them.

### Summary fields

```python
summary = mapping.summary()
```

| Field | Meaning |
| --- | --- |
| `status` | `"ok"` when at least three framework markers are supported; otherwise `"limited_support"`. |
| `offspring` | Number of input offspring rows. |
| `markers` | Number of input marker columns. |
| `bins` | Number of distinguishable co-segregation groups. |
| `framework_markers` | Number of supported representative markers in the framework. |
| `confidence` | Framework support threshold used for this fit. |

`limited_support` is a usable diagnostic result, not a software crash. It says that
the data and requested threshold do not support a framework of at least three
markers.

### Marker-table fields

```python
rows = mapping.marker_table()
```

All ranks are zero-based.

| Field | Meaning |
| --- | --- |
| `marker` | Input marker name. |
| `bin` | Integer co-segregation-bin identifier. Markers with the same value are not distinguishable at the chosen threshold. |
| `order_rank` | Rank of that bin in the inferred complete bin order. Markers in one bin share a rank. |
| `is_representative` | `True` for the marker used to represent its bin during fitting. |
| `framework_rank` | Zero-based framework-anchor rank, or `None` for a non-framework marker. |
| `interval_left` | Supported framework anchor immediately at or to the left of the marker's placement interval. `-1` means the interval can extend before the first anchor. |
| `interval_right` | Supported framework anchor immediately at or to the right of the interval. A value equal to the framework size means it can extend after the last anchor. |

For a framework marker, `interval_left == interval_right == framework_rank`. For a
non-framework marker, the two values describe placement relative to framework
anchors; they are not physical coordinates, centimorgans, or complete-order ranks.

The table uses only Python scalar types, so it can be serialized or converted
directly:

```python
import json
import pandas as pd

json_text = json.dumps(mapping.marker_table())
frame = pd.DataFrame(mapping.marker_table())
```

## Likelihood-MDS ensemble fitting

### `fit_likelihood()`

```python
mapping = softmap.fit_likelihood(
    data,
    marker_names=None,
    stability_mass=0.90,
    maximum_smacof_iterations=500,
)
```

Use this interface when a complete order is required and you want rank bands that
show sensitivity across a prespecified family of likelihood-MDS models. These are
**model-stability bands**, not bootstrap confidence intervals.

| Argument | Type/default | Meaning |
| --- | --- | --- |
| `data` | required | Same accepted inputs as `fit()`. |
| `marker_names` | `None` | Optional names only for array input. |
| `stability_mass` | `0.90` | Central fraction of candidate-model ranks included in each marker's stability band. Must lie between 0 and 1. |
| `maximum_smacof_iterations` | `500` | Maximum optimization iterations for each multidimensional-scaling candidate. Must be positive. |

The returned `LikelihoodMap` exposes `data`, the low-level `result`,
`ordered_markers`, `summary()`, and `marker_table()`.

### Likelihood summary fields

| Field | Meaning |
| --- | --- |
| `method` | Always `"SoftMap-LMDS-Ensemble"`. |
| `status` | `"ok"`, `"limited_order_information"`, or `"insufficient_order_information"`, based on separation of candidate-model rank bands. |
| `offspring` | Number of input offspring. |
| `markers` | Number of input markers. |
| `candidate_orders` | Number of model configurations compared. |
| `selected_config` | `[distance, LOD exponent, embedding dimensions, principal-curve knots]` for the selected candidate. |
| `stability_mass` | Requested mass used to form rank bands. |
| `stability_comparable_pair_fraction` | Fraction of marker pairs whose stability bands do not overlap. Higher values mean more of the order is resolved consistently across candidate models. |
| `unanimous_family_veto_triggered` | Whether unanimous LOD-weighted objectives replaced the default uniformly scored candidate under the ensemble's prespecified veto rule. |

The likelihood marker table contains:

| Field | Meaning |
| --- | --- |
| `marker` | Marker name. |
| `order_rank` | Zero-based rank in the selected complete order. |
| `stability_rank_left` | Lowest rank in the marker's central candidate-model stability band. |
| `stability_rank_right` | Highest rank in that stability band. |

## Complete VCF-to-output example

```python
from pathlib import Path
import csv
import json

import softmap

data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("recurrent_parent", "donor_parent"),
    cross_design="backcross",
)

mapping = softmap.fit(
    data,
    bootstrap=100,
    confidence=0.8,
    seed=7,
    bin_threshold=0.01,
)

Path("summary.json").write_text(
    json.dumps(mapping.summary(), indent=2) + "\n"
)

rows = mapping.marker_table()
with Path("markers.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

mapping.plot("map.png")
mapping.plot_marker_order("marker_order.png")
```

This produces a JSON run summary, a marker-level TSV, and two figures. The Python
object still retains the full inputs and low-level result arrays for further
analysis.

## Common input errors

| Error | Likely cause and action |
| --- | --- |
| `VCF contains multiple chromosomes` | Pass `chromosome="..."` to `read_vcf()`. |
| `VCF contains no usable ... markers` | Check SNP filtering, parent calls, cross design, sample selection, and whether the chromosome name matches. |
| `parent sample not found` / `sample not found` | Use exact sample names from the VCF header. |
| `set cross_design when parents are supplied` | Choose `backcross`, `ril`, or `doubled_haploid`. |
| `probabilities must lie in [0, 1]` | Correct the input scale and replace non-probability genotype labels before fitting. |
| `probabilities contain non-finite values` | Replace `NaN`/infinity with a meaningful probability; use 0.5 only when the binary state is genuinely unknown. |
| `marker_names length ...` | Supply exactly one name per matrix column. |
| `limited_support` status | Add informative offspring, review phase and genotype errors, reduce over-stringent confidence, or inspect binning sensitivity. |

## Generated reference

The following signatures and docstrings are generated directly from the installed
Python package.

### Main interface

::: softmap.api.read_vcf

::: softmap.api.fit

::: softmap.api.fit_likelihood

::: softmap.api.LinkageData

::: softmap.api.Map

::: softmap.api.LikelihoodMap

### Example data

::: softmap.datasets.demo

::: softmap.datasets.contemporary_hybridization

::: softmap.datasets.contemporary_map_positions

### Physical and genetic plotting

::: softmap.plotting.plot_physical_vs_genetic

::: softmap.plotting.plot_marker_order

::: softmap.plotting.plot_physical_order_grid
