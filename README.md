# SoftMap

[![Checks](https://github.com/kevinkorfmann/softmap-linkage/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinkorfmann/softmap-linkage/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-0f766e.svg)](https://kevinkorfmann.github.io/softmap-linkage/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/downloads/)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-4c1d95.svg)](https://github.com/kevinkorfmann/softmap-linkage)
[![License: MIT](https://img.shields.io/badge/license-MIT-c51b8a.svg)](https://github.com/kevinkorfmann/softmap-linkage/blob/main/LICENSE)
[![Status: research software](https://img.shields.io/badge/status-research%20software-fa9fb5.svg)](#scope-and-assumptions)

SoftMap is for researchers who already know which markers belong to one linkage
group and need to determine their order from uncertain genotype data. It converts
phased, two-state data from doubled-haploid, backcross, or RIL populations into a
complete marker order, co-segregation bins, and a confidence-supported framework
with placement uncertainty. It does not discover linkage groups, infer phase for
general F2 or full-sib crosses, or estimate centimorgan distances.

[Documentation](https://kevinkorfmann.github.io/softmap-linkage/) ·
[API reference](https://kevinkorfmann.github.io/softmap-linkage/api/)

## Use your own data

Install SoftMap with plotting support:

```bash
python -m pip install "softmap-linkage[plot] @ git+https://github.com/kevinkorfmann/softmap-linkage.git"
```

For a backcross VCF/BCF with parental samples, list the recurrent parent first and
fit one chromosome at a time:

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

Use `cross_design="ril"` or `"doubled_haploid"` for those populations. SoftMap
also accepts an offspring-by-marker NumPy probability matrix. It returns a complete
marker order, a confidence-supported framework, per-marker bins and placement
bounds, a JSON-serializable summary, and Matplotlib figures.

## Python API at a glance

| Call or type | Input | Output or purpose |
| --- | --- | --- |
| `softmap.read_vcf(path, ...)` | `.vcf`, bgzipped `.vcf.gz`, or `.bcf` | `LinkageData` containing probabilities, marker names, physical positions, and chromosome label. |
| `softmap.LinkageData(...)` | Probability matrix plus names and optional metadata | Validated container that keeps marker metadata aligned with matrix columns. |
| `softmap.fit(data, ...)` | Variant-file path, `LinkageData`, or probability matrix | `Map` with a complete order, supported framework, placement bounds, summaries, and plots. |
| `softmap.fit_likelihood(data, ...)` | Same data inputs as `fit()` | `LikelihoodMap` with a robust complete order and candidate-model stability rank bands. |
| `mapping.summary()` | Fitted `Map` | JSON-serializable status and dataset/bin/framework counts. |
| `mapping.marker_table()` | Fitted `Map` | One JSON-serializable result row per input marker. |
| `mapping.plot(path, colors=...)` | Fitted `Map` | Matplotlib overview figure using customizable state-0/uncertain/state-1 colors. |

Main signatures:

```python
softmap.read_vcf(
    path,
    chromosome=None,
    samples=None,
    parents=None,
    cross_design="auto",
) -> softmap.LinkageData

softmap.fit(
    data,
    marker_names=None,
    bootstrap=20,
    confidence=0.8,
    seed=1,
    bin_threshold=0.01,
) -> softmap.Map

softmap.fit_likelihood(
    data,
    marker_names=None,
    stability_mass=0.90,
    maximum_smacof_iterations=500,
) -> softmap.LikelihoodMap
```

The [full API reference](https://kevinkorfmann.github.io/softmap-linkage/api/)
documents every argument, conversion rule, returned field, and common error.

![Physical and genetic-map order before and after](docs/assets/physical_order_before_after_grid.png)

![Marker order before and after](docs/assets/marker_order_before_after.png)

## Try the included demo

Fit and plot a small simulated example:

```python
import softmap

data = softmap.demo()
mapping = softmap.fit(data)
figure = mapping.plot("map.png")

print(mapping.summary())
print(mapping.ordered_markers[:5])
print(mapping.marker_table()[:5])
```

Open `map.png` to inspect the probability blocks and inferred-versus-reference
order. The returned Matplotlib figure can be customized, and `marker_table()`
contains the bin, order rank, framework rank, and placement interval for each
marker.

## Use a VCF or BCF

The standard file input is a VCF, bgzipped VCF, or BCF containing biallelic SNP
genotypes from one chromosome:

```python
data = softmap.read_vcf("offspring.vcf.gz", chromosome="chr1")
mapping = softmap.fit(data)
```

For a parent-oriented cross, provide the two parental sample names and cross design
separately. In a backcross, list the recurrent parent first:

```python
data = softmap.read_vcf(
    "family.bcf",
    chromosome="chr1",
    parents=("recurrent_parent", "donor_parent"),
    cross_design="backcross",
)
```

| `read_vcf()` argument | Meaning |
| --- | --- |
| `path` | `.vcf`, `.vcf.gz`, or `.bcf` input path. |
| `chromosome` | Exact contig to retain. If omitted, usable markers must come from one contig. |
| `samples` | Optional offspring names and row order; otherwise all non-parent samples are used. |
| `parents` | `(state0_parent, state1_parent)`; for a backcross, state 0 is the recurrent parent. |
| `cross_design` | `auto`, `backcross`, `ril`, or `doubled_haploid`; explicit design is required with parents. |

Parent samples are excluded from offspring automatically. SoftMap uses `PL` or
`GL` genotype likelihoods when available; otherwise it maps compatible hard calls
to 0.01 or 0.99. Calls with no usable likelihood or compatible `GT` become 0.5.

The loader retains passing, biallelic SNPs with two usable inheritance states. It
skips indels, multiallelic variants, failed filters, monomorphic markers, and
parent-uninformative markers. VCF `ID` becomes the marker name, falling back to
`CHROM:POS`; VCF `POS` is retained as the physical coordinate.

Without parents, automatic conversion is suitable only when REF/ALT coding already
has a consistent binary-state orientation across markers. A suitable single-contig
file can be passed directly as `softmap.fit("offspring.vcf.gz")`. Use
`softmap.read_vcf()` when selecting a chromosome, selecting offspring, or supplying
parents.

The loader returns a `LinkageData` object:

| Field | Meaning |
| --- | --- |
| `probabilities` | Offspring-by-marker state-1 probability matrix. |
| `marker_names` | Retained VCF IDs or `CHROM:POS` fallback names. |
| `physical_positions` | One-based VCF positions in base pairs. |
| `label` | Selected or detected chromosome name. |
| `reference_positions` | Optional known genetic positions; `None` for VCF input. |

## Use a probability matrix

Alternatively, your own data can be an offspring-by-marker NumPy array containing
probabilities between zero and one:

```python
mapping = softmap.fit(probabilities, marker_names)
```

Rows are offspring and columns are markers. Values must be finite probabilities in
`[0, 1]`, with at least two offspring and two markers. Near-zero and near-one values
mean strong evidence for the two parental states; 0.5 means the binary state is
unknown.

To retain metadata with the matrix, construct `LinkageData`:

```python
data = softmap.LinkageData(
    probabilities=probabilities,
    marker_names=("m1", "m2", "m3"),
    label="Linkage group 1",
    physical_positions=physical_positions,
)
```

## Fit controls

```python
mapping = softmap.fit(
    data,
    bootstrap=100,
    confidence=0.8,
    seed=7,
    bin_threshold=0.01,
)
```

| Argument | Default | Meaning |
| --- | ---: | --- |
| `bootstrap` | `20` | Number of resampled maps used to measure order support. Use at least 100 for final analysis. |
| `confidence` | `0.8` | Minimum pairwise support for framework-marker order. |
| `seed` | `1` | Reproducible bootstrap seed; use `None` for a non-fixed seed. |
| `bin_threshold` | `0.01` | Maximum expected disagreement for merging co-segregating markers; use `None` for automatic selection. |

## Outputs

`softmap.fit()` returns a `Map` object. Map direction is arbitrary: reversing a
complete order describes the same linkage map.

| Member | Meaning |
| --- | --- |
| `mapping.ordered_markers` | Every input marker in inferred order. |
| `mapping.framework_markers` | Representative markers whose relative order reaches the confidence threshold. |
| `mapping.summary()` | JSON-serializable run status and input/bin/framework counts. |
| `mapping.marker_table()` | One JSON-serializable result row per input marker. |
| `mapping.plot(path, colors=...)` | Probability-block overview as a Matplotlib figure. Colors represent state 0, uncertainty, and state 1. |
| `mapping.plot_marker_order(path)` | Input-order versus inferred-order figure. |
| `mapping.data` | Normalized `LinkageData` used by the fit. |
| `mapping.result` | Low-level `SoftMapResult` arrays for advanced analysis. |

The marker table contains zero-based ranks:

| Field | Meaning |
| --- | --- |
| `marker` | Input marker name. |
| `bin` | Co-segregation-bin identifier. |
| `order_rank` | Rank of the marker's bin in the complete inferred bin order. |
| `is_representative` | Whether the marker represented its bin during fitting. |
| `framework_rank` | Supported framework rank, or `None`. |
| `interval_left`, `interval_right` | Placement bounds relative to framework anchors. `-1` and the framework size represent positions outside the end anchors. |

The `status` in `mapping.summary()` is `ok` when at least three framework markers
are supported and `limited_support` otherwise. Limited support is a diagnostic
result, not a software failure.

For a robust complete order with model-sensitivity rank bands, use:

```python
likelihood_mapping = softmap.fit_likelihood(
    data,
    stability_mass=0.90,
    maximum_smacof_iterations=500,
)
```

`stability_mass` is the central fraction of candidate-model ranks retained in each
marker's band. `maximum_smacof_iterations` limits optimization work for each
candidate ordering model.

Its `marker_table()` returns `order_rank`, `stability_rank_left`, and
`stability_rank_right`. These are stability bands across candidate likelihood-MDS
models, not bootstrap confidence intervals.

## Command line

The CLI accepts the same variant-file formats:

```bash
softmap family.vcf.gz map.tsv \
  --chromosome chr1 \
  --parents recurrent_parent donor_parent \
  --cross-design backcross \
  --bootstrap 100
```

It also retains support for the marker-by-offspring probability TSV format:

```bash
softmap probabilities.tsv map.tsv --bootstrap 100
```

The command writes one marker row per input marker to `map.tsv` and prints a JSON
run summary to standard output.

## Scope and assumptions

The best inputs are doubled-haploid, backcross, or phased RIL data with genotype
probabilities, low missingness, and enough offspring to observe informative
crossovers. More offspring generally improve ordering more than additional markers
that share the same segregation pattern.

Use at least 100 bootstrap replicates for a final analysis. SoftMap currently
supports one linkage group from phased binary parental-origin data, such as a
doubled-haploid or backcross design. It does not infer linkage groups or phase
general F2 and full-sib data.

## Published-data example

The repository includes a reproducible example using chromosome 1 from the
[Rahnamae et al. (2026) contemporary hybridization dataset](https://github.com/nedarahnama/Contemporary_hybridization):

```bash
python examples/contemporary_hybridization.py
```

The example downloads the source table directly, samples 100 markers, fits the map,
and writes linkage-order and physical-versus-genetic figures. Heterozygous and
missing F2 calls are represented as uninformative probabilities, so this is a
software demonstration rather than a replacement analysis of that cross.

## Documentation

The [step-by-step guide](https://kevinkorfmann.github.io/softmap-linkage/guide/)
walks through installation, data preparation, validation, fitting, interpretation,
and troubleshooting. The
[input guide](https://kevinkorfmann.github.io/softmap-linkage/data/) describes
cross-design assumptions, and the
[API reference](https://kevinkorfmann.github.io/softmap-linkage/api/) documents
every input, parameter, output field, and common error.
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
