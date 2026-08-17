# Python API

## Choose a fitter

| Data | Call |
| --- | --- |
| F2 AA/AB/BB | `read_vcf(..., cross_design="f2")`, then `fit_f2()` |
| Backcross, doubled haploid, phased RIL | `read_vcf(...)`, then `fit()` |
| Large binary map with cM and model-stability bands | `fit_likelihood()` |

All fitters operate on one known linkage group.

## F2

```python
data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("PARENT_1", "PARENT_2"),
    cross_design="f2",
)
mapping = softmap.fit_f2(
    data,
    use_physical_scaffold=True,
    stability_mass=0.90,
)
```

`use_physical_scaffold=True` sorts the final order by VCF position. The result
also keeps the de-novo likelihood order. Recombination fractions and Kosambi cM
positions always come from the genotype data.

```python
mapping.ordered_markers
mapping.summary()
mapping.marker_table()
mapping.plot("map.svg")
```

## Binary crosses

```python
data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("PARENT_1", "PARENT_2"),
    cross_design="backcross",
)
mapping = softmap.fit(
    data,
    bootstrap=100,
    confidence=0.8,
    seed=7,
)
```

Use `cross_design="ril"` or `"doubled_haploid"` for those crosses. For a
backcross, put the recurrent parent first.

For model-stability bands and regularized genetic coordinates, use the
likelihood-MDS interface:

```python
mapping = softmap.fit_likelihood(data)
summary = mapping.summary()
```

Important audit fields include:

| Field | Meaning |
| --- | --- |
| `selection_method` | Truth-free route used for the point order. |
| `posterior_calibration_temperature` | Applied log-odds temperature; `1.0` means none. |
| `penalized_curve_effective_degrees_of_freedom` | Fixed curve smoothness for an adverse-data route, otherwise `None`. |
| `weighted_objective_support_filter_applied` | Whether an otherwise uninformative stability family was restricted to objective-supported configurations. |
| `distance_rank_span_weight_exponent` | Rank-span weight used by the composite genetic-distance fit. |
| `stability_comparable_pair_fraction` | Fraction of marker pairs separated by the reported rank bands. |

These diagnostics are computed without truth coordinates or physical marker
positions. A non-`ok` status should be retained and reported, not silently
converted into a total order claim.

## Generated reference

### Main interface

::: softmap.api.read_vcf

::: softmap.api.fit_f2

::: softmap.api.fit

::: softmap.api.fit_likelihood

::: softmap.api.F2LinkageData

::: softmap.api.F2Map

::: softmap.api.LinkageData

::: softmap.api.Map

::: softmap.api.LikelihoodMap

### F2 model

::: softmap.core.fit_f2_likelihood_map

::: softmap.core.f2_pairwise_recombination_likelihood

### Example data

::: softmap.datasets.contemporary_hybridization_f2

::: softmap.datasets.demo

### Plotting

::: softmap.plotting.plot_f2_map

::: softmap.plotting.plot_f2_three_stage

::: softmap.plotting.plot_physical_output_grid

::: softmap.plotting.plot_physical_vs_genetic
