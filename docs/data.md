# Input data

SoftMap fits one known linkage group at a time.

| Cross | Input states | Fit |
| --- | --- | --- |
| F2 | AA, AB, BB probabilities | `fit_f2()` |
| Backcross | two phased parental states | `fit()` |
| Doubled haploid | two phased parental states | `fit()` |
| Phased RIL | two phased parental states | `fit()` |

Full-sib crosses are not currently modeled.

## VCF or BCF

For F2 data, supply two different homozygous parents:

```python
import softmap

data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("PARENT_1", "PARENT_2"),
    cross_design="f2",
)
mapping = softmap.fit_f2(data, use_physical_scaffold=True)
```

The returned `F2LinkageData` contains an offspring × marker × 3 array in
parent-oriented AA/AB/BB order. SoftMap uses `PL` first, then `GL`, then
`GT`. A missing F2 call receives the expected 1:2:1 prior and contributes no
spurious genotype certainty.

For a binary cross:

```python
data = softmap.read_vcf(
    "family.bcf",
    chromosome="chr1",
    parents=("PARENT_1", "PARENT_2"),
    cross_design="backcross",  # or "doubled_haploid" / "ril"
)
mapping = softmap.fit(data)
```

Parent names are exact VCF sample IDs and are excluded from offspring by default.
Records must be passing biallelic SNPs. Parent-oriented loading prevents REF/ALT
changes from reversing inheritance states between markers.

## Physical scaffold

`use_physical_scaffold=True` is an explicit reference-guided analysis:

- final marker order follows VCF `POS`;
- the de-novo genotype-likelihood order remains in `de_novo_order_rank`;
- recombination fractions and Kosambi cM positions are estimated from the F2
  offspring, not copied from physical or published genetic positions.

Leave it `False` for fully de-novo ordering.

## Arrays

Binary data use an offspring × marker matrix:

```python
data = softmap.LinkageData(
    probabilities=probabilities,
    marker_names=("m1", "m2", "m3"),
)
```

Values are probabilities of parental state 1. Use `0.5` for missing information.

F2 data use an offspring × marker × 3 array whose last axis is AA, AB, BB:

```python
data = softmap.F2LinkageData(
    probabilities=f2_probabilities,
    marker_names=("m1", "m2", "m3"),
    physical_positions=physical_positions,
)
mapping = softmap.fit_f2(data, use_physical_scaffold=True)
```

Each three-state vector must be finite, nonnegative, and sum to one.

## Command line

F2 VCF/BCF:

```bash
softmap family.vcf.gz map.tsv --chromosome chr1 \
  --parents PARENT_1 PARENT_2 --cross-design f2 --physical-scaffold
```

Binary probability TSVs use markers in rows:

```text
marker  offspring_1  offspring_2  offspring_3
m1      0.01         0.99         0.50
m2      0.02         0.97         0.92
```

```bash
softmap probabilities.tsv map.tsv --bootstrap 100
```
