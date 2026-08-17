# Guide

## F2: VCF to map

```python
import softmap

data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("PARENT_1", "PARENT_2"),
    cross_design="f2",
)
mapping = softmap.fit_f2(data, use_physical_scaffold=True)
mapping.plot("chr1-map.svg")

print(mapping.summary())
print(mapping.marker_table()[:3])
```

Replace the chromosome and parent names with exact values from your VCF header.
SoftMap excludes the parents from offspring.

The result uses complete AA/AB/BB likelihoods. Its main outputs are:

| Output | Meaning |
| --- | --- |
| `ordered_markers` | Final marker order. |
| `de_novo_order_rank` | Likelihood-only order, retained even with a physical scaffold. |
| `genetic_position_cm` | Genotype-derived Kosambi position. |
| `stability_rank_left/right` | Rank sensitivity across likelihood-ordering models. |
| `summary()` | Method, marker count, certainty, map length, and status. |

Use `use_physical_scaffold=True` when the assembly is trusted and the goal is a
clean reference-guided map. Leave it `False` when the order must be inferred only
from segregation.

## Binary crosses

For a backcross, doubled haploid, or phased RIL:

```python
data = softmap.read_vcf(
    "family.vcf.gz",
    chromosome="chr1",
    parents=("PARENT_1", "PARENT_2"),
    cross_design="backcross",  # or "doubled_haploid" / "ril"
)
mapping = softmap.fit(data, bootstrap=100, confidence=0.8, seed=7)
mapping.plot("chr1-map.svg")
```

The binary result separates a complete point order from the smaller supported
framework. A `limited_support` status is a scientific diagnostic, not a crash.

## Before trusting a map

Check that:

- each run contains only one linkage group;
- parents and cross design are correct;
- marker IDs are unique;
- missingness and isolated genotype errors are not creating false crossovers;
- conclusions distinguish de-novo order from assembly-guided order;
- the map and all filtering/settings are saved reproducibly.

Map direction is arbitrary unless physical positions or another anchor define it.

## Command line

```bash
softmap family.vcf.gz map.tsv --chromosome chr1 \
  --parents PARENT_1 PARENT_2 --cross-design f2 --physical-scaffold
```

The command prints a JSON summary and writes one marker per TSV row.

## Optional Rust acceleration

SoftMap's algorithm remains in Python/NumPy. An optional Rust extension replaces
only the expensive sparse binary-pair and dense F2 recombination kernels:

```bash
python -m pip install "softmap-rust @ git+https://github.com/kevinkorfmann/softmap-linkage.git#subdirectory=rust"
```

No API changes are required. Check activation with:

```python
import softmap

print(softmap.rust_backend_available())
```

The extension is used automatically for nontrivial workloads. Set
`SOFTMAP_DISABLE_RUST=1` to force the NumPy reference path for reproducibility
checks. SoftMap deliberately performs the final LOD reduction in NumPy so both
backends remain bit-for-bit identical.

On the development Apple ARM64 system, 200,000 sparse binary pairs improved from
1.67 s to 0.60 s (2.76x), and a complete 300-marker F2 fit improved from 4.53 s
to 2.83 s (1.60x). On Linux x86_64, the frozen 100,000-marker replay improved
from 74.87 s to 65.42 s (14% faster) with the exact same order hash, likelihood
bins, recombination estimates, LOD values, and map status.
