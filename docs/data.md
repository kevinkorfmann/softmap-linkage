# Input data

## Which data work best?

SoftMap works best when each marker has two possible, phased parental-origin states.
This is most natural for doubled-haploid, backcross, and phased RIL populations.

| Data feature | Better input | Why it helps |
| --- | --- | --- |
| Cross design | Doubled haploid, backcross, or phased RIL | Each allele can be assigned to one of two parental states. |
| Genotypes | Probabilities derived from read or genotype likelihoods | Uncertain observations contribute less without being discarded or forced into a hard call. |
| Offspring | More informative offspring with independent crossover histories | Map resolution comes from observed recombination events. More offspring usually help more than adding nearly identical markers. |
| Markers | Well-distributed, quality-controlled markers with little missing data | Errors and missingness can resemble false crossovers. Co-segregating markers are kept together as bins. |
| Linkage groups | One established linkage group per run | SoftMap orders markers but does not currently discover linkage groups. |

Values near zero or one should mean strong evidence for the two parental-origin
states. A value of 0.5 means that the binary state is unknown; it does not generally
mean “heterozygous.”

Unphased F2 and full-sib populations need an additional phase-aware model before
they are suitable for biological inference. The Rahnamae et al. (2026) F2 example
uses heterozygous calls as 0.5 to test loading, ordering, and plotting. It is a
software demonstration, not a replacement F2 linkage analysis.

## Probability matrix

SoftMap expects an offspring-by-marker matrix. Values near zero or one represent
confident inheritance states. A value of 0.5 represents no information.

```python
import numpy as np
import softmap

probabilities = np.array([
    [0.01, 0.02, 0.95],
    [0.99, 0.97, 0.04],
    [0.50, 0.92, 0.08],
])

data = softmap.LinkageData(
    probabilities=probabilities,
    marker_names=("m1", "m2", "m3"),
)
```

At least two offspring and two markers are required. Marker names must match the
number of columns.

## TSV command-line format

The command-line interface accepts markers in rows:

```text
marker  offspring_1  offspring_2  offspring_3
m1      0.01         0.99         0.50
m2      0.02         0.97         0.92
m3      0.95         0.04         0.08
```

Run it with:

```bash
softmap probabilities.tsv map.tsv --bootstrap 100
```

## Cross design

The binary state model assumes phase is known. Heterozygous genotypes in a general
F2 cannot be assigned to a single parental-origin state without additional modeling.
Do not silently encode them as hard zero or one calls.
