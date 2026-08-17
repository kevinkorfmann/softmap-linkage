# Input data

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
