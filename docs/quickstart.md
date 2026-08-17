# Quick start

## 1. Install

```bash
python -m pip install "softmap-linkage[plot] @ git+https://github.com/kevinkorfmann/softmap-linkage.git"
```

## 2. Fit one chromosome

=== "F2"

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

    `F2LinkageData.probabilities` has shape `(offspring, markers, 3)` for AA,
    AB, and BB. `PL` or `GL` values are retained as probabilities; `GT` is
    used when likelihoods are unavailable. The two parent names must identify
    different homozygotes.

=== "Backcross / DH / RIL"

    ```python
    import softmap

    data = softmap.read_vcf(
        "family.vcf.gz",
        chromosome="chr1",
        parents=("PARENT_1", "PARENT_2"),
        cross_design="backcross",  # or "doubled_haploid" / "ril"
    )
    mapping = softmap.fit(data, bootstrap=100, seed=7)
    ```

=== "Binary probability matrix"

    Rows are offspring, columns are markers, and values are parental-state-1
    probabilities. Use `0.5` for an unknown binary state.

    ```python
    mapping = softmap.fit(
        probabilities,
        marker_names=["m1", "m2", "m3"],
        bootstrap=100,
    )
    ```

## 3. Use the result

```python
mapping.plot("map.svg")
print(mapping.summary())
print(mapping.ordered_markers[:10])
print(mapping.marker_table()[:3])
```

For F2 maps, the table includes the final rank, de-novo likelihood rank,
model-stability band, and inferred genetic position in cM. With
`use_physical_scaffold=True`, final rank follows the assembly and the de-novo rank
remains available for audit.

From the command line:

```bash
softmap family.vcf.gz map.tsv --chromosome chr1 \
  --parents PARENT_1 PARENT_2 --cross-design f2 --physical-scaffold
```

Next: [input details](data.md), [the Rahnamae benchmark](case-studies.md), or the
[API reference](api.md).
