# SoftMap

Turn genotype data into a linkage map, with uncertainty kept visible.

![Raw evidence, genotype-only draft, and final SoftMap map for Rahnamae chromosome 3](assets/rahnamae_chr3_three_stage.png){ .softmap-hero }

*A fully non-randomized progression using Neda Rahnamae's chromosome 3 data: raw
NN/NS/SS evidence, genotype-only de-novo draft, and final reference-guided SoftMap
map. [See all eight chromosomes →](case-studies.md)*

[Make your first map](#make-your-first-map){ .md-button .md-button--primary }
[See the Rahnamae result](case-studies.md){ .md-button }

## Frozen evidence

![SoftMap matched-comparator error and runtime summary](assets/frozen_benchmark_summary.png)

Across six prespecified de-novo simulation blocks, SoftMap had lower mean
point-order error and was 3.3×–44.4× faster than the matched GUSMap or OneMap
workflow. A 100,000-marker release audit completed in 73.75–75.09 seconds using 1.568
GiB and reproduced the same scientific output twice. Caveats, failed gates, and
incomplete comparator runs remain visible in the [full benchmark summary](benchmarks.md).

## Make your first map

Install:

```bash
python -m pip install "softmap-linkage[plot] @ git+https://github.com/kevinkorfmann/softmap-linkage.git"
```

Fit one chromosome from an F2 VCF or BCF:

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

The F2 model uses all three genotype states. The physical scaffold is explicit and
optional; genetic positions are estimated from genotype likelihoods with the
Kosambi map function.

For a backcross, doubled haploid, or phased RIL, load the matching cross design and
call `softmap.fit(data)` instead.

[Input and output examples →](quickstart.md)

SoftMap fits one known linkage group at a time. It does not discover linkage
groups. It is research software, so inspect support and validate the cross model
before biological interpretation.
