# How it works

SoftMap has separate models for complete F2 genotypes and phased binary crosses.
Both fit one linkage group at a time.

## Complete F2 model

For offspring \(i\), marker \(j\), and genotype \(g\in\{AA,AB,BB\}\), the
input is

\[
p_{ijg}=\Pr(G_{ij}=g\mid Y_{ij}).
\]

SoftMap retains all three values. A heterozygote is evidence for \(AB\), not a
missing binary call.

### Pairwise recombination

For every marker pair, SoftMap uses the exact two-locus F2 genotype distribution.
That distribution integrates the two independently recombining F1 gametes in
known parental coupling phase. The recombination fraction \(r\) is fitted by
maximum likelihood and compared with the unlinked model \(r=0.5\) to obtain a LOD
weight.

This is the information loss fixed in the new F2 workflow: NN, NS, and SS all
contribute to the likelihood.

### Order and stability

The pairwise recombination and LOD matrices feed a small ensemble of
likelihood-weighted multidimensional-scaling orders. SoftMap selects the order
whose rank separations best agree with informative pairwise recombination
fractions. Variation among the ensemble orders produces model-stability rank
bands.

With `use_physical_scaffold=False`, that de-novo order is final. With
`use_physical_scaffold=True`, physical position supplies the final curated
order, while the de-novo order and stability bands remain in the result for audit.
Published genetic positions are never used in fitting.

### Genetic distance

F2 recombination estimates are transformed with the Kosambi map function:

\[
d_{cM}=25\log\left(\frac{1+2r}{1-2r}\right).
\]

At mean genotype certainty of at least 0.90, cumulative adjacent distances give
marker positions and total map length. Below 0.90, independently noisy adjacent
estimates can inflate the map severely. SoftMap therefore fits nonnegative
lengths for 16 rank segments to all informative F2 marker pairs and mildly
rebalances long versus short rank spans with exponent `-0.125`. The reported
local adjacent estimate conservatively blends the direct pair estimate with the
regularized coordinate geometry. The route uses genotype certainty only; it does
not inspect truth, physical position, or the input marker order.

A physical scaffold determines adjacency, but not the estimated recombination
fractions or segment lengths.

## Binary model

For a backcross, doubled haploid, or phased RIL, the input is

\[
p_{ij}=\Pr(Z_{ij}=1\mid Y_{ij}),
\]

where \(Z\in\{0,1\}\) has the same parental meaning at every marker. Unknown
states use \(p=0.5\).

SoftMap:

1. groups nearly co-segregating markers;
2. computes certainty-weighted expected disagreement;
3. obtains a spectral point order and local refinement;
4. resamples offspring and genotype uncertainty;
5. reports a supported framework and placement intervals.

The complete order is a useful point estimate. The framework contains only marker
relationships that reach the requested bootstrap support. Several markers can
legitimately remain tied when the offspring contain no recombination evidence that
separates them.

`fit_likelihood()` is the scalable binary alternative when explicit ties,
model-stability bands, and regularized cM coordinates are needed.

### Likelihood-MDS order

`fit_likelihood()` estimates each pairwise recombination fraction directly from
the two probabilistic parental-origin vectors and records its linkage LOD. It
then fits several LOD-weighted multidimensional-scaling geometries and converts
each geometry to a one-dimensional order with a principal curve. The ordinary
dense route uses a fixed high-information geometry. Larger maps first pool
indistinguishable likelihood bins and use deterministic landmarks, so they do
not allocate a full marker-by-marker distance matrix.

Dense adverse-data routing uses only diagnostics computed from the input
likelihoods. Mean genotype certainty in `[0.415, 0.50)` selects a low-variance
Haldane/LOD-squared, 30-dimensional penalized curve with 5.50 effective degrees
of freedom. At high certainty, a composite-distance median residual of at least
0.095 selects the corresponding 5.05-degree curve. Residuals at or above 0.1025
also activate the independent posterior-calibration guard, which divides
posterior log odds by a fixed temperature before refitting. Truth coordinates,
physical position, perturbation labels, and input marker order are absent from
all routing decisions.

### Stability and distance

Rank bands normally span the full prespecified candidate family. If that family
cannot resolve the minimum 35% of marker pairs, SoftMap makes one auditable
fallback to the point configuration plus configurations selected by the
unweighted or nine fixed LOD-weighted objectives. If that supported family is
still insufficient, SoftMap reports `insufficient_order_information` instead of
forcing a confident order.

Genetic coordinates are a nonnegative 16-segment Haldane fit to many
probabilistic marker pairs, rather than a sum of noisy adjacent estimates. The
reported local adjacent recombination estimate combines interval-specific HMM
smoothing with conservative regularization. The clean and moderate-information
routes mildly rebalance rank spans with exponent `-0.125`; the inconsistent
high-information penalized route uses uniform span weights. All choices and
diagnostics are exposed in `summary()`.

## Interpretation

- A map and its complete reversal are equivalent without an external anchor.
- A physical scaffold is reference-guided curation and should be reported as such.
- Dense markers do not replace informative offspring.
- Genotype errors can look like double crossovers and inflate map length.
- Stability intervals summarize the implemented model ensemble or bootstrap; they
  are not posterior probabilities that one complete order is correct.
