# Algorithm

## Aim and notation

SoftMap estimates a supported marker order for one linkage group when inheritance
states are uncertain. Let (N) denote offspring and (M) denote markers. For
offspring (i) and marker (j), the input is

\[
p_{ij}=\Pr(Z_{ij}=1\mid Y_{ij}),
\]

where (Z_{ij}\in\{0,1\}) is the phased parental-origin state and (Y_{ij})
denotes the observed sequence or genotype evidence. Thus, (p_{ij}=0.99) is strong
evidence for state 1, (p_{ij}=0.01) is strong evidence for state 0, and
(p_{ij}=0.5) is uninformative. The model requires the two states to have the same
biological meaning across all markers. This condition is natural in a doubled
haploid or backcross and can be obtained for a recombinant inbred line (RIL) when
phase is known.

The default pipeline is

\[
\text{probabilities}\longrightarrow\text{co-segregation bins}
\longrightarrow\text{point order}\longrightarrow\text{bootstrap orders}
\longrightarrow\text{framework and intervals}.
\]

The important conceptual separation is between a point estimate and supported
claims. The point order provides one useful arrangement of all marker bins. The
framework contains only markers whose relative order is reproducible at the chosen
confidence threshold. Placement intervals describe where the remaining bins can be
placed relative to that framework.

## Expected disagreement between two markers

For markers (j) and (k), the posterior probability that their latent states
differ in offspring (i) is

\[
u_{ijk}
=p_{ij}(1-p_{ik})+(1-p_{ij})p_{ik}.
\]

This is close to zero when both markers confidently support the same state, close to
one when they confidently support opposite states, and equal to one half when either
marker is uninformative. SoftMap also defines a certainty weight

\[
w_{ijk}=|2p_{ij}-1|\,|2p_{ik}-1|.
\]

The pairwise dissimilarity is the certainty-weighted expected disagreement

\[
d_{jk}
=\frac{\sum_{i=1}^{N}w_{ijk}u_{ijk}}
       {\sum_{i=1}^{N}w_{ijk}}.
\]

If the denominator is numerically zero, SoftMap sets (d_{jk}=0.5), representing
no linkage information. The intuition is that an offspring contributes strongly
only when both marker states are known with confidence. This avoids treating a
missing or ambiguous observation as half of a biological recombination event.
Mathematically, the calculation uses the marginal state probabilities and therefore
approximates the two marker posteriors as independent conditional on their local
observations.

## Co-segregation bins

Finite mapping populations cannot distinguish markers that share essentially the
same segregation pattern. Ordering thousands of such markers separately would
create false precision and waste computation. SoftMap therefore assigns an
information score

\[
I_j=\frac{1}{N}\sum_{i=1}^{N}|2p_{ij}-1|
\]

and considers high-information markers first. Candidate neighbors are found in the
certainty-scaled posterior space with coordinates

\[
\boldsymbol{x}_j=2(\boldsymbol{p}_j-0.5).
\]

A candidate joins representative (r) when (d_{rj}\leq\tau), where (	au)
is `bin_threshold`. Membership is deliberately non-transitive: a marker can join a
fixed representative, but two bins are not merged through a chain of slightly
different markers. This prevents successive one-recombinant differences from
collapsing an entire chromosome.

Evidence within bin (B) is pooled on the log-odds scale. With clipped probabilities
to avoid infinite values,

\[
\ell_{ij}=\log\frac{p_{ij}}{1-p_{ij}},\qquad
\ell_{iB}=\operatorname{clip}\!\left(\sum_{j\in B}\ell_{ij},-30,30\right),
\]

and the pooled probability is

\[
p_{iB}=\frac{1}{1+\exp(-\ell_{iB})}.
\]

The sum treats consistent marker observations as accumulating evidence for a common
bin state. It is an evidence-pooling approximation rather than a read-level model of
dependence among markers. Markers within a bin remain unresolved: their adjacency in
the output is meaningful, but their internal order is not.

When `bin_threshold=None`, SoftMap evaluates a fixed increasing threshold grid. It
selects the threshold after the largest substantial reduction in bin count and also
applies a ceiling based on offspring information. This heuristic seeks the boundary
between small observational differences and genuinely distinct crossover patterns;
it does not use physical positions or a known map.

## Spectral point ordering

Let (K\leq M) be the number of marker bins and let (D=(d_{jk})) be their
dissimilarity matrix. For (K\leq2500), SoftMap converts dissimilarities to an
affinity matrix

\[
A_{jk}=\exp\!\left(-\frac{d_{jk}}{s}\right),\qquad A_{jj}=0,
\]

where (s) is the median positive pairwise dissimilarity. If
(G=\operatorname{diag}(\sum_k A_{jk})), the normalized graph Laplacian is

\[
L=I-G^{-1/2}AG^{-1/2}.
\]

The eigenvector associated with the second-smallest eigenvalue of (L), commonly
called the Fiedler vector, gives a one-dimensional seriation coordinate. Sorting
markers by this coordinate places strongly connected markers near one another. The
biological intuition is that nearby loci tend to show similar inheritance patterns,
so a chromosome should appear as an approximately one-dimensional path through the
graph of segregation similarity.

The initial spectral order is refined by minimizing adjacent path cost

\[
C(\pi)=\sum_{t=1}^{K-1}d_{\pi_t,\pi_{t+1}},
\]

where (pi) is a permutation of the bins. A 2-opt search repeatedly reverses the
best improving segment, removing avoidable path crossings. For more than 2500 bins,
SoftMap constructs a sparse nearest-neighbor graph and uses local insertion
polishing rather than a dense (K\times K) matrix.

Chromosome orientation is not identifiable from segregation data alone. Therefore,
(pi) and its reversal represent the same map unless an external anchor defines
left and right.

## One multipoint smoothing step

Pairwise similarities can be disrupted by isolated uncertain calls. After the first
order, SoftMap applies one two-state hidden Markov model smoothing step separately to
each offspring. The transition matrix is

\[
T(r)=
\begin{pmatrix}
1-r&r\\
r&1-r
\end{pmatrix},
\]

where (r) is the median disagreement between adjacent bins, clipped to
([10^{-4},0.05]). The emission weights are

\[
e_{it}(0)=1-p_{i,\pi_t},\qquad e_{it}(1)=p_{i,\pi_t}.
\]

A forward-backward calculation produces smoothed posterior state probabilities.
The spectral ordering is then recomputed once. Intuitively, a single ambiguous call
inside a long inherited block is pulled toward its neighbors, whereas a consistent
state transition across adjacent markers remains evidence for a crossover. The
fixed transition estimate is a stabilizing heuristic; it is not a fitted genetic
distance model.

## Bootstrap uncertainty

The point order alone does not reveal which local decisions are stable. For each
bootstrap replicate (b=1,\ldots,B), SoftMap samples (N) offspring with
replacement. By default it also draws a latent state

\[
Z_{ij}^{(b)}\sim\operatorname{Bernoulli}(p_{ij})
\]

for each selected observation. This second sampling step propagates genotype
uncertainty as well as sampling variation among offspring. The sampled zero and one
states are softened to 0.001 and 0.999 to avoid numerical ties. The full-data bins
remain fixed; every replicate reorders their pooled probability profiles, applies
the smoothing step, and orders them again.

Each replicate has an arbitrary orientation, so its forward and reverse orders are
compared with the full-data order and the closer orientation is retained. Let
(R_{bj}) denote the rank of bin (j) in replicate (b). The reported point order
sorts bins by their mean bootstrap rank,

\[
\bar R_j=\frac{1}{B}\sum_{b=1}^{B}R_{bj}.
\]

This consensus reduces dependence on a single spectral solution. The bootstrap
describes stability under the implemented resampling scheme; it should not be read
as a posterior distribution over all possible biological maps.

## Pairwise precedence and the supported framework

SoftMap summarizes the bootstrap maps with the precedence matrix

\[
P_{jk}=\frac{1}{B}\sum_{b=1}^{B}
\mathbf{1}\!\left(R_{bj}<R_{bk}\right).
\]

Thus, (P_{jk}=0.97) means that marker bin (j) occurs before bin (k) in 97% of
the aligned bootstrap orders. Given a requested confidence (c), the framework is
a long subsequence of the mean-rank order for which every earlier selected marker
precedes every newly admitted marker with (P_{jk}\geq c). The greedy construction
tries every possible first anchor and retains the longest compatible chain.

The resulting guarantee is pairwise: every selected comparison passes the threshold.
It is not the probability that the complete framework order is simultaneously
correct. If fewer than two bins pass, the two endpoint bins are retained as a minimal
orientation-free scaffold. A framework with only two markers should be interpreted
as insufficient internal ordering information, not as a resolved map.

## Placement intervals

Non-framework bin (j) is placed relative to framework anchors
(f_1,\ldots,f_H). For an overall interval confidence (c), each boundary uses

\[
c_{\mathrm{side}}=1-\frac{1-c}{2}=\frac{1+c}{2}.
\]

The supported left and right anchor indices are

\[
L_j=\max\{h:P_{f_hj}\geq c_{\mathrm{side}}\},
\qquad
U_j=\min\{h:P_{jf_h}\geq c_{\mathrm{side}}\}.
\]

If no anchor satisfies one side, that boundary remains outside the framework. A
framework marker has (L_j=U_j) at its own rank. Splitting the error budget between
the two boundaries is analogous to a Bonferroni construction. Because the bounds
are derived from dependent bootstrap order comparisons, they are support intervals,
not exact parametric confidence intervals.

## Diagnostic information and computational scale

SoftMap reports

\[
N_{\mathrm{eff}}=N\left(\frac{1}{NM}\sum_{i=1}^{N}\sum_{j=1}^{M}
|2p_{ij}-1|\right)
\]

as an effective-offspring-information diagnostic. It equals (N) when every call
is certain and approaches zero when all calls are uninformative. It is not a formal
effective sample size and should not replace inspection of missingness, segregation
distortion, or crossover coverage.

After binning, dense pairwise distances require (O(NK^2)) arithmetic and
(O(K^2)) memory; dense eigendecomposition can require (O(K^3)) time. Bootstrap
cost is approximately multiplied by (B). Binning is therefore both a biological
statement about unresolved co-segregating loci and an important computational
reduction. Datasets above 2500 bins use a sparse neighbor graph to avoid the dense
distance matrix.

## What the result does and does not claim

The strongest output is the supported framework and the placement intervals around
it. The complete point order is useful for visualization and hypothesis generation,
but weakly supported local permutations should not be interpreted as resolved
recombination events. SoftMap does not infer linkage groups, phase general F2 or
full-sib crosses, estimate centimorgan distances, or determine chromosome
orientation. Those tasks require additional models or external information.
