# Translation Invariant Unbalanced Sinkhorn (Metal · PyTorch)

The optimal transport target assignment (`TargetAssignment.strategy: optimal_transport`) matches the predictions of
the network to the targets by solving an **unbalanced optimal transport (OT) problem** per sample. This directory
contains the Metal kernels of its solver, [`point_frustums/ops/sinkhorn.py`](../../sinkhorn.py) the host code and
a PyTorch implementation of the same iteration for all other devices.

```python
from point_frustums.ops.sinkhorn import sinkhorn_unbalanced_ti, sinkhorn_unbalanced_ti_batch

plan = sinkhorn_unbalanced_ti(a, b, M, reg=0.1, reg_m=(1.0, 1.0), stop_thr=1e-3, max_iter=1000)  # (N, C)
plans, n_iter = sinkhorn_unbalanced_ti_batch([(a, b, M), ...], reg=0.1)  # several problems in one launch
```

Before, the assignment used `ot.unbalanced.sinkhorn_unbalanced_translation_invariant` from POT. For the
configuration of the model (`reg_m = (1, 1)`), the new solver computes the same iterates (the plans agree up to
float32 round-off, the number of iterations and the assignments are identical on recorded batches of the training).
For `rho_1 != rho_2`, POT's iteration does not converge to the optimum, see [below](#comparison-with-pot).

## The Assignment Problem

For one sample with `N` predictions (feature vectors) and `T` targets:

- **Demand** `a ∈ R^N`: every prediction demands one unit, `a_i = 1`.
- **Supply** `b ∈ R^C`, `C = T + 1`: target `j` supplies `b_j = max(min_k, Σ top-max_k IoU(·, j))` units (the dynamic
  `k` of OTA [4]); the last column is the background, which supplies the rest, `b_T = N - Σ_{j<T} b_j`.
- **Cost** `M ∈ R^{N×C}`: classification (focal loss), `1 - IoU` and the normalized center distance for the targets,
  the classification against "no object" for the background. Pairs outside the receptive field pre-mapping get a
  constant offset, so that they are more expensive than any pre-mapped pair.

Every prediction is then assigned to the column of the plan `π` with the largest entry (after scaling each column to
a maximum of 1), the background column means negative.

OTA [4] solves this as a *balanced* problem (the marginals are constraints). Here both marginals are relaxed with a KL
penalty, so that a target does not have to receive exactly `b_j` predictions (unbalanced OT [1]).

## Derivation

### The problem and its dual

With `KL(p|q) = Σ p log(p/q) - p + q` (the generalized KL divergence, for non-negative vectors that do not have to
sum to the same mass), the entropic unbalanced OT problem is [1, 2]

```
min_{π ≥ 0}  <M, π> + ε KL(π | a ⊗ b) + ρ₁ KL(π 1 | a) + ρ₂ KL(πᵀ 1 | b)                                  (P)
```

`ε` is the entropic regularization (`TargetAssignment.epsilon`, `reg`), `ρ₁, ρ₂` the penalties of the demand and
the supply marginal (`reg_m`). Its dual is [2, Eq. 3]

```
F(f, g) = ρ₁ <a, 1 - e^{-f/ρ₁}> + ρ₂ <b, 1 - e^{-g/ρ₂}> - ε <a ⊗ b, e^{(f ⊕ g - M)/ε} - 1>                    (D)
```

with `(f ⊕ g - M)_ij = f_i + g_j - M_ij`. The optimal plan follows from the optimal potentials as
`π_ij = a_i b_j e^{(f_i + g_j - M_ij)/ε}`. In terms of the **scalings** `u = e^{f/ε}`, `v = e^{g/ε}` and the
**kernel** `K_ij = a_i b_j e^{-M_ij/ε}`:

```
π = diag(u) K diag(v)
```

### Unbalanced Sinkhorn

Maximizing `F` over `f` for a fixed `g`, the stationarity condition of `f_i` is

```
e^{-f_i/ρ₁} = e^{f_i/ε} Σ_j b_j e^{(g_j - M_ij)/ε} = e^{f_i/ε} (K v)_i / a_i
```

which gives the unbalanced Sinkhorn iteration of Chizat et al. [1], alternating in `u` and `v`:

```
u = (a / K v)^{φ₁},   v = (b / Kᵀ u)^{φ₂},   φ_k = ρ_k / (ρ_k + ε)
```

For `ρ → ∞` (`φ → 1`) this is the balanced Sinkhorn of Cuturi [5]. For finite `ρ` the iteration converges linearly,
but slowly when `ε ≪ ρ`: the dual is not invariant to the translation `(f + λ, g - λ)` (unlike the balanced one), and
an error of the translation only decays by `(ρ/(ε+ρ))²` per iteration [2, Sec. 1]. With `ε = 0.1, ρ = 1`, that is a
factor of `0.83` per iteration.

### Translation invariant dual

Séjourné et al. [2] remove the slow mode by optimizing the translation explicitly:

```
H(f̄, ḡ) = sup_λ F(f̄ + λ, ḡ - λ)
```

`H` is invariant to `(f̄ + δ, ḡ - δ)`. The entropic term of `F` does not depend on `λ`, so the optimal translation
follows from `<a, e^{-(f̄+λ)/ρ₁}> = <b, e^{-(ḡ-λ)/ρ₂}>` in closed form [2, Prop. 2]:

```
λ*(f̄, ḡ) = ρ₁ρ₂/(ρ₁+ρ₂) · log(A / B),   A = <a, e^{-f̄/ρ₁}> = <a, ū^{-ε/ρ₁}>,   B = <b, e^{-ḡ/ρ₂}> = <b, v̄^{-ε/ρ₂}>
```

Inserting `λ*` gives [2, Prop. 3]

```
H(f̄, ḡ) = const - ε <a ⊗ b, e^{(f̄ ⊕ ḡ - M)/ε} - 1> - (ρ₁+ρ₂) A^{ρ₁/(ρ₁+ρ₂)} B^{ρ₂/(ρ₁+ρ₂)}
```

### The H-Sinkhorn update

The solver alternates the *exact* maximization of `H` over `f̄` and over `ḡ` (H-Sinkhorn [2, Sec. 3]). The
stationarity condition of `f̄_i` is

```
A^{-ρ₂/(ρ₁+ρ₂)} B^{ρ₂/(ρ₁+ρ₂)} e^{-f̄_i/ρ₁} = e^{f̄_i/ε} (K v̄)_i / a_i
```

i.e. the Sinkhorn condition for the translated potential `f̄ + λ*`. Since `A` depends on all of `f̄`, the solution is
the standard update `û = (a / K v̄)^{φ₁}` (`f̂ = ε log û`) plus a constant: with `f̄ = f̂ - δ`, `A = e^{δ/ρ₁} Â`
(`Â = <a, û^{-ε/ρ₁}>`), and solving the condition for `δ` gives

```
δ = ε ρ₂ / (ε + ρ₁ + ρ₂) · log(Â / B)
```

So, with `θ_k = ρ_k / (ε + ρ₁ + ρ₂)` and symmetrically for `v̄`:

```
û = (a / K v̄)^{φ₁},      ū = û · (B / Â)^{θ₂},     Â = <a, û^{-ε/ρ₁}>,  B = <b, v̄^{-ε/ρ₂}>
v̂ = (b / Kᵀ ū)^{φ₂},     v̄ = v̂ · (A / B̂)^{θ₁},     B̂ = <b, v̂^{-ε/ρ₂}>,  A = <a, ū^{-ε/ρ₁}>
```

For `ρ₁ = ρ₂ = ρ` this is Prop. 4 of [2]: in the log domain, `f̄ = f̂ + ξ (Smin_ρ(f̂) - Smin_ρ(ḡ))` with
`ξ = ε/(ε+2ρ)` and `Smin_ρ(f) = -ρ log <a, e^{-f/ρ}>`. The paper proves a contraction rate that improves on the
standard iteration by the factor `κ_ε(a) κ_ε(b)` [2, Thm. 1].

After each iteration, the dual iterate is translated by `λ*`:

```
t = e^{λ*/ε} = (A / B)^{ρ₁ρ₂ / ((ρ₁+ρ₂) ε)},    u = ū · t,    v = v̄ / t
```

and the iteration stops when the relative change of `u` and `v` is below `stop_thr` (the criterion of POT):

```
err = ½ ( max|u - u_prev| / max(|u|, |u_prev|, 1) + max|v - v_prev| / max(|v|, |v_prev|, 1) )
```

If `Kᵀ u` has a zero or `ū`, `v̄` are not finite, the iteration stops and keeps the previous `u, v` (as POT does). The
plan is `π = diag(u) K diag(v)`.

### Comparison with POT

POT's `sinkhorn_unbalanced_translation_invariant` (0.9.7) applies the factors before the normalization:

```
û' = (a / K v̄)^{φ₁} · <b, v̄^{ε/ρ₂}>^{k₂},  ū = û' <a, û'^{-ε/ρ₁}>^{-θ₂},   k_k = ρ_k² / ((ε+ρ_k)(ρ₁+ρ₂))
v̂' = (b / Kᵀ ū)^{φ₂} · A^{k₁},             v̄ = v̂' <b, v̂'^{-ε/ρ₂}>^{-θ₁}
```

- A constant factor in the update of `ū` only translates `(f̄, ḡ)`, which `H` and the translated `(u, v)` do not
  depend on, so the first factor is harmless.
- The factor `A^{k₁}` of `v̄` is applied before the normalization, which turns it into `A^{k₁ (1 + ε θ₁ / ρ₂)}`. The
  exponent equals the `θ₁` of the exact update only for `ρ₁ = ρ₂`.

Verified numerically (float64, `test/ops/test_sinkhorn.py` and on recorded problems of the training): for
`ρ₁ = ρ₂` the iterates of both agree to `1e-13`; for `ρ = (1, 3)` and `(3, 1)`, POT's iteration converges to a plan
that is `8 %` and `45 %` (maximum relative difference) away from the optimum of the classical unbalanced Sinkhorn,
the exact update reaches it to `1e-11`, in fewer than half the iterations. This is a likely reason why the published
code did not seem to match the paper.

## From the Derivation to the Code

### PyTorch (`_sinkhorn_torch`, CPU/CUDA and the fallback)

The updates above, line by line. The stopping criterion and the error check stay on the device: a problem that is
done (converged or numerical error) is frozen with `torch.where`, and the host only checks every `check_every = 10`
iterations whether to stop. The result is the same as with a check in every iteration (POT synchronizes with the
device several times per iteration).

### Metal (`sinkhorn.metal`)

All problems of a batch are concatenated, the rows of every problem are split into blocks of 256 rows (one
threadgroup each, a threadgroup never spans two problems). Sums over the rows are written as one partial per block and
reduced by the next kernel, so the result is deterministic (no atomics). One iteration consists of five launches:

| Kernel | Grid | Computes |
|---|---|---|
| `sinkhorn_rows_u_hat` | blocks | `û_i = (a_i / (K v̄)_i)^{φ₁}` (one thread per row), partials of `Â` |
| `sinkhorn_rows_ktu` | blocks | `ū = û (B/Â)^{θ₂}`, partials of `Kᵀ ū` and of `A = <a, ū^{-ε/ρ₁}>`, error flags |
| `sinkhorn_columns` | problems | `Kᵀ ū`, `v̂`, `v̄ = v̂ (A/B̂)^{θ₁}`, `B` of the next iteration, `t`, `v = v̄/t`, the change of `v` |
| `sinkhorn_rows_translate` | blocks | `u = ū t`, partials of the change of `u` |
| `sinkhorn_finalize` | problems | `err`, the iteration count and the stopping decision |

- **State on the device:** every kernel first reads the status of its problem and returns immediately if it is not
  running anymore. The host enqueues the iterations and synchronizes only every `check_every = 5` iterations to see
  whether all problems are done.
- **Reading `K`:** `K` is read twice per iteration (`K v̄` and `Kᵀ ū`) and never written during the iteration. For
  `K v̄`, each thread computes the dot product of one row (the rows' powers are then evaluated by all lanes in
  parallel). For `Kᵀ ū`, a simdgroup takes 32 rows, the lanes scale `ū` of their row in parallel and broadcast it,
  then the columns are distributed over the lanes (accumulated in registers, `C ≤ 256`), so that each row is read
  coalesced. The accumulators are reduced over the simdgroups in threadgroup memory.
- **In place:** `M` is turned into `K` and finally into the plan in place.
- **Fast math:** the Metal compiler assumes that there are no NaNs, which can remove `isnan`/`isinf`. The error check
  tests the bit pattern of the exponent instead.
- **Limits:** 32 bit indices; problems with more than 256 columns use the PyTorch implementation.

## Performance

Recorded batches of the training run (nuScenes, 4 samples per batch, `N = 22725` feature vectors, `T = 5…75`
targets, `ε = 0.1`, `stop_thr = 1e-3`, about 20 to 30 iterations), Apple M-series GPU (MPS):

| | per problem |
|---|---|
| POT | 80-125 ms |
| PyTorch implementation (host check every 10 iterations) | 25-35 ms |
| Metal, one problem per call | 9-13 ms |
| Metal, batch of 4 problems | 6-7 ms |

The whole loss of a training step (including the cost matrices and the other losses) went from about 590 ms to
about 255 ms per batch, measured with `get_losses` on 8 real batches.

## References

1. L. Chizat, G. Peyré, B. Schmitzer, F.-X. Vialard. *Scaling Algorithms for Unbalanced Optimal Transport Problems.*
   Mathematics of Computation 87(314), 2018. [arXiv:1607.05816](https://arxiv.org/abs/1607.05816)
2. T. Séjourné, F.-X. Vialard, G. Peyré. *Faster Unbalanced Optimal Transport: Translation Invariant Sinkhorn and 1-D
   Frank-Wolfe.* AISTATS 2022. [arXiv:2201.00730](https://arxiv.org/abs/2201.00730)
3. G. Peyré, M. Cuturi. *Computational Optimal Transport.* Foundations and Trends in Machine Learning 11(5-6), 2019.
   [arXiv:1803.00567](https://arxiv.org/abs/1803.00567) (Chapter 4 on Sinkhorn, Section 10.2 on unbalanced OT)
4. Z. Ge, S. Liu, Z. Li, O. Yoshie, J. Sun. *OTA: Optimal Transport Assignment for Object Detection.* CVPR 2021.
   [arXiv:2103.14259](https://arxiv.org/abs/2103.14259)
5. M. Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* NeurIPS 2013.
   [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
6. R. Flamary et al. *POT: Python Optimal Transport.* JMLR 22(78), 2021. The previous solver:
   `ot.unbalanced.sinkhorn_unbalanced_translation_invariant`.
