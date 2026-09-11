# Radial Johnson IASBS for Fixed-Composition CuAu

## Mathematical specification, exact algorithms, implementation plan, and experiment protocol

**Purpose.** This document specifies a stronger fixed-composition IASBS reference for the CuAu benchmark. It replaces the distance-1 Johnson walk by a time-dependent mixture of Johnson distance shells while preserving the same hard composition constraint and the same permutation-pullback intertwining mechanism.

The document is intended to be executable as an implementation plan. It covers:

- the exact state space and target;
- the radial Johnson reference;
- a proof of admissibility under IASBS conditions (C1)--(C4);
- exact reduced reference kernels for arbitrary shell mixtures;
- exact reference-bridge sampling;
- exact AS and ASBS terminal labels;
- the dependence of computational cost on shell distance `j`;
- a scalable controller parameterization that avoids enumerating the exponentially large shell;
- exact simulation of that represented controlled CTMC;
- CuAu-specific schedules, ablations, metrics, and stop/go tests;
- brute-force mathematical verification tests before any large run.

This is an extension of the existing `CUAU_IASBS_EXPERIMENT_PLAYBOOK.md`. The energy backend, PT reference, L1_0 order parameters, and scientific evaluation protocol from that document remain applicable. This document supersedes the **single-swap reference/controller implementation** portions for the radial experiment.

---

# 0. Executive conclusion

For binary fixed composition

\[
\Omega_{n,k}=\{x\in\{0,1\}^n:\sum_i x_i=k\},
\]

let `d(x,y)` be Johnson distance, i.e. the number of occupied sites replaced when moving from `x` to `y`:

\[
d(x,y)=k-|\operatorname{supp}(x)\cap\operatorname{supp}(y)|.
\]

For each shell `j=1,...,D`, `D=min(k,n-k)`, define the shell averaging operator

\[
(K_jf)(x)=\frac{1}{v_j}\sum_{y:d(x,y)=j}f(y),
\qquad
v_j=\binom{k}{j}\binom{n-k}{j}.
\]

The proposed reference generator is

\[
\boxed{
\mathcal L_t=\sum_{j=1}^{D}\gamma_j(t)(K_j-I),
\qquad \gamma_j(t)\ge 0.
}
\]

The reference rate is therefore

\[
\boxed{
r_t(y,x)=\frac{\gamma_j(t)}{v_j}\quad\text{when }d(x,y)=j.
}
\]

This family is admissible for the same coordinate-permutation pullbacks used by the current fixed-composition construction:

- **C1 reconstruction:** every shell edge `x -> y` can be represented by a coordinate permutation `g` with `gx=y`, hence `f(y)/f(x)=(D^g f)(x)/f(x)`.
- **C2 intertwining:** Johnson distance is permutation invariant, so every `K_j` and every radial mixture commute with every coordinate permutation.
- **C3 terminal tractability:** the full reference kernel is radial and is computed by a `(D+1) x (D+1)` matrix exponential, regardless of the full state count or shell cardinality.
- **C4 coverage:** coordinate permutations map every `x` to every `y` in `Omega_{n,k}`; in particular, every active shell edge is covered.

Moreover, among reference generators invariant under the full coordinate-permutation group, this is the maximal family: an invariant rate can depend on `(x,y)` only through Johnson distance.

For CuAu `n=64,k=32`, the complete state space has

\[
\binom{64}{32}=1,832,624,140,942,590,534
\]

states, but the exact radial reference kernel has only

\[
D+1=33
\]

orbit states.

The crucial implementation point is that **the controller must also avoid shell enumeration**. Do not produce one output per shell neighbor. The recommended controller writes each shell's controlled rate as

\[
u_{\theta,j}(y,x)=\lambda_{\theta,j}(x,t)q_{\theta,j}(y\mid x,t),
\]

where `q_theta,j` is an exactly normalized, exactly sampleable distribution over the distance-`j` shell. A fixed-cardinality product distribution based on elementary symmetric polynomials gives exact normalization and sampling in `O(N j)` time. This removes the `v_j` explosion from controlled simulation.

The main computational tradeoff in `j` is therefore **not** exponential action enumeration. It is:

1. stronger base mixing as `j` increases toward an intermediate shell;
2. linear/polynomial per-jump work (`O(j)`, `O(Nj)`, or `O(j^3)` depending on the operation);
3. potentially much larger terminal energy contrasts and terminal-label variance for large `j`.

For equiatomic `n=64,k=32`, pure shell `j=16` is especially interesting: its normalized shell walk has an exact reduced-chain spectral gap about `0.99919265`, versus `0.0625` for `j=1`. However, `j=32` is the deterministic complement map and is disconnected as a pure reference. Thus "larger `j`" is not monotone; the useful regime is intermediate, with a small `j=1` floor guaranteeing irreducibility on every time interval.

---

# 1. Repository strategy

Do **not** refactor the existing single-swap Ising implementation first. Keep the published experiments frozen.

Recommended files:

```text
common.py
    add generic radial-Johnson kernel helpers only if desired

structured_asbs/
    fixed_ising.py              # leave published implementation unchanged
    cuau.py                     # existing/new CuAu energy + baseline utilities
    cuau_radial.py              # new radial reference/controller experiment
    tests_radial_johnson.py     # exact brute-force mathematical tests
```

If the existing CuAu work has not yet been committed, it is acceptable to put all radial code in `structured_asbs/cuau_radial.py` initially and factor helpers into `common.py` only after the experiment is correct.

The safest rule is:

> Existing paper-result code is immutable. Radial Johnson IASBS enters as a new module and new JSON results.

---

# 2. CuAu target and state representation

Encode

```text
1 = Au
0 = Cu
```

and enforce the equiatomic sector

\[
\Omega_{N,N/2}.
\]

Primary system:

```text
CuAu-M
supercell: 4 x 4 x 4
N:         64
k:         32
states:    C(64,32) = 1.832624140942590534e18
T:         500 K primary
controls:  680 K, 1200 K
```

Target:

\[
\boxed{
\pi_T(x)\propto\exp[-E_{\rm CE}(x)/(k_BT)],
\qquad x\in\Omega_{N,N/2}.
}
\]

Use

```python
KB_EV_PER_K = 8.617333262145e-5
tau = KB_EV_PER_K * temperature_K
```

The energy wrapper must return **total configurational energy in eV for the whole supercell**. Verify the unit and species mapping against the pinned upstream CuAu implementation before training.

Recommended external benchmark assets remain the official MetaDNS CuAu files:

```text
https://github.com/xiaochendu/metadns

data/cuau/cuau_fcc_4x4x4_supercell.vasp
data/cuau/CI_params_ECI_CuAu_Final_Submission.json
```

Do not run DFT/VASP. Use the released fitted cluster-expansion oracle.

---

# 3. Johnson geometry

Identify a binary state `x` with its occupied/Au subset

\[
S(x)=\{i:x_i=1\},\qquad |S|=k.
\]

The Johnson distance is

\[
\boxed{
d(S,Y)=k-|S\cap Y|.
}
\]

A distance-`j` move removes exactly `j` occupied sites and fills exactly `j` empty sites:

\[
R=S\setminus Y,\qquad
A=Y\setminus S,
\qquad |R|=|A|=j.
\]

The number of states at distance `j` from any fixed state is

\[
\boxed{
v_j=\binom{k}{j}\binom{n-k}{j}.
}
\]

This number can be enormous and must never be explicitly enumerated at large `N`.

For `n=64,k=32`:

| `j` | `v_j = C(32,j)^2` |
|---:|---:|
| 1 | 1,024 |
| 2 | 246,016 |
| 4 | 1,293,121,600 |
| 8 | 110,634,634,890,000 |
| 12 | 50,982,406,595,265,600 |
| 16 | 361,297,635,242,552,100 |
| 20 | 50,982,406,595,265,600 |
| 24 | 110,634,634,890,000 |
| 28 | 1,293,121,600 |
| 31 | 1,024 |
| 32 | 1 |

The shell cardinality peaks at `j=16` in the equiatomic `32/32` case. This does **not** imply that computation must peak there; all scalable algorithms below work with the shell implicitly.

---

# 4. Radial Johnson reference

For every `j=1,...,D`, define

\[
(K_jf)(S)
=\frac{1}{v_j}\sum_{Y:d(S,Y)=j}f(Y).
\]

`K_j` is a Markov operator: each row sums to one.

Choose nonnegative shell clocks `gamma_j(t)` and define

\[
\boxed{
\mathcal L_t
=\sum_{j=1}^{D}\gamma_j(t)(K_j-I).
}
\]

Equivalently,

\[
r_t(Y,S)
=\begin{cases}
\gamma_j(t)/v_j,&d(S,Y)=j,\\
0,&\text{otherwise.}
\end{cases}
\]

The total reference event rate from every state is

\[
\boxed{
\gamma_{\rm tot}(t)=\sum_j\gamma_j(t).
}
\]

It is independent of `v_j`.

Thus a shell with `3.6e17` neighbors can still generate one reference event by:

1. choose the shell `j` with probability proportional to `gamma_j(t)`;
2. choose `j` occupied sites uniformly without replacement;
3. choose `j` empty sites uniformly without replacement;
4. exchange their species.

No shell list is constructed.

---

# 5. Admissibility theorem

## 5.1 Statement

Let `Omega_{n,k}` be the binary fixed-composition state space. Let the operator family be all coordinate-permutation pullbacks

\[
(D^g f)(S)=f(gS),\qquad g\in S_n.
\]

Let the reference generator be any radial Johnson mixture

\[
\mathcal L_t=\sum_j\gamma_j(t)(K_j-I).
\]

Then, on every active edge, the triple

\[
(\Omega_{n,k},\{D^g\}_{g\in S_n},p^{\rm base})
\]

satisfies IASBS C1, C2, and C4. If the required reference kernels are positive on the used endpoint pairs, C3 is exactly evaluable from a `(D+1)`-state orbit process. A sufficient condition guaranteeing positivity on every nonzero interval is

\[
\int_s^t\gamma_1(u)\,du>0
\qquad\text{for all }s<t.
\]

A small positive `j=1` floor is therefore the safest implementation.

---

## 5.2 C1: reconstruction

Take any active shell edge `S -> Y`, with `d(S,Y)=j`.

Define

\[
R=S\setminus Y,\qquad A=Y\setminus S.
\]

Choose a deterministic bijection

\[
R=\{r_1,\ldots,r_j\}
\leftrightarrow
A=\{a_1,\ldots,a_j\}.
\]

Define the disjoint-transposition permutation

\[
\boxed{
g=(r_1\ a_1)\cdots(r_j\ a_j).}
\]

Then `gS=Y`, hence

\[
\frac{(D^gf)(S)}{f(S)}
=\frac{f(gS)}{f(S)}
=\frac{f(Y)}{f(S)}.
\]

Thus C1 holds with reconstruction map

\[
\mathcal R(S,z)=z.
\]

### Important implementation fact

Many permutations map `S` to the same `Y`. Any deterministic choice based only on the edge `(S,Y)` is valid. Different choices give different **samplewise terminal labels** but the same conditional mean, because every such permutation maps `S` to `Y` and commutes with the reference.

This gives a useful variance-reduction degree of freedom for CuAu; see Section 12.

---

## 5.3 C2: intertwining

Johnson distance is invariant under simultaneous coordinate permutation:

\[
d(gS,gY)=d(S,Y).
\]

Therefore, for every shell,

\[
D^gK_j=K_jD^g.
\]

Hence

\[
[D^g,\mathcal L_t]=0
\qquad\text{for every }t,
\]

and therefore the transition semigroup obeys

\[
\boxed{
D^gP_{s,t}=P_{s,t}D^g.
}
\]

This is the closed-intertwiner case:

\[
A_{s,t}^g=D^g.
\]

No target-energy symmetry is required. The CuAu cluster-expansion energy is allowed to break arbitrary coordinate permutation symmetry. Only the **reference** needs the permutation symmetry.

---

## 5.4 C4: coverage

The action of `S_n` on `k`-subsets is transitive. For every `S,Y in Omega_{n,k}`, a coordinate permutation exists with `gS=Y`.

Therefore every active edge of every enabled shell is covered.

For an edge at distance `j`, the disjoint-transposition construction in C1 gives an explicit covering operator.

---

## 5.5 C3: terminal tractability

The pair orbits of the diagonal `S_n` action on

\[
\Omega_{n,k}\times\Omega_{n,k}
\]

are exactly classified by Johnson distance. Since the radial reference is invariant under this action,

\[
\boxed{
p^{\rm base}_{t|s}(Y\mid S)=\kappa_{s,t}(d(S,Y)).}
\]

The function `kappa` is computed from a `(D+1)`-state chain, not from the full state space.

The exact construction is derived in Section 7.

Since the state space is finite, positivity of the relevant denominator implies integrability automatically.

---

# 6. Maximality under full permutation symmetry

This is stronger than merely saying that multi-shell references are allowed.

Suppose a CTMC reference rate satisfies full coordinate-permutation equivariance:

\[
r_t(gY,gS)=r_t(Y,S)
\qquad\forall g\in S_n.
\]

Two ordered pairs `(S,Y)` and `(S',Y')` are in the same diagonal `S_n` orbit if and only if

\[
d(S,Y)=d(S',Y').
\]

Therefore the rate can only depend on the Johnson distance:

\[
r_t(Y,S)=c_j(t),\qquad j=d(S,Y).
\]

Writing

\[
\gamma_j(t)=v_jc_j(t)
\]

gives exactly

\[
\mathcal L_t=\sum_j\gamma_j(t)(K_j-I).
\]

Thus:

\[
\boxed{
\text{radial Johnson mixtures are the full family of }S_n\text{-equivariant reference generators on }\Omega_{n,k}.
}
\]

This statement is only about the reference family compatible with the **full permutation-pullback construction**. It does not claim that no other admissible triple can be built with different operators or a smaller symmetry group.

Mathematically, the shell adjacency matrices are the basis relations of the Johnson association scheme and span its commutative Bose--Mesner algebra.

---

# 7. Exact reduced kernel for arbitrary shell mixtures

## 7.1 Orbit coordinate

Fix a source `S0`. For any state `Y`, define

\[
a=d(S_0,Y)\in\{0,\ldots,D\}.
\]

The orbit size is

\[
\boxed{
v_a=\binom{k}{a}\binom{n-k}{a}.}
\]

Let `B_j` be the transition matrix of one **uniform distance-`j` shell jump**, viewed only through distance from `S0`.

---

## 7.2 Exact shell orbit matrix

Assume the current state `Y` is at distance `a` from `S0`.

Partition coordinates into

```text
S0 ∩ Y                  size k-a
Y \ S0                  size a
S0 \ Y                  size a
outside S0 ∪ Y          size n-k-a
```

A distance-`j` move removes `j` elements from `Y` and inserts `j` elements from its complement.

Let

- `r` = number removed from `S0 ∩ Y`;
- `j-r` = number removed from `Y \ S0`;
- `s` = number inserted from `S0 \ Y`;
- `j-s` = number inserted from the outside block.

After the move, the new Johnson distance from `S0` is

\[
\boxed{b=a+r-s.}
\]

The number of distance-`j` destinations producing this choice is

\[
\binom{k-a}{r}
\binom{a}{j-r}
\binom{a}{s}
\binom{n-k-a}{j-s}.
\]

Therefore

\[
\boxed{
(B_j)_{ab}
=\frac{1}{v_j}
\sum_{\substack{0\le r,s\le j\\b=a+r-s}}
\binom{k-a}{r}
\binom{a}{j-r}
\binom{a}{s}
\binom{n-k-a}{j-s}.
}
\]

Invalid binomial coefficients are interpreted as zero.

Each `B_j` is row stochastic.

---

## 7.3 Integrated orbit generator

Define the integrated shell clocks

\[
\Gamma_j(s,t)=\int_s^t\gamma_j(u)\,du.
\]

Because the Johnson shell operators belong to the same commutative association algebra,

\[
[K_i,K_j]=0.
\]

Therefore

\[
[\mathcal L_s,\mathcal L_t]=0
\]

for arbitrary scalar schedules `gamma_j(t)`, even if different shells use different time profiles.

Hence the time-ordered exponential collapses to

\[
\boxed{
P_{s,t}
=\exp\left(\sum_j\Gamma_j(s,t)(K_j-I)\right).
}
\]

On the orbit chain, define

\[
\boxed{
Q_{s,t}^{\rm orb}
=\sum_j\Gamma_j(s,t)(B_j-I).
}
\]

Starting from distance zero,

\[
\boxed{
q_{s,t}=e_0^\top\exp(Q_{s,t}^{\rm orb}).
}
\]

`q_{s,t}(a)` is the probability that the reference endpoint lies somewhere in orbit `a`.

Since the kernel is constant inside that orbit,

\[
\boxed{
\kappa_{s,t}(a)=\frac{q_{s,t}(a)}{v_a}.
}
\]

For CuAu `N=64,k=32`, this requires one `33 x 33` matrix exponential.

---

## 7.4 Direct implementation

```python
import math
import numpy as np
from scipy.linalg import expm


def comb0(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def shell_size(n: int, k: int, j: int) -> int:
    return math.comb(k, j) * math.comb(n - k, j)


def shell_orbit_matrix(n: int, k: int, j: int) -> np.ndarray:
    D = min(k, n - k)
    if not (1 <= j <= D):
        raise ValueError((n, k, j))

    vj = shell_size(n, k, j)
    B = np.zeros((D + 1, D + 1), dtype=np.float64)

    for a in range(D + 1):
        for r in range(j + 1):
            c_remove = comb0(k - a, r) * comb0(a, j - r)
            if c_remove == 0:
                continue
            for s in range(j + 1):
                b = a + r - s
                if not (0 <= b <= D):
                    continue
                cnt = (
                    c_remove
                    * comb0(a, s)
                    * comb0(n - k - a, j - s)
                )
                B[a, b] += cnt / vj

    if not np.allclose(B.sum(axis=1), 1.0, atol=1e-13, rtol=1e-13):
        raise AssertionError("shell orbit matrix is not stochastic")
    return B


def radial_kappa(n: int, k: int, integrated_clocks: dict[int, float]):
    D = min(k, n - k)
    I = np.eye(D + 1)
    G = np.zeros((D + 1, D + 1), dtype=np.float64)

    for j, Gamma_j in integrated_clocks.items():
        if Gamma_j == 0.0:
            continue
        Bj = shell_orbit_matrix(n, k, int(j))
        G += float(Gamma_j) * (Bj - I)

    q = np.zeros(D + 1, dtype=np.float64)
    q[0] = 1.0
    q = q @ expm(G)

    # Roundoff cleanup only; do not hide a materially negative probability.
    if q.min() < -1e-12:
        raise FloatingPointError(f"negative orbit probability: {q.min()}")
    q = np.maximum(q, 0.0)
    q /= q.sum()

    valency = np.array(
        [shell_size(n, k, a) if a > 0 else 1 for a in range(D + 1)],
        dtype=np.float64,
    )
    kappa = q / valency
    return q, kappa
```

For production, cache every `B_j`; never rebuild it inside training.

---

## 7.5 Numerical stability

Use `log_kappa`, not `kappa`, in terminal ratios.

For moderate CuAu clocks (`Gamma_total` on the order of tens), dense `scipy.linalg.expm` on `33 x 33` or `65 x 65` matrices is already cheap and typically stable.

Required safeguards:

1. assert row sums of every `B_j`;
2. assert `B_i @ B_j == B_j @ B_i` numerically on unit tests;
3. reject materially negative entries after matrix exponential;
4. store `log_kappa = log(q) - log(v)`;
5. if a required `q[a]` underflows to zero, recompute by uniformization or higher precision rather than silently clipping the terminal ratio.

Uniformization is straightforward. Let

\[
\Lambda=\sum_j\Gamma_j,
\qquad
\bar B=\Lambda^{-1}\sum_j\Gamma_jB_j.
\]

Then

\[
\exp\left(\sum_j\Gamma_j(B_j-I)\right)
=e^{-\Lambda}\sum_{m=0}^{\infty}\frac{\Lambda^m}{m!}\bar B^m.
\]

This gives a positivity-preserving evaluator with a Poisson-tail truncation criterion.

---

# 8. Positivity and irreducibility

Do not assume every pure distance shell is connected.

Example: when `n=2k` and `j=k`, the only distance-`k` neighbor of a state is its complement. The pure shell reference decomposes into two-state components.

The simplest sufficient condition is a small distance-1 floor:

\[
\boxed{\gamma_1(t)\ge\epsilon>0\quad\text{for all }t.}
\]

Then every positive-length interval contains the connected Johnson graph component and the transition kernel is strictly positive on all states.

If using a piecewise-constant schedule, require `gamma_1 > 0` in every time segment.

This floor can be small. It is a correctness/robustness device, not necessarily the main mixing mechanism.

---

# 9. Exact reference mixing diagnostics

Because the reference law from a Dirac source is radial, its distance from the uniform fixed-composition law can be evaluated **exactly without state enumeration**.

The uniform law places orbit mass

\[
q_{\rm unif}(a)=\frac{v_a}{|\Omega_{n,k}|}.
\]

Therefore

\[
\boxed{
\operatorname{TV}(p^{\rm base}_{t|0}(\cdot\mid x_0),\operatorname{Unif})
=\frac12\sum_{a=0}^{D}|q_{0,t}(a)-q_{\rm unif}(a)|.
}
\]

Likewise,

\[
\operatorname{KL}(p^{\rm base}\|\operatorname{Unif})
=\sum_a q(a)\log\frac{q(a)}{q_{\rm unif}(a)}.
\]

Use these exact diagnostics before training. They reveal how much broader the radial reference is than the current single-swap reference at the same event clock.

---

# 10. Pure-shell spectral diagnostics at CuAu-M size

For the normalized pure shell process

\[
\mathcal L^{(j)}=K_j-I
\]

with one expected reference shell event per unit time, the CTMC spectral gap can be computed exactly from the `33 x 33` reduced matrix `B_j`.

For `n=64,k=32`, direct diagonalization of the exact reduced matrices gives:

| shell `j` | shell degree `v_j` | pure-shell spectral gap |
|---:|---:|---:|
| 1 | 1,024 | 0.0625000000 |
| 2 | 246,016 | 0.1250000000 |
| 4 | 1,293,121,600 | 0.2500000000 |
| 8 | 110,634,634,890,000 | 0.5000000000 |
| 12 | 50,982,406,595,265,600 | 0.7500000000 |
| 16 | 361,297,635,242,552,100 | **0.9991926513** |
| 20 | 50,982,406,595,265,600 | 0.9526209677 |
| 24 | 110,634,634,890,000 | 0.7620967742 |
| 28 | 1,293,121,600 | 0.4445564516 |
| 31 | 1,024 | 0.1230468750 |
| 32 | 1 | 0 (disconnected) |

These numbers are not an argument that `j=16` must be optimal for the learned sampler. They measure only **reference relaxation**.

A useful exact first-harmonic calculation explains the rise toward `j=16`. For a centered occupancy indicator, the normalized shell operator has eigenvalue

\[
\boxed{
\theta_1(j)=1-\frac{nj}{k(n-k)}.
}
\]

At `n=64,k=32`, this is

\[
\theta_1(j)=1-\frac{j}{16},
\]

so the degree-one slow mode is killed exactly at `j=16`. Higher harmonics leave the small residual eigenvalue responsible for the `0.9991926513` rather than exactly `1.0` gap.

Past the middle shell, complement-like structure becomes increasingly important and mixing quality is not monotone. The extreme `j=32` move is only complementation.

### Reproduction snippet

```python
for j in [1, 2, 4, 8, 12, 16, 20, 24, 28, 31, 32]:
    B = shell_orbit_matrix(64, 32, j)
    eig = np.linalg.eigvals(B).real
    n_one = np.sum(np.abs(eig - 1.0) < 1e-9)
    if n_one > 1:
        gap = 0.0
    else:
        second = np.max(eig[np.abs(eig - 1.0) >= 1e-9])
        gap = 1.0 - second
    print(j, shell_size(64, 32, j), gap)
```

---

# 11. Exact AS terminal labels

For a Dirac source `X_0=x_0`,

\[
f_1(z)\propto
\frac{\exp[-E(z)/\tau]}
{p^{\rm base}_{1|0}(z\mid x_0)}.
\]

For a current shell edge `x -> y`, choose a deterministic permutation `g=g(x,y)` satisfying `gx=y`.

The closed intertwiner gives terminal label

\[
\Lambda_g(X_1)=\frac{f_1(gX_1)}{f_1(X_1)}.
\]

Using the radial kernel,

\[
\boxed{
\log\Lambda_g(X_1)
=-\frac{E(gX_1)-E(X_1)}{\tau}
+\log\kappa_{0,1}(d(x_0,X_1))
-\log\kappa_{0,1}(d(x_0,gX_1)).
}
\]

The exact regression identity is

\[
\boxed{
\mathbb E[\Lambda_g(X_1)\mid X_t=x]
=\frac{\varphi_t(y)}{\varphi_t(x)}.
}
\]

The controlled rate is

\[
\boxed{
u_t^*(y,x)=r_t(y,x)\frac{\varphi_t(y)}{\varphi_t(x)}.}
\]

### Per-label cost

A single sampled edge label requires:

- one application of `g` to `X1`;
- one energy evaluation `E(gX1)` if `E(X1)` is cached;
- one updated Johnson distance to the source;
- two `log_kappa` table reads.

It does **not** require evaluation over the `v_j` neighbors.

---

# 12. Choosing the terminal permutation `g`: a free variance-reduction lever

For a shell edge, the removed and added sets are

\[
R=x\setminus y,\qquad A=y\setminus x.
\]

Any bijection `R <-> A` defines a product of disjoint transpositions that maps `x` to `y`.

Every such `g` has the same exact conditional mean label. Therefore the pairing can be chosen to reduce variance as long as it is fixed from the edge and does not inspect the terminal sample `X1` or the unknown target factor.

Use the following hierarchy.

## 12.1 Baseline: deterministic index pairing

```python
R = np.sort(np.flatnonzero((x == 1) & (y == 0)))
A = np.sort(np.flatnonzero((x == 0) & (y == 1)))
pairs = list(zip(R, A))
```

This is deterministic and easy to test.

## 12.2 Recommended CuAu option: geometry-aware pairing

Use the periodic FCC site coordinates and pair `R` to `A` by minimum total periodic squared distance:

\[
\min_{\pi\in S_j}\sum_{m=1}^j d_{\rm PBC}(r_m,a_{\pi(m)})^2.
\]

Solve with `scipy.optimize.linear_sum_assignment`.

This remains mathematically valid because the pairing depends only on the current edge `(x,y)` and known lattice geometry, while the reference commutes with **every** coordinate permutation.

Why try it: `g` is applied to `X1`, and a permutation built from spatially shorter transpositions may induce a smaller CuAu CE energy perturbation than an arbitrary long-range pairing. That may reduce the heavy-tailed part of `Lambda_g`.

Do not claim this reduction in advance; measure it.

## 12.3 Optional averaging over pairings

Because every edge-valid `g` has the same conditional mean, independent randomization over such `g` also remains unbiased. One may average `M_pair` terminal labels from independently sampled bijections:

\[
\bar\Lambda=\frac1{M_{\rm pair}}\sum_m\Lambda_{g_m}.
\]

This costs `M_pair` energy evaluations and should be used only if label variance is the bottleneck.

Do **not** choose `g` after looking at `X1` to minimize the terminal energy difference; that changes the random label selection rule and does not automatically preserve the required conditional expectation.

---

# 13. Exact ASBS corrector label

For non-Dirac ASBS, the reference-side corrector ratio for the same edge operator is

\[
\boxed{
\log Q_g(X_0,X_1)
=
\log\kappa_{0,1}(d(X_0,gX_1))
-
\log\kappa_{0,1}(d(X_0,X_1)).
}
\]

Thus the non-Dirac reference-side label remains an `O(1)` kernel-table operation after distances are known.

For CuAu-M, start with **Dirac AS** for the radial-reference experiment. Add the non-Dirac corrector only after the Dirac implementation is validated and performant. The purpose of the first experiment is to test whether reference geometry, not source generality, closes the MetaDNS gap.

---

# 14. Exact reference bridge

Training can retain reference-bridge resampling exactly.

Let bridge endpoints be

\[
X_s=S,\qquad X_t=Y,
\qquad q=d(S,Y),
\]

and sample an intermediate state `Z=X_u`.

Since the kernel is radial,

\[
P(Z\mid S,Y)
\propto
\kappa_{s,u}(d(S,Z))
\kappa_{u,t}(d(Z,Y)).
\]

---

## 14.1 Two-distance form

Let

\[
a=d(S,Z),\qquad b=d(Z,Y).
\]

If `p_{ab}^q` is the number of intermediate states with these two distances given endpoint distance `q`, then

\[
\boxed{
P(a,b\mid q)
=\frac{
p_{ab}^q
\kappa_{s,u}(a)
\kappa_{u,t}(b)
}{
\kappa_{s,t}(q)
}.
}
\]

This is a `(D+1)^2` distribution for each `q` and time triplet.

---

## 14.2 Exact four-block intersection count

Partition the coordinates by the two endpoints:

```text
A = S ∩ Y                  size k-q
B = S \ Y                  size q
C = Y \ S                  size q
D = outside S ∪ Y          size n-k-q
```

Let

\[
h=|Z\cap A|.
\]

The required occupancies of `Z` in the four blocks are

\[
\begin{aligned}
|Z\cap A| &= h,\\
|Z\cap B| &= k-a-h,\\
|Z\cap C| &= k-b-h,\\
|Z\cap D| &= a+b-k+h.
\end{aligned}
\]

Hence

\[
\boxed{
 p_{ab}^q
 =\sum_h
 \binom{k-q}{h}
 \binom{q}{k-a-h}
 \binom{q}{k-b-h}
 \binom{n-k-q}{a+b-k+h}.
}
\]

The sum is only over feasible `h`.

---

## 14.3 Existing bridge implementation can largely be reused

The current fixed-composition bridge code already groups an intermediate state by occupancies in exactly these four endpoint blocks. That combinatorics does **not** depend on whether the reference uses only shell 1 or a radial shell mixture.

The only reference-dependent inputs are

```text
log kappa_{s,u}(distance to X_s)
log kappa_{u,t}(distance to X_t)
```

Therefore, in the direct-state CuAu bridge sampler:

- retain the existing four-block class table;
- replace the single-swap `kappa_table(...)` with the radial `kappa` table;
- return the sampled binary tensor directly rather than an enumerated-state LUT index.

This is one of the main reasons the radial extension should be relatively low-risk.

---

# 15. Uniform shell sampling

A reference or training proposal from shell `j` is trivial.

```python
import numpy as np


def sample_uniform_shell_np(x: np.ndarray, j: int, rng: np.random.Generator):
    x = np.asarray(x, dtype=np.int8)
    occ = np.flatnonzero(x == 1)
    emp = np.flatnonzero(x == 0)
    if j > len(occ) or j > len(emp):
        raise ValueError("invalid shell")

    R = rng.choice(occ, size=j, replace=False)
    A = rng.choice(emp, size=j, replace=False)

    y = x.copy()
    y[R] = 0
    y[A] = 1
    return y, R, A
```

The count remains exactly `k`.

For GPU batches, use masked random scores plus `topk`, or maintain explicit occupied/empty index lists. At `N=64`, either is cheap.

---

# 16. The controlled-action problem

A naive multi-shell controller is not computationally acceptable.

The optimal controlled rate is

\[
u^*_t(y,x)
=\frac{\gamma_j(t)}{v_j}
\frac{\varphi_t(y)}{\varphi_t(x)},
\qquad d(x,y)=j.
\]

If one follows the current `SwapController` pattern and explicitly emits a log multiplier for every destination, shell `j=16` at `N=64` would require `3.61e17` outputs.

That is impossible.

The reference kernel being tractable is not enough; the **learned controlled rate family must have tractable total rate and tractable destination sampling**.

Two solutions exist:

1. bounded edge multiplier + exact rejection/thinning;
2. normalized shell controller with exact subset normalization.

The second is recommended.

---

# 17. Recommended controller: normalized shell rates

For each active shell, write

\[
\boxed{
u_{\theta,j}(y,x,t)=\lambda_{\theta,j}(x,t)
q_{\theta,j}(y\mid x,t),}
\]

where

\[
\sum_{y:d(x,y)=j}q_{\theta,j}(y\mid x,t)=1.
\]

Then the shell's total controlled rate is known exactly:

\[
\sum_{y:d(x,y)=j}u_{\theta,j}(y,x,t)
=\lambda_{\theta,j}(x,t).
\]

The base shell rate is

\[
r_j(y,x,t)=\gamma_j(t)/v_j.
\]

Therefore the learned rate multiplier is

\[
\boxed{
m_{\theta,j}(y,x,t)
=\frac{u_{\theta,j}}{r_j}
=\frac{\lambda_{\theta,j}}{\gamma_j(t)}
v_jq_{\theta,j}(y\mid x,t).
}
\]

This completely removes shell enumeration if `q_theta,j` is normalized and sampleable without enumerating the shell.

---

# 18. Fixed-cardinality product distribution over shell destinations

A distance-`j` destination is uniquely represented by

\[
R\subset\{i:x_i=1\},\quad |R|=j,
\]

and

\[
A\subset\{i:x_i=0\},\quad |A|=j.
\]

A simple tractable controller distribution is

\[
q_j(y\mid x)
=q^-_j(R\mid x)q^+_j(A\mid x),
\]

with

\[
\boxed{
q^-_j(R\mid x)
=\frac{\exp(\sum_{i\in R}s^-_{j,i}(x,t))}
{e_j(\{e^{s^-_{j,i}}:x_i=1\})},
}
\]

and

\[
\boxed{
q^+_j(A\mid x)
=\frac{\exp(\sum_{i\in A}s^+_{j,i}(x,t))}
{e_j(\{e^{s^+_{j,i}}:x_i=0\})}.
}
\]

Here `e_j` is the elementary symmetric polynomial of degree `j`.

This distribution has four crucial properties:

1. it has support on **every** distance-`j` destination if logits are finite;
2. its log probability is exactly computable;
3. its normalizer is computable by dynamic programming in `O(Nj)`;
4. it can be sampled exactly in `O(Nj)` by backward sampling from the same DP.

The shell may have `10^17` states; none are listed.

---

# 19. Elementary-symmetric-polynomial DP

For positive weights `w_1,...,w_m`, define

\[
e_r(w_1,\ldots,w_m)
=\sum_{|S|=r}\prod_{i\in S}w_i.
\]

Use log weights `ell_i=log w_i` and a log-domain prefix DP.

```python
import torch


def log_esp(logw: torch.Tensor, j: int) -> torch.Tensor:
    """log e_j(exp(logw)); 1D reference implementation."""
    if j < 0 or j > logw.numel():
        return torch.tensor(float("-inf"), device=logw.device, dtype=logw.dtype)

    neginf = torch.tensor(float("-inf"), device=logw.device, dtype=logw.dtype)
    dp = torch.full((j + 1,), neginf, device=logw.device, dtype=logw.dtype)
    dp[0] = 0.0

    for z in logw:
        nxt = dp.clone()
        nxt[1:] = torch.logaddexp(dp[1:], dp[:-1] + z)
        dp = nxt

    return dp[j]
```

For a selected set `R`,

```python
log_q_R = remove_logits[R].sum() - log_esp(remove_logits[occ], j)
```

and analogously for `A`.

For production, vectorize over batches/shells. Start from the scalar implementation and test it exhaustively against enumerated subsets before vectorizing.

---

# 20. Exact sampling from the fixed-cardinality product distribution

Store the full prefix table

\[
E[i,r]=e_r(w_1,\ldots,w_i).
\]

While moving backward from `(i=m,r=j)`, include item `i` with probability

\[
\boxed{
P(i\in S\mid r,i)
=\frac{w_iE[i-1,r-1]}{E[i,r]}.
}
\]

If included, set `r <- r-1`; otherwise retain `r`. Continue until `r=0`.

This samples exactly from

\[
P(S)\propto\prod_{i\in S}w_i,
\qquad |S|=j.
\]

Use the log-domain form for stability.

Required unit test: for `m<=8`, enumerate every size-`j` subset, compare exact probabilities with empirical frequencies from at least `1e6` DP samples, and require agreement within Monte Carlo confidence bounds.

---

# 21. Controller parameterization

For active shell set

```text
J = {1, 2, 4, 8, 16}
```

at `N=64`, a compact controller can use one shared state encoder and shell-specific heads.

Recommended first architecture:

```text
input:
    x in {0,1}^64, encoded as 2x-1
    t, 1-t
    Fourier time features

shared trunk:
    3 x 512 SiLU MLP

for each shell j:
    one scalar b_j                 # total-rate log multiplier
    N removal logits s^-_{j,i}
    N addition logits s^+_{j,i}
```

Total output count for five shells is

```text
5 * (1 + 64 + 64) = 645
```

which is much smaller than the current `64^2=4096` pairwise output head.

Set

\[
\boxed{\lambda_{\theta,j}(x,t)=\gamma_j(t)e^{b_{\theta,j}(x,t)}.}
\]

This automatically enforces support matching: if `gamma_j(t)=0`, the controlled shell rate is zero.

Initialize all heads to zero. Then

- `b_j=0`;
- all subset logits are equal;
- `q_j=1/v_j`;
- `lambda_j=gamma_j`;
- therefore `u_theta=r` exactly at initialization.

That is a desirable initialization.

---

# 22. Exact log multiplier under the normalized controller

For a sampled shell edge represented by removed set `R` and added set `A`,

\[
\log q_j(y\mid x)
=\log q^-_j(R\mid x)+\log q^+_j(A\mid x).
\]

Since `lambda_j/gamma_j=e^{b_j}`,

\[
\boxed{
\log m_{\theta,j}(y,x,t)
=b_{\theta,j}(x,t)
+\log v_j
+\log q_j(y\mid x,t).
}
\]

When all logits and `b_j` are zero,

\[
\log q_j=-\log v_j,
\qquad \log m=0.
\]

Compute `log v_j` directly via `lgamma` or exact Python `comb` followed by `log`.

---

# 23. Poisson-Bregman loss without enumerating a shell

The current IASBS discrete controller uses the exponential-family Bregman objective whose edgewise term is

\[
r(y,x)\left[m_\theta(y,x)-\Lambda(y,x)\log m_\theta(y,x)\right].
\]

For the radial reference, sum over all shell destinations:

\[
\mathcal L(x)
=\sum_j\sum_{y:d(x,y)=j}
\frac{\gamma_j}{v_j}
\left[m_{\theta,j}(y,x)-\Lambda_j(y,x)\log m_{\theta,j}(y,x)\right].
\]

The first term collapses exactly:

\[
\sum_{y:d(x,y)=j}\frac{\gamma_j}{v_j}m_{\theta,j}(y,x)
=\sum_yu_{\theta,j}(y,x)
=\lambda_{\theta,j}(x).
\]

Therefore

\[
\boxed{
\mathcal L(x)
=\sum_j\lambda_{\theta,j}(x)
-\sum_j\gamma_j
\mathbb E_{Y\sim\operatorname{Unif}(\mathcal N_j(x))}
[\Lambda_j(Y,x)\log m_{\theta,j}(Y,x)].
}
\]

Let

\[
\gamma_{\rm tot}=\sum_j\gamma_j,
\qquad
J\sim\gamma_j/\gamma_{\rm tot},
\qquad
Y\sim\operatorname{Unif}(\mathcal N_J(x)).
\]

Then the one-edge unbiased estimator is

\[
\boxed{
\widehat{\mathcal L}(x)
=\sum_j\lambda_{\theta,j}(x)
-\gamma_{\rm tot}\Lambda_J(Y,x)\log m_{\theta,J}(Y,x).
}
\]

Average multiple independently sampled reference edges if label variance is large.

This is the preferred training formulation because:

- the positive-rate term is **exactly normalized**;
- only the target-label term uses edge Monte Carlo;
- neither term scales with `v_j`;
- the minimizer remains the IASBS rate multiplier within the represented controller family.

---

# 24. More expressive shell distributions if one product component is insufficient

The single product distribution assumes conditional independence between the removed set and added set given the network logits. This is an **approximation in controller expressivity**, not in the IASBS label or reference.

Escalate only if necessary.

## Option A: mixture of product subset distributions

Use `C` components:

\[
q_j(y\mid x)=\sum_{c=1}^C\alpha_{j,c}(x,t)
q^-_{j,c}(R\mid x,t)q^+_{j,c}(A\mid x,t).
\]

Sampling:

1. sample `c ~ alpha`;
2. sample `R` by ESP DP;
3. sample `A` by ESP DP.

Likelihood:

\[
\log q_j(y\mid x)
=\operatorname{logsumexp}_c
[\log\alpha_{j,c}+\log q^-_{j,c}(R)+\log q^+_{j,c}(A)].
\]

Cost is `O(C N j)`.

Start with `C=1`. If necessary try `C=4` as a predeclared capacity ablation.

## Option B: condition addition logits on the removed set

After sampling `R`, run a second lightweight head receiving `(x, 1_R, t)` and generate the addition logits. Then

\[
q_j(y\mid x)=q^-_j(R\mid x)q^+_j(A\mid x,R).
\]

This remains exactly normalized and exactly sampleable, but costs a second network pass per accepted event.

Use only if the independent product controller fails to model CuAu rearrangements.

---

# 25. Exact simulation of the represented controlled CTMC

The normalized controller eliminates the astronomical action-space normalization.

For a fixed state `x` and time `t`, compute

\[
\lambda_j(x,t)=\gamma_j(t)e^{b_j(x,t)}.
\]

Total controlled escape rate:

\[
\boxed{R_\theta(x,t)=\sum_j\lambda_j(x,t).}
\]

Conditional shell probability given a jump:

\[
P(J=j\mid\text{jump},x,t)
=\lambda_j/R_\theta.
\]

Conditional destination:

\[
Y\sim q_{\theta,j}(\cdot\mid x,t).
\]

Thus controlled simulation does not require evaluating any other shell destinations.

---

## 25.1 Recommended time discretization

The current discrete implementation uses a time grid. For the radial controller, define network controls to be piecewise constant in time over each grid bin:

\[
t\in[t_s,t_{s+1})
\quad\Rightarrow\quad
u_\theta(\cdot,t)=u_\theta(\cdot,t_s).
\]

Within each bin, simulate the state-dependent CTMC exactly:

```text
remaining = dt
while remaining > 0:
    evaluate lambda_j(x, t_s)
    R = sum_j lambda_j
    draw h ~ Exp(R)
    if h >= remaining:
        finish bin
    else:
        choose shell j proportional to lambda_j
        sample y ~ q_theta,j(. | x,t_s) exactly by ESP DP
        x = y
        remaining -= h
        recompute rates at new x
```

This allows multiple jumps in one time bin and is exact for the piecewise-time-constant neural control.

As the number of bins increases, this approximates the intended continuously time-varying controller.

This is preferable to the current "at most one jump per bin" rule once shell rates become aggressive.

---

# 26. Fallback exact simulator: bounded multiplier thinning

If the normalized subset controller is delayed, a bounded edge-conditioned controller can still simulate without shell enumeration.

Parameterize

\[
a_\theta(t,x,y) = B_j\tanh h_\theta(t,x,y),
\qquad m_\theta=e^{a_\theta}.
\]

Then

\[
m_\theta\le e^{B_j}.
\]

For shell `j`, generate candidate events at rate

\[
\gamma_j e^{B_j},
\]

propose `Y` uniformly from the shell, and accept with probability

\[
\frac{m_\theta(Y,x)}{e^{B_j}}.
\]

The accepted rate for destination `Y` is exactly

\[
\gamma_je^{B_j}\frac1{v_j}\frac{m_\theta}{e^{B_j}}
=\frac{\gamma_j}{v_j}m_\theta.
\]

Correctness is exact. Efficiency depends on rejection acceptance.

Do not use this as the main implementation if large `B_j` is required: the worst-case dominating rate grows as `e^{B_j}`. The normalized controller avoids this problem completely.

---

# 27. How computability depends on shell size `j`

This is the central practical answer.

## 27.1 What grows combinatorially

The number of possible destinations in a shell is

\[
v_j=\binom{k}{j}\binom{n-k}{j}.
\]

At equiatomic `N=64`, it grows from `1024` at `j=1` to `3.61e17` at `j=16` and then decreases symmetrically.

Any algorithm that does

```text
for y in shell_j(x): ...
```

is dead beyond tiny `j`.

This includes:

- explicit controller output per destination;
- explicit normalization of learned shell rates;
- explicit all-edge terminal-label computation;
- explicit shell adjacency storage.

None of these are required.

---

## 27.2 Operations that depend only mildly on `j`

| operation | scalable complexity | `j` dependence | comment |
|---|---:|---:|---|
| sample uniform reference shell neighbor | `O(N)` simple implementation; `O(j)` with maintained lists | linear at worst | choose `j` occupied + `j` empty |
| apply destination state update | `O(j)` sparse / `O(N)` dense copy | linear | exactly `2j` sites change |
| build one reduced shell matrix `B_j` | `O(D j^2)` naive | quadratic | one-time; `D=32` for CuAu-M |
| store `B_j` | `O(D^2)` dense | essentially independent | tiny at `D=32` |
| radial kernel matrix exponential | `O(D^3)` dense | independent of `j` | depends on `D`, not `v_j` |
| one `kappa` query | `O(1)` after distance | none | table lookup |
| update source distance after applying `g` | `O(j)` if cached | linear | otherwise `O(N)` dot product |
| terminal label target-energy calls | 1 new CE call if `E(X1)` cached | **count independent of `j`** | wall cost may vary with CE backend |
| deterministic index pairing for `g` | `O(j log j)` | mild | sorting |
| geometric Hungarian pairing | `O(j^3)` | cubic | `j<=16` still tiny |
| fixed-cardinality `q` normalizer | `O(Nj)` | linear | ESP DP |
| sample controlled shell destination | `O(Nj)` | linear | same DP |
| exact bridge kernel | `(D+1)` orbit matrices | independent of `v_j` | schedule affects only clocks |
| exact four-block bridge sample | `O(N)` after class table | no direct `j` | depends on endpoint distance, not shell enumeration |

The headline conclusion is:

\[
\boxed{
\text{computational complexity can be polynomial in }j\text{ even though }v_j\text{ is exponential/combinatorial.}
}
\]

---

## 27.3 Event count versus changed-site count

For the reference, shell `j` has total event rate `gamma_j`, independent of shell degree.

Expected number of shell-`j` events over `[0,1]` is

\[
\Gamma_j=\int_0^1\gamma_j(t)dt.
\]

But every event changes `2j` coordinates. Therefore a useful physical/computational activity proxy is

\[
\boxed{
C_{\rm changed-sites}
=\sum_j 2j\Gamma_j.
}
\]

When comparing `j=1` and `j=16` references at equal event clock, the latter is performing much more state rearrangement per event.

Report both:

```text
reference/controlled event count
sum of 2j over accepted events
CE energy evaluations
wall time
```

This prevents an unfair claim that a block move is "the same work" as a single swap merely because each counts as one CTMC event.

---

## 27.4 CE energy cost versus `j`

There are two cases.

### Full CE reevaluation

If the backend rebuilds/evaluates the entire 64-site configuration for every energy query, the **number of energy oracle calls per sampled terminal label is independent of `j`**. The wall cost of the call will be nearly independent of `j` except for state-construction overhead.

### Incremental CE delta evaluation

If the backend updates local cluster contributions affected by changed sites, the work can grow roughly with the number of changed sites/clusters, hence approximately with `j` until affected neighborhoods overlap strongly.

Do not assume either behavior. Benchmark the pinned CuAu energy backend with random shell moves at

```text
j in {1,2,4,8,16}
```

and record microseconds per `Delta E` / full `E` call.

---

## 27.5 The real large-`j` bottleneck: statistical label difficulty

The AS terminal label contains

\[
-\Delta E_g/\tau,
\qquad
\Delta E_g=E(gX_1)-E(X_1).
\]

A `j`-pair permutation can perturb many more local CE terms than a one-pair permutation. Therefore the distribution of `log Lambda` will generally widen as `j` grows.

This is **not a theorem of monotone growth** for an arbitrary energy model. It is an empirical/statistical risk. For a local alloy Hamiltonian, one expects the typical number of affected clusters to increase with the number of moved coordinates, so larger `j` can generate heavier-tailed exponentiated labels.

This can dominate computation even when the combinatorics are fully tractable.

Before training, measure for each candidate shell:

```text
mean(log Lambda)
std(log Lambda)
p01 / p50 / p99(log Lambda)
max abs(log Lambda)
mean Lambda if numerically stable
fraction outside current numerical clamp
energy-ratio contribution alone
kernel-ratio contribution alone
```

Evaluate both index pairing and geometry-aware pairing.

The chosen maximum shell should be determined by this diagnostic as well as by reference mixing.

---

## 27.6 Why `j=16` is promising but not guaranteed optimal

At `N=64,k=32`:

- `j=16` maximizes shell cardinality;
- the degree-one Johnson harmonic is killed exactly;
- the pure-shell reduced spectral gap is approximately `0.99919`;
- a uniform shell neighbor can still be sampled in `O(j)`/`O(N)`;
- the kernel is still only `33 x 33`.

But:

- a `j=16` move changes 32 species entries;
- its terminal operator can induce a much larger CE energy difference;
- a learned `q_theta` costs `O(Nj)` rather than `O(N)`;
- pure `j=16` does not by itself guarantee good target-aware dynamics.

Therefore test `j=16`; do not assume it wins.

---

# 28. Recommended radial schedules

Use a **predeclared small schedule family**. Do not do unconstrained hyperparameter search after inspecting final metrics.

Active shell set for CuAu-M:

```text
J = {1, 2, 4, 8, 16}
```

Always keep a distance-1 floor.

## Schedule S0: published baseline

```text
100% j=1
```

This reproduces the current Johnson reference.

## Schedule S1: fixed radial mixture

Start with a conservative fixed mixture such as

```text
j=1    0.10
j=2    0.15
j=4    0.20
j=8    0.25
j=16   0.30
```

with constant total event clock `gamma_total`.

This is a simple test of whether multiscale connectivity alone improves performance.

## Schedule S2: coarse-to-fine

Use five equal time segments and a 5% `j=1` irreducibility floor.

```text
t in [0.0,0.2):   95% j=16 + 5% j=1
t in [0.2,0.4):   95% j=8  + 5% j=1
t in [0.4,0.6):   95% j=4  + 5% j=1
t in [0.6,0.8):   95% j=2  + 5% j=1
t in [0.8,1.0):  100% j=1
```

For each segment,

\[
\gamma_j(t)=\gamma_{\rm total}w_j^{(segment)}.
\]

This schedule is exact and particularly easy to integrate because the shell clocks are piecewise constant.

Rationale:

- early large moves broaden the reference quickly;
- late local moves avoid forcing very large shell transitions when the learned control becomes sharp;
- the full-horizon Dirac kernel remains broad because of the early block events;
- every interval retains the connected distance-1 component.

Treat this as a **starting schedule**, not a theoretical optimum.

## Schedule S3: capped maximum shell

If `j=16` labels are pathological, use the same coarse-to-fine schedule with

```text
8 -> 4 -> 2 -> 1
```

or

```text
4 -> 2 -> 1
```

rather than clipping away most `j=16` labels.

---

# 29. Choosing `gamma_total`

Do not conflate shell size with event clock.

A radial reference has two independent controls:

1. which shell an event uses;
2. how many events occur in expectation.

Start by holding the current integrated total event clock fixed:

\[
\boxed{
\int_0^1\gamma_{\rm total}(t)dt=10
}
\]

if the current CuAu/single-swap implementation uses `gamma=10`.

Then compare S0/S1/S2 at the same expected **reference event count**.

Also report changed-site activity. If the radial method wins only by changing 10--20x more coordinates, that is still a valid algorithmic result, but the compute mechanism should be explicit.

After the first comparison, allow a small predeclared total-clock pilot, e.g.

```text
gamma_total in {5, 10, 20}
```

and freeze the selected setting before final multi-seed results.

---

# 30. Reference-only pretraining diagnostics

Before any neural training, compute for S0/S1/S2:

1. exact endpoint orbit law `q_{0,1}`;
2. exact TV to the uniform canonical law;
3. exact KL to uniform;
4. reduced-generator spectral gap for time-homogeneous mixtures;
5. source-distance distribution under PT target samples;
6. distribution of full-horizon kernel ratio contribution
   `log kappa(d0) - log kappa(dg)`;
7. terminal energy-ratio contribution for sampled edges;
8. full `log Lambda` distribution.

This separates three effects:

```text
reference mixing
reference-kernel correction
CuAu energy-label variance
```

Do not start a 500 K training run until these are understood.

---

# 31. CuAu terminal-label diagnostic protocol by `j`

Use a fixed pool of converged canonical PT samples at 500 K.

For each

```text
j in {1,2,4,8,16}
```

repeat at least `50,000` times:

1. sample `X1` from the same fixed PT pool;
2. sample an intermediate/current `x` from the intended reference bridge or, for a cheap initial proxy, a reference state at representative `t`;
3. sample `y` uniformly from shell `j` around `x`;
4. construct `g(x,y)`;
5. evaluate `E(X1)` from cache and `E(gX1)`;
6. compute kernel-ratio term;
7. compute `log Lambda`.

Do this for representative times

```text
t = 0.1, 0.3, 0.5, 0.7, 0.9
```

because the conditional endpoint distribution changes with time.

Record distributions, not only means.

### Stop condition for a shell

If essentially all useful mass for a shell requires the current hard `[-20,20]` log-label clamp, do not call the corresponding training exact in practice. Prefer dropping/capping that shell or changing the pairing/controller before relying on aggressive clipping.

---

# 32. CuAu source states

Retain the previous playbook's source policy.

## Primary source

A deterministic pseudorandom equiatomic state, e.g. seed `1729`.

## Stress source

A perfect L1_0 state in one orientation.

The radial reference should be tested from both. The ordered source is particularly informative because a successful radial controller should be able to reach all symmetry-related orientation sectors rapidly while preserving exact composition throughout.

---

# 33. Experiment sequence

## Phase 0: pure mathematics on tiny spaces

Before CuAu, run exact tests at

```text
(n,k) = (6,3), (8,4), (10,5)
```

with mixed shell clocks.

Required tests are in Section 40.

No GPU is needed.

## Phase 1: controller mechanics on a cheap synthetic fixed-composition energy

Use `n=8,k=4` or `n=10,k=5` and an arbitrary nonsymmetric energy.

Goals:

- verify normalized shell controller;
- verify ESP subset likelihoods;
- verify exact event simulation;
- verify Bregman training with one sampled shell edge;
- compare learned endpoint law against exact propagation/enumeration.

Do not debug these mechanics on the CE backend.

## Phase 2: CuAu energy parity

Perform the 100-state upstream parity gate from the original CuAu playbook.

Require numerical agreement to the tolerance dictated by the pinned upstream backend.

## Phase 3: CuAu-M reference/label diagnostics

No controller training yet.

Run S0/S1/S2 reference tables and the `j`-dependent label study.

Select:

```text
maximum shell
pairing rule
gamma_total
schedule family
```

using predeclared criteria.

## Phase 4: one-seed CuAu-M radial IASBS

Start at 1200 K if necessary for pipeline validation, then 680 K, then 500 K.

Compare:

```text
single-swap IASBS
radial fixed mixture
radial coarse-to-fine
```

using identical CE-call accounting.

## Phase 5: final 500 K multi-seed run

Only after one radial variant clearly improves the difficult target.

Use at least three independent training seeds; five is preferable for a new headline result.

## Phase 6: MetaDNS comparison

Compare against the same raw/importance/conditioned MetaDNS outputs already being evaluated, but clearly distinguish target/ensemble differences where applicable.

The central question is now:

> Does a hard-constrained radial IASBS reference close the performance gap while keeping every state in the exact equiatomic sector?

## Phase 7: optional N=128

Only after N=64 works.

For `N=128,k=64`, `D=64`; exact reference kernels are still only `65 x 65`. Controller ESP cost grows roughly by a factor of four at fixed relative shell `j/N` because both `N` and `j` grow.

---

# 34. Suggested main CuAu-M method table

Final methods should include at least:

```text
Canonical PT reference             truth/reference, not competitor
Canonical single-swap MCMC         local physical baseline
IASBS j=1                          current fixed-composition reference
IASBS radial fixed mixture         multiscale reference
IASBS radial coarse-to-fine        preferred radial candidate
MetaDNS                             external neural sampler context/comparison
```

DAM at N=64 is optional if its nested rollout cost is prohibitive; retain the small exact DAM comparison already used elsewhere.

---

# 35. Evaluation metrics

Keep the original CuAu metrics.

Primary:

```text
energy distribution distance
Q_max distribution distance
L1_0 x/y/z ordered-sector masses
mode imbalance
ordered-basin mass
CE oracle calls
wall time
hard composition violations
```

Secondary:

```text
mean energy per atom
heat capacity / N
Warren-Cowley SRO / pair correlations
reference/controlled event counts
changed-site activity sum(2j)
```

For the cubic `4x4x4` cell, use the symmetry-equivalent L1_0 orientation sectors as planned.

A radial method that merely lowers energy while collapsing into one orientation is not a success.

---

# 36. Additional radial-specific metrics

Save the following per generated trajectory/run:

```text
number of accepted controlled jumps by shell j
sum_jumps 2*j
mean and quantiles of learned b_j
fraction of total controlled rate assigned to each shell over time
entropy of q_theta,j destination distribution
mean log multiplier by shell
terminal-label quantiles by shell
ESP log-normalizer ranges
controller event rate R_theta over time
```

These diagnostics are needed to explain *why* radial IASBS works or fails.

Especially plot

\[
\frac{\lambda_j(X_t,t)}{\sum_\ell\lambda_\ell(X_t,t)}
\]

versus time. It will show whether the learned process actually uses the aggressive shells or suppresses them.

---

# 37. Cost accounting

Count separately:

```text
CE energy evaluations
controller forward passes
ESP DP calls
reference bridge samples
controlled events
controlled changed sites = sum 2j
wall time
peak GPU memory
```

For terminal labels, cache `E(X1)` in the replay buffer. Each sampled operator then needs only `E(gX1)`.

If averaging over multiple edge operators/pairings, count each additional energy evaluation explicitly.

Do not use `v_j` as a compute proxy; it is an implicit action-space size, not actual executed work under the scalable implementation.

---

# 38. Fairness across shell schedules

There are at least three meaningful normalization regimes. Report which one is used.

## Equal reference event clock

Hold

\[
\sum_j\Gamma_j
\]

fixed. This asks how much more mixing can be obtained per CTMC event.

## Equal changed-site activity

Hold

\[
\sum_j2j\Gamma_j
\]

approximately fixed. This penalizes large block moves for modifying many coordinates.

## Equal wall/CE budget

The most application-relevant comparison. Train/generate until the same CE-call or wall-time budget.

For the paper, use **equal CE-call budget and wall time as primary**, while reporting event/changed-site counts mechanistically.

---

# 39. Recommended initial hyperparameters

These are starting values, not final claims.

```text
N                 64
k                 32
T                 500 K primary
shells            {1,2,4,8,16}
gamma_total       10 first
schedule          S2 coarse-to-fine first after diagnostics
j=1 floor         5% of total reference clock in early segments
controller        normalized fixed-cardinality product controller
mixture comps     C=1 initially
hidden            512
trunk              3 x 512 SiLU
time features     same Fourier convention as current SwapController
loss              Poisson-Bregman
bridge             exact radial Johnson bridge
edge labels/step  1 first; 4 if variance requires
source             deterministic random equiatomic primary
training seeds     0,1,2 minimum final
```

Do not precommit to `j=16` if the label diagnostic shows severe heavy tails.

---

# 40. Mandatory mathematical tests

These tests must pass before CuAu training.

## Test A: shell cardinality

For small enumerated `Omega_{n,k}`, verify

\[
|\{y:d(x,y)=j\}|=v_j
\]

for every state and shell.

## Test B: shell orbit matrix against brute force

Enumerate full shell transition matrix `K_j` for `n<=8`.

For each source orbit distance `a`, aggregate destination probabilities by distance `b` and compare with `B_j`.

Require max absolute error `<1e-13` in float64.

## Test C: commutation

For random permutation matrix/pullback `D_g` and each shell:

```python
np.max(np.abs(Dg @ Kj - Kj @ Dg))
```

must be zero up to numerical precision.

Also test reduced matrices:

```python
np.max(np.abs(Bi @ Bj - Bj @ Bi)) < 1e-12
```

## Test D: mixed radial kernel against full generator

For `n=6,k=3`, use e.g.

```text
Gamma_1 = 0.7
Gamma_2 = 1.1
Gamma_3 = 0.4
```

Build the full generator

\[
Q=\sum_j\Gamma_j(K_j-I)
\]

and compare a source row of `expm(Q)` with

\[
\kappa(d(x_0,y))
\]

from the reduced `4 x 4` orbit calculation.

Require max error around machine precision (`~1e-13` or tighter).

## Test E: Chapman--Kolmogorov

For two time segments with different shell mixtures, require

\[
P_{0,t}P_{t,1}=P_{0,1}
\]

both in the full tiny-state generator and in the reduced radial kernels.

## Test F: exact bridge distribution

For a tiny system, enumerate

\[
P(Z\mid S,Y)
=\frac{P_{s,u}(S,Z)P_{u,t}(Z,Y)}{P_{s,t}(S,Y)}
\]

and compare with the four-block/reduced bridge distribution.

Then sample at least `1e6` bridge states and compare empirical probabilities within Monte Carlo error.

## Test G: full IASBS terminal identity

This is the most important test.

On a tiny state space:

1. draw an arbitrary strictly positive terminal function `f1`;
2. build exact full reference `P_{t,1}`;
3. compute `phi=P @ f1`;
4. choose random active multi-shell edge `x->y`;
5. construct `g` with `gx=y`;
6. compute

\[
LHS=\phi(y)/\phi(x);
\]

7. compute exact Doob conditional

\[
p^*(z\mid x)=\frac{P(x,z)f_1(z)}{\phi(x)};
\]

8. compute

\[
RHS=\sum_zp^*(z\mid x)\frac{f_1(gz)}{f_1(z)}.
\]

Require

```text
abs(LHS - RHS) < 1e-12
```

across many random edges, shells, times, and positive `f1` vectors.

This single test catches mistakes in the operator direction, permutation action, kernel convention, and label ratio.

## Test H: pairing invariance of conditional mean

For one `j>=2` edge, construct several different bijections `R<->A`, hence several `g` mapping `x` to `y`.

Verify that their samplewise labels differ but their exact conditional expectations all equal `phi(y)/phi(x)`.

## Test I: ESP normalizer

For candidate count `m<=8`, enumerate every size-`j` subset and compare

\[
\log e_j(e^{s_i})
\]

with brute-force `logsumexp` over subset scores.

Require `<1e-12` float64 error.

## Test J: ESP sampler

Empirically validate subset probabilities against enumeration.

## Test K: controller normalization

For tiny shells, explicitly sum

\[
u_{\theta,j}(y,x)
\]

over all destinations and require it equals `lambda_theta,j`.

## Test L: sampled Bregman loss unbiasedness

On a tiny enumerated shell, compute the exact all-edge Poisson-Bregman loss and compare with the Monte Carlo estimator averaged over many sampled reference edges.

## Test M: exact controlled simulation

On `n<=8`, freeze a random controller in one time bin, build its full generator explicitly, compute `expm(dt Q_theta)`, and compare with the event-driven normalized-controller simulator.

This validates total-rate and subset-destination sampling together.

## Test N: composition

Every reference state, bridge state, terminal permutation state, and controlled state must satisfy

```python
x.sum() == k
```

exactly. Any violation is a hard failure.

---

# 41. Brute-force full generator helper for tests

For small systems only:

```python
import itertools
import numpy as np


def enumerate_fixed(n, k):
    out = []
    for occ in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=np.int8)
        x[list(occ)] = 1
        out.append(x)
    return np.stack(out)


def johnson_distance(x, y):
    k = int(x.sum())
    return k - int(np.dot(x, y))


def full_shell_matrix(states, j):
    M = len(states)
    K = np.zeros((M, M), dtype=np.float64)
    index = {tuple(x.tolist()): i for i, x in enumerate(states)}
    n = states.shape[1]
    k = int(states[0].sum())
    vj = shell_size(n, k, j)

    for a, x in enumerate(states):
        occ = np.flatnonzero(x == 1)
        emp = np.flatnonzero(x == 0)
        for R in itertools.combinations(occ, j):
            for A in itertools.combinations(emp, j):
                y = x.copy()
                y[list(R)] = 0
                y[list(A)] = 1
                K[a, index[tuple(y.tolist())]] += 1.0 / vj
    return K
```

Never call this for CuAu-M.

---

# 42. Implementation class layout

Recommended interface:

```python
class PiecewiseShellSchedule:
    shells: tuple[int, ...]
    breaks: np.ndarray
    weights: np.ndarray
    gamma_total: float

    def rates(self, t: float) -> dict[int, float]: ...
    def integrated_clocks(self, s: float, t: float) -> dict[int, float]: ...


class RadialJohnsonReference:
    def __init__(self, n, k, schedule): ...

    def shell_matrix(self, j): ...
    def log_kappa(self, s, t): ...
    def precompute_grid(self, times): ...
    def sample_uniform_edge(self, x, t, rng): ...
    def sample_bridge(self, x0, x1, t, rng): ...


class RadialCuAuController(torch.nn.Module):
    def forward(self, t, x):
        # b:         (B, n_shells)
        # remove:    (B, n_shells, N)
        # add:       (B, n_shells, N)
        return b, remove_logits, add_logits

    def log_shell_prob(self, t, x, y, j): ...
    def shell_rates(self, t, x, gamma): ...
    def sample_destination(self, t, x, j, rng): ...


def radial_terminal_log_label(ref, energy, x0, x1, x, y, pairing): ...

def radial_controller_loss(...): ...
def simulate_radial_controlled(...): ...
```

Keep energy-oracle code independent of reference/controller code.

---

# 43. Piecewise schedule implementation

```python
class PiecewiseShellSchedule:
    def __init__(self, shells, breaks, weights, gamma_total):
        self.shells = tuple(map(int, shells))
        self.breaks = np.asarray(breaks, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.gamma_total = float(gamma_total)

        if len(self.breaks) != len(self.weights) + 1:
            raise ValueError("break/weight mismatch")
        if not np.allclose(self.weights.sum(1), 1.0):
            raise ValueError("weights must sum to one")
        if np.any(self.weights < 0):
            raise ValueError("negative shell weight")

    def _segment(self, t):
        i = np.searchsorted(self.breaks, t, side="right") - 1
        return int(np.clip(i, 0, len(self.weights) - 1))

    def rates(self, t):
        w = self.weights[self._segment(t)]
        return {j: self.gamma_total * w[a] for a, j in enumerate(self.shells)}

    def integrated_clocks(self, s, t):
        if not (0.0 <= s <= t <= 1.0):
            raise ValueError((s, t))
        out = {j: 0.0 for j in self.shells}
        for seg in range(len(self.weights)):
            lo = max(s, self.breaks[seg])
            hi = min(t, self.breaks[seg + 1])
            if hi <= lo:
                continue
            dt = hi - lo
            for a, j in enumerate(self.shells):
                out[j] += dt * self.gamma_total * self.weights[seg, a]
        return out
```

For S2 with shell order `[1,2,4,8,16]`:

```python
weights = np.array([
    [0.05, 0.00, 0.00, 0.00, 0.95],
    [0.05, 0.00, 0.00, 0.95, 0.00],
    [0.05, 0.00, 0.95, 0.00, 0.00],
    [0.05, 0.95, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00, 0.00],
], dtype=np.float64)
```

---

# 44. Radial kernel cache on the training grid

For training times

```python
times = np.arange(steps) / steps
```

precompute

```text
log_kappa_0t[t_idx, distance]
log_kappa_t1[t_idx, distance]
```

using

```python
Gamma_0t = schedule.integrated_clocks(0.0, t)
Gamma_t1 = schedule.integrated_clocks(t, 1.0)
```

and one full-horizon vector

```text
log_kappa_01[distance]
```

for terminal AS labels.

Because every matrix is only `33 x 33`, recomputing with `expm` for 128--512 time points is inexpensive. Optimize with simultaneous diagonalization only if profiling shows this matters.

---

# 45. Direct bridge sampler adaptation

The existing bridge-class table depends only on `(n,k)` and endpoint distance, not on the shell reference.

For each sample:

1. compute endpoint distance `m=d(x0,x1)`;
2. retrieve all feasible four-block occupancy classes for that `m`;
3. class log weight is

\[
\log\#\text{class}
+\log\kappa_{0,t}(d(x_0,z))
+\log\kappa_{t,1}(d(z,x_1));
\]

4. sample one class;
5. choose the required number of coordinates uniformly from each block;
6. construct direct binary `z`.

No state LUT is used.

This is the same mathematical bridge sampler already validated for the single-swap direct-state CuAu plan; only `kappa` changes.

---

# 46. Terminal permutation implementation

Never construct `gX1` by writing "removed current sites become zero and added current sites become one". That would apply the current edge operation to `X1`, not the required coordinate permutation.

For paired coordinates `(r_m,a_m)`, **swap the terminal coordinates**:

```python
def apply_disjoint_transpositions(z, R, A):
    z = z.copy()
    old_R = z[R].copy()
    old_A = z[A].copy()
    z[R] = old_A
    z[A] = old_R
    return z
```

If `X1` has the same symbol at a paired coordinate pair, that transposition leaves that pair unchanged. That is correct.

This distinction is mandatory for the IASBS terminal operator.

---

# 47. Radial terminal-label implementation

```python
def radial_log_label(
    x0,
    x1,
    x,
    y,
    log_kappa_01,
    energy_x1,
    energy_fn,
    tau,
    pairing_fn,
):
    R = np.flatnonzero((x == 1) & (y == 0))
    A = np.flatnonzero((x == 0) & (y == 1))
    if len(R) != len(A):
        raise AssertionError("not fixed-composition edge")

    R, A = pairing_fn(R, A)
    gx1 = apply_disjoint_transpositions(x1, R, A)

    Eg = energy_fn(gx1)
    d1 = johnson_distance(x0, x1)
    dg = johnson_distance(x0, gx1)

    return (
        -(Eg - energy_x1) / tau
        + log_kappa_01[d1]
        - log_kappa_01[dg]
    )
```

Use float64 for energy differences, `tau`, kernel logs, and terminal log labels.

---

# 48. Controller loss implementation sketch

At one bridge state `Xt`:

```text
1. network returns b_j and subset logits for every active shell
2. compute lambda_j = gamma_j(t) * exp(b_j)
3. exact positive term = sum_j lambda_j
4. sample shell J ~ gamma_j / gamma_total
5. sample Y uniformly from reference shell J
6. construct terminal permutation g(Xt,Y)
7. compute Lambda_g(X1)
8. compute log q_theta,J(Y | Xt) via ESP normalizers
9. log m = b_J + log v_J + log q
10. loss = sum_j lambda_j - gamma_total * Lambda * log m
```

In code, stabilize the heavy-tailed label carefully. Do not silently turn a mathematically exact label into an aggressively clipped label without reporting the approximation.

If multiple edge samples `M_edge` are used, average only the stochastic second term:

\[
\sum_j\lambda_j
-\gamma_{\rm tot}\frac1M\sum_{m=1}^M\Lambda_m\log m_m.
\]

---

# 49. Controller destination likelihood

Given `x,y,j`, the removed and added sets are unique:

```python
R = ((x == 1) & (y == 0)).nonzero()
A = ((x == 0) & (y == 1)).nonzero()
```

Therefore the product-subset likelihood has no ordering ambiguity:

```python
log_q = (
    remove_logits[R].sum()
    - log_esp(remove_logits[occ], j)
    + add_logits[A].sum()
    - log_esp(add_logits[emp], j)
)
```

This is why fixed-cardinality product distributions are preferable to a naive sequential `top-k` autoregressive sampler: they give a tractable probability for the **unordered destination state**.

---

# 50. Why not use Gumbel top-k as the primary `q`

Gumbel top-k / sequential Plackett--Luce gives easy sampling, but the probability of an **unordered set** generally requires summing over selection orders.

Since an edge is identified by the final destination set, that likelihood complication is undesirable for the Bregman loss.

The elementary-symmetric-polynomial distribution gives both exact unordered-set likelihood and exact sampling.

---

# 51. Numerical treatment of `Lambda`

At 500 K, `tau=k_B T` is small, and large-shell energy differences can create extreme `log Lambda`.

Record the unclipped `log Lambda` in float64.

Preferred escalation order:

1. geometry-aware operator pairing;
2. reduce maximum shell near late time;
3. increase edge minibatch count;
4. use robust/log-domain Bregman implementation;
5. only then consider an explicit numerical clamp.

If a clamp is used, report:

```text
clamp interval
fraction of labels clipped by shell and time
results with at least one wider clamp on a representative run
```

A radial method whose apparent success depends on clipping a large fraction of the exact label is not convincing evidence for the exact construction.

---

# 52. Reference-vs-label tradeoff as `j` increases

There are two opposing trends.

## Reference benefit

Intermediate shells can mix the Dirac reference across `Omega_{n,k}` extremely rapidly. This flattens the full-horizon reference kernel and can reduce the source-kernel correction in

\[
f_1(z)=e^{-E(z)/\tau}/p^{\rm base}(z|x_0).
\]

## Label cost

The operator `g` for a distance-`j` edge contains `j` transpositions and can induce a larger terminal target-energy contrast.

So the useful maximum shell solves a practical tradeoff:

\[
\boxed{
\text{better reference mixing}
\quad\text{vs.}\quad
\text{harder terminal energy-ratio supervision}.
}
\]

This is the central empirical question of the radial experiment.

---

# 53. Proposed `j` ablation

Do not jump directly from `j=1` to the full schedule.

At 500 K run a staged one-seed ablation:

```text
A0: {1}
A1: {1,2}
A2: {1,2,4}
A3: {1,2,4,8}
A4: {1,2,4,8,16}
```

Keep total event clock fixed first.

For each variant report:

```text
reference TV-to-uniform at t=1
label p99 abs(log Lambda)
energy distribution metric
Q_max metric
orientation mode imbalance
CE calls
wall time
changed-site activity
```

This reveals whether performance saturates before `j=16`.

---

# 54. Pairing ablation

At fixed shell set, compare:

```text
P0: sorted-index pairing
P1: periodic-distance Hungarian pairing
```

Primary pairing-selection metric should be terminal-label stability, not final test performance:

```text
std(log Lambda)
p99 abs(log Lambda)
clipped-label fraction
```

Then freeze pairing before final training seeds.

---

# 55. Controller-capacity ablation

Only if the radial reference clearly improves mixing but the controller fails to exploit it:

```text
C1: one product subset component
C4: four-component mixture
CR: addition distribution conditioned on removed set
```

Do not run this ablation before the one-component controller passes all mathematical and small-system tests.

---

# 56. Exact reference benchmark independent of learning

A useful figure can show the reference itself.

For each schedule, plot exact orbit mass `q_{0,t}(a)` against uniform orbit mass at several times.

For `N=64`, this is exact despite the `1.8e18` state space.

Possible panels:

```text
single swap j=1
a radial fixed mixture
coarse-to-fine radial
uniform canonical reference line
```

This gives a clean mechanistic explanation for why the controller sees a less source-localized base process.

---

# 57. What would count as success against MetaDNS

The radial experiment does not need to beat MetaDNS on every number to be scientifically useful, but the strongest outcome is:

> A hard-composition-preserving radial IASBS closes most or all of the performance gap to the masked sampler while retaining exact fixed-composition support at every intermediate and terminal state.

Particularly strong evidence would be:

- correct three-way L1_0 mode mass at 500 K;
- better `Q_max` and energy fidelity than single-swap IASBS at matched CE budget;
- large reduction in mode trapping;
- zero composition violations;
- learned use of coarse shells early and local shells late;
- manageable terminal-label tails without aggressive clipping.

---

# 58. Failure interpretations

## Failure mode 1: reference becomes excellent but labels explode

Interpretation: the admissible triple is computational on the reference side but statistically difficult for terminal supervision at aggressive shells.

Action:

```text
geometry pairing
lower max j
coarse-to-fine schedule
more edge samples
```

## Failure mode 2: labels are stable but controller does not improve sampling

Interpretation: product subset `q_theta` is too restrictive.

Action: mixture components or conditional addition head.

## Failure mode 3: controller rates explode

Inspect `b_j` and total escape rate. Add explicit regularization or parameter bounds, but document them.

## Failure mode 4: large shells are never used

If learned `lambda_j/sum lambda` goes to zero for large `j`, the target-aware optimum within the learned model may prefer local movement even though the base mixes quickly. This is informative, not necessarily a bug.

## Failure mode 5: radial IASBS still loses badly to MetaDNS

Then the remaining advantage is likely not merely the single-swap Johnson geometry. The masked partial-state construction may provide a fundamentally easier generative path for this target. At that point, do not keep tuning radial shell clocks indefinitely; move to the continuous headline benchmark or a masked constrained-state theory project.

---

# 59. Exact small-system paper-worthy proposition if the experiment succeeds

A concise theoretical extension could eventually be stated as:

> **Proposition (radial Johnson admissibility).** On `Omega_{n,k}`, every time-dependent radial Johnson generator `L_t=sum_j gamma_j(t)(K_j-I)` commutes with coordinate-permutation pullbacks. Every active edge admits a covering permutation, and the reference transition kernel is computable from the `(D+1)`-dimensional Johnson association-scheme quotient. Hence the fixed-composition IASBS terminal ratios remain exactly evaluable for arbitrary nonnegative shell schedules satisfying the required positivity condition.

A second structural statement:

> **Proposition (maximal permutation-invariant family).** Every CTMC generator on `Omega_{n,k}` invariant under the diagonal action of `S_n` is a radial Johnson generator.

Do not add these to the paper until the implementation and experiments justify the extra scope.

---

# 60. Suggested output JSON schema

```json
{
  "benchmark": "cuau_4x4x4_radial",
  "temperature_K": 500.0,
  "n": 64,
  "k": 32,
  "source": "random_fixed_seed1729",
  "shells": [1, 2, 4, 8, 16],
  "schedule": "coarse_to_fine_v1",
  "gamma_total": 10.0,
  "j1_floor": 0.05,
  "pairing": "pbc_hungarian",
  "controller": "esp_product",
  "controller_components": 1,
  "train_seed": 0,
  "energy_calls": 0,
  "wall_seconds": 0.0,
  "controlled_events_by_j": {},
  "changed_sites_total": 0,
  "label_stats_by_j": {},
  "metrics": {
    "energy_w1": null,
    "energy_ks": null,
    "qmax_w1": null,
    "mode_imbalance": null,
    "composition_violations": 0
  }
}
```

Do not overwrite old single-swap JSON files.

---

# 61. Suggested CLI

```bash
# Pure mathematical verification
python -m structured_asbs.tests_radial_johnson

# Reference-only diagnostics
python -m structured_asbs.cuau_radial ref-diagnostics \
    --size 4 4 4 --temperature 500 \
    --shells 1 2 4 8 16 --schedule coarse \
    --gamma-total 10

# Label diagnostic by shell
python -m structured_asbs.cuau_radial label-diagnostics \
    --size 4 4 4 --temperature 500 \
    --shells 1 2 4 8 16 \
    --pairing pbc-hungarian \
    --reference-samples data/cuau/pt_500K.npz

# One-seed radial training
python -m structured_asbs.cuau_radial train \
    --size 4 4 4 --temperature 500 \
    --shells 1 2 4 8 16 --schedule coarse \
    --gamma-total 10 --controller esp-product \
    --seed 0

# Evaluation
python -m structured_asbs.cuau_radial evaluate \
    --checkpoint checkpoints/cuau_radial_seed0.pt \
    --reference-samples data/cuau/pt_500K.npz
```

Exact argument names can follow the repository's existing CLI style; the conceptual subcommands should remain separate so reference/label diagnostics do not accidentally train a model.

---

# 62. Hard failure guards

Raise an exception, not a warning, if any of the following occur:

```text
state composition != k
shell move has |R| != |A| != j
B_j row sum differs materially from 1
radial kernel has materially negative probability
required kernel denominator is zero
terminal g does not satisfy g(x) == y
ESP normalizer is nonfinite
sampled subset size != j
q_theta shell likelihood is nonfinite
controlled total rate is nonfinite or negative
CuAu energy parity gate fails
PT reference file composition is not exactly 32/64
```

Log but do not necessarily abort on extreme finite terminal labels; those are a statistical diagnostic.

---

# 63. Minimal implementation order

1. `shell_orbit_matrix`.
2. mixed `radial_kappa`.
3. tiny full-generator tests.
4. piecewise shell schedule.
5. direct uniform shell sampler.
6. direct radial bridge using existing four-block tables.
7. terminal permutation + AS label.
8. terminal-identity brute-force test.
9. ESP log normalizer.
10. ESP exact subset sampler.
11. normalized shell controller.
12. tiny-system Bregman test.
13. exact piecewise controlled event simulator.
14. CuAu CE adapter/parity test.
15. N=64 reference and label diagnostics.
16. one-seed training.
17. multi-seed final experiment.

Do not reverse this order.

---

# 64. Rough implementation footprint

If the direct-state CuAu adapter/metrics from the earlier playbook already exist, the radial extension should roughly require:

```text
radial orbit kernel + schedule        100-150 LOC
ESP normalization/sampling            100-150 LOC
radial controller                     100-180 LOC
controlled simulator                   80-130 LOC
terminal operator/labels               60-100 LOC
math/unit tests                        150-250 LOC
CLI/diagnostics                        150-250 LOC
```

Expected total new code is roughly `700-1,200 LOC` for a robust implementation, less for a prototype.

The normalized controller is more work than simple thinning but is the correct scalable architecture if the radial method is to become a serious large-action-space experiment rather than a mathematical curiosity.

---

# 65. External mathematical references

Useful background references for checking the association-scheme statements:

- Association schemes / Bose--Mesner algebra: https://mathworld.wolfram.com/AssociationScheme.html
- Intersection numbers as structure constants: https://mathworld.wolfram.com/AssociationSchemeIntersectionNumber.html
- Spectral/geometric bases of association schemes: Cambridge Journal article, "Random walks and the Euclidean association scheme in finite vector spaces": https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/random-walks-and-the-euclidean-association-scheme-in-finite-vector-spaces/685F400F48B73B2310FB09536CE99F65
- Current IASBS manuscript already cites Bannai et al. for Johnson/nonbinary-Johnson structure.

CuAu benchmark source:

- MetaDNS official repository: https://github.com/xiaochendu/metadns

---

# 66. Final decision rule

The radial construction is mathematically admissible and reference-computable for arbitrary nonnegative Johnson shell mixtures. The experiment should proceed if and only if the implementation passes the exact small-state tests and the CuAu label diagnostic shows that at least one aggressive shell beyond `j=1` has manageable supervision variance.

For CuAu-M, the most informative first comparison is:

```text
single-swap IASBS
vs
radial coarse-to-fine IASBS with max j in {4,8,16}
```

at matched CE-call budgets.

The computational dependence on `j` should be summarized as:

\[
\boxed{
\begin{array}{l}
\text{shell cardinality: combinatorial/exponential-looking in }j,\\
\text{reference kernel: independent of shell cardinality, }(D+1)\text{-state},\\
\text{reference proposal: }O(j)\text{--}O(N),\\
\text{normalized controlled destination: }O(Nj),\\
\text{terminal label: one energy call + }O(j)\text{ state/permutation work},\\
\text{main practical risk: label variance grows, not action enumeration.}
\end{array}
}
\]

That is the implementation principle to preserve throughout the experiment.
