# Appendix A.2 Fixed-Support Experiments: Larger Exact Toy, GB1, and Optional GB1-Product Scaling

## Implementation and experiment specification

**Repository:** `iasbs-main`  
**Scope:** Appendix A.2 fixed-support state space only  
**Methods:** Dirac IASBS, non-Dirac IASBS, Discrete Adjoint Matching (DAM)  
**Primary outputs:** two new exact-TV rows for Table 1, plus one optional large-scale GB1-product benchmark  
**Hard constraint:** minimal LOC change; do not refactor existing working Ising, occupation, sphere, Stiefel, or DAM code.

---

## 0. Goal and non-goals

Implement one reusable Appendix A.2 fixed-support experiment module, validate it first on a larger synthetic enumerable target, then reuse the **identical state-space/reference/controller/bridge/DAM mathematics** on the real GB1 protein-fitness landscape. After those two exact-TV experiments are complete, optionally reuse the same implementation on a two-block GB1 product target with more than nine million states.

The two required Table 1 rows are:

| benchmark | `n` | `k` | `r` | exact state count |
|---|---:|---:|---:|---:|
| Fixed-support heterogeneous Potts toy | 14 | 4 | 4 | 81,081 |
| GB1, exactly 3 mutations from WT | 4 | 3 | 20 | 27,436 |

For each required Table 1 row run:

1. Dirac IASBS;
2. non-Dirac IASBS with a fixed four-atom source mixture;
3. source-matched Dirac DAM.

The headline metric for every successful Table 1 run is the **exact terminal-law total variation distance**, not empirical TV.

The optional scaling benchmark is:

| benchmark | `n` | `k` | `r` | state count | role |
|---|---:|---:|---:|---:|---|
| two-block GB1 product, exactly 4 total mutations | 8 | 4 | 20 | 9,122,470 | scalability; no exact-TV propagation |

If the optional scaling benchmark is included in the paper, run the same three methods. Its evaluation uses exact iid target samples and analytically known block-mutation-split probabilities, not exact propagation of a 9.12M-dimensional terminal law. It must be described as a **synthetic product of an empirical GB1 landscape**, not as an experimentally assayed eight-site protein landscape.

Do **not** modify or rerun the occupation experiments as part of this task.

Do **not** introduce Hydra, Lightning, a new experiment framework, a generic state-space abstraction, or any other refactor. Preserve the repository's current standalone-script style.

---

# 1. Files to change

Create or edit only the following files unless a test exposes a genuine blocker.

### New file

`structured_asbs/fixed_support.py`

This file owns all Appendix A.2-specific code:

- state enumeration and lookup;
- legal fixed-support moves;
- group actions used by the matching label;
- orbit indices and orbit-chain kernel;
- exact reference-bridge sampler;
- toy target;
- GB1 data loader and target;
- optional two-block GB1-product target and exact target sampler;
- controller and non-Dirac corrector networks;
- Dirac IASBS training;
- non-Dirac IASBS training;
- exact propagation/evaluation for the two Table 1 spaces;
- scalable on-the-fly transition path and bridge cache for the optional product benchmark;
- CLI.

### Existing files with small additions

`dam/discrete.py`

Add:

- `from structured_asbs import fixed_support as FS`;
- `FixedSupportAdapter`;
- `run_fixed_support(args)`;
- CLI command `fixed-support`;
- target/data arguments needed by that command.

`structured_asbs/tests_math.py`

Add the deterministic Appendix A.2 tests in Section 14 below.

`requirements.txt`

Add exactly:

```text
openpyxl==3.1.5
```

`.gitignore`

Ignore the two GB1 `.xlsx` source files if they are not intended to be committed.

`data/gb1/README.md`

Add the original data provenance and filenames given in Section 13.

`README.md`

After the experiments are complete, add the exact commands and results. Do not update claims before the runs exist.

### Do not modify

Unless a deterministic unit test proves that it is unavoidable, do not modify:

- `common.py`;
- `structured_asbs/fixed_ising.py`;
- `structured_asbs/occupation.py`;
- manifold code;
- `dam/core.py`.

Duplicating the small exact-propagation/simulation pattern from `fixed_ising.py` inside `fixed_support.py` is preferable to refactoring working code.

---

# 2. State space

Use exactly the Appendix A.2 state space

$$
\mathcal X_{n,k}^{(r)}
=
\left\{
 x\in\{0,1,\ldots,r-1\}^n:
 \#\{i:x_i\neq 0\}=k
\right\}.
$$

`0` is the inactive/wild-type label. Labels `1,...,r-1` are active/nonzero labels.

Its cardinality is

$$
\left|\mathcal X_{n,k}^{(r)}\right|
=
\binom nk(r-1)^k.
$$

For the toy:

$$
\left|\mathcal X_{14,4}^{(4)}\right|
=
\binom{14}{4}3^4
=
81,081.
$$

For GB1 at exactly three mutations:

$$
\left|\mathcal X_{4,3}^{(20)}\right|
=
\binom43 19^3
=
27,436.
$$

## 2.1 Enumeration order

Implement

```python
def enumerate_fixed_support(n: int, k: int, r: int) -> np.ndarray:
    ...
```

with this deterministic order:

1. support subsets from `itertools.combinations(range(n), k)`;
2. for each support, active labels from `itertools.product(range(1, r), repeat=k)`;
3. inactive positions are zero.

Return an integer array of shape `(M, n)`.

Assert:

```python
M == math.comb(n, k) * (r - 1) ** k
np.all((S != 0).sum(axis=1) == k)
np.unique(S, axis=0).shape[0] == M
```

## 2.2 Exact state ranking; do not allocate a dense `r**n` LUT

A smaller fixed-support toy could afford a dense base-`r` lookup table. The larger
`n=14` toy must **not** do this: a dense `4**14` int64 LUT would consume about 2.15 GB.
Use the exact enumeration rank induced by Section 2.1 instead.

Let the support of `x` be

$$
S(x)=\{c_0<\cdots<c_{k-1}\}.
$$

Precompute all support combinations once:

```python
supports = np.asarray(list(itertools.combinations(range(n), k)), dtype=np.int16)
```

Assign each support its `itertools.combinations` rank using a tiny bit-mask LUT:

```python
support_rank_lut = np.full(1 << n, -1, dtype=np.int32)
for rank, c in enumerate(supports):
    mask = sum(1 << int(i) for i in c)
    support_rank_lut[mask] = rank
```

For the two Table 1 spaces and the optional product space this table has only:

```text
toy:        2**14 = 16,384 entries
GB1 k=3:    2**4  = 16 entries
product2:   2**8  = 256 entries
```

For the nonzero labels on the ordered support, define the base-`(r-1)` label rank

$$
R_{\mathrm{lab}}(x)
=
\sum_{q=0}^{k-1}
(x_{c_q}-1)(r-1)^{k-1-q}.
$$

Then the exact state index is

$$
\operatorname{index}(x)
=
R_{\mathrm{sup}}(S(x))(r-1)^k
+
R_{\mathrm{lab}}(x).
$$

This is **exactly** the row index generated by the enumeration order in Section 2.1.
Implement:

```python
def state_to_index_np(self, x: np.ndarray) -> np.ndarray:
    ...

def state_to_index_torch(self, x: torch.Tensor) -> torch.Tensor:
    ...

def index_to_state(self, idx):
    ...
```

For a Torch batch, compute the support bitmask with powers of two, gather
`support_rank_lut[mask]`, collect the `k` nonzero labels in ascending site order, and evaluate the
base-`(r-1)` rank. Since every valid state has exactly `k` nonzeros, a row-major boolean gather
followed by `.reshape(B, k)` is valid after asserting the constraint.

Mandatory exact round-trip tests are:

```python
idx = np.arange(M)
np.testing.assert_array_equal(state_to_index_np(S), idx)
np.testing.assert_array_equal(index_to_state(idx), S)
```

and the equivalent randomized Torch test.

For the two enumerable paper benchmarks, store `S` in RAM and use this rank map for every transformed
state. For the 9.12M-state optional scaling benchmark, the same map avoids a gigantic `20**8` LUT.

A dense `20**4` **fitness** table for the original four-site GB1 landscape is allowed because it contains
only 160,000 entries. It is a target-data table, not the fixed-support state-index map.

Any transformed state whose support mask has rank `-1`, or whose rank is outside `[0,M)`, is an
implementation bug and must raise immediately.

---

# 3. Legal moves and reference CTMC

There are two legal edge families.

## 3.1 Relabel move

If `x[i] = a != 0`, replace `a` by any

$$
b\in\{1,\ldots,r-1\}\setminus\{a\}.
$$

The number of relabel edges from every state is

$$
E_{\mathrm{lab}}=k(r-2).
$$

## 3.2 Support move

If `x[i] = a != 0` and `x[j] = 0`, set

$$
x_i'=0,
\qquad
x_j'=b,
$$

for any

$$
b\in\{1,\ldots,r-1\}.
$$

The number of support edges from every state is

$$
E_{\mathrm{sup}}
=
k(n-k)(r-1).
$$

Hence

$$
E
=
E_{\mathrm{lab}}+E_{\mathrm{sup}}.
$$

Use one deterministic edge order for every state:

1. **relabel edges first**:
   - active site indices ascending;
   - new labels ascending, skipping the current label;
2. **support edges second**:
   - active source index ascending;
   - empty destination index ascending;
   - new nonzero label ascending.

For the two required Table 1 spaces, precompute `tgt[M,E]` and the per-edge descriptors needed later:

- `edge_type`: relabel/support;
- `edge_i`;
- `edge_j` (`-1` for relabel if convenient);
- `edge_old`;
- `edge_new`;
- `edge_action_id` from Section 7.

Every row of `tgt` must contain `E` distinct valid neighbors. Store large integer tables as `int32` on disk/CPU where possible; cast indices only where a Torch operation requires another integer dtype. The optional 9.12M-state product benchmark must **not** allocate `tgt[M,E]`; Section 27 generates legal moves on the fly.

---

# 4. Reference rates: choose equal per-edge rate

Use one constant total reference escape rate

$$
\gamma=10.
$$

The Appendix A.2 rates are

$$
r(y,x)
=
\frac{\alpha}{k(r-2)}
$$

for a relabel edge, and

$$
r(y,x)
=
\frac{\beta}{k(n-k)(r-1)}
$$

for a support edge.

Choose

$$
\alpha
=
\gamma\frac{E_{\mathrm{lab}}}{E},
$$

and

$$
\beta
=
\gamma\frac{E_{\mathrm{sup}}}{E}.
$$

Then **every individual legal edge has the same rate**

$$
r(y,x)=\frac{\gamma}{E},
$$

and every state has total reference escape rate exactly `gamma`.

This choice is intentional: it preserves the Appendix A.2 two-family reference while allowing the existing fixed-Ising exact propagation and DAM patterns to be reused almost verbatim.

### Toy constants

For `n=14,k=4,r=4`:

$$
E_{\mathrm{lab}}=8,
\qquad
E_{\mathrm{sup}}=120,
\qquad
E=128.
$$

Therefore

$$
\alpha=0.625,
\qquad
\beta=9.375,
\qquad
r(y,x)=\frac{10}{128}=0.078125.
$$

### GB1 constants

For `n=4,k=3,r=20`:

$$
E_{\mathrm{lab}}=54,
\qquad
E_{\mathrm{sup}}=57,
\qquad
E=111.
$$

Therefore

$$
\alpha=\frac{180}{37}\approx4.8648648649,
$$

$$
\beta=\frac{190}{37}\approx5.1351351351,
$$

and

$$
r(y,x)=\frac{10}{111}.
$$

Do not separately tune `alpha`, `beta`, or `gamma` between IASBS and DAM.

---

# 5. Appendix A.2 orbit kernel

For `x,y in X`, define

$$
i(x,y)
=
\#\{\ell:x_\ell\neq0,\ y_\ell\neq0,\ x_\ell\neq y_\ell\},
$$

and

$$
j(x,y)
=
k-
\#\{\ell:x_\ell\neq0,\ y_\ell\neq0\}.
$$

Let

$$
\sigma(x,y)=(i(x,y),j(x,y)).
$$

Valid orbit indices satisfy

$$
i\ge0,
\qquad
j\ge0,
\qquad
i+j\le k,
\qquad
j\le n-k.
$$

Use this deterministic orbit ordering:

```python
orbits = [
    (i, j)
    for j in range(min(k, n-k) + 1)
    for i in range(k - j + 1)
]
```

Thus:

- toy orbit count `C = 15`;
- GB1 `k=3` orbit count `C = 7`.

Create a small integer `orbit_lut[i,j] -> orbit_id` for vectorized Torch lookup.

## 5.1 Orbit valencies

For each valid `(i,j)`, use

$$
N_{ij}
=
\binom{k}{j}
\binom{n-k}{j}
(r-1)^j
\binom{k-j}{i}
(r-2)^i.
$$

Assert

$$
\sum_{i,j} N_{ij}
=
\left|\mathcal X_{n,k}^{(r)}\right|.
$$

Also verify by enumeration that, for the canonical source in Section 8, exactly `N_ij` states have orbit `(i,j)`.

## 5.2 Orbit generators

Construct dense float64 matrices `B_lab` and `B_sup` of shape `(C,C)`.

For

$$
s_{ij}=k-i-j,
$$

the relabel generator has off-diagonal entries

$$
B_{\mathrm{lab}}((i,j),(i+1,j))
=
\frac{k-i-j}{k},
$$

and

$$
B_{\mathrm{lab}}((i,j),(i-1,j))
=
\frac{i}{k(r-2)}.
$$

For support moves define

$$
z_j=n-k-j,
$$

and

$$
D=k(n-k)(r-1).
$$

The six off-diagonal entries are

$$
B_{\mathrm{sup}}((i,j),(i+1,j))
=
\frac{s_{ij}j(r-2)}{D},
$$

$$
B_{\mathrm{sup}}((i,j),(i-1,j))
=
\frac{ij}{D},
$$

$$
B_{\mathrm{sup}}((i,j),(i,j+1))
=
\frac{s_{ij}z_j}{k(n-k)},
$$

$$
B_{\mathrm{sup}}((i,j),(i-1,j+1))
=
\frac{i z_j}{k(n-k)},
$$

$$
B_{\mathrm{sup}}((i,j),(i,j-1))
=
\frac{j^2}{D},
$$

and

$$
B_{\mathrm{sup}}((i,j),(i+1,j-1))
=
\frac{j^2(r-2)}{D}.
$$

Only add a transition if the destination orbit is valid. Set each diagonal to minus its row's off-diagonal sum.

Assert:

```python
np.max(np.abs(B_lab.sum(1))) < 1e-12
np.max(np.abs(B_sup.sum(1))) < 1e-12
np.min(B_lab - np.diag(np.diag(B_lab))) >= 0
np.min(B_sup - np.diag(np.diag(B_sup))) >= 0
np.max(np.abs(B_lab @ B_sup - B_sup @ B_lab)) < 1e-12
```

The last test is mandatory.

## 5.3 Kernel evaluator

For elapsed time `Delta in [0,1]`, define

$$
\Gamma_{\mathrm{lab}}(\Delta)=\alpha\Delta,
$$

and

$$
\Gamma_{\mathrm{sup}}(\Delta)=\beta\Delta.
$$

Compute

$$
q_\Delta
=
e_{(0,0)}^\top
\exp\left(
\Gamma_{\mathrm{lab}}(\Delta)B_{\mathrm{lab}}
+
\Gamma_{\mathrm{sup}}(\Delta)B_{\mathrm{sup}}
\right).
$$

Then

$$
\kappa_\Delta(i,j)
=
\frac{q_\Delta(i,j)}{N_{ij}}.
$$

Use `scipy.linalg.expm`. The matrix is only `15x15` for the toy and `7x7` for GB1.

Implement

```python
def kappa(self, delta: float) -> np.ndarray:
    ...
```

and

```python
def log_kappa_grid(self, steps: int):
    ...
```

for bridge training.

At `delta=0`, do not take `log(0)` and pretend it is a finite kernel. Handle bridge samples at `t=0` exactly by returning the source state. For positive `delta`, clamp only before the logarithm with `1e-300`.

The full-horizon kernel is

```python
kappa_full = kappa(1.0)
log_kappa_full = log(kappa_full)
```

and is used by all terminal ratios.

---

# 6. Canonical group action representation

Appendix A.2 uses site permutations together with independent nonzero-label permutations. Implement only the subset needed by these experiments.

Represent an action by:

```text
perm[n]
swap_a[n]
swap_b[n]
```

with the convention

$$
(Tx)_q
=
\lambda_q\!\left(x_{\operatorname{perm}[q]}\right),
$$

where `lambda_q` is either identity or the transposition `swap_a[q] <-> swap_b[q]`. Label `0` is always fixed.

Use `-1,-1` to encode identity at a site.

Implement:

```python
def apply_action(x, perm, swap_a, swap_b):
    ...
```

for Torch batches.

Implement the inverse exactly. If

```python
inv_perm = np.argsort(perm)
```

then the inverse action at output position `p` uses the transposition stored at original output position `inv_perm[p]`. A transposition is its own inverse.

Unit-test `T_inv(T(x)) == x` on random feasible states and all action families used below.

---

# 7. Group action associated with each legal edge

The terminal label needs one fixed group action `g` satisfying `T^g(x)=y` for each legal edge `(x,y)`.

## 7.1 Relabel edge

For a relabel edge

```text
site i: a -> b
```

use:

- identity site permutation;
- label transposition `(a,b)` at site `i`;
- identity label maps at all other sites.

This maps the current state exactly to its relabeled neighbor.

## 7.2 Support edge

For

```text
site i: a -> 0
site j: 0 -> b
```

use:

- site transposition `i <-> j`;
- at **output site `j`**, apply the label transposition `(a,b)` if `a != b`;
- if `a == b`, use no label transposition;
- all other label maps are identity.

With the convention of Section 6 this maps the current state exactly to the desired support neighbor.

## 7.3 Global action IDs

The learned non-Dirac corrector is indexed by the group action, not by a local edge slot.

Create a deterministic global action table.

Relabel actions are identified by

```text
("lab", site, min(a,b), max(a,b))
```

with `1 <= a < b <= r-1` after sorting.

Support actions are identified by

```text
("sup", i, j, label_action)
```

where `i != j` is ordered and `label_action` is either:

- `("id",)` for the pure site swap `a == b`, or
- `(min(a,b), max(a,b))` for a nonzero-label transposition.

Therefore the number of global actions is

$$
A_{\mathrm{lab}}
=
n\binom{r-1}{2},
$$

and

$$
A_{\mathrm{sup}}
=
n(n-1)\left(1+\binom{r-1}{2}\right).
$$

This gives:

- toy: `42 + 728 = 770` global actions;
- GB1: `684 + 2064 = 2748` global actions.

Precompute the action representation for every action ID. For the two required Table 1 spaces also precompute `edge_action_id[M,E]` for every legal local edge. The optional product benchmark constructs the local action ID on the fly and must not allocate a 9.12M-by-376 action table.

Mandatory assertion:

```python
apply_action_id(S[x], edge_action_id[x,e]) == S[tgt[x,e]]
```

for all micro-test states and a large random sample from both required Table 1 spaces.

---

# 8. Canonical source and exact bridge sampler

The canonical Dirac source is

```python
x0 = np.zeros(n, dtype=np.int64)
x0[:k] = 1
```

so the first `k` sites are active with label `1`.

The exact reference bridge must not enumerate all `M` candidate states per training sample.

Use the following finite-class construction.

## 8.1 Canonical endpoint for an orbit

For orbit `c=(i,j)`, let

$$
s=k-i-j.
$$

Define canonical endpoint `y_c` as:

- among sites `0,...,k-1`:
  - first `s` entries are `1`;
  - next `i` entries are `2`;
  - next `j` entries are `0`;
- among sites `k,...,n-1`:
  - first `j` entries are `1`;
  - all remaining entries are `0`.

Assert

$$
\sigma(x_0,y_c)=c.
$$

## 8.2 Precompute bridge classes for canonical endpoints

For each orbit `c` and each enumerated state `z`, compute

$$
a=\sigma(x_0,z),
$$

and

$$
b=\sigma(z,y_c).
$$

States with the same pair `(a,b)` have identical bridge weight.

For every endpoint orbit `c`, precompute a compact partition:

```text
bridge_members[c]       flattened state indices
bridge_offsets[c]       offsets for keys a*C+b
bridge_counts[c,key]
```

The total stored membership is only `C*M` integers:

- toy: `15 * 81,081 = 1,216,215`;
- GB1: `7 * 27,436 = 192,052`.

Do not store an `M x M` transition matrix.

For a canonical bridge from `x0` to `y_c` at time `t`, class `(a,b)` has unnormalized probability

$$
\#\mathcal C_{c,a,b}
\;\kappa_t(a)
\;\kappa_{1-t}(b).
$$

Sample a class from these at most `C^2` weights, then sample one member uniformly from that class.

## 8.3 Map an arbitrary endpoint to the canonical endpoint

For any endpoint `y`, construct deterministically an action `h_y` that:

$$
h_y(x_0)=x_0,
$$

and

$$
h_y(y_{\sigma(x_0,y)})=y.
$$

Construct it deterministically as follows.

Within the first `k` sites of `x0`, partition the positions of `y` into:

1. shared-same: `y=1`;
2. shared-different: `y != 0` and `y != 1`;
3. vacated: `y=0`.

Within the last `n-k` sites partition into:

4. gained: `y != 0`;
5. inactive: `y=0`.

Map the corresponding canonical blocks to these target positions in ascending-position order. This site permutation stays within the active block and inactive block of `x0`, so it fixes `x0`.

Then:

- at each shared-different output site, the canonical value is `2`; transpose `2` with the desired target label if the desired target label is not `2`;
- at each gained output site, the canonical value is `1`; transpose `1` with the desired target label if the target label is not `1`;
- all other sitewise label maps are identity.

Mandatory tests:

```python
apply(h_y, x0) == x0
apply(h_y, canonical_endpoint[orbit(y)]) == y
```

for every state in the micro test and random states from both required Table 1 spaces. For the optional 9.12M-state product benchmark, construct `h_y` on demand and do not precompute one action object per endpoint.

## 8.4 Dirac bridge sampling

To sample

$$
X_t\mid X_0=x_0, X_1=y,
$$

perform:

1. if `t_idx == 0`, return `x0` exactly;
2. `c = sigma(x0,y)`;
3. sample canonical `z_c` from the precomputed classes for endpoint `y_c`;
4. return `h_y(z_c)`.

Map the transformed state back to its enumeration index with `state_to_index_torch`; do not use a dense base-`r` LUT.

This sampler is exact up to floating-point evaluation of the small matrix exponential.

---

# 9. Controller network

Use one controller class for toy and GB1.

```python
class FixedSupportController(nn.Module):
    ...
```

Input state encoding is flattened categorical one-hot:

```python
F.one_hot(x, num_classes=r).flatten(1).float()
```

Use exactly the same time features as `SwapController`:

```text
t
1-t
sin(pi*q*t), q=1,...,4
cos(pi*q*t), q=1,...,4
```

Network:

```text
Linear(n*r + 10, 512)
SiLU
Linear(512, 512)
SiLU
Linear(512, 512)
SiLU
Linear(512, n*(r-1) + n*n*(r-1))
```

Use float32 network parameters, as in the current discrete controllers. Zero-initialize the final layer weight and bias.

Interpret outputs as:

```text
lab  : (B, n, r-1)
sup  : (B, n, n, r-1)
```

For a legal relabel edge `i: old -> b`, read

```python
lab[:, i, b-1]
```

For a legal support edge `i -> j` with new label `b`, read

```python
sup[:, i, j, b-1]
```

The old label does not need its own output axis because it is already encoded in the input state.

Implement:

```python
net_mult_on_index(space, net, t, idx) -> (B,E)
net_mult_all_states(space, net, t, chunk=4096) -> (M,E)
```

with the same clamping convention as the existing discrete experiments when converting log multipliers to rates.

---

# 10. Dirac IASBS

## 10.1 Target and terminal factor

For any target energy `E(y)` and temperature `tau`, define

$$
\pi(y)
=
\frac{e^{-E(y)/\tau}}
{\sum_z e^{-E(z)/\tau}}.
$$

For the canonical Dirac source,

$$
\log f_1(y)
=
-\frac{E(y)}{\tau}
-
\log\kappa_1(\sigma(x_0,y))
+
\mathrm{const}.
$$

Store a numerically shifted `space.logf1` exactly as `fixed_ising.py` does.

## 10.2 Terminal IASBS label

For the group action `g` associated with a current legal edge, and terminal state `Y`, use

$$
\log\Lambda_g(Y)
=
-\frac{E(gY)-E(Y)}{\tau}
+
\log\kappa_1(\sigma(x_0,Y))
-
\log\kappa_1(\sigma(x_0,gY)).
$$

Compute `gY` using `edge_action_id` and the global action table from Section 7. Do not approximate this ratio and do not roll out an inner adjoint process.

The controller target is

$$
\Lambda_g(Y)=\exp(\log\Lambda_g(Y)).
$$

Use the existing clamp before exponentiation:

```python
lam = exp(log_lam.clamp(-20, 20))
```

The controller loss is the same Poisson/Bregman loss already used by the discrete IASBS code:

$$
\mathcal L_{\mathrm{ctrl}}
=
\mathbb E\left[e^a-a\Lambda\right].
$$

MSE may exist as a development option but **all paper runs use `--loss poisson`**.

## 10.3 Forward process and replay buffer

Follow `fixed_ising.py`:

1. simulate `batch` terminal states on-policy from the current controlled chain;
2. append terminal indices to a replay buffer of `buffer` outer iterations;
3. for each inner update sample terminal states from the buffer;
4. sample time index uniformly from `0,...,steps-1`;
5. sample `X_t` from the exact reference bridge in Section 8;
6. compute every legal-edge target at that `X_t`;
7. perform one Poisson regression update.

No target samples are used in training.

---

# 11. Non-Dirac IASBS

Use a fixed finite source so the branch is genuinely non-Dirac while remaining easy to audit.

## 11.1 Four-atom source

For either target, define these four source states in integer-label coordinates:

```python
s0 = x0.copy()

s1 = x0.copy()
s1[0] = 2

s2 = x0.copy()
s2[k-1] = 0
s2[k] = 1

s3 = x0.copy()
s3[k-1] = 0
s3[k] = 2
```

This is valid for both required Table 1 configurations and for the optional product configuration because `n > k` and `r >= 3`.

Use exactly

$$
\nu_0
=
\frac14\sum_{c=0}^3\delta_{s_c}.
$$

The source is target-independent. Do not select source states using toy energy or GB1 fitness.

Assert all four source states are distinct and feasible.

## 11.2 Source actions for bridge reuse

For each source `s_c`, construct and store an action `a_c` with

$$
a_c(x_0)=s_c.
$$

These actions are simple:

- `s0`: identity;
- `s1`: label transposition `1 <-> 2` at site `0`;
- `s2`: site transposition `(k-1) <-> k`;
- `s3`: site transposition `(k-1) <-> k`, then label transposition `1 <-> 2` at output site `k`.

Store both each action and its inverse using Section 6.

## 11.3 Non-Dirac bridge sampling

To sample

$$
X_t\mid X_0=s_c, X_1=y,
$$

perform:

1. if `t_idx == 0`, return `s_c` exactly;
2. transform endpoint to canonical-source coordinates:

$$
y'=a_c^{-1}(y);
$$

3. use Section 8 to sample `z'` from the canonical-source bridge from `x0` to `y'`;
4. map back:

$$
z=a_c(z').
$$

This reuses exactly the same bridge-class tables.

The replay buffer for non-Dirac training must store both:

```text
source_id
terminal_state_index
```

for each trajectory.

## 11.4 Corrector regression target

For source state `X0`, terminal state `Y`, and global action `g`, use the one-sample corrector label

$$
\log Q_g(X_0,Y)
=
\log\kappa_1(\sigma(X_0,gY))
-
\log\kappa_1(\sigma(X_0,Y)).
$$

The learned corrector satisfies

$$
e^{h_g(Y)}
\approx
\mathbb E[Q_g(X_0,Y)\mid Y].
$$

Use the Poisson/Bregman loss

$$
\mathcal L_h
=
\mathbb E\left[e^h-hQ\right].
$$

## 11.5 Corrector architecture

Do **not** create one dense output for all 2,748 GB1 actions.

Use a selected-action model:

```python
class FixedSupportCorrector(nn.Module):
    # state encoder -> h_y in R^H
    # action embedding -> e_g in R^H
    # scalar bias per action
    # output = <h_y, e_g>/sqrt(H) + b_g
```

Use:

```text
state one-hot dimension : n*r
hidden H                : 256
state encoder           : Linear -> SiLU -> Linear -> SiLU -> Linear
n_actions               : global action count from Section 7
```

Initialize the corrector deterministically after `torch.manual_seed(args.seed)`: use normal PyTorch initialization for the state-encoder `Linear` layers, initialize `action_embed.weight` with `torch.nn.init.normal_(..., mean=0.0, std=0.02)`, and initialize the scalar action-bias embedding to zero. Do not zero-initialize the state encoder. This guarantees a nonzero first-step gradient through the bilinear state/action term while keeping the initial log-corrector near zero.

For each corrector minibatch sample `A_corr = 32` global action IDs uniformly and independently for each terminal state. Compute `Q` only for those selected actions. This gives every global action direct supervision without materializing a `(B,2748,n)` tensor.

Add CLI argument

```text
--corrector-actions 32
```

with paper default `32`.

## 11.6 Exact corrector diagnostic

The four-atom source makes the exact base terminal marginal available:

$$
\widehat f_1(y)
\propto
\frac14
\sum_{c=0}^3
\kappa_1(\sigma(s_c,y)).
$$

Therefore the exact corrector ratio is

$$
C_g(y)
=
\frac{\widehat f_1(gy)}{\widehat f_1(y)}.
$$

This quantity is **evaluation only**. Do not feed it to controller training.

At the end of a non-Dirac run, evaluate the learned `h_g(y)` against

$$
\log C_g(y)
$$

on at least 100,000 uniformly sampled `(y,g)` pairs from the enumerated state/action sets. Save:

- RMSE of `h - log C`;
- mean absolute error;
- maximum absolute error.

This is an important correctness diagnostic for the new branch.

## 11.7 Non-Dirac controller label

The non-Dirac controller label differs from the Dirac label only by replacing the explicit source-kernel correction with the learned corrector:

$$
\log\Lambda_g^{\mathrm{ND}}(Y)
=
-\frac{E(gY)-E(Y)}{\tau}
-
h_g(Y).
$$

Everything else in controller training is identical to the Dirac branch.

---

# 12. Toy benchmark

## 12.1 Configuration

Use exactly:

```text
name   = fixed_support_toy_v2
n      = 14
k      = 4
r      = 4
tau    = 1.0
gamma  = 10.0
```

State count:

```text
81,081
```

Canonical source:

```text
[1,1,1,1,0,0,0,0,0,0,0,0,0,0]
```

## 12.2 Target energy

Use a **heterogeneous dilute-Potts ring** deliberately chosen to break the site/label symmetries of the reference.

Generate coefficients once using the repository's pinned NumPy version:

```python
rng = np.random.default_rng(20260908)
H = rng.normal(0.0, 0.7, size=(n, r))
J = rng.normal(0.0, 0.4, size=(n, r, r))
H[:, 0] = 0.0
J[:, 0, :] = 0.0
J[:, :, 0] = 0.0
```

Define

$$
E_{\mathrm{toy}}(x)
=
\sum_{i=0}^{n-1} H_{i,x_i}
+
\sum_{i=0}^{n-1}
J_{i,x_i,x_{(i+1)\bmod n}}.
$$

Use

$$
\tau=1.
$$

Do not symmetrize `H`, `J`, or the target.

Save the generated `H` and `J` arrays in each checkpoint's metadata, together with the seed, so the target is reconstructable even if the RNG implementation changes in a future environment.

The target is exactly enumerable:

```python
logw = -E
logw -= logw.max()
pi = exp(logw)
pi /= pi.sum()
```

Before training, print and save:

- `E.min()`;
- `E.max()`;
- `E.mean()`;
- `E.std()`;
- `pi.max()`;
- entropy `-(pi*log(pi)).sum()`.

These are diagnostics only, not acceptance gates.

---

# 13. GB1 benchmark

## 13.1 Scientific dataset and required download

**Yes: GB1 requires a one-time external dataset download.** The toy requires no external data, and the optional product2 benchmark reuses the same downloaded GB1 files. Training/evaluation scripts must never download data automatically.

Use the four-site GB1 fitness landscape from:

> Wu NC, Dai L, Olson CA, Lloyd-Smith JO, Sun R. *Adaptation in protein fitness landscapes is facilitated by indirect paths.* eLife 5:e16965 (2016). DOI: `10.7554/eLife.16965`.

Use the original supplementary files:

- Supplementary file 1, profiled/measured variants: DOI `10.7554/eLife.16965.024`;
- Supplementary file 2, author-imputed fitness for missing variants: DOI `10.7554/eLife.16965.025`.

Expected local filenames:

```text
data/gb1/elife-16965-supp1-v4.xlsx
data/gb1/elife-16965-supp2-v4.xlsx
```

Do not silently substitute FLIP, ProteinGym, a third-party transformed fitness, or a reprocessed CSV in the primary experiment.

## 13.2 Required data assertions

Load both workbooks with `openpyxl` in read-only/data-only mode.

Do not hardcode a worksheet name or row number. For each workbook:

1. scan sheets and the first 25 rows for a header row;
2. identify a sequence column whose normalized header is one of `variant`, `variants`, `sequence`, `sequences`;
3. identify the fitness column as a header containing the substring `fitness`;
4. read nonempty rows below it.

Normalize a sequence by `str(value).strip().upper()` and fitness by `float(value)`.

The loader must assert before any experiment runs:

```text
measured unique variants = 149,361
imputed unique variants  = 10,639
intersection             = 0
merged unique variants   = 160,000
```

It must also assert:

- every sequence has length 4;
- every character belongs to the 20 canonical amino-acid alphabet;
- all fitness values are finite;
- all fitness values are nonnegative;
- WT `VDGV` exists;
- WT fitness from the measured table is numerically consistent with `1` to ordinary spreadsheet precision.

Merge policy is exact:

```text
measured values take their measured fitness;
only the 10,639 absent variants receive Supplementary-file-2 imputed fitness.
```

Do not impute anything yourself.

Do not drop author-imputed variants. A 149,361-state measured-only subset is not closed under the Appendix A.2 group actions/reference moves and therefore is not the state space being claimed.

## 13.3 Amino-acid encoding

Use the fixed canonical alphabet

```text
ACDEFGHIKLMNPQRSTVWY
```

The four GB1 sites are ordered exactly as in the four-letter variant string:

```text
V39, D40, G41, V54
```

WT is

```text
VDGV
```

At each site separately:

- integer label `0` is that site's WT amino acid;
- labels `1,...,19` are the remaining amino acids in the order obtained by removing that WT amino acid from the fixed canonical alphabet above.

Implement both directions and unit-test round trips for all 160,000 variants.

## 13.4 Fixed-support sector

The primary GB1 row is exactly Hamming distance 3 from WT:

$$
k=3.
$$

Therefore

```text
n = 4
k = 3
r = 20
M = 27,436
```

After merging the complete 160,000 landscape, filter by

```python
sum(seq[pos] != WT[pos] for pos in range(4)) == 3
```

and assert exactly

```text
27,436 variants
```

Do not use `k=4` as the primary A.2 benchmark because `n-k=0`, which removes the support-move family and fails to exercise the defining two-family fixed-support construction.

`k=2` may be used as an optional development smoke test but is not a Table 1 row.

## 13.5 GB1 target

Use the authors' relative fitness `F(x)` directly to define a strictly positive target.

Set

```text
GB1_FITNESS_EPS = 1e-4
```

and

$$
E_{\mathrm{GB1}}(x)
=
-\log\left(F(x)+10^{-4}\right).
$$

Use

$$
\tau=1.
$$

Thus the exact target within the `k=3` sector is

$$
\pi_{\mathrm{GB1},k=3}(x)
=
\frac{F(x)+10^{-4}}
{\sum_{z:\operatorname{HD}(z,\mathrm{WT})=3}
(F(z)+10^{-4})}.
$$

The epsilon has only one purpose: retain strictly positive support for lethal/zero-fitness variants so the finite-energy bridge problem remains closed. Do not delete zero-fitness variants.

Save in every GB1 result:

```text
paper DOI
supplement DOIs
measured count
imputed count
sector count
WT sequence
epsilon
amino-acid alphabet
```

---

# 14. Deterministic mathematical tests: mandatory before training

Add all of these to `structured_asbs/tests_math.py` or an imported helper called by it. No neural training is allowed until every deterministic test passes.

## Test A2.1 — enumeration and edge closure

Use micro state space

```text
n=5, k=2, r=4, M=90
```

Assert:

- exact cardinality;
- every state has exactly `k` nonzero entries;
- no duplicate states;
- `state_to_index_np(S) == np.arange(M)` exactly and `index_to_state(np.arange(M)) == S` exactly;
- every state has exactly

$$
2(4-2)+2(5-2)(4-1)=22
$$

legal edges;
- every target maps to a valid exact state rank;
- every move preserves exactly `k` active sites;
- no duplicate legal neighbor within a row.

## Test A2.2 — full generator

Construct the micro full reference generator explicitly from `tgt` and the Appendix A.2 rates.

Assert:

```python
max_abs_row_sum < 1e-12
max_abs((-diag(Q)) - gamma) < 1e-12
all offdiagonal entries >= 0
```

## Test A2.3 — orbit valency

For the micro space and both required Table 1 spaces, verify:

$$
\sum_{ij}N_{ij}=M
$$

and the enumerated number of states at each orbit from `x0` equals `N_ij` exactly.

## Test A2.4 — orbit-generator commutation

For micro, toy, GB1, and `gb1-product2` configurations assert:

$$
\|B_{\mathrm{lab}}B_{\mathrm{sup}}-B_{\mathrm{sup}}B_{\mathrm{lab}}\|_\infty
<10^{-12}.
$$

## Test A2.5 — orbit kernel equals full matrix exponential

On the `M=90` micro state, for elapsed clocks

```text
Delta in {0.05, 0.5, 1.0}
```

compare the canonical source row of

```python
scipy.linalg.expm(Delta * Q)
```

against

```python
kappa_Delta[sigma(x0,y)]
```

for all states `y`.

Require

```text
max absolute error < 1e-10
```

and row mass error `<1e-12`.

## Test A2.6 — local edge action

On all micro states and all their edges assert

```python
T_g(S[x]) == S[tgt[x,e]]
```

where `g=edge_action_id[x,e]`.

Also assert action inverse round trip.

## Test A2.7 — equivariance/orbit preservation

For random micro `(x,z,g)` pairs assert

$$
\sigma(T^g x,T^g z)=\sigma(x,z).
$$

This is the implementation-level symmetry check used by the terminal ratio.

## Test A2.8 — canonical endpoint actions

For every micro state `y` assert:

```python
h_y(x0) == x0
h_y(y_c[orbit(y)]) == y
```

Repeat for at least 5,000 random toy states and all 27,436 GB1 states after the GB1 loader is available.

## Test A2.9 — Chapman-Kolmogorov bridge-class identity

For every micro endpoint orbit `c` and

```text
t in {0.1, 0.37, 0.8}
```

compute from the precomputed class counts

$$
S_c(t)
=
\sum_{a,b}
\#\mathcal C_{c,a,b}
\kappa_t(a)
\kappa_{1-t}(b).
$$

Require

$$
|S_c(t)-\kappa_1(c)|<10^{-11}.
$$

Repeat the same identity for the toy and GB1 class tables with tolerance `1e-10`.

## Test A2.10 — empirical bridge sampler

On the micro space only, select at least three endpoint states from different orbits and one `t=0.37`.

Compute the exact full bridge law

$$
P(z\mid x_0,y)
=
\frac{P_t(x_0,z)P_{1-t}(z,y)}{P_1(x_0,y)}.
$$

Draw 100,000 samples from the class sampler and require empirical TV to the exact bridge law `<0.02` for each endpoint.

This is a stochastic unit test; mark it slow if desired, but run it before the final experiments.

## Test A2.11 — terminal IASBS identity on the micro state space

Choose a strictly positive arbitrary terminal function `f1`, a time `t=0.43`, a state `x`, and several legal edge actions `g` from `x`.

Using the explicit full reference matrix, compute

$$
\phi_t=P_{t,1}f_1.
$$

Then verify

$$
\frac{\phi_t(gx)}{\phi_t(x)}
=
\mathbb E_{p^*}
\left[
\frac{f_1(gX_1)}{f_1(X_1)}
\mid X_t=x
\right]
$$

by explicit summation over all 90 terminal states.

Require relative error `<1e-10`.

This test is mandatory because it jointly checks the action convention, kernel equivariance, and matching-label direction.

## Test A2.12 — non-Dirac corrector identity

On the micro space use the exact four-atom source from Section 11.

Compute

$$
\widehat f_1(y)
=
\frac14\sum_c p_1^{\mathrm{base}}(y\mid s_c).
$$

For random `(y,g)` pairs, explicitly compute the posterior over source atoms and verify

$$
\mathbb E
\left[
\frac{p_1^{\mathrm{base}}(gY\mid X_0)}
{p_1^{\mathrm{base}}(Y\mid X_0)}
\middle|Y=y
\right]
=
\frac{\widehat f_1(gy)}{\widehat f_1(y)}.
$$

Require relative error `<1e-10`.

## Test A2.13 — exact controlled propagation

With zero controller output, exact propagation for any number of steps must:

- preserve total mass to `<1e-12`;
- preserve nonnegativity up to roundoff;
- remain entirely in the enumerated state space;
- agree with the corresponding discretized constant-rate reference behavior within the intended one-jump-per-step scheme.

Do not compare this discretized chain to `expm(Q)` as if they were identical; they are different numerical time integrations.

## Test A2.14 — GB1 loader

Before the GB1 training commands become available, require all assertions in Section 13.2 plus:

```text
HD=3 count = 27,436
encode(decode(x)) == x for all 27,436 sector states
decode(encode(sequence)) == sequence for all 160,000 sequences
```

---

# 15. Exact terminal-law evaluation

The two required Table 1 state spaces are enumerable. Therefore do not use sample histogram TV as the headline number for them. The optional 9.12M-state product benchmark is evaluated separately under Section 27 and must not use this exact-propagation path.

Copy the exact propagation pattern from `fixed_ising.py` into `fixed_support.py`.

For a current-state log multiplier matrix `a(x,e)`:

$$
u_e(x)
=
\frac{\gamma}{E}e^{a(x,e)}.
$$

Let

$$
R(x)=\sum_e u_e(x).
$$

For step size

$$
\Delta t=\frac1{\text{steps}},
$$

use

$$
p_{\mathrm{stay}}(x)
=e^{-R(x)\Delta t},
$$

and

$$
p_e(x)
=
\left(1-p_{\mathrm{stay}}(x)\right)
\frac{u_e(x)}{R(x)}.
$$

Propagate mass by sparse `index_add_` into `tgt` exactly as in `fixed_ising.py`.

For Dirac IASBS and DAM, initial law is a delta at `x0`.

For non-Dirac IASBS, initial law has mass exactly `0.25` on each of the four source states.

At the end compute

$$
\mathrm{TV}(p,\pi)
=
\frac12\sum_x|p(x)-\pi(x)|.
$$

Also save:

- KL `KL(p || pi)` with safe positive support;
- Hellinger distance;
- mean target energy;
- exact expected raw target score:
  - toy: not needed beyond energy;
  - GB1: expected GB1 fitness under samples/law;
- maximum probability state and its target probability;
- mass error `abs(p.sum()-1)`;
- constraint violations in forward samples.

Empirical TV and an iid empirical-TV floor may be reported as secondary diagnostics, but they are not the Table 1 value.

---

# 16. Training configurations

Do not perform a broad hyperparameter search. Start from the successful fixed-Ising regime.

## 16.1 Dirac IASBS — both toy and GB1

Use the paper configuration:

```text
steps       = 256
iters       = 3000
batch       = 2048
buffer      = 8
inner       = 40
mb          = 1024
hidden      = 512
lr          = 3e-4
loss        = poisson
seed        = 0
eval_every  = target-specific; toy 500, GB1 250
n_samples   = 20,000
gamma       = 10
tau         = 1
```

Use `eval_every = 500` for the toy and `eval_every = 250` for GB1. The toy exact propagation is substantially larger, so do not run it every 100 iterations. These values are fixed for the primary runs. The final exact evaluation is mandatory regardless.

Do not change any mathematical target/reference parameter because of training behavior.

## 16.2 Non-Dirac IASBS — both toy and GB1

Controller settings are identical to Dirac IASBS.

Corrector settings:

```text
corrector_hidden   = 256
corrector_actions  = 32
inner_h            = 40
lr_h               = 3e-4
```

Use a separate cosine LR scheduler for controller and corrector, following `fixed_ising.py`.

Replay source is always the four-atom mixture in Section 11.

## 16.3 DAM — both toy and GB1

The primary DAM experiment is source-matched to the **Dirac** IASBS bridge.

Use exactly:

```text
K           = 16
K_num       = 1
steps       = 256
iters       = 1200
inner       = 4
batch       = 512
mb          = 256
buffer      = 8
hidden      = 512
lr          = 1e-3
eval_every  = 50
n_samples   = 20,000
seed        = 0
gamma       = 10
tau         = 1
```

Keep the existing DAM numerical defaults:

```text
a_clamp     = adapter default 20
m_clip      = current LOG_M_CLIP
ESS minimum = 0
coef_cap    = 0
```

Do not introduce a fixed-support-specific DAM rescue configuration into the primary Table 1 run.

If the run becomes non-finite or finishes without a valid finite terminal law, save the failure diagnostics and report `div.`. A DAM failure is a result, not permission to alter IASBS or target settings.

If DAM is stable but misses TV `0.05`, report its finite TV. Do not relabel a merely inaccurate finite run as `div.`.

---

# 17. `FixedSupportAdapter` for DAM

Implement the same adapter contract as `IsingAdapter`.

```python
class FixedSupportAdapter:
    name = "fixed-support"
```

Required behavior:

### `rate_state(net,t,idx)`

Read the `(B,E)` IASBS-style controller log multipliers from `FS.net_mult_on_index`.

Use

$$
u_e=\frac{\gamma}{E}e^{a_e}.
$$

Model escape rate is their row sum. Base escape rate is the constant `gamma`.

### `sample_edge`

Reuse the categorical sampling logic in `IsingAdapter`.

### `apply_edge`

For the two required Table 1 spaces use

```python
space.tgt[idx,e]
```

for active samples. For `gb1-product2`, reconstruct the current encoded state, apply the selected on-the-fly legal edge from Section 27.4, and convert the result back with `state_to_index_torch`. The adapter interface remains unchanged.

### `a_on_edge`

Gather the selected legal-edge controller output.

### `base_rate_on_edge`

Return constant

```python
space.gamma / space.n_edges
```

for each batch member.

### `log_f1`

Return the **Dirac-source** terminal factor from Section 10.1.

### `source_state`

Return `space.i0` repeated across the batch.

### `violations`

Count

```python
(space.S[idx] != 0).sum(1) != space.k
```

No new logic is needed in `dam/core.py`.

`run_fixed_support` should mirror `run_ising`, replacing only:

- space constructor;
- controller constructor;
- bridge call;
- exact propagation/evaluation;
- benchmark diagnostics.

---

# 18. CLI

## 18.1 `structured_asbs/fixed_support.py`

Use subcommands:

```text
train
train-nondirac
build-cache
```

`build-cache` is required only for `--target gb1-product2`; it builds the exact reference-bridge class cache described in Section 27 before training.

and target choice:

```text
--target toy|gb1|gb1-product2
```

There is no need for a paper `exact` controller subcommand. Mathematical exactness is established by the deterministic tests and exact terminal-law propagation. Avoid implementing an expensive full optimal-controller oracle solely for this benchmark.

Common arguments:

```text
--target
--steps
--iters
--batch
--buffer
--inner
--mb
--hidden
--lr
--seed
--eval-every
--n-samples
--loss
--out
--ckpt-dir
--tag
```

Non-Dirac adds:

```text
--inner-h
--lr-h
--corrector-hidden
--corrector-actions
```

GB1 adds:

```text
--gb1-measured data/gb1/elife-16965-supp1-v4.xlsx
--gb1-imputed  data/gb1/elife-16965-supp2-v4.xlsx
```

Target-specific `n,k,r,tau,gamma` are fixed by `--target` for paper runs. They do not need public CLI overrides.

## 18.2 `dam/discrete.py`

Add command:

```text
fixed-support
```

and arguments:

```text
--target toy|gb1|gb1-product2
--gb1-measured ...
--gb1-imputed ...
```

When `cmd == "fixed-support"`, set benchmark defaults:

```text
tau   = 1.0
gamma = 10.0
hidden = 512
```

All DAM `K`, training, and stabilizer arguments remain the existing shared arguments.

---

# 19. Run order

Do not start GB1 training until every preceding stage passes.

## Stage 0 — dependency/data setup

```bash
pip install -r requirements.txt
```

Place the two original eLife workbooks at:

```text
data/gb1/elife-16965-supp1-v4.xlsx
data/gb1/elife-16965-supp2-v4.xlsx
```

## Stage 1 — all deterministic math tests

```bash
python structured_asbs/tests_math.py
```

All existing tests and all Appendix A.2 tests must pass.

## Stage 2 — toy Dirac IASBS

```bash
python structured_asbs/fixed_support.py train \
  --target toy \
  --steps 256 --iters 3000 --batch 2048 --buffer 8 \
  --inner 40 --mb 1024 --hidden 512 --lr 3e-4 \
  --loss poisson --seed 0 --eval-every 500 --n-samples 20000 \
  --tag fs_toy_v2_dirac \
  --out json/results_fs_toy_v2_dirac.json
```

## Stage 3 — toy non-Dirac IASBS

```bash
python structured_asbs/fixed_support.py train-nondirac \
  --target toy \
  --steps 256 --iters 3000 --batch 2048 --buffer 8 \
  --inner 40 --mb 1024 --hidden 512 --lr 3e-4 \
  --inner-h 40 --lr-h 3e-4 --corrector-hidden 256 \
  --corrector-actions 32 \
  --loss poisson --seed 0 --eval-every 500 --n-samples 20000 \
  --tag fs_toy_v2_nd \
  --out json/results_fs_toy_v2_nd.json
```

## Stage 4 — toy DAM

```bash
python -m dam.discrete fixed-support \
  --target toy --K 16 --K-num 1 \
  --steps 256 --iters 1200 --inner 4 \
  --batch 512 --mb 256 --buffer 8 --hidden 512 --lr 1e-3 \
  --eval-every 50 --n-samples 20000 --seed 0 \
  --tag dam_fs_toy_v2_K16 \
  --out json/results_dam_fs_toy_v2_K16.json
```

Only after Stages 2-4 have produced interpretable results proceed to GB1.

## Stage 5 — GB1 loader-only validation

Expose a cheap command or test path that constructs the GB1 space and exits after all data assertions. If implemented as a target constructor test, running `tests_math.py` after the two workbooks are present is sufficient.

The log must visibly print:

```text
measured = 149361
imputed  = 10639
merged   = 160000
HD3      = 27436
WT       = VDGV
```

## Stage 6 — GB1 Dirac IASBS

```bash
python structured_asbs/fixed_support.py train \
  --target gb1 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx \
  --steps 256 --iters 3000 --batch 2048 --buffer 8 \
  --inner 40 --mb 1024 --hidden 512 --lr 3e-4 \
  --loss poisson --seed 0 --eval-every 250 --n-samples 20000 \
  --tag fs_gb1_k3_dirac \
  --out json/results_fs_gb1_k3_dirac.json
```

## Stage 7 — GB1 non-Dirac IASBS

```bash
python structured_asbs/fixed_support.py train-nondirac \
  --target gb1 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx \
  --steps 256 --iters 3000 --batch 2048 --buffer 8 \
  --inner 40 --mb 1024 --hidden 512 --lr 3e-4 \
  --inner-h 40 --lr-h 3e-4 --corrector-hidden 256 \
  --corrector-actions 32 \
  --loss poisson --seed 0 --eval-every 250 --n-samples 20000 \
  --tag fs_gb1_k3_nd \
  --out json/results_fs_gb1_k3_nd.json
```

## Stage 8 — GB1 DAM

```bash
python -m dam.discrete fixed-support \
  --target gb1 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx \
  --K 16 --K-num 1 \
  --steps 256 --iters 1200 --inner 4 \
  --batch 512 --mb 256 --buffer 8 --hidden 512 --lr 1e-3 \
  --eval-every 50 --n-samples 20000 --seed 0 \
  --tag dam_fs_gb1_k3_K16 \
  --out json/results_dam_fs_gb1_k3_K16.json
```

---

# 20. Acceptance gates

## Structural gates — all runs

A run is invalid unless:

```text
constraint violations = 0
invalid state-rank lookups = 0
NaN/Inf rates = 0
NaN/Inf terminal law = 0
absolute terminal mass error < 1e-10
```

## IASBS success gate

Use the existing exact-law gate:

$$
\mathrm{TV}(p_1,\pi)\le0.05.
$$

This gate applies separately to Dirac and non-Dirac IASBS.

If a run misses `0.05`, report the miss. Do not hide it by replacing exact TV with empirical metrics.

## DAM reporting rule

If the DAM run remains finite, report its exact-law TV whether or not it is below `0.05`.

Use `div.` only if the completed run does not yield a finite valid terminal comparison because of numerical/optimization divergence.

Save the existing DAM diagnostics:

- adjoint rollouts;
- terminal `f1` evaluations;
- simulated CTMC jumps;
- ESS statistics;
- wall time;
- loss history;
- any clipping counts already tracked by `dam/discrete.py`.

---

# 21. Result JSON and checkpoint requirements

Every IASBS result JSON must include at least:

```text
method
target
config
n, k, r, M
n_edges
n_actions
alpha, beta, gamma, tau
source type
history
final_exact_TV
KL
Hellinger
mean_energy
violations
mass_error
wall_seconds
```

Non-Dirac additionally includes:

```text
source_states
source_weights
corrector_log_ratio_RMSE
corrector_log_ratio_MAE
corrector_log_ratio_max_abs
```

Toy additionally includes:

```text
toy_rng_seed
H
J
```

or an exact serialized copy/hash of those arrays in the checkpoint.

GB1 additionally includes:

```text
GB1 paper DOI
supp1 DOI
supp2 DOI
measured_count = 149361
imputed_count = 10639
merged_count = 160000
sector_count = 27436
WT = VDGV
fitness_epsilon = 1e-4
amino_acid_alphabet = ACDEFGHIKLMNPQRSTVWY
mean_fitness under exact target
mean_fitness under learned terminal law
```

Checkpoint every full run using the repository's existing `C.save_ckpt` convention. Save enough metadata to remeasure exact TV without retraining.

---

# 22. Table 1 update

Do not remove the existing Table 1 rows.

Add exactly these two rows after the current enumerable discrete rows:

| benchmark | states | IASBS | IASBS (non-Dirac) | DAM |
|---|---:|---:|---:|---:|
| Fixed support toy | 81,081 | exact TV | exact TV | exact TV or `div.` |
| GB1, `k=3` | 27,436 | exact TV | exact TV | exact TV or `div.` |

All numeric entries in these two rows are exact-law TV values obtained by deterministic propagation of the learned discretized controlled chain.

The table caption/note should state:

- fixed-support toy uses `X^(4)_{14,4}` and a heterogeneous nonsymmetric Potts-ring energy;
- GB1 uses the Wu et al. complete measured+author-imputed four-site landscape and the exact Hamming-distance-3 sector;
- DAM is source-matched to the Dirac IASBS bridge;
- non-Dirac IASBS uses the fixed four-atom source mixture;
- zero constraint violations are checked separately and are not encoded by the TV value.

Do not describe the GB1 experiment as using 27,436 *experimentally measured* variants. Some members of the complete sector may use the authors' published imputed values. Describe it as the **Wu et al. complete measured+author-imputed landscape**.

---

# 23. What is allowed to change after a failed run

The mathematical implementation is frozen after all deterministic tests pass.

If training fails, use this order only:

1. inspect the saved loss/rate/TV diagnostics for an implementation bug;
2. if there is a bug, fix it and rerun the deterministic tests;
3. if there is no bug, report the result;
4. do not change `n,k,r`, target energy, GB1 epsilon, source mixture, reference rates, or terminal label to make the number look better.

Do not alter the primary IASBS configuration after seeing the toy or GB1 result. Any later capacity or optimization sweep is a separate ablation and must not replace the predeclared Table 1 run without being explicitly identified as such.

For DAM, do not silently tune stabilizers. The primary K=16 run stands as the Table 1 baseline. Any K or stabilizer sweep is a separate diagnostic/cost ablation.

---

# 24. Definition of done

This task is complete only when all of the following are true:

- [ ] all old deterministic tests still pass;
- [ ] all Appendix A.2 deterministic tests pass;
- [ ] toy state count is exactly 81,081;
- [ ] toy Dirac IASBS run completed and exact TV saved;
- [ ] toy non-Dirac IASBS run completed and exact TV + corrector error saved;
- [ ] toy DAM run completed or validly recorded as divergent;
- [ ] GB1 loader reconstructs exactly 160,000 variants from 149,361 measured + 10,639 author-imputed values;
- [ ] GB1 Hamming-distance-3 sector contains exactly 27,436 states;
- [ ] GB1 Dirac IASBS run completed and exact TV saved;
- [ ] GB1 non-Dirac IASBS run completed and exact TV + corrector error saved;
- [ ] GB1 DAM run completed or validly recorded as divergent;
- [ ] every forward sample in every method remains in the exact fixed-support sector;
- [ ] Table 1 uses exact-law TV for both new rows;
- [ ] README contains the exact final commands and paths;
- [ ] no occupation result/code was changed as part of this experiment task.

If the optional GB1-product2 benchmark is included, all of the following are additionally required:

- [ ] product2 state count is exactly 9,122,470;
- [ ] the exact product target sampler passes the sector-count and block-split tests;
- [ ] all 15 memory-mapped bridge caches pass shape, coverage, and Chapman--Kolmogorov checks;
- [ ] product2 Dirac IASBS run completed and sample metrics saved;
- [ ] product2 non-Dirac IASBS run completed and sample metrics + corrector diagnostics saved;
- [ ] product2 DAM run completed or validly recorded as divergent;
- [ ] all product2 samples have exactly four active sites;
- [ ] the product2 result is reported separately from Table 1 and explicitly labeled synthetic-product scaling.

---

# 25. One-paragraph intended paper interpretation

The toy row is the controlled mathematical validation of Appendix A.2: the target deliberately breaks the reference's permutation/label symmetry, yet the nonbinary-Johnson orbit kernel makes the exact terminal matching labels tractable, and the learned controlled law can be compared to the exact 81,081-state target in total variation. GB1 then changes only the target-energy/data adapter: the same `X^(r)_{n,k}` machinery samples the experimentally motivated four-site protein-fitness landscape subject to an exact three-mutation budget. Together the two rows show that Appendix A.2 is not merely an abstract group-action construction: it produces a functioning exact-constraint sampler on both a fully auditable synthetic law and a published experimentally profiled biological landscape completed with the original authors' imputed missing values. If the optional GB1-product2 benchmark is run, it adds a separate scalability result at 9,122,470 constrained states using exact target sampling rather than full-law propagation; it does not replace either exact-TV validation row.

---

# 26. Data references

Wu, N. C., Dai, L., Olson, C. A., Lloyd-Smith, J. O., & Sun, R. (2016). **Adaptation in protein fitness landscapes is facilitated by indirect paths.** *eLife*, 5, e16965. DOI: `10.7554/eLife.16965`.

Supplementary file 1: **The fitness of each profiled variant.** DOI: `10.7554/eLife.16965.024`.

Supplementary file 2: **Imputed fitness values for missing variants.** DOI: `10.7554/eLife.16965.025`.

---

# 27. Optional headline scaling benchmark: two-block GB1 product

This section is **optional**. Do not begin it until both required Table 1 benchmarks have completed for
Dirac IASBS, non-Dirac IASBS, and DAM. Nothing in this section is needed to validate Appendix A.2 or
to keep the two exact-TV Table 1 rows.

Its purpose is to show that the same fixed-support construction can be trained and sampled without
exact propagation when the constrained state space is already in the multi-million-state regime.

## 27.1 Interpretation and naming

Construct two independent copies of the four-site GB1 landscape. This is a **synthetic product target
built from empirical GB1 fitness data**. It is not an experimentally measured eight-site protein
landscape and must never be described as one.

Use the paper name:

```text
GB1-product2, k=4
```

Configuration:

```text
name   = gb1_product2_k4
n      = 8
k      = 4
r      = 20
tau    = 1.0
gamma  = 10.0
```

The first four coordinates are one copy of GB1 sites `(39,40,41,54)` and the second four coordinates
are a second copy in the same order.

The exact constrained state count is

$$
\left|\mathcal X^{(20)}_{8,4}\right|
=
\binom84 19^4
=
9,122,470.
$$

Both Appendix A.2 move families remain active because `0 < k < n`.

The number of legal neighbors from every state is

$$
E_{\mathrm{lab}}=4(20-2)=72,
$$

$$
E_{\mathrm{sup}}=4(8-4)(20-1)=304,
$$

and therefore

$$
E=376.
$$

With `gamma=10`, use

$$
\alpha
=10\frac{72}{376}
=\frac{90}{47}
\approx1.914893617,
$$

$$
\beta
=10\frac{304}{376}
=\frac{380}{47}
\approx8.085106383,
$$

and every individual legal edge has rate

$$
\frac{10}{376}
=\frac{5}{188}
\approx0.02659574468.
$$

The orbit count is `C=15`.

The global action count is

$$
A_{\mathrm{lab}}
=8\binom{19}{2}
=1368,
$$

$$
A_{\mathrm{sup}}
=8\cdot7\left(1+\binom{19}{2}\right)
=9632,
$$

so

$$
A=11,000.
$$

The selected-action corrector architecture in Section 11 is therefore mandatory; do not replace it
with a dense `11,000`-output corrector head.

## 27.2 Product target

Let the complete four-site GB1 weight from Section 13 be

$$
w(z)=F(z)+10^{-4},
$$

where `F` is the measured value when available and the original authors' imputed value only for the
10,639 missing variants.

For an eight-site state `x`, split it into two four-site blocks:

$$
x=(x^{(1)},x^{(2)}).
$$

Decode each block using exactly the site-specific GB1 encoding from Section 13.3 and define

$$
E_{\mathrm{prod}}(x)
=
-\log w(x^{(1)})
-\log w(x^{(2)}).
$$

The constrained target is

$$
\pi_{\mathrm{prod}}(x)
\propto
w(x^{(1)})w(x^{(2)})
\mathbf 1\{\|x\|_0=4\}.
$$

No additional dataset is downloaded for this benchmark. It reuses exactly the same two eLife
supplementary workbooks as Section 13.

## 27.3 Exact target sampler without enumerating the 9.12M target law

For one four-site GB1 block define sectors

$$
\mathcal G_h
=
\{z:\operatorname{HD}(z,\mathrm{WT})=h\},
\qquad h=0,1,2,3,4,
$$

with exact sector partition functions

$$
Z_h
=
\sum_{z\in\mathcal G_h} w(z).
$$

The sector sizes must be asserted:

```text
h=0:       1
h=1:      76
h=2:   2,166
h=3:  27,436
h=4: 130,321
sum: 160,000
```

Conditioned on four total mutations across the two blocks, the exact block-mutation split is

$$
\Pr(H_1=h,H_2=4-h)
=
\frac{Z_hZ_{4-h}}
{\sum_{a=0}^4 Z_aZ_{4-a}},
\qquad h=0,\ldots,4.
$$

Precompute the five normalized categorical distributions

$$
\Pr(Z=z\mid H=h)
=
\frac{w(z)}{Z_h},
\qquad z\in\mathcal G_h.
$$

An exact iid target draw is therefore:

1. sample `h1 in {0,...,4}` from the analytic split distribution above;
2. set `h2 = 4-h1`;
3. independently sample block 1 from the weighted GB1 sector `h1`;
4. independently sample block 2 from the weighted GB1 sector `h2`;
5. concatenate the two encoded blocks.

Implement:

```python
def sample_gb1_product2_exact(self, batch: int, device=None):
    ...
```

This exact target sampler is evaluation only. No target samples may enter IASBS, non-Dirac IASBS,
or DAM training.

Mandatory checks:

- every returned state has exactly four active positions;
- empirical `H1` frequencies from 1,000,000 iid draws agree with the analytic five-bin distribution
  within ordinary multinomial sampling error;
- no sequence outside the complete 160,000-state four-site GB1 table is ever queried.

## 27.4 State representation at 9.12M states

Do **not** allocate any of the following:

```text
20**8 dense LUT
M x M transition matrix
M x 376 tgt table
M x 376 edge_action_id table
M-dimensional propagated terminal probability at every training evaluation
```

Use the exact support-rank/label-rank map in Section 2.2.

For controlled trajectories, keep the current state either as:

```text
(B,8) uint8/int64 encoded state
```

or as its integer fixed-support rank plus `index_to_state` conversion. Do not introduce an approximate
hash map.

Generate legal edges on the fly from a batch state:

- collect its four active positions in ascending order;
- collect its four inactive positions in ascending order;
- generate `4*18=72` relabel descriptors;
- generate `4*4*19=304` support descriptors;
- concatenate in the same relabel-first/support-second ordering as Section 3.

Implement scalable helpers:

```python
legal_edges_on_state(x)          # descriptors for a batch
apply_selected_edge(x, edge)     # new encoded state
local_action_id(x, edge)         # global action ID from Section 7
```

`FixedSupportAdapter` must use this path automatically when `space.precompute_tgt == False`.
No change to `dam/core.py` is permitted.

## 27.5 Exact reference-bridge cache for IASBS training

IASBS still requires exact reference-bridge samples. Do not replace the A.2 bridge with an
uncontrolled forward sample, rejection sampler, approximate bridge, or DAM rollout.

The orbit identity still reduces the bridge to pairs `(a,b)`, but storing every endpoint-specific state
list naively would duplicate too much Python-object overhead. Build one compact on-disk cache per
canonical endpoint orbit.

Cache directory:

```text
data/cache/fixed_support/gb1_product2_k4/
```

There are `C=15` canonical endpoint orbits. For each endpoint orbit `c`, create:

```text
perm_cXX.npy      int32, shape (9_122_470,)
counts_cXX.npy    int64, shape (C*C,)
offsets_cXX.npy   int64, shape (C*C + 1,)
```

`perm_cXX` groups every fixed-support state index by

$$
\mathrm{key}=aC+b,
$$

where

$$
a=\sigma(x_0,z),
\qquad
b=\sigma(z,y_c).
$$

The 15 permutation files together occupy approximately

$$
15\times9,122,470\times4
\approx547\ \text{MB}
$$

decimal, before filesystem overhead. This is acceptable; do not store Python lists of millions of
integers.

### Cache construction algorithm

For each canonical endpoint orbit `c`:

1. iterate fixed-support state indices in chunks, e.g. `500_000`;
2. reconstruct states with `index_to_state`;
3. compute the uint8 class key `a*C+b`;
4. first pass: accumulate `np.bincount(key, minlength=C*C)`;
5. compute prefix-sum offsets;
6. second pass: recompute keys, stably group each chunk by key, and write its global int32 state
   indices into the correct segment of an `open_memmap` permutation file using per-key cursors;
7. verify every cursor reaches its final offset and the permutation contains exactly `0,...,M-1`
   once.

Do not use one global `argsort` on a Python object array.

Build with:

```bash
python structured_asbs/fixed_support.py build-cache \
  --target gb1-product2 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx
```

The command must be restart-safe: if a cache file exists, validate its shape/dtype/checksum metadata
before reusing it. Do not silently reuse a partial file.

### Bridge draw from the cache

For endpoint `y` at time `t`:

1. compute `c=sigma(x0,y)`;
2. construct `h_y` on demand as in Section 8.3;
3. for each class key `(a,b)`, compute

$$
W_{a,b}
=
\mathrm{count}_{c,a,b}
\kappa_t(a)
\kappa_{1-t}(b);
$$

4. sample a class key from normalized `W`;
5. sample a uniform integer position inside that class's offset interval;
6. read canonical state index `z_idx = perm_cXX[position]`;
7. reconstruct `z_c=index_to_state(z_idx)`;
8. return `h_y(z_c)`.

This is the same exact class sampler as Section 8; only the class-member storage is memory-mapped.

Before training, verify for every endpoint orbit and several times `t` the class-count
Chapman--Kolmogorov identity from Test A2.9 to tolerance `1e-10`.

## 27.6 Controller and corrector

Reuse the same classes as the required benchmarks. For `n=8,r=20`, the controller output dimension is

$$
8(19)+8^2(19)=1368.
$$

Only 376 outputs are legal at any state and are gathered by the on-the-fly legal-edge descriptors.

The non-Dirac branch uses the same four-atom source construction in Section 11, interpreted at
`n=8,k=4,r=20`. Do not select source atoms from GB1 fitness.

The exact four-atom corrector diagnostic remains available because

$$
\widehat f_1(y)
\propto
\frac14\sum_{c=0}^3
\kappa_1(\sigma(s_c,y)).
$$

For the 9.12M benchmark, evaluate corrector error on 200,000 uniformly sampled `(y,g)` pairs rather
than exhaustively.

## 27.7 Training configurations

This benchmark is a scaling experiment, so use one predeclared configuration and do not tune each
method separately after seeing results.

### Dirac IASBS

```text
steps       = 256
iters       = 3000
batch       = 1024
buffer      = 8
inner       = 40
mb          = 512
hidden      = 512
lr          = 3e-4
loss        = poisson
seed        = 0
gamma       = 10
tau         = 1
```

### Non-Dirac IASBS

Use the same controller configuration plus:

```text
corrector_hidden   = 256
corrector_actions  = 32
inner_h            = 40
lr_h               = 3e-4
```

### DAM

Use:

```text
K           = 16
K_num       = 1
steps       = 256
iters       = 1200
inner       = 4
batch       = 256
mb          = 128
buffer      = 8
hidden      = 512
lr          = 1e-3
seed        = 0
gamma       = 10
tau         = 1
```

The smaller DAM batch only controls memory from its nested rollouts and larger 376-edge state; it is
not a change to `K`, the bridge, target, or reference process.

If this optional benchmark is included, attempt all three methods. Apply the same reporting rule as
Section 20: a finite DAM run gets its finite metrics; `div.` is reserved for a run that cannot produce a
valid finite terminal sample set. Do not use `n/r` merely because DAM is slow after deciding to include
this benchmark.

## 27.8 Training commands

### Dirac IASBS

```bash
python structured_asbs/fixed_support.py train \
  --target gb1-product2 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx \
  --steps 256 --iters 3000 --batch 1024 --buffer 8 \
  --inner 40 --mb 512 --hidden 512 --lr 3e-4 \
  --loss poisson --seed 0 --n-samples 100000 \
  --tag fs_gb1_product2_k4_dirac \
  --out json/results_fs_gb1_product2_k4_dirac.json
```

### Non-Dirac IASBS

```bash
python structured_asbs/fixed_support.py train-nondirac \
  --target gb1-product2 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx \
  --steps 256 --iters 3000 --batch 1024 --buffer 8 \
  --inner 40 --mb 512 --hidden 512 --lr 3e-4 \
  --inner-h 40 --lr-h 3e-4 --corrector-hidden 256 \
  --corrector-actions 32 \
  --loss poisson --seed 0 --n-samples 100000 \
  --tag fs_gb1_product2_k4_nd \
  --out json/results_fs_gb1_product2_k4_nd.json
```

### DAM

```bash
python -m dam.discrete fixed-support \
  --target gb1-product2 \
  --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
  --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx \
  --K 16 --K-num 1 \
  --steps 256 --iters 1200 --inner 4 \
  --batch 256 --mb 128 --buffer 8 --hidden 512 --lr 1e-3 \
  --n-samples 100000 --seed 0 \
  --tag dam_fs_gb1_product2_k4_K16 \
  --out json/results_dam_fs_gb1_product2_k4_K16.json
```

Do not pass an exact-propagation/eval flag for this target. `--target gb1-product2` must automatically
select sample-based scaling evaluation.

## 27.9 Scaling evaluation

Generate exactly `100,000` terminal samples from each learned sampler and two independent
`100,000`-sample iid target sets from Section 27.3.

Report the following metrics:

1. **Energy-law KS** between sampler and exact target samples for

$$
E_{\mathrm{prod}}.
$$

2. **Block mutation-split TV**. For a terminal state define

$$
H_1=\|x^{(1)}\|_0.
$$

Compare its five-bin empirical histogram to the analytic exact distribution

$$
\Pr(H_1=h\mid H_1+H_2=4).
$$

3. **Mean site activity error**:

$$
\frac18\sum_{i=1}^8
\left|
\Pr_{\mathrm{sampler}}(x_i\ne0)
-
\Pr_{\mathrm{target}}(x_i\ne0)
\right|.
$$

Use the exact iid target sample to estimate the target term.

4. **Mean site amino-acid marginal TV**. For each site compare the 20-label empirical categorical
marginal and average the eight TVs.

5. **Pairwise activity-correlation MAE** over the 28 unordered site pairs, using indicators
`1[x_i != 0]`.

6. **Mean product energy** and its difference from the exact iid target estimate.

7. **Constraint violations**, which must be zero.

For metrics estimated using target samples, report an iid floor obtained by comparing the two
independent exact target sets with the same sample size. The analytic block-split TV has exact target
probabilities and therefore needs no sampled target approximation.

Do **not** report an empirical histogram TV over all 9,122,470 states; with 100,000 samples it is not a
meaningful approximation to full-law TV.

## 27.10 Scaling result table

Keep this separate from Table 1. Suggested table:

| method | states | KS(E) | block-split TV | site-active MAE | AA-marginal TV | pair-corr MAE | violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid floor | 9,122,470 | value | -- | value | value | value | 0 |
| IASBS | 9,122,470 | value | value | value | value | value | 0 |
| IASBS (non-Dirac) | 9,122,470 | value | value | value | value | value | 0 |
| DAM | 9,122,470 | value or `div.` | value or `--` | value or `--` | value or `--` | value or `--` | 0 or `--` |

The caption must state that the target is a synthetic product of two copies of the complete
measured+author-imputed four-site GB1 landscape, conditioned on exactly four total mutations.

## 27.11 Scaling acceptance gates

This optional benchmark has no predeclared accuracy threshold because it is not an exact-law
validation row. It is valid only if:

```text
constraint violations = 0
invalid state-rank lookups = 0
NaN/Inf rates = 0
all reported sample metrics are finite
exact-target sampler tests pass
bridge-cache CK tests pass
```

Do not replace a poor scaling result with a smaller state space after observing it. If the product2
benchmark is attempted and performs poorly, report that result or omit the optional scaling section
entirely with no claim based on it.

## 27.12 Additional result metadata

Every product2 result JSON must include:

```text
synthetic_product = true
base_dataset = Wu et al. GB1 complete landscape
n_blocks = 2
n = 8
k = 4
r = 20
M = 9122470
n_edges = 376
n_actions = 11000
exact_target_sampler = sector-DP/product sampler
n_eval_samples = 100000
block_partition_functions_Z0_to_Z4
analytic_block_split_probabilities
bridge_cache_directory
bridge_cache_file_metadata/checksums
```

This metadata prevents the product benchmark from later being mistaken for a distinct experimental
protein dataset.

---

# 28. External-data download checklist

GB1 is the only experiment in this specification that requires an external dataset download. The toy
uses generated coefficients and the product2 benchmark reuses the same GB1 files.

Download exactly the two original eLife supplementary workbooks from the article's **Figures and
data / Additional files** page:

- Supplementary file 1, *The fitness of each profiled variant*: 149,361 measured variants,
  DOI `10.7554/eLife.16965.024`, expected filename `elife-16965-supp1-v4.xlsx`;
- Supplementary file 2, *Imputed fitness values for missing variants*: 10,639 author-imputed variants,
  DOI `10.7554/eLife.16965.025`, expected filename `elife-16965-supp2-v4.xlsx`.

Official article/data page:

```text
https://elifesciences.org/articles/16965/figures
```

DOI links:

```text
https://doi.org/10.7554/eLife.16965.024
https://doi.org/10.7554/eLife.16965.025
```

Place them at:

```text
data/gb1/elife-16965-supp1-v4.xlsx
data/gb1/elife-16965-supp2-v4.xlsx
```

Do not require network access from training scripts. Data download is a one-time setup step. The
loader must verify the counts and merge policy in Section 13 every time the GB1 target is constructed.
