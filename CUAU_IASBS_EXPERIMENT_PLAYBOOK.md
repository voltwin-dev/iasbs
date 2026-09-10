# CuAu IASBS Experiment Playbook

**Purpose.** This document is a self-contained implementation and execution plan for adding a realistic, large-scale, fixed-composition Cu-Au alloy benchmark to the supplied IASBS repository with minimal disturbance to the existing code. It is written so that the experiment can be implemented from this document alone, while preserving the mathematical construction already used by `structured_asbs/fixed_ising.py`.

**Reliability policy.** No written plan can literally guarantee zero software errors across an external repository, CE-library versions, GPU/CPU environments, and unpublished experiment outputs. This playbook is therefore designed to make silent errors difficult: energy parity, state invariants, bridge-law checks, terminal-label equivalence, detailed balance, reference convergence, Hamiltonian symmetry, cost accounting, and exact-small-law checks are **hard stop/go gates**. Do not proceed past a failed gate.

**Repository inspected.** The instructions below refer to the supplied repository layout containing at least:

- `common.py`
- `structured_asbs/fixed_ising.py`
- `dam/core.py`
- `dam/discrete.py`
- `requirements.txt`

The relevant current code positions in the supplied snapshot are:

- `structured_asbs/fixed_ising.py:71-154`: enumerated `FixedIsingSpace`.
- `structured_asbs/fixed_ising.py:156-219`: exact Johnson bridge occupancy-class tables.
- `structured_asbs/fixed_ising.py:301-325`: frozen-rate controlled simulation.
- `structured_asbs/fixed_ising.py:365-389`: `SwapController`.
- `structured_asbs/fixed_ising.py:428-455`: exact Dirac IASBS terminal label.
- `structured_asbs/fixed_ising.py:488-530`: exact reference-bridge sampler.
- `structured_asbs/fixed_ising.py:619-748`: Dirac training loop.
- `common.py:146-170`: Johnson orbit generator/kernel.
- `common.py:205-215`: binary AS terminal-ratio identity.
- `dam/core.py`: generic finite-state/tensor-state DAM machinery.
- `dam/discrete.py:52-113`: the existing fixed-composition Ising DAM adapter.

The paper's fixed-composition construction is already exactly the one required here: binary states with a conserved number of active sites, uniform legal swaps, transposition readouts, and the Johnson-orbit reference kernel. Therefore **CuAu should be implemented as an application of the existing fixed-composition triple, not as a new IASBS construction.**

---

## 1. Executive decision: what to implement

Implement **one new experiment module** and avoid refactoring the existing working Ising code.

Recommended new files:

```text
structured_asbs/cuau.py              # IASBS CuAu: energy wrapper, scalable state ops, training, eval
structured_asbs/cuau_reference.py    # canonical MCMC / PT runner + reference export
structured_asbs/cuau_metrics.py      # optional; may be folded into cuau.py to minimize files
tests/test_cuau.py                   # correctness gates
data/cuau/PROVENANCE.md              # exact external source/commit/hash record
requirements-cuau.txt                # separate environment, do not disturb original requirements.txt
```

Recommended modifications to existing source:

```text
common.py                            # ideally 0 LOC changed
structured_asbs/fixed_ising.py       # 0 LOC changed
dam/core.py                          # 0 LOC changed
dam/discrete.py                      # only add a CuAu-small adapter/CLI branch if desired
```

The highest-value minimal implementation is:

1. Reuse `fixed_ising.SwapController` unchanged.
2. Reuse `common.binary_orbit_kernel` unchanged.
3. Copy the small occupancy-class bridge-table construction into `cuau.py` and return direct binary states instead of enumeration indices.
4. Carry states as `(B, N)` binary tensors for `N=64/128`; never enumerate the state space.
5. Replace the current all-transpositions terminal-label calculation with **uniform legal-edge minibatching**. This preserves the same expected training objective while changing target-energy work from `O(N^2)` labels per state to `O(1)` labels per state.
6. Use a published DFT-fitted CuAu cluster-expansion energy from the MetaDNS CuAu benchmark.
7. Use canonical swap MCMC as the direct large-system baseline and canonical parallel tempering as the high-quality reference.
8. Use DAM only on the exact small canonical CuAu problem unless large-system DAM proves unexpectedly cheap.

Do **not** initially add a new Transformer/GNN, non-Dirac corrector, special lattice-equivariant architecture, free-energy estimator, or new IASBS theorem. Those additions increase reviewer surface area and LOC without strengthening the main claim as efficiently.

---

## 2. Scientific claim the experiment should support

The experiment is successful if it supports the following statement:

> On a realistic DFT-derived CuAu alloy energy with exactly fixed equiatomic composition, IASBS scales the same fixed-composition construction validated on enumerable systems to a state space of order `10^18` (and optionally `10^37`), recovers the canonical energy/order-parameter law and symmetry-related ordered basins, and does so more effectively than local canonical swap MCMC at matched target-energy cost.

This is stronger than simply showing a low energy. It requires evidence for **distributional correctness, mode coverage, exact constraint preservation, and computational efficiency**.

The experiment should explicitly avoid claiming that published semi-grand-canonical MetaDNS/SEGAL/MAM numbers are direct target-matched baselines. They are prior-work context unless rerun/conditioned carefully.

---

## 3. Target distribution and representation

### 3.1 State space

Encode Au as `1`, Cu as `0`.

For a system with `N` lattice sites and equiatomic composition:

```math
\Omega_{N,N/2}
= \left\{x\in\{0,1\}^{N}: \sum_i x_i=N/2\right\}.
```

State counts:

| system | N | k | canonical state count |
|---|---:|---:|---:|
| CuAu-S | 16 | 8 | `C(16,8) = 12,870` |
| CuAu-M | 64 | 32 | `C(64,32) = 1,832,624,140,942,590,534` |
| CuAu-L | 128 | 64 | `C(128,64) = 23,951,146,041,928,082,866,135,587,776,380,551,750` |

### 3.2 Canonical target

Use

```math
\pi_T(x) \propto \exp\left[-\frac{E_{\rm CE}(x)}{k_B T}\right],
\qquad x\in\Omega_{N,N/2},
```

with

```text
k_B = 8.617333262145e-5 eV/K
```

and therefore the IASBS `tau` is

```python
tau = KB_EV_PER_K * temperature_K
```

**Energy-unit gate:** the energy wrapper must return **total configurational energy in eV for the entire supercell**. If the upstream backend returns eV/atom, multiply by `N` before using it in the Boltzmann exponent. Never infer the unit from magnitude; verify it against the upstream CuAu code on an identical configuration.

At fixed composition, a semi-grand chemical-potential term is constant across the sector, so it cancels from all target ratios. The CuAu IASBS target should therefore use only `E_config` and not a concentration-changing chemical-potential term.

---

## 4. External CuAu source: use the established MetaDNS benchmark energy

### 4.1 Required external source

Use the official MetaDNS repository:

```text
https://github.com/xiaochendu/metadns
```

Verified public CuAu assets/commands in that repository include:

```text
data/cuau/cuau_fcc_4x4x4_supercell.vasp
data/cuau/CI_params_ECI_CuAu_Final_Submission.json
```

and the official `4x4x4`, `500 K` training command points to exactly those two files. The repository describes the CuAu energy as a cluster-expansion model using CLEASE/iCET. It also provides a CuAu reproduction notebook and an optional Zenodo archive with checkpoints and benchmark samples.

MetaDNS source/evidence:

- Repository/README: https://github.com/xiaochendu/metadns
- ICML 2026 MetaDNS citation in repo: https://openreview.net/forum?id=OY7Qe2ZSx9
- Zenodo benchmark archive: https://doi.org/10.5281/zenodo.20301979

### 4.2 Download only what is required

From the root of the IASBS repository:

```bash
mkdir -p external data/cuau

git clone https://github.com/xiaochendu/metadns.git external/metadns

git -C external/metadns rev-parse HEAD | tee data/cuau/METADNS_COMMIT.txt

cp external/metadns/data/cuau/cuau_fcc_4x4x4_supercell.vasp data/cuau/
cp external/metadns/data/cuau/CI_params_ECI_CuAu_Final_Submission.json data/cuau/

sha256sum \
  data/cuau/cuau_fcc_4x4x4_supercell.vasp \
  data/cuau/CI_params_ECI_CuAu_Final_Submission.json \
  | tee data/cuau/SHA256SUMS.txt
```

Do not download raw DFT data and do not run VASP. The fitted CE is the energy oracle.

The Zenodo `data_cuau.tar.gz` is **optional**. Download it only if doing the optional direct MetaDNS-conditioned comparison or reproducing their published curves.

### 4.3 Do not copy upstream API names blindly

The public repository documents an `energy_cuau.py` / `AuCuAlloyModel`, but upstream APIs can change. **Do not guess the constructor or atom-symbol convention from this document.** Immediately after pinning the MetaDNS commit, inspect the exact checked-out implementation and its caller:

```bash
sed -n '1,260p' external/metadns/energy_cuau.py
grep -n "AuCuAlloyModel\|eci_file\|input_file\|energy" \
  external/metadns/train_cuau.py | head -80
```

If importable, this gives an even more explicit API dump:

```bash
python - <<'PY'
import inspect, sys
sys.path.insert(0, 'external/metadns')
import energy_cuau
print(inspect.signature(energy_cuau.AuCuAlloyModel))
print(inspect.getsource(energy_cuau.AuCuAlloyModel))
PY
```

Then put one stable adapter in IASBS with this interface:

```python
class CuAuEnergy:
    """Only adapter allowed to depend on MetaDNS/CLEASE/ASE details."""

    def energy_np(self, x: np.ndarray) -> np.ndarray:
        """x shape (..., N), IASBS convention 1=Au, 0=Cu; total eV."""
        raise NotImplementedError  # replace from the pinned upstream implementation

    def energy_torch(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        y = self.energy_np(x.detach().to('cpu').numpy())
        return torch.as_tensor(y, dtype=torch.float64, device=dev)
```

`NotImplementedError` is intentional in this playbook: the constructor/call signature belongs to the **pinned external commit**, and inventing it here would be less reproducible than copying the exact three-to-ten upstream lines after inspection. The wrapper is the only location allowed to know CLEASE/iCET/ASE object details.

**Species-mapping gate:** IASBS uses `1=Au, 0=Cu`, but the upstream code may internally encode species in the opposite order or via atomic symbols. The adapter must map explicitly rather than relying on integer coincidence. The 100-state energy-parity corpus below is the authority on this mapping.

### 4.4 Mandatory energy parity test

Before training anything:

1. In the exact upstream MetaDNS environment, generate/save at least 100 deterministic equiatomic binary configurations and their upstream `E_config` values.
2. Load those configurations in the IASBS CuAu environment.
3. Evaluate them through `CuAuEnergy`.
4. Require:

```python
max_abs_error < 1e-10  # eV if both paths use the same deterministic CE arithmetic
```

If library-version floating arithmetic prevents `1e-10`, inspect first; only relax after confirming the discrepancy is roundoff rather than a configuration-ordering or unit bug. A practical relaxed ceiling is `1e-8 eV`, still vastly below `kBT` at 500 K.

Also test at least 100 single Au<->Cu swaps and compare `Delta E` between the two implementations.

**Stop the experiment if parity fails.** Do not tune IASBS around an unverified energy mapping.

---

## 5. Environment strategy: do not break the original reproducibility pins

The supplied IASBS `requirements.txt` pins NumPy `2.4.6`. MetaDNS documents CLEASE `1.1.0`; CLEASE 1.1.0 explicitly pins `numpy < 2`. Therefore do **not** append MetaDNS's old CLEASE pins to the repository's existing requirements file.

CLEASE 1.2.0 removed the `numpy < 2` restriction, while `mchammer-pt 0.27.1` requires `icet >= 3.2`. This makes a separate environment the safest path.

Recommended policy:

- Keep original `requirements.txt` unchanged.
- Create `requirements-cuau.txt` or a Conda environment dedicated to CuAu.
- Record exact package versions in every result JSON.
- Run the energy parity gate above so newer CE-library versions cannot silently alter the Hamiltonian.

A reasonable current Python 3.11 reference stack is:

```text
# start from the IASBS torch version
python == 3.11
torch == 2.5.1

# current CE/reference stack; pin after successful parity test
ase
clease >= 1.2.0
icet >= 3.2
mchammer-pt == 0.27.1
```

If the MetaDNS CE loader only works unchanged under its original CLEASE 1.1 environment, use **two environments**:

- `iasbs-cuau`: IASBS + exact energy wrapper.
- `cuau-reference`: current icet/mchammer-pt reference sampling.

Then require the same 100-configuration energy parity between them.

External package evidence:

- CLEASE 1.2 release notes (`numpy < 2` restriction removed): https://clease.readthedocs.io/en/stable/releasenotes.html
- icet canonical ensemble docs: https://icet.materialsmodeling.org/dev/moduleref/ensembles.html
- `mchammer-pt 0.27.1`: https://pypi.org/project/mchammer-pt/0.27.1/

---

## 6. Why the existing IASBS mathematics is reused unchanged

The current fixed-composition Ising branch already implements the exact ingredients needed by CuAu:

1. Legal state transition: swap an occupied site `i` with empty site `j`.
2. Number of ordered legal edges: `k * (N-k)`.
3. Uniform base rate per legal edge:

```math
r(y,x) = \frac{\gamma}{k(N-k)}.
```

4. Johnson reference kernel `kappa[d]`, with

```math
d(x_0,x)=k-\langle x_0,x\rangle.
```

5. For the transposition `g=(i j)`, the Dirac IASBS terminal log-label is

```math
\log\Lambda_g(X_1)
= -\frac{E(gX_1)-E(X_1)}{\tau}
+ \log\kappa[d(x_0,X_1)]
- \log\kappa[d(x_0,gX_1)].
```

This is exactly what `fixed_ising.py:428-455` currently computes. Only the energy oracle and state storage need to change.

---

## 7. Critical implementation change #1: direct tensor states, no enumeration

### 7.1 Why this is necessary

`FixedIsingSpace` currently calls `enumerate_fixed_count(n,k)`, builds a `2^n` lookup table, and stores target indices for all legal swaps. This is valid for the current small Ising experiments but impossible at `N=64`.

Do not alter `FixedIsingSpace`; introduce a direct state container for CuAu.

### 7.2 Minimal scalable space object

In `structured_asbs/cuau.py`:

```python
class CuAuSpace:
    def __init__(self, n, k, temperature, gamma, energy, source, device):
        assert 0 < k < n
        assert int(source.sum()) == k
        self.n = int(n)
        self.k = int(k)
        self.n_edges = k * (n - k)
        self.temperature = float(temperature)
        self.tau = KB_EV_PER_K * self.temperature
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.energy = energy
        self.x0 = source.to(self.device, dtype=torch.int64)

        kap = C.binary_orbit_kernel(self.n, self.k, self.gamma)
        self.log_kappa_full = torch.tensor(
            np.log(np.maximum(kap, 1e-300)),
            dtype=torch.float64,
            device=self.device,
        )
        self._build_bridge_tables()  # copy mathematics from FixedIsingSpace._bridge_tables

    def distance(self, x0, x):
        return self.k - (x0 * x).sum(dim=-1)

    def kappa_table(self, gammas):
        rows = [np.maximum(C.binary_orbit_kernel(self.n, self.k, float(g)), 1e-300)
                for g in gammas]
        return torch.tensor(np.log(np.stack(rows)), dtype=torch.float64,
                            device=self.device)
```

No state enumeration. No LUT. No `M`. No `tgt` table. No full pair table is needed for the scalable branch.

For true copy-paste reproducibility, the bridge-table method can be lifted directly from the supplied `FixedIsingSpace` with only the method name changed. Add `from scipy.special import gammaln` and use:

```python
def _build_bridge_tables(self):
    """Exact occupancy classes for the Johnson reference bridge."""
    n, k = self.n, self.k
    J = min(k, n - k)

    def lc(N, r):
        return (gammaln(N + 1.0) - gammaln(r + 1.0)
                - gammaln(N - r + 1.0))

    rows = []
    for m in range(J + 1):
        sz = (k - m, m, m, n - k - m)
        cur = [
            (a, b, c, k - a - b - c,
             sum(lc(sz[q], v) for q, v in
                 enumerate((a, b, c, k - a - b - c))))
            for a in range(sz[0] + 1)
            for b in range(sz[1] + 1)
            for c in range(sz[2] + 1)
            if 0 <= k - a - b - c <= sz[3]
        ]
        rows.append(cur)

    Lm = max(len(r) for r in rows)
    A = np.zeros((J + 1, Lm, 4), dtype=np.int64)
    Wt = np.full((J + 1, Lm), -np.inf, dtype=np.float64)
    D0 = np.zeros((J + 1, Lm), dtype=np.int64)
    D1 = np.zeros((J + 1, Lm), dtype=np.int64)

    for m, cur in enumerate(rows):
        for q, (a, b, c, d, w) in enumerate(cur):
            A[m, q] = (a, b, c, d)
            Wt[m, q] = w
            D0[m, q] = k - a - b
            D1[m, q] = k - a - c

    dev = self.device
    self.br_abcd = torch.tensor(A, dtype=torch.int64, device=dev)
    self.br_lmult = torch.tensor(Wt, dtype=torch.float64, device=dev)
    self.br_d0 = torch.tensor(D0, dtype=torch.int64, device=dev)
    self.br_d1 = torch.tensor(D1, dtype=torch.int64, device=dev)
    self.br_size = torch.tensor(
        np.array([[k - m, m, m, n - k - m] for m in range(J + 1)]),
        dtype=torch.int64, device=dev,
    )
```

This is deliberately the same combinatorial construction as the existing benchmark; do not replace it with a new bridge algorithm for CuAu.

The copied occupancy-class bridge table itself remains manageable: for equiatomic `N=64`, the maximum padded class count is 3,281 (about 6 MB for the principal float64/int64 tables); for `N=128` it is 23,969 (about 87 MB for those padded tables). The per-minibatch bridge logits add temporary memory proportional to `batch_size * class_count`, which is another reason to establish the `N=64` result before scaling to `N=128`.

### 7.3 State invariants

Every externally visible state function must assert or be testable against:

```python
x.dtype in (torch.int64, torch.int32, torch.int8, torch.bool)
x.shape[-1] == space.n
(x == 0 | x == 1).all()
(x.sum(-1) == space.k).all()
```

Do not silently round or threshold neural outputs; states are always discrete tensors.

---

## 8. Critical implementation change #2: exact direct-state reference bridge

### 8.1 Reuse the current occupancy-class bridge exactly

Copy `_bridge_tables()` from `FixedIsingSpace` into `CuAuSpace`. It depends only on `(n,k)`, not on the Ising energy or enumeration.

Then copy `sample_bridge()` with only these conceptual changes:

- Input `X1` is `(B,N)` rather than an enumeration index.
- Use `X1` directly wherever the current code uses `space.S[X1_idx]`.
- Return the binary tensor `x` directly rather than mapping it through `lut_t`.

Reference implementation shape:

```python
@torch.no_grad()
def sample_bridge_direct(space, X1, t_idx, log_kap0, log_kap1,
                         generator=None, X0=None):
    B, n = X1.shape
    x0b = (space.x0.unsqueeze(0).expand(B, -1)
           if X0 is None else X0)

    m = space.k - (x0b * X1).sum(dim=1)

    lw = (
        space.br_lmult[m]
        + torch.gather(log_kap0[t_idx], 1, space.br_d0[m])
        + torch.gather(log_kap1[t_idx], 1, space.br_d1[m])
    )
    lw = lw - lw.max(dim=1, keepdim=True).values
    sel = torch.multinomial(torch.exp(lw), 1, generator=generator).squeeze(1)

    cnt = space.br_abcd[m, sel]
    sizes = space.br_size[m]

    # Same four block labels as current fixed_ising.py.
    blk = 2 * (1 - x0b) + (1 - X1)
    key = torch.rand(B, n, device=space.device, generator=generator)
    order = torch.argsort(blk.to(key.dtype) * 2.0 + key, dim=1)
    bs = torch.gather(blk, 1, order)

    start = torch.cat(
        [torch.zeros_like(sizes[:, :1]), sizes.cumsum(1)[:, :3]], dim=1
    )
    pos = torch.arange(n, device=space.device).expand(B, n)
    take = (pos - torch.gather(start, 1, bs)) < torch.gather(cnt, 1, bs)

    x = torch.zeros(B, n, dtype=torch.int64, device=space.device)
    x.scatter_(1, order, take.long())

    assert torch.all(x.sum(dim=1) == space.k)
    return x
```

### 8.2 Validation already performed while preparing this playbook

The above direct-state bridge construction was independently exercised against the exact enumerated Johnson bridge on `n=16,k=8`:

- 100,000 bridge samples.
- Every sample had exactly 8 active sites.
- Every sample mapped to a valid enumerated state.
- Aggregating by the exact pair of endpoint Johnson distances gave empirical TV `0.008385`, consistent with ordinary finite-sample noise.

This is an implementation sanity check, not a paper result; reproduce it in `tests/test_cuau.py` before merging.

### 8.3 Required automated bridge test

At `n=8,k=4` or `n=10,k=5`, where enumeration is very cheap:

1. Choose several endpoint pairs and `t in {0.1,0.3,0.5,0.8}`.
2. Compute exact bridge weights over every state:

```math
p(x_t=x|x_0,x_1)
\propto
\kappa_{\gamma t}[d(x_0,x)]
\kappa_{\gamma(1-t)}[d(x,x_1)].
```

3. Draw at least `200,000` direct bridge samples.
4. Require aggregate discrepancy to be statistically compatible with multinomial sampling.
5. Require zero fixed-count violations.

Do not compare two implementations with the same bug; compare against the explicit formula above.

---

## 9. Scalable controlled simulation

Reuse the **same frozen-rate one-jump-per-grid-bin approximation** as the existing IASBS Ising branch so CuAu does not change the numerical interpretation of the method.

### 9.1 Legal edge mask

```python
def legal_mask(x):
    # x: (B,N); row=edge source Au(1), col=edge target Cu(0)
    return (x[:, :, None] == 1) & (x[:, None, :] == 0)
```

There are exactly `k*(N-k)` true entries per state.

### 9.2 Direct simulation

```python
@torch.no_grad()
def simulate_direct(space, net, batch, steps, generator=None, x0=None):
    if x0 is None:
        x = space.x0.unsqueeze(0).expand(batch, -1).clone()
    else:
        x = x0.clone().to(space.device)

    dt = 1.0 / steps
    base = space.gamma / space.n_edges

    for s in range(steps):
        t_scalar = s * dt
        tt = torch.full((batch,), t_scalar, device=space.device)
        a = net(tt, x).to(torch.float64).clamp(-20.0, 20.0)
        mask = legal_mask(x)

        u = base * torch.exp(a) * mask
        flat = u.reshape(batch, -1)
        R = flat.sum(dim=1)
        p_stay = torch.exp(-R * dt)

        jump = torch.rand(batch, device=space.device,
                          generator=generator) > p_stay
        if jump.any():
            e = torch.multinomial(flat[jump], 1,
                                  generator=generator).squeeze(1)
            i = e // space.n
            j = e % space.n
            rows = torch.nonzero(jump, as_tuple=False).squeeze(1)

            x = x.clone()
            vi = x[rows, i].clone()
            vj = x[rows, j].clone()
            x[rows, i] = vj
            x[rows, j] = vi

    assert torch.all(x.sum(1) == space.k)
    return x
```

### 9.3 Do not use set-to-0/set-to-1 for arbitrary transpositions

For a **legal current-state move**, setting source to 0 and target to 1 is correct. For the **terminal readout `gX1`**, however, the same transposition selected at `Xt` can act on two equal species at `X1`; then it should leave `X1` unchanged. Therefore the shared utility for `gX1` must exchange values, not assume the edge is legal at terminal time.

Use:

```python
def transpose_batch(x, i, j):
    y = x.clone()
    b = torch.arange(x.shape[0], device=x.device)
    vi = x[b, i].clone()
    vj = x[b, j].clone()
    y[b, i] = vj
    y[b, j] = vi
    return y
```

This detail is easy to get wrong and would silently corrupt terminal labels.

---

## 10. Edge-minibatched IASBS training: the main LOC/cost reduction

### 10.1 Why all-pairs terminal labels must not be retained

The current Ising implementation creates every unordered transposition of every terminal state and evaluates the energy of all of them. For `N=64`, this means `C(64,2)=2016` transpositions per terminal state; for `N=128`, `8128`.

The controller loss, however, is averaged over the **legal edges of the intermediate state `Xt`**. Because the reference rate is uniform across legal edges, the edge sum can be Monte-Carlo minibatched without changing the expected objective.

### 10.2 Exact expectation argument

The current per-state Poisson/Bregman objective is

```math
L(x_t)
=\frac{1}{k(N-k)}
\sum_{(i,j)\in E(x_t)}
\left[\exp(a_{ij})-a_{ij}\Lambda_{ij}\right].
```

If

```math
(i,j)\sim\mathrm{Uniform}(E(x_t)),
```

then

```math
\mathbb E_{(i,j)}
[\exp(a_{ij})-a_{ij}\Lambda_{ij}]
=L(x_t).
```

Therefore one legal edge per training example is an unbiased stochastic estimator of the existing edge-average loss.

### 10.3 Numerical sanity check already performed

On random `n=16,k=8` states and random synthetic controller/label tensors:

```text
full all-edge loss     = 1.0196249276849847
sampled-edge MC mean   = 1.019600418821145
MC standard error      = 0.0002525343642693844
z-score of difference  = -0.0971
```

This confirms the code-level edge sampling matches the all-edge loss expectation. Reproduce this as a unit test with a looser stochastic criterion.

### 10.4 Uniform legal-edge sampler

```python
def sample_uniform_legal_edge(x, generator=None):
    B, n = x.shape
    m = legal_mask(x).reshape(B, -1).to(torch.float32)
    e = torch.multinomial(m, 1, generator=generator).squeeze(1)
    return e // n, e % n
```

### 10.5 Edge-specific terminal label

```python
@torch.no_grad()
def terminal_log_label_edge(space, X1, E1, i, j):
    Xg = transpose_batch(X1, i, j)
    Eg = space.energy.energy_torch(Xg).to(torch.float64)

    d1 = space.distance(space.x0.unsqueeze(0), X1).long()
    dg = space.distance(space.x0.unsqueeze(0), Xg).long()

    return (
        -(Eg - E1) / space.tau
        + space.log_kappa_full[d1]
        - space.log_kappa_full[dg]
    )
```

When `X1[:,i] == X1[:,j]`, `Xg == X1`, so this correctly yields `log Lambda = 0` up to energy roundoff.

Add an assertion in debug mode:

```python
same = X1[torch.arange(B), i] == X1[torch.arange(B), j]
assert torch.allclose(log_lam[same], torch.zeros_like(log_lam[same]), atol=1e-10)
```

### 10.6 Cache terminal energies in the replay buffer

Without caching, every terminal label would require evaluating both `E(X1)` and `E(gX1)`. `E(X1)` is shared by every inner update that reuses the endpoint.

Store replay items as `(X1, E1)`. The replay buffer is small, so keep it on the same device as the controller and avoid host/device copies inside the inner loop:

```python
with torch.no_grad():
    X1_new = simulate_direct(
        space, net, batch=args.batch, steps=args.steps,
        generator=train_gen,
    )
    E1_new = space.energy.energy_torch(X1_new).to(torch.float64)

replay.append((X1_new.detach(), E1_new.detach()))
if len(replay) > args.buffer:
    replay.pop(0)
```

Then each sampled edge requires only **one new CE evaluation**, `E(gX1)`. If GPU memory becomes limiting at `N=128`, store replay tensors on pinned CPU memory and move only the selected minibatch to GPU; do not introduce that complexity for `N=64` unless profiling requires it.

### 10.7 Minimal training-loop skeleton

```python
for it in range(1, args.iters + 1):
    with torch.no_grad():
        X1_new = simulate_direct(space, net, args.batch, steps)
        E1_new = space.energy.energy_torch(X1_new).to(torch.float64)

    replay.append((X1_new, E1_new))
    if len(replay) > args.buffer:
        replay.pop(0)

    pool_x = torch.cat([p[0] for p in replay], 0)
    pool_e = torch.cat([p[1] for p in replay], 0)

    for _ in range(args.inner):
        sel = torch.randint(len(pool_x), (args.mb,), device=space.device)
        X1 = pool_x[sel]
        E1 = pool_e[sel]
        ti = torch.randint(steps, (args.mb,), device=space.device)

        with torch.no_grad():
            Xt = sample_bridge_direct(space, X1, ti, log_kap0, log_kap1)
            i, j = sample_uniform_legal_edge(Xt)
            log_lam = terminal_log_label_edge(space, X1, E1, i, j)
            lam = torch.exp(log_lam.clamp(-20.0, 20.0))

        tt = ti.to(torch.float64) / steps
        a = net(tt, Xt)
        b = torch.arange(args.mb, device=space.device)
        av = a[b, i, j].clamp(-20.0, 20.0).to(torch.float64)

        loss = (torch.exp(av) - av * lam).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
```

Start with `--edge-samples 1`. If gradient variance is visibly high, allow `edge_samples in {2,4}`; implement this by repeating the selected endpoint/bridge row or sampling multiple edges, not by reverting to all pairs.

---

## 11. Target-energy call accounting: instrument it from day one

A major value of this benchmark is computational evidence. Wrap the energy oracle with a counter:

```python
class CountedEnergy:
    def __init__(self, backend):
        self.backend = backend
        self.n_configs = 0
        self.wall_seconds = 0.0

    @staticmethod
    def _batch_count(x):
        # x shape (..., N); a single 1-D state counts as one configuration.
        return 1 if x.ndim == 1 else int(np.prod(x.shape[:-1]))

    def energy_np(self, x):
        t0 = time.perf_counter()
        out = self.backend.energy_np(x)
        self.wall_seconds += time.perf_counter() - t0
        self.n_configs += self._batch_count(np.asarray(x))
        return out

    def energy_torch(self, x):
        t0 = time.perf_counter()
        out = self.backend.energy_torch(x)
        self.wall_seconds += time.perf_counter() - t0
        self.n_configs += self._batch_count(x)
        return out

    def snapshot(self):
        return {'n_configs': self.n_configs, 'wall_seconds': self.wall_seconds}
```

Create separate `CountedEnergy` instances (or take/reset explicit snapshots) for training, baseline generation, and evaluation so those costs cannot be mixed accidentally.

Count **configurations evaluated**, not Python function calls. If a batch of 256 configurations is passed once, count 256 target-energy evaluations.

For IASBS, report at minimum:

```text
training CE evaluations
reference/precomputation CE evaluations (normally zero for Johnson kernel)
evaluation-only CE evaluations
inference CE evaluations (should be zero)
training wall time
10k-sample inference wall time
```

For canonical MCMC with current energy cached, count one candidate energy evaluation per proposed swap unless the calculator exposes an exact local delta-energy primitive. If a local primitive is used, report that separately rather than pretending it equals a full CE evaluation.

**Primary fairness curve:** quality vs target-energy oracle evaluations.

**Secondary fairness curve:** quality vs end-to-end wall time.

Do not combine PT-reference cost with a method's benchmark cost; PT is ground-truth generation.

---

## 12. Network: reuse `SwapController` first

Use exactly:

```python
from structured_asbs.fixed_ising import SwapController
net = SwapController(space.n, hidden=args.hidden).to(space.device)
```

Do not initially introduce lattice positional encodings or a Transformer. Reusing the controller has two scientific advantages:

1. Improvement cannot be attributed to a CuAu-specific architecture.
2. The experiment demonstrates that the new scale/physics comes from the state-space construction and energy oracle, not a separate modeling contribution.

Recommended first attempt:

```text
N=16: hidden=512 (same as current fixed-composition branch)
N=64: hidden=512
N=128: try hidden=512 first; reduce only for memory, change architecture only if necessary
```

The output has `N^2` entries. For `N=64`, this is 4096 outputs and is entirely reasonable. `N=128` has 16384 outputs and is still feasible, though memory/throughput should be measured.

---

## 13. Source state policy

The current Ising default `first k sites = 1` has no physical meaning on the alloy lattice and can accidentally encode a spatially special state. Use explicit reproducible sources.

### Primary source

Use a deterministic pseudorandom equiatomic configuration:

```python
def random_fixed_source(n, k, seed=1729):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    x = torch.zeros(n, dtype=torch.int64)
    x[perm[:k]] = 1
    return x
```

This is the default headline source.

### Stress-test source

Construct a perfect `L1_0-x` ordered equiatomic state and train/sample from that single orientation. If IASBS recovers the other symmetry-equivalent ordered sectors, this is a strong global-transport stress test.

Do **not** replace the primary run with the ordered-source stress test; present it as an ablation so reviewers cannot argue that the benchmark was chosen to maximize a source-dependent effect.

---

## 14. Benchmark suite

### 14.1 CuAu-S: exact canonical real-energy validation

```text
size:           2 x 2 x 4
N:              16
k:              8
state count:    12,870 in canonical sector
T:              500 K primary; 680 K and 1200 K controls
reference:      exact enumeration of the 12,870 canonical states
baselines:      DAM direct; local canonical MCMC optional
```

The 2026 lattice-thermodynamics CuAu study uses a 16-site `2x2x4` problem and computes exact thermodynamics by enumerating all `2^16=65,536` unconstrained configurations. Your canonical sector is only 12,870 states, giving a clean exact benchmark.

**Important practical gate:** the official MetaDNS README publicly documents the `4x4x4` VASP file, not necessarily a ready-made `2x2x4` VASP file. Do not improvise lattice-site ordering. Before implementing CuAu-S, inspect the upstream CuAu builder/energy code and generate a compatible 16-site supercell using the same primitive structure and site ordering. Then verify the 100-state energy parity test. If a compatible 16-site structure cannot be generated without changing the CE model, omit CuAu-S rather than silently using a different Hamiltonian; the `N=64` benchmark remains valid.

### 14.2 CuAu-M: headline benchmark

```text
size:           4 x 4 x 4
N:              64
k:              32
state count:    1.832624140942590534e18
T:              500 K primary
controls:       680 K, 1200 K
reference:      long canonical parallel tempering
baseline:       ordinary single-temperature canonical swap MCMC
optional:       conditioned/reweighted MetaDNS released samples
```

This should carry the main paper figure.

### 14.3 CuAu-L: optional scaling benchmark

```text
size:           4 x 4 x 8
N:              128
k:              64
state count:    2.3951146041928083e37
T:              500 K first
reference:      canonical PT
baseline:       canonical swap MCMC
```

Only run after CuAu-M is correct and stable. The 2026 scaling paper explicitly studies CuAu up to `4x4x8`, so this is excellent scale precedent. However, the rectangular cell should **not** be assigned exact three-way x/y/z orientation symmetry; use PT-measured sector weights there.

---

## 15. Temperature choices

Use the same three temperatures highlighted in the MetaDNS public CuAu sampling interface:

```text
500 K   low-temperature/ordered, primary difficult target
680 K   near the order-disorder regime, transition-region control
1200 K  high-temperature/disordered control
```

Do not assert in the paper that 680 K is exactly the finite-cell transition temperature for your fixed-composition CE. Treat it as a near-transition benchmark inherited from prior CuAu work, and let the reference order-parameter/heat-capacity data show the actual finite-cell behavior.

The purpose of the three temperatures is:

| T | scientific role |
|---:|---|
| 500 K | demonstrate multimodal ordered sampling and barrier crossing |
| 680 K | demonstrate fidelity near a difficult thermodynamic crossover |
| 1200 K | demonstrate that IASBS does not artificially impose ordering |

---

## 16. Reference sampler: canonical parallel tempering

### 16.1 Why PT is the reference

Single-temperature canonical swap MCMC is the direct baseline and may be trapped. It should not simultaneously define the "truth" against which it is judged.

Use independent canonical parallel-tempering runs as the high-quality reference.

### 16.2 Preferred implementation: same energy wrapper, tiny explicit MCMC/PT code

The safest reference implementation is a small canonical Metropolis + replica-exchange loop that calls the **same verified `CuAuEnergy` wrapper** used by IASBS. This avoids a subtle but important integration risk: `mchammer-pt` expects an icet `ClusterExpansion`, whereas the released MetaDNS CuAu artifact is exposed through its CLEASE/iCET energy code and ECI JSON. Do not assume the JSON can be handed directly to `mchammer-pt`.

The canonical single-temperature step is exactly:

```python
# x has exactly k Au atoms; E is cached total energy
i = uniform_random_Au_site(x)
j = uniform_random_Cu_site(x)
y = transpose(x, i, j)
Ey = energy(y)
log_alpha = -(Ey - E) / (KB_EV_PER_K * T)
if log(U) < min(0.0, log_alpha):
    x, E = y, Ey
```

For adjacent PT replicas `a,b` with inverse temperatures `beta_a,beta_b`, propose exchanging their whole configurations and accept with

```math
\log \alpha_{swap}
= (\beta_a-\beta_b)(E_a-E_b),
```

clipped at zero in log space. Because every replica has the same `N,k`, replica exchange preserves the hard composition constraint. Alternate even and odd adjacent pairs so every neighboring temperature pair is proposed symmetrically.

This reference code should be roughly 100-150 LOC and is easier to audit than a cross-library CE conversion. Validate it on CuAu-S against the exact canonical law before using it as the large-system gold reference.

A drop-in reference core is below. Put it in `structured_asbs/cuau_reference.py` and make the CLI/provenance code call these functions rather than duplicating transition logic. `energy_np` must be the same verified wrapper used by IASBS and must return total eV.

```python
import math
import numpy as np

KB_EV_PER_K = 8.617333262145e-5


def _log_uniform(rng: np.random.Generator) -> float:
    # Avoid log(0) without changing the distribution at machine precision.
    return math.log(max(float(rng.random()), np.finfo(np.float64).tiny))


def canonical_step(x, E, beta, energy_np, rng):
    """One symmetric unlike-species swap Metropolis step.

    x: 1-D binary NumPy array, 1=Au, 0=Cu.
    E: cached total energy [eV].
    beta: 1/(k_B T) [1/eV].
    Returns (possibly new x, cached E, accepted). Exactly one candidate
    configuration energy is evaluated.
    """
    occ = np.flatnonzero(x == 1)
    emp = np.flatnonzero(x == 0)
    if occ.size == 0 or emp.size == 0:
        raise ValueError('canonical swap requires 0 < k < N')

    i = int(occ[rng.integers(occ.size)])
    j = int(emp[rng.integers(emp.size)])
    y = x.copy()
    y[i], y[j] = y[j], y[i]

    Ey = float(np.asarray(energy_np(y[None, :])).reshape(-1)[0])
    if not np.isfinite(Ey):
        raise FloatingPointError('non-finite CuAu candidate energy')

    log_alpha = -beta * (Ey - E)
    if _log_uniform(rng) < min(0.0, log_alpha):
        return y, Ey, True
    return x, E, False


def canonical_mcmc(energy_np, x0, temperature_K, n_steps, *,
                   burn=0, thin=1, seed=0):
    """Single-temperature direct baseline."""
    if not (0 <= burn < n_steps) or thin <= 0:
        raise ValueError('invalid burn/thin')
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.int8).copy()
    k = int(x.sum())
    beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
    E = float(np.asarray(energy_np(x[None, :])).reshape(-1)[0])

    samples, energies = [], []
    accepted = 0
    for step in range(n_steps):
        x, E, acc = canonical_step(x, E, beta, energy_np, rng)
        accepted += int(acc)
        if step >= burn and (step - burn) % thin == 0:
            if int(x.sum()) != k:
                raise RuntimeError('composition changed in canonical MCMC')
            samples.append(x.copy())
            energies.append(E)

    return {
        'samples': np.asarray(samples, dtype=np.int8),
        'energies_eV': np.asarray(energies, dtype=np.float64),
        'acceptance': accepted / n_steps,
        # Candidate calls only. Add the single initialization call separately
        # if reporting absolute oracle counts.
        'candidate_energy_evals': int(n_steps),
    }


def canonical_pt(energy_np, x0_by_replica, temperatures_K, *,
                 rounds, local_steps_per_round=10, burn_rounds=0,
                 record_every=1, seed=0):
    """Canonical replica exchange with fixed temperatures in slots.

    `x0_by_replica` has shape (R,N), one fixed-composition state per
    temperature slot. Whole configurations (and cached energies/walker IDs)
    are exchanged; beta values remain attached to temperature slots.
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(temperatures_K, dtype=np.float64)
    xs = np.asarray(x0_by_replica, dtype=np.int8).copy()
    if xs.ndim != 2 or xs.shape[0] != T.size:
        raise ValueError('x0_by_replica must have shape (n_temperatures, N)')
    if np.any(T <= 0):
        raise ValueError('all temperatures must be positive')
    if not np.all(xs.sum(axis=1) == xs[0].sum()):
        raise ValueError('all replicas must have the same fixed composition')
    if burn_rounds < 0 or burn_rounds >= rounds or record_every <= 0:
        raise ValueError('invalid PT burn/record settings')

    beta = 1.0 / (KB_EV_PER_K * T)
    E = np.asarray(energy_np(xs), dtype=np.float64).reshape(-1)
    if E.shape != (T.size,) or not np.all(np.isfinite(E)):
        raise ValueError('energy_np must return one finite energy per replica')

    R = T.size
    walker = np.arange(R, dtype=np.int64)  # walker ID currently in each T slot
    local_accept = np.zeros(R, dtype=np.int64)
    local_trials = np.zeros(R, dtype=np.int64)
    swap_accept = np.zeros(R - 1, dtype=np.int64)
    swap_trials = np.zeros(R - 1, dtype=np.int64)
    walker_at_temperature = []
    samples_by_slot = [[] for _ in range(R)]
    energies_by_slot = [[] for _ in range(R)]

    for r in range(rounds):
        # Local canonical dynamics at each fixed temperature slot.
        for q in range(R):
            for _ in range(local_steps_per_round):
                xs[q], E[q], acc = canonical_step(
                    xs[q], E[q], beta[q], energy_np, rng
                )
                local_accept[q] += int(acc)
                local_trials[q] += 1

        # Alternate even and odd neighboring pairs. Cached energies make
        # replica exchange require zero new CE evaluations.
        parity = r & 1
        for a in range(parity, R - 1, 2):
            b = a + 1
            log_alpha = (beta[a] - beta[b]) * (E[a] - E[b])
            swap_trials[a] += 1
            if _log_uniform(rng) < min(0.0, float(log_alpha)):
                xs[[a, b]] = xs[[b, a]]
                E[[a, b]] = E[[b, a]]
                walker[[a, b]] = walker[[b, a]]
                swap_accept[a] += 1

        # Keep the full walker-label trajectory for round-trip diagnostics.
        walker_at_temperature.append(walker.copy())

        if r >= burn_rounds and (r - burn_rounds) % record_every == 0:
            for q in range(R):
                samples_by_slot[q].append(xs[q].copy())
                energies_by_slot[q].append(float(E[q]))

    k = int(xs[0].sum())
    for q in range(R):
        arr = np.asarray(samples_by_slot[q], dtype=np.int8)
        if arr.size and not np.all(arr.sum(axis=1) == k):
            raise RuntimeError('composition changed during PT')
        samples_by_slot[q] = arr
        energies_by_slot[q] = np.asarray(energies_by_slot[q], dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        local_rate = np.divide(local_accept, local_trials,
                               out=np.zeros_like(local_accept, dtype=float),
                               where=local_trials > 0)
        swap_rate = np.divide(swap_accept, swap_trials,
                              out=np.zeros_like(swap_accept, dtype=float),
                              where=swap_trials > 0)

    return {
        'temperatures_K': T,
        'samples_by_slot': samples_by_slot,
        'energies_eV_by_slot': energies_by_slot,
        'local_acceptance': local_rate,
        'swap_acceptance': swap_rate,
        'walker_at_temperature': np.asarray(walker_at_temperature),
        'candidate_energy_evals': int(R * rounds * local_steps_per_round),
    }
```

**Mandatory reference-code test before N=64:** on any verified exact canonical system (prefer CuAu-S; otherwise a synthetic `n<=10` fixed-composition target), run the single-temperature routine and PT long enough that their retained marginal at each checked temperature agrees with the explicitly normalized exact law within sampling uncertainty. Also verify detailed balance numerically on randomly chosen neighbor pairs:

```math
\pi(x)P(x,y)=\pi(y)P(y,x)
```

up to floating-point tolerance. This catches sign errors in both the local Metropolis and replica-exchange formulas before the reference is used as ground truth.

For the round-trip convergence gate, `walker_at_temperature[t,slot]` stores the walker ID occupying each temperature slot. The inverse trajectory and a conservative low->high->low round-trip count are:

```python
def walker_slot_trajectories(walker_at_temperature):
    w = np.asarray(walker_at_temperature, dtype=np.int64)
    # Every row is a permutation of walker IDs. argsort gives slot per walker.
    return np.argsort(w, axis=1)  # (round, walker)


def count_low_high_low_roundtrips(walker_at_temperature):
    slot = walker_slot_trajectories(walker_at_temperature)
    hi = slot.shape[1] - 1
    counts = np.zeros(slot.shape[1], dtype=np.int64)
    for walker in range(slot.shape[1]):
        phase = 0  # 0=waiting for low, 1=waiting for high, 2=waiting for low
        for q in slot[:, walker]:
            if phase == 0 and q == 0:
                phase = 1
            elif phase == 1 and q == hi:
                phase = 2
            elif phase == 2 and q == 0:
                counts[walker] += 1
                phase = 1
    return counts
```

Inspect the per-walker counts, not only their sum; one mobile walker does not prove the entire PT ensemble is mixing.

### 16.3 Optional independent cross-check with icet/mchammer-pt

Current `mchammer-pt` provides `CanonicalParallelTempering` on top of icet/mchammer and exposes round-trip/autocorrelation diagnostics. Use it as an **independent reference cross-check only if** you can construct an icet `ClusterExpansion` that reproduces the MetaDNS energy parity corpus. Never assume the CLEASE ECI JSON is directly compatible.

icet/mchammer's canonical semantics are exactly the desired target: fixed numbers of species, unlike-atom swaps, and Metropolis probability proportional to `exp(-DeltaE/kBT)`. If an icet representation passes the 100-state and swap-Delta-E parity gates, its PT output is an excellent second implementation check.

### 16.4 Suggested ladder

Start with a pilot ladder that includes the paper temperatures exactly, e.g.

```python
T_ladder = [350, 400, 450, 500, 550, 600, 640, 680,
            720, 760, 820, 900, 1050, 1200, 1400]
```

This is a starting point, not a sacred hyperparameter. After a short pilot, adjust spacing so neighboring replica-exchange acceptance is neither near zero nor trivially one. A reasonable target is roughly `0.15-0.5` acceptance for most adjacent pairs.

### 16.5 Independent starts

Run at least four independent PT jobs, with their initial 500 K structures chosen from:

```text
random equiatomic
perfect L10-x
perfect L10-y
perfect L10-z
```

If the PT package replicates one initial structure across the full temperature ladder, use one such structure per independent PT job.

### 16.6 Reference convergence gates

Do not choose a fixed production length and declare convergence. Increase it until all of these hold:

1. **Round trips:** each independent PT job shows repeated temperature-ladder round trips after burn-in; target at least 20 as a practical starting gate.
2. **Independent-start agreement:** post-burn-in estimates from the different starts agree for `E/N`, `Qmax`, ordered mass, and sector masses within statistical uncertainty.
3. **Autocorrelation/ESS:** obtain an effective sample size of at least `~5000` for both `E/N` and `Qmax` in the 500 K pooled reference if feasible.
4. **Split-half stability:** first and second post-burn-in halves give small KS/W1 discrepancies relative to the method effects being claimed.
5. **Symmetry at 4x4x4:** x/y/z ordered-sector masses agree within reference uncertainty after applying the same symmetric basin definition.

If these fail, the reference is not ready; increasing IASBS training does not fix an unconverged reference.

### 16.7 PT split-half noise floor

This is essential for reviewer-proof evaluation.

Split independent converged reference samples into two disjoint pools `R1`, `R2`. For every metric used to compare IASBS to PT, calculate the same metric between `R1` and `R2`.

Example table layout:

| comparison | E/N KS | E/N W1 | Qmax KS | Qmax W1 | SRO MAE |
|---|---:|---:|---:|---:|---:|
| PT half 1 vs half 2 | floor | floor | floor | floor | floor |
| IASBS vs PT | ... | ... | ... | ... | ... |
| local MCMC vs PT | ... | ... | ... | ... | ... |

If IASBS approaches the PT-vs-PT floor, that is much stronger than an isolated small distance number.

---

## 17. Direct large-system baseline: local canonical swap MCMC

Run ordinary fixed-composition Metropolis swap MCMC at each target temperature, using the **same CE energy oracle** if possible.

Algorithm:

```text
cache E(x)
repeat:
    choose i uniformly among Au sites
    choose j uniformly among Cu sites
    y = swap(x,i,j)
    evaluate E(y)
    Delta = E(y)-E(x)
    accept with min(1, exp(-Delta/(kB*T)))
    if accepted: x=y; E(x)=E(y)
```

This is exactly the canonical fixed-composition sampler documented by icet/mchammer.

### 17.1 Fair starts

For each budget, run multiple seeds from the same start families used to diagnose PT:

```text
random
L10-x
L10-y
L10-z
```

For the primary matched-budget method comparison, use the **same deterministic random source convention** as IASBS unless the comparison's purpose is explicitly the ordered-basin stress test.

### 17.2 Matched-budget curves

Do not report one arbitrarily selected MCMC run length. Evaluate quality at logarithmically spaced cumulative target-energy budgets, e.g.

```text
1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, ...
```

adjusted to the actual IASBS training budget.

For IASBS, the quality point at training budget `B` should use a checkpoint trained with `B` cumulative target-energy evaluations, then generate a fixed number (e.g. 10,000 or 50,000) of samples. IASBS inference requires no CE evaluation.

For MCMC, use the same target-energy evaluation budget and then evaluate the retained/decorrelated states up to that budget.

Report both:

- quality vs CE configurations evaluated;
- quality vs end-to-end wall time.

---

## 18. DAM baseline: exact small CuAu only by default

DAM is methodologically close and already implemented in the repository, so include it where it is fair and computationally interpretable.

### 18.1 Use DAM on CuAu-S

For `N=16,k=8`, build an exact enumerated CuAu space with the same fields used by the current `IsingAdapter`:

```text
S               enumerated states
E               CE energy table
pi              exact canonical target
x0/i0           source
edge_i/edge_j   legal edge tensors
tgt             enumeration target index per legal swap
logf1            -E/tau - log kappa[d(x0,x)] up to constant
```

Then the existing `IsingAdapter` logic is almost generic already. The lowest-risk route is to add a tiny `CuAuExactAdapter` copied from `IsingAdapter` and point it at `CuAuExactSpace`; do not generalize/refactor Ising just to save 40 lines.

The generic `dam/core.py` should require no changes.

### 18.2 Why not require N=64 DAM

DAM's estimator performs nested terminal rollouts and terminal `f1` evaluations. With an expensive CE target, a large-system DAM run can become dominated by target-energy calls. That fact may itself support IASBS, but a huge failed/incomplete baseline is not necessary if the small exact comparison and the paper's existing DAM results already establish methodology.

If budget permits, a short N=64 DAM attempt can be reported as a cost diagnostic, but do not make the CuAu experiment dependent on it.

---

## 19. Prior work: what to compare directly and what not to compare directly

### 19.1 MetaDNS, ICML 2026

**Use for:** strongest modern CuAu neural-sampling precedent, same public `4x4x4` CE benchmark, same 500/680/1200 K temperature family, public code/checkpoints/data.

Their public benchmark is semi-grand canonical; the official training example uses a concentration collective variable. Their paper reports CuAu quantities including energy-distribution JS, concentration JS, NESS, free-energy/PMF information, and computational timings.

Published 4x4x4 500 K values can be quoted as **context** but not as direct canonical leaderboard numbers. In the reported semi-grand problem, MetaDNS has approximately energy-JS `0.079`, concentration-JS `0.085`, and NESS `0.321` at 500 K, while the unenhanced model shows stronger mode collapse. Those numbers are for a different target distribution.

**Do not write:** `IASBS E-JS X < MetaDNS E-JS 0.079, therefore IASBS is better.`

### 19.2 SEGAL, npj Computational Materials 2022

**Use for:** establishes DFT-derived CuAu cluster expansion as a realistic neural lattice-sampling application and provides public code/data/model lineage.

SEGAL used semi-grand canonical sampling. The paper reports thermodynamic/phase behavior, NESS, and energy-evaluation counts; for its larger CuAu experiment it reports about `3.0e7` CE energy evaluations for SEGAL and `3.6e8` for a metadynamics calculation while explicitly warning that these costs were not directly comparable because the reference had higher accuracy.

Source: https://www.nature.com/articles/s41524-022-00736-4

### 19.3 Scaling Autoregressive Models for Lattice Thermodynamics, 2026 preprint

**Use for:** establishes the exact 16-site CuAu benchmark and scaling to `4x4x4` and `4x4x8`; supplies contemporary thermodynamic metrics and scale context.

The paper reports:

- exact `2^16` enumeration on the 16-site system;
- `4x4x4` and `4x4x8` CuAu;
- energy/free-energy/phase behavior and heat-capacity style observables;
- directly trained/out-painted MAM Transformer results with free-energy deviations typically below `5 meV/atom` on much of the `4x4x8` regime;
- semi-grand/metadynamics references.

Source: https://arxiv.org/abs/2603.14695

At the time of this playbook, treat its code as **not relied upon** unless a public release is independently verified. Your implementation needs only MetaDNS CE assets plus canonical reference tools.

### 19.4 icet / mchammer

**Use for:** established canonical MCMC semantics and standardized materials observables.

Canonical ensemble documentation explicitly samples fixed composition by swapping unlike atoms with Metropolis probability `exp(-DeltaE/kBT)`.

Sources:

- https://icet.materialsmodeling.org/dev/moduleref/ensembles.html
- https://icet.materialsmodeling.org/moduleref/observers.html

### 19.5 Direct-comparison summary

| method/work | target-matched canonical comparison? | role |
|---|---|---|
| IASBS | yes | proposed method |
| DAM on N=16 | yes | direct exact method baseline |
| local canonical swap MCMC | yes | direct large-system baseline |
| canonical PT | yes | gold reference, not competitor |
| MetaDNS published numbers | **no** | prior-work context |
| SEGAL published numbers | **no** | prior-work context |
| MAM published numbers | **no** | prior-work context |
| conditioned/reweighted MetaDNS samples | potentially yes | optional direct prior-model comparison |

---

## 20. Optional direct MetaDNS comparison: how to make it mathematically defensible

This is optional. Do it only after the core IASBS-vs-canonical-MCMC experiment is complete.

### 20.1 Exact semi-grand reference samples

For an exact semi-grand target

```math
p_{\rm SGC}(x) \propto
\exp[-\beta(E(x)-\mu\Delta N(x))],
```

conditioning on a fixed composition makes the chemical-potential term constant. Therefore

```math
p_{\rm SGC}(x\mid N_{Au}=N/2)
\propto \exp[-\beta E(x)],
```

which is exactly the canonical target.

Thus, if the released MetaDNS archive contains well-converged **MCMC target-reference** samples and enough samples land exactly at `N_Au=N/2`, filtering those reference samples gives an additional independent canonical reference sample.

### 20.2 Neural MetaDNS samples are different

Do not simply filter raw neural samples and call them target samples. MetaDNS samples are generated from a learned proposal. If the release supplies proposal probabilities/importance weights, then:

1. retain samples with `N_Au=N/2`;
2. retain their correct target/proposal importance weights;
3. renormalize those weights within the fixed-composition subset;
4. calculate weighted canonical observables.

If effective sample size after conditioning is poor, report that and do not force the comparison.

Label any plotted result explicitly as e.g.

```text
MetaDNS (conditioned/reweighted to x_Au=0.5)
```

not simply `MetaDNS`.

---

## 21. Order parameters: make the multimodality physically visible

### 21.1 L10 orientation coordinates

For a cubic FCC CuAu supercell, use the three X-point ordering directions. With Ising-like species variable

```math
s_j = 2x_j-1 \in \{-1,+1\},
```

define

```math
Q_\alpha(x)
=\frac{1}{N}
\left|\sum_{j=1}^N s_j e^{i q_\alpha^T r_j}\right|,
\qquad \alpha\in\{x,y,z\},
```

with

```math
q_x=(2\pi/a)(1,0,0),\quad
q_y=(2\pi/a)(0,1,0),\quad
q_z=(2\pi/a)(0,0,1)
```

for the conventional cubic axes.

Do **not** hard-code site phases before checking the actual VASP lattice vectors and coordinate convention. Build `q` from the physical cell/known conventional lattice constant or use icet's `StructureFactorObserver`, which explicitly supports these q-points.

icet documents this exact family of q-points for Cu-Au long-range-order monitoring:

```python
2*np.pi/alat * np.array([1,0,0])
2*np.pi/alat * np.array([0,1,0])
2*np.pi/alat * np.array([0,0,1])
```

A minimal implementation, using Cartesian ASE positions, is:

```python
def fcc_conventional_lattice_constant(atoms):
    # An FCC Bravais lattice has primitive volume a^3/4 per lattice site.
    # This is valid for an unrelaxed commensurate FCC supercell.
    N = len(atoms)
    return float((4.0 * atoms.get_volume() / N) ** (1.0 / 3.0))


def l10_q_vectors(atoms):
    a = fcc_conventional_lattice_constant(atoms)
    return (2.0 * np.pi / a) * np.eye(3, dtype=np.float64)


def l10_order_parameters(x, positions_cart, q_vectors):
    """Return (...,3) absolute L10 structure-factor coordinates."""
    x = np.asarray(x)
    r = np.asarray(positions_cart, dtype=np.float64)
    q = np.asarray(q_vectors, dtype=np.float64)
    if x.shape[-1] != r.shape[0] or q.shape != (3, 3):
        raise ValueError('state/position/q shape mismatch')
    s = 2.0 * x.astype(np.float64) - 1.0
    phase = np.exp(1j * (r @ q.T))  # (N,3)
    z = np.einsum('...n,nq->...q', s, phase) / x.shape[-1]
    return np.abs(z)


def ideal_l10_state(positions_cart, q_vector):
    """Construct one ideal orientation; binary complement is equivalent."""
    c = np.cos(np.asarray(positions_cart) @ np.asarray(q_vector))
    x = (c < 0.0).astype(np.int8)
    if x.sum() * 2 != x.size:
        raise ValueError('q/positions do not generate an equiatomic L10 pattern')
    return x
```

Use the actual atom positions in the exact order consumed by the energy adapter. If the VASP cell axes are not aligned with the conventional cubic axes assumed above, do **not** silently rotate the q-vectors by hand: obtain the cubic orientation from the upstream structure builder or use icet's observer, then repeat the ideal-state and Hamiltonian-symmetry gates below.

### 21.2 Order-parameter validation gate

Before any sampling comparison, construct ideal ordered test configurations and require:

```text
L10-x: Qx is maximal and near its expected ideal value; Qy,Qz small
L10-y: Qy maximal
L10-z: Qz maximal
random equiatomic: no deterministic orientation preference across many random draws
```

Also apply cubic rotations/permutations to a state and verify that `(Qx,Qy,Qz)` permutes correspondingly.

For the exact `1/3` cubic-sector claim, add a **Hamiltonian symmetry gate**: build the site permutation induced by the relevant cubic rotations and verify on at least 100 random equiatomic states that

```text
abs(E(x) - E(R x)) < energy_parity_tolerance
```

for rotations exchanging x/y/z. This catches a non-cubic supercell convention, a site-mapping error, or a CE implementation that does not preserve the assumed finite-cell symmetry. If this gate fails, compare orientation masses only to PT and do not claim exact `1/3` target mass.

If the order-parameter geometry test fails, do not use the metric.

### 21.3 Main scalar order statistic

```math
Q_{\max}=\max(Q_x,Q_y,Q_z).
```

This distinguishes ordered from disordered configurations while orientation identifies the basin.

### 21.4 Ordered-basin definition

Do not classify every disordered sample into x/y/z by raw `argmax`. Use a symmetric threshold:

```python
ordered = Qmax >= q0
sector = argmax([Qx,Qy,Qz]) if ordered else "disordered"
```

Choose `q0` using the **PT reference only**, before inspecting IASBS-vs-baseline differences. A robust protocol is:

1. inspect the 500 K PT `Qmax` distribution;
2. if clearly bimodal/separated, choose the local density minimum between disordered and ordered peaks;
3. freeze this value for all methods/seeds;
4. report a threshold sensitivity check at `q0 +/- 0.05` or a similarly small physically meaningful interval.

If the PT distribution is not separable, do not fabricate a basin threshold; use continuous `Q` distribution metrics as primary evidence.

### 21.5 Symmetry metric for cubic 4x4x4

Among ordered samples define conditional orientation masses

```math
p_\alpha = P(\mathrm{sector}=\alpha\mid Q_{\max}\ge q_0).
```

For a cubic symmetry-preserving Hamiltonian/reference cell they should agree. Report

```math
\mathrm{ModeImbalance}
=\frac12\sum_{\alpha\in\{x,y,z\}}
|p_\alpha-1/3|.
```

Also report the total ordered mass `P(Qmax>=q0)`. Mode imbalance alone is insufficient because a disordered sampler can have a superficially balanced argmax.

For `4x4x8`, do not use `1/3` as the expected finite-cell orientation weight; compare sector masses to PT instead.

---

## 22. Metrics: exact definitions and priorities

### 22.1 Primary large-system metrics

| metric | definition/use | reason |
|---|---|---|
| `E/N KS` | two-sample KS on energy per atom | bin-free thermodynamic distribution |
| `E/N W1` | 1-Wasserstein, report meV/atom | interpretable energy-law error |
| `Qmax KS` | two-sample KS | global long-range-order law |
| `Qmax W1` | Wasserstein on Qmax | magnitude-sensitive order error |
| ordered mass error | `abs(p_ord_method-p_ord_ref)` | ordered/disordered balance |
| orientation mass error | TV of x/y/z conditional sector vector | multimodal coverage |
| cubic mode imbalance | vs 1/3 at `4x4x4` only | symmetry diagnostic independent of sampler |
| mean `E/N` error | absolute error, meV/atom | standard thermodynamics |
| heat-capacity error | relative/absolute `Cv/N` difference | energy fluctuations |
| SRO/pair error | MAE across chosen shells | local chemistry |
| violations | number with `sum x != k` | must be exactly zero |
| CE evaluations | counted target evaluations | method cost |
| wall time | train + sampling separately | practical cost |

### 22.2 Heat capacity

With total energy in eV,

```math
C_V
= \frac{\operatorname{Var}(E)}{k_B T^2},
\qquad
C_V/N
= \frac{\operatorname{Var}(E)}{N k_B T^2}.
```

Use block/bootstrap uncertainty that respects autocorrelation for the PT/MCMC reference. IID IASBS generated samples can use ordinary bootstrap across generated samples, but training-seed variation should be reported separately.

### 22.3 SRO

Prefer an established observer implementation such as icet's binary short-range-order observer. Select a fixed set of neighbor shells before method comparison and report MAE of the vector relative to PT.

### 22.4 JS divergence

JS on energy/order histograms may be reported as a secondary metric to align with MetaDNS's evaluation vocabulary. It is bin-dependent; therefore KS/W1 should remain primary.

If JS is included:

1. define bin edges from the PT reference only;
2. freeze them for all methods;
3. document the bin rule;
4. never directly compare the resulting number to MetaDNS's published semi-grand JS as if it were the same target.

### 22.5 Do not prioritize NESS

NESS is especially natural when a model exposes tractable proposal probabilities for importance weighting. IASBS's primary evidence should be distributional sample quality and cost, not an artificially engineered likelihood metric. Do not add a new density estimator merely to copy a prior paper's metric.

---

## 23. CuAu-S exact evaluation protocol

If a verified compatible 16-site CE structure is available:

### 23.1 Build exact canonical law

```python
S = C.enumerate_fixed_count(16, 8)       # (12870,16)
E = energy.energy_np(S)                  # total eV
logw = -E / (KB_EV_PER_K * T)
logw -= logw.max()
pi = np.exp(logw)
pi /= pi.sum()
```

### 23.2 Exact metrics

For IASBS's exactly propagated discretized controlled law `p` report:

```text
TV(p,pi)
KL(p||pi)
Hellinger(p,pi)
energy-histogram TV
abs mean E/N error [meV/atom]
Cv/N error
Q/order-distribution discrepancy
constraint violations
```

This mirrors the current fixed-composition Ising evaluation and demonstrates that changing to a real CE energy does not break the exact structured sampler.

### 23.3 Exact discretization sweep

As in the current Ising appendix, compute an oracle-controller discretization curve over, e.g.

```text
32, 64, 128, 256, 512 steps
```

Choose the production discretization only after seeing that the oracle floor is comfortably below learned-model error. This distinguishes controller-learning error from finite-grid integration error.

### 23.4 DAM comparison

Run DAM using the same source, target, step grid, and CE energy table. Match the paper's DAM conventions as closely as possible; report exact law TV and DAM nested `f1`/CE-evaluation cost.

This is the cleanest direct method-vs-method CuAu comparison because the target law is known exactly.

---

## 24. CuAu-M reference characterization before IASBS training

Do **not** train IASBS first and then decide which physics to plot. Characterize the canonical target independently.

At `N=64,k=32`, run PT and answer:

1. Is 500 K strongly ordered according to `Qmax`?
2. Are x/y/z sectors visibly populated in converged cubic PT?
3. How separated are the sectors in order-parameter space?
4. How slowly does single-temperature local swap MCMC change orientation at 500 K?
5. What is the actual 680 K distribution: ordered, mixed, or mostly disordered?
6. Does 1200 K look disordered and easy?

Record these before selecting the headline plot.

**If 500 K does not produce the expected separated within-composition orientation basins under this exact CE and finite cell, do not claim that it does.** The experiment remains useful as a real canonical thermodynamics benchmark; adjust the narrative, or evaluate a lower T only if that choice is documented as a pre-IASBS target-characterization decision.

---

## 25. Gamma/reference-clock selection

The existing Ising run uses `gamma=10`. Reuse `10` first, but do not assume it remains numerically ideal as N grows.

The base reference makes an expected order-`gamma` number of swaps over the unit interval. A too-small gamma can make the Dirac reference endpoint extremely concentrated relative to the target and cause very large `f1`/terminal kernel ratios; a too-large gamma increases controlled path activity.

Before full `N=64` training, run a small **predeclared** gamma pilot:

```text
gamma in {10, 20, 40}
```

For each gamma:

1. compute the Johnson `log_kappa_full[j]` table;
2. on a fixed PT pilot sample, compute the source-to-target distance distribution;
3. evaluate the distribution of the **reference kernel-ratio component** for uniformly sampled terminal transpositions;
4. run a short one-seed IASBS pilot with identical CE-call budget;
5. choose gamma using training stability and held-out PT metric quality, then freeze it for all final seeds/temperatures of that size.

Do not sweep dozens of values after seeing the final test curves.

Record the selection protocol in the paper appendix.

---

## 26. Step-count selection

The direct CuAu simulator deliberately retains the current frozen-rate, at-most-one-jump-per-grid-bin approximation.

Protocol:

1. On exact CuAu-S, run the oracle-control step sweep.
2. Require the discretization floor to be well below the learned TV/error at the selected step count.
3. Start CuAu-M with `steps=128` because that matches current discrete practice, but also produce a generated-sample stability check at `256` and/or `512` using the same trained controller if computationally feasible.
4. If metrics materially change, increase production steps.

Do not hide the finite-step approximation under PT comparison.

---

## 27. Suggested training defaults

The safest first run is to preserve current optimizer/loss conventions but reduce CE-heavy batch work because each label now calls an external energy model.

Start with:

```text
loss             poisson
hidden           512
lr               3e-4
grad clip         10
log a clamp       [-20,20]
log Lambda clamp  [-20,20]
buffer            8
edge_samples      1
steps             128 initially
gamma             10 initially, then predeclared {10,20,40} pilot if needed
training seeds    0,1,2 minimum; 5 seeds preferable for headline N=64
```

For `batch`, `mb`, `inner`, and `iters`, do **not** blindly copy the current Ising values because its all-pair energy is cheap arithmetic while CE evaluation is external/CPU-heavy. Choose them based on a short throughput benchmark, then report the exact target-energy budget.

A practical start is:

```text
batch   256
mb      256
inner   4 to 8
iters   1000 to 5000, checkpoint by CE-call budget
```

but treat these as starting values, not final claims. Save checkpoints at cumulative CE-call thresholds so final comparisons can be drawn at matched budgets regardless of iteration structure.

---

## 28. Required result artifacts per run

Every run should write enough information that no expensive training must be repeated merely to calculate a new metric.

Checkpoint/result bundle should include:

```text
config/CLI arguments
Git commit of IASBS repo
Git commit of MetaDNS external repo
SHA256 of VASP/ECI files
Python/package versions
random seed
source state
controller state_dict
optimizer/scheduler state if resumable
training history
CE-call counter history
energy-evaluation wall time
training wall time
final generated binary samples
final sample energies
Qx,Qy,Qz,Qmax per sample
constraint-violation count
reference file identifier/hash used for evaluation
```

Suggested paths:

```text
ckpt/cuau/4x4x4/T500/seed0.pt
json/cuau/4x4x4/T500/seed0.json
samples/cuau/4x4x4/T500/seed0.npz
reference/cuau/4x4x4/pt_T500.npz
```

---

## 29. Reproducibility commands to expose from `cuau.py`

Implement a CLI whose final public interface is approximately:

```bash
# validate CE mapping and fixed-composition utilities
python -m structured_asbs.cuau validate \
  --size 4 4 4 \
  --vasp data/cuau/cuau_fcc_4x4x4_supercell.vasp \
  --eci data/cuau/CI_params_ECI_CuAu_Final_Submission.json

# exact small target (only after verified 16-site structure support)
python -m structured_asbs.cuau exact \
  --size 2 2 4 --temperature 500 --gamma 10 --steps-sweep 32 64 128 256 512

# headline IASBS
python -m structured_asbs.cuau train \
  --size 4 4 4 --temperature 500 --gamma 10 --steps 128 \
  --hidden 512 --edge-samples 1 --seed 0

# evaluate an existing checkpoint against PT reference
python -m structured_asbs.cuau eval \
  --ckpt ckpt/cuau/4x4x4/T500/seed0.pt \
  --reference reference/cuau/4x4x4/pt_T500.npz

# direct MCMC baseline
python -m structured_asbs.cuau_reference mcmc \
  --size 4 4 4 --temperature 500 --seed 0 \
  --budget 1000000

# PT gold reference
python -m structured_asbs.cuau_reference pt \
  --size 4 4 4 \
  --temperatures 350 400 450 500 550 600 640 680 720 760 820 900 1050 1200 1400 \
  --seed 0
```

The exact argument names can differ, but do not create separate ad-hoc scripts for every temperature. One CLI should parameterize size/T/seed/source/budget.

---

## 30. Unit and integration test checklist

`tests/test_cuau.py` should contain at least the following. Run them before any production job.

### A. Representation

```text
[ ] random_fixed_source returns exactly k ones
[ ] every legal sampled edge has x_i=1 and x_j=0
[ ] applying a legal edge preserves sum(x)=k
[ ] transpose_batch is involutive: g(g(x))=x
[ ] transpose_batch leaves x unchanged when terminal symbols at i,j are equal
```

### B. Johnson kernel

```text
[ ] sum_j C(k,j)C(n-k,j) kappa[j] == 1 to numerical tolerance
[ ] kappa[j] > 0 for gamma>0
[ ] direct distance agrees with common.binary_orbit_distance on random states
```

### C. Bridge

```text
[ ] direct bridge always has exactly k active sites
[ ] direct bridge empirical law agrees with explicit enumerated bridge on n<=10
[ ] t near 0 concentrates toward x0
[ ] t near 1 concentrates toward x1
```

### D. Terminal label

```text
[ ] edge-specific label matches common.binary_as_label on random n<=10 examples
[ ] edge-specific label matches current all-pairs fixed_ising terminal_labels when energy is Ising
[ ] terminal equal-symbol transposition gives Lambda=1
[ ] kernel-ratio sign/order matches current fixed_ising formula exactly
```

### E. Edge minibatching

```text
[ ] average sampled-edge Poisson loss agrees statistically with exact all-edge loss
[ ] edge sampler is uniform over legal edges (chi-square or frequency tolerance)
```

### F. Controlled simulation

```text
[ ] zero controller has empirical base endpoint Johnson-distance law matching kappa orbit masses
[ ] all generated states preserve k exactly
[ ] direct simulation on n<=10 agrees with enumerated simulation distribution within MC noise
```

### G. Energy backend

```text
[ ] 100-state MetaDNS parity
[ ] 100 swap Delta-E parity
[ ] binary index -> Cu/Au atom mapping verified
[ ] total-vs-per-atom energy unit verified
[ ] repeated evaluation deterministic
```

### H. Exact N=16 target

```text
[ ] pi sums to 1
[ ] no underflow after log-sum-exp stabilization
[ ] exact energy moments independently recomputed from saved E table
[ ] MCMC empirical law converges toward exact pi on long test run
```

### I. Order parameter

```text
[ ] ideal L10-x/y/z classify correctly
[ ] cubic rotation permutes Qx/Qy/Qz as expected
[ ] random equiatomic controls have no hard-coded orientation preference
```

### J. Cost counter

```text
[ ] batch of B energy configs increments counter by B, not by 1
[ ] cached E1 is not double-counted in inner loop
[ ] eval-only calls reported separately from train calls
```

---

## 31. Failure guards that should raise exceptions, not warnings

Add explicit runtime errors for:

```text
non-binary state values
incorrect fixed count
non-finite CE energy
non-finite log kappa
invalid/negative kappa
non-finite terminal labels before clipping
wrong number of lattice sites relative to CE structure
energy parity failure
order-parameter geometry validation failure
```

Clipping is a training stabilization device, not a license to hide `nan`, `inf`, mapping mistakes, or unit mistakes.

---

## 32. Experiment execution order and stop/go gates

### Phase 0 — freeze provenance

```text
[ ] clean IASBS git branch
[ ] clone/pin MetaDNS commit
[ ] copy/hash VASP + ECI
[ ] create CuAu environment without modifying original requirements.txt
```

**Go only if:** dependencies import reproducibly.

### Phase 1 — energy correctness

```text
[ ] implement CuAuEnergy wrapper
[ ] 100-state parity
[ ] swap Delta-E parity
[ ] atom/site ordering validation
```

**Go only if:** parity passes.

### Phase 2 — generic fixed-composition mechanics

```text
[ ] direct bridge
[ ] direct simulator
[ ] edge minibatching
[ ] label equivalence against existing Ising code on a tiny problem
```

**Go only if:** all test groups A-F pass.

### Phase 3 — exact CuAu-S

```text
[ ] verified compatible 16-site structure
[ ] exact canonical E/pi table
[ ] oracle discretization sweep
[ ] IASBS 3 seeds
[ ] DAM comparison
```

**Go only if:** IASBS exact-law error is sensible relative to current Ising results and oracle discretization floor; no constraint violations.

If the 16-site geometry cannot be verified with the same CE, skip rather than substitute a different energy silently.

### Phase 4 — characterize CuAu-M target independently

```text
[ ] PT pilot
[ ] Q metric verified
[ ] determine whether 500 K actually has separated orientation basins
[ ] determine q0 only from reference if thresholding is justified
[ ] confirm PT convergence diagnostics
```

**Go only if:** reference is defensible.

### Phase 5 — headline IASBS CuAu-M

```text
[ ] gamma pilot {10,20,40} if needed
[ ] freeze gamma/steps/training budget protocol
[ ] >=3 training seeds, preferably 5
[ ] save samples at matched CE-call checkpoints
```

### Phase 6 — direct canonical MCMC

```text
[ ] same CE
[ ] same T/composition
[ ] multiple starts/seeds
[ ] matched CE-call budget curve
```

### Phase 7 — controls

```text
[ ] 680 K
[ ] 1200 K
```

### Phase 8 — optional scale/prior-model additions

```text
[ ] 4x4x8 N=128
[ ] MetaDNS conditioned/reweighted comparison if ESS permits
```

---

## 33. Main result table design

### Exact small table

```text
CuAu-S, N=16, k=8, 500 K
```

| method | exact TV ↓ | KL ↓ | Hellinger ↓ | E-hist TV ↓ | violations | target-energy cost |
|---|---:|---:|---:|---:|---:|---:|
| IASBS | ... | ... | ... | ... | 0 | ... |
| DAM | ... | ... | ... | ... | 0 | ... |
| oracle finite-step | ... | - | - | - | 0 | - |

The value of this table is not scale; it establishes exact correctness on the same real CE family.

### Headline large table

```text
CuAu-M, N=64, k=32, 500 K
```

| method | E/N KS ↓ | E/N W1 (meV/atom) ↓ | Qmax KS ↓ | orient. mass TV ↓ | ordered-mass err ↓ | Cv/N err ↓ | violations | CE calls | wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PT split-half floor | ... | ... | ... | ... | ... | ... | 0 | reference | reference |
| IASBS | ... | ... | ... | ... | ... | ... | **0** | ... | ... |
| canonical swap MCMC | ... | ... | ... | ... | ... | ... | **0** | matched | ... |

If conditioned MetaDNS is valid, add it as a separate clearly labeled row and explain its conditioning/weighting.

---

## 34. Main figure design

Use one figure with four panels rather than many weak figures.

### Panel A — physical target

Show a `4x4x4` FCC CuAu configuration and small schematics of the three L10 orientation variants. This establishes the target as an alloy problem, not a binary-string toy.

### Panel B — PT reference mode structure

Plot the reference in a symmetry-aware 2D projection of `(Qx,Qy,Qz)`, for example

```math
u=Q_x-Q_y,
\qquad
v=(Q_x+Q_y-2Q_z)/\sqrt{3},
```

with density/free-energy-style contours if sample density supports it.

Do not call `-kBT log histogram` a precise free energy outside well-sampled bins; label it an empirical projected free-energy surface if used.

### Panel C — mode coverage vs CE calls

Plot x/y/z ordered-sector occupancy or orientation-mass TV versus cumulative target-energy evaluations for:

```text
IASBS
canonical swap MCMC
```

with the PT target/floor indicated.

If MCMC is trapped, this panel will make the global-sampling advantage immediately visible. If MCMC is not trapped, retain the plot and let the data show comparable mixing; do not force a trapping narrative.

### Panel D — distributional fidelity

Overlay PT / IASBS / MCMC empirical CDFs or density estimates for:

```text
E/N
Qmax
```

CDFs are preferable when possible because they align directly with KS and avoid kernel-density bandwidth choices.

---

## 35. Seed/statistical reporting

For IASBS headline results:

```text
minimum: 3 independent training seeds
preferred: 5 independent training seeds
```

Report mean +/- sample standard deviation across **training seeds** for method metrics.

For each trained sampler, generate enough terminal samples that sampling noise is smaller than training-seed variation where feasible. `50,000-200,000` is reasonable because IASBS generation needs no CE energy during simulation; only metric energies need to be computed afterwards and should be counted as evaluation-only.

For MCMC, do not treat correlated trajectory frames as IID. Use block/bootstrap or autocorrelation-aware errors, and multiple independent chains.

For PT reference, report independent-run consistency and split-half floors rather than pretending the finite reference is exact.

---

## 36. What counts as a convincing result

A very strong outcome is:

1. CuAu-S IASBS reproduces the exact canonical law with errors comparable to the current fixed-composition Ising accuracy, and beats/competes favorably with DAM at much lower nested target-evaluation cost.
2. At CuAu-M 500 K, IASBS `E/N` and `Qmax` distributions are close to the PT split-half floor.
3. IASBS recovers the reference x/y/z ordered-sector masses across training seeds.
4. Local MCMC at the same CE-call budget shows materially larger order-parameter/mode discrepancy or mixes more slowly.
5. At 1200 K, IASBS correctly produces the disordered law rather than retaining low-temperature order.
6. Zero fixed-composition violations in every IASBS/MCMC run.
7. Optional N=128 retains reasonable distributional accuracy and confirms scale.

A still-valid but weaker outcome is that MCMC mixes well at N=64. Then the evidence becomes **agreement with an established canonical sampler/reference on a realistic `10^18` constrained target**, not superiority through barrier crossing. Do not manufacture a negative baseline.

---

## 37. What not to do

Avoid these mistakes:

```text
DO NOT enumerate N=64 states.
DO NOT build a 2^64 lookup table.
DO NOT evaluate all 2016 terminal transpositions per N=64 training example.
DO NOT change the IASBS theorem for CuAu.
DO NOT use a different energy for reference and IASBS without parity testing.
DO NOT compare canonical IASBS JS numerically to published semi-grand MetaDNS JS as if targets match.
DO NOT use composition as the main mode variable: composition is fixed by construction.
DO NOT classify all high-temperature states into x/y/z and call that mode coverage.
DO NOT claim exact 1/3 x/y/z weights for the rectangular 4x4x8 cell.
DO NOT make PT and ordinary MCMC the same finite run and call one truth and one baseline.
DO NOT use a finite reference without a split-half/noise-floor diagnostic.
DO NOT edit the original requirements.txt and invalidate prior experiment reproducibility.
DO NOT report wall time without also reporting hardware and CE-evaluation counts.
DO NOT tune q0/gamma/temperature after inspecting which choice makes IASBS look best.
```

---

## 38. Minimal-LOC patch budget

A realistic implementation budget is:

| file | type | expected LOC |
|---|---|---:|
| `structured_asbs/cuau.py` | new | ~350-500 |
| `structured_asbs/cuau_reference.py` | new | ~150-250 if using library PT/MCMC wrappers |
| `tests/test_cuau.py` | new | ~150-250 |
| `requirements-cuau.txt` | new | <20 |
| `data/cuau/PROVENANCE.md` | new | <40 |
| `dam/discrete.py` | existing change | ~30-60 for CuAu-S branch |
| `common.py` | existing change | preferably 0 |
| `fixed_ising.py` | existing change | **0** |
| `dam/core.py` | existing change | **0** |

The exact count is less important than keeping the working original benchmark path untouched.

---

## 39. Suggested code organization inside `cuau.py`

Keep it linear and readable, following the repository's current plain-script style:

```text
constants / imports
CuAuEnergy / CountedEnergy
CuAuSpace
bridge-table builder
legal_mask / transpose_batch / edge sampler
sample_bridge_direct
simulate_direct
terminal_log_label_edge
order-parameter helpers
exact small-space helper
train()
evaluate()
validate()
CLI main()
```

Avoid an experiment framework/Hydra/config system; the current repo explicitly uses plain scripts and argparse.

---

## 40. Paper-writing interpretation once results exist

The clean narrative is:

1. **Same construction, real target.** The CuAu canonical ensemble is exactly the binary fixed-composition state space already covered by IASBS.
2. **No enumeration.** The Johnson reference/kernel and exact bridge remain low-dimensional while the target state count grows to `~1.8e18`.
3. **Real energy.** The target is a published DFT-fitted cluster expansion used in contemporary neural-sampler literature.
4. **Exact small validation.** A 16-site version permits complete canonical enumeration if the same CE geometry is verified.
5. **Independent large validation.** Canonical PT supplies a physically standard reference, with a reported reference noise floor.
6. **Meaningful modes.** Long-range L10 ordering coordinates test within-composition multimodality rather than trivial composition changes.
7. **Efficiency.** IASBS is compared with ordinary canonical swap MCMC at matched target-energy evaluations and wall time.
8. **Prior-work honesty.** MetaDNS/SEGAL/MAM establish CuAu as a modern neural-sampling benchmark, but their published semi-grand numbers are not misrepresented as canonical direct baselines.

A concise contribution sentence, if supported by the measurements, would be:

> On an equiatomic 64-site CuAu lattice with a published DFT-fitted cluster-expansion energy, the same binary fixed-composition IASBS construction scales from exact enumerable validation to a `1.8 x 10^18`-state canonical target, reproducing reference energy and L10-order distributions and symmetry-related ordering sectors while preserving composition exactly.

Only add the MCMC-superiority clause if the matched-budget data actually support it.

---

# Appendix A. Exact mapping from current Ising code to CuAu code

| current fixed-Ising concept | current implementation | CuAu replacement |
|---|---|---|
| state | enumeration index | direct `(B,N)` tensor |
| explicit binary state | `space.S[idx]` | `x` itself |
| state transition | `space.tgt[idx,e]` | swap tensor entries `i,j` |
| legal edges | precomputed `edge_i/edge_j` | boolean legal mask or sampled occupied/empty indices |
| base rate | `gamma / n_edges` | unchanged |
| controller | `SwapController` | unchanged |
| Johnson kernel | `C.binary_orbit_kernel` | unchanged |
| source distance | `space.dist0[idx]` | `k-(x0*x).sum(-1)` |
| bridge classes | `_bridge_tables` | copied unchanged mathematically |
| bridge result | map binary state through LUT | return binary state directly |
| terminal energy | Ising nearest-neighbor arithmetic | CE oracle |
| terminal labels | every pair | sampled legal intermediate edge only |
| loss | Poisson/Bregman | unchanged |
| replay | endpoint indices | `(X1,E1)` tensors |
| exact propagation | all states | CuAu-S only |
| large evaluation | impossible by enumeration | samples vs PT reference |

This table is the guiding principle: **change storage and energy evaluation, not the IASBS construction.**

---

# Appendix B. Acceptance criteria before a number enters the paper

A CuAu result is paper-eligible only if all relevant boxes are true:

```text
[ ] exact VASP/ECI provenance recorded and hashed
[ ] CE energy parity test passed
[ ] atom/site ordering test passed
[ ] total-energy units verified
[ ] direct bridge test passed
[ ] edge-minibatch equivalence test passed
[ ] zero composition violations
[ ] production hyperparameters frozen before final multi-seed runs
[ ] reference PT convergence gates passed
[ ] PT split-half noise floor computed
[ ] order-parameter geometry/rotation tests passed
[ ] all IASBS training seeds retained, not cherry-picked
[ ] local MCMC compared at matched target-energy budgets
[ ] CE-call accounting includes every training label evaluation
[ ] evaluation-only CE calls separated from training cost
[ ] published semi-grand baselines clearly labeled as non-target-matched context
[ ] code/result commit hashes recorded
```

---

# Appendix C. Evidence register

The external claims used to design this experiment can be checked at the following sources.

## C.1 MetaDNS (ICML 2026)

Official code:

```text
https://github.com/xiaochendu/metadns
```

Verified from its public README/metadata:

- CuAu support is via a 3D FCC cluster-expansion energy.
- `data/cuau/` contains a VASP supercell and ECI parameters.
- Official 4x4x4 command names:
  - `data/cuau/cuau_fcc_4x4x4_supercell.vasp`
  - `data/cuau/CI_params_ECI_CuAu_Final_Submission.json`
- Public sampling interface accepts CuAu temperatures including `500`, `680`, `1200 K`.
- `examples/cuau_4x4x4_benchmark.ipynb` is the CuAu reproduction notebook.
- Checkpoints/reference artifacts are hosted at Zenodo DOI `10.5281/zenodo.20301979`.
- The CuAu model uses CLEASE/iCET according to the repository's developer documentation.

Paper/OpenReview:

```text
https://openreview.net/forum?id=OY7Qe2ZSx9
```

Use its semi-grand CuAu metrics as context only unless rerun/conditioned to the same canonical target.

## C.2 SEGAL (2022)

```text
https://www.nature.com/articles/s41524-022-00736-4
https://github.com/learningmatter-mit/Segal
```

Relevant evidence:

- CuAu is treated as a realistic materials sampling problem.
- The energy is a DFT-trained CLEASE cluster expansion.
- 16-site CuAu is small enough for complete configuration enumeration in this literature.
- Larger CuAu thermodynamics and energy-evaluation costs are reported.
- The native ensemble is semi-grand canonical, so published numbers are context rather than direct canonical IASBS comparisons.

## C.3 Scaling Autoregressive Models for Lattice Thermodynamics (2026)

```text
https://arxiv.org/abs/2603.14695
```

Relevant evidence:

- CuAu scales from `2x2x4` to `4x4x4` and `4x4x8`.
- Exact thermodynamics are computed by enumerating all `2^16` configurations for the 16-site system.
- Larger systems use MCMC/metadynamics references.
- Reported observables include free energy, phase behavior, energy distributions, ESS, and heat capacity.
- `4x4x8` provides strong contemporary scale precedent.

## C.4 icet/mchammer canonical sampling and structure factors

```text
https://icet.materialsmodeling.org/dev/moduleref/ensembles.html
https://icet.materialsmodeling.org/moduleref/observers.html
```

Relevant evidence:

- `CanonicalEnsemble` fixes species counts and uses unlike-atom swaps with Metropolis acceptance.
- `StructureFactorObserver` is intended to monitor long-range order and documents Cu-Au q-points along x/y/z cubic axes.

## C.5 mchammer-pt

```text
https://pypi.org/project/mchammer-pt/0.27.1/
```

Relevant evidence:

- current public package exposes `CanonicalParallelTempering` for cluster expansions;
- requires Python >=3.11;
- supports icet-based canonical PT and round-trip/autocorrelation diagnostics.

## C.6 CLEASE environment compatibility

```text
https://clease.readthedocs.io/en/stable/releasenotes.html
```

Relevant evidence:

- CLEASE 1.1.0 pinned `numpy < 2`;
- CLEASE 1.2.0 removed that requirement.

This is why CuAu dependencies should not be merged casually into the repository's original NumPy-2.4.6 requirements file.

---

# Appendix D. Final recommended minimum experiment set

If compute/time is limited, **do exactly these and stop**:

1. **Energy/implementation validation** against MetaDNS CE.
2. **CuAu-S 16-site exact IASBS + DAM**, only if the same CE geometry can be verified cleanly.
3. **CuAu-M 4x4x4, 500 K**: IASBS (3-5 seeds), canonical PT reference, canonical swap-MCMC matched-budget baseline.
4. **CuAu-M 1200 K**: one high-temperature control with the same evaluation suite.
5. **CuAu-M 680 K** if budget permits, because it probes the transition regime and aligns with prior CuAu settings.
6. Add **CuAu-L 4x4x8** only after the above is stable.

The single experiment that matters most is item 3. If it shows correct `E/N`, `Qmax`, ordered mass, and x/y/z mode coverage near the PT noise floor, while local swap MCMC is worse at matched CE cost, it is already strong evidence for the paper.

---

# Appendix E. Final pre-run checklist

Copy this into the issue/experiment log and check each item literally.

```text
DATA
[ ] MetaDNS commit pinned
[ ] VASP hash saved
[ ] ECI hash saved
[ ] site count equals requested N

ENERGY
[ ] 100 upstream energy parity cases pass
[ ] 100 Delta-E parity cases pass
[ ] total eV unit confirmed
[ ] fixed-composition chemical-potential term excluded

IASBS MECHANICS
[ ] direct bridge exact-small test pass
[ ] direct controlled simulation fixed-count pass
[ ] label equals current fixed-Ising formula on a toy energy
[ ] one-edge loss unbiasedness test pass
[ ] Lambda=1 for terminal equal-symbol transposition

PHYSICS METRICS
[ ] L10 ideal x/y/z tests pass
[ ] rotation permutation test pass
[ ] q0 chosen only from PT if threshold used

REFERENCE
[ ] >=4 independent PT starts
[ ] round trips sufficient
[ ] independent runs agree
[ ] ESS/autocorrelation acceptable
[ ] PT split-half floor saved

FINAL RUNS
[ ] hyperparameters frozen
[ ] all seeds run
[ ] no seed discarded
[ ] IASBS samples saved
[ ] MCMC matched CE budgets run
[ ] 0 constraint violations
[ ] train CE calls separated from eval CE calls
[ ] wall time + hardware recorded

PAPER
[ ] exact-small table if valid
[ ] 4x4x4 500 K main table
[ ] PT floor row
[ ] mode/order figure
[ ] 1200 K control
[ ] semi-grand prior-work numbers described only as context
```

---

## Bottom line

The lowest-risk, highest-evidence path is **not** to port a materials framework into IASBS. Keep IASBS exactly recognizable as the existing fixed-composition experiment:

```text
existing SwapController
+ existing Johnson kernel
+ existing exact bridge mathematics
+ direct binary states instead of enumeration indices
+ one uniformly sampled legal edge instead of all terminal transpositions
+ published CuAu CE energy
+ independent canonical PT/MCMC evaluation
```

That combination minimizes code changes while producing an experiment that is materially stronger than another synthetic constrained benchmark.
