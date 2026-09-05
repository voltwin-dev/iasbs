Experiment Execution Plan

Adjoint Schrödinger Bridge Sampling on Structured State Spaces via Markov Semigroup Intertwining

Purpose. This is an implementation-grade plan for the first complete experimental package for the paper. It is deliberately written so that a researcher can implement the experiments without reconstructing any mathematics from the manuscript.

Recommended philosophy. Follow the compact style of the R-ASBS repository: one standalone script per experiment, a tiny shared utility file, no Hydra/Lightning/framework machinery unless a later experiment genuinely needs it.

Official R-ASBS repository:
https://github.com/mattiamosso/Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-via-Adjoint-Schrodinger-Bridges

Paper:
https://arxiv.org/abs/2608.25838

────────

0. What we are trying to prove experimentally

The paper is not primarily claiming “a better neural sampler on every benchmark.” The experiments should answer four much narrower questions.

1. Bijective structured discrete case: does the permutation/swap construction recover a known constrained target while preserving the exact composition constraint?
2. Non-bijective structured discrete case: does the occupation-space intertwiner train and sample correctly, including the stochastic estimator of the terminal operator?
3. Sphere: does the scalar Killing-field readout recover an exact spherical target and improve over the published R-ASBS implementation on the same benchmark?
4. Stiefel: does the canonical-metric Killing-field construction work on St(4,2), using the corrected heat-kernel clock, and does it outperform Extended R-ASBS on matched targets?

These four experiments exercise every mathematically important branch of the proposal.

Do not begin with CuAu, Cantor HEA, dilute Potts, ZRP, earthquake, cosmic-ray, or Wahba. Add one large scientific application only after the four core experiments work.

────────

1. Minimal repository layout

```text
structured_asbs/
├── README.md
├── requirements.txt
├── common.py
├── fixed_ising.py
├── occupation.py
├── sphere.py
├── stiefel.py
├── tests_math.py
└── figures.py
```

Target size:

|File            |Approximate target|
|----------------|-----------------:|
|`common.py`     |200–300 LOC       |
|`fixed_ising.py`|300–450 LOC       |
|`occupation.py` |400–600 LOC       |
|`sphere.py`     |300–450 LOC       |
|`stiefel.py`    |450–700 LOC       |
|`tests_math.py` |300–500 LOC       |

The R-ASBS repository itself explicitly uses one standalone script per experiment. Its Stiefel script is only about 230 nonblank LOC. We should preserve that spirit.

Suggested dependencies:

```text
torch>=2.4
numpy>=2.0
scipy>=1.13
matplotlib>=3.9
tqdm>=4.66
```

Optional for exact small-state tests:

```text
networkx
```

────────

2. Rules before training anything

Every mathematical component gets a deterministic unit test before a neural network is trained.

Run:

```bash
python tests_math.py
```

and require all tests to pass.

The tests should check:

```text
[1] Binary single-swap generator rows sum to zero.
[2] Binary reference preserves particle number.
[3] Small binary orbit kernel equals full matrix exponential.
[4] Occupation commutator identity.
[5] Occupation terminal intertwining F P = P A.
[6] Concentrated-source multinomial kernel ratio.
[7] Sphere Killing-frame resolution of identity.
[8] Sphere tangent controller.
[9] Stiefel canonical Killing-frame resolution.
[10] Stiefel collapsed readout equals explicit generator sum.
[11] Stiefel reference first moment uses exp(-3 r), NOT exp(-6 r).
[12] Stiefel heat-kernel helper calls S^3 kernel at r/2.
```

No experiment proceeds until these pass.

────────

3. Shared conventions

3.1 Time and noise clock

Use continuous time (t\in[0,1]).

For constant noise (\sigma),

$$
r(t,s)=\frac12\int_t^s \sigma_u^2,du
=\frac{\sigma^2}{2}(s-t).
$$

For the direct sphere heat kernel, use the sphere time r.

For the St(4,2) spin-cover kernel, each S^3 factor uses r/2.

This distinction is mandatory.

```python
def heat_clock(t0, t1, sigma):
    """r = 1/2 ∫ sigma^2 dt for constant sigma."""
    return 0.5 * sigma * sigma * (t1 - t0)

def sphere_factor_clock_for_stiefel(t0, t1, sigma):
    """Each SU(2) ≅ S^3 factor runs at half the SO(4) quotient heat time."""
    return 0.5 * heat_clock(t0, t1, sigma)
```

────────

4. Experiment A — fixed-magnetization Ising

Goal: validate the bijective discrete construction

4.1 State space

Use a 2D periodic square lattice with (n=L^2) binary variables and exactly (k=n/2) active sites:

$$ \Omega_{n,k}

\left{
x\in{0,1}^n:
\sum_i x_i=k
\right}.
$$

Convert to Ising spins with

$$
s_i=2x_i-1.
$$

Energy:

$$ E(x)

-J\sum_{\langle i,j\rangle}s_i s_j.
$$

Start small:

```text
L = 4  -> n = 16, k = 8  (enumerable)
L = 8  -> n = 64, k = 32
L = 16 -> n = 256, k = 128
```

The L=4 case is the mathematical correctness benchmark.

────────

4.2 Reference process

One jump swaps an occupied site and an empty site.

For (i\in S), (j\notin S),

$$
S^{ij}=S-{i}+{j}.
$$

Rate:

$$ r_t(S^{ij},S)

\frac{\gamma_t}{k(n-k)}.
$$

Total rate is exactly (\gamma_t).

Verified implementation:

```python
import torch

def swap_state(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    x: (..., n), binary tensor.
    Requires x[..., i] = 1, x[..., j] = 0 for the selected sample.
    """
    y = x.clone()
    y[..., i] = 0
    y[..., j] = 1
    return y

def sample_uniform_swap(x: torch.Tensor, generator=None):
    """
    Single state x of shape (n,).
    Samples uniformly from k(n-k) legal ordered occupied->empty swaps.
    """
    occ = torch.nonzero(x > 0.5, as_tuple=False).flatten()
    emp = torch.nonzero(x < 0.5, as_tuple=False).flatten()
    ii = occ[torch.randint(len(occ), (), generator=generator)]
    jj = emp[torch.randint(len(emp), (), generator=generator)]
    return int(ii), int(jj)
```

Invariant check:

```python
def assert_fixed_count(x, k):
    assert torch.all(x.sum(dim=-1) == k)
```

────────

4.3 Exact small-state reference

For L=4, enumerate all (\binom{16}{8}=12870) states. This is small enough to compute the exact Gibbs target.

$$ \pi(x)

\frac{e^{-E(x)/\tau}}
{\sum_{z\in\Omega_{n,k}}e^{-E(z)/\tau}}.
$$

Use exact enumeration to score:

• total variation distance;
• energy histogram;
• nearest-neighbor correlation;
• magnetization count (must be exact by construction).

Code:

```python
import itertools
import numpy as np

def enumerate_fixed_count(n, k):
    states = []
    for occ in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=np.int8)
        x[list(occ)] = 1
        states.append(x)
    return np.stack(states)

def exact_boltzmann_prob(energies, tau):
    logw = -np.asarray(energies, dtype=np.float64) / tau
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()
```

────────

4.4 Binary orbit kernel for AS labels

Fix source (S_0). Define

$$ j

k-|S\cap S_0|.
$$

The number of states at distance (j) is

$$ N_j

\binom{k}{j}\binom{n-k}{j}.
$$

The distance chain has birth/death rates

$$ B_{j,j+1}

\frac{(k-j)(n-k-j)}
{k(n-k)},
$$

$$ B_{j,j-1}

\frac{j^2}
{k(n-k)}.
$$

The diagonal is minus the row sum.

For integrated rate

$$ \Gamma_{0,1}

\int_0^1\gamma_t,dt,
$$

compute

$$ q

e_0^\top
\exp(\Gamma_{0,1}B).
$$

Then

$$ p^{\mathrm{base}}_{1|0}(S\mid S_0)

\frac{q_j}{N_j}.
$$

Implementation:

```python
import math
import numpy as np
from scipy.linalg import expm

def binary_orbit_generator(n, k):
    J = min(k, n - k)
    B = np.zeros((J + 1, J + 1), dtype=np.float64)

    for j in range(J + 1):
        if j + 1 <= J:
            up = ((k - j) * (n - k - j)) / (k * (n - k))
            B[j, j + 1] = up
        if j - 1 >= 0:
            down = (j * j) / (k * (n - k))
            B[j, j - 1] = down

    B[np.diag_indices_from(B)] = -B.sum(axis=1)
    return B

def binary_orbit_kernel(n, k, Gamma):
    B = binary_orbit_generator(n, k)
    q = np.zeros(len(B), dtype=np.float64)
    q[0] = 1.0
    q = q @ expm(Gamma * B)

    kappa = np.empty_like(q)
    for j in range(len(q)):
        Nj = math.comb(k, j) * math.comb(n - k, j)
        kappa[j] = q[j] / Nj
    return kappa

def binary_orbit_distance(x0, x):
    # x0 and x are binary arrays with equal fixed count k
    k = int(np.sum(x0))
    overlap = int(np.sum(np.asarray(x0) * np.asarray(x)))
    return k - overlap
```

Unit test

At tiny n,k, build the full generator on all states, exponentiate it, and compare every entry against binary_orbit_kernel.

Tolerance:

```text
max absolute error < 1e-10
```

────────

4.5 AS terminal label

For a queried neighboring move represented by permutation (g),

$$ \Lambda_g(X_1)

\exp\left(
-\frac{E(gX_1)-E(X_1)}{\tau}
\right)
\frac{
p^{\mathrm{base}}{1|0}(X_1\mid X_0)
}{
p^{\mathrm{base}}{1|0}(gX_1\mid X_0)
}.
$$

Implementation skeleton:

```python
def binary_as_label(x1, x1_g, x0, energy_fn, tau, kappa):
    e0 = energy_fn(x1)
    e1 = energy_fn(x1_g)

    j0 = binary_orbit_distance(x0, x1)
    j1 = binary_orbit_distance(x0, x1_g)

    log_energy_ratio = -(e1 - e0) / tau
    kernel_ratio = kappa[j0] / kappa[j1]

    return np.exp(log_energy_ratio) * kernel_ratio
```

Never evaluate the full state-space transition matrix during training.

────────

4.6 Controller parameterization

The true controlled rate is positive:

$$ u_t^*(y,x)

r_t(y,x)
\frac{\varphi_t(y)}{\varphi_t(x)}.
$$

Use a network that predicts a log multiplier:

$$
a_\theta(t,x,i,j)\in\mathbb R,
$$

$$ \widehat u_\theta(y,x)

r_t(y,x)\exp(a_\theta(t,x,i,j)).
$$

This guarantees positivity.

A compact architecture:

```python
class SwapController(torch.nn.Module):
    def __init__(self, n, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n + 3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, t, x, i, j):
        # x: (B,n), i,j: (B,)
        B, n = x.shape
        ii = i.float().unsqueeze(-1) / max(n - 1, 1)
        jj = j.float().unsqueeze(-1) / max(n - 1, 1)
        inp = torch.cat([x.float(), t[:, None], ii, jj], dim=-1)
        return self.net(inp).squeeze(-1)
```

For large lattices later replace the MLP with a small convolutional architecture. Do not do that until L=4 and L=8 work.

────────

4.7 Pass/fail gates

Gate A0 — math

All binary kernel/unit tests pass.

Gate A1 — exact tiny target

At L=4:

```text
TV(sample distribution, exact target) <= 0.05
```

with enough terminal samples.

Gate A2 — exact constraint

For all generated samples:

```text
sum_i x_i == k
```

zero violations.

Gate A3 — nontrivial scale

At L=16, energy and pair-correlation observables agree with long-run Kawasaki/PT reference within reference Monte Carlo uncertainty.

────────

5. Experiment B — occupation / inclusion process

Goal: validate the genuinely non-bijective intertwiner

This is the most important discrete experiment.

5.1 State space

$$ \mathcal X^{\mathrm{occ}}_{m,N}

\left{
\eta\in\mathbb N_0^m:
\sum_i\eta_i=N
\right}.
$$

Use:

```text
toy:     m=4, N=4          full enumeration
small:   m=N=32
medium:  m=N=128
large:   m=N=1000
```

────────

5.2 Inclusion-process target with exact samples

Choose (d>0).

Target:

$$ \pi(\eta)

\frac{1}{Z_{m,N}}
\prod_{i=1}^m
\frac{\Gamma(\eta_i+d)}
{\eta_i!\Gamma(d)}.
$$

Normalization:

$$ Z_{m,N}

\frac{\Gamma(N+dm)}
{N!\Gamma(dm)}.
$$

This is Dirichlet-multinomial.

Exact iid sampler:

1. sample (p\sim\mathrm{Dirichlet}(d,\ldots,d));
2. sample (\eta\sim\mathrm{Multinomial}(N,p)).

```python
def sample_inclusion_exact(batch, m, N, d, device="cpu"):
    alpha = torch.full((m,), float(d), device=device)
    p = torch.distributions.Dirichlet(alpha).sample((batch,))
    eta = torch.distributions.Multinomial(
        total_count=N, probs=p
    ).sample()
    return eta.to(torch.long)
```

This gives exact iid target samples, so we do not need MCMC ground truth.

────────

5.3 Reference process

A particle at mode (i) moves to (j\neq i) at rate

$$ r_t(\eta-e_i+e_j,\eta)

\frac{\gamma_t}{m-1}\eta_i.
$$

Total rate:

$$ \sum_{i\neq j} \frac{\gamma_t}{m-1}\eta_i

\gamma_t N.
$$

This identity should be checked in code.

A direct Gillespie reference step:

```python
def sample_occupation_reference_jump(eta, generator=None):
    """
    eta: shape (m,), integer occupancies, total N.
    Samples source i proportional to eta_i, then destination j uniformly
    among the other m-1 modes.
    """
    m = eta.numel()
    probs = eta.float() / eta.sum()
    i = torch.multinomial(probs, 1, generator=generator).item()

    r = torch.randint(m - 1, (), generator=generator).item()
    j = r if r < i else r + 1

    out = eta.clone()
    out[i] -= 1
    out[j] += 1
    return out, i, j
```

────────

5.4 Verified operator algebra

Define

$$ (E_{ab}f)(\eta)

\eta_b f(\eta-e_b+e_a)
$$

for (a\neq b), and

$$
(E_{aa}f)(\eta)=\eta_a f(\eta).
$$

Let

$$
F_{ji}=E_{ji}-E_{ii}.
$$

Then

$$ (F_{ji}f)(\eta)

\eta_i
\left[
f(\eta-e_i+e_j)-f(\eta)
\right].
$$

Reference generator:

$$ L

\frac{1}{m-1}
\sum_{i\neq j}F_{ji}.
$$

The commutators are

$$ [L,F_{ji}]

\frac{1}{m-1}
\sum_{a=1}^m(E_{ja}-E_{ia}),
$$

and

$$ \left[ L, \sum_{a=1}^m(E_{ja}-E_{ia}) \right]

\frac{m}{m-1}
\sum_{a=1}^m(E_{ja}-E_{ia}).
$$

Therefore

$$ A_{ji}^{(t)}

F_{ji}

c_t
\sum_{a=1}^m(E_{ja}-E_{ia}),
$$

with

$$ c_t

\frac{
1-\exp\left[-\frac{m}{m-1}\Gamma_{t,1}\right]
}{m}.
$$

The controlled edge rate is

$$ u_t^*(\eta-e_i+e_j,\eta)

\frac{\gamma_t}{m-1}
\left[
\eta_i+
\mathbb E
\left[
\frac{A_{ji}^{(t)}f_1(X_1)}
{f_1(X_1)}
\mid X_t=\eta
\right]
\right].
$$

────────

5.5 Concentrated Dirac source

Choose

$$
\eta_0=Ne_c.
$$

For integrated rate

$$
\Gamma=\int_0^t\gamma_s,ds,
$$

define

$$ \rho

\exp\left(
-\frac{m}{m-1}\Gamma
\right),
$$

$$ \omega

\frac{1-\rho}{m}.
$$

One-particle probabilities:

$$
q_c=\omega+\rho,
$$

$$
q_\ell=\omega,\qquad \ell\neq c.
$$

Hence the reference endpoint is multinomial.

Verified code:

```python
def occupation_single_particle_probs(m, Gamma, source_mode=0):
    rho = np.exp(-m * Gamma / (m - 1))
    omega = (1.0 - rho) / m
    q = np.full(m, omega, dtype=np.float64)
    q[source_mode] += rho
    return q

def occupation_neighbor_kernel_ratio(xi, a, b, q):
    """
    ratio p(xi) / p(xi - e_b + e_a)
    Requires a != b and xi[b] > 0.
    """
    assert a != b
    assert xi[b] > 0
    return ((xi[a] + 1.0) / xi[b]) * (q[b] / q[a])
```

Unit test this against direct multinomial probabilities.

────────

5.6 Stable energy evaluation

For the inclusion process,

$$ E(\eta)

-\tau
\sum_i
\left[
\log\Gamma(\eta_i+d)
-\log\Gamma(\eta_i+1)
-\log\Gamma(d)
\right].
$$

Use torch.lgamma.

```python
def inclusion_energy(eta, tau, d):
    x = eta.float()
    logw = torch.lgamma(x + d) - torch.lgamma(x + 1.0) - torch.lgamma(
        torch.as_tensor(d, dtype=x.dtype, device=x.device)
    )
    return -tau * logw.sum(dim=-1)
```

Do not literally evaluate Gamma functions.

────────

5.7 Terminal-ratio helper

For a legal transfer (a\to b), define

$$ R_{ba}(\xi)

\frac{f_1(\xi-e_a+e_b)}
{f_1(\xi)}.
$$

Under Dirac-source AS:

$$ R_{ba}(\xi)

\exp\left(
-\frac{
E(\xi-e_a+e_b)-E(\xi)
}{\tau}
\right)
\frac{
p^{\mathrm{base}}(\xi\mid\eta_0)
}{
p^{\mathrm{base}}(\xi-e_a+e_b\mid\eta_0)
}.
$$

Code:

```python
def transfer_state(xi, a, b):
    # source a -> destination b
    assert a != b
    assert xi[a] > 0
    out = xi.clone()
    out[a] -= 1
    out[b] += 1
    return out

def occupation_f_ratio(xi, a, b, q, energy_fn, tau):
    """
    R_{b a}(xi) = f1(xi - e_a + e_b) / f1(xi).
    """
    if a == b:
        return 1.0
    if int(xi[a]) <= 0:
        raise ValueError("illegal transfer")

    xi2 = transfer_state(xi, a, b)

    # kernel ratio p(xi)/p(xi2)
    kr = occupation_neighbor_kernel_ratio(
        np.asarray(xi.cpu()), b, a, q
    )

    dE = float(energy_fn(xi2) - energy_fn(xi))
    return np.exp(-dE / tau) * kr
```

Notice the orientation carefully:
occupation_neighbor_kernel_ratio(xi, destination=b, source=a, q=q).

────────

5.8 Full terminal label

For queried edge (i\to j),

$$ \Lambda_{ji}(\xi)

\xi_i(R_{ji}-1)

c_t
\sum_{a:\xi_a>0}
\xi_a
\left(
R_{ja}-R_{ia}
\right).
$$

Exact code for small m:

```python
def occupation_terminal_label_full(
    xi, i, j, q, energy_fn, tau, c_t
):
    assert i != j

    if int(xi[i]) > 0:
        Rji = occupation_f_ratio(xi, i, j, q, energy_fn, tau)
        first = float(xi[i]) * (Rji - 1.0)
    else:
        first = 0.0

    corr = 0.0
    for a in range(len(xi)):
        na = int(xi[a])
        if na == 0:
            continue

        Rja = 1.0 if a == j else occupation_f_ratio(
            xi, a, j, q, energy_fn, tau
        )
        Ria = 1.0 if a == i else occupation_f_ratio(
            xi, a, i, q, energy_fn, tau
        )
        corr += na * (Rja - Ria)

    return first - c_t * corr
```

This implementation is for verification and small runs, not the final large-scale estimator.

────────

5.9 Unbiased one-sample correction estimator

Let (A\sim\mathrm{Unif}{1,\ldots,m}).

Because

$$ m c_t

1-\rho_{t,1},
$$

an unbiased estimator is

$$ \widehat\Lambda_{ji}

\xi_i(R_{ji}-1)

(1-\rho_{t,1})
\xi_A(R_{jA}-R_{iA}).
$$

Implementation:

```python
def occupation_terminal_label_uniform_one_sample(
    xi, i, j, q, energy_fn, tau, rho_t1, rng=np.random
):
    if int(xi[i]) > 0:
        Rji = occupation_f_ratio(xi, i, j, q, energy_fn, tau)
        first = float(xi[i]) * (Rji - 1.0)
    else:
        first = 0.0

    m = len(xi)
    a = int(rng.randint(m))
    na = int(xi[a])

    if na == 0:
        correction = 0.0
    else:
        Rja = 1.0 if a == j else occupation_f_ratio(
            xi, a, j, q, energy_fn, tau
        )
        Ria = 1.0 if a == i else occupation_f_ratio(
            xi, a, i, q, energy_fn, tau
        )
        correction = (1.0 - rho_t1) * na * (Rja - Ria)

    return first - correction
```

────────

5.10 Recommended variance-reduced estimator

Uniform mode sampling can have variance growing with (m).

Instead sample

$$
A\sim\Pr(A=a)=\frac{\xi_a}{N}.
$$

Then

$$ \sum_a \xi_a(R_{ja}-R_{ia})

N,
\mathbb E_{A\sim \xi/N}
[R_{jA}-R_{iA}].
$$

Therefore

$$ \boxed{ \widehat\Lambda^{\mathrm{occ}}_{ji}

\xi_i(R_{ji}-1)

c_t N(R_{jA}-R_{iA})
}
$$

is unbiased.

Recommended code:

```python
def occupation_terminal_label_occupancy_sample(
    xi, i, j, q, energy_fn, tau, c_t
):
    if int(xi[i]) > 0:
        Rji = occupation_f_ratio(xi, i, j, q, energy_fn, tau)
        first = float(xi[i]) * (Rji - 1.0)
    else:
        first = 0.0

    probs = xi.float() / xi.sum()
    a = int(torch.multinomial(probs, 1).item())

    Rja = 1.0 if a == j else occupation_f_ratio(
        xi, a, j, q, energy_fn, tau
    )
    Ria = 1.0 if a == i else occupation_f_ratio(
        xi, a, i, q, energy_fn, tau
    )

    correction = c_t * float(xi.sum()) * (Rja - Ria)
    return first - correction
```

Ablate:

```text
full sum
uniform one-sample
occupancy-weighted one-sample
occupancy-weighted 4-sample
```

This ablation is scientifically important.

────────

5.11 Positivity of learned rates

The label itself can be negative. The final controlled rate cannot.

The identity implies

$$ \eta_i + \mathbb E[\Lambda_{ji}\mid X_t=\eta]

\eta_i
\frac{\varphi_t(\eta-e_i+e_j)}
{\varphi_t(\eta)}

> 1. 

$$

Do not let a raw neural output directly become a rate.

Recommended parameterization:

$$ u_\theta(i\to j\mid t,\eta)

\frac{\gamma_t}{m-1}
\eta_i
\exp(a_\theta(t,\eta,i,j)).
$$

Train the log-ratio or positive multiplier in a way consistent with the regression target.

For the first implementation, one may regress the additive quantity and clamp only for simulation, but this is not the preferred final method because clamping changes the learned operator. The clean version predicts a positive multiplier.

────────

5.12 Exact metrics

Against iid exact samples report:

• one-site occupancy histogram (P(\eta_i=n));
• maximum occupancy (P(\max_i\eta_i=n));
• condensate fraction (\max_i\eta_i/N);
• energy distribution;
• MMD on normalized occupancy vectors;
• sorted-occupancy Wasserstein distance;
• exact constraint residual.

Do not claim full-state total variation at m=N=1000; it is not estimable from reasonable sample counts.

────────

5.13 Pass/fail gates

B0

All operator and kernel-ratio unit tests pass.

B1 — full enumeration

At m=N=4, compare exact terminal state probabilities:

```text
TV <= 0.05
```

B2 — exact target sampling comparison

At m=N=32 and 128:

```text
1D occupancy KS <= 0.05
max-occupancy Wasserstein error <= 5% of N
constraint violations = 0
```

B3 — estimator variance

Occupancy-weighted estimator should have lower or equal empirical variance than uniform one-sample over the chosen scale sweep.

B4 — scale

Demonstrate successful sampling at m=N=1000.

────────

6. Experiment C — sphere S^2

Goal: exact continuous validation and direct R-ASBS comparison

This is the cleanest continuous experiment.

6.1 Target

Use exactly the R-ASBS analytic benchmark:

$$ E(x)

6(1-x_3^2),
\qquad
x\in\mathbb S^2.
$$

Thus

$$
\pi(x)
\propto
e^{6x_3^2}.
$$

The target is symmetric under (x_3\mapsto-x_3), so

$$
\Pr_\pi(x_3>0)=\frac12.
$$

The marginal of (z=x_3\in[-1,1]) is

$$ p(z)

\frac{e^{6z^2}}
{\int_{-1}^1e^{6s^2}ds}.
$$

This gives a one-dimensional exact distributional benchmark.

────────

6.2 Published R-ASBS number to beat

The R-ASBS paper reports:

|Metric                   |Exact target|R-ASBS reported|Desired result |
|-------------------------|-----------:|--------------:|--------------:|
|Northern hemisphere mass |0.500       |**0.438**      |closer to 0.500|
|Absolute hemisphere error|0           |**0.062**      |< 0.062        |

The paper also reports that geometric Langevin MCMC placed 86% of particles in one basin in their finite-run comparison.

Important: hemisphere mass alone is too weak. We must additionally report the exact x3 marginal discrepancy.

Recommended headline:

```text
KS(x3 marginal, exact)
```

R-ASBS did not publish this value. Re-run their public script and compute it under the same sample budget.

────────

6.3 Sphere tangent projection

For (x\in\mathbb S^{d-1}),

$$ P_x

I-xx^\top.
$$

```python
def sphere_project_tangent(x, v):
    return v - x * (x * v).sum(dim=-1, keepdim=True)
```

Test:

```python
v_tan = sphere_project_tangent(x, v)
assert torch.max(torch.abs((x * v_tan).sum(-1))) < 1e-6
```

────────

6.4 Killing-field collapsed readout

For terminal ambient gradient

$$
G(Y)=\nabla \log f_1(Y),
$$

the exact controller score is

$$ \boxed{ \nabla_{\mathbb S^{d-1}}\log\varphi_t(x)

\mathbb E \left[ (x^\top Y)G(Y)

Y(x^\top G(Y))
\mid X_t=x
\right].
}
$$

Verified implementation:

```python
def sphere_terminal_readout(x, y, G):
    """
    x: (B,d) intermediate point
    y: (B,d) terminal point
    G: (B,d) ambient gradient representative of log f1 at y

    Returns the per-sample terminal vector whose conditional expectation
    is grad_S log phi_t(x).
    """
    xy = (x * y).sum(dim=-1, keepdim=True)
    xG = (x * G).sum(dim=-1, keepdim=True)
    out = xy * G - y * xG
    return out
```

It is tangent at x:

```python
out = sphere_terminal_readout(x, y, G)
assert torch.max(torch.abs((x * out).sum(-1))) < 1e-6
```

Do not multiply this readout by an additional Ricci damping factor.

────────

6.5 AS terminal gradient for Dirac source

For source (x_0),

$$
f_1(y)
\propto
\frac{e^{-E(y)/\tau}}
{p^{\mathrm{base}}_{1|0}(y\mid x_0)}.
$$

Therefore

$$ G(y)

-\frac1\tau\nabla E(y)

\nabla_y
\log p^{\mathrm{base}}_{1|0}(y\mid x_0).
$$

For the benchmark energy:

$$
E(x)=6(1-x_3^2),
$$

an ambient gradient is

$$ \nabla E(x)

(0,0,-12x_3).
$$

```python
def sphere_energy(x):
    return 6.0 * (1.0 - x[..., 2] ** 2)

def sphere_energy_grad(x):
    g = torch.zeros_like(x)
    g[..., 2] = -12.0 * x[..., 2]
    return g
```

────────

6.6 S^2 heat-kernel score

For the first implementation, avoid relying on a short-time approximation. Compute the spherical heat kernel with a truncated spectral series.

On the unit (S^2),

$$ p_r(\cos\theta)

\frac{1}{4\pi}
\sum_{\ell=0}^\infty
(2\ell+1)e^{-\ell(\ell+1)r}
P_\ell(\cos\theta).
$$

Differentiate with respect to (c=\cos\theta):

$$ \partial_c p_r(c)

\frac{1}{4\pi}
\sum_{\ell=1}^\infty
(2\ell+1)e^{-\ell(\ell+1)r}
P_\ell’(c).
$$

Because (c=x_0^\top y),

$$ \nabla_y\log p_r

\frac{\partial_c p_r(c)}{p_r(c)}
P_y x_0.
$$

Numerically stable SciPy reference implementation:

```python
import numpy as np
from scipy.special import eval_legendre

def s2_heat_kernel_and_dc(c, r, Lmax=200):
    c = np.asarray(c, dtype=np.float64)
    p = np.zeros_like(c)
    dp = np.zeros_like(c)

    for ell in range(Lmax + 1):
        coeff = (2 * ell + 1) * np.exp(-ell * (ell + 1) * r)
        P = eval_legendre(ell, c)
        p += coeff * P

        if ell >= 1:
            # P'_ell(c) = ell (P_{ell-1}(c) - c P_ell(c))/(1-c^2)
            Pm1 = eval_legendre(ell - 1, c)
            den = np.maximum(1.0 - c * c, 1e-14)
            dP = ell * (Pm1 - c * P) / den
            dp += coeff * dP

    p /= (4.0 * np.pi)
    dp /= (4.0 * np.pi)
    return p, dp

def s2_reference_score(y, x0, r, Lmax=200):
    """
    numpy implementation, y/x0 shape (3,)
    """
    c = float(np.dot(x0, y))
    p, dp = s2_heat_kernel_and_dc(np.array(c), r, Lmax=Lmax)
    coeff = float(dp / p)
    tangent = x0 - c * y
    return coeff * tangent
```

At c≈±1, use either a stabilized recurrence or autograd through a Torch Legendre recurrence. Do not trust the naïve derivative formula exactly at the endpoints.

Convergence test

Double Lmax until:

```text
relative kernel change < 1e-10
relative score change  < 1e-8
```

at a grid of r and theta values used during training.

────────

6.7 Reference simulator

The simplest compact implementation is intrinsic Euler followed by normalization:

$$ x_{n+1}

\mathrm{normalize} \left( x_n + \sigma P_{x_n}\Delta W_n

\frac{d-1}{2}\sigma^2x_n\Delta t
\right).
$$

But normalization plus an Itô correction mixes two discretizations.

For clean experiments, use an exponential-map increment on the sphere:

Given tangent (v\in T_xS^{d-1}),

$$ \operatorname{Exp}_x(v)

\cos(|v|)x
+
\sin(|v|)
\frac{v}{|v|}.
$$

```python
def sphere_exp(x, v, eps=1e-12):
    nv = torch.linalg.norm(v, dim=-1, keepdim=True)
    direction = v / torch.clamp(nv, min=eps)
    out = torch.cos(nv) * x + torch.sin(nv) * direction

    small = nv < 1e-7
    approx = x + v
    approx = approx / torch.linalg.norm(approx, dim=-1, keepdim=True)
    return torch.where(small, approx, out)
```

A single Gaussian exponential increment is constraint-preserving but is still a numerical SDE discretization, not an exact finite-time heat-kernel sample. Perform a step-size sweep.

────────

6.8 Exact target metrics

Hemisphere

```python
north_mass = (samples[:, 2] > 0).float().mean().item()
north_error = abs(north_mass - 0.5)
```

KS for x3

Exact CDF can be tabulated numerically:

```python
from scipy.integrate import cumulative_trapezoid

def exact_s2_z_cdf_grid(num=20001):
    z = np.linspace(-1.0, 1.0, num)
    w = np.exp(6.0 * z * z)
    cdf = np.concatenate([[0.0], cumulative_trapezoid(w, z)])
    cdf /= cdf[-1]
    return z, cdf
```

Use interpolation to compute one-sample KS.

Constraint

$$
\max_i ||x_i|_2-1|.
$$

Expected numerical value should be around floating-point/retraction tolerance.

────────

6.9 Sphere pass/fail gates

C0 — exactness

Killing readout and heat-kernel tests pass.

C1 — beat published hemisphere result

Published R-ASBS error:

```text
|0.438 - 0.5| = 0.062
```

Require:

```text
mean hemisphere error across 5 seeds < 0.03
```

A stronger target is <0.02.

C2 — full distribution

Require:

```text
KS(x3, exact) < 0.05
```

and compare directly with a re-run of public R-ASBS under the same budget.

C3 — discretization

At increasing integration steps:

```text
32, 64, 128, 256, 512
```

our error should decrease or plateau near statistical error.

────────

7. Experiment D — Stiefel St(4,2)

Goal: validate the canonical Killing construction and corrected kernel

7.1 Manifold

$$ \mathrm{St}(4,2)

\left{
X\in\mathbb R^{4\times2}:
X^\top X=I_2
\right}.
$$

Use the canonical metric in the paper.

────────

7.2 Constraint helper

```python
def stiefel_constraint_error(X):
    # X: (..., n, p)
    p = X.shape[-1]
    I = torch.eye(p, dtype=X.dtype, device=X.device)
    gram = X.transpose(-1, -2) @ X
    return torch.linalg.norm(gram - I, dim=(-2, -1))
```

────────

7.3 Canonical Killing readout

Let

$$ G(Y)

\nabla_{\mathrm{ambient}}\log f_1(Y).
$$

Then

$$ \boxed{ \nabla^c\log\varphi_t(X)

\mathbb E \left[ \left( G(Y)Y^\top

YG(Y)^\top
\right)X
\mid X_t=X
\right].
}
$$

Do not form a (4\times4) matrix unnecessarily.

Equivalent efficient computation:

$$ G(Y)(Y^\top X)

Y(G(Y)^\top X).
$$

```python
def stiefel_terminal_readout(X, Y, G):
    """
    X,Y,G: (B,n,p)
    G is an ambient covector representative of grad log f1 at Y.
    """
    return G @ (Y.transpose(-1, -2) @ X) \
         - Y @ (G.transpose(-1, -2) @ X)
```

This output is tangent at X.

Test:

```python
R = stiefel_terminal_readout(X, Y, G)
skew_test = X.transpose(-1, -2) @ R + R.transpose(-1, -2) @ X
assert skew_test.abs().max() < 1e-6
```

────────

7.4 Explicit Killing-sum unit test

For each (i<j), let

$$
\Omega_{ij}=E_{ij}-E_{ji}.
$$

Define

$$
V_{ij}(X)=\Omega_{ij}X.
$$

The explicit sum should equal the collapsed expression.

```python
def omega(n, i, j, dtype=torch.float64):
    O = torch.zeros((n, n), dtype=dtype)
    O[i, j] = 1.0
    O[j, i] = -1.0
    return O

def stiefel_readout_explicit(X, Y, G):
    # single sample, shape (n,p)
    n = X.shape[0]
    out = torch.zeros_like(X)
    for i in range(n):
        for j in range(i + 1, n):
            O = omega(n, i, j, dtype=X.dtype).to(X.device)
            VX = O @ X
            VY = O @ Y
            coeff = torch.sum(VY * G)
            out = out + VX * coeff
    return out

def stiefel_readout_collapsed(X, Y, G):
    return G @ (Y.T @ X) - Y @ (G.T @ X)
```

Test:

```text
max |explicit - collapsed| < 1e-10 in float64
```

────────

7.5 Corrected St(4,2) heat-kernel clock

This is the correction discovered in the audit.

Define

$$ r

\frac12\int_0^1\sigma_t^2dt.
$$

The SO(4) group heat kernel lifts to two SU(2) ≅ S^3 factors. Under the normalization used in the manuscript, each factor runs at time (r/2).

Correct:

$$
p_{r/2}(\theta_p)
p_{r/2}(\theta_q).
$$

Incorrect:

$$
p_r(\theta_p)p_r(\theta_q).
$$

Implementation rule:

```python
def stiefel_spin_factor_time(r):
    return 0.5 * r
```

and every call inside the Stiefel kernel uses:

```python
rs = stiefel_spin_factor_time(r)
kp = s3_heat_kernel(theta_p, rs)
kq = s3_heat_kernel(theta_q, rs)
```

Direct S^3 experiments still use r, not r/2.

────────

7.6 Mandatory first-moment test

For the stated St(4,2) reference:

$$ \boxed{ \mathbb E[X_t\mid X_0]

e^{-3r}X_0.
}
$$

The incorrect product-time convention produces (e^{-6r}X_0).

Monte Carlo validation:

```python
def expected_st42_mean_factor(r):
    return np.exp(-3.0 * r)
```

For each chosen r, simulate at least 1e5 cheap reference trajectories at a sufficiently fine step size and check

```text
|| sample_mean - exp(-3r) X0 ||_F
```

against Monte Carlo standard error plus discretization error.

This test is non-negotiable.

────────

7.7 S^3 heat kernel

Use the winding sum from the manuscript:

$$ p_s(\theta)

\frac{e^s}
{(4\pi s)^{3/2}}
\frac1{\sin\theta}
\sum_{k\in\mathbb Z}
(\theta+2\pi k)
\exp\left[
-\frac{(\theta+2\pi k)^2}{4s}
\right].
$$

Implement in log-safe form and truncate symmetrically.

```python
def s3_heat_kernel(theta, s, K=8):
    """
    Reference implementation for moderate s.
    theta can be torch tensor in (0, pi).
    """
    ks = torch.arange(-K, K + 1, device=theta.device, dtype=theta.dtype)
    z = theta[..., None] + 2.0 * torch.pi * ks
    terms = z * torch.exp(-(z * z) / (4.0 * s))
    series = terms.sum(dim=-1)

    denom = torch.sin(theta)
    pref = torch.exp(torch.as_tensor(s, dtype=theta.dtype, device=theta.device))
    pref = pref / ((4.0 * torch.pi * s) ** 1.5)

    return pref * series / denom
```

At (\theta\approx0,\pi), direct evaluation is numerically unstable because numerator and denominator vanish. Implement limiting formulas or use an alternate spectral representation near the endpoints.

Convergence criterion:

```text
K -> K+2 changes log kernel by < 1e-10
K -> K+2 changes score by < 1e-8
```

for the entire training clock range.

────────

7.8 Fiber quadrature

The quotient kernel is a one-dimensional integral over the SO(2) fiber.

Use fixed Gauss-Legendre quadrature rather than adaptive Python loops so the operation can be vectorized and differentiated.

Prototype:

```python
from numpy.polynomial.legendre import leggauss

def periodic_legendre_nodes(nq=64, device="cpu", dtype=torch.float64):
    z, w = leggauss(nq)
    phi = np.pi * (z + 1.0)      # [0, 2pi]
    weight = np.pi * w
    return (
        torch.tensor(phi, device=device, dtype=dtype),
        torch.tensor(weight, device=device, dtype=dtype),
    )
```

The exact spin-lift map from a 4x4 rotation to the quaternion pair should be unit-tested independently.

Do not optimize this kernel before the first-moment and normalization checks pass.

────────

7.9 R-ASBS Stiefel benchmark

Their benchmark:

$$ E(X)

\operatorname{tr}(X^\top H X),
$$

with

$$
\operatorname{spec}(H)={1,2,5,8},
\qquad n=4,\quad p=2.
$$

Exact asymptotes:

High temperature:

$$ \lim_{\beta\to0}\mathbb E[E]

\frac{p}{n}\operatorname{tr}(H)

\frac24(1+2+5+8)

8. 

$$

Low temperature:

$$ \lim_{\beta\to\infty}\mathbb E[E]

\lambda_1+\lambda_2

3. 

$$

The public R-ASBS script sweeps approximately:

```text
beta =
0.001, 0.01, 0.01, 0.1, 0.5, 1.3, 2, 5, 7, 10,
20, 50, 100, 200, 1000, 10000, 1000000
```

and uses 5000 samples per beta.

Important limitation of the published benchmark

This energy is right-orthogonally invariant:

$$
E(XQ)=E(X),\qquad Q\in O(2).
$$

Therefore it is effectively a Grassmannian target. Reproduce it for an apples-to-apples comparison, but add a frame-sensitive target.

────────

7.10 Frame-sensitive Stiefel target

Use

$$ E_{\mathrm{frame}}(X)

\operatorname{tr}(X^\top H X)

\lambda\operatorname{tr}(C^\top X).
$$

Choose deterministic C and lambda, recorded in the script.

Example:

```python
H = torch.diag(torch.tensor([1., 2., 5., 8.]))
C = torch.tensor([
    [ 0.7, -0.2],
    [ 0.1,  0.8],
    [-0.4,  0.3],
    [ 0.2, -0.5],
])
lam = 1.0

def stiefel_frame_energy(X):
    quad = torch.einsum("...np,nm,...mp->...", X, H, X)
    linear = torch.einsum("np,...np->...", C, X)
    return quad - lam * linear
```

This genuinely distinguishes frames spanning the same subspace.

For St(4,2), obtain high-quality ground truth using long constrained MCMC or direct numerical quadrature over a dense parameterization only for this tiny dimension. The exact method used must be documented.

────────

8. R-ASBS continuous results to beat

8.1 Sphere benchmark

From the R-ASBS paper:

|Task                 |Metric                   |Reference|R-ASBS      |Ours target            |
|---------------------|-------------------------|--------:|-----------:|----------------------:|
|`S^2`, (E=6(1-x_3^2))|northern hemisphere mass |0.500    |**0.438**   |closer to 0.500        |
|same                 |absolute hemisphere error|0        |**0.062**   |< 0.03 preferred       |
|same                 |`KS(x3, exact)`          |0        |not reported|report and minimize    |
|same                 |max norm residual        |0        |not reported|near floating precision|

The first headline number to beat is therefore 0.438 vs 0.500.

Do not overfit to this single number. A sampler could obtain exactly 50/50 while having the wrong distribution inside each hemisphere.

────────

8.2 Stiefel benchmark

The R-ASBS paper primarily reports the expected-energy curve rather than a table of exact numerical values.

Hard reference values:

|Regime          |Exact reference|
|----------------|--------------:|
|(\beta\to0)     |(E[E]=8)       |
|(\beta\to\infty)|(E[E]=3)       |

For a fair comparison, run their official alg2_stiefel.m or faithfully port it and extract the full expected-energy curve at the same beta grid.

The comparison table in our final paper should become:

|beta |exact/reference MCMC|Extended R-ASBS|ours    |
|----:|-------------------:|--------------:|-------:|
|0.001|TBD                 |measured rerun |measured|
|0.01 |TBD                 |measured rerun |measured|
|0.1  |TBD                 |measured rerun |measured|
|0.5  |TBD                 |measured rerun |measured|
|1.3  |TBD                 |measured rerun |measured|
|2    |TBD                 |measured rerun |measured|
|5    |TBD                 |measured rerun |measured|
|10   |TBD                 |measured rerun |measured|
|50   |TBD                 |measured rerun |measured|
|100+ |approaches 3        |measured rerun |measured|

Do not invent values from their plotted curve. Re-run the compact public script and save them.

Primary metric should be error against an independent target, not merely “lower energy.”

────────

8.3 Wahba published/released numbers — optional appendix benchmark

The manuscript currently contains these R-ASBS values:

TLS-gap statistic

|Outlier ratio|R-ASBS relative gap (%)|
|------------:|----------------------:|
|25%          |(0.143\pm0.146)        |
|50%          |(0.012\pm0.017)        |
|75%          |(-0.025\pm0.025)       |
|85%          |(-0.007\pm0.012)       |
|90%          |(-0.044\pm0.052)       |
|95%          |(-0.058\pm0.029)       |

Rotation error of best selected sample

|Outlier ratio|R-ASBS rotation error (deg)|
|------------:|--------------------------:|
|25%          |(0.806\pm0.246)            |
|50%          |(0.733\pm0.258)            |
|75%          |(1.119\pm0.316)            |
|85%          |(0.892\pm0.688)            |
|90%          |(1.253\pm0.429)            |
|95%          |(2.245\pm1.111)            |

Released CSV diagnostics at (\beta=1), (N_s=10^4), as summarized in the proposal:

```text
median sample rotation error: about 15.4–15.7 degrees
fraction samples within 5 degrees: about 2.9–3.3%
```

Treat Wahba as an optimization benchmark, not a clean distributional benchmark.

The released CSV column named trueTLS is the objective evaluated at the synthetic ground-truth rotation in the public implementation, not a certificate of the global TLS optimum. Negative relative gaps therefore do not mean the sampler solved a certified optimum better than an optimizer.

This is why Wahba should not be one of the first four experiments.

────────

9. Fair R-ASBS comparison protocol

9.1 Clone and freeze

Record exact commit:

```bash
git clone https://github.com/mattiamosso/Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-via-Adjoint-Schrodinger-Bridges.git
cd Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-via-Adjoint-Schrodinger-Bridges
git rev-parse HEAD
```

Store the commit hash in our README.

────────

9.2 Match budgets

At minimum match:

• number of training iterations;
• number of terminal energy/gradient evaluations;
• number of integration steps;
• number of generated samples;
• network parameter count within a reasonable factor.

Report both:

```text
energy/gradient oracle calls
wall-clock
```

The first is the scientific budget; the second is implementation-dependent.

────────

9.3 Step-count sweep

For sphere and Stiefel:

```text
N_steps = 32, 64, 128, 256, 512
```

Reason:

• our controller identity is exact;
• numerical integration still introduces discretization error;
• R-ASBS also has geometric-surrogate error.

A strong empirical signature is:

```text
ours: error decreases as step size decreases
R-ASBS: possible nonzero surrogate floor
```

Do not claim such a floor before measuring it.

────────

10. Shared neural-training skeleton

The exact distribution used for matching depends on AS/ASBS details. Keep the experiment scripts explicit rather than hiding it inside abstractions.

A minimal generic training skeleton:

```python
for step in range(num_updates):
    # 1. sample current controlled endpoints
    x0 = sample_source(batch_size)

    with torch.no_grad():
        x1, path_info = simulate_controlled(
            controller, x0, return_endpoint=True
        )

    # 2. sample t
    t = torch.rand(batch_size, device=device)

    # 3. sample/reference-bridge intermediate x_t
    #    The exact routine is experiment-specific.
    xt = sample_reference_bridge(x0, x1, t)

    # 4. construct exact terminal label from the theorem
    label = terminal_label(
        t=t, xt=xt, x0=x0, x1=x1
    )

    # 5. regression
    pred = controller_matching_output(t, xt)
    loss = ((pred - label.detach()) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This pseudocode is deliberately incomplete: reference bridge sampling is state-space-specific and must be implemented correctly.

Do not replace it casually with a forward reference sample unless the objective being implemented mathematically permits that change.

────────

11. Reference bridge issue: resolve before final training code

For a Markov reference,

$$
p^{\mathrm{base}}{t|0,1}(z\mid x_0,x_1)
\propto
p^{\mathrm{base}}{t|0}(z\mid x_0)
p^{\mathrm{base}}_{1|t}(x_1\mid z).
$$

Each experiment must have a bridge sampler or another mathematically equivalent training construction.

Sphere

Use the same bridge construction as the corresponding exact-manifold AS/R-ASBS setup if reproducing their matching objective.

Stiefel

This is the most delicate implementation point. The quotient kernel evaluator gives bridge density factors, but not automatically an iid bridge sampler. Start with importance/resampling for St(4,2) because the state is tiny, and only optimize later.

Binary fixed-composition

At small L, exact bridge sampling is possible using enumeration. Use this first to validate learning. For larger systems, exploit the orbit/reference structure or use a bridge CTMC construction.

Occupation

Because the reference is independent-particle motion conditioned on occupancy, exploit particle labels internally for bridge simulation if convenient, then collapse to occupation counts. The observable state remains unlabeled.

This is one of the few places where compact code must not take precedence over correctness.

────────

12. Suggested development order

Phase 1 — no neural networks

Implement only tests_math.py.

Pass:

```text
binary orbit kernel
occupation intertwiner
sphere readout
Stiefel readout
Stiefel first moment
```

Estimated code: <500 LOC.

────────

Phase 2 — exact tiny discrete experiments

Implement:

```text
fixed_ising.py   L=4
occupation.py    m=N=4
```

Use enumeration wherever necessary.

Goal:

```text
learned sampler reproduces exact state probabilities
```

Do not scale before this works.

────────

Phase 3 — sphere

Implement the analytic S^2 target.

Goal:

```text
hemisphere mass closer to 0.5 than 0.438
KS to exact x3 marginal small
```

This is the first direct R-ASBS result.

────────

Phase 4 — Stiefel

Before neural training:

```text
[ ] corrected r/2 factor
[ ] kernel normalization test
[ ] first-moment exp(-3r) test
[ ] score finite-difference test
[ ] collapsed Killing readout test
```

Then reproduce the published trace-energy benchmark.

Then add the frame-sensitive target.

────────

Phase 5 — scale occupation

Scale:

```text
m=N=32
128
1000
```

This should become the strongest discrete result if variance remains controlled.

────────

Phase 6 — optional scientific application

Only now choose one:

```text
CuAu
Cantor HEA
dilute Potts
```

Do not add all three unless needed by reviewers.

────────

13. Figures for the first paper

Keep the main paper compact.

Figure 1 — method cartoon

Four columns:

```text
cyclic DASBS -> fixed composition -> occupation -> manifold
```

Show corresponding operator:

```text
translation
permutation
non-bijective F/A
Killing derivative
```

────────

Figure 2 — discrete correctness

Panels:

```text
A: fixed-Ising exact energy distribution
B: fixed-Ising exact state-probability scatter on tiny instance
C: occupation P(eta_i=n), ours vs exact
D: occupation max-occupancy distribution, ours vs exact
```

────────

Figure 3 — occupation scale/variance

Panels:

```text
A: error vs m=N
B: label variance: full / uniform-1 / occupancy-1 / occupancy-4
C: wall clock or oracle calls
D: exact constraint residual
```

────────

Figure 4 — sphere vs R-ASBS

Panels:

```text
A: x3 exact density + samples
B: KS vs integration steps
C: hemisphere error vs integration steps
D: constraint residual
```

Include horizontal reference:

```text
R-ASBS hemisphere error = 0.062
```

────────

Figure 5 — Stiefel

Panels:

```text
A: expected trace energy vs beta
B: error vs integration steps
C: constraint residual
D: frame-sensitive target comparison
```

────────

14. What counts as paper success?

Minimum viable paper result

All of:

```text
[PASS] tiny binary exact target
[PASS] tiny occupation exact target
[PASS] occupation m=N>=128 against iid exact reference
[PASS] sphere beats R-ASBS 0.438 hemisphere result and has good KS
[PASS] Stiefel corrected-kernel implementation reproduces expected energy behavior
[PASS] zero structural constraint violations up to numerical precision
```

This is enough to justify writing a serious experimental paper draft.

────────

Strong result

Additionally:

```text
[PASS] occupation m=N=1000
[PASS] variance-reduced one-sample estimator works
[PASS] sphere distributional metric substantially beats rerun R-ASBS
[PASS] Stiefel matched-target metric beats rerun Extended R-ASBS
[PASS] frame-sensitive Stiefel target works
```

────────

Headline result

Add one real scientific target, preferably after the mathematical benchmarks are stable.

────────

15. Failure interpretation

Do not simply tune forever. Diagnose mathematically.

If fixed Ising fails on tiny exact instance

Likely:

```text
wrong bridge sampling
wrong orientation of kernel ratio
controller loss mismatch
rate positivity issue
```

It is almost certainly not a capacity problem.

────────

If occupation full-sum works but stochastic estimator fails

The theorem is fine; variance is the problem.

Test:

```text
uniform sampling
occupancy-weighted sampling
4/8-sample averages
control variates
```

────────

If sphere hemisphere works but KS is bad

The network may learn basin weights without correct within-basin density.

Increase:

```text
controller capacity
training convergence
integration resolution
```

and verify heat-kernel score.

────────

If Stiefel fails the first-moment test

Stop immediately.

The error is in:

```text
reference simulator
spin-cover conversion
r vs r/2 clock
quadrature normalization
```

Do not train.

────────

If Stiefel trace-energy works but frame-sensitive target fails

That is especially informative: the published benchmark is Grassmannian-like and may be hiding an orientation/frame failure.

Debug the vertical part of the canonical-metric readout and the ambient-gradient/covector convention.

────────

16. Final checklist before any reported number enters the paper

```text
[ ] exact git commit recorded
[ ] random seeds recorded
[ ] all math unit tests pass
[ ] all constraint checks pass
[ ] exact target/reference provenance stated
[ ] energy-call budget stated
[ ] integration-step count stated
[ ] sample count stated
[ ] network parameter count stated
[ ] 5 independent seeds minimum
[ ] mean and standard deviation reported
[ ] no value copied visually from a plot when source code can reproduce it
[ ] R-ASBS public scripts rerun for all new metrics
[ ] Stiefel uses p_{r/2} inside each S^3 spin factor
[ ] direct sphere code still uses p_r
```

────────

17. Recommended immediate coding task

The first repository milestone should not train a sampler.

Implement tests_math.py containing:

```text
test_binary_orbit_kernel()
test_occupation_commutator()
test_occupation_intertwining()
test_occupation_multinomial_ratio()
test_sphere_killing_identity()
test_sphere_readout_tangent()
test_stiefel_killing_identity()
test_stiefel_collapsed_readout()
test_stiefel_reference_first_moment()
test_stiefel_spin_clock()
```

Once these pass, implement fixed_ising.py on L=4.

This gives us a clean sequence:

```text
mathematics -> tiny exact experiment -> continuous exact experiment -> scale
```

rather than debugging mathematics, neural training, and large systems simultaneously.

────────

18. Continuous comparison scoreboard

Keep this table at the top of README.md and fill it as experiments complete.

|Benchmark              |Metric                     |Exact/reference|R-ASBS      |Ours|
|-----------------------|---------------------------|--------------:|-----------:|---:|
|S2 analytic            |north mass                 |0.500          |**0.438**   |TBD |
|S2 analytic            |absolute north error       |0              |**0.062**   |TBD |
|S2 analytic            |KS(x3)                     |0              |rerun needed|TBD |
|S2 analytic            |max norm residual          |0              |rerun needed|TBD |
|St(4,2)                |E as beta -> 0             |8              |curve       |TBD |
|St(4,2)                |E as beta -> infinity      |3              |curve       |TBD |
|St(4,2)                |finite-beta reference error|0              |rerun needed|TBD |
|St(4,2)                |orthogonality residual     |0              |rerun needed|TBD |
|frame-sensitive Stiefel|reference discrepancy      |0              |new rerun   |TBD |

Optional Wahba table:

|Outliers|R-ASBS TLS gap %|R-ASBS best rotation error deg|
|-------:|---------------:|-----------------------------:|
|25%     |0.143 +/- 0.146 |0.806 +/- 0.246               |
|50%     |0.012 +/- 0.017 |0.733 +/- 0.258               |
|75%     |-0.025 +/- 0.025|1.119 +/- 0.316               |
|85%     |-0.007 +/- 0.012|0.892 +/- 0.688               |
|90%     |-0.044 +/- 0.052|1.253 +/- 0.429               |
|95%     |-0.058 +/- 0.029|2.245 +/- 1.111               |

────────

19. Final recommendation

Build only the four core scripts first:

```text
fixed_ising.py
occupation.py
sphere.py
stiefel.py
```

The likely paper narrative is then:

> A single semigroup-intertwining principle yields practical adjoint readouts on structured spaces. We verify it in a bijective discrete constrained system, a genuinely non-bijective occupation process, and two homogeneous manifolds. Every experiment preserves its structural constraint by construction, the discrete non-bijective case agrees with exact iid ground truth at enormous state-space size, and the continuous constructions can be compared directly against R-ASBS on its own public benchmarks.

That is much cleaner than attempting seven unrelated application benchmarks.