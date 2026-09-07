# Minimal, mathematically faithful DAM implementation plan for the IASBS repository

**Status:** implementation plan only.  No DAM code has been added yet.

**Primary source:** Oswin So, Brian Karrer, Chuchu Fan, Ricky T. Q. Chen, and Guan-Horng Liu, *Discrete Adjoint Matching*, arXiv:2602.07132v2 / ICLR 2026.  The equations referenced below are the paper's Eq. (6), (11)--(14), and the CTMC Radon--Nikodym identity in Eq. (40).

**Repository target:** add one compact `dam/` subfolder beside the existing `rasbs/` folder, while reusing the existing state spaces, reference bridges, controllers, metrics, checkpoint helpers, and target evaluators in `structured_asbs/` and `common.py`.

---

## 0. Executive decision

Implement **generic on-policy DAM for the existing Dirac-source discrete benchmarks first**:

- fixed-composition Ising, `L=4` and `L=5`;
- occupation, `m=N=4`;
- scalable occupation, `m=N=32,128,1000` if computationally feasible.

This fills the `DAM` cells in Section 6 without adding a DAM-specific neural architecture.

The DAM implementation should use:

1. the **same base CTMC** as IASBS;
2. the **same controller network family** as IASBS;
3. the **same exact reference bridge sampler** already in the repository;
4. the paper's **importance-weighted discrete-adjoint estimator**;
5. the paper's **generalized-KL matching objective**;
6. the **full CTMC path likelihood ratio** for our non-masked CTMCs.

The last point is non-negotiable.  The paper's masked-diffusion simplification cannot be used for the Ising swap process or occupation process.

### Important correction to the current experimental prose

For the clean DAM baseline, use the **same Dirac source as the existing IASBS run**.

Do **not** make the primary DAM baseline share the non-Dirac source with `IASBS (non-Dirac)` unless a valid SB terminal factor `f_1` has already been obtained from a corrector.  DAM, as presented in the paper, learns the controlled CTMC for a prescribed terminal loss `g`; it does not by itself solve the extra two-sided Schrödinger corrector required by our general non-Dirac ASBS problem.

Therefore the clean three-column interpretation is:

- `IASBS`: Dirac source, structured terminal adjoint;
- `IASBS (non-Dirac)`: broad source, learned ASBS corrector + structured terminal adjoint;
- `DAM`: Dirac source, generic discrete-adjoint estimator.

This is the mathematically clean comparison for the current paper.  If a source-matched non-Dirac DAM experiment is desired later, use the **same frozen `f_1` corrector** as the non-Dirac IASBS run and label it explicitly as `DAM + shared corrector`; do not pretend DAM learned that corrector itself.

---

# 1. Exact mathematical object DAM must learn in our experiments

Let the reference CTMC have off-diagonal rate

$$
r_t(y,x), \qquad y\neq x.
$$

Let the terminal factor be

$$
f_1(z)=e^{-g(z)}.
$$

Define

$$
\varphi_t(x)
=
\mathbb E_{p^{\mathrm{base}}(\cdot\mid X_t=x)}
[f_1(X_1)].
$$

Then the exact optimal controlled rate is

$$
u_t^*(y,x)
=
r_t(y,x)
\frac{\varphi_t(y)}{\varphi_t(x)}.
$$

For the Dirac-source experiments already implemented in the repository,

$$
f_1(z)
\propto
\frac{\mu(z)}{p^{\mathrm{base}}_{1|0}(z\mid x_0)}.
$$

Therefore choosing

$$
g(z)=-\log f_1(z)
$$

makes DAM target **exactly the same terminal law and exact continuous-time controlled rate as Dirac IASBS**.

This is critical.  Do **not** set `g = E/tau` unless the reference terminal law happens to be constant.  For our current reference processes it generally is not.

The existing repository already contains the correct Dirac-source `logf1` for:

- `FixedIsingSpace.logf1`;
- `OccupationSpace.logf1`.

For scalable occupation, `logf1` can be evaluated analytically without enumeration; see Section 7 below.

---

# 2. DAM equations to implement

## 2.1 Analytic discrete adjoint

The paper's analytic discrete adjoint can be written in our notation as

$$
\widetilde m_t(y;X_1)
=
\sum_z
p^{\mathrm{base}}_{1|t}(z\mid y)
\exp[-g(z)+g(X_1)].
$$

Equivalently,

$$
\widetilde m_t(y;X_1)
=
\mathbb E_{Z\sim p^{\mathrm{base}}_{1|t}(\cdot\mid y)}
\left[
\frac{f_1(Z)}{f_1(X_1)}
\right].
$$

Under the optimal conditional law, its conditional expectation is the optimal rate multiplier

$$
\mathbb E_{p^*}
[\widetilde m_t(y;X_1)\mid X_t=x]
=
\frac{\varphi_t(y)}{\varphi_t(x)}.
$$

## 2.2 Practical importance-weighted DAM estimator

For a current stop-gradient model `\bar u`, the paper estimates the multiplier using model rollouts and importance weighting.

For our implementation, sample independently:

- one trajectory `\zeta^y` from time `t`, state `y`, under `p^{\bar u}` with endpoint `Z`;
- `K` trajectories `\zeta^{x,k}` from time `t`, state `x`, under `p^{\bar u}` with endpoints `X_1^{(k)}`.

Let

$$
W(\zeta;s,t)
=
\frac{dP^{\mathrm{base}}_{t:1}(\zeta\mid X_t=s)}
{dP^{\bar u}_{t:1}(\zeta\mid X_t=s)}.
$$

Then use the pathwise implementation of the paper's estimator

$$
\widehat m_t(y,x)
=
\frac{
W(\zeta^y;y,t)f_1(Z)
}{
\frac1K\sum_{k=1}^K
W(\zeta^{x,k};x,t)f_1(X_1^{(k)})
}.
$$

This is the quantity that multiplies the base rate:

$$
\widehat u_t^*(y,x)
=
r_t(y,x)\widehat m_t(y,x).
$$

Use **one numerator rollout** by default, matching the paper's practical estimator.  A `K_num > 1` option may be implemented only as an ablation; it must not silently become the default.

### Why path weights are needed in our repository

The paper writes the estimator using terminal conditional-density ratios such as

$$
\frac{p^{\mathrm{base}}_{1|t}(Z\mid y)}
{p^{\bar u}_{1|t}(Z\mid y)}.
$$

Those marginal model probabilities are not tractable for our learned CTMCs.  However, for a sampled CTMC path the Radon--Nikodym derivative is tractable, and

$$
\mathbb E_{P^{\bar u}}
[W(\zeta)h(X_1)]
=
\mathbb E_{P^{\mathrm{base}}}
[h(X_1)].
$$

Hence path importance sampling is the correct generic implementation.

---

# 3. The CTMC path likelihood ratio: implement this exactly

For a path under the model with jump times `\tau_j` and jumps

$$
X_{\tau_j^-}\to X_{\tau_j},
$$

define the total escape rates

$$
R_{\bar u}(t,x)
=
\sum_{y\neq x}\bar u_t(y,x),
$$

and

$$
R_{\mathrm{base}}(t,x)
=
\sum_{y\neq x}r_t(y,x).
$$

The log likelihood ratio needed by DAM is

$$
\log W
=
\int_t^1
\left[
R_{\bar u}(s,X_s)-R_{\mathrm{base}}(s,X_s)
\right]ds
+
\sum_j
\log
\frac{r_{\tau_j}(X_{\tau_j},X_{\tau_j^-})}
{\bar u_{\tau_j}(X_{\tau_j},X_{\tau_j^-})}.
$$

Our model is parameterized as a multiplicative control

$$
\bar u_t(y,x)
=
r_t(y,x)e^{a_{\bar\theta}(t,x,y)}.
$$

Therefore every jump contributes simply

$$
\log
\frac{r_t(y,x)}{\bar u_t(y,x)}
=
-a_{\bar\theta}(t,x,y).
$$

This makes the implementation compact.

### Never use the masked-DAM shortcut here

In the paper's masked diffusion specialization, model and base share the same total escape rate and the integrated escape-rate term cancels.  That is why their path weight can reduce to a product of jump-probability ratios.

Our Ising and occupation controllers change the total escape rate.  Therefore

$$
\int_t^1
[R_{\bar u}-R_{\mathrm{base}}]ds
$$

must be retained.

Dropping it would make the estimator mathematically wrong.

---

# 4. Time dependence: exact piecewise-constant CTMC, not an uncontrolled approximation

The existing controllers take continuous `t`, but the experiments already have a fixed integration grid with `steps` points.

For DAM, define the learned rate to be piecewise constant in time:

$$
a_\theta(t,x,y)
:=
a_\theta(t_s,x,y),
\qquad
 t\in[t_s,t_{s+1}),
$$

where

$$
t_s=\frac{s}{\texttt{steps}}.
$$

Within each time bin, rates remain state-dependent but are constant in the clock variable.  This permits an **exact Gillespie simulation of the piecewise-constant CTMC** and an exact evaluation of the integral in the path weight.

Do not use the existing one-jump-per-bin sampler to compute DAM importance weights.  That sampler is useful as the existing common numerical evaluator, but its path law is not the CTMC path law appearing in the DAM derivation.

---

# 5. Minimal repository layout

Add only:

```text
dam/
    __init__.py
    core.py
    discrete.py
    tests_math.py
```

Target new-code budget:

| file | purpose | target LOC |
|---|---|---:|
| `dam/core.py` | CTMC rollout + path RN weight + DAM estimator + gKL loss | 180--240 |
| `dam/discrete.py` | Ising / occupation adapters, training CLI, evaluation, JSON | 260--360 |
| `dam/tests_math.py` | mathematical gates | 140--220 |
| `dam/__init__.py` | empty or exports | <10 |
| **total** | | **~600--800** |

No new neural-network file is needed.

No new dependency is needed.

Avoid modifying the existing IASBS scripts unless a truly generic helper needs to be exported.  Direct imports from the existing modules are preferable.

---

# 6. Reuse map: do not duplicate these components

## Fixed Ising

Import from `structured_asbs.fixed_ising`:

- `FixedIsingSpace`
- `SwapController`
- `sample_bridge`
- `net_mult_on_index`
- `net_mult_all_states`
- `propagate_exact`
- `report`

Reuse precomputed objects:

- `space.tgt`
- `space.edge_i`, `space.edge_j`
- `space.logf1`
- `space.pi`
- exact reference-kernel / bridge tables.

## Occupation, enumerable

Import from `structured_asbs.occupation`:

- `OccupationSpace`
- `OccController`
- `sample_bridge`
- `net_mult_on`
- `net_mult_all`
- `propagate_exact`
- `occ_hist`, `maxocc_hist`

Reuse:

- `space.etgt`
- `space.edge_i`, `space.edge_j`
- `space.eocc`, `space.emask`
- `space.logf1`
- `space.pi`.

## Occupation, scalable

Import from `structured_asbs.occupation`:

- `ScaleOccupation`
- `ScaleController`
- `bridge_scale`
- `scale_metrics`

Import from `common.py`:

- `occupation_single_particle_probs`
- `sample_inclusion_exact`
- `save_ckpt`.

Do not duplicate the target sampler or Section-6 metrics.

---

# 7. Terminal factor `f_1` for every DAM benchmark

## 7.1 Ising

Use the repository's existing shifted value

$$
\log f_1(x)
=
-\frac{E(x)}{\tau}
-
\log p^{\mathrm{base}}_{1|0}(x\mid x_0)
+C.
$$

The additive constant `C` is irrelevant because it cancels between numerator and denominator of the DAM multiplier.

Implementation: just index

```python
log_f1 = space.logf1[idx]
```

for enumerated endpoint states.

## 7.2 Occupation, `m=N=4`

Use

```python
log_f1 = space.logf1[idx]
```

from `OccupationSpace`.

## 7.3 Scalable occupation

The target unnormalized log density is

$$
\log \widetilde\mu(\eta)
=
\sum_{i=1}^m
\left[
\log\Gamma(\eta_i+d)
-
\log\Gamma(\eta_i+1)
-
\log\Gamma(d)
\right].
$$

For the Dirac source `Ne_c`, the reference terminal law is multinomial with one-particle probabilities `q`:

$$
p^{\mathrm{base}}_{1|0}(\eta\mid Ne_c)
=
\frac{N!}{\prod_i\eta_i!}
\prod_i q_i^{\eta_i}.
$$

Therefore, up to a state-independent constant,

$$
\log f_1(\eta)
=
\sum_i \log\Gamma(\eta_i+d)
-
\sum_i\eta_i\log q_i
+C.
$$

The `\log(\eta_i!)` terms cancel exactly.

Implement the minimal evaluator:

```python
def scale_logf1(sp, eta):
    q = torch.as_tensor(
        C.occupation_single_particle_probs(sp.m, sp.gamma, sp.source),
        device=eta.device,
        dtype=torch.float64,
    )
    return torch.lgamma(eta.to(torch.float64) + sp.d).sum(-1) \
           - (eta.to(torch.float64) * torch.log(q)).sum(-1)
```

No normalization constant is needed.

Add a unit test at `m=N=4` comparing this formula, after subtracting a constant, against `OccupationSpace.logf1`.

---

# 8. `dam/core.py`: exact API and algorithms

The core file should know nothing about Ising or occupation beyond a tiny adapter interface.

## 8.1 Adapter contract

Each adapter needs only:

```python
class Adapter:
    def log_rates(self, net, t, x): ...
    def base_escape(self, x): ...
    def sample_edge(self, log_rates, x, generator=None): ...
    def apply_edge(self, x, edge): ...
    def base_log_rate(self, x, edge): ...
    def log_f1(self, x1): ...
```

`log_rates` should represent the actual model off-diagonal rates on valid edges, or expose a factorized equivalent for the scalable occupation adapter.

The adapter must guarantee common support:

$$
r_t(y,x)>0
\iff
u_t^\theta(y,x)>0.
$$

This is required for a finite Radon--Nikodym derivative.

## 8.2 Exact piecewise-CTMC rollout

Implement one batched function:

```python
rollout_ctmc(adapter, net, x0, t0, steps, need_logw, generator=None)
```

Return only:

```python
x1, logw, jump_count
```

No full trajectory storage is necessary.

Pseudo-code:

```text
x <- x0
tau <- t0
logw <- 0
jumps <- 0
while any(tau < 1):
    s <- floor(tau * steps)
    bin_end <- (s + 1) / steps
    evaluate stop-gradient model rates at t_s and current x
    Ru <- sum model off-diagonal rates
    Rb <- base escape rate
    Delta <- Exp(Ru)
    h <- min(Delta, bin_end - tau)
    logw += (Ru - Rb) * h

    if Delta < bin_end - tau:
        sample edge e proportional to model rate
        logw += log r(e|x) - log u(e|x)
        x <- jump(x,e)
        tau += Delta
        jumps += 1
    else:
        tau <- bin_end
return x, logw, jumps
```

The model network is always evaluated under `torch.no_grad()` inside this rollout.

The function must support a vector `t0`, because each reciprocal sample uses its own randomly selected time.

### Numerical rules

- store `tau`, rates, `logw`, `log_f1`, and `logsumexp` in float64;
- sample waiting time as `-log(U)/Ru` with `U` clipped away from exactly zero;
- no path-weight clipping in the core implementation;
- assert all model rates and base rates on sampled edges are positive and finite;
- count non-finite weights and fail the run rather than silently replacing them.

## 8.3 Candidate edge sampling

At training state `x=X_t`, compute stop-gradient model rates and sample

$$
Y\sim q_{\bar\theta,t}(\cdot\mid x),
$$

where

$$
q_{\bar\theta,t}(y\mid x)
=
\frac{\bar u_t(y,x)}{R_{\bar u}(t,x)}.
$$

Return:

```python
y, edge, log_q
```

Do not sample `Y` uniformly.  The `1/q` correction in the DAM loss is part of the published method.

## 8.4 DAM multiplier estimator

Implement in log space:

```python
estimate_log_adjoint(adapter, net, t, x, y, K, steps)
```

For the numerator:

$$
\ell_y
=
\log W_y
+
\log f_1(Z).
$$

For the denominator:

$$
\ell_k
=
\log W_k
+
\log f_1(X_1^{(k)}).
$$

Then

$$
\log \widehat m_t(y,x)
=
\ell_y
-
\left[
\operatorname{LSE}(\ell_1,\ldots,\ell_K)-\log K
\right].
$$

Return additionally:

- denominator importance-weight ESS;
- total number of rollout jumps;
- number of terminal `f_1` evaluations, exactly `K+1` per label.

For diagnostic ESS use normalized weights based on

$$
\omega_k
=
\exp(\ell_k-\max_j\ell_j),
$$

and

$$
\mathrm{ESS}
=
\frac{(\sum_k\omega_k)^2}{\sum_k\omega_k^2}.
$$

Do not use ESS to alter the estimator.

## 8.5 Generalized-KL loss

The paper uses

$$
D_{\mathrm{gKL}}(u,w)
=
u-w+w\log\frac{w}{u}
$$

per edge.

Let the network output the log multiplier

$$
a_\theta(t,x,y)
=
\log\frac{u_t^\theta(y,x)}{r_t(y,x)}.
$$

Let

$$
w=r_t(y,x)\widehat m_t(y,x).
$$

After sampling `y` from `q`, the exact importance-sampled objective is

$$
\ell
=
\frac{1}{q_{\bar\theta,t}(y\mid x)}
D_{\mathrm{gKL}}
\left(
 u_t^\theta(y,x),
 r_t(y,x)\widehat m_t(y,x)
\right).
$$

For implementation, terms independent of `theta` may be dropped.  The gradient-equivalent compact loss is

$$
\ell_{\mathrm{train}}
=
\frac{r_t(y,x)}{q_{\bar\theta,t}(y\mid x)}
\left[
 e^{a_\theta(t,x,y)}
-
\operatorname{stopgrad}(\widehat m_t(y,x))
 a_\theta(t,x,y)
\right].
$$

This is preferable because the existing IASBS controllers already output `a_theta` as a log multiplier.

Unit-test its gradient against the literal scalar gKL expression.

---

# 9. Fixed-composition Ising adapter

## Rates

For every legal occupied-to-empty swap edge,

$$
r(y,x)
=
\frac{\gamma}{k(n-k)}.
$$

Therefore

$$
R_{\mathrm{base}}(x)=\gamma.
$$

The learned rate is

$$
u_\theta(y,x)
=
\frac{\gamma}{k(n-k)}e^{a_\theta(t,x,y)}.
$$

Reuse `SwapController` exactly.

## State representation

Keep paths as state indices `idx`, not explicit binary vectors, wherever possible.

- legal destinations: `space.tgt[idx]`;
- network input state: `space.S[idx]`;
- edge log multipliers: same indexing used by `net_mult_on_index`.

## Jump

Sample edge `e` from normalized controlled rates and set

```python
idx = space.tgt[idx, e]
```

## Bridge state

Reuse `sample_bridge` unchanged.

## Evaluation

For Section 6 Table 1:

- feed the learned DAM controller into existing `propagate_exact`;
- compute full terminal-law TV exactly under the same finite-step evaluation scheme already used for IASBS.

This gives directly comparable `DAM` cells for:

- Ising `4x4`;
- Ising `5x5`.

The DAM training rollouts remain exact for the piecewise-constant CTMC; `propagate_exact` is only the common terminal evaluator.

---

# 10. Occupation adapter, enumerable `m=N=4`

For state

$$
\eta=(\eta_1,\ldots,\eta_m),
$$

a legal transfer `i -> j`, `i != j`, has base rate

$$
r_{ji}(\eta)
=
\frac{\gamma}{m-1}\eta_i.
$$

The total base escape rate is

$$
R_{\mathrm{base}}(\eta)
=
\gamma N.
$$

The model rate is

$$
u_{ji}^\theta(\eta)
=
\frac{\gamma}{m-1}\eta_i
 e^{a_{ji}^\theta(t,\eta)}.
$$

Reuse `OccController` exactly.

Do not sample edges with `eta_i=0`.

Use:

- `space.etgt` to apply a transfer;
- `space.eocc` / `space.emask` for legal-edge rate factors;
- `space.logf1` for the terminal factor;
- `sample_bridge` for reciprocal states;
- `propagate_exact` for final Table-1 TV.

---

# 11. Scalable occupation adapter

Reuse the existing `ScaleController`, which returns per-mode factors

$$
\alpha_i(t,\eta),\qquad \beta_i(t,\eta).
$$

Interpret the actual DAM rate as

$$
u_{ji}^\theta(t,\eta)
=
\frac{\gamma}{m-1}\eta_i
\exp[\alpha_i(t,\eta)+\beta_j(t,\eta)],
\qquad i\neq j.
$$

This avoids materializing an `m x m` rate matrix.

Define

$$
A_i
=
\eta_i e^{\alpha_i},
$$

$$
B_j
=
e^{\beta_j},
$$

and

$$
B_{\mathrm{tot}}
=
\sum_j B_j.
$$

Then the total model escape rate is

$$
R_\theta
=
\frac{\gamma}{m-1}
\sum_i A_i
(B_{\mathrm{tot}}-B_i).
$$

This is `O(m)`, not `O(m^2)`.

## Exact `O(m)` edge sampling

The departure-mode marginal is

$$
q(i\mid\eta)
\propto
A_i(B_{\mathrm{tot}}-B_i).
$$

After choosing `i`, sample

$$
q(j\mid i,\eta)
=
\frac{B_j}{B_{\mathrm{tot}}-B_i},
\qquad j\neq i.
$$

Implement `j` by exact masked categorical sampling or rejection from `softmax(beta)`; do not permit `j=i` as an actual CTMC jump.

Then

$$
q(i,j\mid\eta)
=
\frac{u_{ji}^\theta(\eta)}{R_\theta(\eta)}.
$$

Applying the edge is only

```python
eta[i] -= 1
eta[j] += 1
```

and preserves the constraint exactly.

### Important distinction from existing `scale_step`

`scale_step` is an efficient finite-step sampler and permits virtual/self destination handling for speed.  The DAM path likelihood must instead use the genuine off-diagonal CTMC with `i != j` so that the Radon--Nikodym derivative is mathematically exact.

Use the DAM event simulator for DAM rollouts and large-scale DAM evaluation.  Reuse `scale_metrics` unchanged for the final samples.

---

# 12. Training loop in `dam/discrete.py`

The loop should mirror the paper and reuse the repository's reciprocal-projection machinery.

## Outer step

Under `torch.no_grad()`:

1. sample terminal states `X1` from the current DAM CTMC using `rollout_ctmc` from the Dirac source at `t=0`;
2. append terminal states to a small replay buffer.

No trajectory needs to be stored.

## Inner update

For every minibatch:

1. sample `X1` from the replay pool;
2. sample time index `ti` uniformly;
3. set `t=ti/steps`;
4. sample `X_t` from the **existing exact reference bridge** conditioned on `(x0,X1)`;
5. sample one legal edge `X_t -> Y` from the stop-gradient model jump distribution `q`;
6. from `(t,Y)`, sample one model rollout and compute numerator `log W + log f1`;
7. from `(t,X_t)`, sample `K` independent model rollouts and compute denominator terms;
8. form `log_adj` with `logsumexp`;
9. evaluate the local network log multiplier with gradient enabled;
10. take one generalized-KL matching step.

Pseudo-code:

```text
for outer in range(iters):
    with no_grad:
        X1_new = rollout_ctmc(net_bar, source, t=0)
    buffer.add(X1_new)

    for inner in range(inner_updates):
        X1 <- buffer sample
        ti <- Uniform{0,...,steps-1}
        Xt <- exact_reference_bridge(x0, X1, ti)

        with no_grad:
            Y, edge, logq <- sample_model_edge(net_bar, ti, Xt)
            log_adj, stats <- estimate_log_adjoint(
                net_bar, ti, Xt, Y, K
            )

        a <- local network log multiplier at (ti, Xt, edge)
        loss <- sampled gKL loss(a, log_adj, base_rate, logq)
        backward / clip_grad_norm / Adam step
```

Use `net_bar` only semantically: calls inside `torch.no_grad()` are sufficient.  No EMA or duplicate network is required unless an instability later justifies it.

---

# 13. What not to implement

To keep the code small and the comparison scientifically clean, do **not** add:

- masked-diffusion-specific code;
- LLM/token machinery;
- a DAM-specific corrector network;
- a second DAM controller architecture;
- full state-space discrete-adjoint ODE integration;
- full trajectories in the replay buffer;
- terminal marginal model-density estimation;
- a separate bridge learner;
- a custom target sampler;
- path-weight clipping in the mathematical core;
- an approximation that drops the escape-rate integral.

The repository already contains almost every benchmark-specific component DAM needs.

---

# 14. Mathematical tests required before any headline experiment

`dam/tests_math.py` should be treated as a hard gate.

## T0. Terminal-loss identity

For Ising `4x4` and occupation `m=N=4`, verify numerically that

$$
p^{\mathrm{base}}_{1|0}(x\mid x_0)e^{-g(x)}
$$

normalized over the state space equals the exact target `pi` to float64 tolerance.

Target:

```text
max_abs_error < 1e-12
```

This catches the most dangerous implementation mistake: using energy alone instead of the correct density ratio.

## T1. Model equals base => path weight equals one

Zero-initialize the controller, for which

$$
u_\theta=r.
$$

Every sampled path must satisfy

$$
\log W=0
$$

up to floating-point error.

Target:

```text
max |logW| < 1e-10
```

## T2. Mean-one Radon--Nikodym test

For a deliberately nonzero frozen toy controller and several start states/times, verify

$$
\mathbb E_{P^{\bar u}}[W]=1.
$$

Use confidence intervals rather than a hard deterministic tolerance.

## T3. Weighted terminal expectation

At Ising `4x4` and occupation `4`, exact `phi_t(x)` is already available.

Verify

$$
\mathbb E_{P^{\bar u}}
[Wf_1(X_1)]
=
\varphi_t(x).
$$

for several states and times.

This simultaneously validates path simulation, the RN sign, the escape-rate integral, and `f1` indexing.

## T4. DAM multiplier convergence

For random legal edges `(x,y)`, compare the DAM estimator against the exact multiplier

$$
m_t^*(y,x)
=
\frac{\varphi_t(y)}{\varphi_t(x)}.
$$

Run

```text
K = 1, 4, 16, 64, 256
```

and confirm decreasing error of the denominator estimator / expected adjoint.  Do not require a finite-`K` ratio estimator to be unbiased.

## T5. gKL gradient identity

For random positive scalar `r`, `m_hat`, and network log multiplier `a`, compare autograd gradients of:

1. literal generalized KL;
2. compact gradient-equivalent loss

$$
\frac{r}{q}[e^a-m_{\rm hat}a].
$$

Target:

```text
relative gradient error < 1e-12 in float64
```

## T6. Reference endpoint law

Run the event simulator with zero controller and compare empirical endpoints against the exact reference terminal law:

- Ising `4x4`;
- occupation `m=N=4`.

This checks the Gillespie implementation separately from importance weighting.

## T7. Constraint preservation

Require zero violations for all DAM path simulations:

- Ising: exact fixed composition;
- occupation: exact particle number.

## T8. Scale `logf1` identity

At `m=N=4`, compare `scale_logf1` against `OccupationSpace.logf1` after removing a single additive constant.

Target:

```text
max centered difference < 1e-12
```

## T9. Scale factorized rate identity

At small `m`, explicitly materialize all `i != j` rates and verify that the `O(m)` formulas for:

- total escape rate;
- departure distribution;
- conditional destination distribution

match the full `m(m-1)` calculation.

## T10. Direct CTMC likelihood check on a tiny chain

Include one tiny hand-constructed 3-state CTMC with a fixed piecewise-constant control.  For sampled paths, compare `logW` from the generic DAM code with a direct path-density calculation.

This test is independent of the IASBS benchmark code and is the strongest guard against a hidden sign/indexing error.

**Do not run Section-6 DAM experiments until T0--T10 pass.**

---

# 15. Experiment sequence

## Phase A. Smoke test on occupation `m=N=4`

Why first: only 35 states, exact `phi`, exact target, cheap CTMC, easiest place to diagnose every quantity.

Run a tiny configuration first:

```bash
python -m dam.discrete occupation \
  --m 4 --N 4 --K 4 \
  --steps 64 --iters 20 --inner 2 --mb 64 \
  --out json/results_dam_occ4_smoke.json
```

Required before proceeding:

- finite loss;
- finite path weights;
- zero constraint violations;
- ESS logged;
- TV moves away from the uncontrolled reference in the correct direction.

Then run a `K` sweep:

```text
K = 1, 4, 16, 64
```

Use `K=16` as the initial primary setting because the DAM paper uses `K=16` on one of its low-dimensional synthetic tasks; do not hard-code this if the estimator audit shows that our problem requires a different value.

## Phase B. Ising `4x4`

Run the same `K` sweep.

Headline cell:

```text
Table 1 / Ising 4x4 / DAM / TV
```

Also record in Appendix diagnostics:

- `K`;
- wall clock;
- terminal `f1` evaluations;
- average / p10 ESS;
- exact log-multiplier RMSE at selected times.

## Phase C. Ising `5x5`

Reuse the same code and network family.

Headline cell:

```text
Table 1 / Ising 5x5 / DAM / TV
```

No new metric is needed.

If full exact terminal propagation remains the runtime bottleneck, use the repository's existing L5 propagation path unchanged.

## Phase D. Occupation `m=N=32`

This is the first non-enumerable DAM scale benchmark.

Headline cells:

```text
Table 2 / 32 / DAM / KS_occ
Table 2 / 32 / DAM / KS_max
```

Use exact Dirichlet--multinomial target samples already provided by the repository.

## Phase E. Occupation `m=N=128`

Same metrics only.

Headline cells:

```text
Table 2 / 128 / DAM / KS_occ
Table 2 / 128 / DAM / KS_max
```

## Phase F. Occupation `m=N=1000`

Run only after measuring the actual jump/second rate at `m=128`.

The base process has total escape rate

$$
R_{\mathrm{base}}=\gamma N.
$$

With `gamma=4` and a uniformly sampled training time, the expected number of remaining base jumps per rollout is approximately

$$
\frac{\gamma N}{2}=2N.
$$

Thus one DAM label with one numerator rollout and `K` denominator rollouts costs roughly

$$
2N(K+1)
$$

jump events before control-induced changes in escape rate.

At `N=1000, K=16`, this is about

$$
34{,}000
$$

jump events **per training label**.

Therefore `m=1000` is expected to be the main computational stress point of generic DAM.  Do not hide this by changing the reference process or dropping path-weight terms.

Headline cells, if the run is feasible:

```text
Table 2 / 1000 / DAM / KS_occ
Table 2 / 1000 / DAM / KS_max
```

If `K=16` exceeds the predeclared compute budget, run a clearly identified `K=4` and then `K=1` fallback rather than silently changing mathematics.  Record the chosen `K` in the JSON and Appendix.

---

# 16. Recommended CLI

A single script is enough:

```bash
python -m dam.discrete ising --L 4 ...
python -m dam.discrete ising --L 5 ...
python -m dam.discrete occupation --m 4 --N 4 ...
python -m dam.discrete occupation-scale --m 32 --N 32 ...
python -m dam.discrete occupation-scale --m 128 --N 128 ...
python -m dam.discrete occupation-scale --m 1000 --N 1000 ...
```

Common DAM-only options:

```text
--K                 denominator rollout count
--steps             piecewise-constant time grid
--iters
--inner
--batch
--mb
--buffer
--hidden
--lr
--seed
--eval-every
--n-samples
--out
--ckpt-dir
--tag
```

Optional diagnostic-only flags:

```text
--K-num             default 1; numerator-rollout ablation only
--weight-stats      save ESS/logW quantiles
--max-hours         external/manual run policy, not mathematical logic
```

Do not expose dozens of unnecessary hyperparameters.

---

# 17. Primary hyperparameter policy

The DAM paper uses `K=16` for Checkerboard and `K=64` for Pinwheel, and uses on-policy samples for its low-dimensional synthetic examples.

For our benchmarks:

1. audit `K in {1,4,16,64}` on occupation-4 and Ising-4;
2. select one primary `K` before seeing large-scale test metrics;
3. preserve the same controller hidden size and time features as IASBS;
4. preserve the same base generator and target;
5. report total terminal `f1` evaluations and wall clock;
6. never improve DAM by using exact `phi_t` in the training target except in an explicitly labeled oracle test.

The exact `phi_t` tables at small scale are for **verification only**, not for the practical DAM label.

---

# 18. Section-6 outputs and only the necessary headline metrics

The DAM implementation needs to fill exactly these blank cells.

## Table 1

| benchmark | DAM output |
|---|---|
| Ising `4x4` | terminal-law TV |
| Ising `5x5` | terminal-law TV |
| Occupation `m=N=4` | terminal-law TV |

No DAM-specific variance or cost metric should be added to Table 1.

## Table 2

| benchmark | DAM outputs |
|---|---|
| `m=N=32` | KS_occ, KS_max |
| `m=N=128` | KS_occ, KS_max |
| `m=N=1000` | KS_occ, KS_max |

No additional headline statistic is required.

## Appendix-only DAM diagnostics

These are sufficient and necessary:

1. `K`;
2. total terminal `f1` evaluations;
3. wall clock;
4. path-weight ESS;
5. small-scale log-multiplier RMSE against exact control.

Do not create a large DAM metric zoo.

---

# 19. Result JSON schema

Keep the same style as the current repository.

Suggested minimum:

```json
{
  "method": "DAM",
  "benchmark": "occupation_scale",
  "config": {
    "m": 32,
    "N": 32,
    "K": 16,
    "steps": 128,
    "seed": 0
  },
  "headline": {
    "KS_occ": 0.0,
    "KS_max": 0.0
  },
  "dam": {
    "f1_evals": 0,
    "rollout_jumps": 0,
    "wall_sec": 0.0,
    "ess_mean": 0.0,
    "ess_p10": 0.0,
    "nonfinite_weights": 0
  }
}
```

For enumerable runs add:

```text
TV
KL                   # optional appendix diagnostic, already available
log_multiplier_RMSE  # verification/appendix only
```

Save the controller checkpoint through `common.save_ckpt` so the result can be remeasured without retraining.

---

# 20. Failure rules

A run is invalid and must not populate Section 6 if any of the following occurs:

- nonzero constraint violations;
- non-finite path weights;
- `T0`, `T1`, `T3`, `T5`, `T8`, `T9`, or `T10` fails;
- the escape-rate integral is omitted;
- an invalid/self edge is treated as a genuine occupation jump;
- `g` is replaced by energy alone;
- exact `phi` is accidentally used to train the practical DAM controller;
- the model proposal distribution is differentiated through inside the DAM label;
- `1/q(Y|X_t)` is omitted from edge-sampled gKL training without replacing it by an exact full-edge sum;
- a non-Dirac source is used without a mathematically valid terminal factor `f_1`.

Estimator variance or low ESS is **not** an implementation failure.  It is a legitimate empirical property of DAM.

---

# 21. Optional non-Dirac DAM extension -- not needed for the primary baseline

If later we insist on source-matching DAM to `IASBS (non-Dirac)`, first obtain a valid SB terminal factor

$$
f_1
=
\frac{\mu}{\widehat f_1}
$$

from the non-Dirac corrector.

Then freeze this same `f_1` evaluator and set

$$
g=-\log f_1.
$$

The generic DAM controller code above remains unchanged.

But the experiment must be described as using a **shared/frozen corrector**, and any efficiency comparison must account for the corrector cost consistently.

This optional extension requires almost zero DAM-specific LOC because `dam/core.py` only consumes `log_f1(x)`.

---

# 22. Implementation order

Do the work in this order:

```text
1. dam/core.py: exact path simulator + log RN weight
2. T1 / T2 / T10 path-weight tests
3. terminal logf1 adapters + T0 / T8
4. DAM log-adjoint estimator + T3 / T4
5. gKL loss + T5
6. occupation-4 adapter and smoke training
7. Ising-4 adapter
8. Ising-5
9. scalable occupation O(m) rate/edge adapter + T9
10. occupation-32
11. occupation-128
12. occupation-1000 feasibility run
13. only then write Section-6 DAM numbers
```

This ordering isolates mathematical bugs before expensive experiments.

---

# 23. Definition of done

The DAM subfolder is complete when:

- all mathematical tests `T0--T10` pass;
- the code uses the published importance-weighted discrete-adjoint and gKL matching logic;
- the full CTMC path RN derivative is used for our non-masked references;
- no new controller network is introduced;
- no benchmark code is duplicated unnecessarily;
- `DAM` produces the three Table-1 TV cells;
- `DAM` produces the Table-2 KS cells at every computationally feasible occupation scale;
- every run records `K`, wall time, terminal `f1` evaluations, jump count, and weight ESS;
- checkpoints and JSONs can be remeasured by the existing repository infrastructure;
- the manuscript does not claim a non-Dirac DAM corrector that the method does not implement.

The intended final structure remains compact: **one generic DAM core, one discrete benchmark wrapper, one mathematical test file.**
