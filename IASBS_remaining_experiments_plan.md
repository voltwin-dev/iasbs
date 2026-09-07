# Remaining Experiment Plan: non-Dirac IASBS + R-ASBS Sphere

## Scope

This plan intentionally **holds DAM completely out**. It covers only the experiments needed to fill the current Section 6 cells for:

1. **IASBS (non-Dirac)** on all current discrete benchmarks and on the sphere benchmark.
2. **R-ASBS sphere**, only if the existing PyTorch port can be brought into credible agreement with the released MATLAB implementation / published behavior.

The implementation constraint is **minimal LOC change**. The current repository already contains almost all required machinery: exact reference bridges, terminal-kernel ratios, controller networks, metrics, checkpointing, exact target samplers, and the R-ASBS corrector implementation.

---

# 0. Important correction: which IASBS branch needs a corrector?

The additional corrector network is needed for **non-Dirac IASBS**, not for the current Dirac IASBS experiments.

For a Schrödinger bridge,

$$
f_1(z)=\frac{\mu(z)}{\widehat f_1(z)},
$$

with

$$
\widehat f_1(z)
=
\int p^{\mathrm{base}}_{1|0}(z\mid x_0) f_0(x_0)\,\nu_0(dx_0).
$$

For a Dirac source, $\nu_0=\delta_{x_0}$, the source-side factor is absorbed into a constant and

$$
f_1(z)\propto
\frac{e^{-E(z)/\tau}}
{p^{\mathrm{base}}_{1|0}(z\mid x_0)},
$$

so the current repository can evaluate the terminal label directly. This is why `fixed_ising.py`, `occupation.py`, and `sphere.py` are corrector-free today.

For a genuinely non-Dirac source, $\widehat f_1$ is unknown and must be estimated. The controller terminal label then contains a learned corrector.

Continuous sphere:

$$
\nabla \log f_1(y)
=
-\frac{1}{\tau}\nabla E(y)
-
\nabla\log\widehat f_1(y).
$$

Discrete symmetry edge $g$:

$$
\frac{f_1(gy)}{f_1(y)}
=
\exp\!\left[-\frac{E(gy)-E(y)}{\tau}\right]
\frac{\widehat f_1(y)}{\widehat f_1(gy)}.
$$

Thus the intended non-Dirac experiment is the **full IASBS controller-corrector loop**, analogous in role to the controller/corrector loop already present in `rasbs/rasbs_sphere_port.py`.

---

# 1. What the current repository already gives us

Relevant files:

```text
structured_asbs/fixed_ising.py
structured_asbs/occupation.py
structured_asbs/sphere.py
structured_asbs/remeasure.py
common.py
rasbs/rasbs_sphere_port.py
rasbs/rasbs_sphere_audit.py
json/results_rasbs_sphere_bimodal*.json
json/results_sphere_train_anti.json
```

The important reusable pieces are:

- `fixed_ising.py`
  - exact Johnson-graph reference kernel;
  - exact endpoint-conditioned bridge sampler;
  - exact full-law propagation at $4\times4$ and $5\times5$;
  - `SwapController` already outputs all swap-edge log multipliers in one pass;
  - exact target TV.

- `occupation.py`
  - exact small-state reference semigroup;
  - exact small-state bridge;
  - scalable independent-particle bridge for concentrated source states;
  - target Dirichlet-multinomial iid sampler;
  - scalable KS$_{\rm occ}$ and KS$_{\max}$ metrics;
  - existing controller architectures and stochastic terminal-label estimators.

- `sphere.py`
  - exact spherical heat kernel and derivative tables;
  - exact Brownian bridge simulator;
  - exact target sampler / CDF metrics;
  - `ScoreNet`, EMA, replay buffer, geodesic simulator;
  - all Section 6 sphere metrics are already computable.

- `rasbs/rasbs_sphere_port.py`
  - controller network;
  - corrector network;
  - R-ASBS alternating controller/corrector training loop;
  - R-ASBS bridge surrogate and parallel transport;
  - shared Section 6 metric evaluator;
  - checkpoint and JSON output.

Therefore **do not create a new framework, trainer abstraction, dataset class, Hydra config, Lightning module, etc.** Add small helper functions and one command-line switch per script.

---

# 2. Section 6 cells to fill

## Table 1 — enumerable discrete TV

Fill IASBS (non-Dirac) only:

```text
Ising 4x4              TV
Ising 5x5              TV
Occupation m=N=4       TV
```

DAM remains blank for now.

## Table 2 — scalable occupation

Fill IASBS (non-Dirac):

```text
m=N=32      KS_occ, KS_max
m=N=128     KS_occ, KS_max
m=N=1000    KS_occ, KS_max
```

DAM remains blank for now.

## Sphere table

Fill IASBS (non-Dirac):

```text
north mass
KS(x3)
W1(x3)
KS(phi)
|Delta E[x3^2]|
```

For R-ASBS, the published north mass 0.438 remains the primary-source number. The other R-ASBS cells should only be populated from our port if the port passes the fidelity gate in Section 6 below.

---

# 3. Shared non-Dirac corrector identity

The minimal implementation should learn **the terminal corrector quantity that the IASBS label actually needs**, rather than introducing a generic density model.

Let $(X_0,X_1)$ be endpoints sampled from the current controlled process. Under the fixed-point Schrödinger bridge,

$$
p^*(x_0\mid X_1=z)
\propto
f_0(x_0)\nu_0(x_0)
\,p^{\mathrm{base}}_{1|0}(z\mid x_0).
$$

This gives a direct regression identity.

## Continuous version

$$
\nabla\log\widehat f_1(z)
=
\mathbb E_{p^*}
\left[
\nabla_z\log p^{\mathrm{base}}_{1|0}(z\mid X_0)
\mid X_1=z
\right].
$$

So the corrector network on the sphere should regress the **exact spherical heat-kernel terminal score**.

## Discrete ratio version

For any admissible operator move $T$,

$$
\frac{\widehat f_1(Tz)}{\widehat f_1(z)}
=
\mathbb E_{p^*}
\left[
\frac{p^{\mathrm{base}}_{1|0}(Tz\mid X_0)}
{p^{\mathrm{base}}_{1|0}(z\mid X_0)}
\mid X_1=z
\right].
$$

This is ideal for the existing positive-ratio / Poisson-Bregman training style. No normalizing constant and no explicit $f_0$ are required.

At the fixed point, alternate:

```text
simulate current controller -> endpoint pairs (X0, X1)
corrector regression from endpoint pairs
reference-bridge resampling Xt | X0, X1
controller regression using target ratio / learned corrector ratio
repeat
```

This is the discrete/IASBS analogue of the alternating loop already visible in `rasbs_sphere_port.py`.

---

# 4. non-Dirac IASBS on S2 — highest priority

## 4.1 Source

Use **Haar/uniform on $\mathbb S^2$**.

Reasons:

1. It is unquestionably non-Dirac.
2. It is trivial to sample.
3. It matches the source used by the R-ASBS sphere benchmark.
4. It makes the eventual IASBS-vs-R-ASBS comparison maximally interpretable.
5. No new dataset or source model is needed.

Do **not** use a mixture-of-Diracs for the headline sphere experiment; Haar is cleaner and directly exercises the corrector.

## 4.2 Minimal code changes in `structured_asbs/sphere.py`

### A. Generalize the simulator initial state

Current:

```python
def simulate(prob, score_fn, batch, steps, generator=None, keep_path=False):
    x = prob.x0[None, :].expand(batch, 3).contiguous().clone()
```

Change to:

```python
def simulate(prob, score_fn, batch, steps, generator=None,
             keep_path=False, x_init=None):
    if x_init is None:
        x = prob.x0[None, :].expand(batch, 3).contiguous().clone()
    else:
        x = x_init.clone()
```

Estimated change: **~5 LOC**.

### B. Generalize `bridge_path`

Current bridge starts from `prob.x0`.

Add `x0=None`; if supplied use the batch-specific source points.

The bridge drift already depends only on the current point, endpoint `y1`, and exact heat-kernel score, so no mathematical rewrite is needed.

Estimated change: **~5 LOC**.

### C. Reuse `ScoreNet` for the corrector

Do not create a second architecture. Instantiate another `ScoreNet`, but call it at terminal time only:

```python
hnet = ScoreNet(hidden=args.hidden, symmetric=False)
```

The corrector is a tangent vector field $h(y)\approx\nabla\log\widehat f_1(y)$.

It does not need time as an information variable; for minimal LOC simply call the existing model with `t=1` rather than creating `TerminalScoreNet`.

### D. Exact corrector target

For endpoint pair $(x_0,y)$, compute

$$
b(x_0,y)
=
\nabla_y\log p_{r_{0,1}}(x_0\cdot y).
$$

With the existing heat table,

$$
b(x_0,y)
=
\frac{d}{dc}\log p_{r_{0,1}}(c)
\left(x_0-cy\right),
\qquad c=x_0\cdot y.
$$

This is already essentially implemented inside `SphereProblem.G`; factor the heat-kernel part into one helper:

```python
def terminal_kernel_score(prob, x0, y):
    c = (x0 * y).sum(-1).clamp(-1.0, 1.0)
    si = torch.zeros(len(y), device=y.device, dtype=torch.long)
    k = prob.tab.dlogp_dc(si, c)
    return k[:, None] * (x0 - c[:, None] * y)
```

Estimated change: **~6 LOC**.

Corrector loss:

$$
\mathcal L_H
=
\mathbb E
\left[
\|H_\psi(X_1)-b(X_0,X_1)\|^2
\right].
$$

Use the same EMA strategy as the controller if instability appears; initially keep only controller EMA to minimize changes.

### E. Controller terminal gradient

Replace the Dirac terminal gradient `prob.G(y)` by

$$
G_{\rm ND}(y)
=
-\frac{1}{\tau}\nabla E(y)-H_\psi(y).
$$

Then keep the existing Killing readout unchanged:

$$
\Lambda(x,y)
=
(x\cdot y)G_{\rm ND}(y)
-y\,[x\cdot G_{\rm ND}(y)].
$$

This is important: **the IASBS geometric machinery does not change at all. Only the terminal $f_1$ gradient changes.**

### F. Add one new subcommand

Prefer:

```text
python structured_asbs/sphere.py train-nondirac ...
```

rather than adding many conditionals to `train`.

However the actual implementation should reuse `run_train` through one boolean argument wherever possible.

Estimated net new sphere code: **~45–70 LOC**, mostly the alternating corrector update and source sampling.

## 4.3 Training loop

Minimal loop:

```python
for it in range(1, args.iters + 1):
    x0 = random_sphere(args.batch)

    with torch.no_grad():
        y1 = simulate(prob, controller_fn, args.batch, steps, x_init=x0)

    # corrector update
    b = terminal_kernel_score(prob, x0, y1)
    h = hnet(torch.ones(len(y1)), y1)
    loss_h = ((h - b) ** 2).sum(-1).mean()
    update(hnet, loss_h)

    # controller update: same bridge/readout pipeline already used today
    with torch.no_grad():
        path = bridge_path(prob, y1, steps, x0=x0)
        h1 = hnet(torch.ones(len(y1)), y1)
        G1 = -project_tangent(y1, energy_grad(y1) / tau) - h1

    Xt = replay_sample(path)
    label = sphere_terminal_readout(Xt, y1, G1)
    loss_u = mse(controller(Xt, t), label)
    update(controller, loss_u)
```

Do not copy R-ASBS's approximate geodesic-Gaussian bridge or log-map corrector. IASBS already has the **exact spherical reference heat kernel and exact bridge code** in this repo; use it.

## 4.4 Run protocol

Use the same headline protocol as the existing IASBS sphere result unless memory forces a reduction:

```text
steps       128
iters       4000
inner       16
batch       8192
mb          16384
hidden      256
lr          1e-3
EMA         0.9995
seeds       5
samples     200000
source      Haar S2
```

If alternating the corrector makes this too expensive, use one corrector minibatch update per outer iteration and retain the existing `inner=16` controller updates.

## 4.5 Section 6 outputs

Use `remeasure.py` / existing sphere metrics unchanged.

Required only:

```text
north mass
KS(x3)
W1(x3)
KS(phi)
abs(E[x3^2] - exact)
```

Also save, Appendix only:

```text
loss_u
loss_h
norm residual
wall time
controller params
corrector params
```

## 4.6 Gates

The experiment should be considered valid before worrying whether it beats the Dirac branch:

```text
G0 corrector sanity:
   on controlled endpoint pairs, held-out corrector MSE decreases materially

G1 feasibility:
   max | ||x|| - 1 | < 1e-10 in float64 runtime samples

G2 no mode collapse:
   |north - 0.5| < 0.03

G3 target law:
   KS(x3) < 0.05

G4 no hidden angular collapse:
   KS(phi) < 0.02
```

The Section 6 cell should report the measured result even if a gate fails.

---

# 5. non-Dirac IASBS on fixed-magnetization Ising

This is the easiest discrete non-Dirac extension and should be implemented second.

## 5.1 Source

Use the **uniform distribution on $\Omega_{n,k}$**.

Reasons:

- exact sampler is one random state index;
- respects the constrained state space;
- reference single-swap chain is permutation symmetric;
- exact kernel still depends only on Johnson distance;
- Table 1 exact-law evaluation remains available;
- no new source model.

## 5.2 Corrector ratio

For swap permutation $g$ and endpoint pair $(X_0,Y)$,

$$
C_g(Y)
=
\frac{\widehat f_1(gY)}{\widehat f_1(Y)}.
$$

Unbiased terminal label:

$$
Q_g(X_0,Y)
=
\frac{p^{\mathrm{base}}_{1|0}(gY\mid X_0)}
{p^{\mathrm{base}}_{1|0}(Y\mid X_0)}.
$$

The Johnson reference kernel depends only on

$$
d(X_0,Y)=k-X_0^\top Y,
$$

so

$$
\log Q_g
=
\log\kappa_{d(X_0,gY)}
-
\log\kappa_{d(X_0,Y)}.
$$

All of this is already available from `space.log_kappa_full`; only the fixed `space.x0` assumption must be removed from one helper.

## 5.3 Reuse `SwapController` as corrector

Instantiate:

```python
net_u = SwapController(...)
net_h = SwapController(...)
```

Call `net_h` at terminal time $t=1$ and interpret its relevant pair entries as **log corrector ratios**.

No new architecture.

Corrector Poisson-Bregman loss:

$$
\mathcal L_H
=
\mathbb E
\left[
\exp(h_g(Y))-h_g(Y)Q_g(X_0,Y)
\right].
$$

At its conditional optimum,

$$
\exp(h_g(Y))
=
\mathbb E[Q_g(X_0,Y)\mid Y]
=C_g(Y).
$$

Controller terminal ratio becomes

$$
\Lambda_g(Y)
=
\exp\!\left[-\frac{E(gY)-E(Y)}{\tau}-h_g(Y)\right].
$$

Then the current controller regression is unchanged.

## 5.4 Minimal helper changes

### `simulate`

Add optional vector `idx0` rather than always starting at `space.i0`.

~5 LOC.

### `propagate_exact`

Add optional initial law `p0`; default stays Dirac.

For non-Dirac Ising use

```python
p0 = torch.full((space.M,), 1.0 / space.M)
```

~5 LOC.

### `sample_bridge`

Generalize from fixed `space.x0` to batch-specific source state `X0`.

The current combinatorial bridge sampler can remain intact. Only compute

```text
m = k - <X0, X1>
block labels from (X0, X1)
```

instead of using `space.dist0` and `space.x0`.

~10–15 LOC.

### Corrector labels

Add one helper, roughly:

```python
def corrector_labels(space, X0, X1_idx):
    # all unordered swap pairs
    # return log p(gY|X0) - log p(Y|X0)
```

~15 LOC.

### New command

```text
python structured_asbs/fixed_ising.py train-nondirac ...
```

Estimated total new Ising code: **~50–80 LOC**.

## 5.5 Runs

Keep current hyperparameters per lattice size.

```bash
python structured_asbs/fixed_ising.py train-nondirac \
    --L 4 --steps 256 --iters 3000 \
    --tag ising_nd_L4 --out json/results_ising_nd_L4.json

python structured_asbs/fixed_ising.py train-nondirac \
    --L 5 --steps 512 --iters 6000 --hidden 512 \
    --tag ising_nd_L5 --out json/results_ising_nd_L5.json
```

The exact-law final evaluation must start from the same uniform source used in training.

## 5.6 Required output

Only Table 1 TV:

```text
Ising 4x4 IASBS(non-Dirac) TV
Ising 5x5 IASBS(non-Dirac) TV
```

Appendix-only diagnostics:

```text
corrector loss
controller loss
corrector ratio error at L=4 against an exact Sinkhorn/IPF solution if computed
constraint violations
wall time
```

### Strong optional correctness check at L=4

Because $|\Omega|=12,870$, solve the two-marginal Schrödinger system by exact matrix/IPF once and compare the learned corrector ratio against the exact one. This is highly valuable but should be an evaluation helper, not a separate training method.

---

# 6. non-Dirac IASBS on occupation spaces

This is the only part where the non-Dirac extension is not completely free. The current large-$m$ implementation exploits a product/rank-1 structure induced by the concentrated Dirac source. A non-Dirac corrector changes the terminal ratio and is not guaranteed to preserve exact rank-1 structure.

Therefore implement this in two stages and **do not redesign the scalable architecture before measuring whether the existing one is sufficient**.

## 6.1 Source: uniform mixture of concentrated states

Use

$$
\nu_0
=
\frac1m\sum_{c=1}^m\delta_{Ne_c}.
$$

This is genuinely non-Dirac on the occupation state space, but preserves the key reason the current reference code is cheap: conditional on the sampled source component $c$, all $N$ particles start at one mode.

This choice is specifically preferable to a generic multinomial source because:

- the current exact/scalable bridge machinery can be reused almost unchanged;
- reference endpoint probabilities have the existing multinomial closed form;
- terminal kernel ratios are $O(1)$ per queried transfer;
- source sampling is one integer `c` per trajectory.

Do **not** condition the controller on $c$. The source component is used only to construct endpoint-pair corrector labels and the reference bridge; the controlled process remains a sampler on the original occupation state space.

## 6.2 Corrector transfer ratio

For endpoint $\xi$ and transfer $a\to b$,

$$
C_{ba}(\xi)
=
\frac{\widehat f_1(\xi-e_a+e_b)}{\widehat f_1(\xi)}.
$$

For source component $c$, the reference endpoint is multinomial with one-particle probabilities $q^{(c)}$. Therefore the sampled corrector label has the closed form

$$
Q_{ba}(c,\xi)
=
\frac{p^{\mathrm{base}}(\xi-e_a+e_b\mid Ne_c)}
{p^{\mathrm{base}}(\xi\mid Ne_c)}
=
\frac{\xi_a}{\xi_b+1}
\frac{q_b^{(c)}}{q_a^{(c)}}
$$

whenever $\xi_a>0$.

Thus no state-space enumeration is required even at $m=N=1000$.

## 6.3 Small $m=N=4$

This should be fully rigorous.

Reuse the existing `OccupationController` as both:

```text
net_u : controller
net_h : terminal corrector-ratio predictor
```

At `m=4`, train all directed transfer ratios. Substitute

$$
R_{ab}^{f_1}(\xi)
=
\frac{f_1(\xi-e_a+e_b)}{f_1(\xi)}
=
\frac{\mu(\xi-e_a+e_b)}{\mu(\xi)}
\frac{1}{C_{ba}(\xi)}
$$

into the existing `labels_full` / $A^{(t)}$ formula.

Because the whole state space has 35 states, also compute an exact two-marginal SB/IPF solution and use it as a unit test for:

```text
learned corrector ratios
learned controller
final TV
```

This is the cleanest correctness gate for the non-bijective non-Dirac construction.

Expected new code for the small branch: **~60–90 LOC**.

## 6.4 Large $m=N=32,128,1000$

### First attempt: retain the current `ScaleController`

Do not immediately replace the rank-1 controller.

The current architecture returns

$$
a_{ij}=\alpha_i+\beta_j.
$$

For a non-Dirac bridge this factorization is no longer guaranteed by the same product-form argument. Nevertheless it is a neural approximation family and may remain sufficient for the symmetric source/target used here.

The first experiment should therefore be exactly:

```text
same ScaleController
+ second ScaleController used as a terminal corrector
+ sampled A^(t) labels using learned f1 transfer ratios
+ source component c sampled uniformly per trajectory
```

The existing stochastic occupation estimator is particularly useful here because it requires only a small number of queried terminal transfer ratios; we do not need an $m\times m$ exact correction matrix.

### Corrector architecture

For minimal LOC, reuse `ScaleController` and evaluate its terminal output at `t=1`. Interpret its `alpha_a + beta_b` as the log corrector transfer ratio approximation.

This is again a restricted family. Measure before changing it.

### If the existing rank-1 family fails

Only then add a low-rank residual, e.g.

$$
a_{ij}
=
\alpha_i+\beta_j
+u_i^\top v_j,
\qquad r\in\{2,4\}.
$$

But this is **fallback work, not part of the first implementation** because it complicates the current $O(m)$ simulator.

### Required metrics

Exactly the current Table 2 metrics:

```text
KS_occ
KS_max
```

for

```text
m=N=32
m=N=128
m=N=1000
```

No new headline metric.

### Interpretation rule

If the non-Dirac large-scale run fails while $m=4$ passes exact-IPF validation, report it as a **scalable approximation/architecture limitation**, not a failure of the IASBS intertwining identity.

---

# 7. R-ASBS sphere: resolve the port before using its extra metrics

The existing port already computes every metric needed for the Section 6 sphere table. The problem is not missing measurement code; it is the unresolved disagreement:

```text
published R-ASBS north mass = 0.438
current PyTorch port = near-complete one-pole collapse in 5/5 seeds
```

Therefore the remaining task is an **implementation-fidelity audit**, not a new experiment implementation.

## 7.1 First fix: MATLAB-equivalent initialization

The repository itself identifies an untested mismatch:

- MATLAB `fullyConnectedLayer`: Glorot/Xavier weights, zero bias;
- current `torch.nn.Linear`: PyTorch default initialization, including random/nonzero bias.

For a symmetric two-mode target, a tiny fixed directional bias can seed winner-take-all symmetry breaking. This mismatch should be removed before interpreting the port.

Add one helper to `rasbs/rasbs_sphere_port.py`:

```python
def matlab_init(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return net
```

and wrap only the R-ASBS networks.

Estimated change: **~8 LOC**.

Do not change any bridge, loss, optimizer, network width, sigma, step count, or geometry in this first rerun.

## 7.2 Fidelity run sequence

### Gate R0 — current self-checks

```bash
python rasbs/rasbs_sphere_port.py --check
```

Must pass.

### Gate R1 — null target audit

The repo already contains `rasbs_sphere_audit.py`. Run the $E=0$/Haar test to completion before the bimodal benchmark.

A Haar target should not spontaneously choose a pole. If it does, the port/training loop still has a symmetry-breaking bug or instability.

### Gate R2 — single-mode vMF audit

Run the existing analytic unimodal audit. A unimodal problem removes the two-basin ambiguity and tests score/corrector orientation.

### Gate R3 — bimodal published benchmark

Run at least five seeds with exactly the published/released settings:

```text
B=500
epochs=600
N=500
sigma=1
controller 64/64 tanh
corrector 48/48 tanh
Adam lr=2e-3
```

Save 100k terminal points per seed for stable Section 6 metrics.

Suggested commands:

```bash
python rasbs/rasbs_sphere_port.py --problem bimodal --seed 0 \
  --tag rasbs_sphere_matlabinit_s0 \
  --out json/results_rasbs_sphere_matlabinit_s0.json
```

repeat seeds 1--4.

## 7.3 When may the Section 6 R-ASBS blank metrics be filled?

### Accept port metrics in the main table only if

1. R0 passes;
2. the null and unimodal audits behave correctly;
3. the bimodal runs no longer exhibit pathological one-pole collapse in essentially every seed;
4. northern mass is at least qualitatively compatible with the published 0.438 result.

Then compute, with exactly the same evaluator used for IASBS:

```text
KS(x3)
W1(x3)
KS(phi)
abs(E[x3^2] - exact)
```

and fill the R-ASBS column as **our reproduction**, explicitly distinguished in the caption from the published 0.438 north-mass value.

### If the port still collapses

Do **not** fill the main R-ASBS cells from it.

Keep:

```text
north mass = 0.438  (published)
other metrics = --  (not published)
```

and report the failed reproduction only in Appendix C.

This is preferable to placing numerically precise but scientifically unresolved port values beside a published result they contradict.

---

# 8. Result files and checkpoint naming

Keep the current repo naming style.

Suggested outputs:

```text
json/results_ising_nd_L4.json
json/results_ising_nd_L5.json
json/results_occ_nd_m4.json
json/results_occ_nd_m32.json
json/results_occ_nd_m128.json
json/results_occ_nd_m1000.json
json/results_sphere_nd.json

json/results_rasbs_sphere_matlabinit_s0.json
...
json/results_rasbs_sphere_matlabinit_s4.json
```

Checkpoints:

```text
ckpt/ising_nd_L4.pt
ckpt/ising_nd_L5.pt
ckpt/occ_nd_m4.pt
ckpt/occ_nd_m32.pt
ckpt/occ_nd_m128.pt
ckpt/occ_nd_m1000.pt
ckpt/sphere_nd_seed0.pt ... seed4.pt
ckpt/rasbs_sphere_matlabinit_s0.pt ... s4.pt
```

Every non-Dirac checkpoint should contain **both controller and corrector state dicts**. If `common.save_ckpt()` currently accepts only one `net`, minimally extend its `extra` payload with `corrector_state_dict` rather than redesigning the checkpoint API.

---

# 9. Do not add new headline metrics

The new Section 6 tables are already correctly scoped.

## Table 1

Only:

$$
\mathrm{TV}.
$$

## Table 2

Only:

$$
\mathrm{KS}_{\rm occ},
$$

$$
\mathrm{KS}_{\max}.
$$

## Sphere

Only:

$$
P(X_3>0),
$$

$$
\mathrm{KS}(X_3),
$$

$$
W_1(X_3),
$$

$$
\mathrm{KS}(\phi),
$$

$$
\left|\mathbb E[X_3^2]-\mathbb E_\pi[X_3^2]\right|.
$$

Corrector loss, controller loss, oracle calls, wall clock, constraint residuals, label variance and IPF ratio errors belong in Appendix C only.

---

# 10. Minimal-diff implementation order

Implement in this order because each stage validates machinery reused by the next.

## Phase 1 — R-ASBS port fidelity, ~8 LOC

```text
[ ] add MATLAB Glorot + zero-bias initializer
[ ] run --check
[ ] run null-target audit
[ ] run unimodal audit
[ ] rerun 5 bimodal seeds
[ ] decide whether main-table R-ASBS metrics are scientifically usable
```

No other R-ASBS code change unless an audit demonstrates an actual mismatch.

## Phase 2 — non-Dirac sphere, ~45–70 LOC

```text
[ ] simulator accepts batch x0
[ ] exact bridge accepts batch x0
[ ] reuse ScoreNet as hnet
[ ] exact heat-kernel corrector target
[ ] alternating corrector/controller training
[ ] 1-seed smoke run
[ ] 5-seed headline run
[ ] fill sphere IASBS(non-Dirac) column
```

This is the most important remaining non-Dirac experiment because it directly compares against R-ASBS under a common Haar source.

## Phase 3 — non-Dirac Ising, ~50–80 LOC

```text
[ ] simulator accepts source index vector
[ ] exact propagation accepts p0
[ ] bridge accepts per-sample X0
[ ] reuse SwapController as corrector
[ ] kernel-ratio corrector labels
[ ] L=4 exact/IPF validation
[ ] L=4 headline run
[ ] L=5 headline run
[ ] fill Table 1 Ising cells
```

## Phase 4 — occupation m=4, ~60–90 LOC

```text
[ ] source = uniform mixture {N e_c}
[ ] corrector transfer-ratio labels
[ ] reuse OccupationController as corrector
[ ] substitute learned corrector into A^(t) label
[ ] exact IPF validation
[ ] fill Table 1 occupation cell
```

## Phase 5 — occupation scale, first attempt with existing architecture

```text
[ ] vectorize source component c per trajectory
[ ] reuse ScaleController as corrector
[ ] stochastic A^(t) labels query learned terminal ratios
[ ] m=32
[ ] if stable: m=128
[ ] if stable: m=1000
[ ] fill Table 2
```

Do not implement a more expressive scalable pairwise controller unless this phase demonstrates that the rank-1 family is the actual bottleneck.

---

# 11. Smoke tests before expensive runs

Each new branch should have a tiny run that can finish quickly.

## Sphere

```text
steps=32
iters=50
batch=256
mb=512
seeds=1
samples=5000
```

Check:

```text
finite losses
no NaN heat-kernel ratios
corrector norm finite
controller norm finite
sphere constraint exact
both hemispheres occupied
```

## Ising

Use $L=4$, ~50 outer iterations.

Check:

```text
all states remain in Omega_n,k
corrector labels positive and finite
mean predicted ratio not exploding
exact propagated law sums to 1
```

## Occupation

Use $m=N=4$ first.

Check:

```text
particle number exactly conserved
corrector ratio agrees with exact-IPF ratio direction
A^(t) label finite
TV improves from uncontrolled reference
```

Only after these pass launch the headline runs.

---

# 12. Scientific pass/fail interpretation

The desired outcome is not “every blank cell must beat Dirac IASBS.” The non-Dirac experiment answers a different question:

> Does IASBS remain a functioning Schrödinger-bridge sampler when the source is genuinely distributed and the missing terminal factor is supplied by a learned corrector?

Therefore:

- **Sphere non-Dirac succeeds** if it remains balanced and reaches the target distribution at reasonable error.
- **Ising/occupation small succeeds** if exact-law TV is good and the learned corrector agrees with an exact SB/IPF check.
- **Large occupation succeeds** if KS metrics remain controlled as $m$ scales.
- If large occupation degrades while the exact small problem passes, that diagnoses the scalable neural parameterization, not the intertwining theorem.
- R-ASBS port values belong in the main table only after the reproduction discrepancy is materially resolved.

---

# 13. Final expected Section 6 state after this plan

Table 1:

```text
benchmark                 IASBS        IASBS(non-Dirac)     DAM
Ising 4x4                 existing     NEW                  blank
Ising 5x5                 existing     NEW                  blank
Occupation 4              existing     NEW                  blank
```

Table 2:

```text
m=N           IASBS KSs             IASBS(non-Dirac) KSs     DAM KSs
32            existing              NEW                      blank
128           existing              NEW                      blank
1000          existing              NEW                      blank
```

Sphere:

```text
metric              target   iid   R-ASBS              IASBS      IASBS(non-Dirac)
north mass          existing        published 0.438      existing   NEW
KS(x3)                            port only if valid     existing   NEW
W1(x3)                            port only if valid     existing   NEW
KS(phi)                           port only if valid     existing   NEW
|Delta E[x3^2]|                  port only if valid     existing   NEW
```

At that point the only intentionally incomplete experimental column in the paper is DAM, which can be addressed separately.
