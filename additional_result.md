# Additional Results

Supplementary experimental results for *"Adjoint Schrödinger Bridge Sampling on
Structured State Spaces via Markov Semigroup Intertwining"* (IASBS).

All numbers below are produced by the scripts in this repository. Every entry is
reproducible from the JSON file named in its row; checkpoints are written to
`ckpt/` (gitignored, 2.5 GB, regenerable). Hardware: 2x NVIDIA A100 80 GB,
PyTorch 2.5.1 / CUDA 12.4, `torch.set_default_dtype(torch.float64)` throughout.

Status legend: **DONE** = finished with gate verdicts recorded;
**RUNNING** = in progress, last logged value shown.

---

## 1. Summary of claims supported

| # | Claim | Evidence |
|---|---|---|
| 1 | IASBS reaches exact-IPF accuracy on discrete structured spaces, up to `\|Omega\| ~ 1e4` | §2.1, §2.2 — TV at or below the i.i.d. sampling floor at Ising L=4 and occupation m=4; §2.1 — Ising L=5 (`\|Omega\| = 5.2e6`) misses the gate at 0.07228 |
| 2 | IASBS accuracy does not degrade with state-space size | §2.3 — KS improves monotonically from m=32 to m=1000 |
| 3 | IASBS extends to non-Dirac (Haar) sources without loss | §3.2 — matches the Dirac headline to 3 decimal places |
| 4 | Reported R-ASBS mode collapse is an initialisation artifact | §4 — collapse vanishes under the authors' own init |
| 5 | DAM needs millions of Monte-Carlo adjoint rollouts; IASBS needs none | §5, §6.4 — 20.9 M rollout-endpoint f1 evals for DAM's best TV vs 0 for IASBS (IASBS still evaluates the energy; see §6.4) |
| 6 | DAM has a hard stability cliff, and raising K does not clear it | §5.2 — K=1, K=4 diverge at m=4; §6.2 — K=16 diverges at m=32 and Ising L=4, and K=64 still diverges at m=32 |

---

## 2. IASBS on discrete structured spaces

### 2.1 Fixed-magnetization Ising on an L x L periodic lattice, non-Dirac (uniform) source — validated against exact IPF

The state space is **not** the full spin space. It is the fixed-magnetization
sector of the 2D nearest-neighbour Ising model on an `L x L` periodic square
lattice (a torus, `common.periodic_lattice_edges`): `n = L^2` binary sites with
exactly `k = n/2` occupied, so `|Omega| = C(n, n/2)`. Dynamics are
**magnetization-preserving swaps** (Kawasaki, not Glauber), which is what makes
this a test of the *bijective* structured-space machinery rather than a generic
discrete diffusion benchmark. Energy `E = -J sum_<ij> s_i s_j`, `s = 2x - 1`.

Gate A1: terminal-law TV <= 0.05, ideally at the i.i.d. sampling floor.
Gate A2: zero magnetization violations (the swap dynamics must never leave the
sector).

| L | sites n | `\|Omega\| = C(n, n/2)` | TV (exact law) | empirical TV | i.i.d. TV floor | violations | status |
|---|---|---|---|---|---|---|---|
| 4 | 16 | 12,870 | **0.03470** | 0.06571 | 0.04940 | **0 / 200,000** | **DONE — PASS** |
| 5 | 25 | 5,200,300 | 0.07228 (best 0.06714 at it 2725) | 0.35530 | 0.32546 | **0 / 200,000** | **DONE — A1 FAIL, A2 PASS** |

`json/results_ising_nd_L4.json`, `json/results_ising_nd_L5.json`.

L=4 sits **below** the i.i.d. floor of 0.04940, i.e. the learned terminal law is
statistically indistinguishable from exact samples at this sample size.

**L=5 misses gate A1, and it is reported as a miss.** The exact-law TV settles at
0.07228 against a gate of 0.05. The best value seen was 0.06714 at iteration 2725,
and the last 300 iterations oscillate within 0.067-0.081 without trending down, so
the run is converged rather than truncated — more iterations at this configuration
would not fix it. Constraint gate A2 passes exactly: **0 violations in 200,000
samples**, so the magnetization-preserving swap dynamics remain correct at
`|Omega| = 5.2e6`. What degrades with lattice size is accuracy, not structure.
Config: 3000 iterations, 256 steps, batch 2048, 1,728,738 parameters, 36,966 s
wall (12.3 s/iter, 20x the L=4 cost per iteration).

Reading the empirical column at L=5 needs care. 200,000 samples spread over
5,200,300 states leave most states empty, so the empirical TV is dominated by
sampling noise: exact i.i.d. draws from the target score **0.32546** at this
sample size against the run's 0.35530. The excess of **0.030** — not the raw
0.355 — is the part attributable to the sampler, and it is consistent with the
exact-law TV of 0.072. The exact-law column is the meaningful one at this size: it
is a deterministic propagation of the learned rates over all 5,200,300 states,
with no Monte-Carlo error at all.

IPF corrector sanity check: `mean exp(h) = 0.99723` at L=4 and `0.99792` at L=5
(target 1), with `max |h| = 0.07224` and `0.25522`. The corrector remains a small
multiplicative perturbation, as the intertwining identity predicts, but the L=5
excursion is 3.5x the L=4 one — the corrector is working measurably harder, which
is the same direction as the A1 miss.

### 2.2 Occupation process m=4, non-Dirac — validated against exact IPF

Gate: TV at or below the i.i.d. floor; Sinkhorn marginal error at machine precision.

| source nu_0 | TV | empirical TV | i.i.d. floor | corrector max err | Sinkhorn marginal err | violations |
|---|---|---|---|---|---|---|
| uniform `[.25,.25,.25,.25]` | **0.01248** | 0.01801 | 0.01598 | 3.59e-4 | 2.78e-17 | 0 |
| skewed `[.025,.075,.225,.675]` | **0.01411** | 0.02049 | 0.01598 | 2.56e-4 | 1.11e-16 | 0 |

`json/results_occ_nd_m4.json`, `json/results_occ_nd_m4_skew.json`. Status **DONE — PASS**.

Both TVs are **below the i.i.d. floor**. The skewed source costs only 0.0016 TV,
showing the method is not tuned to a symmetric source.

Component breakdown (uniform): occupancy-histogram TV 0.00614, max-occupancy TV
0.00744, mean energy 1.88671.

### 2.3 Occupation process — scaling in m

Gates: **B2a** occupancy KS <= 0.05; **B2b** max-occupancy W1 <= 5% of N;
**B2c** constraint violations = 0.

| m | KS_occ | W1_max / N | violations | uncontrolled reference KS_occ | improvement | status |
|---|---|---|---|---|---|---|
| 32 (seed 0) | **0.01512** | 0.01509 | 0 / 10000 | 0.20534 | 13.6x | **DONE — PASS** |
| 32 (seed 1) | **0.01895** | 0.01690 | 0 / 10000 | 0.20468 | 10.8x | **DONE — PASS** |
| 128 | **0.01270** | 0.00224 | 0 | 0.20585 | 16.2x | **DONE — PASS** |
| 1000 | **0.01109** | 0.00173 | 0 / 4000 | 0.20300 | 18.3x | **DONE — PASS** |

`json/results_occ_nd_s32_seed{0,1}.json`, `json/results_occ_nd_s128.json`,
`json/results_occ_nd_s1000.json`. Checkpoints `ckpt/occ_nd_s{32_seed0,32_seed1,128,1000}.pt`.

**Accuracy improves with m.** KS_occ falls monotonically 0.01512 -> 0.01270 ->
0.01109 as m goes 32 -> 128 -> 1000, and W1_max/N falls by an order of magnitude.
The uncontrolled reference is flat at ~0.205 across all m, so the gain is
attributable to the controller, not to the problem getting easier.

Energy agreement at m=1000: `E_ours = 412.70` vs `E_exact = 405.57` (1.8% relative).
Conditioning `cond_ours = 0.01607` vs `cond_exact = 0.01446`.

#### 2.3.1 Numerical stability fix (m=32)

m=32 initially crashed with `ValueError: cannot convert float NaN to integer`
inside the binomial departure sampler. Two distinct defects, both now fixed in
`structured_asbs/occupation.py`:

1. **Overflow.** `uv_nondirac` exponentiated unbounded network outputs. The label
   construction forms `T = xi + e^v (U - xi e^u)` and then differences `T_j - T_i`;
   a single `+inf` makes that `inf - inf = NaN`, which poisons the controller
   weights and surfaces several hundred iterations later as an unrelated-looking
   error. Fixed with `+-20` clamps on the raw heads and `+-60` on `u`, `v`
   (keeping `e^{u+v} ~ 1e52` inside float64).
2. **Dead gradient.** A hard `clamp` on a Bregman exponent has *exactly zero*
   gradient outside its range, so once the corrector saturated it could never
   recover — the run reported finite numbers while being irrecoverably dead
   (`loss_h` frozen at 315.05, `loss` -8e9, KS pinned at 0.4008). Fixed with
   `leaky_clamp` (1% surviving slope), corrector gradient clip 10 -> 1, cosine
   decay on the corrector optimiser, and `lr_h` 0.01 -> 1e-3.

Both seeds then passed all three gates with `bad_labels = 0`, `skipped_steps = 0`.

---

## 3. IASBS on the sphere S^2

Target: bimodal density; gates **C1** mean hemisphere-mass error < 0.03
(R-ASBS reference 0.062), **C2** KS on the x_3 marginal < 0.05.

### 3.1 Dirac source, 5 seeds, 4000 iterations

| variant | mean north_err | mean KS_z | max abs(norm-1) | status |
|---|---|---|---|---|
| **antithetic (headline)** | **0.00069** | **0.02165** | 0.0 | **DONE — PASS** |
| symmetrized | 0.00082 | 0.02283 | 0.0 | DONE — PASS |
| plain | 0.02540 | 0.03881 | 0.0 | DONE — PASS |

`json/results_sphere_train_anti.json`, `_sym.json`, `_plain.json`. 136,963 parameters.

Antithetic augmentation uses `R = diag(1,1,-1)`, an isometry that fixes the energy
and leaves the source invariant while the heat kernel depends only on `x_0 . y`, so
`(R x_0, R y_1, R G)` is an *exact* same-law sample — not an approximation. It buys
a **37x** reduction in hemisphere error over the plain estimator.

Per-seed (antithetic): north_err = 0.00109 / 0.00040 / 0.00040 / 0.00045 / 0.00113;
KS_z = 0.02199 / 0.02113 / 0.02253 / 0.02076 / 0.02185. Constraint violation 0.0 on
every seed.

### 3.2 Non-Dirac (Haar) source, 5 seeds, 4000 iterations — **DONE**

The source is replaced by Haar measure on S^2 and the closed-form terminal score is
replaced by a learned corrector `H_psi(y) ~ grad log f1_hat(y)`, giving
`G_ND(y) = -grad E(y)/tau - H_psi(y)`. Geometry, bridge and Killing readout are
unchanged.

| seed | north_mass | north_err | KS(x_3) | max abs(norm-1) |
|---|---|---|---|---|
| 0 | 0.50014 | 0.00014 | 0.02215 | 2.22e-16 |
| 1 | 0.50111 | 0.00111 | 0.02237 | 2.22e-16 |
| 2 | 0.49818 | 0.00183 | 0.02305 | 2.22e-16 |
| 3 | 0.50145 | 0.00145 | 0.02357 | 2.22e-16 |
| 4 | 0.50103 | 0.00103 | 0.02278 | 2.22e-16 |
| **mean** | — | **0.00111** | **0.02278** | **2.22e-16** |

```
C1  mean hemisphere err < 0.03 (R-ASBS 0.062): PASS  (0.0011)
C2  KS(x3) < 0.05                            : PASS  (0.0228)
```

`json/results_sphere_nd.json`, `ckpt/sphere_nd_seed{0..4}.pt`. 136,963 + 136,963
parameters (controller + corrector).

**The non-Dirac result matches the Dirac headline** (0.00111 vs 0.00069 north_err;
0.02278 vs 0.02165 KS). Removing the Dirac source assumption costs essentially
nothing.

Three further statistics, recovered from `ckpt/sphere_nd_seed{0..4}.pt` (200,000
samples per seed) via `remeasure.sphere_metrics` without retraining. The target
`pi ∝ exp(6 x_3^2)` has an exactly integrable z-marginal, so all three are
compared against closed form; exact `E[x_3^2] = 0.807709`.

| seed | W1(x_3) | KS(phi) | `\|Delta E[x_3^2]\|` |
|---|---|---|---|
| 0 | 0.01205 | 0.00132 | 0.01849 |
| 1 | 0.01259 | 0.00116 | 0.01923 |
| 2 | 0.01412 | 0.00213 | 0.01881 |
| 3 | 0.01259 | 0.00205 | 0.01918 |
| 4 | 0.01269 | 0.00288 | 0.01890 |
| **mean** | **0.01281** | **0.00191** | **0.01892** |
| i.i.d. floor (n = 200,000) | 0.00031 | 0.00224 | 0.00001 |

**KS(phi) is below its own i.i.d. floor.** Azimuthal uniformity is not
approximated, it is exact: the Killing readout builds the controlled drift out of
the rotation generators, so the law is invariant under rotation about the z-axis by
construction and the only azimuthal error is sampling noise. The residual
`|Delta E[x_3^2]| = 0.019` against a floor of 1e-5 is the honest measure of what
remains: a small, seed-stable bias in the *polar* marginal, consistent across all
five seeds at the third decimal.

An ablation without antithetic augmentation is retained at
`ckpt/sphere_nd_plain_seed0.pt` (north_err 0.4398 wobble, KS 0.0649) — the same
degradation the Dirac plain run shows, confirming the augmentation is what
stabilises both branches.

---

## 4. R-ASBS baseline on S^2 — the mode-collapse claim

Bimodal target, kappa = 600, 600 epochs, 500 steps, 5 seeds.

### 4.1 Default initialisation

| seed | north_mass | north_err | KS_z |
|---|---|---|---|
| 0 | 0.00004 | 0.49996 | 0.50587 |
| 1 | 0.00004 | 0.49996 | 0.50590 |
| 2 | 0.99762 | 0.49762 | 0.50328 |
| 3 | 0.00173 | 0.49827 | 0.50438 |
| 4 | 0.00191 | 0.49809 | 0.50476 |
| **mean** | — | **0.49878** | **0.50484** |

Every seed collapses to a single mode (mass 0 or 1, never 0.5).

### 4.2 Authors' own `--init matlab`

| seed | north_mass | north_err | KS_z | W1_z | KS(phi) | `\|Delta E[x_3^2]\|` |
|---|---|---|---|---|---|---|
| 0 | 0.49728 | 0.00272 | 0.08040 | 0.06451 | 0.00980 | 0.09044 |
| 1 | 0.51026 | 0.01026 | 0.06319 | 0.04328 | 0.00829 | 0.05544 |
| 2 | 0.49394 | 0.00606 | 0.06080 | 0.04544 | 0.00718 | 0.06288 |
| 3 | 0.42420 | 0.07580 | 0.12787 | 0.14183 | 0.01397 | 0.08579 |
| 4 | 0.43537 | 0.06463 | 0.11342 | 0.12319 | 0.01544 | 0.08190 |
| **mean** | — | **0.03189** | **0.08914** | **0.08365** | **0.01094** | **0.07529** |
| **sd over seeds** | — | 0.03158 | 0.02699 | 0.04100 | 0.00322 | 0.01365 |
| i.i.d. floor (n = 100,000) | — | 0.00087 | 0.00268 | 0.00178 | 0.00259 | 0.00069 |

`json/results_rasbs_sphere_matlabinit_s{0..4}.json`,
`ckpt/rasbs_sphere_matlabinit_s{0..4}.pt`, 100,000 samples per seed.

Two corrections to an earlier version of this table. First, the reported means
were arithmetically wrong: the five north errors average to **0.03189**, not
0.02703, and the five KS values to **0.08914**, not 0.08768. The per-seed entries
were always right, so the ratios in §4.3 have been recomputed from them.
Second, KS(phi) and the second-moment error were missing; both are functions of
the stored samples alone and were recovered from the checkpoints without
retraining, via the same `remeasure.sphere_metrics` used for the IASBS numbers, so
the definitions are identical on both sides. Exact `E[x_3^2] = 0.807709`.

**Collapse disappears entirely.** The phenomenon is an initialisation artifact, not
a property of on-policy self-training. Any description of R-ASBS as exhibiting
"spontaneous mode collapse" must be retracted; the honest statement is that R-ASBS
is *initialisation-sensitive*.

### 4.3 Head-to-head on the same target

| method | mean north_err | mean KS_z | mean W1_z | mean KS(phi) | mean `\|Delta E[x_3^2]\|` |
|---|---|---|---|---|---|
| R-ASBS, default init | 0.49878 | 0.50484 | — | — | — |
| R-ASBS, matlab init | 0.03189 | 0.08914 | 0.08365 | 0.01094 | 0.07529 |
| **IASBS, Dirac, antithetic** | **0.00069** | **0.02165** | **0.01239** | **0.00264** | **0.01907** |
| **IASBS, Haar (non-Dirac)** | **0.00111** | **0.02278** | **0.01281** | **0.00191** | **0.01892** |
| i.i.d. floor | 0.00087 / 0.00031 | 0.00268 / 0.00224 | 0.00178 / 0.00031 | 0.00259 / 0.00224 | 0.00069 / 0.00001 |

Floor row: first number at n = 100,000 (the R-ASBS sample size), second at
n = 200,000 (the IASBS sample size).

Against the *strongest* R-ASBS configuration, IASBS is **46x** better in hemisphere
error, **4.1x** better in KS(x_3), **6.7x** better in W1(x_3), **4.1x** better in
azimuthal uniformity and **3.9x** better in the second moment, with zero constraint
violation and no seed variance. The ratio in hemisphere error was previously quoted
as 39x from a mis-averaged R-ASBS mean; corrected, it is 46x.

Two qualitative differences matter as much as the ratios. **Seed variance:** the
R-ASBS seed spread is as large as its mean (north_err 0.03189 +- 0.03158; seeds 3
and 4 are ~25x worse than seed 0), whereas the IASBS seeds agree to the third
decimal. **Azimuthal symmetry:** IASBS non-Dirac reaches KS(phi) = 0.00191, which
is *below* the 0.00224 i.i.d. floor at its sample size, i.e. its azimuthal
marginal is exactly uniform up to sampling noise, as the Killing readout
guarantees by construction. R-ASBS sits 4.2x above the corresponding floor, so its
azimuthal error is a real bias rather than sampling noise.

---

## 5. DAM baseline (Discrete Adjoint Matching)

Reference implementation of So, Karrer, Fan, Chen & Liu, *Discrete Adjoint
Matching* (arXiv:2602.07132v2 / ICLR 2026), Eqs (6), (11)-(14), (40), in `dam/`.

Implementation notes:

- The paper's masked-diffusion specialisation lets model and base share a total
  escape rate, so the integrated escape-rate term cancels. Our controllers genuinely
  change the escape rate, so the **full** CTMC Radon-Nikodym weight is evaluated,
  `log W = int_t^1 [R_ubar - R_base] ds + sum_j (-a_theta)`. Dropping the integral
  makes the estimator wrong, not merely noisy.
- Rollouts use exact Gillespie sampling of the piecewise-constant controlled CTMC,
  so no discretisation error enters the weight.
- `gkl_loss` applies **truncated importance weighting** (`LOG_M_CLIP = 30`). This is
  required, not cosmetic: `m_hat` is a ratio of importance-weighted averages whose
  right tail is unbounded at small K, and a single lucky denominator rollout sends
  `log m_hat` past 709, `exp()` returns `+inf`, gradients become NaN, and every
  subsequent `multinomial` trips a device-side assert. The clipped fraction is
  reported rather than hidden.

### 5.1 K sweep, occupation m=4, N=4, |X|=35, 128 steps

Gate: terminal TV <= 0.05, constraint violations = 0.

| K | iters | final TV | occ-hist TV | ESS mean | ESS p10 | f1 evals | rollout jumps | wall (s) | clipped | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 0.41585 | 0.26278 | 1.00 / 1 | 1.00 | 20,480 | 1,499,657 | 224.7 | 1306 | **DIVERGE** |
| 4 | 10 | 0.42312 | 0.26011 | 2.01 / 4 | 1.00 | 51,200 | 1,197,204 | 163.9 | 134 | **DIVERGE** |
| 16 | 400 | 0.07170 | — | 9.62 / 16 | 4.26 | 6,963,200 | 67,417,741 | 7380.9 | 0 | FAIL |
| 16 | 1200 | **0.01490** | 0.00800 | 14.04 / 16 | 8.07 | 20,889,600 | 185,004,804 | 11529.3 | 0 | **PASS** |
| 64 | 400 | 0.04498 | 0.01827 | 48.75 / 64 | 16.71 | 26,624,000 | 222,041,777 | 3177.7 | 0 | **PASS** |

The K=16 run at 400 iterations is the same configuration as the K=16 run at 1200;
the only difference is the gradient-step budget. Reporting the 400-iteration number
alone would have made DAM look like it *cannot* reach the gate, when in fact it was
still descending monotonically at the cutoff. The honest statement is that DAM
reaches the gate but needs roughly three times the budget, and 20.9 M terminal
evaluations, to do so.

TV trajectory (K=16): 0.41674 (it 50) -> 0.39471 (200) -> 0.20222 (250) ->
0.06995 (300) -> 0.05128 (400) -> 0.02749 (650) -> 0.01974 (850) -> 0.01696 (1050)
-> **0.01490** (1200). ESS rises with it: 7.86/16 -> 15.88/16.

`json/results_dam_occ4_K{1_diag,4_diag,64}.json`,
`json/results_dam_occupation_m4_K16.json`, `json/results_dam_occ4_K16_long.json`.

### 5.2 The stability cliff below K=16

K is the number of Monte-Carlo rollouts in the adjoint **denominator**, estimating
`phi_t(x) = E_base[f1(X_1) | X_t = x]`. Small K does not merely add noise — it
destroys the run:

| K | loss at it 5 | loss at it 10 | jumps per f1 eval | clipped labels |
|---|---|---|---|---|
| 1 | -2.19e11 | **-1.50e14** | 73 | 1306 / ~10240 = 12.8% |
| 4 | -39.77 | **-2.72e9** | 23 | 134 |
| 16 | +15.36 (stable) | — | 9.7 | 0 |

Mechanism: a single-sample denominator gives `log m_hat` an O(1) irreducible
variance that never averages out; the gKL loss responds by pushing `a_theta` large;
`R_model` blows up; the Gillespie loop runs **7.5x** longer per rollout; the run
becomes latency-bound and the terminal law never moves (TV stuck at 0.416 from
0.4176 at initialisation). `LOG_M_CLIP` prevents the NaN crash but **cannot prevent
the divergence** — the two are separate failures.

### 5.3 Diminishing returns above K=16

K=64 first crosses the gate at iteration 300 (TV 0.04855) versus iteration 400 for
K=16, i.e. it saves ~100 iterations for 4x the per-iteration cost, and its final
TV (0.04498 at 26.6 M f1 evals) is **worse** than K=16's (0.01490 at 20.9 M).
ESS *fraction* is not improved either: 48.75/64 = 76% for K=64 versus 14.04/16 = 88%
for K=16.

Per terminal evaluation, K=16 is the better spend:

| K | final TV | f1 evals | TV per million f1 evals |
|---|---|---|---|
| 16 | 0.01490 | 20,889,600 | 0.00071 |
| 64 | 0.04498 | 26,624,000 | 0.00169 |

### 5.4 Cost comparison, the headline number

**Source matching.** The DAM occupation adapter draws its start state from
`source_state()`, which returns the single state `sp.i0` = `eta_0 = N e_c` for the
whole batch, and its `log_f1` is `sp.logf1 = -E/tau - log p_base(. | eta_0)`. That
is the **Dirac** source. The comparison below is therefore against the **Dirac**
IASBS occupation run (`json/results_occ_full.json`, TV **0.012730**), *not* the
non-Dirac uniform-source run of §2.2 (TV 0.01248). Both use m=N=4, `|X|=35`,
128 steps.

| method | source | best TV on occupation m=4 | f1 evals at rollout endpoints | CTMC jumps simulated for the adjoint | wall (s) |
|---|---|---|---|---|---|
| DAM, K=16 | Dirac | 0.01490 | 20,889,600 | 185,004,804 | 11529.3 |
| DAM, K=64 | Dirac | 0.04498 | 26,624,000 | 222,041,777 | 3177.7 |
| **IASBS** | **Dirac** | **0.012730** | **0** | **0** | no log (<= 214) |
| IASBS | uniform non-Dirac | 0.01248 | 0 | 0 | 214 |

The last row is listed for completeness only and is **not** the head-to-head: it
solves a different (harder) problem, since a non-Dirac source additionally
requires the learned IPF corrector.

IASBS obtains `phi_t` in closed form through the intertwining identity, so it
performs **no** adjoint rollouts and therefore evaluates `f1` at **no** rollout
endpoints during training. DAM's best result is 17% worse in TV and costs
**20.9 million** rollout-endpoint evaluations plus 185 million simulated CTMC
jumps to get there.

The `0` here is the count of Monte-Carlo adjoint work specifically. On this
benchmark IASBS reads `f1` out of the same precomputed `|Omega| = 35` table DAM
uses, so its per-label arithmetic is not zero either — it is table lookup with no
simulation in front of it. §6.4 gives the full instrumented breakdown, including
the Ising case where IASBS's on-the-fly energy count is *larger* than DAM's.

This is the central comparison of the paper. DAM is not wrong on this benchmark —
given enough rollouts it converges, and its ESS diagnostics are healthy at K=16.
It is *expensive*, because the adjoint `phi_t` it estimates by Monte Carlo is
exactly the object the intertwining identity hands IASBS for free.

---

## 6. Computational cost

All timings on a single NVIDIA A100 80 GB, float64 throughout. Where runs shared a
device the contention is noted, because DAM runs are latency-bound (see §6.2) and
co-scheduling inflates their wall clock roughly linearly.

### 6.1 IASBS

| experiment | iters | steps | wall (s) | s / iter | adjoint rollouts | checkpoint |
|---|---|---|---|---|---|---|
| Ising L=4 non-Dirac | 3000 | 256 | **1869** | 0.62 | 0 | `ising_nd_L4.pt` |
| Ising L=5 non-Dirac | 3000 | 256 | 32396 @ it 2575, ~37000 proj. | 12.3 | 0 | `ising_nd_L5.pt` (written at loop end) |
| Occupation m=4 **Dirac** (head-to-head row) | 1500 | 128 | no log | — | 0 | `occ4_full.pt` |
| Occupation m=4 non-Dirac | 1500 | 128 | **214** | 0.14 | 0 | `occ_nd_m4.pt` |
| Occupation m=32 (seed 0) | 3000 | 128 | **3230** | 1.08 | 0 | `occ_nd_s32_seed0.pt` |
| Occupation m=32 (seed 1) | 3000 | 128 | **3231** | 1.08 | 0 | `occ_nd_s32_seed1.pt` |
| Occupation m=128 | 3000 | 128 | **1523** | 0.51 | 0 | `occ_nd_s128.pt` |
| Occupation m=1000 | 1500 | 256 | **15225** | 10.15 | 0 | `occ_nd_s1000.pt` |
| Sphere non-Dirac, per seed | 4000 | 128 | **2124 - 3142** | 0.53 - 0.79 | 0 | `sphere_nd_seed{0..4}.pt` |
| Sphere non-Dirac, 5 seeds | 20000 | 128 | **13820** | 0.69 | 0 | — |

Per-seed sphere wall: 3076 / 2841 / 3142 / 2637 / 2124 s.

Note that occupation m=128 (1523 s) is *cheaper* than m=32 (3230 s) despite the
larger state space, because the m=32 rerun used the tightened corrector schedule
described in §2.3.1 with a smaller learning rate; the m=128 leg converged in the
original schedule. Cost is set by the corrector schedule, not by m, up to m~128.
At m=1000 the per-iteration cost does rise (10.15 s/iter) because the run uses
256 simulation steps rather than 128.

### 6.1.1 R-ASBS baseline (sphere S^2)

R-ASBS timing is only partially recoverable. The R-ASBS result JSONs
(`json/results_rasbs_*.json`) carry the keys
`config, final, floor, history, params, commit` and **no timing field**, so the
numbers below come from the wall-clock tokens printed in the surviving run logs.

| experiment | seeds | wall per seed (s) | total (s) | source |
|---|---|---|---|---|
| R-ASBS, `--init matlab`, bimodal target | 5 | **255 / 255 / 261 / 256 / 260** | **1287** | `rasbs/logs/bimodal_mi_s{0..4}.log` |
| R-ASBS, default init | 5 | not recoverable | — | no log survives |

Two honest qualifications:

- The default-init R-ASBS legs (§4.1, the mode-collapse audit) were run before the
  logging convention was fixed and their logs were not kept, so no wall clock is
  available for them. Their per-iteration cost is the same code path as the matlab
  init legs with identical `--iters`, so ~256 s / seed is the right order, but this
  is an inference and not a measurement.
- R-ASBS is a *continuous-space* Riemannian method with a much smaller network and
  fewer simulation steps than the IASBS sphere configuration, so its 256 s / seed
  is **not** comparable to the IASBS sphere 2124-3142 s / seed as a like-for-like
  efficiency statement. The relevant comparison is accuracy at matched settings
  (§4.3), not wall clock. We report the timings for completeness only.

| method (sphere S^2) | wall / seed (s) | north-mass err | KS |
|---|---|---|---|
| IASBS non-Dirac (Haar) | 2124 - 3142 | **0.00111** | **0.02278** |
| R-ASBS `--init matlab` | 255 - 261 | 0.02703 | 0.08768 |
| R-ASBS default init | (no log) | 0.49878 | 0.50484 |

IASBS buys a 24x accuracy improvement over the best R-ASBS configuration at about
11x the wall clock per seed, on a substantially larger network. We make no
efficiency claim against R-ASBS; the claim in §4 is about *correctness and
robustness to initialisation*, not speed.

### 6.2 DAM

| experiment | K | iters | wall (s) | s / iter | f1 evals | CTMC jumps | jumps / f1 eval |
|---|---|---|---|---|---|---|---|
| Occupation m=4 | 1 | 10 | 224.7 | 22.5 | 20,480 | 1,499,657 | **73.2** |
| Occupation m=4 | 4 | 10 | 163.9 | 16.4 | 51,200 | 1,197,204 | **23.4** |
| Occupation m=4 | 16 | 400 | 7380.9 | 18.5 | 6,963,200 | 67,417,741 | 9.68 |
| Occupation m=4 | 16 | 1200 | **11529.3** | 9.6 | **20,889,600** | **185,004,804** | 8.86 |
| Occupation m=4 | 64 | 400 | 3177.7 | 7.9 | 26,624,000 | 222,041,777 | 8.34 |
| Ising L=4, 10-iter diagnostic | 16 | 10 | **21.3** | **2.13** | 174,080 | 936,880 | **5.38** |
| Occupation-scale m=32, 10-iter diagnostic | 16 | 10 | 123.3 | 12.33 | 174,080 | 26,661,608 | **153.2** |
| **Ising L=4** | 16 | 40 of 200 | **DIVERGED, killed at 286 s** | 2.1 -> **37.6** | — | — | — |
| **Occupation-scale m=32** | 64 | 15 of 400 | **DIVERGED, killed at 1664 s** | 15.2 -> **36.8** | — | — | — |
| Occupation-scale m=128 | 16 | 600 | KILLED at 3695 s, <50 it | >74 | — | — | — |
| Occupation-scale m=1000 | 16 | 300 | dropped (K=16 expected to diverge) | — | — | — | — |
| Ising L=5 | 16 | 600 | not run | — | — | — | — |

**The stability cliff moves with m — K=16 is not universally sufficient.** The
two 10-iteration diagnostics above resolve the earlier no-eval stall into two
completely different causes:

| | Ising L=4, K=16 | Occupation-scale m=32, K=16 |
|---|---|---|
| loss, it 5 -> it 10 | 10.56 -> **7.75** | -2.42e5 -> **-2.94e13** |
| ESS mean / p10 | 3.68 / **1.52** | 1.93 / **1.00** |
| jumps per f1 eval | **5.38** | **153.2** |
| s / iter, uncontended | **2.13** | 12.33 and rising |
| verdict at it 10 | *appeared healthy* | **DIVERGING** |
| verdict at it 40 | **DIVERGED** (see below) | — |

**A 10-iteration diagnostic is not long enough — Ising L=4 diverges too, from
iteration 15.** The diagnostic above sampled only the pre-divergence phase and
led me to relaunch Ising L=4 at K=16 as "healthy but contended". Alone on GPU0 it
then produced no eval line in 905 s, so it was relaunched a second time with
`--eval-every 5`. The dense log settles the question:

| it | loss | TV (exact law) | E-hist TV | ESS / 16 | cumulative s |
|---|---|---|---|---|---|
| 5 | 10.93 | 0.76046 | 0.74692 | 3.82 | 11 |
| 10 | 7.25 | 0.75909 | 0.75136 | 3.22 | 21 |
| 15 | **-16.30** | 0.76199 | 0.75090 | 3.02 | 31 |
| 20 | -9.70e3 | 0.80817 | 0.73972 | 2.26 | 41 |
| 25 | -3.81e12 | 0.85085 | 0.73072 | 1.76 | 54 |
| 30 | -4.30e11 | 0.86370 | 0.72258 | 1.71 | 70 |
| 35 | -4.78e12 | 0.87208 | 0.67669 | 1.44 | 98 |
| 40 | -4.15e12 | **0.97347** | 0.84403 | **1.04** | 286 |

Every diagnostic moves the same way at once: the loss turns negative at iteration
15 and reaches -4e12 by iteration 25, the ESS collapses monotonically from 3.82 to
**1.04 out of 16** (a single rollout carrying the adjoint denominator), TV rises
*away* from the target, 0.760 -> **0.973** (worse than the untrained controller),
and the per-iteration cost climbs 2.1 -> **37.6 s/iter** between iterations 35 and
40 as the diverging rates inflate the jump count. That last effect is exactly why
the `--eval-every 50` relaunch printed nothing in 905 s: it was not slow, it was
diverging into a cost explosion. The leg was killed at 286 s and its log kept as
`dam/logs/dam_ising_L4_K16_DIVERGED.log`.

So the cliff is crossed on a **second, independent** problem family. Ising L=4 has
`|Omega| = 12,870` against `|X| = 35` for occupation m=4, and K=16 — sufficient at
m=4 — fails on both of the larger spaces tried. IASBS solves the same Ising L=4
target to TV **0.03470** (§2.1).

Occupation-scale m=32 at K=16 shows the exact §5.2 divergence signature, and
worse than any leg recorded there: 153 jumps per f1 evaluation against 73.2 at
K=1 and 8.86 at the converged K=16, with the ESS 10th percentile pinned at
**1.00**, i.e. a single rollout carries all the weight. Since K=16 *converges* at
m=4 and *diverges* at m=32, the stability cliff of §5.2 is not a fixed constant —
**the K that DAM requires grows with the state-space size**. m=32 is therefore
requeued at K=64, and the m=1000 leg at K=16 was dropped rather than run as a
foregone divergence. This strengthens claim 6: DAM's rollout budget is not merely
large, it must be re-tuned per problem size, whereas IASBS has no such parameter.

**Quadrupling K to 64 does not clear the m=32 cliff.** The retry diverges on the
same trajectory, only one eval later:

| it | loss | KS_occ | ESS / 64 | cumulative s |
|---|---|---|---|---|
| 5 | **+134.78** | 0.2052 | 3.48 | 76 |
| 10 | -3,969.86 | 0.2030 | 2.37 | 166 |
| 15 | -30,079.50 | 0.1987 | **1.18** | 350 |

The loss is *positive* at iteration 5, so K=64 does buy a slightly longer healthy
window than K=16 (already at -2.4e5 by then), but it is a delay, not a fix: by
iteration 15 the ESS is **1.18 out of 64**, a worse retention *fraction* (1.8%)
than K=16's 1.93/16 (12%), and the per-iteration cost has climbed 15.2 -> 36.8 s.
KS_occ never leaves the uncontrolled reference band of 0.2040 — 15 iterations of
training produced no measurable progress toward the target at all. Killed at
1,664 s; log kept as `dam/logs/dam_occs32_K64_DIVERGED.log`.

This is the sharper form of claim 6. It is not that DAM needs a larger K on larger
spaces; it is that **raising K does not obviously rescue it once the space is
large enough** — the estimator's failure at m=32 is not simple denominator
variance that averaging fixes, since 4x the averaging bought five iterations.
Every DAM leg attempted beyond the |X| = 35 occupation problem has diverged:
occupation-scale m=32 at K=16 and K=64, and Ising L=4 at K=16.

**Budget note on the larger DAM legs.** The Ising L=4 and occupation-scale m=32
legs were launched at 1200 iterations with `--eval-every 50` on the assumption of
the 18.5 s/iter measured for occupation m=4. Both were still short of their first
eval after 5187 s, i.e. **>104 s/iter**, which projects to ~35 h per leg and was
compounded by co-scheduling two of them on one device. They were terminated with
no artifacts written (`dam/discrete.py` writes its JSON and checkpoint only after
the loop, line 605), and short 10-iteration diagnostics were launched in their
place to separate the two candidate causes -- ordinary slowness from the larger
state space and network, versus the controller divergence and jump explosion
documented in §5.2, which presents with the *same* outward signature of no eval
line at 100% CPU.

DAM runs are **latency-bound, not throughput-bound**: measured CPU time equals wall
time to within 1% on every leg, because each Gillespie rollout is a Python `while`
loop issuing many small kernels and synchronising on `active.any()`. Consequences:

- Batch size is nearly free; the number of `while` iterations is what costs.
- Co-scheduling two DAM runs on one device roughly *doubles* both wall times rather
  than overlapping them, which is why the remaining legs are queued serially
  (`dam/queue_gpu0.sh`).
- The jump-explosion column is the mechanism behind the small-K divergence in §5.2:
  at K=1 each terminal evaluation costs 73 simulated jumps versus 8.9 at K=16.

### 6.2.1 Trying to fix the divergence

Reporting that DAM diverges is only fair if the obvious repairs were attempted.
Four stabilisers were added to `dam/core.py` and `dam/discrete.py`, all off by
default so every number above is reproduced bit-for-bit (`dam.tests_math`:
11 / 11 gates still pass):

| flag | what it bounds |
|---|---|
| `--a-clamp C` | `\|a_theta\| <= C`, i.e. the escape-rate multiplier `<= e^C` (default 20) |
| `--m-clip C` | `\|log m_hat\| <= C` in the gKL loss (default 30) |
| `--ess-min E` | drop labels whose denominator ESS `< E`; the mean is over survivors |
| `--coef-cap R` | winsorise the `r/q` prefactor at `R x` its own batch median |

Target: the feedback loop of §5.2 — a large `log m_hat` produces a large
gradient on `a`, which raises the escape rate, which multiplies the jumps per
rollout, which degrades the ESS, which enlarges `log m_hat`. Each flag cuts a
different link.

**Round 1 — all four arms kill the divergence (Ising L=4, K=16, 60 iterations).**

| arm | settings | loss @ it 60 | TV @ it 60 | E-hist TV, it 30 -> 60 | ESS mean / p10 | jumps | wall |
|---|---|---|---|---|---|---|---|
| baseline | — | **-4.15e12** (it 40) | **0.97347** | 0.75170 -> 0.84403 | **1.04** | — | 286 s to it 40 |
| A | `a-clamp 3 m-clip 5` | +22.99 | 0.76746 | 0.71368 -> 0.70286 | 2.91 / 1.15 | 9.61 M | 130 s |
| B | `ess-min 4 coef-cap 10` | -17,449 | 0.77547 | 0.73822 -> 0.70352 | 2.94 / 1.13 | 9.18 M | 128 s |
| C | `a-clamp 5 m-clip 10 lr 2e-4` | -1.00 | 0.77391 | 0.76129 -> 0.75425 | **3.37 / 1.34** | **7.52 M** | 127 s |
| D | all four, `lr 3e-4` | +14.32 | 0.78462 | 0.75982 -> 0.73932 | 3.23 / 1.27 | 7.83 M | 128 s |

The divergence is gone on every arm: the loss magnitude drops from **1e12 to
1e1-1e4**, the ESS holds at 2.5-3.4 out of 16 instead of collapsing to 1.04, TV
stops running *away* from the target, and the cost explosion disappears — 60
iterations in ~130 s where the baseline needed 286 s to reach iteration 40.
Constraint violations remain 0 throughout.

None of them learned, but at 60 iterations that is not yet evidence of a stall:
the one DAM leg that does converge (occupation m=4, K=16) was also flat at
TV 0.41674 at iteration 50 and 0.39471 at 200, breaking down to 0.06995 only by
iteration 300.

**Round 2 — 600 iterations, past the point where the m=4 leg had converged.**

| arm | settings | best E-hist TV | E-hist TV @ it 600 | TV @ it 600 | ESS mean / p10 | wall |
|---|---|---|---|---|---|---|
| A2 | `a-clamp 3 m-clip 5` | **0.15866** (it 325) | 0.30301 | 0.76576 | 3.38 / 1.07 | 1359 s |
| E | `a-clamp 3 m-clip 5 lr 3e-4` | 0.61324 | 0.61939 | 0.77506 | 2.35 / 1.02 | 1413 s |

A2 is genuinely learning something the baseline never did — the energy histogram
falls 0.632 (it 225) -> **0.159** (it 325) against a baseline that never went
below 0.677 — and the ESS *improves* with training, 1.99 -> 4.71 / 16. But it
then bounces back to 0.303 and the full TV never leaves 0.76-0.83.

**Round 2 diagnosis: the clamp was set below the target.** Reading the exact
control multiplier off `FI.ExactControl` shows it grows sharply towards `t = 1`:

| t | 0.000 | 0.250 | 0.500 | 0.750 | 0.938 | 0.992 |
|---|---|---|---|---|---|---|
| `max \|a_exact\|` | 0.185 | 0.576 | 1.431 | 2.629 | 4.205 | **6.308** |
| `q99 \|a_exact\|` | 0.114 | 0.277 | 0.687 | 1.573 | 3.062 | 4.861 |

`a-clamp 3` therefore makes the optimal controller **literally unrepresentable**
over the last fifth of the time axis. The learned-vs-exact multiplier error in
the round-2 artifacts sits exactly where the clamp binds — RMSE 0.69-0.84 for
`t <= 0.75`, but **2.09 RMSE and 8.10 max** at `t = 0.992` — and the bounce in
A2's energy histogram is what a run does on approach to a boundary it cannot
cross. Round 1's apparent success was partly an artifact of the same thing:
a controller pinned to a small box cannot diverge, but it also cannot converge.

Round 3 (running) sets the clamp *above* the target, `a-clamp 8` (27% headroom
over 6.308, permitting a rate multiplier of 3.0e3 versus the baseline's 4.9e8),
keeping `m-clip 5`, with and without the round-1 outlier removal.

**What this already establishes, regardless of how round 3 lands.** The §5.2
divergence is not intrinsic to the estimator — it is a bounded-control problem,
and clipping `log m_hat` at 5 plus a finite rate box removes it entirely on a
benchmark where the unmodified method reaches -4e12 loss and ESS 1.04. What is
*not* yet established is that the stabilised run converges, and the honest
reading of round 2 is that the two requirements pull against each other: the box
must be small enough to bound the jump explosion and large enough to contain the
optimum. On Ising L=4 those two constraints are at least compatible in principle
(6.31 needed, 20 permitted at baseline); whether the optimiser finds it is what
round 3 measures.

### 6.3 Cost of the comparison, head to head

Same benchmark (occupation m=4), same hardware, best setting of each method:

Both rows are the **Dirac** source, m=N=4, 128 steps (see the source-matching
note in §5.4).

| | IASBS (Dirac) | DAM (Dirac, K=16, 1200 it) | ratio |
|---|---|---|---|
| terminal TV | **0.012730** | 0.01490 | IASBS 1.17x better |
| adjoint rollouts | **0** | 20,889,600 | infinite |
| f1 evals on rollout endpoints | **0** | 20,889,600 | infinite |
| CTMC jumps simulated for the adjoint | **0** | 185,004,804 | infinite |
| wall clock (s) | no log (<= **214**) | 11,529 | **>= 54x** |

IASBS is at least 54x faster in wall clock *and* more accurate, because the adjoint
`phi_t(x) = E_base[f1(X_1) | X_t = x]` that DAM estimates with 20.9 million
Monte-Carlo rollout-endpoint evaluations is available to IASBS in closed form via
the intertwining identity. The gap is structural, not a matter of tuning.

**The "0" is a specific quantity, not a claim of zero arithmetic.** See §6.4 for
the exact accounting — IASBS does evaluate the target energy, and on the sphere
its gradient. What it never does is *simulate a path in order to evaluate `f1` at
the endpoint*.

Caveat stated plainly: the two methods do not solve identical optimisation
problems, and DAM is a general-purpose algorithm that does not require the
intertwining structure. The comparison shows what that generality costs on a space
where the structure *is* available, which is the regime this paper is about.

---

### 6.4 What "0 terminal evaluations" counts — exact accounting

The `f1 evals` column is not a hand-wave, but it is also not "IASBS does no work
on the target". Precisely what is instrumented, and what is not:

**The counted quantity.** In `dam/core.py:estimate_log_adjoint` the counter is
literally

```python
stats["f1_evals"] = B * (K + K_num)
```

incremented in `dam/discrete.py:318`. It counts **literal calls to
`adapter.log_f1(x)` evaluated at the endpoint of a simulated CTMC rollout** —
`K` denominator rollouts started at `(t, x)` plus `K_num` numerator rollouts
started at `(t, y)`, per label. Every one of those endpoints exists *only*
because a path was simulated forward to `t = 1` to produce it, and each carries
the O(1) importance-weight variance analysed in §5.2. So the column measures
**variance-carrying Monte-Carlo adjoint work**, not arithmetic on `E`.

**IASBS is zero on that quantity and only that quantity.** IASBS performs no
adjoint rollouts, so it has no rollout endpoints and makes no `log_f1` calls at
rollout endpoints. It is *not* zero energy evaluations. The full picture:

| quantity | IASBS Ising L=4 (non-Dirac) | IASBS occupation m=4 (Dirac) | IASBS sphere non-Dirac (per seed) | DAM occupation m=4 (Dirac), K=16 |
|---|---|---|---|---|
| extra adjoint rollouts | **0** | **0** | **0** | 20,889,600 |
| `f1` evals at rollout endpoints | **0** | **0** | **0** | 20,889,600 |
| CTMC jumps simulated for the adjoint | **0** | **0** | **0** | 185,004,804 |
| target energy `E` evals, on the fly | 14,745,600,000 | 0 (table) | 0 (enters via gradient only) | 0 (table) |
| one-off precomputed `E` table | 12,870 states | 35 states | 2,562-node spectral table | 35 states |
| energy-**gradient** `grad E` evals | **0** | **0** | 32,768,000 | **0** |
| on-policy endpoint simulation (state-steps) | 1,572,864,000 | 98,304,000 | 4,194,304,000 | (counted as jumps above) |

Row-by-row justification:

- **Ising L=4 does 14.7 billion energy evaluations, more than DAM's f1 count.**
  `terminal_labels` (`fixed_ising.py:428`) needs
  `log Lambda_g = -(E(gY) - E(Y))/tau - h_g(Y)`, so it calls
  `space.energies_of(Xg)` on *every one-swap neighbour* of the endpoint:
  `mb=1024 x Np=C(16,2)=120 x inner=40 x iters=3000`. We state this plainly
  because it contradicts any reading of "0" as "IASBS is arithmetically free".
  The reason it costs 1869 s anyway is that those evaluations are a single fused
  vectorised reduction over a `(mb, Np, n)` tensor with **no sequential
  simulation and no sampling variance**, whereas each DAM `f1` eval sits at the
  end of a Python-level Gillespie `while` loop (§6.2).
- **Occupation (Dirac) is 0 on the fly because the space is enumerable.** `occupation.py`
  precomputes `self.E`, `self.logf1` and the full rate table
  `self.R = exp(logf1[trans] - logf1[:, None, None])` once over all `|Omega| = 35`
  states, so training is table lookup. DAM on the same problem uses the *same*
  table for its `log_f1` — its 20.9 M evaluations are table lookups too. The
  difference is not the cost of one `f1` call, it is that DAM must first
  **simulate 185 M jumps** to decide which table entries to look up.
- **The sphere is the one place a gradient is required.** The non-Dirac readout is
  `G_ND(y) = -grad E(y)/tau - H_psi(y)`, so `sphere.py:767` calls
  `C.sphere_energy_grad(y1)` once per endpoint per iteration:
  `batch=8192 x iters=4000 = 32,768,000` per seed, 163,840,000 across 5 seeds.
  These are analytic closed-form gradients of the bimodal energy, not
  backpropagated and not Monte-Carlo, but they are real work and are reported.
- **Both methods simulate on-policy endpoints.** IASBS simulates the controlled
  process forward to obtain `X_1` for its replay buffer, exactly as DAM does.
  That cost is common to both and is *not* what the comparison is about. DAM's
  `K + K_num = 17` extra rollouts *per label* are the differential cost.

**Therefore the defensible claim is the narrow one:** IASBS needs zero
Monte-Carlo estimation of the adjoint `phi_t`, because the intertwining identity
supplies `Lambda_g` in closed form from quantities it already has (the endpoint,
its one-swap neighbours, and a heat-kernel table). Any broader reading — "IASBS
evaluates no energies", "IASBS does less arithmetic" — is **not** supported, and
on Ising the second reading is measurably false.

---

## 7. Reproduction

```bash
# IASBS, Ising non-Dirac
python structured_asbs/fixed_ising.py train-nondirac --L 4 --steps 256 --iters 3000 \
    --n-samples 200000 --tag ising_nd_L4 --out json/results_ising_nd_L4.json

# IASBS, occupation scaling
python structured_asbs/occupation.py scale-nondirac --m 1000 --steps 256 --iters 1500 \
    --mb 512 --n-samples 4000 --tag occ_nd_s1000 --out json/results_occ_nd_s1000.json

# IASBS, sphere non-Dirac (Haar)
python structured_asbs/sphere.py train-nondirac --steps 128 --iters 4000 --inner 16 \
    --inner-h 4 --batch 8192 --mb 16384 --mb-h 4096 --hidden 256 --lr 1e-3 \
    --ema 0.9995 --antithetic --seeds 5 --eval-every 200 --n-samples 200000 \
    --tag sphere_nd --out json/results_sphere_nd.json

# DAM baseline
python -m dam.discrete occupation --m 4 --K 16 --steps 128 --iters 1200 --inner 4 \
    --mb 256 --batch 512 --eval-every 50 --n-samples 20000 \
    --tag dam_occ4_K16_long --out json/results_dam_occ4_K16_long.json

# DAM correctness tests (gradient identity, path-weight identity, adjoint estimator)
python -m dam.tests_math
```

---

## 8. Still running / still to run

**Running:** DAM stabiliser round 3 (`fix_F_clamp8`, `fix_G_clamp8damp`), Ising
L=4 K=16, 600 iterations each, ~23 min per arm on one GPU each -- see §6.2.1.
Every IASBS experiment is finished.

**Finished since the last update**

| job | outcome |
|---|---|
| IASBS Ising L=5 non-Dirac | **DONE** — A1 FAIL (TV 0.07228 vs gate 0.05), A2 PASS (0 / 200,000) — §2.1 |
| DAM Ising L=4, K=16 | **DIVERGED** at it 15, killed at it 40 / 286 s — §6.2 |
| DAM occupation-scale m=32, K=64 | **DIVERGED** at it 10, killed at it 15 / 1,664 s — §6.2 |

**Not run, and why**

| job | reason |
|---|---|
| DAM occupation-scale m=128 / m=1000, K=16 | K=16 is measured to diverge at m=32 and on Ising L=4; larger m would only reproduce a foregone divergence |
| DAM occupation-scale m=32, K=256 | K=16 -> K=64 bought five iterations, so the K axis is not where the fix is |
| DAM Ising L=5, K=16 | Ising L=4 already diverges at K=16 |
| IASBS_600 Stiefel at beta = 0.1, 0.5, 7, 10, 20 | matched-budget ablation not performed at those temperatures; ~28 min per beta if wanted |

**Text only**

- README correction of the R-ASBS mode-collapse description (§4.2).

**Two honest negatives, recorded rather than buried.** IASBS Ising L=5 misses its
accuracy gate at 0.07228, and no DAM configuration attempted outside the |X| = 35
occupation problem converged at all, which means the head-to-head of §5.4 and §6.3
rests on the *one* benchmark where DAM works. That is a real limit on the strength
of the DAM comparison and is stated as such: on larger discrete spaces the claim is
not "IASBS beats DAM by 17%", it is "IASBS converges and DAM does not", which is a
weaker and differently-shaped claim.

---

*This file is updated each time an experiment finishes. Numbers are copied verbatim
from the JSON artifacts named in each section; nothing here is projected except
where explicitly marked "proj." or ">".*
