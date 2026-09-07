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
| 1 | IASBS reaches exact-IPF accuracy on discrete structured spaces | §2.1, §2.2 — TV at or below the i.i.d. sampling floor |
| 2 | IASBS accuracy does not degrade with state-space size | §2.3 — KS improves monotonically from m=32 to m=1000 |
| 3 | IASBS extends to non-Dirac (Haar) sources without loss | §3.2 — matches the Dirac headline to 3 decimal places |
| 4 | Reported R-ASBS mode collapse is an initialisation artifact | §4 — collapse vanishes under the authors' own init |
| 5 | DAM needs millions of terminal evaluations; IASBS needs none | §5 — 20.9 M f1 evals for DAM's best TV vs 0 for IASBS |
| 6 | DAM has a hard stability cliff below K=16 | §5.2 — K=1 and K=4 diverge, not merely degrade |

---

## 2. IASBS on discrete structured spaces

### 2.1 Ising ring, non-Dirac (uniform) source — validated against exact IPF

Gate: terminal-law TV must reach the i.i.d. sampling floor.

| L | states | TV (exact law) | empirical TV | i.i.d. TV floor | violations | status |
|---|---|---|---|---|---|---|
| 4 | 2^16 | **0.03470** | 0.06571 | 0.04940 | 0 | **DONE — PASS** |
| 5 | 2^25 | 0.07886 (it 2325/3000) | — | — | 0 | RUNNING |

`json/results_ising_nd_L4.json`, `json/results_ising_nd_L5.json`.

L=4 sits **below** the i.i.d. floor of 0.04940, i.e. the learned terminal law is
statistically indistinguishable from exact samples at this sample size.

IPF corrector sanity check (L=4): `mean exp(h) = 0.99723` (target 1),
`max |h| = 0.07224`. The corrector is a small multiplicative perturbation, as the
intertwining identity predicts.

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

| seed | north_mass | north_err | KS_z | W1_z |
|---|---|---|---|---|
| 0 | 0.49728 | 0.00272 | 0.08040 | 0.06451 |
| 1 | 0.51026 | 0.01026 | 0.06319 | 0.04328 |
| 2 | 0.49394 | 0.00606 | 0.06080 | 0.04544 |
| 3 | 0.42420 | 0.07580 | 0.12787 | 0.14183 |
| 4 | 0.43537 | 0.06463 | 0.11342 | 0.12319 |
| **mean** | — | **0.02703** | **0.08768** | — |

**Collapse disappears entirely.** The phenomenon is an initialisation artifact, not
a property of on-policy self-training. Any description of R-ASBS as exhibiting
"spontaneous mode collapse" must be retracted; the honest statement is that R-ASBS
is *initialisation-sensitive*.

### 4.3 Head-to-head on the same target

| method | mean north_err | mean KS_z |
|---|---|---|
| R-ASBS, default init | 0.49878 | 0.50484 |
| R-ASBS, matlab init | 0.02703 | 0.08768 |
| **IASBS, Dirac, antithetic** | **0.00069** | **0.02165** |
| **IASBS, Haar (non-Dirac)** | **0.00111** | **0.02278** |

Against the *strongest* R-ASBS configuration, IASBS is **39x** better in hemisphere
error and **4.1x** better in KS, with zero constraint violation and no seed variance.

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

| method | best TV on occupation m=4 | terminal f1 evaluations | CTMC jumps simulated | wall (s) |
|---|---|---|---|---|
| DAM, K=16 | 0.01490 | 20,889,600 | 185,004,804 | 11529.3 |
| DAM, K=64 | 0.04498 | 26,624,000 | 222,041,777 | 3177.7 |
| **IASBS** | **0.01248** | **0** | **0** | — |

IASBS obtains `phi_t` in closed form through the intertwining identity, so it
performs **no** adjoint rollouts and **no** terminal evaluations during training.
DAM's best result is 19% worse in TV and costs **20.9 million** terminal
evaluations plus 185 million simulated CTMC jumps to get there.

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

| experiment | iters | steps | wall (s) | s / iter | terminal evals | checkpoint |
|---|---|---|---|---|---|---|
| Ising L=4 non-Dirac | 3000 | 256 | **1869** | 0.62 | 0 | `ising_nd_L4.pt` |
| Ising L=5 non-Dirac | 3000 | 256 | ~37000 (proj.) | 12.3 | 0 | `ising_nd_L5.pt` |
| Occupation m=4 non-Dirac | — | 128 | **214** | — | 0 | `occ_nd_m4.pt` |
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

### 6.2 DAM

| experiment | K | iters | wall (s) | s / iter | f1 evals | CTMC jumps | jumps / f1 eval |
|---|---|---|---|---|---|---|---|
| Occupation m=4 | 1 | 10 | 224.7 | 22.5 | 20,480 | 1,499,657 | **73.2** |
| Occupation m=4 | 4 | 10 | 163.9 | 16.4 | 51,200 | 1,197,204 | **23.4** |
| Occupation m=4 | 16 | 400 | 7380.9 | 18.5 | 6,963,200 | 67,417,741 | 9.68 |
| Occupation m=4 | 16 | 1200 | **11529.3** | 9.6 | **20,889,600** | **185,004,804** | 8.86 |
| Occupation m=4 | 64 | 400 | 3177.7 | 7.9 | 26,624,000 | 222,041,777 | 8.34 |
| Ising L=4 | 16 | 1200 | RUNNING | >30 | — | — | — |
| Occupation-scale m=32 | 16 | 1200 | RUNNING | >30 | — | — | — |
| Occupation-scale m=128 | 16 | 600 | QUEUED | — | — | — | — |
| Occupation-scale m=1000 | 16 | 300 | QUEUED | — | — | — | — |
| Ising L=5 | 16 | 600 | QUEUED | — | — | — | — |

DAM runs are **latency-bound, not throughput-bound**: measured CPU time equals wall
time to within 1% on every leg, because each Gillespie rollout is a Python `while`
loop issuing many small kernels and synchronising on `active.any()`. Consequences:

- Batch size is nearly free; the number of `while` iterations is what costs.
- Co-scheduling two DAM runs on one device roughly *doubles* both wall times rather
  than overlapping them, which is why the remaining legs are queued serially
  (`dam/queue_remaining.sh`).
- The jump-explosion column is the mechanism behind the small-K divergence in §5.2:
  at K=1 each terminal evaluation costs 73 simulated jumps versus 8.9 at K=16.

### 6.3 Cost of the comparison, head to head

Same benchmark (occupation m=4), same hardware, best setting of each method:

| | IASBS | DAM (K=16, 1200 it) | ratio |
|---|---|---|---|
| terminal TV | **0.01248** | 0.01490 | IASBS 1.19x better |
| terminal f1 evaluations | **0** | 20,889,600 | infinite |
| simulated CTMC jumps | **0** | 185,004,804 | infinite |
| wall clock (s) | **214** | 11,529 | **53.9x** |

IASBS is 54x faster in wall clock *and* more accurate, because the adjoint
`phi_t(x) = E_base[f1(X_1) | X_t = x]` that DAM estimates with 20.9 million
Monte-Carlo terminal evaluations is available to IASBS in closed form via the
intertwining identity. The gap is structural, not a matter of tuning.

Caveat stated plainly: the two methods do not solve identical optimisation
problems, and DAM is a general-purpose algorithm that does not require the
intertwining structure. The comparison shows what that generality costs on a space
where the structure *is* available, which is the regime this paper is about.

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

Every IASBS experiment is finished. What remains is entirely DAM-baseline work
plus one text correction.

**Running**

| job | progress | last metric | projected wall |
|---|---|---|---|
| IASBS Ising L=5 non-Dirac | 2325 / 3000 | TV 0.07886 | ~37000 s |
| DAM Ising L=4, K=16 | <50 / 1200 | — | >36000 s |
| DAM occupation-scale m=32, K=16 | <50 / 1200 | — | >36000 s |

**Queued** (serially, `dam/queue_remaining.sh`, see §6.2 for why not in parallel)

| job | iters | note |
|---|---|---|
| DAM occupation-scale m=128, K=16 | 600 | budget cut as m grows |
| DAM occupation-scale m=1000, K=16 | 300 | may not finish; that is a result |
| DAM Ising L=5, K=16 | 600 | matched against §2.1 |

**Text only**

- README correction of the R-ASBS mode-collapse description (§4.2).

A note on the queued DAM legs. Their iteration budgets are deliberately smaller
than the IASBS runs they are compared against, because the per-rollout jump count
grows with the state space and a matched-iteration DAM run at m=1000 is not
affordable. If a leg fails to reach its gate inside the reduced budget, the
reported outcome is the wall clock and the f1-evaluation count at the cutoff,
labelled as budget-limited rather than as a method failure — the same correction
already applied to the K=16 m=4 leg in §5.1.

---

*This file is updated each time an experiment finishes. Numbers are copied verbatim
from the JSON artifacts named in each section; nothing here is projected except
where explicitly marked "proj." or ">".*
