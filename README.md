# Adjoint Schrödinger Bridge Sampling on Structured State Spaces via Markov Semigroup Intertwining

Experimental package for the paper. One standalone script per experiment, one
shared utility file. This README is a **catalogue of numbers**: for every
experiment it gives the configuration, the exact command that produced the
artifacts, the measured results, and the `json/` + `ckpt/` names. Discussion,
comparison and interpretation belong to the paper, not here.

```
LICENSE                   MIT for our code; see its scope note about rasbs/
requirements.txt          pinned versions every number below was produced with
MULTISEED_RESULTS.md      per-seed tables + the original seed-request table
APPENDIX_OCCUPATION_SIMULATORS.md
                          the two occupation simulators; what `steps` counts
common.py                 shared kernels, quadrature, Stiefel/S^3 helpers, ckpt IO
json/  ckpt/  fig/        artifacts, shared by both methods

iasbs/          OUR method (IASBS)
  fixed_ising.py          Exp A  fixed-magnetisation Ising   (bijective discrete)
  occupation.py           Exp B  occupation process      (non-bijective discrete)
  sphere.py               Exp C  S^2                       (scalar Killing readout)
  stiefel.py              Exp D  St(4,2)                   (matrix Killing readout)
  earthquake.py           Exp E  S^2, real data            (vMF mixture, 3343 modes)
  fixed_support.py        Appendix A.2  fixed-support space X^(r)_{n,k} + GB1
  tests_math.py           standalone mathematical unit tests (27 gates)
  figures.py              all paper figures + the comparison table
  gallery.py              sample gallery
  remeasure.py            re-derives the Ising and sphere tables from ckpt/
  _a2_tests.py            A.2 deterministic tests, imported by tests_math.py
  scripts/                the shell scripts that produced our json/ entries
  analysis/               read-only diagnostics -- they load ckpt/ + json/,
                          train nothing, and each writes one json/ artifact.
                          Indexed in §10.

dam/                      BASELINE 1 -- Discrete Adjoint Matching
  core.py                 gKL loss, adjoint estimator, control box
  discrete.py             the benchmark adapters + CLI
  tests_math.py           gradient identity, path-weight identity, estimator (11 gates)

rasbs/                    BASELINE 2 -- R-ASBS, kept apart from our code
  rasbs_port.py           PyTorch port of alg2_stiefel.m @ bb71d14
  rasbs_sphere_port.py    port of asbs_sphere_sampler.m + earthquake_sphere_ex.m
  rasbs_sphere_audit.py   closed-form falsification tests for that port
```

Every entry point calls `common.use_repo_root()`, so relative paths
(`json/`, `ckpt/`, `fig/`) resolve to the repository root regardless of the
launch directory. **All commands below are run from the repository root.**

Hardware and environment: 2x NVIDIA A100 80 GB, PyTorch 2.5.1, CUDA 12.4,
NumPy 2.4.6, conda env `SML_env` (Python 3.11). Manifold state and all metrics
are `float64`; networks are `float32`. `requirements.txt` pins exact versions.
`PY` defaults to whatever `python` is first on `PATH`.

`ckpt/` is gitignored (~10 GB, regenerable). `rasbs_ref/` (the frozen R-ASBS
clone, source of the earthquake `query.csv`) and `data/gb1/*.xlsx` (Olson et al.
supplements) are also gitignored; download instructions for the latter are in
`data/gb1/README.md`.

Every `json/results_*.json` carries the full `config` block it was produced
with, so any flag not spelled out below can be read off the artifact.

```bash
python iasbs/tests_math.py   # 27 mathematical unit tests
python -m dam.tests_math               # 11 gates for the DAM baseline
python iasbs/figures.py      # all figures -> fig/
python iasbs/gallery.py      # sample gallery -> fig/
```

Gates used below: **A1** exact-law TV <= 0.05; **A2** zero constraint
violations; **B2a** occupancy KS <= 0.05; **B2b** max-occupancy W1 <= 5% of N;
**B2c** zero violations; **C1** mean hemisphere-mass error < 0.03; **C2**
KS(x3) < 0.05; **D0** verification gates incl. the spin-clock test; **E1**
KS(E) < 0.05; **E2** mode-histogram TV < 0.15; **E3** norm residual at machine
precision.

---

# 1. Ising lattice — fixed magnetisation

State space: `L x L` periodic lattice, `n = L^2` spins, magnetisation fixed at
zero, so `|Omega| = C(n, n/2)`. Energy `E = -J sum_<ij> s_i s_j`, `J = 1`,
`tau = 2`. Dynamics are magnetisation-preserving swaps (Kawasaki). The chain
law is propagated by **exact enumeration**, so the TV column carries no
Monte-Carlo error.

| L | sites | `\|Omega\|` |
|---|---:|---:|
| 4 | 16 | 12,870 |
| 5 | 25 | 5,200,300 |

## 1.1 IASBS, Dirac source, L = 4

| | configuration |
|---|---|
| script | `iasbs/fixed_ising.py train` |
| network | `SwapController`, hidden 512, 670,464 parameters |
| integration steps | 256 |
| iters x inner | 3000 x 40 |
| batch / minibatch | 2048 / 1024 |
| lr | 3e-4 |
| buffer | 8 |
| loss | Poisson |
| eval samples | 20,000 |
| seeds | 1 |

| metric | exact | IASBS |
|---|---:|---:|
| TV of the chain law vs pi | 0 | **0.0416** |
| KL(ours \|\| pi) | 0 | 0.0056 |
| Hellinger | 0 | 0.0372 |
| ESS fraction (importance reweighting) | 1 | 0.9885 |
| max_x \|p(x)/pi(x) - 1\| | 0 | 224.27 |
| mean energy <E> | -9.2457 | -8.9850 |
| heat capacity Var(E)/tau^2 | 5.8843 | 6.0237 |
| <s_i s_j> | 0.2889 | 0.2808 |
| violations / 20,000 | 0 | **0** |

Multiplier error (all 12,870 states x 32 edges), mean absolute: 0.060 at t = 0,
0.028-0.048 in the interior, 0.170 (max 1.92) at t -> 1.

```bash
python iasbs/fixed_ising.py train --steps 256 --iters 3000 \
    --eval-every 200 --n-samples 20000 --loss poisson --ckpt-dir ckpt \
    --tag ising_poisson256 --out json/results_ising_poisson256.json
python iasbs/remeasure.py ising   # re-derives the table from ckpt/
```

Artifacts: `json/results_ising_poisson256.json`, `ckpt/ising_poisson256.pt`.
Also produced by `bash iasbs/scripts/rerun_ckpt.sh` (Ising leg).

## 1.2 Exact control — discretisation floor

Same algorithm, multiplier from enumeration instead of a network. No training.

| steps | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| L = 4, TV vs pi | 0.13610 | 0.06823 | 0.03385 | 0.01681 | 0.00837 | 0.00417 |
| L = 5, TV vs pi | — | 0.09849 | 0.04823 | 0.02377 | 0.01179 | — |

First-order, no bias floor; mass leak <= 5e-15 at both sizes. At L = 4, 1024
steps and 200,000 samples the empirical TV is 0.0517 against an iid floor of
0.0511.

```bash
python iasbs/fixed_ising.py exact       # L=4, gates A0/A1/A2
python iasbs/fixed_ising.py exact --L 5 --steps-sweep 64 128 256 512 \
    --tag ising_t1_L5_exact --out json/results_ising_t1_L5_exact.json
```

Artifacts: `json/results_ising_exact.json`, `json/results_ising_t1_L5_exact.json`
(closed form, no checkpoint).

## 1.3 IASBS, Dirac source, L = 5

Same target and loss; hidden 512, 864,369 parameters, `--iters 6000 --inner 40`,
512 steps. Reported as **A1 FAIL, A2 PASS**.

| metric | L = 4 (12,870) | L = 5 (5,200,300) |
|---|---:|---:|
| TV of the chain law vs pi | 0.0416 | **0.0770** |
| KL(ours \|\| pi) | 0.0056 | 0.0193 |
| Hellinger | 0.0372 | 0.0692 |
| ESS fraction | 0.9885 | 0.9617 |
| max_x \|p(x)/pi(x) - 1\| | 224.3 | 62,240 |
| TV of the energy histogram | 0.0262 | 0.0347 |
| mean energy <E> (exact) | -8.9850 (-9.2457) | -17.1348 (-17.6038) |
| heat capacity (exact) | 6.0237 (5.8843) | 7.3246 (7.1195) |
| <s_i s_j> (exact) | 0.2808 (0.2889) | 0.3427 (0.3521) |
| violations / 200,000 | 0 | **0** |

Step ablation at L = 5, learned vs exact control:

| steps | iters | exact-control TV | learned TV | ratio |
|---:|---:|---:|---:|---:|
| 256 | 3000 | 0.02377 | 0.0923 | 3.9x |
| 512 | 6000 | 0.01179 | **0.0770** | 6.5x |

Multiplier error over 5,200,300 states x 50 edges, mean absolute: 0.088 at
t = 0, 0.053-0.089 interior, 0.310 (max 5.59) at t -> 1.

Sample-based statistics at this size: empirical TV 0.3546 against an iid floor
of 0.3231 +- 0.0008; the exact control scores 0.3298 on the same measure at a
true TV of 0.0118.

```bash
python iasbs/fixed_ising.py train --L 5 --steps 512 --iters 6000 \
    --hidden 512 --tag ising_t1_L5_s512 --out json/results_ising_t1_L5_s512.json
python iasbs/remeasure.py ising5
```

Artifacts: `json/results_ising_t1_L5.json`, `json/results_ising_t1_L5_s512.json`,
`ckpt/ising_t1_L5.pt`, `ckpt/ising_t1_L5_s512.pt`.

## 1.4 IASBS, non-Dirac (uniform) source, L = 4 and L = 5

| L | TV (exact law) | empirical TV | iid TV floor | violations | verdict |
|---|---:|---:|---:|---:|---|
| 4 | **0.03470** | 0.06571 | 0.04940 | 0 / 200,000 | A1 PASS, A2 PASS |
| 5 | **0.07228** (best 0.06714 at it 2725) | 0.35530 | 0.32546 | 0 / 200,000 | A1 FAIL, A2 PASS |

IPF corrector: `mean exp(h)` = 0.99723 (L = 4) and 0.99792 (L = 5), target 1;
`max |h|` = 0.07224 and 0.25522.

L = 5 config: 3000 iterations, 256 steps, batch 2048, 1,728,738 parameters,
36,966 s wall (12.3 s/iter).

```bash
python iasbs/fixed_ising.py train-nondirac --L 4 --steps 256 \
    --iters 3000 --n-samples 200000 \
    --tag ising_nd_L4 --out json/results_ising_nd_L4.json
python iasbs/fixed_ising.py train-nondirac --L 5 --tau 2.0 --gamma 10.0 \
    --steps 256 --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 \
    --hidden 512 --lr 3e-4 --loss poisson --inner-h 40 --lr-h 3e-4 \
    --eval-every 25 --n-samples 200000 \
    --tag ising_nd_L5 --out json/results_ising_nd_L5.json
```

Artifacts: `json/results_ising_nd_L4.json`, `json/results_ising_nd_L5.json`,
`ckpt/ising_nd_L4.pt`, `ckpt/ising_nd_L5.pt`.

## 1.5 DAM baseline on the Ising lattice

Control box for every DAM leg in this README: `--a-clamp 8 --m-clip 5
--ess-min 3 --coef-cap 10`. Exact-control multiplier magnitude for reference:
`max |a_exact|` = 0.185 / 0.576 / 1.431 / 2.629 / 4.205 / 6.308 at
t = 0 / 0.25 / 0.5 / 0.75 / 0.938 / 0.992.

| L | K | iters | TV | E-hist TV | gate | ESS end | ESS mean / p10 | f1 evals | CTMC jumps | wall (s) | s/it |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | 32 | 5000 | **0.06168** | 0.04759 | FAIL | 30.47 / 32 | 23.70 / 7.43 | 168,960,000 | 974,800,000 | 11,428 | 2.29 |
| 4 | 64 | 2500 | 0.08103 | 0.06567 | FAIL | 58.62 / 64 | 42.47 / 8.36 | 166,400,000 | 924,000,000 | 6,950 | 2.78 |
| 4 | 32 | 7000 | 0.06339 | 0.05206 | FAIL | 30.64 / 32 | 24.91 / 11.40 | 236,544,000 | 1,339,809,998 | 16,226 | 2.32 |
| 5 | 32 | 5000 | 0.39956 | 0.19000 | FAIL | 13.39 / 32 | 8.53 / 2.01 | 168,960,000 | 1,372,615,519 | 15,391 | 3.08 |

Violations: 0 on every row. Clipped labels: 0 on every row.

L = 4 K = 32 tail (plateau check):

| it | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 | 7000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TV | 0.08227 | 0.08055 | 0.06934 | 0.06731 | 0.06784 | 0.06193 | **0.06339** |
| E-hist TV | 0.05425 | 0.05636 | 0.05033 | 0.05087 | 0.05172 | 0.04901 | 0.05206 |
| ESS / 32 | 28.86 | 29.47 | 29.85 | 30.04 | 30.56 | 30.63 | 30.64 |

L = 5 K = 32 tail (still descending at cutoff):

| it | 3250 | 3500 | 4000 | 4250 | 4500 | 4750 | 5000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TV | 0.49883 | 0.46154 | 0.41855 | 0.41352 | 0.41032 | 0.40511 | **0.39956** |
| E-hist TV | 0.28391 | 0.23797 | 0.19383 | 0.21109 | 0.19141 | 0.19710 | 0.19000 |
| ESS / 32 | 10.18 | 11.18 | 11.64 | 12.78 | 13.35 | 12.60 | 13.39 |

```bash
bash dam/run_dam.sh ising      # L=4 legs
bash dam/run_dam.sh isingL5    # L=5 leg
```

Artifacts: `json/results_dam_ising_L4_K32_5000.json`,
`json/results_dam_ising_L4_K64_2500.json`,
`json/results_dam_ising_L4_K32_7000.json`,
`json/results_dam_ising_L5_K32_5000.json` + identically-named `ckpt/*.pt`.

---

# 2. Occupation process

`m` indistinguishable particles on `N` sites, state `eta` in `N^N` with
`sum_i eta_i = m`, so `|X| = C(m+N-1, N-1)`. Target
`pi(eta) ∝ prod_i Gamma(eta_i+d)/(eta_i! Gamma(d))`, `d = 0.5`, `tau = 1`.
Moves are non-bijective.

| | configuration |
|---|---|
| script | `iasbs/occupation.py` |
| network | `OccController`, hidden 256, 139,536 parameters (136,450 in the scale sweep) |
| integration steps | 128 (256 at m = 1000) |
| iters x inner | 1500 x 4 (3000 at m = 32, 128) |
| batch / minibatch | 512 / 1024 (128 at m = 1000) |
| lr | 1e-3 |
| buffer | 8 |
| eval samples | 20,000 |

## 2.1 IASBS, Dirac source, m = N = 4 (`|X| = 35`)

| metric | exact | IASBS |
|---|---:|---:|
| TV vs exact law, learned control | 0 | **0.012730** |
| TV vs exact law, exact control | 0 | 0.0199 |
| iid floor, 20,000 samples | — | 0.0167 |
| violations | 0 | **0** |

Exact-control step sweep: TV 0.2965 / 0.1410 / 0.0379 / 0.0189 / 0.0094 /
0.0047 at 8 / 16 / 32 / 64 / 128 / 256 steps.

**Estimator and loss ablation** (m = N = 4, all with 0 violations):

| variant | flags | TV vs exact law |
|---|---|---:|
| full enumeration over moves | `--estimator full` | **0.0127** |
| uniform move sampling | `--estimator uniform` | 0.0162 |
| occupancy-weighted sampling | `--estimator occupancy` | 0.0164 |
| Bregman -> MSE loss | `--estimator full --loss mse` | 0.0131 |

**Estimator variance, gate B3** (`json/results_occ_var.json`, B3 true). At
t = 0 occupancy weighting is exact (variance 3.5e-13 vs uniform 1.059); in the
interior four-sample occupancy weighting cuts variance ~4x against one sample
(0.418 vs 1.672 at t = 0.25) and 4.6-4.8x against uniform. The same measurement
at m = N = 128 and 1000 also passes.

```bash
python iasbs/occupation.py verify                  # gate B0
python iasbs/occupation.py exact                   # exact control, step sweep
python iasbs/occupation.py train --m 4 --iters 1500 --eval-every 250 \
    --n-samples 20000 --estimator full --ckpt-dir ckpt \
    --tag occ4_full --out json/results_occ_full.json
python iasbs/occupation.py train --m 4 --estimator uniform \
    --ckpt-dir ckpt --tag occ4_uniform --out json/results_occ_uniform.json
python iasbs/occupation.py train --m 4 --estimator occupancy \
    --ckpt-dir ckpt --tag occ4_occupancy --out json/results_occ_occupancy.json
python iasbs/occupation.py train --m 4 --estimator full --loss mse \
    --ckpt-dir ckpt --tag occ4_mse --out json/results_occ_mse.json
python iasbs/occupation.py var                     # gate B3
```

Artifacts: `json/results_occ_exact.json`,
`json/results_occ_{full,uniform,occupancy,mse}.json`,
`json/results_occ_var.json` + `ckpt/occ4_{full,uniform,occupancy,mse}.pt`.
All four training legs are also produced by
`bash iasbs/scripts/rerun_ckpt.sh`.

## 2.2 IASBS, non-Dirac source, m = N = 4

| source nu_0 | TV | empirical TV | iid floor | corrector max err | Sinkhorn marginal err | violations |
|---|---:|---:|---:|---:|---:|---:|
| uniform `[.25,.25,.25,.25]` | **0.01248** | 0.01801 | 0.01598 | 3.59e-4 | 2.78e-17 | 0 |
| skewed `[.025,.075,.225,.675]` | **0.01411** | 0.02049 | 0.01598 | 2.56e-4 | 1.11e-16 | 0 |

Both A1 PASS. Component breakdown (uniform): occupancy-histogram TV 0.00614,
max-occupancy TV 0.00744, mean energy 1.88671.

```bash
python iasbs/occupation.py train-nondirac --m 4 \
    --tag occ_nd_m4 --out json/results_occ_nd_m4.json
python iasbs/occupation.py train-nondirac --m 4 --nu-skew \
    --tag occ_nd_m4_skew --out json/results_occ_nd_m4_skew.json
```

Artifacts: `json/results_occ_nd_m4.json`, `json/results_occ_nd_m4_skew.json`,
`ckpt/occ_nd_m4.pt`, `ckpt/occ_nd_m4_skew.pt`.

## 2.3 IASBS, Dirac source, scaling in m

Same network (136,450 parameters) and code at every size. The `m = N = 4` rows
above and the `m = N = 32, 128, 1000` rows here use **different simulators** --
one jump per bin vs. tau-leap -- so the `steps` column does not mean the same
thing in both. `APPENDIX_OCCUPATION_SIMULATORS.md` spells that out.

| m = N | `\|X\|` | source KS(occ) | KS(occ) | KS(max) | W1(max)/m | E ours | E exact | violations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | C(63,31) ≈ 9.2e17 | 0.2054 | **0.0043** | 0.0305 | 0.0052 | 13.317 | 13.212 | **0** |
| 128 | C(255,127) ≈ 5.8e75 | 0.2054 | **0.0084** | 0.0796 | 0.0044 | 52.871 | 52.108 | **0** |
| 1000 | C(1999,999) ≈ 1e600 | 0.2035 | **0.0102** | 0.2215 | 0.0014 | 412.32 | 405.57 | **0** |

The three rows are not budget-matched: m = 32 and 128 use 3000 iterations,
128 steps, batch 512; m = 1000 uses 1500 iterations, 256 steps, batch 128.

These rows use a different simulator from §2.1/§2.2. The enumerated simulator
(`simulate`) moves at most one particle per grid interval. The large-m
simulator (`scale_step`) is a tau-leap: departures from mode i are
`Binomial(eta_i, p_i)` and all arrivals are one `Multinomial(K, softmax beta)`,
so one bin moves many particles. `steps` counts leap bins here, not jumps.

Jump audit, 2000 paths per row:

| m = N | steps | transfers/path mean (min–max) | max particles in one bin | max_i eta_i at t=1 mean (min–max) | peak departure prob | clamp hits |
|---:|---:|---|---:|---|---:|---:|
| 32 | 128 | 127.5 (94–168) | 7 | 6.62 (3–17) | — | 0 |
| 128 | 128 | 513.1 (442–584) | 15 | 9.30 (4–24) | — | 0 |
| 1000 | 256 | 4103.8 (3868–4341) | 36 | 13.03 (8–26) | 1.6993e-01 | 0 / 2.50e8 |

`scale_step` clamps the per-particle departure probability to `[0, 0.9]`; the
clamp is inactive at every setting reported here.

```bash
python iasbs/occupation.py scale-jumps --m 1000 --N 1000 --steps 256 \
    --batch 2000 --hidden 256 --ckpt-dir ckpt --tag occ_s1000 \
    --out json/results_occ_jumps_s1000.json
```

Artifacts: `json/results_occ_jumps_s{32,128,1000}.json`.

```bash
bash iasbs/scripts/run_scale.sh
# or, per leg:
python iasbs/occupation.py scale --m 32 --N 32 --iters 3000 \
    --eval-every 500 --n-samples 20000 --ckpt-dir ckpt \
    --tag occ_s32 --out json/results_occ_s32.json
python iasbs/occupation.py scale --m 128 --N 128 --iters 3000 \
    --eval-every 500 --n-samples 10000 --ckpt-dir ckpt \
    --tag occ_s128 --out json/results_occ_s128.json
python iasbs/occupation.py scale --m 1000 --N 1000 --steps 256 \
    --batch 128 --mb 512 --iters 1500 --eval-every 250 --n-samples 4000 \
    --ckpt-dir ckpt --tag occ_s1000 --out json/results_occ_s1000.json
python iasbs/occupation.py scalevar --m 1000 --N 1000 \
    --out json/results_occ_var1000.json
```

Artifacts: `json/results_occ_s{32,128,1000}.json`,
`json/results_occ_var{,128,1000}.json` + `ckpt/occ_s{32,128,1000}.pt`.

## 2.4 IASBS, non-Dirac source, scaling in m

Gates B2a / B2b / B2c. Same tau-leap simulator and same `steps` semantics as
§2.3, not the one-jump-per-bin simulator of §2.1/§2.2.

| m | KS(occ) | W1(max)/N | violations | uncontrolled KS(occ) | improvement | verdict |
|---|---:|---:|---:|---:|---:|---|
| 32 (seed 0) | **0.01512** | 0.01509 | 0 / 10,000 | 0.20534 | 13.6x | PASS |
| 32 (seed 1) | **0.01895** | 0.01690 | 0 / 10,000 | 0.20468 | 10.8x | PASS |
| 128 | **0.01270** | 0.00224 | 0 | 0.20585 | 16.2x | PASS |
| 1000 | **0.01109** | 0.00173 | 0 / 4,000 | 0.20300 | 18.3x | PASS |

Energy at m = 1000: `E_ours = 412.70` vs `E_exact = 405.57` (1.8% relative);
conditioning 0.01607 vs 0.01446. Both m = 32 seeds report `bad_labels = 0`,
`skipped_steps = 0`.

```bash
python iasbs/occupation.py scale-nondirac --m 32 --N 32 \
    --tag occ_nd_s32_seed0 --out json/results_occ_nd_s32_seed0.json
python iasbs/occupation.py scale-nondirac --m 128 --N 128 \
    --tag occ_nd_s128 --out json/results_occ_nd_s128.json
python iasbs/occupation.py scale-nondirac --m 1000 --steps 256 \
    --iters 1500 --mb 512 --n-samples 4000 \
    --tag occ_nd_s1000 --out json/results_occ_nd_s1000.json
```

Artifacts: `json/results_occ_nd_s32_seed{0,1}.json`,
`json/results_occ_nd_s128.json`, `json/results_occ_nd_s1000.json` +
`ckpt/occ_nd_s{32_seed0,32_seed1,128,1000}.pt`.

## 2.5 DAM baseline on the occupation process

**K sweep at m = N = 4** (`|X| = 35`, 128 steps):

| K | iters | TV | occ-hist TV | ESS mean | ESS p10 | f1 evals | CTMC jumps | wall (s) | clipped | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 400 | 0.07170 | — | 9.62 / 16 | 4.26 | 6,963,200 | 67,417,741 | 7,380.9 | 0 | FAIL |
| 16 | 1200 | **0.01490** | 0.00800 | 14.04 / 16 | 8.07 | 20,889,600 | 185,004,804 | 11,529.3 | 0 | PASS |
| 64 | 400 | 0.04498 | 0.01827 | 48.75 / 64 | 16.71 | 26,624,000 | 222,041,777 | 3,177.7 | 0 | PASS |

TV trajectory (K = 16): 0.41674 (it 50) -> 0.39471 (200) -> 0.20222 (250) ->
0.06995 (300) -> 0.05128 (400) -> 0.02749 (650) -> 0.01974 (850) -> 0.01696
(1050) -> **0.01490** (1200). ESS rises 7.86/16 -> 15.88/16.

Per terminal evaluation:

| K | final TV | f1 evals | TV per million f1 evals |
|---:|---:|---:|---:|
| 16 | 0.01490 | 20,889,600 | 0.00071 |
| 64 | 0.04498 | 26,624,000 | 0.00169 |

**Occupation at scale:**

| m | K | iters | KS(occ) | KS(max) | W1(max)/N | ESS | clipped | f1 evals | CTMC jumps | wall (s) | s/it | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 64 | 1000 | **0.0128** | 0.0399 | 0.0068 | 55.58 / 64 (mean 31.39, p10 1.10) | 0 | 66,560,000 | 5,849,804,432 | 16,572 | 16.6 | PASS |
| 128 | 64 | 500 | 0.17554 | 0.9014 | 0.03820 | 1.69 / 64 (p10 1.00) | 451,665 | 33,280,000 | 12,889,415,391 | 67,693 | 135.4 | FAIL |

Violations 0 on both. Uncontrolled reference: m = 32 KS(occ) 0.2040,
KS(max) 0.8024, W1(max)/N 0.1007; m = 128 KS(occ) 0.20529, KS(max) 0.8986,
W1(max)/N 0.03997.

m = 32 trajectory:

| it | 25 | 125 | 225 | 325 | 425 | 625 | 825 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | +134.4 | -43.8 | -1029.9 | +163.4 | +121.5 | +118.3 | +118.7 | +121.0 |
| KS(occ) | 0.2024 | 0.1865 | 0.0395 | 0.0200 | 0.0138 | 0.0139 | 0.0105 | **0.0128** |
| ESS / 64 | 3.03 | 2.45 | 2.70 | 9.88 | 27.62 | 48.80 | 53.42 | **55.58** |

m = 128 trajectory: KS(occ) 0.2071 (it 25) -> 0.2086 (125) -> 0.2027 (250) ->
0.1834 (400) -> 0.1731 (475) -> 0.1755 (500). Energy `E_ours = 65.89` vs
`E_exact = 52.11`; uncontrolled 67.86; E-histogram KS 0.9983.

```bash
bash dam/run_dam.sh occ4        # the three m=4 legs
bash dam/run_dam.sh occs32      # the m=32 leg
bash dam/run_occs128_K64.sh     # the m=128 leg
```

Artifacts: `json/results_dam_occupation_m4_K16.json` (400 it),
`json/results_dam_occ4_K16_long.json` (1200 it),
`json/results_dam_occ4_K64.json`, `json/results_dam_occs32_K64_1000.json`,
`json/results_dam_occs128_K64_500.json` + identically-named `ckpt/*.pt`.

---

# 3. Fixed-support space (Appendix A.2)

State space `X^(r)_{n,k} = {x in {0..r-1}^n : #{i : x_i != 0} = k}`. The
nonbinary-Johnson orbit kernel makes the terminal matching labels exactly
computable, so the learned controlled law is compared to the target in exact TV
over all states.

## 3.1 Heterogeneous Potts toy, (n, k, r) = (14, 4, 4), `|X| = 81,081`

Target: heterogeneous nonsymmetric Potts ring, 128 energy edges (8 label + 120
support), `C = 15` orbits, `A = 770` global actions. Both runs: `steps = 256`,
`iters = 3000`, `batch = 2048`, `buffer = 8`, `inner = 40`, `mb = 1024`,
`hidden = 512`, `lr = 3e-4`, Poisson loss, `seed = 0`, `n_samples = 20000`.

| source | params | exact-law TV | KL | Hellinger | A1 | A2 | wall (s) |
|---|---:|---:|---:|---:|---|---|---:|
| Dirac(x0) | 882,806 | **0.00774** | 0.000202 | 0.00709 | PASS | PASS (0 / 20,000) | 3,050 |
| 4-atom mixture [0, 27, 81, 82] | 882,806 + 344,066 | **0.00776** | 0.000201 | 0.00708 | PASS | PASS (0 / 20,000) | 3,460 |

Non-Dirac corrector (`corrector_hidden = 256`, `corrector_actions = 32`):
RMSE 0.00705, MAE 0.00336, max abs 0.1169 over 100,000 probes.

Mean energy -2.3617 (Dirac) and -2.3572 (non-Dirac) vs target -2.3753. Modal
state (index 78,344) ranked first by both, `p = 0.000908 / 0.000910` vs
`pi = 0.000935`. Terminal-law mass error 7e-16 / 2e-16. Empirical TV over
20,000 samples 0.52736 / 0.52521 against iid floors 0.52772 / 0.52162.

| it | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---:|---:|---:|---:|---:|---:|
| TV, Dirac | 0.02040 | 0.01881 | 0.01767 | 0.01145 | 0.01195 | **0.00774** |
| TV, non-Dirac | 0.02157 | — | 0.01394 | 0.01918 | 0.01031 | **0.00776** |

```bash
python iasbs/fixed_support.py train --target toy --steps 256 \
    --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 \
    --lr 3e-4 --loss poisson --n-samples 20000 \
    --tag fs_toy_v2_dirac --out json/results_fs_toy_v2_dirac.json
python iasbs/fixed_support.py train-nondirac --target toy \
    (same flags) --tag fs_toy_v2_nd --out json/results_fs_toy_v2_nd.json
```

Artifacts: `json/results_fs_toy_v2_dirac.json`, `json/results_fs_toy_v2_nd.json`,
`ckpt/fs_toy_v2_{dirac,nd}.pt`.

## 3.2 GB1 protein-fitness landscape, k = 3 sector, `|X| = 27,436`

Four-site GB1 landscape of Wu et al. (eLife 2016;5:e16965) restricted to the
exact Hamming-distance-3 sector from wild type `VDGV`: `n = 4`, `k = 3`,
`r = 20`, `|X| = C(4,3) * 19^3 = 27,436`. Complete measured + author-imputed
landscape (149,361 + 10,639 = 160,000 variants), Boltzmann-weighted on
`E = -log(F + 1e-4)` at `tau = 1`, `gamma = 10`. Target statistics: E_min
-1.8366, E_max 9.2103, E_mean 4.9582, E_std 2.5893, pi_max 0.0015412, entropy
8.2118 (of `log 27436 = 10.22`). Orbit count `C = 7`, `A = 2,748` global
actions, `alpha = 4.8649`, `beta = 5.1351`, 111 energy edges (54 label + 57
support).

**Schedule: temperature-annealed, three warm-started stages of 1000 iterations
at `tau = 2.0, 1.4, 1.0`** (3000 iterations total). Only stage C is reported,
evaluated at `tau = 1` against the unmodified Wu et al. Boltzmann law.

| source | params | exact-law TV | KL | Hellinger | A1 | A2 |
|---|---:|---:|---:|---:|---|---|
| Dirac(x0) | 766,844 | **0.13568** | 0.08132 | 0.15205 | FAIL | PASS (0 / 20,000) |
| four-atom mixture | 766,844 + 858,556 | **0.13226** | 0.07650 | 0.14738 | FAIL | PASS (0 / 20,000) |

Per-stage exact TV, each stage measured against its own `tau`:

| stage | tau | iters | Dirac TV | non-Dirac TV |
|---|---:|---:|---:|---:|
| A | 2.0 | 1000 | 0.10566 | 0.10695 |
| B | 1.4 | 1000 | 0.11564 | 0.10703 |
| C | **1.0** | 1000 | **0.13568** | **0.13226** |

Stage C opens at 0.18333 (Dirac) / 0.15945 (non-Dirac) and recovers past its own
starting point within 750 iterations. Mean fitness 1.85499 / 1.91098 against
target 1.83499. Modal-variant (state 6,464) overweight 1.29x (Dirac,
`p = 0.00199`) and 1.51x (non-Dirac, `p = 0.00233`) against `pi = 0.00154`.
Terminal-law mass error exactly 0.0 on both. Non-Dirac corrector RMSE 0.02353,
MAE 0.00899, max abs 0.63775.

**Discretisation and reference-mixing controls.** Re-propagating the trained
control on finer grids gives TV 0.21465 at 256 steps, 0.21759 at 512, 0.21909
at 1024 — finer grids are not better, so the residual is not the Euler grid.
The reference chain at `gamma = 10` is fully mixed at `t = 1`: propagating with
the control off gives TV 0.77976 against the target, and the exactly uniform law
sits at 0.78220; `gamma` = 20, 40, 80 all sit on that same floor to five
decimals.

```bash
bash iasbs/scripts/run_gb1_anneal.sh both
```

Artifacts: `json/results_fs_gb1_k3_{dirac,nd}_{A,B,C}.json` +
`ckpt/fs_gb1_k3_{dirac,nd}_{A,B,C}.pt`; the reported row is stage `C`.
Requires `data/gb1/elife-16965-supp1-v4.xlsx` and
`data/gb1/elife-16965-supp2-v4.xlsx` (see `data/gb1/README.md`).

## 3.3 DAM baseline on the two A.2 spaces

Identical schedules: 1200 flat `tau = 1` iterations on the toy, and the same
`2.0 -> 1.4 -> 1.0` annealed chain on GB1 (3 x 400 iterations, warm started via
`--init-from`). Same 20,000-sample constraint audit.

| space | method | source | exact-law TV | A1 | A2 | iters | f1 evals | CTMC jumps | wall (s) | s/it |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| toy, `\|X\|=81,081` | IASBS | Dirac | **0.00774** | PASS | PASS (0 / 20,000) | 3000 | 0 | 0 | 3,098 | 1.03 |
| toy | IASBS | four-atom | 0.00776 | PASS | PASS (0 / 20,000) | 3000 | 0 | 0 | 3,294 | 1.10 |
| toy | DAM, K=16 | Dirac | 0.08285 | FAIL | PASS (0 / 20,000) | 1200 | 20,889,600 | 118,996,727 | 7,508 | 6.26 |
| GB1 k=3, `\|X\|=27,436` | IASBS | Dirac | **0.13568** | FAIL | PASS (0 / 20,000) | 3 x 1000 | 0 | 0 | 5,814 | 1.94 |
| GB1 k=3 | IASBS | four-atom | 0.13226 | FAIL | PASS (0 / 20,000) | 3 x 1000 | 0 | 0 | 6,079 | 2.03 |
| GB1 k=3 | DAM, K=16 | Dirac | 0.26472 | FAIL | PASS (0 / 20,000) | 3 x 400 | 20,889,600 | 127,340,484 | 7,512 | 6.26 |

DAM toy secondaries: KL 0.02942, Hellinger 0.08532, mean energy -2.29261 vs
target -2.37526, modal-state mass 0.00088 vs `pi = 0.00093`, ESS 13.15 / 16
mean, 9.58 p10, 0 clipped labels, 0 non-finite weights, terminal-law mass error
6.7e-16.

DAM per-stage on the annealed GB1 chain:

| stage | tau | iters | DAM TV | jumps | wall (s) |
|---|---:|---:|---:|---:|---:|
| A | 2.0 | 400 | 0.25745 | 41,657,667 | 2,508 |
| B | 1.4 | 400 | 0.27040 | 42,548,716 | 2,511 |
| C | **1.0** | 400 | **0.26472** | 43,134,101 | 2,493 |

DAM GB1 mean fitness 1.45539 vs target 1.83499; modal-state overweight 1.18x
(`p = 0.00144` vs `pi = 0.00122`); terminal-law mass error 2.2e-15.

```bash
bash dam/run_dam_a2.sh both
```

Artifacts: `json/results_dam_fs_toy_v2_K16.json`,
`json/results_dam_fs_gb1_k3_K16_{A,B,C}.json` + `ckpt/dam_fs_toy_v2_K16.pt`,
`ckpt/dam_fs_gb1_k3_K16_{A,B,C}.pt`; the reported GB1 row is stage `C`.

---

# 4. Sphere S^2

Bimodal target `pi ∝ exp(6 x_3^2)`, i.e. `E(x) = 6(1 - x_3^2)` at `tau = 1` —
the same energy and scale as R-ASBS's `asbs_sphere_sampler.m`. The reference is
exact: the z-marginal is `∝ exp(6z^2)` on [-1, 1] with uniform azimuth, so
inverse-CDF sampling draws iid target points.

| | configuration |
|---|---|
| script | `iasbs/sphere.py` |
| source `x_0` | `(1, 0, 0)` (Dirac) or Haar (non-Dirac) |
| network | `ScoreNet`, hidden 256, 136,963 parameters |
| integration steps | 128 |
| iters x inner | 4000 x 16 |
| batch / minibatch | 8192 / 16384 |
| lr | 1e-3 |
| EMA / buffer | 0.9995 / 4 |
| eval samples | 200,000 |
| seeds | 5 |

## 4.1 IASBS, Dirac source — symmetry-handling ablation

| variant | mean north_err | mean KS(x_3) | max \|norm-1\| | gates C1 / C2 |
|---|---:|---:|---:|---|
| **antithetic (headline)** | **0.00069** | **0.02165** | 2.2e-16 | PASS / PASS |
| symmetrized | 0.00082 | 0.02283 | 2.2e-16 | PASS / PASS |
| plain | 0.02540 | 0.03881 | 2.2e-16 | PASS / PASS |

Antithetic augmentation uses `R = diag(1,1,-1)`, an exact same-law isometry.
Per-seed north mass: antithetic 0.5001 +- 0.0006; plain 0.460 / 0.474 / 0.495 /
0.528 / 0.528 (spread +- 0.0275). Per-seed antithetic north_err = 0.00109 /
0.00040 / 0.00040 / 0.00045 / 0.00113; KS(x_3) = 0.02199 / 0.02113 / 0.02253 /
0.02076 / 0.02185.

Full antithetic table, 200,000 samples per seed, recomputed from the checkpoints
by `remeasure.py sphere`:

| metric | exact | IASBS Dirac (5 seeds) | iid floor |
|---|---:|---:|---:|
| north mass | 0.5 | **0.5001 +- 0.0006** | 0.5002 +- 0.0008 |
| absolute north error | 0 | **0.0006 +- 0.0003** | 0.0006 +- 0.0005 |
| KS(x_3) | 0 | **0.0222 +- 0.0004** | 0.0018 +- 0.0004 |
| W1(x_3) | 0 | **0.01239 +- 0.00022** | 0.00118 +- 0.00073 |
| KS(azimuth) | 0 | **0.0026 +- 0.0010** | 0.0018 +- 0.0005 |
| <x_3^2> | 0.80771 | **0.78864 +- 0.00019** | 0.80778 +- 0.00033 |
| mean energy <E> | 1.15375 | **1.26816 +- 0.00112** | 1.15333 +- 0.00197 |
| max norm residual | 0 | **2.2e-16** | — |

**Iteration-budget control.** `json/results_sphere_train_plain2500.json` is the
plain variant stopped at 2500 iterations instead of 4000: north error 0.0599,
KS(x_3) 0.0667.

**Exact control step sweep** (no network): KS(x_3) = 0.0655 / 0.0353 / 0.0184 /
0.0102 / 0.0063 at 32 / 64 / 128 / 256 / 512 steps; `<E>` falls monotonically
1.543 -> 1.177 toward the exact 1.15375. First-order, no bias floor.

**Residual against iid target draws:** 99% of 1800 equal-area bins within
+-3 sigma, max |z| = 4.3 at 200,000 samples a side.

```bash
python iasbs/sphere.py verify              # gate C0
bash iasbs/scripts/rerun_sphere.sh         # exact + plain/anti/sym, 5 seeds each
# headline leg alone:
python iasbs/sphere.py train --antithetic --iters 4000 --inner 16 \
    --batch 8192 --mb 16384 --ema 0.9995 --seeds 5 --n-samples 200000 \
    --tag sphere_anti --out json/results_sphere_train_anti.json
python iasbs/remeasure.py sphere
```

Artifacts: `json/results_sphere_exact.json`,
`json/results_sphere_train_{plain,anti,sym}.json`,
`json/results_sphere_train_plain2500.json` +
`ckpt/sphere_{plain,anti,sym}_seed{0..4}.pt`.

## 4.2 IASBS, non-Dirac (Haar) source

Source replaced by Haar measure; the closed-form terminal score is replaced by a
learned corrector `H_psi(y) ~ grad log f1_hat(y)`, giving
`G_ND(y) = -grad E(y)/tau - H_psi(y)`. 136,963 + 136,963 parameters.

| seed | north_mass | north_err | KS(x_3) | max \|norm-1\| | W1(x_3) | KS(phi) | `\|Delta E[x_3^2]\|` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.50014 | 0.00014 | 0.02215 | 2.22e-16 | 0.01205 | 0.00132 | 0.01849 |
| 1 | 0.50111 | 0.00111 | 0.02237 | 2.22e-16 | 0.01259 | 0.00116 | 0.01923 |
| 2 | 0.49818 | 0.00183 | 0.02305 | 2.22e-16 | 0.01412 | 0.00213 | 0.01881 |
| 3 | 0.50145 | 0.00145 | 0.02357 | 2.22e-16 | 0.01259 | 0.00205 | 0.01918 |
| 4 | 0.50103 | 0.00103 | 0.02278 | 2.22e-16 | 0.01269 | 0.00288 | 0.01890 |
| **mean** | — | **0.00111** | **0.02278** | **2.22e-16** | **0.01281** | **0.00191** | **0.01892** |
| iid floor (n = 200,000) | — | — | — | — | 0.00031 | 0.00224 | 0.00001 |

Gates: C1 PASS (0.0011 < 0.03), C2 PASS (0.0228 < 0.05). Exact
`E[x_3^2] = 0.807709`.

**Antithetic ablation on the non-Dirac branch:** without it, north_err 0.4398
wobble, KS 0.0649 (`ckpt/sphere_nd_plain_seed0.pt`).

```bash
python iasbs/sphere.py train-nondirac --steps 128 --iters 4000 \
    --inner 16 --inner-h 4 --batch 8192 --mb 16384 --mb-h 4096 --hidden 256 \
    --lr 1e-3 --ema 0.9995 --antithetic --seeds 5 --eval-every 200 \
    --n-samples 200000 --tag sphere_nd --out json/results_sphere_nd.json
python iasbs/remeasure.py sphere
```

Artifacts: `json/results_sphere_nd.json` + `ckpt/sphere_nd_seed{0..4}.pt`,
`ckpt/sphere_nd_plain_seed0.pt`.

## 4.3 R-ASBS baseline on S^2

Bimodal target, 600 epochs, 500 steps, sigma = 1, B = 500, 64/64 and 48/48 tanh
networks, Adam at 2e-3, 5 seeds. Port of `asbs_sphere_sampler.m`.

**Default initialisation:**

| seed | north_mass | north_err | KS(x_3) |
|---|---:|---:|---:|
| 0 | 0.00004 | 0.49996 | 0.50587 |
| 1 | 0.00004 | 0.49996 | 0.50590 |
| 2 | 0.99762 | 0.49762 | 0.50328 |
| 3 | 0.00173 | 0.49827 | 0.50438 |
| 4 | 0.00191 | 0.49809 | 0.50476 |
| **mean** | — | **0.49878** | **0.50484** |

Epoch controls at seed 0: north mass 0.0004 at 100 epochs, 0.0002 at 300;
`<E>` 1.267 (100) -> 1.172 (300) -> 1.046 (600) against the exact 1.15375.

**Authors' own `--init matlab`:**

| seed | north_mass | north_err | KS(x_3) | W1(x_3) | KS(phi) | `\|Delta E[x_3^2]\|` |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.49728 | 0.00272 | 0.08040 | 0.06451 | 0.00980 | 0.09044 |
| 1 | 0.51026 | 0.01026 | 0.06319 | 0.04328 | 0.00829 | 0.05544 |
| 2 | 0.49394 | 0.00606 | 0.06080 | 0.04544 | 0.00718 | 0.06288 |
| 3 | 0.42420 | 0.07580 | 0.12787 | 0.14183 | 0.01397 | 0.08579 |
| 4 | 0.43537 | 0.06463 | 0.11342 | 0.12319 | 0.01544 | 0.08190 |
| **mean** | — | **0.03189** | **0.08914** | **0.08365** | **0.01094** | **0.07529** |
| **sd over seeds** | — | 0.03158 | 0.02699 | 0.04100 | 0.00322 | 0.01365 |
| iid floor (n = 100,000) | — | 0.00087 | 0.00268 | 0.00178 | 0.00259 | 0.00069 |

Port self-checks (`--check`): Haar uniformity, transport isometry and tangency,
`grad E` against central differences, `V(1)` against quadrature, and the
cotangent series at its switch point — all pass. `rasbs_sphere_audit.py` holds
two closed-form falsification targets (`E = 0`, target Haar; `E = -6 x_3`,
single mode with `north = (e^6-1)/(e^6-e^-6)`, `<x_3> = coth 6 - 1/6`).

## 4.4 S^2 summary table

| method | mean north_err | mean KS(x_3) | mean W1(x_3) | mean KS(phi) | mean `\|Delta E[x_3^2]\|` |
|---|---:|---:|---:|---:|---:|
| R-ASBS, default init | 0.49878 | 0.50484 | — | — | — |
| R-ASBS, matlab init | 0.03189 | 0.08914 | 0.08365 | 0.01094 | 0.07529 |
| IASBS, Dirac, antithetic | **0.00069** | **0.02165** | **0.01239** | **0.00264** | **0.01907** |
| IASBS, Haar (non-Dirac) | **0.00111** | **0.02278** | **0.01281** | **0.00191** | **0.01892** |
| iid floor | 0.00087 / 0.00031 | 0.00268 / 0.00224 | 0.00178 / 0.00031 | 0.00259 / 0.00224 | 0.00069 / 0.00001 |

Floor row: first number at n = 100,000 (R-ASBS sample size), second at
n = 200,000 (IASBS sample size).

```bash
python rasbs/rasbs_sphere_port.py --check                      # port self-checks
python rasbs/rasbs_sphere_port.py --problem bimodal --seed 0 \
    --tag rasbs_sphere_bimodal --out json/results_rasbs_sphere_bimodal.json
bash rasbs/run_fidelity_audit.sh                               # --init matlab, 5 seeds
python iasbs/analysis/rasbs_extra.py                         # KS(phi), 2nd moment, from ckpt/
python rasbs/rasbs_sphere_audit.py --test vmf                  # closed-form audit
```

Artifacts: `json/results_rasbs_sphere_bimodal{,_s1,_s2,_s3,_s4}.json`,
`json/results_rasbs_sphere_bimodal_e{100,300}.json`,
`json/results_rasbs_sphere_matlabinit_s{0..4}.json`,
`json/results_rasbs_audit_{uniform,vmf}_mi.json` + matching `ckpt/*.pt`.

## 4.5 Fair comparison against R-ASBS on S^2

The §4.4 rows are not budget-matched. IASBS calls `C.sphere_energy_grad` once
per endpoint per iteration, `batch 8192 x iters 4000 = 32,768,000` terminal
oracle calls per seed; R-ASBS uses `600 epochs x B 500 = 300,000`, a factor of
109. This section equalises that. Everything else is `rerun_sphere.sh`'s
antithetic leg unchanged: 128 steps, `--inner 16`, EMA 0.9995, 200,000
evaluation samples, 5 seeds.

| leg | oracle calls | params | mean north_err +- sd | mean KS(x_3) +- sd | max \|\|x\|-1\| |
|---|---:|---:|---:|---:|---:|
| R-ASBS, matlab init | 300,000 | 7,366 | 0.03189 +- 0.03158 | 0.08914 +- 0.02699 | — |
| IASBS, budget-matched | 300,000 | 136,963 | **0.00072 +- 0.00055** | **0.03042 +- 0.00052** | 2.2e-16 |
| IASBS, budget + parameter matched | 300,000 | **5,715** | **0.00132 +- 0.00080** | **0.02948 +- 0.00083** | 2.2e-16 |
| IASBS, native budget | 32,768,000 | 136,963 | 0.00069 | 0.02165 | 2.2e-16 |

The parameter-matched leg uses `--hidden 48`, which is 5,715 parameters against
R-ASBS's 7,366 — 0.78x, so it is matched on oracle calls and strictly smaller in
capacity. Per-seed KS(x_3): 0.03098 / 0.02967 / 0.03079 / 0.03018 / 0.03051
(hidden 256) and 0.02993 / 0.02868 / 0.03047 / 0.02855 / 0.02977 (hidden 48).
Gates C1 and C2 PASS on both legs. Wall 140-153 s per seed.

On the four §4.3 secondaries, all recomputed from `ckpt/` by the same
`remeasure.py sphere` path that produced the §4.2 and §4.3 rows (population sd,
5 seeds, 200,000 samples per seed):

| leg | north mass | W1(x_3) | KS(phi) | `\|dE[x_3^2]\|` |
|---|---:|---:|---:|---:|
| R-ASBS, matlab init | 0.47221 +- 0.03524 | 0.08365 +- 0.04100 | 0.01094 +- 0.00322 | 0.07529 +- 0.01365 |
| IASBS, budget-matched | **0.49961 +- 0.00136** | **0.01883 +- 0.00034** | **0.00371 +- 0.00152** | **0.02813 +- 0.00046** |
| IASBS, budget + parameter matched | **0.49954 +- 0.00096** | **0.01762 +- 0.00029** | **0.00298 +- 0.00068** | **0.02659 +- 0.00031** |
| IASBS, native budget | 0.50013 +- 0.00062 | 0.01239 +- 0.00022 | 0.00264 +- 0.00097 | 0.01907 +- 0.00019 |
| iid floor (n = 200,000) | 0.50019 +- 0.00079 | 0.00118 +- 0.00073 | 0.00182 +- 0.00053 | 0.00028 +- 0.00019 |

Per-seed, budget-matched then parameter-matched:

| metric | hidden 256 | hidden 48 |
|---|---|---|
| north mass | 0.49978 / 0.50194 / 0.49772 / 0.49912 / 0.49949 | 0.49970 / 0.50112 / 0.49975 / 0.49890 / 0.49824 |
| W1(x_3) | 0.01908 / 0.01875 / 0.01933 / 0.01865 / 0.01836 | 0.01785 / 0.01757 / 0.01739 / 0.01726 / 0.01804 |
| KS(phi) | 0.00192 / 0.00457 / 0.00280 / 0.00623 / 0.00302 | 0.00213 / 0.00348 / 0.00249 / 0.00400 / 0.00281 |
| `\|dE[x_3^2]\|` | 0.02894 / 0.02757 / 0.02816 / 0.02815 / 0.02784 | 0.02702 / 0.02655 / 0.02652 / 0.02609 / 0.02676 |

KS(phi) at both matched budgets is within a factor 2 of the iid floor, so the
azimuthal marginal is already at sampling resolution and the 109x budget cut
does not touch it. `north mass` differs by 5e-4 from the value the training run
records in its own JSON (0.50012 for hidden 256) because the two are separate
200,000-sample draws; the table above uses the `ckpt/` draw so that all four
columns share one provenance with §4.2 and §4.3.

Cutting the budget 109x moves KS(x_3) from 0.02165 to 0.03042 (1.4x) and leaves
north_err unchanged within seed noise. Halving capacity below R-ASBS's costs
nothing further.

```bash
python iasbs/sphere.py train --antithetic --seeds 5 --iters 600 \
    --batch 500 --mb 16384 --ema 0.9995 --inner 16 --eval-every 1000 \
    --n-samples 200000 --ckpt-dir ckpt --tag sphere_m300 \
    --out json/results_sphere_matched300.json
python iasbs/sphere.py train --antithetic --seeds 5 --iters 600 \
    --batch 500 --mb 16384 --ema 0.9995 --inner 16 --hidden 48 \
    --eval-every 1000 --n-samples 200000 --ckpt-dir ckpt --tag sphere_m300h48 \
    --out json/results_sphere_matched300_h48.json
```

Artifacts: `json/results_sphere_matched300.json`,
`json/results_sphere_matched300_h48.json` +
`ckpt/sphere_m300{,h48}_seed{0..4}.pt`.

## 4.6 Controlled ablation — every nuisance factor matched

§4.5 matches the oracle budget only. This section matches source law,
diffusion coefficient, integrator resolution, precision, training budget,
evaluation draw, metric code, and reflection augmentation, and leaves only the
supervision/control formulation different.

Matched: Haar source on both sides (IASBS runs `train-nondirac`); `sigma = 1.0`
on both (IASBS's default is `sqrt(2)`; the terminal law is `pi` for any
`sigma`); 500 integrator steps; float64 end to end; 300,000 terminal oracle
calls; 100,000 evaluation samples; 5 seeds; `iasbs/remeasure.py` scores both;
reflection augmentation `R = diag(1,1,-1)` applied to both arms or to neither
(a `--antithetic` flag was added to `rasbs/rasbs_sphere_port.py` for this, off
by default).

Not matched, by construction: learning rate (1e-3 IASBS / 2e-3 R-ASBS, each
its own tuned value for a different loss), parameter count (136,963 /
7,366 — see §4.5 for the parameter-matched cut), and gradient steps per
oracle batch.

| leg | north_err | KS(x_3) | W1(x_3) | KS(E) | KS(phi) | `<E>` | wall/seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| IASBS Haar, aug | **0.00171 +- 0.00124** | **0.01435 +- 0.00123** | **0.00878 +- 0.00073** | **0.02597 +- 0.00176** | 0.00385 +- 0.00109 | **1.22349 +- 0.00389** | 478 s |
| IASBS Haar, no aug | 0.02553 +- 0.02405 | 0.03531 +- 0.01845 | 0.04825 +- 0.03918 | 0.03193 +- 0.00204 | 0.00290 +- 0.00078 | 1.24430 +- 0.00711 | 473 s |
| R-ASBS, no aug | 0.03189 +- 0.03158 | 0.08914 +- 0.02699 | 0.08365 +- 0.04100 | 0.13489 +- 0.02138 | 0.01094 +- 0.00322 | 1.60549 +- 0.08190 | 257 s |
| R-ASBS, aug | 0.01337 +- 0.01138 | 0.07786 +- 0.01762 | 0.06101 +- 0.01653 | 0.13638 +- 0.02140 | 0.01360 +- 0.00700 | 1.61276 +- 0.08121 | 250 s |
| iid floor (n = 100,000) | 0.00087 +- 0.00049 | 0.00268 +- 0.00076 | 0.00178 +- 0.00083 | 0.00253 +- 0.00075 | 0.00259 +- 0.00064 | 1.15076 +- 0.00338 | — |

Exact `<E> = 1.15375`. Constraint residual `max ||x| - 1|`: 2.2e-16 (IASBS),
3.3e-16 (R-ASBS).

Per-seed:

| leg | north_err | KS(x_3) | KS(E) | `<E>` |
|---|---|---|---|---|
| IASBS, aug | 0.00340 / 0.00204 / 0.00021 / 0.00251 / 0.00037 | 0.01244 / 0.01585 / 0.01437 / 0.01366 / 0.01541 | 0.02269 / 0.02730 / 0.02701 / 0.02558 / 0.02727 | 1.21682 / 1.22692 / 1.22416 / 1.22197 / 1.22757 |
| IASBS, no aug | 0.00050 / 0.01383 / 0.05393 / 0.00434 / 0.05506 | 0.01621 / 0.02719 / 0.05669 / 0.01826 / 0.05818 | 0.03162 / 0.03548 / 0.02998 / 0.02996 / 0.03261 | 1.24308 / 1.25568 / 1.24220 / 1.23376 / 1.24677 |
| R-ASBS, no aug | 0.00272 / 0.01026 / 0.00606 / 0.07580 / 0.06463 | 0.08040 / 0.06319 / 0.06080 / 0.12787 / 0.11342 | 0.15996 / 0.10487 / 0.11529 / 0.15311 / 0.14121 | 1.69640 / 1.48640 / 1.53103 / 1.66849 / 1.64515 |
| R-ASBS, aug | 0.03431 / 0.00002 / 0.00822 / 0.01123 / 0.01305 | 0.10579 / 0.06452 / 0.05539 / 0.07643 / 0.08719 | 0.15907 / 0.12580 / 0.10030 / 0.14227 / 0.15446 | 1.68953 / 1.54967 / 1.48949 / 1.63667 / 1.69842 |

Equal-optimizer-step leg, `--inner 1 --inner-h 1 --mb 500 --mb-h 500`, same
matched config otherwise, 5 seeds:

| leg | north_err | KS(x_3) | KS(E) | `<E>` | wall/seed |
|---|---:|---:|---:|---:|---:|
| IASBS Haar, aug, inner 1 | 0.00308 +- 0.00207 | 0.31953 +- 0.00089 | 0.63650 +- 0.00147 | 3.94837 +- 0.00475 | 440 s |

IASBS's construction emits `batch x steps = 500 x 500 = 250,000` regression
pairs per oracle batch and R-ASBS draws one time per trajectory, 500 pairs per
epoch; at `--inner 1 --mb 500` IASBS consumes 0.2% of the pairs it generated.

Caveat on §4.2: the antithetic collapse quoted there (`north_err 0.4398`,
`KS 0.0649`, one seed, `ckpt/sphere_nd_plain_seed0.pt`) is specific to that
configuration (128 steps, `sigma = sqrt(2)`, batch 8192, 4000 iters). At the
§4.6 configuration the no-augmentation leg gives 0.02553 +- 0.02405 /
0.03531 +- 0.01845 over 5 seeds — degraded, not collapsed.

```bash
# IASBS, augmentation on / off
python iasbs/sphere.py train-nondirac --steps 500 --sigma 1.0 --iters 600 \
    --batch 500 --inner 16 --inner-h 4 --mb 16384 --mb-h 4096 --hidden 256 \
    --lr 1e-3 --ema 0.9995 --antithetic --seeds 5 --eval-every 1000 \
    --n-samples 100000 --ckpt-dir ckpt --tag sphere_w2_nd_anti \
    --out json/results_w2_iasbs_nd_anti.json
# (drop --antithetic, --tag sphere_w2_nd_plain for the no-aug leg;
#  add --inner 1 --inner-h 1 --mb 500 --mb-h 500, --tag sphere_w2_nd_inner1
#  for the equal-optimizer-step leg)

# R-ASBS, augmentation on (seed s = 0..4); drop --antithetic for the no-aug leg
python rasbs/rasbs_sphere_port.py --bimodal --steps 500 --epochs 600 \
    --batch 500 --init matlab --antithetic --seed $s --n-samples 100000 \
    --ckpt-dir ckpt --tag rasbs_sphere_w2anti_s$s \
    --out json/results_w2_rasbs_anti_s$s.json

# aggregate
python iasbs/analysis/weakness2_sphere.py
```

Artifacts: `json/results_weakness2_sphere.json`,
`json/results_w2_iasbs_nd_{anti,plain,inner1}.json`,
`json/results_w2_rasbs_anti_s{0..4}.json`,
`json/results_rasbs_sphere_matlabinit_s{0..4}.json` (the no-aug R-ASBS leg,
reused from §4.5) + `ckpt/sphere_w2_nd_{anti,plain,inner1}_seed{0..4}.pt`.

---

# 5. Stiefel manifold St(4,2)

`E(X) = tr(X^T H X)`, `H = diag(1, 2, 5, 8)` — R-ASBS's own `H` up to a change
of basis, and their own beta grid. `tau = 1/beta`. IASBS uses `sigma = sqrt(2)`,
`nq = 64` fibre quadrature nodes and `--antithetic` (the exact 16-fold symmetry
`T_s(X) = SXD`, automatically disabled with `--frame`).

| | IASBS | R-ASBS (`rasbs_port.py`) |
|---|---|---|
| script | `iasbs/stiefel.py` | port of `alg2_stiefel.m` @ `bb71d14` |
| basis | eigenbasis of `H` | ambient Z2xZ2 basis |
| source | `E_0 = [e_1, e_2]` (Dirac) | Haar |
| constraint | exact geodesic step | ambient Euler + QR retraction |
| network | `ScoreNet`, hidden 256 | `netU` + `netH` |
| parameters | 139,528 | 140,560 (1.01x) |
| integration steps | 199 | 199 |
| iters x inner | 1500 x 8 | 1000 epochs |
| batch / minibatch | 2048 / 16384 | 600 |
| lr | 1e-3 | 1e-3 |
| EMA / buffer | 0.9995 / 4 | — |
| eval samples | 100,000 | 5,000 (199-step sweep), 100,000 (step and high-beta reruns) |
| reference | MCMC, 2e5 chains x 4000 sweeps | same |

Both samples sets are scored with the `H` they were trained against; scoring the
port's samples with the IASBS `H` returns `E ≈ 8.00` at every beta (basis
mismatch, not failure).

## 5.1 Energy sweep in beta

`N` is the number of integration steps. `err` is measured against the shared
`reference` column.

| beta | reference | R-ASBS | err | N | IASBS | err | N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 7.9965 | 7.9546 | -0.042 | 199 | 7.9958 | **-0.001** | 199 |
| 0.01 | 7.9628 | 7.9407 | -0.022 | 199 | 7.9679 | **+0.005** | 199 |
| 0.1 | 7.6671 | 7.7262 | +0.059 | 199 | 7.6893 | **+0.022** | 199 |
| 0.5 | 6.4171 | 6.5426 | +0.126 | 199 | 6.5152 | **+0.098** | 199 |
| 1.3 | 4.7503 | 5.2008 | +0.451 | 199 | 4.9016 | **+0.151** | 199 |
| 2 | 4.0976 | 4.6444 | +0.547 | 199 | 4.2335 | **+0.136** | 199 |
| 5 | 3.4104 | 3.8893 | +0.479 | 199 | 3.5212 | **+0.111** | 199 |
| 7 | 3.2905 | 3.6104 | +0.320 | 199 | 3.3995 | **+0.109** | 199 |
| 10 | 3.2025 | 3.5275 | +0.325 | 199 | 3.3102 | **+0.108** | 199 |
| 20 | 3.1005 | 3.3858 | +0.285 | 199 | 3.2170 | **+0.116** | 199 |
| 50 | 3.0418 | 3.2897 | +0.248 | 1024 | 3.0534 | **+0.012** | 1592 |
| 100 | 3.0279 | 3.2487 | +0.221 | 1024 | 3.0496 | **+0.022** | 3184 |
| 200 | 3.0243 (MCMC) | 3.2482 | +0.224 | 199 | — | — | — |
| 1e6 | 3.0233 (MCMC, frozen) | 3.1847 | +0.161 | 199 | — | — | — |

Orthogonality residual `|X^T X - I|`: IASBS 1.1e-14 (32 steps) to 6.6e-14 (512
steps), largest measured anywhere 3.2e-13 (beta = 100, 3184 steps); R-ASBS
pinned near 3.1e-07 to 3.8e-07 at every beta and every step count.

MCMC reference caveat: for beta >~ 1000 acceptance goes to zero and the chain
freezes at 3.0233 for every beta from 1e3 to 1e6 (true limit 3), so the
reference itself carries about +0.023 of error there. At beta <= 100 two
independent chains agree to 0.0001-0.01.

```bash
python iasbs/stiefel.py verify        # gate D0, incl. spin-clock test
python iasbs/stiefel.py ref --mcmc-sweeps 4000 \
    --betas "0.001,0.01,0.1,0.5,1.3,2,5,7,10,20,50,100,200,1000,10000,1000000"
bash iasbs/scripts/run_scalefix.sh    # beta = 50, 100 at 199 steps
bash iasbs/scripts/run_anneal.sh      # 50 -> 100 anneal
bash iasbs/scripts/run_anneal_chain.sh    # 50 -> 65 -> 80 -> 100
bash iasbs/scripts/run_anneal_b100.sh     # single beta=100 leg
bash iasbs/scripts/run_anneal_b100_fine.sh # same at 796 steps
python rasbs/rasbs_port.py --check-retraction   # GS == sign-corrected QR
python rasbs/rasbs_port.py --out json/results_rasbs_stiefel.json
```

Artifacts: `json/results_stiefel_{grid,fill,scalefix,ref,sweep}.json`,
`json/results_stiefel_anneal_b{50,100,100_fine}.json`,
`json/results_chain_b{65,80,100}.json`,
`json/results_rasbs_stiefel.json` + `ckpt/stiefel_*_b*_seed0.pt` (each with its
`_mcmc.pt` reference), `ckpt/chain_b*_seed0.pt`, `ckpt/rasbs_b*.pt`.
`results_stiefel_grid.json` holds beta = 0.1 / 0.5 / 1.3 / 2 / 5 / 10;
`results_stiefel_fill.json` holds beta = 0.001 / 0.01 / 7 / 20 / 50 / 100.

## 5.2 Beyond the mean energy — KS(E), dispersion, constraint

Recovered from the stored samples by `analysis/stiefel_extra.py` (no retraining).
IASBS_600 is the budget-matched run (`--batch 400 --iters 1500`, exactly 600,000
terminal-oracle calls, matching R-ASBS).

| beta | metric | R-ASBS | IASBS | IASBS_600 |
|---:|---|---:|---:|---:|
| 1.3 | \|dE\| | 0.4581 | **0.1548** | 0.1656 |
| | KS(E) | 0.1407 | **0.0537** | 0.0552 |
| | E_std / ref (1.1628) | 1.25x | **1.05x** | 1.06x |
| | \|X^T X - I\| | 3.2e-07 | **3.5e-14** | 3.4e-14 |
| 2 | \|dE\| | 0.5513 | **0.1372** | 0.1610 |
| | KS(E) | 0.1857 | **0.0718** | 0.0753 |
| | E_std / ref (0.7807) | 1.65x | **1.10x** | 1.14x |
| | \|X^T X - I\| | 3.1e-07 | **3.3e-14** | 3.3e-14 |
| 5 | \|dE\| | 0.4781 | **0.1100** | 0.1106 |
| | KS(E) | 0.2750 | **0.1369** | 0.1353 |
| | E_std / ref (0.2903) | 3.27x | **1.23x** | 1.24x |
| | \|X^T X - I\| | 3.1e-07 | **3.2e-14** | 3.4e-14 |

KS null floor at n = 5,000 (the R-ASBS sample size) is **0.011**. Subsampling
IASBS to 5,000 moves its KS 0.0537 -> 0.0597, 0.0718 -> 0.0761,
0.1369 -> 0.1374.

Oracle-call accounting: the IASBS default uses 3,072,000 terminal-oracle calls
against R-ASBS's 600,000. Sample-size control: the one configuration measured
both ways (199 steps, beta = 50 and 100) gives 3.3116 / 3.2614 at 5,000 samples
and 3.3139 / 3.2626 at 100,000.

Wall clock, single A100: IASBS ~1130 s per beta at 199 steps, ~2180 s at 398,
~4380 s at 796; R-ASBS ~780 s per beta at 512 steps, ~1560 s at 1024. Every
`results_*.json` records `train_s` (IASBS) or `wall_s` (R-ASBS).

```bash
python iasbs/analysis/stiefel_extra.py    # no training; reads ckpt/
```

Artifacts: `json/results_stiefel_extra.json`,
`json/results_stiefel_matched.json`.

## 5.3 Step-count ablation

At beta = 2, everything except the step count held fixed:

| N_steps | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| R-ASBS error | 0.805 | 0.661 | 0.601 | 0.526 | 0.493 |
| IASBS error | 0.621 | 0.323 | 0.186 | 0.121 | **0.072** |

Per doubling IASBS falls ~1.7x, R-ASBS ~1.12x. Richardson extrapolation in 1/N
on the two finest R-ASBS grids gives a floor of **≈0.46**.

At beta >= 50, each cell measured against its own run's MCMC reference:

| beta | 199 | 398 | 512 | 796 | 1024 | 1592 | 3184 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| R-ASBS, 50 | +0.272 | — | +0.255 | — | **+0.248** | — | — |
| IASBS, 50 | +4.391 | +0.0685 | — | +0.0223 | — | **+0.0101** | — |
| R-ASBS, 100 | +0.235 | — | **+0.218** | — | +0.221 | — | — |
| IASBS, 100 | +5.290 | +4.063 | — | +3.535 | — | +0.750 | **+0.0187** |

The IASBS beta = 100 row is the 50 -> 65 -> 80 -> 100 chain
(`json/results_chain_b100.json`). The direct beta = 50 warm start
(`json/results_stiefel_anneal_b100_fine.json`) gives +3.4695 / +0.7458 / +0.0461
in the same three cells; at 3184 steps the chain gives dE +0.0187, KS(E) 0.258,
2.1x spread, 2602 s against the direct warm start's +0.0461, KS 0.312, 16x
spread, 4393 s.

**Loss-scale ablation.** `ScoreNet` factorises `score = out_scale * raw(t, X)`
with `out_scale` a running RMS of the label. Removing the beta^2 loss scale
(0.055 at beta = 0.001, 2.8e5 at beta = 100) changes the beta = 50 / 100
199-step results by nothing: +4.463 / +5.271 with the factorisation against
+4.391 / +5.290 without. `json/results_stiefel_scalefix.json`.

```bash
python iasbs/stiefel.py sweep --beta 2 --iters 1500 --mb 16384 \
    --ema 0.9995 --antithetic --tag stiefel_d3 \
    --out json/results_stiefel_sweep.json
bash rasbs/rasbs_steps.sh                   # their beta=2 step ablation
bash rasbs/rasbs_steps_highbeta.sh          # their beta=50/100 step ablation
bash rasbs/regen_f64.sh                     # float64 rerun at beta=2
```

Artifacts: `json/results_stiefel_sweep.json`,
`json/results_rasbs_steps_{32,64,128,256,512}.json`,
`json/results_rasbs_highbeta_steps_{199,512,1024}.json`,
`json/results_{rasbs_b2,stiefel_b2}_f64.json` + `ckpt/stiefel_d3_sweep_steps*.pt`,
`ckpt/rasbs_*.pt`.

## 5.4 Frame-sensitive target

`E(X) = tr(X^T H X) - lambda tr(C^T X)` with `lambda = 1`, `beta = 1`,
`H = diag(1,2,5,8)`, `C = [[0.7,-0.2],[0.1,0.8],[-0.4,0.3],[0.2,-0.5]]`. The
frame term breaks the 16-fold sign symmetry, so `--antithetic` is
auto-disabled. No analytic reference; MCMC is the only ground truth.

MCMC reference: `E = 4.6978 +- 1.5670`, `tr(C^T X) = 0.4977`, 200,000 chains
x 3,000 sweeps. Five seeds, native budget 2048 x 2500 = 5,120,000 terminal
oracle calls, 100,000 evaluation samples.

| steps | \|dE\| mean +- sd | KS(E) mean +- sd | E | tr(C^T X) | \|X^T X - I\| |
|---:|---:|---:|---:|---:|---:|
| 199 | 0.1264 +- 0.0065 | 0.0307 +- 0.0019 | 4.8242 +- 1.6161 | 0.5094 | 3.2e-14 |
| 398 | 0.0779 +- 0.0061 | 0.0185 +- 0.0021 | 4.7705 +- 1.6017 | 0.5125 | 5.5e-14 |
| 796 | 0.0564 +- 0.0021 | 0.0137 +- 0.0007 | 4.7537 +- 1.5972 | 0.5126 | 9.5e-14 |

Gate D2a (mean \|dE\| < 0.05) FAILs at 199 steps (0.1264); D2b (mean KS(E) <
0.05) PASSes (0.0307). Wall 1,889 s per seed.

```bash
bash iasbs/scripts/run_fair_stiefel.sh frame
```

Artifacts: `json/results_stiefel_frame_s5.json` +
`ckpt/stiefel_frame_s5_b1_seed{0..4}.pt`. The earlier 2-seed probe is
`json/results_stiefel_frame.json`.

## 5.5 Second moment, from disk

At beta = 2, after rotating the R-ASBS samples into the common eigenbasis,
`||E[XX^T] - target||_F` is **0.028** (IASBS) and **0.132** (R-ASBS) against
0.755 for Haar. Orthogonality residual recomputed across all 100,000 samples:
**3.0e-14** (IASBS) against **2.9e-07** (R-ASBS). Checkpoints store sample
tensors in float64 (`common.py:save_ckpt` promotes them), which is what makes
the second comparison resolvable — float32 epsilon is 1.2e-07.

## 5.6 Fair comparison against R-ASBS

Both samplers run on the same two targets, against the same MCMC reference,
with the same 100,000 evaluation samples and the same inference resolutions,
and each keeps its own native source, retraction and architecture. The
oracle budget is equalised at 600,000 terminal calls: R-ASBS's own
600 x 1000, matched by IASBS at 400 x 1500. `IASBS` is the unmatched native
budget (2048 x 2500 = 5,120,000) and is listed only to show what the extra
oracle calls buy.

### 5.6.1 Frame-sensitive target, beta = 1, 5 seeds per method

| method | budget | steps | \|dE\| mean +- sd | KS(E) mean +- sd | \|X^T X - I\| | wall/seed |
|---|---:|---:|---:|---:|---:|---:|
| IASBS | 5.12M | 199 | 0.1264 +- 0.0065 | 0.0307 +- 0.0019 | 3.2e-14 | 1,889 s |
| IASBS | 5.12M | 398 | 0.0779 +- 0.0061 | 0.0185 +- 0.0021 | 5.5e-14 | |
| IASBS | 5.12M | 796 | 0.0564 +- 0.0021 | 0.0137 +- 0.0007 | 9.5e-14 | |
| IASBS600 | 600k | 199 | 0.1888 +- 0.0102 | 0.0458 +- 0.0032 | 3.2e-14 | 1,117 s |
| IASBS600 | 600k | 398 | 0.1458 +- 0.0135 | 0.0353 +- 0.0038 | 5.5e-14 | |
| IASBS600 | 600k | 796 | 0.1263 +- 0.0071 | 0.0304 +- 0.0010 | 9.7e-14 | |
| R-ASBS | 600k | 199 | 0.3222 +- 0.0097 | 0.0820 +- 0.0025 | 4.0e-07 | 310 s |
| R-ASBS | 600k | 398 | 0.3078 +- 0.0139 | 0.0792 +- 0.0036 | 4.0e-07 | |
| R-ASBS | 600k | 796 | 0.3018 +- 0.0160 | 0.0775 +- 0.0036 | 3.8e-07 | |

IASBS600 gates: D2a FAIL (0.1888), D2b PASS (0.0458).

### 5.6.2 Trace target, 3 seeds per method, matched 600k budget

| beta | method | steps | \|dE\| mean +- sd | KS(E) mean +- sd | \|X^T X - I\| | wall/seed |
|---:|---|---:|---:|---:|---:|---:|
| 1.3 | IASBS600 | 199 | 0.1654 +- 0.0074 | 0.0547 +- 0.0010 | 3.3e-14 | 1,119 s |
| 1.3 | IASBS600 | 398 | 0.1237 +- 0.0054 | 0.0421 +- 0.0027 | 5.5e-14 | |
| 1.3 | R-ASBS | 199 | 0.4709 +- 0.0141 | 0.1433 +- 0.0052 | 4.0e-07 | 304 s |
| 1.3 | R-ASBS | 398 | 0.4520 +- 0.0125 | 0.1390 +- 0.0046 | 4.0e-07 | |
| 2 | IASBS600 | 199 | 0.1589 +- 0.0040 | 0.0764 +- 0.0004 | 3.2e-14 | 1,119 s |
| 2 | IASBS600 | 398 | 0.1146 +- 0.0052 | 0.0569 +- 0.0022 | 5.5e-14 | |
| 2 | R-ASBS | 199 | 0.5544 +- 0.0109 | 0.1876 +- 0.0039 | 4.1e-07 | 302 s |
| 2 | R-ASBS | 398 | 0.5291 +- 0.0108 | 0.1797 +- 0.0032 | 3.9e-07 | |
| 5 | IASBS600 | 199 | 0.1111 +- 0.0020 | 0.1365 +- 0.0014 | 3.2e-14 | 1,122 s |
| 5 | IASBS600 | 398 | 0.0699 +- 0.0006 | 0.0921 +- 0.0006 | 5.6e-14 | |
| 5 | R-ASBS | 199 | 0.4497 +- 0.0253 | 0.2599 +- 0.0162 | 3.9e-07 | 303 s |
| 5 | R-ASBS | 398 | 0.4231 +- 0.0222 | 0.2444 +- 0.0163 | 4.0e-07 | |

IASBS600 gates over the three betas: D1a FAIL (0.1451), D1b FAIL (0.0892).

The single-seed IASBS600 numbers these tables replace are in
`json/results_stiefel_matched.json`; the single-seed native-budget row per beta
is in `json/results_stiefel_extra.json` (beta = 1.3 / 2 / 5: \|dE\| 0.1548 /
0.1372 / 0.1100, KS(E) 0.0537 / 0.0718 / 0.1369).

```bash
bash iasbs/scripts/run_fair_stiefel.sh frame      # IASBS native, 5 seeds
bash iasbs/scripts/run_fair_stiefel.sh frame600   # IASBS600 frame, 5 seeds
bash iasbs/scripts/run_fair_stiefel.sh trace600   # IASBS600 trace, 3 seeds/beta
bash rasbs/run_fair.sh frame                                # R-ASBS frame, 5 seeds
bash rasbs/run_fair.sh trace                                # R-ASBS trace, 3 seeds/beta
```

Artifacts:

| leg | json | ckpt |
|---|---|---|
| IASBS native frame | `results_stiefel_frame_s5.json` | `stiefel_frame_s5_b1_seed{0..4}.pt` |
| IASBS600 frame | `results_stiefel_frame600.json` | `stiefel_frame600_b1_seed{0..4}.pt` |
| IASBS600 trace | `results_stiefel_matched_s3.json` | `stiefel_matched_s3_b{1.3,2,5}_seed{0,1,2}.pt` |
| R-ASBS frame | `results_rasbs_frame.json` | `rasbs_frame_b1_seed{0..4}.pt` |
| R-ASBS trace | `results_rasbs_m600.json` | `rasbs_m600_b{1.3,2,5}_seed{0,1,2}.pt` |

Figures: `fig5_stiefel` (energy curves, step refinement), `fig7_stiefel_frames`
(96 raw frames per sampler + second moment), `fig/table_stiefel.md` (generated
comparison table).

### 5.6.3 Consolidated table, frame target, beta = 1

Energy law, frame law, feasibility, oracle calls and wall time in one place.
Energy-law columns are read back from the training JSONs; MMD^2 is recomputed
for every row in a single pass so that legs added after §5.7 share one
bandwidth, one 4000-vs-4000 draw and one generator seed. MMD^2 values therefore
differ from §5.7's in the third significant figure — the draw order is not the
same — and neither differs from the other by more than the floor's own sd.

**Equal oracle budget (600,000 terminal calls), 199 steps:**

| method | oracle calls | \|dE\| | KS(E) | MMD^2 vs MCMC | \|X^T X - I\| | wall/seed | seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| IASBS600 | 600,000 | **0.1888 +- 0.0102** | **0.0458 +- 0.0032** | **1.77e-05 +- 1.8e-05** | **3.2e-14** | 1,117 s | 5 |
| R-ASBS | 600,000 | 0.3222 +- 0.0097 | 0.0820 +- 0.0025 | 4.28e-04 +- 1.8e-04 | 4.3e-07 | 310 s | 5 |
| IASBS native (unmatched) | 5,120,000 | 0.1264 +- 0.0065 | 0.0307 +- 0.0019 | 1.43e-05 +- 3.2e-05 | 3.3e-14 | 1,889 s | 5 |
| MMD^2 split-half floor | — | — | — | 6.94e-06 +- 7.37e-05 | — | — | — |

**Equal wall time (~310 s), 199 steps:**

| method | oracle calls | \|dE\| | KS(E) | MMD^2 vs MCMC | \|X^T X - I\| | wall/seed | seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| IASBS wall310 | 166,000 | 1.4509 +- 0.0329 | 0.3290 +- 0.0071 | 1.75e-03 | 3.2e-14 | 309 s | 5 |
| R-ASBS | 600,000 | 0.3222 +- 0.0097 | 0.0820 +- 0.0025 | 4.28e-04 | 4.3e-07 | 310 s | 5 |

`IASBS wall310` is `--iters 415 --batch 400`, i.e. `run_fair_stiefel.sh frame600`
truncated to R-ASBS's wall clock; every other flag is the `frame600` leg's.

**Training-budget ladder for R-ASBS, 199 steps.** Same script and flags as the
600k row, `--epochs` raised from 1000 to 6100 so the oracle budget is 6.1x and
the wall clock matches `IASBS native`; 3 seeds.

| method | oracle calls | \|dE\| | KS(E) | MMD^2 vs MCMC | \|X^T X - I\| | wall/seed | seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| R-ASBS | 600,000 | 0.3222 +- 0.0097 | 0.0820 +- 0.0025 | 4.28e-04 +- 1.8e-04 | 4.3e-07 | 310 s | 5 |
| R-ASBS 6.1x | 3,660,000 | 0.1461 +- 0.0426 | 0.0367 +- 0.0080 | 4.67e-04 +- 1.5e-04 | 4.1e-07 | 1,846 s | 3 |
| IASBS native | 5,120,000 | 0.1264 +- 0.0065 | 0.0307 +- 0.0019 | 1.43e-05 +- 3.2e-05 | 3.3e-14 | 1,889 s | 5 |

Per-seed `|dE|` for `R-ASBS 6.1x`: 0.1737 / 0.1676 / 0.0971; KS(E) 0.0410 /
0.0415 / 0.0275; `E[tr(C^T X)]` 0.4793 / 0.4995 / 0.5458 against the MCMC
reference 0.4977.

**Step-refinement ladder, `|dE|` at 199 / 398 / 796 steps:**

| method | 199 | 398 | 796 | change |
|---|---:|---:|---:|---:|
| IASBS native | 0.1264 | 0.0779 | 0.0564 | -55% |
| IASBS600 | 0.1888 | 0.1458 | 0.1263 | -33% |
| R-ASBS | 0.3222 | 0.3078 | 0.3018 | -6% |
| R-ASBS 6.1x | 0.1461 | 0.1256 | 0.1151 | -21% |

```bash
# the 6.1x budget ladder
python rasbs/rasbs_port.py --frame --lam 1.0 --betas 1.0 --epochs 6100 \
    --seeds 3 --refine 2,4 --n-samples 100000 --ref-stem stiefel_frame \
    --verbose --tag rasbs_w2_big --out json/results_w2_rasbs_frame_big.json
# the table above
python iasbs/analysis/weakness2_stiefel.py
```

Artifacts: `json/results_weakness2_stiefel.json`,
`json/results_w2_stiefel_wall310.json`,
`json/results_w2_rasbs_frame_big.json` +
`ckpt/stiefel_w2_wall310_b1_seed{0..4}.pt`,
`ckpt/rasbs_w2_big_b1_seed{0,1,2}.pt`. All other rows reuse the §5.6.1
checkpoints.

## 5.7 Full frame-law accuracy, beta = 1, 5 seeds per method

Frame-sensitive statistics of `X` itself, against the 100,000-frame MCMC
reference `ckpt/stiefel_frame_s5_b1_mcmc.pt`. Sections 5.6.1 / 5.6.2 score the
energy law only; this section scores the law of the frame. Joint metric is
unbiased MMD^2 on `vec(X)` in R^8 with kernel `exp(-|a-b|^2 / (2 sigma^2))`,
`sigma^2 = 3.8249` (median heuristic on the reference), 4000 vs 4000 frames,
8 draws per seed.

R-ASBS checkpoints store `X` in the R-ASBS basis (`X_r = U X`); every R-ASBS row
below is mapped back by `U^T` first.

| method | steps | E[tr(C^T X)] mean +- sd | MMD^2 vs MCMC mean +- sd | max entry err of E[X] | \|Cov - Cov_ref\|_F |
|---|---:|---:|---:|---:|---:|
| MCMC ref | — | 0.495298 | — | — | — |
| IASBS | 199 | 0.50720 +- 0.00432 | 4.00e-05 +- 6.86e-05 | 0.00767 | 0.0308 |
| IASBS600 | 199 | 0.50548 +- 0.00546 | 3.77e-05 +- 4.39e-05 | 0.00773 | 0.0399 |
| R-ASBS | 199 | 0.52744 +- 0.00631 | 4.594e-04 +- 1.75e-04 | 0.02245 | 0.0599 |
| R-ASBS | 398 | 0.52735 +- 0.00736 | 4.626e-04 +- 2.38e-04 | 0.02292 | 0.0584 |
| R-ASBS | 796 | 0.52886 +- 0.01003 | 5.014e-04 +- 2.28e-04 | 0.02503 | 0.0582 |

Noise floor, disjoint halves of the MCMC reference, 16 draws:
MMD^2 = 6.94e-06 +- 7.37e-05.

`E[X]`, reference vs IASBS(199) vs R-ASBS(796), row-major 4x2:

| entry | MCMC ref | IASBS | R-ASBS |
|---|---:|---:|---:|
| (0,0) | 0.27047 | 0.27815 | 0.29550 |
| (0,1) | -0.07015 | -0.07328 | -0.07518 |
| (1,1) | 0.28200 | 0.28845 | 0.29260 |
| (2,0) | -0.06012 | -0.06112 | -0.06938 |
| (3,0) | 0.01862 | 0.01660 | 0.02556 |

Var of `vec(X)` coordinates 1, 5, 7 (diag of the 8x8 covariance):

| coord | MCMC ref | IASBS | IASBS600 | R-ASBS 796 |
|---|---:|---:|---:|---:|
| 1 | 0.35457 | 0.34217 | 0.33848 | 0.32740 |
| 5 | 0.15436 | 0.16053 | 0.16360 | 0.17314 |
| 7 | 0.08390 | 0.09181 | 0.09491 | 0.09866 |

```bash
# json/results_frame_law_audit.json (per-seed values + full 8x8 covariances)
```

## 5.8 Reference-bridge audit

`iasbs/analysis/bridge_audit.py`. Production resolutions: `steps = 128`, fibre quadrature
`nq = 64`, fibre sampling grid `nfib = 512`, S^3 table `ntheta = 16385`.

Tangent-to-skew map `Omega_X(S) = S X^T - X S^T - X (X^T S) X^T` (`skew_lift`),
512 random tangent vectors at random frames:

| quantity | value |
|---|---:|
| tangency residual \|X^T S + S^T X\| | 4.89e-15 |
| max \|Omega X - S\| | 3.55e-15 |
| max \|Omega + Omega^T\| | 3.03e-15 |
| control: max \|Omega X - D\|, D non-tangent | 5.646 |

The same numbers appear in `iasbs/stiefel.py verify` (gate D0).

Bridge correctness by the mixture identity: draw `Y` from the terminal law of
the unconditioned reference process, run the bridge to `Y`, compare against the
unconditioned marginal at intermediate times. 8000 paths, MMD^2 on 2000 vs 2000,
8 draws, median-heuristic bandwidth per time.

| manifold | t | MMD^2 | split-half floor | floor sd |
|---|---:|---:|---:|---:|
| S^2 | 0.25 | -1.91e-05 | -3.16e-05 | 1.99e-04 |
| S^2 | 0.50 | -7.78e-05 | 2.05e-05 | 2.80e-04 |
| S^2 | 0.75 | 1.39e-04 | -3.79e-05 | 3.11e-04 |
| St(4,2) | 0.25 | 3.38e-05 | 4.03e-05 | 1.91e-04 |
| St(4,2) | 0.50 | 5.38e-05 | 7.31e-05 | 1.74e-04 |
| St(4,2) | 0.75 | -1.25e-05 | 1.33e-05 | 1.50e-04 |

St(4,2) bridge endpoint: `max |bp[0] - E0| = 0.0`.

Production bridge (128 steps) against an 8x finer bridge (1024 steps) on the
same terminal points, t = 0.5: MMD^2 = -5.67e-05, floor 1.32e-05.

Resolution sensitivity. Fibre quadrature order for `log p^St`, against
`nq = 512`, 4000 frames:

| nq | max \|dlog_pst\| | mean \|dlog_pst\| |
|---:|---:|---:|
| 16 | 1.97e-11 | 1.82e-12 |
| 32 | 7.11e-15 | 2.38e-15 |
| 64 | 6.66e-15 | 2.21e-15 |
| 128 | 1.78e-14 | 6.18e-15 |
| 256 | 7.11e-15 | 2.30e-15 |

Fibre sampling grid, law of the lifted pair `(a, b)` against `nfib = 4096`,
split-half floor -1.77e-05:

| nfib | MMD^2 vs 4096 |
|---:|---:|
| 64 | -2.17e-04 |
| 128 | 1.81e-04 |
| 512 | -1.42e-05 |
| 2048 | 2.02e-04 |

S^3 table resolution against `ntheta = 65537`, 4001 points:

| ntheta | max \|dlogp\| | max \|d dlogp/du\| |
|---:|---:|---:|
| 4097 | 7.09e-07 | 3.55e-06 |
| 16385 | 4.15e-08 | 1.98e-07 |
| 32769 | 1.14e-08 | 5.84e-08 |

```bash
# json/results_bridge_audit.json
PYTHONPATH=.:iasbs python iasbs/stiefel.py verify
```

---

# 6. Earthquakes — S^2, real data

Target: epicentre distribution of the 4,776 magnitude >= 6.0 earthquakes in
`rasbs_ref/query.csv`. Each epicentre becomes a von Mises-Fisher mode,
`pi(x) ∝ (1/N) sum_i exp(kappa <m_i, x>)` at `kappa = 600` (angular width
≈ 2.4 deg). Fit on a random 70% (3,343 modes), hold out 30% (1,433).

Exactly solvable despite being real data: `int_{S^2} exp(kappa<m,x>) dx =
4 pi sinh(kappa)/kappa` is independent of `m`, so `pi` is exactly an
equal-weight vMF(kappa) mixture — iid reference samples are available in closed
form and `log Z = log(4 pi sinh kappa / kappa) = 595.4409474111932`.

| | configuration |
|---|---|
| script | `iasbs/earthquake.py` |
| source `x_0` | fixed point on S^2 (Dirac) |
| network | multi-scale random-Fourier score net, hidden 512, 931,331 parameters |
| kappa schedule | annealed 150 -> 300 -> 450 -> 600, one warm-started net + EMA |
| integration steps | 512 |
| iters x inner | 6000 x 16 |
| batch / minibatch | 2048 / 4096 |
| lr / EMA | 1e-3 / 0.999 |
| drift clip | 200 |
| eval samples | 100,000 |

R-ASBS column is measured, not quoted: `rasbs_sphere_port.py` ports their
`earthquake_sphere_ex.m` (same kappa staircase, 250 steps, drift clip 30, same
random-Fourier network, their Haar source and tangent-space Gaussian bridge) and
is scored by `earthquake.py`'s metrics.

| metric | IASBS | R-ASBS (measured) | iid floor |
|---|---:|---:|---:|
| dE (ref -594.7238 / -594.7293) | **-0.0771** | +0.8466 | -0.0069 / +0.0052 |
| KS(E) — gate E1 | **0.10048** FAIL | 0.22614 FAIL | 0.00327 / 0.00403 |
| mode-histogram TV — gate E2 | 0.50884 FAIL | **0.29385** FAIL | 0.08910 / 0.08982 |
| KS(angle to nearest mode) | **0.08048** | 0.13445 | 0.00744 / 0.00383 |
| mode coverage (ref 0.9737 / 0.9722) | 0.7679 | **0.9432** | 0.9707 / 0.9749 |
| energy distance | 0.30171 | **0.10079** | 0.00031 / 0.00120 |
| `\|norm-1\|` — gate E3 | **2.2e-16** PASS | 3.3e-16 PASS | 2.2e-16 |

Two iid floors: each column was scored against its own independently drawn
100,000-sample reference.

**Held-out check** (1,433 modes the sampler never saw): KS(E) 0.05457, coverage
0.88625, mode TV 0.51013 — against 0.10048 / 0.7679 / 0.50884 on the training
modes.

**Training trace over the four kappa stages:** dE improved monotonically
5.47 -> -0.07; KS(theta) improved monotonically 0.495 -> 0.080; mode TV
oscillated in 0.34-0.55 throughout, best 0.339 at the kappa = 300 stage, ending
at 0.509. Within the final kappa = 600 block mode TV went 0.481 -> 0.509 and
coverage 0.825 -> 0.767.

**Exact control** (quadrature score instead of learned): only run at a reduced
`kappa = 20`, where it is clean first-order with no bias floor. Not run at
`kappa = 600`.

Haar-source ablation is not available: for a Dirac `mu` the h-transform's
initial density is `mu(x_0) * h(x_0, 0)` with `h(x_0, 0) = int pi = 1`, so the
source survives exactly; for a Haar `mu` it is `4 pi (P_{r01} pi)(x_0)`, the
heat-smoothed target. Measured at `r01 = 1`: initial density over uniform ranges
0.833 to 1.175 (1.41x spread), `TV(initial law, Haar) = 0.0376`.

No figure code path exists for this experiment.

```bash
python iasbs/earthquake.py verify                    # sampler + log Z identities
python iasbs/earthquake.py exact --kappa 20          # quadrature control, step sweep
python iasbs/earthquake.py train --kappa 600 \
    --anneal 150 300 450 600 --steps 512 --iters 6000 --hidden 512 \
    --n-dir 384 --bw-hi 128 --max-drift 200 \
    --tag earthquake_k600 --out json/results_earthquake_k600.json
python rasbs/rasbs_sphere_port.py --problem quake              # the R-ASBS column
```

Artifacts: `json/results_earthquake_k600.json`, `json/results_earthquake.json`,
`json/results_rasbs_sphere_quake.json` + `ckpt/earthquake_k600.pt`,
`ckpt/rasbs_sphere_quake.pt`. Needs `rasbs_ref/query.csv` (gitignored).

---

# 7. Computational cost

## 7.1 IASBS

| experiment | iters | steps | wall (s) | s / iter | adjoint rollouts | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| Ising L=4 non-Dirac | 3000 | 256 | 1,869 | 0.62 | 0 | `ising_nd_L4.pt` |
| Ising L=5 non-Dirac | 3000 | 256 | 36,966 | 12.3 | 0 | `ising_nd_L5.pt` |
| Occupation m=4 Dirac | 1500 | 128 | no log (<= 214) | — | 0 | `occ4_full.pt` |
| Occupation m=4 non-Dirac | 1500 | 128 | 214 | 0.14 | 0 | `occ_nd_m4.pt` |
| Occupation m=32 (seed 0) | 3000 | 128 | 3,230 | 1.08 | 0 | `occ_nd_s32_seed0.pt` |
| Occupation m=32 (seed 1) | 3000 | 128 | 3,231 | 1.08 | 0 | `occ_nd_s32_seed1.pt` |
| Occupation m=128 | 3000 | 128 | 1,523 | 0.51 | 0 | `occ_nd_s128.pt` |
| Occupation m=1000 | 1500 | 256 | 15,225 | 10.15 | 0 | `occ_nd_s1000.pt` |
| Fixed-support toy, Dirac | 3000 | 256 | 3,050 | 1.02 | 0 | `fs_toy_v2_dirac.pt` |
| Fixed-support toy, non-Dirac | 3000 | 256 | 3,460 | 1.15 | 0 | `fs_toy_v2_nd.pt` |
| GB1 k=3 annealed, Dirac | 3 x 1000 | 256 | 5,814 | 1.94 | 0 | `fs_gb1_k3_dirac_{A,B,C}.pt` |
| GB1 k=3 annealed, non-Dirac | 3 x 1000 | 256 | 6,079 | 2.03 | 0 | `fs_gb1_k3_nd_{A,B,C}.pt` |
| Sphere non-Dirac, per seed | 4000 | 128 | 2,124 - 3,142 | 0.53 - 0.79 | 0 | `sphere_nd_seed{0..4}.pt` |
| Sphere non-Dirac, 5 seeds | 20000 | 128 | 13,820 | 0.69 | 0 | — |

Per-seed sphere wall: 3076 / 2841 / 3142 / 2637 / 2124 s.

R-ASBS sphere (`--init matlab`, 5 seeds): 255 / 255 / 261 / 256 / 260 s per
seed, 1,287 s total (`rasbs/logs/bimodal_mi_s{0..4}.log`). The default-init legs
predate the logging convention and have no recoverable wall clock. R-ASBS result
JSONs carry `config, final, floor, history, params, commit` and no timing field.

## 7.2 DAM

Every DAM leg uses the same control box: `--a-clamp 8 --m-clip 5 --ess-min 3
--coef-cap 10`. The flags are off by default, so the K sweep of §2.5 reproduces
bit-for-bit (`dam.tests_math`: 11/11 gates pass).

| flag | what it bounds | value |
|---|---|---:|
| `--a-clamp C` | `\|a_theta\| <= C`, escape-rate multiplier `<= e^C` | 8 |
| `--m-clip C` | `\|log m_hat\| <= C` in the gKL loss | 5 |
| `--ess-min E` | drop labels whose denominator ESS `< E` | 3 |
| `--coef-cap R` | winsorise the `r/q` prefactor at `R x` its batch median | 10 |

| experiment | K | iters | wall (s) | s / iter | f1 evals | CTMC jumps | jumps / f1 eval |
|---|---:|---:|---:|---:|---:|---:|---:|
| Occupation m=4 | 16 | 400 | 7,380.9 | 18.5 | 6,963,200 | 67,417,741 | 9.68 |
| Occupation m=4 | 16 | 1200 | 11,529.3 | 9.6 | 20,889,600 | 185,004,804 | 8.86 |
| Occupation m=4 | 64 | 400 | 3,177.7 | 7.9 | 26,624,000 | 222,041,777 | 8.34 |
| Ising L=4 | 32 | 5000 | 11,428 | 2.29 | 168,960,000 | 974,800,000 | 5.77 |
| Ising L=4 | 64 | 2500 | 6,950 | 2.78 | 166,400,000 | 924,000,000 | 5.55 |
| Ising L=4 | 32 | 7000 | 16,226 | 2.32 | 236,544,000 | 1,339,809,998 | 5.66 |
| Ising L=5 | 32 | 5000 | 15,391 | 3.08 | 168,960,000 | 1,372,615,519 | 8.12 |
| Occupation-scale m=32 | 64 | 1000 | 16,572 | 16.6 | 66,560,000 | 5,849,804,432 | 87.9 |
| Occupation-scale m=128 | 64 | 500 | 67,693 | 135.4 | 33,280,000 | 12,889,415,391 | 387.3 |
| Fixed-support toy | 16 | 1200 | 7,508 | 6.26 | 20,889,600 | 118,996,727 | 5.70 |
| GB1 k=3 annealed | 16 | 3 x 400 | 7,512 | 6.26 | 20,889,600 | 127,340,484 | 6.10 |

Measured CPU time equals wall time to within 1% on every DAM leg (each Gillespie
rollout is a Python `while` loop). `dam/run_dam.sh` queues legs serially.

## 7.3 What `f1 evals = 0` counts

The counter is `stats["f1_evals"] = B * (K + K_num)` in
`dam/core.py:estimate_log_adjoint`, incremented at `dam/discrete.py:318`. It
counts literal calls to `adapter.log_f1(x)` **at the endpoint of a simulated
CTMC rollout**. IASBS is zero on that quantity and only that quantity.

| quantity | IASBS Ising L=4 (nD) | IASBS occ m=4 (Dirac) | IASBS sphere nD (per seed) | DAM occ m=4 (Dirac), K=16 |
|---|---:|---:|---:|---:|
| extra adjoint rollouts | **0** | **0** | **0** | 20,889,600 |
| `f1` evals at rollout endpoints | **0** | **0** | **0** | 20,889,600 |
| CTMC jumps simulated for the adjoint | **0** | **0** | **0** | 185,004,804 |
| target energy `E` evals, on the fly | 14,745,600,000 | 0 (table) | 0 (via gradient only) | 0 (table) |
| one-off precomputed `E` table | 12,870 states | 35 states | 2,562-node spectral table | 35 states |
| energy-gradient `grad E` evals | **0** | **0** | 32,768,000 | **0** |
| on-policy endpoint simulation (state-steps) | 1,572,864,000 | 98,304,000 | 4,194,304,000 | (counted as jumps) |

Ising L=4: `terminal_labels` (`fixed_ising.py:428`) calls `space.energies_of(Xg)`
on every one-swap neighbour of the endpoint — `mb=1024 x Np=C(16,2)=120 x
inner=40 x iters=3000` = 14.7e9, more than DAM's `f1` count, as a single fused
vectorised reduction. Sphere: `sphere.py:767` calls `C.sphere_energy_grad(y1)`
once per endpoint per iteration, `batch=8192 x iters=4000 = 32,768,000` per
seed, 163,840,000 across 5 seeds.

---

# 8. Multi-seed reproducibility

Every number published elsewhere in this README is a single run at seed 0. This
section re-runs the 18 `group:main` rows at seeds 1 and 2 -- 35 additional runs,
identical in every respect except the seed -- and reports the across-seed
spread. The five long DAM rows (`ising_L4_dam`, `ising_L5_dam`, `occ4_dam`,
`occs32_dam`, `occs128_dam`) are not included.

`sd` here is the **sample** standard deviation (ddof = 1) over n = 3, which is
what `aggregate_seeds.py` emits. Note this differs from the sphere tables in
sections 4.2 and 4.5, which quote a population sd (ddof = 0) over n = 5 seeds.

That the added runs differ from the published ones *only* in the seed is checked
mechanically rather than by inspection:

```bash
python iasbs/scripts/verify_multiseed.py
# 0 mismatch(es), 0 skip(s)   over 30 row/stage combinations
```

The script re-parses each main's argument parser, replays the recorded command
line for the seed-0 artifact and for every added seed, and diffs the resulting
config dicts field by field.

## 8.1 Headline metric, three seeds

TV is exact (full enumeration) on the Ising, occupation m = 4, toy and GB1 rows.
The occupation m >= 32 rows cannot be enumerated, so they report KS(occ)
against the exact reference; those values are the last training-history entry,
**not** the top-level `reference` block, which is the untrained baseline sampler
(KS(occ) ~ 0.205).

| row | metric | seed 0 (paper) | 3-seed mean +- sd | seed 1 | seed 2 | spread | viol |
|---|---|---:|---:|---:|---:|---:|---:|
| Ising 4x4, Dirac | TV | 0.04158 | **0.04134 +- 0.00592** | 0.03531 | 0.04715 | 1.34x | 0 |
| Ising 4x4, non-Dirac | TV | 0.03470 | **0.03685 +- 0.00269** | 0.03597 | 0.03987 | 1.15x | 0 |
| Ising 5x5, Dirac | TV | 0.07698 | **0.07832 +- 0.00541** | 0.07371 | 0.08427 | 1.14x | 0 |
| Ising 5x5, non-Dirac | TV | 0.07228 | **0.07296 +- 0.00416** | 0.06919 | 0.07743 | 1.12x | 0 |
| occupation m=4, Dirac | TV | 0.01273 | **0.01340 +- 0.00058** | 0.01373 | 0.01373 | 1.08x | 0 |
| occupation m=4, non-Dirac | TV | 0.01248 | **0.01449 +- 0.00188** | 0.01621 | 0.01479 | 1.30x | 0 |
| occupation m=32, Dirac | KS_occ | 0.00432 | **0.00733 +- 0.00566** | 0.00381 | 0.01386 | 3.64x | 0 |
| occupation m=32, non-Dirac | KS_occ | 0.01512 | **0.01686 +- 0.00194** | 0.01895 | 0.01650 | 1.25x | 0 |
| occupation m=128, Dirac | KS_occ | 0.00835 | **0.00862 +- 0.00100** | 0.00777 | 0.00972 | 1.25x | 0 |
| occupation m=128, non-Dirac | KS_occ | 0.01270 | **0.01029 +- 0.00232** | 0.00808 | 0.01007 | 1.57x | 0 |
| occupation m=1000, Dirac | KS_occ | 0.01024 | **0.01322 +- 0.00538** | 0.00999 | 0.01943 | 1.94x | 0 |
| occupation m=1000, non-Dirac | KS_occ | 0.01109 | **0.01109 +- 0.00321** | 0.01430 | 0.00788 | 1.81x | 0 |
| toy fixed-support, Dirac | TV | 0.00774 | **0.01043 +- 0.00332** | 0.01414 | 0.00940 | 1.83x | 0 |
| toy fixed-support, non-Dirac | TV | 0.00776 | **0.00900 +- 0.00115** | 0.00923 | 0.01003 | 1.29x | 0 |
| toy fixed-support, DAM | TV | 0.08285 | **0.08400 +- 0.00166** | 0.08325 | 0.08591 | 1.04x | 0 |
| GB1 k=3 (stage C), Dirac | TV | 0.13568 | **0.14371 +- 0.01123** | 0.15654 | 0.13889 | 1.15x | 0 |
| GB1 k=3 (stage C), non-Dirac | TV | 0.13226 | **0.13110 +- 0.00396** | 0.12669 | 0.13435 | 1.06x | 0 |
| GB1 k=3 (stage C), DAM | TV | 0.26472 | **0.25750 +- 0.00662** | 0.25606 | 0.25170 | 1.05x | 0 |

## 8.2 What moved

Every gate verdict is unchanged from seed 0, and `violations` is 0 in all 53
runs. `bad_labels` and `skipped_steps` are 0 wherever the main reports them.

Two published numbers are noticeably lucky at seed 0:

- **Occupation m = 32, Dirac, KS(occ).** Section 2.3 publishes 0.0043; the three
  seeds are 0.00432 / 0.00381 / 0.01386, a 3.6x spread and the widest in the
  table. The scaling story survives -- the 3-seed means are still monotone in m
  (0.00733, 0.00862, 0.01322 for m = 32, 128, 1000) -- but the single-seed value
  understates the run-to-run variation by a large factor.
- **Occupation m = 4, non-Dirac, corrector max err.** Section 2.2 publishes
  3.59e-4; the three seeds are 3.59e-4 / 1.40e-3 / 9.33e-4, mean 8.96e-4 +-
  5.19e-4, a 3.9x spread. All three are small in absolute terms and the A1 gate
  (which is on TV, not on this diagnostic) passes at every seed.

One published number is mildly pessimistic:

- **Occupation m = 1000, Dirac, conditioning** (section 2.4). Seed 0 gives
  0.01607 against the exact 0.01443; the 3-seed mean 0.01401 is closer to exact,
  so seed 0 is the worst of the three.

Structural observations that hold across the sweep:

- The non-Dirac source is consistently *more* seed-stable than the Dirac source
  on the same space: sd ratio Dirac/non-Dirac is 2.8x on GB1, 2.9x on toy and
  2.2x on Ising 4x4. On GB1 and toy the non-Dirac 3-seed mean is also the better
  of the two (0.13110 vs 0.14371, and 0.00900 vs 0.01043).
- DAM is the most seed-stable method in the table (spread 1.04x-1.05x), which is
  expected: its rollouts average over K = 16 annealing stages.
- `KS_max` at m = 1000 varies by 1.5x across seeds (0.2215 / 0.1470 / 0.1975).
  It is a max over 1000 marginals and should not be quoted as a point estimate.

## 8.3 Reproduction

```bash
# one row, seeds 1 and 2
bash iasbs/scripts/multiseed.sh <row> "1 2" <keep|drop>

# the whole group:main sweep, two GPUs sharing one atomic work pool
bash iasbs/scripts/multiseed_pool.sh init
CUDA_VISIBLE_DEVICES=0 bash iasbs/scripts/multiseed_pool.sh work w0 &
CUDA_VISIBLE_DEVICES=1 bash iasbs/scripts/multiseed_pool.sh work w1 &

# collapse to mean +- sd
python iasbs/scripts/aggregate_seeds.py --json json/seed_summary_main.json <bases...>
python iasbs/scripts/aggregate_seeds.py --stage C --json json/seed_summary_gb1.json \
    fs_gb1_k3_dirac fs_gb1_k3_nd dam_fs_gb1_k3_K16
```

`multiseed_pool.sh` runs the three `scale-nondirac` rows (`occs32_nd`,
`occs128_nd`, `occs1000_nd`) in `keep` mode, because their mid-training
evaluation block draws `torch.randint` and therefore advances the training RNG
stream; changing `--eval-every` on those rows would change the trajectory.
Every other row runs in `drop` mode (`--eval-every 1000000000`), whose
evaluation calls are all `no_grad` and RNG-free, leaving the stream identical
while removing a large diagnostic overhead.

Artifacts: `json/seed_summary_main.json`, `json/seed_summary_gb1.json`, and
`json/results_<base>_s{1,2}.json` for every row above.

---

# 9. Analytic corrector, non-Dirac occupation

The uniform source over the `m` corner states `N e_c` makes the static bridge
`S_m`-invariant, so `f_0` is constant on the source support and
`f_1(xi) ∝ (1/m) Σ_c p_base(xi | N e_c)`.  That mixture has a closed form,
giving an exact terminal ratio with no corrector network.  `--corrector
analytic` in `iasbs/occupation.py` uses it (`labels_analytic`), which drops the
corrector head entirely (`--inner-h 0`, half the parameters).

## 9.1 Symmetry check, m = N = 4, exact Sinkhorn

| quantity | value |
|---|---|
| Sinkhorn marginal residual | 2.776e-17 |
| `f_0` relative spread over the 4 source atoms | 1.332e-15 |
| spread of `log f_1 - log pbar` | 8.882e-16 |
| closed-form `log pbar` vs enumerated | 2.665e-15 |
| `q_atoms` vs `occupation_single_particle_probs` | 0.0 |
| `log` conditional-mean training target vs `h` | 4.441e-16 |
| exact `log R` vs analytic `log R` | 2.442e-15 |
| O(m) `labels_analytic` vs enumerated full sum | 8.882e-15 |

`json/results_weakness3_corrector.json`

## 9.2 Comparison table

10000 samples at m = 32, 128; 4000 at m = 1000.  128 tau-leap steps
(256 at m = 1000).  All four rows scored against one common exact-target draw.
`exact_floor` is a second independent exact draw (finite-sample floor).

| m | row | KS_occ | KS_max | W1(max) | W1(max)/N | E | E[max_i eta_i] | viol |
|---|---|---|---|---|---|---|---|---|
| 32 | reference | 0.2052 | 0.8119 | 3.2092 | 0.1003 | 17.1085 | 3.5512 | 0 |
| 32 | exact_floor | 0.0002 | 0.0077 | 0.0403 | 0.0013 | 13.2299 | 6.7519 | 0 |
| 32 | learned | 0.0167 | 0.0988 | 0.4662 | 0.0146 | 13.5924 | 6.2942 | 0 |
| 32 | analytic | 0.0205 | 0.1177 | 0.5572 | 0.0174 | 13.6734 | 6.2034 | 0 |
| 128 | reference | 0.2052 | 0.9095 | 5.1256 | 0.0400 | 67.8589 | 4.6536 | 0 |
| 128 | exact_floor | 0.0005 | 0.0081 | 0.0318 | 0.0002 | 52.1571 | 9.7656 | 0 |
| 128 | learned | 0.0129 | 0.0316 | 0.2755 | 0.0022 | 53.2908 | 9.5037 | 0 |
| 128 | analytic | 0.0112 | 0.0976 | 0.6102 | 0.0048 | 53.1775 | 9.1692 | 0 |
| 1000 | reference | 0.2031 | 0.4902 | 4.2055 | 0.0042 | 526.1046 | 18.6128 | 0 |
| 1000 | exact_floor | 0.0002 | 0.0082 | 0.0630 | 0.0001 | 405.6946 | 14.4563 | 0 |
| 1000 | learned | 0.0113 | 0.3473 | 1.7858 | 0.0018 | 412.8590 | 16.0715 | 0 |
| 1000 | analytic | 0.0135 | 0.2917 | 1.8098 | 0.0018 | 415.2987 | 12.6245 | 0 |
| 1000 | analytic s1 | 0.0077 | 0.2507 | 1.5810 | 0.0016 | 411.5978 | 12.8533 | 0 |

Exact-target `E[max_i eta_i]`: 6.7604 (m = 32), 9.7792 (m = 128), 14.4343
(m = 1000).  `json/results_weakness3_table.json`, and
`json/results_weakness3_table_m1000_analytic_s1.json` for the second m = 1000
analytic seed (same exact draw, same 256 steps, 4000 samples).

## 9.3 Learned corrector vs analytic, held-out target states

`h = log R` on ~3900 held-out exact draws.  `sd(h)` is the spread of the exact
value, for scale.

| ckpt | m | RMSE | max abs | bias | sd(h) |
|---|---|---|---|---|---|
| `occ_nd_s32_seed0` | 32 | 0.0170 | 0.1046 | +0.0087 | 0.8991 |
| `occ_nd_s32_seed1` | 32 | 0.0204 | 0.1610 | +0.0127 | 0.9056 |
| `occ_nd_s32_s2` | 32 | 0.0178 | 0.1353 | +0.0074 | 0.9061 |
| `occ_nd_s128` | 128 | 0.0852 | 0.8187 | -0.0266 | 0.8746 |
| `occ_nd_s128_s1` | 128 | 0.0414 | 0.5765 | -0.0021 | 0.8671 |
| `occ_nd_s128_s2` | 128 | 0.0453 | 0.6311 | -0.0023 | 0.8740 |
| `occ_nd_s1000` | 1000 | 0.1927 | 2.9071 | +0.0105 | 0.9543 |
| `occ_nd_s1000_s1` | 1000 | 0.4852 | 4.0304 | -0.0894 | 0.9309 |
| `occ_nd_s1000_s2` | 1000 | 0.2922 | 3.0093 | -0.0235 | 0.9411 |

Propagated into the Bregman regression target `log y`, at bridge time t = 0.75
(4000 states, `ctrl_sd` = sd of the control the net must fit):

| ckpt | d log y RMSE | p99 | sign flips | ctrl_sd | RMSE / ctrl_sd |
|---|---|---|---|---|---|
| `occ_nd_s32_seed0` | 0.0936 | 0.1994 | 0.0025 | 0.8505 | 0.110 |
| `occ_nd_s32_seed1` | 0.1046 | 0.2606 | 0.0023 | 0.8089 | 0.129 |
| `occ_nd_s32_s2` | 0.0665 | 0.1519 | 0.0023 | 0.7970 | 0.083 |
| `occ_nd_s128` | 0.1477 | 0.6535 | 0.0060 | 0.7759 | 0.190 |
| `occ_nd_s128_s1` | 0.1110 | 0.2416 | 0.0018 | 0.8045 | 0.138 |
| `occ_nd_s128_s2` | 0.0791 | 0.2841 | 0.0033 | 0.7939 | 0.100 |
| `occ_nd_s1000` | 0.2411 | 0.9098 | 0.0128 | 0.7615 | 0.317 |
| `occ_nd_s1000_s1` | 0.3804 | 1.7113 | 0.0222 | 0.7750 | 0.491 |
| `occ_nd_s1000_s2` | 0.2118 | 0.8833 | 0.0103 | 0.7900 | 0.268 |

At t = 0 the same error is ~0.001 nats.
`json/results_weakness3_labelerr.json`

## 9.4 Step refinement, no retraining

Same controller re-evaluated at finer tau-leap grids.

| run | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| `m32` learned KS_max | 0.0942 | 0.0822 | 0.0805 | 0.0713 |
| `m32` analytic KS_max | 0.1131 | 0.1101 | 0.0979 | 0.0953 |
| `m32` analytic s1 KS_max | 0.1063 | — | 0.0814 | 0.0883 |
| `m128` learned KS_max | 0.0347 | 0.0198 | 0.0108 | 0.0142 |
| `m128` analytic KS_max | 0.0979 | 0.0821 | 0.0746 | 0.0627 |
| `m128` analytic s1 KS_max | 0.1085 | — | 0.0901 | 0.0886 |
| `m1000` learned KS_max | — | 0.3330 | 0.3463 | 0.3472 |
| `m32` learned KS_occ | 0.0163 | 0.0126 | 0.0110 | 0.0105 |
| `m128` learned KS_occ | 0.0123 | 0.0092 | 0.0078 | 0.0064 |
| `m128` analytic KS_occ | 0.0109 | 0.0081 | 0.0065 | 0.0053 |
| `m1000` learned KS_occ | — | 0.0110 | 0.0096 | 0.0093 |

`json/results_weakness3_tauleap.json`

## 9.5 Non-rank-1 residual of the Bregman optimum

Two-way ANOVA on `a*_ji = log(xi_i + Lambda_ji) - log xi_i` over a 96 x 96
(i, j) grid, 48 held-out states, t = 0.95.  Fraction of variance left after
removing `mu + row_i + col_j`.

| ckpt | learned R | analytic R |
|---|---|---|
| `occ_nd_s32_seed0` | 0.1852 | 0.1831 |
| `occ_nd_s128` | 0.4224 | 0.4224 |
| `occ_nd_s128_s1` | 0.4652 | 0.4646 |
| `occ_nd_s1000` | 0.3997 | 0.3999 |

`json/results_weakness3_rank1.json`

## 9.6 Reproduction

```bash
P=/root/miniconda3/envs/cuau_env/bin/python
export PYTHONPATH=/home/RESEARCH/iasbs:/home/RESEARCH/iasbs/iasbs

# analytic-corrector training (m = 32, 128; --steps 256 --iters 1500 at m = 1000)
$P -m iasbs.occupation scale-nondirac --m 32 --N 32 --d 0.5 --tau 1.0 \
  --gamma 4.0 --steps 128 --iters 3000 --inner 4 --batch 512 --hidden 256 \
  --lr 1e-3 --estimator full --corrector analytic --nu-skew 1.0 --seed 0 \
  --tag occ_nd_s32_analytic --out json/results_occ_nd_s32_analytic.json

# the five diagnostics
$P iasbs/analysis/weakness3_corrector.py   # symmetry proof + corrector error
$P iasbs/analysis/weakness3_table.py       # comparison table
$P iasbs/analysis/weakness3_labelerr.py    # label-error propagation
$P iasbs/analysis/weakness3_tauleap.py     # step refinement
$P iasbs/analysis/weakness3_rank1.py       # non-rank-1 ANOVA residual
```

---

# 10. Reproduction index

Everything here runs from the repository root. `$P` is any Python with
`requirements.txt` installed.

## 10.1 Gates first

```bash
$P iasbs/tests_math.py        # 27 mathematical unit tests (imports _a2_tests.py)
$P -m dam.tests_math          # 11 gates for the DAM baseline
$P rasbs/rasbs_sphere_audit.py  # closed-form falsification tests for the R-ASBS port
```

## 10.2 Training entry points

One script per experiment; the exact flags for every published row are in that
row's section, and every `json/results_*.json` also carries its own `config`
block.

| experiment | script | sections |
|---|---|---|
| A, fixed-magnetisation Ising | `iasbs/fixed_ising.py` | §1.1–1.4 |
| B, occupation process | `iasbs/occupation.py` | §2.1–2.4, §9 |
| C, sphere S^2 | `iasbs/sphere.py` | §4.1–4.6 |
| D, Stiefel St(4,2) | `iasbs/stiefel.py` | §5.1–5.7 |
| E, earthquakes | `iasbs/earthquake.py` | §6 |
| A.2, fixed-support + GB1 | `iasbs/fixed_support.py` | §3 |
| baseline 1, DAM | `dam/discrete.py` | §1.5, §2.5, §3.3 |
| baseline 2, R-ASBS Stiefel | `rasbs/rasbs_port.py` | §5.6, §5.7 |
| baseline 2, R-ASBS sphere | `rasbs/rasbs_sphere_port.py` | §4.3, §4.5, §4.6 |

## 10.3 Shell wrappers

| script | produces |
|---|---|
| `iasbs/scripts/rerun_ckpt.sh` | `results_ising_poisson256`, `results_occ_{full,uniform,occupancy,mse}`, `results_occ_s{32,128,1000}` |
| `iasbs/scripts/run_occ.sh` | `results_occ_{full,mse}` |
| `iasbs/scripts/run_scale.sh` | `results_occ_s{32,128,1000}`, `results_occ_var{128,1000}` |
| `iasbs/scripts/rerun_sphere.sh` | `results_sphere_exact`, `results_sphere_train_{plain,anti,sym}` |
| `iasbs/scripts/run_scalefix.sh` | `results_stiefel_{scalefix,fill}` |
| `iasbs/scripts/run_anneal*.sh` | `results_stiefel_anneal_b{50,100,100_fine}`, `results_chain_b{65,80,100}` |
| `iasbs/scripts/run_gb1_anneal.sh` | `results_fs_gb1_k3_{dirac,nd}_{A,B,C}` |
| `iasbs/scripts/run_fair_stiefel.sh` | §5.6 IASBS legs (`frame`, `frame600`, `trace`) |
| `rasbs/run_fair.sh` | §5.6 R-ASBS legs |
| `rasbs/rasbs_steps.sh`, `rasbs_steps_highbeta.sh` | `results_rasbs_steps_*`, `results_rasbs_highbeta_steps_*` |
| `rasbs/run_fidelity_audit.sh` | `results_rasbs_sphere_matlabinit_s{0..4}`, `results_rasbs_audit_{uniform,vmf}_mi` |
| `rasbs/regen_f64.sh` | `results_{rasbs_b2,stiefel_b2}_f64` |
| `dam/run_dam.sh` | the eight §1.5 / §2.5 legs, `results_dam_*` |
| `dam/run_occs128_K64.sh` | `results_dam_occs128_K64_500` |
| `dam/run_dam_a2.sh` | `results_dam_fs_*` |

## 10.4 Read-only diagnostics — `iasbs/analysis/`

None of these train. Each loads `ckpt/` + `json/`, writes one artifact, and is
safe to re-run at any time.

| script | writes `json/` | feeds |
|---|---|---|
| `remeasure.py {ising,ising5,sphere}` | — (prints) | §1, §4 |
| `analysis/stiefel_extra.py` | `results_stiefel_extra` | §5.2 |
| `analysis/rasbs_extra.py` | — (prints) | §4.3 |
| `analysis/bridge_audit.py` | `results_bridge_audit` | §5.8 |
| `analysis/frame_law_audit.py` | `results_frame_law_audit` | §5.7 |
| `analysis/weakness2_sphere.py` | `results_weakness2_sphere` | §4.6 |
| `analysis/weakness2_stiefel.py` | `results_weakness2_stiefel` | §5.6.3 |
| `analysis/weakness3_corrector.py` | `results_weakness3_corrector` | §9.1, §9.3 |
| `analysis/weakness3_table.py` | `results_weakness3_table` | §9.2 |
| `analysis/weakness3_labelerr.py` | `results_weakness3_labelerr` | §9.3 |
| `analysis/weakness3_tauleap.py` | `results_weakness3_tauleap` | §9.4 |
| `analysis/weakness3_rank1.py` | `results_weakness3_rank1` | §9.5 |
| `analysis/weakness5_bridge.py` | `results_weakness5_bridge` | not published |
| `analysis/weakness5_verify.py` | `results_weakness5_verify` | not published |
| `analysis/weakness5_ref.py <refB\|refC>` | — (writes `ckpt/`) | not published |
| `analysis/weakness5_refaudit.py` | `results_weakness5_refaudit` | not published |

The four `weakness5_*` scripts are a fixed-endpoint bridge validation and an
independent-reference audit for the frame target. They run and write artifacts,
but no README section quotes them yet.

`analysis/_paths.py` only puts `iasbs/`, the repository root and `rasbs/` on
`sys.path`; it is imported by the others and is not an entry point.

## 10.5 Multi-seed replication

```bash
bash iasbs/scripts/multiseed.sh list              # row names
bash iasbs/scripts/multiseed.sh <row|group:X|all> [seeds] [drop|keep]
$P iasbs/scripts/aggregate_seeds.py <base> [...]  # collapse to mean +- sd
$P iasbs/scripts/verify_multiseed.py              # re-check the recorded rows
```

`multiseed.sh` re-issues each row's recorded flags with `--seed S --tag
<base>_sS --out json/results_<base>_sS.json`. The third argument defaults to
`drop`, which appends `--eval-every 1000000000` so only the final evaluation
runs. Add `--stage C` to `aggregate_seeds.py` for the GB1 chain. Results and the
request table are in `MULTISEED_RESULTS.md`.

## 10.6 Figures

```bash
$P iasbs/figures.py     # all paper figures + fig/table_stiefel.md
$P iasbs/gallery.py     # sample gallery
```

---


# License

Our code is MIT (see `LICENSE`).

The MIT grant covers the original work only. `rasbs/` holds ports of the R-ASBS
reference implementation, whose upstream repository carries no license file at
the commit we read (`bb71d14`). We claim no ownership of that material; anyone
wanting to reuse `rasbs/` should ask its authors. The upstream clone is not
redistributed here — `.gitignore` excludes `rasbs_ref/`.
