# Adjoint Schrödinger Bridge Sampling on Structured State Spaces via Markov Semigroup Intertwining

Experimental package for the paper. One standalone script per experiment, one
small shared utility file, no framework machinery — the layout deliberately
follows the R-ASBS repository's style.

Our method and the baseline live in separate directories, so that no reader has
to take on trust which code produced which column.

```
LICENSE                   MIT for our code; see its scope note about rasbs/
requirements.txt          pinned versions every number below was produced with
common.py                 shared kernels, quadrature, Stiefel/S^3 helpers, ckpt IO
json/  ckpt/  fig/        artifacts, shared by both methods

structured_asbs/          OUR method
  fixed_ising.py          Exp A  fixed-magnetisation Ising   (bijective discrete)
  occupation.py           Exp B  occupation process      (non-bijective discrete)
  sphere.py               Exp C  S^2                       (scalar Killing readout)
  stiefel.py              Exp D  St(4,2)                   (matrix Killing readout)
  earthquake.py           Exp E  S^2, real data            (vMF mixture, 3343 modes)
  tests_math.py           standalone mathematical unit tests
  figures.py              all paper figures + the comparison table
  gallery.py              sample gallery -- the samples, not their metrics
  remeasure.py            re-derives the Ising and sphere tables from ckpt/
  scripts/                the shell scripts that produced our json/ entries

rasbs/                    THE BASELINE, kept apart from our code
  rasbs_port.py           faithful PyTorch port of R-ASBS alg2_stiefel.m
  rasbs_sphere_port.py    port of asbs_sphere_sampler.m + earthquake_sphere_ex.m
  rasbs_sphere_audit.py   closed-form falsification tests for that port
  rasbs_steps.sh          their beta = 2 step ablation
  rasbs_steps_highbeta.sh their beta = 50/100 step ablation
  regen_f64.sh            float64 rerun at beta = 2
```

Code is split by *whose method it is*; artifacts are not. Most figures plot
ours and theirs on the same axes, and the file names (`results_rasbs_*` vs the
rest) already say which is which, so two artifact trees would buy nothing and
cost a second search path in every loader. `json/`, `ckpt/` and `fig/`
therefore sit at the root next to `common.py`, above both method directories
rather than inside either.

Every entry point calls `common.use_repo_root()` before parsing arguments, so
the bare relative paths (`json/results_*.json`, `ckpt/`, `fig/`) resolve to
these root directories no matter which directory the script was launched from.

Five benchmarks, in order of increasing structure: two discrete state spaces
where the answer can be enumerated exactly, then two manifolds where it cannot,
then one real dataset on S² whose target happens to be exactly solvable anyway.
Each section below carries its own configuration, results, figures, limitations
and reproduction commands. **The earthquake experiment fails two of its own
gates**; it is reported in full rather than dropped. Every number is what the
committed
`json/results_*.json` files were produced with, and each of those files also
carries its own full `config` block, so any number here traces back to the
flags that made it.

Run on a single NVIDIA A100 80GB (two available, used only to run independent
experiments concurrently), PyTorch 2.5.1, CUDA 12.4, NumPy 2.4.6, conda
environment `SML_env` (Python 3.11). Manifold state and all metrics are
`float64`; the networks are `float32`. `requirements.txt` pins those exact
versions; nearby ones almost certainly work, the pins are there so a reader who
gets a different number knows the environment is not the reason.

The `bash …` command blocks below call the interpreter through `PY`, which
defaults to whatever `python` is first on `PATH`. Point it somewhere else with
`PY=/path/to/python bash rasbs/rasbs_steps.sh`.

**The source is a Dirac in every experiment.** We start from a *fixed point*
`x_0` and learn a control that transports the reference process to the target,
following Adjoint Sampling; we never sample the source from a guessed
distribution. This is the structural point of the method, and it is what the
R-ASBS comparison in the Stiefel section turns on.

**Provenance.** Every R-ASBS number below is either marked *quoted* with its
section in the paper, or carries the `json/` file our rerun produced. Only two
are quoted: their paper's §4.1–4.3 report figures only, and its single results
table (Table 2) is for the Wahba problem, which we do not attempt, so the rest
had to be regenerated. `rasbs/rasbs_port.py` is pinned to upstream commit
`bb71d14` (2026-08-28). The Ising and Occupation sections carry no R-ASBS
column: no counterpart experiment exists in their paper or their repository.

Checkpoints under `ckpt/` are not committed (2.5 GB); the `json/results_*.json`
files are, and almost every table here is built from them. The exception is the
extra columns `remeasure.py` derives — KL, Hellinger, χ², ESS fraction, the
max density ratio, ⟨sᵢsⱼ⟩ and the sphere's KS(E) — which are recomputed from
the checkpoints rather than stored in any JSON. Those numbers need `ckpt/`,
which means retraining, and the commands that produce it are in each section.

**One input is not committed either.** `.gitignore` excludes `rasbs_ref/`, the
frozen R-ASBS clone kept locally for reference, and the earthquake target is
built from `rasbs_ref/query.csv` inside it. Experiment E and the R-ASBS
earthquake column are therefore not runnable from a fresh clone until that
clone is restored; every other experiment is self-contained.

All commands below are run from the repository root.

```bash
python structured_asbs/tests_math.py   # mathematical unit tests
python structured_asbs/figures.py      # all figures -> fig/
python structured_asbs/gallery.py      # sample gallery -> fig/
```

`figures.py` plots metrics. `gallery.py` plots the samples themselves, as the
objects they actually are, beside the exact or MCMC reference drawn the same
way. The claim those panels support is "you cannot tell the two columns
apart", and that is a claim the eye should adjudicate. `figures.py` also writes
`fig/table_stiefel.md`, the Stiefel comparison table in markdown. That generated
table fills each (β, steps) cell with the smaller |ΔE| of the two β = 100
controls, so at 796 and 1592 steps it quotes the direct warm start while the
README quotes the chain throughout; the two agree at 3184 steps, which is the
cell either of them is judged on.

Every experiment script takes a subcommand. `sphere.py` and `earthquake.py`
expose `verify` (correctness gates), `exact` and `train`; `occupation.py` adds
`var`, `scale` and `scalevar` to those three;
`stiefel.py` exposes `verify`, `ref`, `train` and `sweep`; `fixed_ising.py`
exposes only `exact` and `train`, because its gate A0 is checked inside the
`exact` control rather than as a separate mode.

---

## Ising lattice

**Experiment A — bijective discrete.** A 4×4 periodic Ising lattice with the
magnetisation held fixed at zero, so the state space is the constraint set
Ω = {s ∈ {±1}¹⁶ : Σᵢ sᵢ = 0}, of size **C(16, 8) = 12,870**. The target is the
Gibbs law of `E = −J Σ_⟨ij⟩ sᵢsⱼ` at `J = 1`, `τ = 2`. The controlled process
moves by spin *swaps*, which is what makes the constraint structural rather
than enforced: every move is a bijection of Ω onto itself, so magnetisation
cannot drift even in principle.

The point of running it at 4×4 is that everything is computable exactly by
enumeration. We can (a) run the exact optimal control, (b) propagate the law of
the *discretised* controlled chain exactly rather than estimating it from
samples, and (c) compare the learned multiplier against the exact one state by
state. A TV distance against truth therefore means something here, which it
would not at a size where the reference is itself a sampler.

| | configuration |
|---|---|
| script | `fixed_ising.py` |
| state space | 4×4 periodic lattice, 16 spins, magnetisation fixed |
| target | `E = −J Σ_⟨ij⟩ s_i s_j`, `J = 1`, `τ = 2` |
| source `x_0` | fixed spin configuration (Dirac) |
| reference | exact enumeration |
| network | `SwapController`, hidden 512, **670,464** parameters |
| integration steps | 256 |
| iters × inner | 3000 × 40 |
| batch / minibatch | 2048 / 1024 |
| lr | 3e-4 |
| buffer | 8 |
| loss | Poisson |
| eval samples | 20,000 |
| seeds | 1 |

### Results

Because Ω can be enumerated, the law of the discretised controlled chain is
propagated exactly rather than estimated from samples, so the divergences below
are closed-form, with no Monte-Carlo error. Source `ckpt/ising_poisson256.pt`
and `json/results_ising_exact.json`; `remeasure.py ising` regenerates all of it,
and can emit further metrics on request.

| metric | exact | ours |
|---|---:|---:|
| TV of the chain law vs π | 0 | **0.0416** |
| KL(ours ‖ π) | 0 | **0.0056** |
| Hellinger | 0 | **0.0372** |
| importance-reweighting ESS fraction | 1 | **0.9885** |
| max_x \|p(x)/π(x) − 1\| | 0 | **224.27** |
| mean energy ⟨E⟩ | −9.2457 | **−8.9850** |
| heat capacity Var(E)/τ² | 5.8843 | **6.0237** |
| constraint violations in 20,000 samples | 0 | **0** |

Reweighting our samples onto the exact Gibbs law would cost about 1% of them.
The max-ratio of 224 is the dissenting number and is printed for that reason:
it is dominated by states where π itself is ~1e-9, so the mass-weighted
divergences say the bulk is close while the ratio says the tails are not. A
sampler used for rare-event estimation would have to be judged on that column.

The **exact control** — same algorithm, multiplier from enumeration instead of
a network — isolates pure discretisation error: TV vs π is 0.13610, 0.06823,
0.03385, 0.01681, 0.00837, 0.00417 at 32…1024 steps. It halves per doubling,
first-order with no bias floor, the discrete analogue of the Stiefel step
ablation below. Mass leak stays at 1e-14: the chain never leaves Ω even
numerically. At 1024 steps and 200,000 samples its empirical TV is 0.0517
against an iid floor of 0.0511.

At matched budget the Poisson loss beats MSE but only narrowly (TV 0.0496 vs
0.0509 at 128 steps); doubling to 256 steps and 3000 iters gives 0.0416. The
learned log-multiplier, compared at all 12,870 states × 32 edges, has mean
absolute error 0.060 at t = 0, 0.028–0.048 in the middle and 0.170 at t → 1 —
the terminal end, where the sharp state dependence encoding the Gibbs weight
lives, is where the residual TV comes from.

`fig8_ising_configs` draws 50 raw 4×4 spin configurations from our sampler
beside 50 drawn by exact enumeration; `fig2_discrete_correctness` panel A
carries the metric panel, specifically the **exact control's** TV against step
count with the iid floor marked, not the learned run's single number. **Both
figures are 4×4 only** — neither `figures.py` nor `gallery.py` has a 5×5 code
path, so the 5×5 subsection below is tables only.

### Scaling the exact check: the 5×5 lattice

The 4×4 lattice is small enough that "exact" costs nothing. The point of
redoing it at 5×5 is that the constraint set grows from 12,870 states to
**C(25, 12) = 5,200,300** — a factor of 404 — while everything above stays
computable: the chain's law is still propagated by enumeration, so the TV
column is still exact. Same target, `τ = 2`, magnetisation fixed at zero on a
25-spin odd lattice (so k = 12 rather than 12.5), 50 edges, same network
(hidden 512, 864,369 parameters), Poisson loss, `--iters 6000 --inner 40`.
Source `ckpt/ising_t1_L5_s512.pt` and `json/results_ising_t1_L5*.json`;
`remeasure.py ising5` regenerates the whole subsection.

| metric | 4×4 (12,870) | 5×5 (5,200,300) |
|---|---:|---:|
| TV of the chain law vs π | 0.0416 | **0.0770** |
| KL(ours ‖ π) | 0.0056 | **0.0193** |
| Hellinger | 0.0372 | **0.0692** |
| ESS fraction | 0.9885 | **0.9617** |
| max_x \|p(x)/π(x) − 1\| | 224.3 | **62,240** |
| TV of the energy histogram | 0.0262 | **0.0347** |
| mean energy ⟨E⟩ (exact) | −8.9850 (−9.2457) | **−17.1348** (−17.6038) |
| heat capacity (exact) | 6.0237 (5.8843) | **7.3246** (7.1195) |
| ⟨sᵢsⱼ⟩ (exact) | 0.2808 (0.2889) | **0.3427** (0.3521) |
| constraint violations / 200,000 | 0 | **0** |

The constraint still holds structurally — 404× more states, still zero
violations — and the bulk agreement degrades only about twofold. The tail does
not: the max ratio goes up 280×, because π's smallest atom falls by roughly the
same factor and the controller has no incentive to resolve it.

**The interesting result is that more integration steps stop helping.** Running
the exact control (multiplier from enumeration) gives TV 0.09849, 0.04823,
0.02377, 0.01179 at 64/128/256/512 steps — still exactly first-order, still no
bias floor, mass leak ≤ 5e-15. But the learned controller does not track it:

| steps | iters | exact-control TV (the floor) | learned TV | ratio |
|---:|---:|---:|---:|---:|
| 256 | 3000 | 0.02377 | 0.0923 | 3.9× |
| 512 | 6000 | 0.01179 | 0.0770 | **6.5×** |

Doubling the grid halved the floor and moved the achieved TV by 17%. At 4×4 the
same comparison is 0.0416 against a floor of 0.01681, a ratio of 2.5×, and
refining the grid there *did* help. So the two lattices are limited by different
things: at 4×4 the residual is mostly the integrator, at 5×5 it is mostly the
controller. Adding steps at 5×5 is the wrong knob; capacity, iterations or a
better terminal parametrisation is the right one.

The multiplier error says where: mean absolute error over all 5,200,300 states ×
50 edges is 0.088 at t = 0, 0.053–0.089 in the middle, and **0.310 (max 5.59)**
at t → 1, against 0.170 (max 1.92) for the same quantity at 4×4. The terminal
end is where the Gibbs weight's state dependence is sharpest, it is where the
error concentrated at 4×4, and enlarging the state space made it worse rather
than merely larger.

One methodological note. The 200,000 stored samples give an empirical TV of
0.3546 against an iid floor of 0.3231 ± 0.0008 — both numbers are dominated by
having 26 states per sample, not by sampler quality, and the exact control
scores 0.3298 on the same measure despite a true TV of 0.0118. That is exactly
why this experiment propagates the law instead of estimating it: at this size a
sample-based TV cannot distinguish a good sampler from a perfect one.

We report the run as a gate **failure**: the A1 threshold of TV < 0.05 was
inherited unchanged from 4×4 and the 5×5 run missed it at 0.0770. The gate was
not re-tuned for a 404× larger state space, so this is a threshold that does not
transfer, not a run that misbehaved. A2 (zero violations) passed.

**Limitations.**

- **Two lattice sizes, both small.** 4×4 and 5×5 are the sizes at which the
  constrained distribution can be enumerated exactly, which is the entire
  reason for the experiment — but it does mean we have no evidence beyond
  5.2 million states.
- **The 5×5 controller is not converged.** The step ablation above shows the
  error is learning-limited there, and we did not run the larger-capacity or
  longer-schedule variants that would say how much of the 0.0770 is removable.
- **No R-ASBS counterpart exists**, in either direction: their method is
  formulated for embedded Riemannian manifolds and does not apply to discrete
  state spaces. The comparison here is against exact enumeration instead.

```bash
python structured_asbs/fixed_ising.py exact          # gate A0/A1/A2 with the exact control
python structured_asbs/fixed_ising.py train --steps 256 --iters 3000 \
    --tag ising_poisson256 --out json/results_ising_poisson256.json
python structured_asbs/remeasure.py ising            # every table in this section, from ckpt/
# 5x5: 5,200,300 states, still enumerated exactly
python structured_asbs/fixed_ising.py exact --L 5 --steps-sweep 64 128 256 512 \
    --tag ising_t1_L5_exact --out json/results_ising_t1_L5_exact.json
python structured_asbs/fixed_ising.py train --L 5 --steps 512 --iters 6000 \
    --hidden 512 --tag ising_t1_L5_s512 --out json/results_ising_t1_L5_s512.json
python structured_asbs/remeasure.py ising5
```

---

## Occupation

**Experiment B — non-bijective discrete, and the scale test.** m indistinguishable
particles distributed over N sites; a state is the occupancy vector η ∈ ℕᴺ with
Σᵢ ηᵢ = m, and the constraint set has size C(m+N−1, N−1). The target is
`π(η) ∝ Πᵢ Γ(ηᵢ+d) / (ηᵢ! Γ(d))` at `d = 0.5`, `τ = 1`. Unlike the Ising swap
chain the moves here are **not** bijective — particles are indistinguishable,
so many microscopic moves collapse onto the same occupancy transition, and the
intertwining has to carry that degeneracy.

This is the one benchmark in the repository that is actually scaled. The other
three are small by construction: the Ising lattice is 4×4 so that it can be
enumerated, S² is 2-dimensional and St(4,2) is 5-dimensional. Here the same
network and the same code run at every size and only the state space grows,
from 10¹⁸ to **10⁶⁰⁰**.

| | configuration |
|---|---|
| script | `occupation.py` |
| state space | m particles on N sites, C(m+N−1, N−1) states |
| target | `π(η) ∝ Π_i Γ(η_i+d)/(η_i! Γ(d))`, `d = 0.5`, `τ = 1` |
| source `x_0` | fixed occupancy (Dirac) |
| reference | exact (matrix exponential) |
| network | `OccController`, hidden 256, **139,536** parameters (136,450 in the scale sweep) |
| integration steps | 128 (256 at m = 1000) |
| iters × inner | 1500 × 4 (3000 at m = 32, 128) |
| batch / minibatch | 512 / 1024 (128 at m = 1000) |
| lr | 1e-3 |
| buffer | 8 |
| eval samples | 20,000 |
| seeds | 1 |

### Results at m = N = 4, where the answer can be enumerated

| metric | exact | ours |
|---|---:|---:|
| TV vs exact law, learned control | 0 | **0.0127** (`results_occ_full.json`) |
| TV vs exact law, exact control | 0 | 0.0199 (`results_occ_exact.json`) |
| iid floor, 20,000 samples | — | 0.0167 |
| constraint violations | 0 | **0** |

As with the Ising lattice, the TV value sits essentially at the iid sampling
floor, so the residual is finite-sample error rather than sampler bias. The
exact control's 0.0199 is the empirical TV of the same integrator with the
multiplier taken from enumeration; its step sweep is first-order and clean —
0.2965, 0.1410, 0.0379, 0.0189, 0.0094, 0.0047 at 8…256 steps, halving per
doubling from 32 steps on (the 16 → 32 step is a 3.7× drop, the coarse end
still leaving the asymptotic regime) — which is what `fig2` panel B plots.

**Estimator and loss ablation, m = N = 4.** The terminal expectation can be
estimated three ways, selected by `--estimator`, and the loss by `--loss`:

| variant | flags | TV vs exact law |
|---|---|---:|
| full enumeration over moves | `--estimator full` | **0.0127** |
| uniform move sampling | `--estimator uniform` | 0.0162 |
| occupancy-weighted sampling | `--estimator occupancy` | 0.0164 |
| Bregman → MSE loss | `--estimator full --loss mse` | 0.0131 |

All four land within 0.004 TV of each other and all four have zero constraint
violations, so neither choice is load-bearing at this size; `full` is the
default because it is exact and affordable when the move set is small.

**Gate B3 — estimator variance.** `occupation.py var` measures the variance of
the terminal estimator at t = 0, 0.25, 0.5, 0.75, 0.9 and passes
(`json/results_occ_var.json`, `B3: true`). At t = 0 occupancy weighting is exact
(variance 3.5e−13 against uniform's 1.059); in the interior four-sample
occupancy weighting cuts the variance about 4× against one sample
(0.418 vs 1.672 at t = 0.25) and 4.6–4.8× against uniform. The same
measurement at m = N = 128 and 1000 is in `results_occ_var128.json` and
`results_occ_var1000.json`, and also passes.

### Scaling to 10⁶⁰⁰ states

Same network (136,450 parameters) and the same code at every size; only the
state space grows. `python structured_asbs/occupation.py scale --m M --N M`:

| m = N | constraint-set size | source KS(occ) | ours KS(occ) | ours KS(max) | W₁(max)/m | E ours | E exact | violations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | C(63, 31) ≈ 9.2e17 | 0.2054 | **0.0043** | 0.0305 | 0.0052 | 13.317 | 13.212 | **0** |
| 128 | C(255, 127) ≈ 5.8e75 | 0.2054 | **0.0084** | 0.0796 | 0.0044 | 52.871 | 52.108 | **0** |
| 1000 | C(1999, 999) ≈ 10⁶⁰⁰ | 0.2035 | **0.0102** | 0.2215 | 0.0014 | 412.32 | 405.57 | **0** |

The "source" column is the uncontrolled reference process measured the same
way, so the KS numbers are a 20× to 48× reduction rather than a small number
quoted alone. The occupancy KS stays near 0.01 across four orders of magnitude
of state-space size, the constraint `Σᵢ ηᵢ = m` is satisfied exactly at every
size, and W₁ on the hardest statistic (the maximum occupancy) *improves* in
relative terms as m grows, from 0.52% to 0.14% of m.

`fig3_occupation_scale` plots this sweep; `fig9_occupation_raster` draws raw
occupancy vectors at m = N = 1000 with their per-box marginals.

**Limitations.**

- **`KS(max)` degrades with scale**: 0.031 → 0.080 → 0.222. That is the
  extreme-value statistic — the law of the single largest occupancy — and it is
  the one we do worst on at m = 1000. It is reported rather than left out.
- **The three rows are not budget-matched.** m = 32 and 128 use 3000 iterations,
  128 steps, batch 512; m = 1000 uses 1500 iterations, 256 steps, batch 128. So
  the row-to-row comparison should not be read as a clean scaling law.
- **No R-ASBS counterpart**, for the same reason as the Ising lattice.

```bash
python structured_asbs/occupation.py verify                  # gate B0
python structured_asbs/occupation.py exact                   # exact control, step sweep
python structured_asbs/occupation.py train --tag occ4_full --out json/results_occ_full.json
python structured_asbs/occupation.py var                     # gate B3, estimator variance
python structured_asbs/occupation.py scale --m 1000 --N 1000
python structured_asbs/occupation.py scalevar --m 1000 --N 1000 \
    --out json/results_occ_var1000.json
# the estimator / loss ablation
python structured_asbs/occupation.py train --estimator uniform \
    --tag occ4_uniform --out json/results_occ_uniform.json
python structured_asbs/occupation.py train --estimator occupancy \
    --tag occ4_occupancy --out json/results_occ_occupancy.json
python structured_asbs/occupation.py train --loss mse \
    --tag occ4_mse --out json/results_occ_mse.json
```

---

## Sphere

**Experiment C — S², scalar Killing readout.** The bimodal target
`π ∝ exp(6 x₃²)` on the unit sphere, i.e. `E(x) = 6(1 − x₃²)` at `τ = 1`, which
is the same energy and the same scale as R-ASBS's `asbs_sphere_sampler.m`. Mass
concentrates on the two poles, and the diagnostic that matters is whether both
poles get half of it: a sampler that collapses onto one pole still gets the
radial statistics right, so the north mass and the KS distance of the `x₃`
marginal are what separate a correct sampler from a plausible-looking one.

The reference is exact — the density is ∝ exp(6x₃²) with uniform azimuth, so
inverse-CDF sampling draws genuinely iid target points.

| | configuration |
|---|---|
| script | `sphere.py` |
| state space | unit sphere in R³, continuous, dim 2 |
| target | `E(x) = 6(1 − x₃²)`, `π ∝ exp(6x₃²)`, `τ = 1` |
| source `x_0` | `(1, 0, 0)` (Dirac) |
| reference | exact inverse-CDF |
| network | `ScoreNet`, hidden 256, **136,963** parameters |
| integration steps | 128 |
| iters × inner | 4000 × 16 |
| batch / minibatch | 8192 / 16384 |
| lr | 1e-3 |
| EMA / buffer | 0.9995 / 4 |
| eval samples | 200,000 |
| seeds | 5 |

### Results

There are now two R-ASBS columns and **they disagree with each other.** The
first is quoted from their §4.1 text — "R–ASBS allocates 43.8% of its particles
to the northern hemisphere". The second is measured, by running
`rasbs/rasbs_sphere_port.py`, a port of their `asbs_sphere_sampler.m`, over five
seeds at their own settings (600 epochs, 500 steps, σ = 1, B = 500, the 64/64
and 48/48 tanh networks, Adam at 2e-3). The port collapses onto **one** pole,
every time. This is reported before our own results because how much weight the
rest of the section carries depends on how that disagreement is resolved, and we
have not resolved it.

Five seeds of the antithetic variant, 200,000 samples each, recomputed from the
checkpoints by `remeasure.py sphere`, which can emit further metrics on
request. The exact column is closed form: the z-marginal of π ∝ exp(6x₃²) is
∝ exp(6z²) on [−1, 1] with uniform azimuth. The iid column is the finite-sample
floor at the same n, so anything at that level is sample noise.

| metric | exact | R-ASBS (quoted, §4.1) | R-ASBS (our port, 5 seeds) | ours | iid floor |
|---|---:|---:|---:|---:|---:|
| north mass | 0.5 | 0.438 (= 43.8%) | 0.0000, 0.0000, 0.9976, 0.0017, 0.0019 | **0.5001 ± 0.0006** | 0.5002 ± 0.0008 |
| absolute north error | 0 | 0.062 | 0.4988 ± 0.0011 | **0.0006 ± 0.0003** | 0.0006 ± 0.0005 |
| KS(x₃) | 0 | not reported | 0.50486 ± 0.00110 | **0.0222 ± 0.0004** | 0.0018 ± 0.0004 |
| Wasserstein-1 on x₃ | 0 | not reported | 0.90224 ± 0.00234 | **0.01239 ± 0.00022** | 0.00118 ± 0.00073 |
| KS(azimuth) | 0 | not reported | 0.0163 ± 0.0108 | **0.0026 ± 0.0010** | 0.0018 ± 0.0005 |
| ⟨x₃²⟩ | 0.80771 | not reported | 0.82666 ± 0.00271 | **0.78864 ± 0.00019** | 0.80778 ± 0.00033 |
| mean energy ⟨E⟩ | 1.15375 | not reported | 1.04007 ± 0.01626 | **1.26816 ± 0.00112** | 1.15333 ± 0.00197 |
| max norm residual | 0 | not reported | 3.3e-16 | **2.2e-16** | — |

The per-seed north masses are printed rather than averaged because their mean,
0.200 ± 0.446, describes no run that happened. Four seeds emptied the northern
hemisphere and one emptied the southern; the ± is the coin flip, not a spread.
KS(x₃) ≈ 0.505 in every run is precisely the missing half of the mass.

**What the collapsed runs get right is the point.** ⟨x₃²⟩ = 0.8267 against an
exact 0.8077, and ⟨E⟩ = 1.040 against 1.154 — the radial profile is roughly
correct, and on ⟨E⟩ the collapsed sampler is *closer to exact than ours is*
(0.114 low versus our 0.114 high). A report consisting of ⟨E⟩ and a picture of
one pole would look like a success. Only the hemisphere mass and KS(x₃) show
that half the distribution is missing, which is why this section gates on them.

Collapse is fast and it is not undertraining: at 100 epochs north mass is
already 0.0004, at 300 epochs 0.0002. Longer training makes the *profile* worse,
not better — at seed 0, ⟨E⟩ goes 1.267 (100 epochs) → 1.172 (300) → 1.046
(600), passing through the exact 1.154 on its way past it. (The 1.040 in the
table above is the mean over five seeds at 600 epochs; these three are one seed,
so that they differ only in epoch count.) The sampler spends its training
budget over-sharpening the single pole it kept.

**The symmetry is exact** — north mass and azimuthal uniformity are both at the
iid floor. **The radial profile is not** — KS(x₃) is 12× the floor, ⟨x₃²⟩ low
by 0.019 and ⟨E⟩ high by 0.114, all one direction: slightly too little mass at
the poles. That is discretisation, not learning. The exact control (no network
at all) gives KS(x₃) = 0.0655, 0.0353, 0.0184, 0.0102, 0.0063 at 32…512 steps
and ⟨E⟩ falling monotonically 1.543 → 1.177 toward 1.15375 — halving per
doubling, **no bias floor**, the opposite of what the R-ASBS geometric
surrogate does on Stiefel below. Our runs use 128 steps and land at 0.0222
against the exact control's 0.0184 there, so nearly all the residual is the
integrator rather than the network. Reporting only north mass would have hidden
this.

The norm residual is 2.2e-16 because the update is an exact geodesic step, not
a projection; it is quoted from the run-time metrics, since the checkpointed
samples are float32 and cannot resolve a float64 residual.

**Symmetry handling.** The target is invariant under x₃ ↦ −x₃. The plain
variant is not worse on average so much as *unreliable*: per-seed north masses
0.460, 0.474, 0.495, 0.528, 0.528, a spread of ± 0.0275 against ± 0.0006 for
antithetic and symmetrised, which are indistinguishable from each other and fix
it completely. Neither improves ⟨E⟩ — the pole imbalance and the radial bias
are independent defects. `fig4_sphere` plots the comparison. All three variants
are 4000 iterations; `json/results_sphere_train_plain2500.json` is the same
plain variant stopped at 2500 (north error 0.0599, KS(x₃) 0.0667), kept only to
show that the plain run is still improving when the others have converged.

**Figures.** `fig6_sphere_cloud` shows four S² clouds: the uncontrolled source,
ours, exact iid target samples, and the residual between the last two.
`fig10_sphere_rasbs_style` renders our samples in R-ASBS's own visual
convention — their plasma colormap, `view(40, 25)`, the FaceAlpha 0.45 shaded
globe, 12 black contour lines, 1500 black sample dots, colour limits [0, 6] and
their "Bi-Modal Distribution" title — so the two figures can be placed side by
side without re-styling either.

One thing the gallery had to get right, because getting it wrong would have
flattered us: **the S² reference is not the `sphere_reference` checkpoint.**
That checkpoint holds the *uncontrolled* process — the bridge's source,
KS(x₃) = 0.324 — not the target. Comparing against it measures the transport,
not the error. Figure 6 draws exact iid target samples by inverse-CDF instead,
and against that reference 99% of the 1800 equal-area bins fall within ±3σ with
max |z| = 4.3: at 200,000 samples a side there is no resolvable structure left
in the residual.

**Limitations.**

- **S² is 2-dimensional.** This is a correctness benchmark, not a hard one.
- **Our port collapses and their paper says it does not, and we did not settle
  which is right.** This is the largest open item in the repository. What we
  checked, and what we did not:

  *Checked.* The port was read line by line against `asbs_sphere_sampler.m`:
  Haar `X0`, `t = (step−1)dt`, `X + u dt + σ√dt · P_x(noise)` then normalise,
  `grad_ambient_E = [0, 0, −12x₃]` projected, the geodesic bridge
  `μ_t = x₀cos(tθ) + u sin(tθ)` with `std = σ√(t(1−t))`, the parallel transport
  `a_⊥ + a_∥cos φ − ⟨a,u⟩x₁ sin φ`, the corrector `logmap(x₁→x₀)/σ²`, the losses
  `mean|P(u) + σa|²` and `mean|P(h) − b|²`, and every hyperparameter. It also
  passes five internal checks (`--check`): Haar uniformity, transport isometry
  and tangency, ∇E against central differences, `V(1)` against quadrature, and
  the cotangent series at its switch point. Two seeds collapse to *opposite*
  poles, which is the signature of a bistable solution rather than a sign error.

  *Not checked.* Their binary. MATLAB is not installed here and their script
  needs the Deep Learning Toolbox, so there is no cross-check available. Three
  hypotheses remain live and untested: (a) weight initialisation — MATLAB's
  `fullyConnectedLayer` uses Glorot weights with **zero bias** while PyTorch's
  `nn.Linear` gives a nonzero bias, i.e. a preferred ambient direction from
  epoch 1, which is exactly the seed a winner-take-all loop needs; (b) a seed
  lottery, if the collapse probability is high but below 1 and their single
  `rng(1,'twister')` run landed in the balanced basin; (c) drift between the
  released script and the number in the paper. `rasbs/rasbs_sphere_audit.py`
  was written to attack a fourth — a symmetry-breaking defect in the port — by
  driving the same code path with `E = 0` (target is Haar) and `E = −6x₃`
  (single mode, `north = (e⁶−1)/(e⁶−e⁻⁶)`, `⟨x₃⟩ = coth 6 − 1/6`), but **it was
  not run to completion**, so it is a designed test, not evidence.

  Until one of these is settled, the quoted 43.8% and the measured collapse are
  both in the table and neither is presented as *the* R-ASBS result. The Stiefel
  section, where their algorithm is rerun and the disagreement does not arise,
  is the stronger comparison.

```bash
python structured_asbs/sphere.py verify              # gate C0
python structured_asbs/sphere.py train --antithetic --iters 4000 --inner 16 \
    --batch 8192 --mb 16384 --ema 0.9995 --seeds 5 --n-samples 200000 \
    --tag sphere_anti --out json/results_sphere_train_anti.json
python structured_asbs/remeasure.py sphere           # every table in this section, from ckpt/
python rasbs/rasbs_sphere_port.py --check            # port self-checks
python rasbs/rasbs_sphere_port.py --problem bimodal --seed 0   # the collapse
python rasbs/rasbs_sphere_audit.py --test vmf        # designed, never run to completion
```

---

## Earthquakes

**Experiment E — S², real data, and the one experiment we do not pass.** The
target is the epicentre distribution of the 4,776 magnitude ≥ 6.0 earthquakes in
`rasbs_ref/query.csv`, the USGS query R-ASBS ship with their repository and use
for their §4.2 figure. Each epicentre `mᵢ` becomes a von Mises–Fisher mode:

```
π(x) ∝ (1/N) Σᵢ exp(κ ⟨mᵢ, x⟩)
```

at `κ = 600`, so each mode has angular width ≈ κ^(−1/2) ≈ 2.4°. We fit on a
random 70% (3,343 modes) and hold out the remaining 30% (1,433) to check that
the sampler is reproducing a distribution rather than memorising a point set.

The reason this target is worth the trouble is that it is **exactly solvable
despite being real data.** Because `∫_{S²} exp(κ⟨m, x⟩) dx = 4π sinh(κ)/κ` does
not depend on `m`, the normalising constant of every mode is the same, so π is
*exactly* an equal-weight vMF(κ) mixture. Two things follow that a generic
multimodal benchmark does not give you:

- **iid reference samples are available in closed form** — pick a mode
  uniformly, then draw from vMF(κ) by inverse CDF, `w = 1 + log(u + (1−u)
  e^(−2κ))/κ`. So every metric below has a measured finite-sample floor, not a
  guessed one.
- **log Z is closed form**: `log(4π sinh κ / κ) = 595.4409474111932` at κ = 600.

R-ASBS use the same dataset but report only a plot, so there is no number to
quote. The R-ASBS column below is therefore **measured, not quoted**:
`rasbs/rasbs_sphere_port.py` ports their `earthquake_sphere_ex.m` — same κ
staircase, same 250 steps, same drift clip of 30, same random-Fourier network,
their uniform Haar source and their tangent-space Gaussian bridge both left
unrepaired — and is then scored by *our* `earthquake.py` metrics, so both
columns come off the same ruler.

| | configuration |
|---|---|
| script | `earthquake.py` |
| target | equal-weight vMF mixture, `κ = 600`, 3,343 modes (70% split) |
| data | `rasbs_ref/query.csv`, 4,776 quakes, M ≥ 6.0 |
| source `x_0` | fixed point on S² (Dirac) |
| network | multi-scale random-Fourier score net, hidden 512, **931,331** parameters |
| κ schedule | annealed 150 → 300 → 450 → 600, one warm-started net + EMA |
| integration steps | 512 |
| iters × inner | 6000 × 16 |
| batch / minibatch | 2048 / 4096 |
| lr / EMA | 1e-3 / 0.999 |
| drift clip | 200 |
| eval samples | 100,000 |

### Results

| metric | ours | R-ASBS (measured) | iid floor | better |
|---|---:|---:|---:|:--:|
| ΔE (ref −594.7238 / −594.7293) | **−0.0771** | +0.8466 | −0.0069 / +0.0052 | ours, 11× |
| KS(E) — **gate E1** | **0.10048** ✗ | 0.22614 ✗ | 0.00327 / 0.00403 | ours, 2.3× |
| mode-histogram TV — **gate E2** | **0.50884** ✗ | **0.29385** ✗ | 0.08910 / 0.08982 | R-ASBS, 1.7× |
| KS(angle to nearest mode) | **0.08048** | 0.13445 | 0.00744 / 0.00383 | ours, 1.7× |
| mode coverage (ref 0.9737 / 0.9722) | **0.7679** | **0.9432** | 0.9707 / 0.9749 | R-ASBS |
| energy distance | **0.30171** | **0.10079** | 0.00031 / 0.00120 | R-ASBS, 3.0× |
| \|‖x‖−1\| — **gate E3** | **2.2e−16** ✓ | 3.3e−16 ✓ | 2.2e−16 | tie |

Two iid floors are listed because each column was scored against its own
independently drawn 100,000-sample reference; the gap between them (e.g. 0.00327
vs 0.00403 on KS(E)) is the sampling noise on the floor itself.

**E1 and E2 fail for us.** The mean energy is right to 1.3e−4 relative and the
constraint is exact, but the *law* is not: the sampler reaches only 77% of the
3,343 modes against the reference's 97%, and its mode histogram is half a
TV unit away from uniform-over-modes.

**And this is the one benchmark where R-ASBS beats us**, on exactly the axis we
fail: mode TV 0.294 against our 0.509, coverage 0.943 against our 0.768, energy
distance 0.101 against our 0.302. We are better everywhere the *shape* of the
distribution around a mode is measured — ΔE by 11×, KS(E) by 2.3×, KS(θ) by
1.7× — and worse everywhere the *allocation of mass across modes* is measured.
Neither sampler passes E1 or E2.

That split is not a coincidence, and it is the same mechanism that costs them
the sphere and Stiefel benchmarks. **Their uniform Haar source is a liability
when the target is concentrated and an asset when it is spread out.** With 3,343
modes scattered over the whole sphere, initialising every particle uniformly
means every mode starts with particles near it, and 250 steps of a clipped drift
never has to move mass between modes. Our Dirac source has to transport all of
its mass out of a single point, and the transport reaches the modes it reaches:
it sharpens the profile it has instead of finding the ones it is missing. On
this target, starting from the answer's support is worth more than being
unbiased about it.

The honest reading is that these are two different failures, not a ranking:
theirs is a source-tilted sampler that inherits good coverage from a source it
should not have, ours is a correctly sourced sampler with a transport that
abandons modes. Reporting only mode TV would hand them the benchmark; reporting
only ⟨E⟩ would hand it to us. Both are in the table.

It is not overfitting. Evaluated against the 1,433 **held-out** modes the
sampler never saw, KS(E) is 0.05457 and coverage 0.88625 — *better* than on the
modes it trained on (0.10048 / 0.7679), with mode TV essentially unchanged at
0.51013. A memoriser would show the opposite gap.

The training trace says the failure is specific. Over the four κ stages, ΔE
improved monotonically (5.47 → −0.07), KS(θ) improved monotonically
(0.495 → 0.080), and mode TV **never improved at all** — it oscillated in
0.34–0.55 for the entire run, best 0.339 at the κ = 300 stage, and drifted back
to 0.509 by the end. Within the final κ = 600 block, mode TV went 0.481 → 0.509
and coverage 0.825 → 0.767 while KS(θ) kept improving. So the sampler
progressively sharpens onto the correct radial and angular *profile* around the
modes it has, while quietly abandoning modes. Getting the profile right and the
mode weights wrong is exactly the failure mode a mean-energy-only report would
hide, which is why coverage and mode TV are gated.

**What we have not measured, and therefore do not claim.** The decisive control
was never run: `earthquake.py exact` — the same integrator with the score
computed by quadrature instead of learned — has only been run at a reduced
κ = 20, where it is clean first-order with no bias floor. Until it is run at
κ = 600 we cannot say whether mode TV ≈ 0.51 is a 512-step discretisation floor
that no controller could beat, or a genuine failure of the learned control.
Two specific suspects, in order:

1. `--max-drift 200` clips the drift, whose natural magnitude at κ = 600 is
   O(κ). The clip value was chosen by analogy with R-ASBS's 30 and never tuned.
   A clip below the true drift would systematically prevent transitions between
   modes, which is the observed symptom.
2. 512 steps may be too coarse for 2.4°-wide modes.

Both are cheap to test (`exact --kappa 600 --sweep 64 128 256 512`, ~10 min, no
network) and neither has been tested, so the section is reported as a failure
with an open cause rather than a diagnosis.

**Limitations.**

- **Gates E1 and E2 fail**, as above. This is the one benchmark in the
  repository where our sampler does not reach its own threshold.
- **There is no figure for this experiment.** `figures.py` and `gallery.py`
  have no earthquake code path, so everything above is tables and prose. The
  R-ASBS §4.2 globe plot has no counterpart here.
- **The target data is not in the repository.** `rasbs_ref/` is gitignored, and
  `rasbs_ref/query.csv` is where the 4,776 epicentres come from.
- **The exact-control baseline is missing at κ = 600**, so the failure is not
  yet attributed to the learned control as opposed to the integrator.
- **R-ASBS beats us on mode coverage here**, and we have not shown that our
  Dirac source is the reason — the argument above is a mechanism, not a
  measurement. The obvious experiment, rerunning ours from a Haar source to see
  whether coverage rises to ≈0.94, **is not available to us**, and the reason is
  worth stating because it is the same reason the method is corrector-free.
  Weighting the reference by `f₁(X₁)` produces a process whose *initial* density
  is `μ(x₀) · h(x₀, 0)` with `h(x₀, 0) = ∫ p_{r₀₁}(x₀·y) f₁(y) dy`. For a Dirac
  `μ` that integral is `∫π = 1`, so the source survives the h-transform exactly
  — the Dirac source is what buys the closed form, not an incidental choice. For
  a Haar `μ` it is `4π(P_{r₀₁}π)(x₀)`, the heat-smoothed target, so the run
  would no longer start uniform. We measured the distortion at `r₀₁ = 1`:
  initial density over uniform ranges 0.833 to 1.175, a **1.41× spread**, with
  **TV(initial law, Haar) = 0.0376**. Getting `p_base` constant — which Haar
  does give, since it is heat-stationary — is necessary and not sufficient. The
  ablation therefore needs a corrector we have not derived, or a mixture-of-
  Diracs source in which each component is individually exact. Neither was run.
- **The R-ASBS column is a port, not their binary.** MATLAB's Deep Learning
  Toolbox is not available here, so their script cannot be executed for a
  cross-check; see the porting audit in the Sphere section.
- **Still S².** Real data, but a 2-dimensional manifold.

```bash
python structured_asbs/earthquake.py verify                    # sampler + log Z identities
python structured_asbs/earthquake.py exact --kappa 20          # quadrature control, step sweep
python structured_asbs/earthquake.py train --kappa 600 --anneal 150 300 450 600 \
    --steps 512 --iters 6000 --hidden 512 --n-dir 384 --bw-hi 128 \
    --max-drift 200 --tag earthquake_k600 \
    --out json/results_earthquake_k600.json
python rasbs/rasbs_sphere_port.py --problem quake              # the R-ASBS column
```

---

## Stiefel

**Experiment D — St(4,2), matrix Killing readout, and the head-to-head against
R-ASBS.** The state space is the 4×2 orthonormal frames, dimension 5, with
`E(X) = tr(XᵀHX)` and `H = diag(1, 2, 5, 8)` — R-ASBS's own `H` up to a change
of basis, and their own β grid. Temperature is `τ = 1/β`, β swept from 1e-3 to
1e6. Experiment D additionally uses `σ = √2`, `nq = 64` fibre quadrature nodes
and `--antithetic` (the exact 16-fold symmetry `T_s(X) = SXD`, automatically
disabled with `--frame`). Gate `D0` includes the first-moment test for the spin
clock: each SU(2) ≅ S³ factor of Spin(4) runs at heat time `s = r/2`, not `r`,
and the test separates the two conventions by ~160× at `r = 0.25`, so a wrong
clock cannot pass silently.

Their paper reports the energy curve as a plot rather than a table, so every
R-ASBS number below was produced by rerunning their algorithm. `alg2_stiefel.m`
needs the MATLAB Deep Learning Toolbox, which Octave does not implement, so
`rasbs_port.py` is a line-for-line PyTorch port of frozen upstream commit
`bb71d1496468658f4ff1c9995773968fd12706a8` (2026-08-28), keeping their Z₂×Z₂
basis, the Haar source, the ambient Euler step plus QR retraction, the backward
sequential projection, σ = 1, N = 199, B = 600, 1000 epochs, lr = 1e-3.

| | ours | R-ASBS (`rasbs_port.py`) |
|---|---|---|
| script | `stiefel.py` | port of `alg2_stiefel.m` @ `bb71d14` |
| target | `E(X) = tr(XᵀHX)`, `H = diag(1,2,5,8)` | same `H`, ambient Z₂×Z₂ basis |
| source | `E₀ = [e₁, e₂]` (**Dirac**) | **Haar** |
| constraint | exact geodesic step | ambient Euler + QR retraction |
| network | `ScoreNet`, hidden 256 | `netU` + `netH` |
| parameters | **139,528** | **140,560** (1.01× ours) |
| integration steps | 199 | 199 |
| iters × inner | 1500 × 8 | 1000 epochs |
| batch / minibatch | 2048 / 16384 | 600 |
| lr | 1e-3 | 1e-3 |
| EMA / buffer | 0.9995 / 4 | — |
| eval samples | 100,000 | **5,000** on the 199-step sweep, 100,000 on the step and high-β reruns |
| reference | MCMC, 2e5 chains × 4000 sweeps (`results_stiefel_ref.json`; the per-run references in the other files use 3000, and the β = 65/80 chain legs 5e4 × 1000) | same |

### Results

`N` is the number of integration steps. β ≤ 20 is the shared 199-step grid both
methods were originally run on; at β = 50 and 100 each column is quoted at the
finest grid we ran for it, for the reason given in the ablation below. R-ASBS
column from `rasbs/rasbs_port.py` → `json/results_rasbs_stiefel.json` at 199
steps and `json/results_rasbs_highbeta_steps_1024.json` at β = 50 and 100; our
β = 50 and 100 cells from `json/results_stiefel_anneal_b50.json` and
`json/results_chain_b100.json`, the rest from `json/results_stiefel_fill.json`.
The `err` column is always measured against the shared `reference` column, so
the β = 50 and 100 entries differ by 0.002–0.003 from the per-run MCMC quoted
in the refinement table below.

| β | reference | R-ASBS | err | N | ours | err | N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 7.9965 | 7.9546 | −0.042 | 199 | 7.9958 | **−0.001** | 199 |
| 0.01 | 7.9628 | 7.9407 | −0.022 | 199 | 7.9679 | **+0.005** | 199 |
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
| 200 | 3.0243 (MCMC) | 3.2482 | +0.224 | 199 | not trained | — | — |
| 10⁶ | 3.0233 (MCMC, frozen) | 3.1847 | +0.161 | 199 | not trained | — | — |

Worst finite-β error: **0.547** (theirs, β = 2) against **0.166** (ours, β = 1.3
at matched budget). Orthogonality residual: **3.3e-14** at β = 2 against
3.4e-07, and 3.0e-14 when recomputed from disk across all 100,000 samples
(figure 7). It grows slowly with the step count — 1.1e-14 at 32 steps to
6.6e-14 at 512 in `json/results_stiefel_sweep.json` — and the largest we
measure anywhere is 3.2e-13, at β = 100 and 3184 steps. Every one of those is
eleven orders of magnitude below their 3.4e-07, which is what it is because
the update is an exact geodesic step rather than a retraction. Two things the
curve alone does not show:

**The error changes sign at β ≈ 0.05.** At β → 0 the target *is* Haar, so their
source is exactly right and the −0.042 residual is pure QR discretisation. As β
grows the target leaves Haar, the source-tilting bias switches on and dominates,
peaking at +0.547 near β = 2. The correct initial law for the h-transform is
Haar·φ₀ / ⟨Haar, φ₀⟩ and φ₀ is not constant — the Dirac-vs-Haar distinction from
the top of this README, measured.

**The β → ∞ limit is never reached.** Their own paper states the limit is 3; the
rerun plateaus just above 3.18 — 3.2086 at β = 10³, 3.1873 at 10⁴, 3.1847 at
10⁶ — and stops moving there.

**Budget matching.** A fair comparison has to match training iterations,
terminal energy/gradient evaluations, integration steps, generated samples and
parameter count — and of those the oracle call count is the one that is
implementation-independent, so it is the one we hold fixed and report
alongside wall-clock rather than instead of it. Counting exactly revealed that our default uses
**3,072,000** oracle calls against their **600,000**, a 5.1× advantage. The
budget-matched run (`--batch 400 --iters 1500`, exactly 600,000 calls, same 199
steps) gives +0.166 / +0.161 / +0.111 at β = 1.3 / 2 / 5 against their +0.451 /
+0.547 / +0.479 — removing the advantage costs 0.01–0.03 and changes no
conclusion. Parameters (1.01×) and step counts are matched throughout. Sample
counts are not: their 199-step sweep evaluates 5,000 terminal samples against
our 100,000, which is their script's own default and which we left alone. It
makes their column noisier, not biased: the one configuration measured both
ways — 199 steps at β = 50 and 100 — gives 3.3116 / 3.2614 at 5,000 samples and
3.3139 / 3.2626 at 100,000, a shift of 0.002 against errors of 0.25.
**Wall-clock**, single A100: ~1130 s per β at 199 steps, ~2180 s at 398, ~4380 s
at 796 for us; ~780 s per β at 512 steps and ~1560 s at 1024 for them. Every
`results_*.json` records `train_s` (ours) or `wall_s` (theirs).

### Beyond the mean energy

The table above compares `E[E]` and nothing else, which is the weakest question
one can ask of a sampler: a badly wrong distribution can still have the right
mean. `json/results_rasbs_stiefel.json` left `KS_E` as `NaN`, so three further
statistics were recovered from the stored samples by
`structured_asbs/_stiefel_extra.py` (no retraining) and written to
`json/results_stiefel_extra.json`, at the three temperatures where the
budget-matched ablation also exists.

| β | metric | R-ASBS | IASBS | IASBS₆₀₀ |
|---:|---|---:|---:|---:|
| 1.3 | \|ΔE\| | 0.4581 | **0.1548** | 0.1656 |
| | KS(E) | 0.1407 | **0.0537** | 0.0552 |
| | E_std / ref (1.1628) | 1.25× | **1.05×** | 1.06× |
| | \|XᵀX − I\| | 3.2e-07 | **3.5e-14** | 3.4e-14 |
| 2 | \|ΔE\| | 0.5513 | **0.1372** | 0.1610 |
| | KS(E) | 0.1857 | **0.0718** | 0.0753 |
| | E_std / ref (0.7807) | 1.65× | **1.10×** | 1.14× |
| | \|XᵀX − I\| | 3.1e-07 | **3.3e-14** | 3.3e-14 |
| 5 | \|ΔE\| | 0.4781 | **0.1100** | 0.1106 |
| | KS(E) | 0.2750 | **0.1369** | 0.1353 |
| | E_std / ref (0.2903) | 3.27× | **1.23×** | 1.24× |
| | \|XᵀX − I\| | 3.1e-07 | **3.2e-14** | 3.4e-14 |

`KS(E)` is the Kolmogorov–Smirnov distance between the sampler's energy law and
the MCMC reference law, so it sees the whole distribution rather than its first
moment. IASBS is 2.0–2.6× closer on it, and the budget-matched IASBS₆₀₀ is
indistinguishable from the full-budget run — at β = 5 it is very slightly
*better*, which is noise, not a real ordering.

**Sample sizes differ and KS is sensitive to that.** The port evaluates 5,000
terminal samples, IASBS 100,000, and KS has an O(1/√n) null floor. Scoring exact
MCMC draws of size 5,000 against the reference CDF gives a floor of **0.011**,
and subsampling IASBS to the same 5,000 moves its KS only 0.0537 → 0.0597,
0.0718 → 0.0761, 0.1369 → 0.1374. So the gap is real: R-ASBS sits 13–25× above
the floor, IASBS 5–12×.

**Dispersion.** Both methods over-disperse the energy, and the gap widens with β:
by β = 5 R-ASBS is 3.27× too wide against the MCMC reference while IASBS is
1.23×. Since the mean error is roughly flat in β for both, the second moment is
where the two methods actually separate.

**Constraint.** Seven orders of magnitude, at every β, because the IASBS update
is an exact geodesic step on the manifold and the port's is a QR retraction.
This one is structural rather than statistical: it does not improve with more
training or more samples.

**A trap worth recording.** IASBS uses `H = diag(1, 2, 5, 8)`; the port uses the
R-ASBS paper's circulant `H_RASBS`, which has the *same spectrum* but a different
eigenbasis. St(4,2) is invariant under left `O(4)`, so the law of `E` under the
Gibbs target is identical for both and every scalar above is comparable — but the
samples are not interchangeable, and scoring the port's samples with the IASBS
`H` returns `E ≈ 8.00` at every β, i.e. exactly the β → 0 asymptote, which looks
like total failure and is in fact a basis mismatch. Each set of samples is scored
with the `H` it was trained against.

### The step-count ablation

Our error is discretisation and falls with `N`; theirs is not, and does not.
That is the predicted signature — our controller identity is exact, so only
the integrator is approximate, while their geometric surrogate should leave a
nonzero floor — but a floor must not be claimed before it is measured.
Measured at β = 2, where their bias peaks, everything except the
step count held fixed:

R-ASBS row from `rasbs/rasbs_steps.sh` → `json/results_rasbs_steps_*.json`; our
row from `stiefel.py`'s `sweep` subcommand → `json/results_stiefel_sweep.json`.

| N_steps | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| R-ASBS error | 0.805 | 0.661 | 0.601 | 0.526 | 0.493 |
| our error | 0.621 | 0.323 | 0.186 | 0.121 | **0.072** |

Per doubling ours falls ~1.7×, theirs ~1.12×. Richardson extrapolation in 1/N on
their two finest grids gives a floor of **≈0.46** — roughly 93% of their
remaining error at N = 512 is not discretisation at all. Their orthogonality
residual likewise stays pinned near 3.8e-07 at every step count.

At β ≥ 50 the same effect runs the other way first. The score is O(β), so `h·score`
is O(1) per step at β = 100 on a 199-point grid, and our sampler falls apart
there — that is the honest matched-grid number and R-ASBS beats us by an order
of magnitude:

R-ASBS rows from `rasbs/rasbs_steps_highbeta.sh` →
`json/results_rasbs_highbeta_steps_*.json`; our β = 50 row from
`json/results_stiefel_anneal_b50.json`, our β = 100 row from
`json/results_chain_b100.json` (the 50 → 65 → 80 → 100 chain, which is the
better of the two β = 100 controls; the direct β = 50 warm start in
`json/results_stiefel_anneal_b100_fine.json` gives +3.4695 / +0.7458 / +0.0461
in the same three cells). The two 199-step "ours" cells are the unconverged
199-step controls from `json/results_stiefel_fill.json` and
`json/results_stiefel_anneal_b100.json`, carried over from the headline table
so the refinement has a starting point; every cell in this table is measured
against its own run's MCMC reference, which is why the β = 50 and 100 endpoints
read +0.0101 / +0.0187 here and +0.012 / +0.022 in the headline table above.

| β | 199 | 398 | 512 | 796 | 1024 | 1592 | 3184 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| R-ASBS, 50 | +0.272 | — | +0.255 | — | **+0.248** | — | — |
| ours, 50 | +4.391 | +0.0685 | — | +0.0223 | — | **+0.0101** | — |
| R-ASBS, 100 | +0.235 | — | **+0.218** | — | +0.221 | — | — |
| ours, 100 | +5.290 | +4.063 | — | +3.535 | — | +0.750 | **+0.0187** |

Ours falls by 435× and 283× over that refinement; theirs moves by −0.024 at
β = 50 and −0.014 at β = 100 over the same 199 → 1024 span, and by −0.003 from
512 to 1024 — run-to-run noise on a quantity that has stopped moving. The refined
cells reuse a trained control and never retrain, but they are warm-started from
the β below, a recipe the rest of the column does not use. Three by-products
worth recording:

- **It is not the loss scale.** The regression loss grows like β² (0.055 at
  β = 0.001, 2.8e5 at β = 100) but Adam is scale-invariant. We removed the scale
  anyway — `ScoreNet` factorises `score = out_scale · raw(t, X)` with `out_scale`
  a running RMS of the label — and the answer changed by nothing (+4.463 /
  +5.271 against +4.391 / +5.290). Clean negative result; the factorisation was
  kept because it is what makes the refined cells reachable.
- **The gate cannot rank controls at β = 100.** It evaluates on the training
  grid. Two β = 100 controls both fail it identically (E = 6.50 and 6.57 at 796
  steps), yet at 3184 steps the 50 → 65 → 80 → 100 chain gives +0.0187 / KS 0.258
  / 2.1× spread / 2602 s against the direct β = 50 warm start's +0.0461 / KS
  0.312 / 16× spread / 4393 s. We reported the worse one until we checked.
- **Refinement fixes the mean, not the law.** At β = 100 and 3184 steps ΔE is
  +0.019 and the spread 2.1× the reference's, but KS(E) = 0.258 still fails the
  gate every other β passes.

### Figures

`fig5_stiefel` carries the energy curves and the step-refinement panels.
`fig7_stiefel_frames` draws 96 raw St(4,2) frames per sampler plus the second
moment E[XXᵀ]. **`stiefel.py` and `rasbs_port.py` do not use the same basis** —
ours works in the eigenbasis of H, theirs in R-ASBS's ambient Z₂×Z₂ basis — so
figure 7 rotates their samples into the common eigenbasis first; without that
the two second moments look like each other's answers. Once rotated, at β = 2
the distance ‖E[XXᵀ] − target‖_F is **0.028** for us and **0.132** for them
against 0.755 for Haar, i.e. R-ASBS sits 17% of the way back to its own source
and we sit 4% — the source-tilting bias visible directly in a quantity that has
nothing to do with the scalar energy.

Figure 7's bottom row recomputes the orthogonality residual from disk:
**3.0e-14 across all 100,000 of our samples against 2.9e-07 across theirs**.
That comparison is only meaningful because **checkpoints store samples in
float64**; float32 epsilon is 1.2e-07, which happens to look exactly like their
retraction error. `common.py:save_ckpt` promotes sample tensors to float64,
figures draw the residual only for genuinely-float64 files, and `regen_f64.sh`
refreshes the pre-change β = 2 pair.

**Limitations.**

- **At β ≥ 50 on the shared 199-step grid our sampler does not converge and
  R-ASBS is better than we are** (+0.27 against +4.39 at β = 50). Refining the
  grid and warm-starting recovers +0.010 and +0.019, far below their floor, but
  at 8–16× their step budget and with a recipe the rest of the sweep does not
  use. We keep the 199-step number in the headline.
- **The β = 100 refined cell fixes the mean and not the law** (KS(E) = 0.258).
  We did not find a grid fine enough to fix the distribution there.
- **We cannot predict which (β, steps, warm start) combinations are stable.**
  `out_scale` sits near 0.55 β on-mode and near 2 β off-mode and the failing runs
  are the off-mode ones — but a 2.5× β jump succeeded where a 1.3× jump on a
  finer grid failed, and every leg of the winning chain sat off-mode. The step
  counts above were found by measurement, not derived.
- **Our self-imposed gate `|E − E_MCMC| < 0.05` is not met at β ≥ 0.5**
  (0.10–0.17 at 199 steps). Refining the same trained control drops the β = 1
  error to 0.050 at 796 steps — that measurement is on the frame-sensitive
  target D2 (`json/results_stiefel_frame.json`: ΔE 0.117 at 199 steps, 0.0493
  and 0.0516 across two seeds at 796, KS(E) 0.029) — so about half the residual
  is discretisation. The gate is stricter than anything R-ASBS achieves at any β, but it is reported as failed
  rather than relaxed.
- **The MCMC reference stops converging for β ≳ 1000**, where acceptance goes to
  zero and the chain freezes at 3.0233 for every β from 10³ to 10⁶; the true
  limit is 3, so the reference itself carries about +0.023 of error there, which
  is a tenth of R-ASBS's residual and does not change the comparison.
  At β ≤ 100 two independent chains agree to 0.0001–0.01.
- **KS(E) degrades at large β** (0.23 at β = 10) even as the mean error improves,
  because the target concentrates and KS becomes very sensitive.
- **The frame-sensitive target D2 has no analytic reference** — MCMC is the only
  ground truth, so its error bar is the reference's own.
- **St(4,2) is 5-dimensional.** Like S², a correctness benchmark, not a
  high-dimensional one.

```bash
python rasbs/rasbs_port.py --check-retraction   # GS == sign-corrected QR
python rasbs/rasbs_port.py --out json/results_rasbs_stiefel.json
python structured_asbs/stiefel.py ref --mcmc-sweeps 4000 \
    --betas "0.001,0.01,0.1,0.5,1.3,2,5,7,10,20,50,100,200,1000,10000,1000000"
python structured_asbs/stiefel.py verify                       # gate D0, incl. spin-clock test
python structured_asbs/stiefel.py train --frame --betas 1.0 --steps 199 \
    --iters 2500 --mb 16384 --ema 0.9995 --seeds 2 --refine 2,4 \
    --tag stiefel_frame --out json/results_stiefel_frame.json   # target D2
python structured_asbs/stiefel.py sweep --beta 2 --iters 1500 --mb 16384 \
    --ema 0.9995 --antithetic --tag stiefel_d3 \
    --out json/results_stiefel_sweep.json   # our beta=2 step ablation
bash rasbs/rasbs_steps.sh                   # their beta=2 step ablation
bash rasbs/rasbs_steps_highbeta.sh          # their beta=50/100 step ablation
bash rasbs/regen_f64.sh                     # float64 rerun at beta=2
```

Every run in this repository writes a checkpoint. `--ckpt-dir` now defaults to
`ckpt` in every entry point rather than to the empty string, and the two step
ablations above pass it explicitly. Both were originally written without it,
which silently discarded the eleven controls they trained; the numbers in this
section survived only because they were already in `json/`. The default was
changed so that the omission cannot recur.

---

## Additional Results

Supplementary experimental results for *"Adjoint Schrödinger Bridge Sampling on
Structured State Spaces via Markov Semigroup Intertwining"* (IASBS).

All numbers below are produced by the scripts in this repository. Every entry is
reproducible from the JSON file named in its row; checkpoints are written to
`ckpt/` (gitignored, 2.5 GB, regenerable). Hardware: 2x NVIDIA A100 80 GB,
PyTorch 2.5.1 / CUDA 12.4, `torch.set_default_dtype(torch.float64)` throughout.

Status legend: **DONE** = finished with gate verdicts recorded;
**RUNNING** = in progress, last logged value shown.

---

### 1. Summary of claims supported

| # | Claim | Evidence |
|---|---|---|
| 1 | IASBS reaches exact-IPF accuracy on discrete structured spaces, up to `\|Omega\| ~ 1e4` | §2.1, §2.2 — TV at or below the i.i.d. sampling floor at Ising L=4 and occupation m=4; §2.1 — Ising L=5 (`\|Omega\| = 5.2e6`) misses the gate at 0.07228 |
| 2 | IASBS accuracy does not degrade with state-space size | §2.3 — KS improves monotonically from m=32 to m=1000 |
| 3 | IASBS extends to non-Dirac (Haar) sources without loss | §3.2 — matches the Dirac headline to 3 decimal places |
| 4 | Reported R-ASBS mode collapse is an initialisation artifact | §4 — collapse vanishes under the authors' own init |
| 5 | DAM needs millions of Monte-Carlo adjoint rollouts; IASBS needs none | §5, §6.4 — 20.9 M rollout-endpoint f1 evals for DAM's best TV vs 0 for IASBS (IASBS still evaluates the energy; see §6.4) |
| 6 | DAM is fragile and expensive: it diverges out of the box on every space larger than `\|X\| = 35`, and repairing it needs a per-problem control box, label truncation and 5x the budget | §5.2, §6.2 — divergence at K=1/4 (m=4), K=16 (m=32, Ising L=4), K=64 (m=32); §6.2.1 — the repair reaches TV 0.0617 on Ising L=4 for 169 M f1 evals and 975 M jumps, still 1.8x worse than IASBS at zero adjoint rollouts |

---

### 2. IASBS on discrete structured spaces

#### 2.1 Fixed-magnetization Ising on an L x L periodic lattice, non-Dirac (uniform) source — validated against exact IPF

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

#### 2.2 Occupation process m=4, non-Dirac — validated against exact IPF

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

#### 2.3 Occupation process — scaling in m

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

##### 2.3.1 Numerical stability fix (m=32)

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

### 3. IASBS on the sphere S^2

Target: bimodal density; gates **C1** mean hemisphere-mass error < 0.03
(R-ASBS reference 0.062), **C2** KS on the x_3 marginal < 0.05.

#### 3.1 Dirac source, 5 seeds, 4000 iterations

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

#### 3.2 Non-Dirac (Haar) source, 5 seeds, 4000 iterations — **DONE**

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

### 4. R-ASBS baseline on S^2 — the mode-collapse claim

Bimodal target, kappa = 600, 600 epochs, 500 steps, 5 seeds.

#### 4.1 Default initialisation

| seed | north_mass | north_err | KS_z |
|---|---|---|---|
| 0 | 0.00004 | 0.49996 | 0.50587 |
| 1 | 0.00004 | 0.49996 | 0.50590 |
| 2 | 0.99762 | 0.49762 | 0.50328 |
| 3 | 0.00173 | 0.49827 | 0.50438 |
| 4 | 0.00191 | 0.49809 | 0.50476 |
| **mean** | — | **0.49878** | **0.50484** |

Every seed collapses to a single mode (mass 0 or 1, never 0.5).

#### 4.2 Authors' own `--init matlab`

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

#### 4.3 Head-to-head on the same target

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

### 5. DAM baseline (Discrete Adjoint Matching)

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

#### 5.1 K sweep, occupation m=4, N=4, |X|=35, 128 steps

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

#### 5.2 The stability cliff below K=16

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

#### 5.3 Diminishing returns above K=16

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

#### 5.4 Cost comparison, the headline number

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

### 6. Computational cost

All timings on a single NVIDIA A100 80 GB, float64 throughout. Where runs shared a
device the contention is noted, because DAM runs are latency-bound (see §6.2) and
co-scheduling inflates their wall clock roughly linearly.

#### 6.1 IASBS

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

#### 6.1.1 R-ASBS baseline (sphere S^2)

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

#### 6.2 DAM

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

#### 6.2.1 Trying to fix the divergence

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

**Round 3 — clamp above the target (`a-clamp 8`, 27% headroom over 6.308,
rate multiplier 3.0e3 against the baseline's 4.9e8).**

| arm | settings | best E-hist TV | E-hist TV @ 600 | TV @ 600 | ESS mean / p10 | peak ESS | jumps | wall |
|---|---|---|---|---|---|---|---|---|
| F | `a-clamp 8 m-clip 5` | 0.40969 (it 550) | 0.41500 | 0.77586 | 2.34 / 1.01 | 2.93 | 176.7 M | 1446 s |
| G | F + `ess-min 3 coef-cap 10` | **0.15688** (it 325) | 0.20890 | 0.81692 | **4.32 / 1.18** | **5.68** | 102.5 M | 1296 s |

The clamp diagnosis is confirmed in the direction predicted. Arm G retains
**4.32 of 16** rollouts against 2.35 at clamp 3 and **1.04** for the diverging
baseline, holds the energy histogram near 0.19-0.21 instead of bouncing off a
boundary, and does it with **42% fewer simulated jumps** than arm F — the ESS
filter and the `r/q` cap are doing real work, not just trimming. Arm F, the same
clamp without that damping, is markedly worse on every axis, so at clamp 8 the
outlier removal is what keeps the run healthy.

**The full TV still does not descend on any arm.** Across all eight
configurations it sits in 0.76-0.83 while the energy histogram falls as low as
0.157. That gap is informative: the energy *marginal* is being learned and the
within-energy-level distribution is not. Under the Gibbs target, equal-energy
states are equiprobable, but `f1 propto mu / p_base(. | x_0)` is not flat within
an energy level — it has to undo the base process's geometry-dependent
preference among equal-energy states — and that is exactly the component a
Monte-Carlo adjoint with 4 effective rollouts out of 16 estimates worst.

**Round 4 — the budget test, and the answer. DAM converges.**

Every arm so far had 600 iterations. Occupation m=4 needed **1200** at K=16 on a
space 368x smaller, so "600 was simply not enough" had not been excluded. It was
the whole story. Both round-4 arms descend monotonically, with a **positive**
loss and a **rising** ESS for the entire run:

| arm | K | iters | TV trajectory | E-hist TV @ end | ESS start -> end | ESS mean / p10 | f1 evals | jumps | wall |
|---|---|---|---|---|---|---|---|---|---|
| H | 16 | 3000 | 0.795 (100) -> 0.814 (800) -> 0.399 (2000) -> **0.20278** | 0.11677 | 2.61 -> **11.19** / 16 | 6.49 / 1.52 | 52,224,000 | 428.6 M | 6488 s |
| I | 32 | 1500 | 0.685 (300) -> 0.216 (800) -> **0.12905** | 0.06858 | 4.71 -> **26.01** / 32 | 15.66 / 2.31 | 50,688,000 | 346.9 M | **3493 s** |

Constraint violations 0 on both.

**Rounds 1-3 were reading a transient.** Arm H's loss is *negative* until
iteration ~700 and its TV *rises* to 0.870 at iteration 200 — at 600 iterations
it looks exactly like the stalled runs of rounds 1-3, and only then does it turn
around. Every earlier arm was cut off inside that transient. The stabilisers had
already fixed the method; the diagnosis "stabilised but not learning" was wrong,
and it was wrong because of a budget that had been sized from a state space 368x
smaller.

**Both axes bind.** Arm I reaches a *better* TV than H in **half** the wall
clock. Retention is why: the damped arm ends at 70% of its rollouts at K=16 and
**81%** at K=32, and that extra retention more than pays for the doubled
per-iteration cost. So the §5.2 cliff and the round-2 clamp were two of three
constraints — control-box size, adjoint retention, and gradient budget — and all
three had to be right at once.

Neither run had flattened at its cutoff: I was still falling 0.0110 per 100
iterations over its last 200.

**Round 5 — running the K axis out to the gate (~3.2 h per arm).**

| arm | K | iters | TV trajectory | TV final | E-hist TV | ESS end | ESS mean / p10 | f1 evals | jumps | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| J | 32 | 5000 | 0.129 (1500) -> 0.0824 (3500) -> **0.06168** | **0.06168** | 0.04759 | **30.47 / 32** | 23.70 / 7.43 | 168,960,000 | 974.8 M | 11,428 s |
| L | 64 | 2500 | 0.125 (1500) -> **0.0798** (2250) -> 0.08103 | 0.08103 | 0.06567 | 58.62 / 64 | 42.47 / 8.36 | 166,400,000 | 924.0 M | 6,950 s |

Constraint violations 0 on both. Neither crosses the TV <= 0.05 gate, and two
things follow.

**The K axis has saturated.** L spends essentially the same terminal-evaluation
budget as J (166.4 M vs 169.0 M) and lands *worse* — 0.081 against 0.062.
Retention at K=32 is already 95% (30.47 of 32) so the extra 32 rollouts per
label buy nothing that was missing. Combined with round 4, where K=32 beat K=16
at half the wall clock, the picture is a single optimum around K=32 on this
benchmark rather than a monotone "more rollouts is better".

**The gate is a budget question, not a wall.** J was still descending at its
cutoff, 0.06435 -> 0.06168 over its last 250 iterations, with the loss positive
and the ESS still creeping up. Extrapolating that rate puts TV = 0.05 near
iteration 6100.

**Round 6 (running), ~4.4 h per arm.** Arm **M** takes K=32 to 7000 iterations,
~15% past the extrapolated crossing. Arm **N** is the transfer test, and it is
the more important of the two: the repair has only ever been exercised on Ising
L=4, so N applies the identical control box and truncation to
**occupation-scale m=32 at K=64**, the other benchmark that diverged (at
iteration 15, with 153 jumps per f1 evaluation and ESS p10 pinned at 1.00).
Note that the clamp there *cannot* be sized against a known optimum —
`ScaleOccupation` has no `ExactControl` to read one off — which is exactly the
practical difficulty the revised claim 6 asserts.

**Where this leaves claim 6.** The claim as originally written — "DAM has a
hard stability cliff, and raising K does not clear it" — is **too strong and is
retracted in that form**. Raising K alone does not clear it, which is what the
K=64 leg at m=32 showed, but K was never the only axis. Three things had to hold
simultaneously, and the unmodified method gets two of them wrong:

1. **a bounded control box that still contains the optimum.** Baseline
   `|a| <= 20` permits a rate multiplier of 4.9e8 and diverges; `|a| <= 3`
   is stable but excludes the exact optimum of 6.31 and cannot converge;
   `|a| <= 8` does both.
2. **a truncated adjoint label.** `|log m_hat| <= 5` in place of 30. At 30 a
   single label can carry a weight of 1e13 into the gradient, which is the
   mechanism of §5.2.
3. **enough gradient steps.** 3000 at K=16, against the 600 that a state space
   368x smaller had needed.

With all three, DAM on Ising L=4 goes from a diverging run at **-4e12 loss,
TV 0.973 and ESS 1.04/16** to a converging one at **TV 0.06168, ESS 30.47/32**.
The honest revised claim is therefore not that DAM fails, but that it is
**fragile and expensive**: it needs a per-problem control box sized against an
optimum one does not know in advance, a truncation constant, and **169 million**
rollout-endpoint evaluations plus 975 million simulated jumps to reach a TV that
IASBS beats by **1.8x** (0.03470, §2.1) with **zero** adjoint rollouts, in
11,428 s against IASBS's 1,869 s. The comparison in §6.3 is
unaffected in direction and is now made against a DAM that actually works on
this benchmark rather than one that has fallen over.

#### 6.3 Cost of the comparison, head to head

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

#### 6.4 What "0 terminal evaluations" counts — exact accounting

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

### 7. Reproduction

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

### 8. Still running / still to run

**Running:** DAM stabiliser round 6 -- `fix_M_K32_7000` (Ising L=4, K=32, 7000
iterations, ~15% past the extrapolated TV = 0.05 crossing) and `fix_N_occs32`
(occupation-scale m=32, K=64, 1000 iterations, the transfer test of the repair
onto the *other* benchmark that diverged). ~4.4 h per arm on one GPU each --
see §6.2.1.
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
| DAM Ising L=5, K=16 | Ising L=4 diverges at K=16 unstabilised; the stabilised configuration of §6.2.1 would be the one to try, at an estimated 5x the L=4 cost |
| IASBS_600 Stiefel at beta = 0.1, 0.5, 7, 10, 20 | matched-budget ablation not performed at those temperatures; ~28 min per beta if wanted |

**Text only**

- README correction of the R-ASBS mode-collapse description (§4.2).

**Two honest negatives, recorded rather than buried.** IASBS Ising L=5 misses its
accuracy gate at 0.07228, and no *unmodified* DAM configuration attempted outside
the |X| = 35 occupation problem converged at all — §6.2.1 repairs it on Ising L=4
but only with a control box sized against the exact optimum, a tighter truncation
and 5x the gradient budget, so the head-to-head of §5.4 and §6.3 still rests on
the one benchmark where stock DAM works. That is a real limit on the strength
of the DAM comparison and is stated as such: on larger discrete spaces the claim is
not "IASBS beats DAM by 17%", it is "IASBS converges and DAM does not", which is a
weaker and differently-shaped claim.

---

*This file is updated each time an experiment finishes. Numbers are copied verbatim
from the JSON artifacts named in each section; nothing here is projected except
where explicitly marked "proj." or ">".*


---

## License

Our code is MIT (see `LICENSE`).

The MIT grant covers the original work only. `rasbs/` holds ports of the R-ASBS
reference implementation, whose upstream repository carries no license file at
the commit we read (`bb71d14`). We claim no ownership of that material; anyone
wanting to reuse `rasbs/` should ask its authors. The upstream clone is not
redistributed here — `.gitignore` excludes `rasbs_ref/`.
