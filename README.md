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
  fixed_support.py        Appendix A.2  fixed-support space X^(r)_{n,k} + GB1
  tests_math.py           standalone mathematical unit tests
  figures.py              all paper figures + the comparison table
  gallery.py              sample gallery -- the samples, not their metrics
  remeasure.py            re-derives the Ising and sphere tables from ckpt/
  _stiefel_extra.py       KS(E) and E_err for the Stiefel sweep, from ckpt/
  _rasbs_extra.py         KS(phi) and second-moment error for R-ASBS, from ckpt/
  scripts/                the shell scripts that produced our json/ entries
    rerun_ckpt.sh           Ising + occupation trains and scaling legs
    rerun_sphere.sh         sphere exact + plain/anti/sym, 5 seeds each
    run_occ.sh              occupation estimator ablation (1-sample legs)
    run_scale.sh            occupation scaling + variance probes
    run_scalefix.sh         Stiefel beta = 50/100 at 199 steps
    run_anneal.sh           Stiefel beta 50 -> 100 anneal
    run_anneal_chain.sh     Stiefel beta 50 -> 65 -> 80 -> 100 chain
    run_anneal_b100.sh      single beta = 100 leg from the beta = 50 control
    run_anneal_b100_fine.sh same at 796 steps
    run_gb1_anneal.sh       GB1 k=3 tau 2 -> 1.4 -> 1 chain (the GB1 default)

dam/                      THE OTHER BASELINE (Discrete Adjoint Matching)
  core.py                 gKL loss, adjoint estimator, control box
  discrete.py             the three benchmark adapters + CLI
  tests_math.py           gradient identity, path-weight identity, estimator
  run_dam.sh              every DAM leg reported in section 6.2
  run_dam_a2.sh           the two Appendix A.2 legs of section 2.6 (toy, GB1 annealed)
  run_occs128_K64.sh      occupation m=128 leg   (section 6.2.1)

rasbs/                    THE BASELINE, kept apart from our code
  rasbs_port.py           faithful PyTorch port of R-ASBS alg2_stiefel.m
  rasbs_sphere_port.py    port of asbs_sphere_sampler.m + earthquake_sphere_ex.m
  rasbs_sphere_audit.py   closed-form falsification tests for that port
  rasbs_steps.sh          their beta = 2 step ablation
  rasbs_steps_highbeta.sh their beta = 50/100 step ablation
  regen_f64.sh            float64 rerun at beta = 2, for the fig-7 residual panel
  run_fidelity_audit.sh   --init matlab audits + the five repaired seeds
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

**Repro:** `bash structured_asbs/scripts/rerun_ckpt.sh` (its Ising legs) -> `json/results_ising_{poisson,poisson256,mse}.json` + `ckpt/ising_{poisson,poisson256,mse}.pt`; `python structured_asbs/fixed_ising.py exact` -> `json/results_ising_exact.json` (closed form, no checkpoint); `python structured_asbs/remeasure.py ising` re-derives every table above from `ckpt/ising_poisson256.pt`.

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

**Repro:** `structured_asbs/fixed_ising.py train --L 5 ...` and `... exact` (the block above) -> `json/results_ising_t1_L5.json`, `json/results_ising_t1_L5_s512.json`, `json/results_ising_t1_L5_exact.json` + `ckpt/ising_t1_L5.pt`, `ckpt/ising_t1_L5_s512.pt`.

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

**Repro:** `bash structured_asbs/scripts/rerun_ckpt.sh` (its occupation legs) -> `json/results_occ_{full,uniform,occupancy,mse}.json` + `ckpt/occ4_{full,uniform,occupancy,mse}.pt`; `bash structured_asbs/scripts/run_occ.sh` adds the 1-sample estimator ablation `json/results_occ_{occ1,unif1}.json`.

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

**Repro:** `bash structured_asbs/scripts/run_scale.sh` -> `json/results_occ_s{32,128,1000}.json` plus the variance probes `json/results_occ_var{,128,1000}.json`; `rerun_ckpt.sh` runs the same three legs with `--ckpt-dir ckpt` and writes `ckpt/occ_s{32,128,1000}.pt`.

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

**Repro:** `bash structured_asbs/scripts/rerun_sphere.sh` -> `json/results_sphere_exact.json`, `json/results_sphere_train_{plain,anti,sym}.json` + `ckpt/sphere_{plain,anti,sym}_seed{0..4}.pt`; `python structured_asbs/remeasure.py sphere` re-derives every table above from those checkpoints.

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

**Repro:** `structured_asbs/earthquake.py train` then `verify` (the block above) -> `json/results_earthquake_k600.json` and `json/results_earthquake.json` + `ckpt/earthquake_k600.pt`. The R-ASBS column is `python rasbs/rasbs_sphere_port.py --problem quake` -> `json/results_rasbs_sphere_quake.json` + `ckpt/rasbs_sphere_quake.pt`.

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

**Repro:** `bash structured_asbs/scripts/run_scalefix.sh` (beta = 50, 100 at 199 steps), `run_anneal.sh` (50 -> 100 anneal), `run_anneal_chain.sh` (50 -> 65 -> 80 -> 100) and `run_anneal_b100_fine.sh` (796 steps) -> `json/results_stiefel_{fill,scalefix,anneal_b50,anneal_b100,anneal_b100_fine}.json` and `json/results_chain_b{65,80,100}.json` + `ckpt/stiefel_{fill,scalefix,anneal,anneal_fine}_b*_seed0.pt`, `ckpt/chain_b*_seed0.pt` (each with its `_mcmc.pt` reference). The R-ASBS column is `python rasbs/rasbs_port.py` -> `json/results_rasbs_stiefel.json` + `ckpt/rasbs_b*.pt`.

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

**Repro:** `python structured_asbs/_stiefel_extra.py` -> `json/results_stiefel_extra.json`. No training: it scores `ckpt/rasbs_b{1.3,2,5}.pt` against the same MCMC references `ckpt/stiefel_grid_b*_mcmc.pt` that IASBS is scored against.

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

**Repro:** `bash rasbs/rasbs_steps.sh` and `bash rasbs/rasbs_steps_highbeta.sh` -> `json/results_rasbs_steps_{32,64,128,256,512}.json`, `json/results_rasbs_highbeta_steps_{199,512,1024}.json` + `ckpt/rasbs_*.pt`; our row is `python structured_asbs/stiefel.py sweep` -> `json/results_stiefel_sweep.json` + `ckpt/stiefel_d3_sweep_steps*.pt`.

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

**Repro:** `python structured_asbs/figures.py` draws every figure and the comparison table from the committed `json/` and `ckpt/` files; `python structured_asbs/gallery.py` draws the sample galleries. The float64 orthogonality panel needs `bash rasbs/regen_f64.sh` -> `json/results_rasbs_b2_f64.json`, `json/results_stiefel_b2_f64.json` + float64 `ckpt/rasbs_b2.pt`, `ckpt/stiefel_grid_b2_seed0.pt`.

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
| 6 | DAM is fragile and expensive: it needs a per-problem control box sized against an optimum one does not know in advance, an adjoint-label truncation, and a gradient budget that grows with the state space | §6.2 — TV 0.0617 on Ising L=4 for 169 M f1 evals and 975 M simulated jumps (plateau confirmed at 7000 iterations, TV 0.0634); KS_occ 0.0128 on occupation-scale m=32 for 5.85 B jumps; both 1.8-3x worse than IASBS at zero adjoint rollouts, and at m=128 the adjoint estimator collapses outright (ESS 1.69 / 64, KS_occ 0.1755 vs IASBS 0.0127, §6.2.1) |

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

**Repro:** `structured_asbs/fixed_ising.py train-nondirac --L 4` and `--L 5` (see §7) -> `json/results_ising_nd_L4.json`, `json/results_ising_nd_L5.json` + `ckpt/ising_nd_L4.pt`, `ckpt/ising_nd_L5.pt`.

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

**Repro:** `structured_asbs/occupation.py train-nondirac --m 4` (and `--nu-skew` for the skewed source) -> `json/results_occ_nd_m4.json`, `json/results_occ_nd_m4_skew.json` + `ckpt/occ_nd_m4.pt`, `ckpt/occ_nd_m4_skew.pt`.

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

**Repro:** `structured_asbs/occupation.py scale-nondirac --m {32,128,1000}` (see §7 for the m=1000 flags) -> `json/results_occ_nd_s{32_seed0,32_seed1,128,1000}.json` + `ckpt/occ_nd_s{32_seed0,32_seed1,128,1000}.pt`.

---

#### 2.4 Fixed-support heterogeneous Potts space (Appendix A.2) — toy, validated against the exact 81,081-state law

The state space is the Appendix A.2 fixed-support set
`X^(r)_{n,k} = {x in {0..r-1}^n : #{i : x_i != 0} = k}` at `n = 14`, `k = 4`,
`r = 4`, so `|X| = C(14,4) * 3^4 = 81,081`. The target is a heterogeneous
nonsymmetric Potts ring (128 energy edges: 8 label + 120 support), which
deliberately breaks the permutation/label symmetry of the reference construction;
the nonbinary-Johnson orbit kernel (`C = 15` orbits, `A = 770` global actions)
still makes the terminal matching labels exactly computable, so the learned
controlled law is compared to the target in exact total variation over all
81,081 states. Both runs use `steps = 256`, `iters = 3000`, `batch = 2048`,
`buffer = 8`, `inner = 40`, `mb = 1024`, `hidden = 512`, `lr = 3e-4`, Poisson
loss, `seed = 0`, `n_samples = 20000`. Gates as in §2.1: **A1** exact-law
TV <= 0.05, **A2** zero support-cardinality violations.

| source | params | exact-law TV | KL | Hellinger | A1 | A2 | wall |
|---|---|---|---|---|---|---|---|
| Dirac(x0) | 882,806 | **0.00774** | 0.000202 | 0.00709 | **PASS** | **PASS** (0 / 20,000) | 3,050 s |
| 4-atom mixture [0, 27, 81, 82] | 882,806 + 344,066 | **0.00776** | 0.000201 | 0.00708 | **PASS** | **PASS** (0 / 20,000) | 3,460 s |

Both land at essentially the same TV, ~6.5x below the gate. The non-Dirac run
carries an extra 344,066-parameter corrector (`corrector_hidden = 256`,
`corrector_actions = 32`); its Radon-Nikodym head is accurate to
RMSE 0.00705, MAE 0.00336, max abs 0.1169 over 100,000 probes — i.e. the
non-Dirac machinery costs 13% more wall time and buys back the same accuracy,
matching the pattern already seen on the sphere (§3.2).

Empirical TV over 20,000 samples is 0.52736 (Dirac) and 0.52521 (non-Dirac)
against i.i.d. floors of 0.52772 and 0.52162 — at `|X| = 81,081` with 20,000
draws the empirical statistic is dominated by sampling noise, which is exactly
why the exact-law TV is the gate. Mean energy `-2.3617` (Dirac) and `-2.3572`
(non-Dirac) vs target `-2.3753`; the modal state (index 78,344) is ranked first
by both, at `p = 0.000908 / 0.000910` vs `pi = 0.000935`. Terminal-law mass
error is at machine precision (7e-16, 2e-16).

| it | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| TV, Dirac | 0.02040 | 0.01881 | 0.01767 | 0.01145 | 0.01195 | **0.00774** |
| TV, non-Dirac | 0.02157 | — | 0.01394 | 0.01918 | 0.01031 | **0.00776** |

**Repro:** `structured_asbs/fixed_support.py train --target toy` and
`structured_asbs/fixed_support.py train-nondirac --target toy` (flags above) ->
`json/results_fs_toy_v2_dirac.json`, `json/results_fs_toy_v2_nd.json` +
`ckpt/fs_toy_v2_{dirac,nd}.pt`.

---

#### 2.5 Fixed-support GB1 protein-fitness landscape (Appendix A.2) — exact three-mutation sector, 27,436 states

Same `X^(r)_{n,k}` machinery as §2.4 with only the target adapter changed: the
four-site GB1 landscape of Wu et al. (eLife 2016;5:e16965, `10.7554/eLife.16965`)
restricted to the exact Hamming-distance-3 sector from wild type `VDGV`, i.e.
`n = 4`, `k = 3`, `r = 20` so `|X| = C(4,3) * 19^3 = 27,436`. The target is the
complete **measured + author-imputed** landscape (149,361 measured + 10,639
imputed = 160,000 variants), Boltzmann-weighted on `E = -log(F + 1e-4)` at
`tau = 1`, `gamma = 10`: E_min -1.8366, E_max 9.2103, E_mean 4.9582,
E_std 2.5893, pi_max 0.0015412, entropy 8.2118. Orbit count `C = 7`,
`A = 2,748` global actions, `alpha = 4.8649`, `beta = 5.1351`, 111 energy edges
(54 label + 57 support). Both runs: `steps = 256`, `iters = 3000`,
`batch = 2048`, `buffer = 8`, `inner = 40`, `mb = 1024`, `hidden = 512`,
`lr = 3e-4`, Poisson loss, `seed = 0`, `eval_every = 250`, `n_samples = 20000`.

| source | params | exact-law TV | KL | Hellinger | A1 (<= 0.05) | A2 | wall |
|---|---|---|---|---|---|---|---|
| Dirac(x0) | 766,844 | 0.21465 (best 0.20672 at it 2750) | 0.16552 | 0.21871 | **FAIL** | **PASS** (0 / 20,000) | 2,990 s |
| four-atom mixture | 766,844 + 858,556 | **0.19501** (best 0.18426 at it 2750) | 0.14681 | 0.20513 | **FAIL** | **PASS** (0 / 20,000) | 3,340 s |

**This row does not pass A1 at 3000 iterations, and the reason is visible in the
trajectory, not in the constraint machinery.** The A.2 group action is exact —
zero support violations out of 20,000 samples on both runs, and terminal-law
mass error 8.9e-16 (Dirac) and 0.0 (non-Dirac). What is not converged is the
control: TV is still falling at cutoff, 0.41124 -> 0.21465 over 3000 iterations
with no plateau.

| it | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 | 2500 | 2750 | 3000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TV, Dirac | 0.41124 | 0.37348 | 0.30884 | 0.29591 | 0.28265 | 0.26552 | 0.22242 | 0.22909 | 0.22215 | 0.20718 | 0.20672 | 0.21465 |
| TV, non-Dirac | 0.42944 | 0.35118 | 0.32679 | 0.27711 | 0.29705 | 0.28746 | 0.23627 | 0.22129 | 0.21286 | 0.19755 | 0.18426 | 0.19501 |

GB1 is harder than the toy at a third of the state count: the toy is a smooth
Potts ring (entropy 10.2658 out of `log 81081 = 11.30`), whereas the GB1
Boltzmann law is an empirical fitness landscape with `E_std 2.59` and entropy
8.2118 out of `log 27436 = 10.22` — a much more peaked target that the same
3000-iteration budget does not resolve. The empirical statistic is
uninformative here for the same reason as in §2.4: 0.29349 / 0.28759 over 20,000
draws against i.i.d. floors of 0.21491 / 0.20909.

Physical agreement is already reasonable: mean fitness 1.9798 (Dirac) and 1.9527
(non-Dirac) vs target 1.8350; mean energy -0.2347 / -0.1879 vs target -0.0999.
Both runs rank the same modal variant first (state index 6,464), but overweight
it 2x, `p = 0.003087 / 0.003004` vs `pi = 0.0015412` — i.e. the residual TV is
concentrated on the peak, which is what an unconverged control on a peaked
target looks like. The non-Dirac corrector reaches RMSE 0.02709, MAE 0.00960,
max abs 0.61340 over 100,000 probes — 3.8x looser than the toy's 0.00705, again
consistent with a target the budget has not resolved.

##### 2.5.1 What the residual gap is not

Two diagnostics rule out the cheap explanations before any schedule change.

**Not a time-discretisation floor.** Re-propagating the trained flat-tau=1
Dirac control on finer grids gives TV 0.21465 at 256 steps, 0.21759 at 512 and
0.21909 at 1024 — finer grids are marginally *worse* (the control was fitted at
256), so the gap is not the Euler grid.

**Not a reference-mixing floor.** The reference chain at `gamma = 10` is already
fully mixed at `t = 1`: propagating it with the control switched off gives
TV 0.77976 against the target, and the distance from the exactly uniform law to
the target is 0.78220. Raising gamma cannot help — 20, 40 and 80 all sit on the
0.78220 uniform floor to five decimals. So the 0.195-0.215 above is a genuine
75% reduction of an 0.78 starting distance, and the remaining gap is the
optimiser's, not the discretisation's or the reference's.

The one reference-side knob that *would* matter, tilting the base rates toward
the target's per-site amino-acid marginals, is not admissible here: it breaks
the permutation-equivariance of the reference and with it the exactness of the
A.2 orbit-kernel propagation that makes this row's TV computable at all.

##### 2.5.2 Temperature-annealed schedule — the GB1 default

What does move the number is the optimiser's path. Because the target enters
only as `exp(-E/tau)`, one can train through a flatter surrogate and finish on
the real one. This is now the **default schedule for the GB1 target**, at the
*same* 3000-iteration budget as the flat run above, spent as three warm-started
stages of 1000 iterations at `tau = 2.0`, `1.4`, `1.0`. Only the final stage is
reported, and it is evaluated at `tau = 1` against the unmodified Wu et al.
Boltzmann law, so the benchmark is untouched — the flat and annealed rows differ
in optimiser trajectory only, and the matched budget keeps the DAM head-to-head
honest. The same schedule is available to DAM.

| source | flat tau=1, 3000 it | annealed, 3000 it | change | A1 | A2 |
|---|---|---|---|---|---|
| Dirac(x0) | 0.21465 | **0.13568** | **-37%** | FAIL | **PASS** (0 / 20,000) |
| four-atom mixture | 0.19501 | **0.13226** | **-32%** | FAIL | **PASS** (0 / 20,000) |

Per-stage exact TV, each stage measured against its own `tau`:

| stage | tau | iters | Dirac TV | non-Dirac TV |
|---|---|---|---|---|
| A | 2.0 | 1000 | 0.10566 | 0.10695 |
| B | 1.4 | 1000 | 0.11564 | 0.10703 |
| C | **1.0** | 1000 | **0.13568** | **0.13226** |

Each `tau` step costs a transient: stage C opens at 0.18333 (Dirac) and 0.15945
(non-Dirac) as the network absorbs the sharper target, then recovers past its
own starting point within 750 iterations. The annealed runs also improve every
secondary quantity — KL 0.08132 / 0.07650 (flat: 0.16552 / 0.14681), Hellinger
0.15205 / 0.14738 (flat: 0.21871 / 0.20513), mean fitness 1.85499 / 1.91098
against target 1.83499 (flat: 1.97982 / 1.95267), and the modal-variant
overweight at state 6,464 drops from 2.0x to 1.29x (Dirac, `p = 0.00199`) and
1.51x (non-Dirac, `p = 0.00233`) against `pi = 0.00154`. Terminal-law mass error
is exactly 0.0 on both. The non-Dirac corrector improves to RMSE 0.02353,
MAE 0.00899, max abs 0.63775.

**A1 still fails.** 0.132 is 2.6x the gate. The annealing buys a third of the
gap and every physical observable moves the right way, but GB1 is not a solved
row: it is reported as a hardness result, exactly like Ising L=5 in §2.1, with
the constraint machinery exact on both.

**Repro (annealed, the default):**
`bash structured_asbs/scripts/run_gb1_anneal.sh both` ->
`json/results_fs_gb1_k3_{dirac,nd}_{A,B,C}.json` +
`ckpt/fs_gb1_k3_{dirac,nd}_{A,B,C}.pt`; the reported row is stage `C`.

**Repro:** `structured_asbs/fixed_support.py train --target gb1` and
`structured_asbs/fixed_support.py train-nondirac --target gb1` (flags above,
plus `--gb1-measured data/gb1/elife-16965-supp1-v4.xlsx --gb1-imputed
data/gb1/elife-16965-supp2-v4.xlsx`) -> `json/results_fs_gb1_k3_dirac.json`,
`json/results_fs_gb1_k3_nd.json` + `ckpt/fs_gb1_k3_{dirac,nd}.pt`.

#### 2.6 DAM on the same two A.2 spaces — matched head-to-head

The Appendix A.2 rows above are the only place in this repo where DAM and IASBS
solve the *same* constrained sampling problem on a space large enough for the
comparison to mean something (81,081 and 27,436 states, both with an exactly
computable terminal law).  DAM was given the identical schedules: 1200 flat
`tau = 1` iterations on the toy, and the same temperature-annealed
`2.0 -> 1.4 -> 1.0` chain on GB1 that is the IASBS default of section 2.5.2
(3 x 400 iterations, warm started through `--init-from`).  Both methods hit the
same 20,000-sample constraint audit at the end.

| space | method | source | exact-law TV | A1 | A2 | iters | f1 evals at rollout endpoints | CTMC jumps simulated | wall (s) | s / it |
|---|---|---|---|---|---|---|---|---|---|---|
| toy, \|X\|=81,081 | **IASBS** | **Dirac** | **0.00774** | **PASS** | **PASS** (0 / 20,000) | 3000 | **0** | **0** | 3098 | 1.03 |
| toy | IASBS | four-atom | 0.00776 | PASS | PASS (0 / 20,000) | 3000 | 0 | 0 | 3294 | 1.10 |
| toy | DAM, K=16 | Dirac | 0.08285 | FAIL | PASS (0 / 20,000) | 1200 | 20,889,600 | 118,996,727 | 7508 | 6.26 |
| GB1 k=3, \|X\|=27,436 | **IASBS** | **Dirac** | **0.13568** | FAIL | **PASS** (0 / 20,000) | 3 x 1000 | **0** | **0** | 5814 | 1.94 |
| GB1 k=3 | IASBS | four-atom | 0.13226 | FAIL | PASS (0 / 20,000) | 3 x 1000 | 0 | 0 | 6079 | 2.03 |
| GB1 k=3 | DAM, K=16 | Dirac | 0.26472 | FAIL | PASS (0 / 20,000) | 3 x 400 | 20,889,600 | 127,340,484 | 7512 | 6.26 |

**Accuracy.** DAM is **10.7x** worse than Dirac IASBS on the toy (0.08285 vs
0.00774) and **1.95x** worse on GB1 (0.26472 vs 0.13568), and it misses the A1
gate on the toy space that IASBS clears with a 6.5x margin.  On GB1 neither
method reaches A1, so that row is a hardness statement about the target rather
than a separation - but the gap between them is still a factor of two.

**Cost.** The separation is not bought with compute.  DAM spends
**20.9 million** rollout-endpoint `f1` evaluations and **119 / 127 million**
simulated CTMC jumps on the adjoint on each space; IASBS performs **zero** of
either, because the intertwining identity gives `phi_t` in closed form on the
orbit quotient (C = 15 orbits for the toy, C = 7 for GB1).  Per wall-clock
second DAM is also 6x slower per iteration on the identical device, since each
rollout is a Python-level Gillespie `while` loop.  Both sides were run two legs
at a time on one A100, so the wall column is comparably loaded on both.

**Constraints.** This is the one thing the two methods agree on exactly: 0
support-cardinality violations in 20,000 samples for every run in the table, and
terminal-law mass error at machine precision (6.7e-16 toy, 2.2e-15 GB1 for DAM).
The A.2 machinery is a property of the state space and the reference chain, not
of the loss, so it transfers to DAM unchanged.

Secondary quantities on the toy: DAM KL 0.02942 and Hellinger 0.08532 against
IASBS 0.00020 and 0.00709; mean energy -2.29261 vs target -2.37526 (IASBS
-2.36170); modal-state mass 0.00088 vs `pi = 0.00093`.  DAM's ESS is healthy
throughout (13.15 / 16 mean, 9.58 p10, 0 clipped labels, 0 non-finite weights),
so the gap is the estimator's variance, not a stability failure.

Per-stage DAM on the annealed GB1 chain, each stage measured against its own
`tau` (compare the IASBS table in section 2.5.2):

| stage | tau | iters | DAM TV | IASBS Dirac TV | jumps | wall (s) |
|---|---|---|---|---|---|---|
| A | 2.0 | 400 | 0.25745 | 0.10566 | 41,657,667 | 2508 |
| B | 1.4 | 400 | 0.27040 | 0.11564 | 42,548,716 | 2511 |
| C | **1.0** | 400 | **0.26472** | **0.13568** | 43,134,101 | 2493 |

DAM's annealed chain is flat: it does not pick up the stage-A advantage that
IASBS gets from the softer target (0.257 vs 0.106), so warm starting through the
temperature ladder carries almost nothing forward for it.  DAM's GB1 mean fitness
is 1.45539 against the target's 1.83499, i.e. it *under*-weights the fit tail,
where annealed IASBS overshoots slightly (1.85499); DAM's modal-state overweight
is 1.18x (`p = 0.00144` vs `pi = 0.00122`).

**Repro:** `bash dam/run_dam_a2.sh both` ->
`json/results_dam_fs_toy_v2_K16.json`,
`json/results_dam_fs_gb1_k3_K16_{A,B,C}.json` +
`ckpt/dam_fs_toy_v2_K16.pt`, `ckpt/dam_fs_gb1_k3_K16_{A,B,C}.pt`; the reported
GB1 row is stage `C`.

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

**Repro:** `bash structured_asbs/scripts/rerun_sphere.sh` -> `json/results_sphere_train_{anti,sym,plain}.json` + `ckpt/sphere_{anti,sym,plain}_seed{0..4}.pt`.

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

**Repro:** `structured_asbs/sphere.py train-nondirac` (see §7) -> `json/results_sphere_nd.json` + `ckpt/sphere_nd_seed{0..4}.pt`; the plain-init control is `ckpt/sphere_nd_plain_seed0.pt`. The three extra statistics come from `python structured_asbs/remeasure.py sphere` reading those checkpoints.

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

**Repro:** `python rasbs/rasbs_sphere_port.py --problem bimodal --seed {0..4}` -> `json/results_rasbs_sphere_bimodal{,_s1,_s2,_s3,_s4}.json` + `ckpt/rasbs_sphere_bimodal*.pt`; the epoch controls are `json/results_rasbs_sphere_bimodal_e{100,300}.json`.

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

**Repro:** `bash rasbs/run_fidelity_audit.sh` -> `json/results_rasbs_sphere_matlabinit_s{0..4}.json` and the two closed-form audits `json/results_rasbs_audit_{uniform,vmf}_mi.json` + `ckpt/rasbs_sphere_matlabinit_s{0..4}.pt`, `ckpt/rasbs_audit_{uniform,vmf}_mi.pt`; `python structured_asbs/_rasbs_extra.py` adds KS(phi) and the second-moment error from those same checkpoints, without retraining.

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

**Repro:** no new run — the table scores `json/results_rasbs_sphere_matlabinit_s{0..4}.json` against `json/results_sphere_nd.json`.

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

**Repro:** `bash dam/run_dam.sh occ4` -> `json/results_dam_occupation_m4_K16.json` (400 it), `json/results_dam_occ4_K16_long.json` (1200 it), `json/results_dam_occ4_K64.json` + the identically-named `ckpt/dam_occ4_K16_long.pt`, `ckpt/dam_occ4_K64.pt`, `ckpt/dam_occupation_m4_K16.pt`.

#### 5.2 Diminishing returns above K=16

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

**Repro:** as §5.1 — `bash dam/run_dam.sh occ4`.

#### 5.3 Cost comparison, the headline number

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

**Repro:** IASBS side `json/results_occ_full.json` + `ckpt/occ4_full.pt` (`bash structured_asbs/scripts/rerun_ckpt.sh`); DAM side `bash dam/run_dam.sh occ4`.

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

**Repro:** no separate run — every wall-clock figure is the `wall`/`time` field of the json artifact named in the section that reports the corresponding accuracy number.

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

**Repro:** `python rasbs/rasbs_sphere_port.py` and `bash rasbs/run_fidelity_audit.sh`; timings are stored in `json/results_rasbs_sphere_*.json`.

#### 6.2 DAM

All DAM numbers below use the same control box and adjoint-label truncation,
exposed as four flags in `dam/core.py` and `dam/discrete.py`.  They are off by
default, so the K-sweep of §5.1 reproduces bit-for-bit (`dam.tests_math`:
11 / 11 gates still pass).

| flag | what it bounds | value used |
|---|---|---|
| `--a-clamp C` | `\|a_theta\| <= C`, i.e. the escape-rate multiplier `<= e^C` | 8 |
| `--m-clip C` | `\|log m_hat\| <= C` in the gKL loss | 5 |
| `--ess-min E` | drop labels whose denominator ESS `< E`; the mean is over survivors | 3 |
| `--coef-cap R` | winsorise the `r/q` prefactor at `R x` its own batch median | 10 |

The clamp has to contain the optimum.  Reading the exact control multiplier off
`FI.ExactControl` shows it growing sharply towards `t = 1`:

| t | 0.000 | 0.250 | 0.500 | 0.750 | 0.938 | 0.992 |
|---|---|---|---|---|---|---|
| `max \|a_exact\|` | 0.185 | 0.576 | 1.431 | 2.629 | 4.205 | **6.308** |
| `q99 \|a_exact\|` | 0.114 | 0.277 | 0.687 | 1.573 | 3.062 | 4.861 |

so `\|a\| <= 8` leaves 27% headroom over the 6.308 optimum.  The number is
problem-specific, and on `ScaleOccupation` there is no `ExactControl` to read it
off — which is the practical difficulty claim 6 asserts.

**Cost.**

| experiment | K | iters | wall (s) | s / iter | f1 evals | CTMC jumps | jumps / f1 eval |
|---|---|---|---|---|---|---|---|
| Occupation m=4 | 16 | 400 | 7,380.9 | 18.5 | 6,963,200 | 67,417,741 | 9.68 |
| Occupation m=4 | 16 | 1200 | **11,529.3** | 9.6 | **20,889,600** | **185,004,804** | 8.86 |
| Occupation m=4 | 64 | 400 | 3,177.7 | 7.9 | 26,624,000 | 222,041,777 | 8.34 |
| Ising L=4 | 32 | 5000 | 11,428 | 2.29 | 168,960,000 | 974,800,000 | 5.77 |
| Ising L=4 | 64 | 2500 | 6,950 | 2.78 | 166,400,000 | 924,000,000 | 5.55 |
| Ising L=4 | 32 | 7000 | 16,226 | 2.32 | 236,544,000 | 1,339,809,998 | 5.66 |
| Occupation-scale m=32 | 64 | 1000 | 16,572 | 16.6 | 66,560,000 | 5,849,804,432 | 87.9 |
| Occupation-scale m=128 | 64 | 500 | 67,693 | 135.4 | 33,280,000 | 12,889,415,391 | 387.3 |
| Ising L=5 | 32 | 5000 | 15,391 | 3.08 | 168,960,000 | 1,372,615,519 | 8.12 |

**Accuracy.**  Constraint violations are 0 on every row.

| experiment | K | iters | headline | gate | ESS end | ESS mean / p10 |
|---|---|---|---|---|---|---|
| Occupation m=4 | 16 | 400 | TV 0.07170 | FAIL | — | 9.62 / 4.26 |
| Occupation m=4 | 16 | 1200 | TV **0.01490** (occ-hist 0.00800) | **PASS** | 14.04 / 16 | — / 8.07 |
| Occupation m=4 | 64 | 400 | TV 0.04498 (occ-hist 0.01827) | **PASS** | 48.75 / 64 | — / 16.71 |
| Ising L=4 | 32 | 5000 | TV **0.06168** (E-hist 0.04759) | FAIL | 30.47 / 32 | 23.70 / 7.43 |
| Ising L=4 | 64 | 2500 | TV 0.08103 (E-hist 0.06567) | FAIL | 58.62 / 64 | 42.47 / 8.36 |
| Ising L=4 | 32 | 7000 | TV 0.06339 (E-hist 0.05206) | FAIL | 30.64 / 32 | 24.91 / 11.40 |
| Occupation-scale m=32 | 64 | 1000 | KS_occ **0.0128** (KS_max 0.0399, W1max/N 0.0068) | **PASS** | 55.58 / 64 | 31.39 / 1.10 |
| Occupation-scale m=128 | 64 | 500 | KS_occ 0.1755 (KS_max 0.9014, W1max/N 0.0382) | FAIL | 1.73 / 64 | 1.69 / 1.00 |
| Ising L=5 | 32 | 5000 | TV 0.39956 (E-hist 0.19000) | FAIL | 13.39 / 32 | 8.53 / 2.01 |

IASBS on the same targets, for reference: occupation m=4 TV **0.0117**
(Dirac **0.012730**), Ising L=4 TV **0.03470**, Ising L=5 TV **0.07228**,
occupation-scale m=32 KS_occ **0.0043** (KS_max 0.0305, W1max/N 0.0052).
On occupation-scale m=128 IASBS reaches KS_occ **0.01270** (W1max/N 0.00224,
0 violations) in 1,523 s (§2.3).

**Ising L=4: DAM plateaus at TV ~0.062 and never reaches the TV <= 0.05 gate.**
The 7000-iteration run is healthy throughout — loss positive, ESS at 96%
retention and still creeping up — and still flattens:

| it | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 | 7000 |
|---|---|---|---|---|---|---|---|
| TV | 0.08227 | 0.08055 | 0.06934 | 0.06731 | 0.06784 | 0.06193 | **0.06339** |
| E-hist TV | 0.05425 | 0.05636 | 0.05033 | 0.05087 | 0.05172 | 0.04901 | 0.05206 |
| ESS / 32 | 28.86 | 29.47 | 29.85 | 30.04 | 30.56 | 30.63 | 30.64 |

From iteration 5000 onward TV oscillates in **0.062-0.069** with no trend; the
last 2000 iterations move it by 0.006, well inside the eval-to-eval noise of
±0.005, and cost 67.6 M additional f1 evaluations.  The 5000-iteration K=32 run
is, within noise, the same number.  This is a floor, not a transient: DAM lands
at TV 0.062 against IASBS's **0.03470**, a factor of **1.8**.

**The K axis saturates around K=32.**  At 2500 iterations K=64 spends essentially
the same terminal-evaluation budget as K=32 at 5000 (166.4 M vs 169.0 M) and
lands *worse*, 0.081 against 0.062.  Retention at K=32 is already 95%
(30.47 of 32), so the extra 32 rollouts per label buy nothing that was missing.

**Occupation-scale m=32 is the strongest DAM result here.**  It clears its
`KS_occ <= 0.05` gate, and it does so on the benchmark where the clamp could
*not* be sized against a known optimum:

| it | 25 | 125 | 225 | 325 | 425 | 625 | 825 | 1000 |
|---|---|---|---|---|---|---|---|---|
| loss | +134.4 | -43.8 | -1029.9 | +163.4 | +121.5 | +118.3 | +118.7 | +121.0 |
| KS_occ | 0.2024 | 0.1865 | 0.0395 | 0.0200 | 0.0138 | 0.0139 | 0.0105 | **0.0128** |
| ESS / 64 | 3.03 | 2.45 | 2.70 | 9.88 | 27.62 | 48.80 | 53.42 | **55.58** |

Against the uncontrolled reference of KS_occ 0.2040, KS_max 0.8024, W1max/N
0.1007, it ends at **0.0128 / 0.0399 / 0.0068** — a 16x reduction on KS_occ, 0
violations, 87% rollout retention.  Two qualifications keep this honest.  It is
still **3x worse than IASBS**, which reaches KS_occ **0.0043** on the same
target.  And the cost is not comparable: **5.85 billion simulated jumps** and
66.6 M f1 evaluations over 4.6 h, against zero adjoint rollouts for IASBS.  The
ESS 10th percentile of **1.10** also says the healthy mean of 31.4 hides a tail
of batches carried by a single rollout.

**Ising L=5 is where the gap becomes an order of magnitude.** The 404x larger
constraint set (`|Omega| = 5,200,300` against 12,870) gets the identical control
box and 5000 iterations, which is what L=4 needed to reach its floor. It
converges — loss positive from iteration ~3250, ESS rising monotonically
6.91 -> **13.39 / 32**, 0 violations — but only to:

| it | 3250 | 3500 | 4000 | 4250 | 4500 | 4750 | 5000 |
|---|---|---|---|---|---|---|---|
| TV | 0.49883 | 0.46154 | 0.41855 | 0.41352 | 0.41032 | 0.40511 | **0.39956** |
| E-hist TV | 0.28391 | 0.23797 | 0.19383 | 0.21109 | 0.19141 | 0.19710 | 0.19000 |
| ESS / 32 | 10.18 | 11.18 | 11.64 | 12.78 | 13.35 | 12.60 | 13.39 |

TV **0.39956** against IASBS's **0.07228** on the same target (§2.1) — a factor
of **5.5** — for 169 M f1 evaluations, 1.37 billion simulated jumps and 4.3 h.
Unlike L=4 this one has *not* flattened: it is still falling ~0.005 per 250
iterations at the cutoff, so a larger budget would improve it. That is the point.
The budget L=4 needed to reach 0.062 leaves L=5 at 0.400, and the 32x-per-space
scaling of the gradient budget that would close it is a cost IASBS does not pay:
IASBS reaches 0.07228 at L=5 in 3000 iterations with zero adjoint rollouts.

**Budget is the dominant axis, and it scales with the state space.**  Occupation
m=4 (`|X| = 35`) converges in 1200 iterations at K=16.  Ising L=4
(`|Omega| = 12,870`, 368x larger) needs 3000 at K=16 just to leave its transient
— the loss is negative until iteration ~700 and TV *rises* to 0.870 at iteration
200 before turning around — and 5000 at K=32 to reach its floor.  Any read taken
inside that transient is uninformative, which is why every leg above is run to a
flat tail.

**What claim 6 rests on.**  Three things have to hold simultaneously, and only
one of them is a knob the method exposes as such:

1. **a bounded control box that still contains the optimum.**  At `|a| <= 3` the
   box excludes the exact optimum of 6.31 and caps accuracy; `|a| <= 8` does not.
   Sizing it requires knowing an optimum one does not have in advance.
2. **a truncated adjoint label.**  `|log m_hat| <= 5` in place of the default 30;
   at 30 a single label can carry a weight of 1e13 into the gradient.
3. **enough gradient steps.**  3000 at K=16 on Ising L=4, against the 1200 that
   a state space 368x smaller needed.

With all three, DAM on Ising L=4 reaches **TV 0.06168 at ESS 30.47/32** — for
**237 million** rollout-endpoint evaluations and 1.34 billion simulated jumps, a
TV that IASBS beats by **1.8x** (0.03470, §2.1) with **zero** adjoint rollouts,
in 16,226 s against IASBS's 1,869 s.  On occupation-scale m=32 the same box
transfers and passes its gate at KS_occ 0.0128, still 3x behind IASBS's 0.0043
and at 5.85 billion jumps.  So the claim is not that DAM fails; it is that DAM
is **fragile and expensive** — a per-problem control box sized against an unknown
optimum, a truncation constant, and several orders of magnitude more sampling
work for accuracy that remains 1.8-3x behind.

DAM runs are **latency-bound, not throughput-bound**: measured CPU time equals wall
time to within 1% on every leg, because each Gillespie rollout is a Python `while`
loop issuing many small kernels and synchronising on `active.any()`. Consequences:

- Batch size is nearly free; the number of `while` iterations is what costs.
- Co-scheduling two DAM runs on one device roughly *doubles* both wall times rather
  than overlapping them, which is why `dam/run_dam.sh` queues the legs serially.
- The jumps-per-f1-eval column is the cost multiplier that makes the larger
  spaces expensive: 8.9 on occupation m=4, **87.9** on occupation-scale m=32.

#### 6.2.1 Occupation-scale m=128 — where DAM stops working

The m=32 leg above is DAM's best row in this repo.  Pushing the same
configuration to m=128 (`|X|` = compositions of N=128 into m=128 parts, 128
steps, K=64, the identical §6.2 control box) breaks it:

| method | m | iters | KS_occ | KS_max | W1max / N | viol | ESS | f1 evals | simulated jumps | wall (s) | s / it |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **IASBS** | **128** | 3000 | **0.01270** | — | **0.00224** | **0** | — | **0** | **0** | **1,523** | **0.51** |
| DAM, K=64 | 128 | 500 | 0.17554 | 0.9014 | 0.03820 | 0 | 1.69 / 64 | 33,280,000 | 12,889,415,391 | 67,693 | 135.4 |
| uncontrolled reference | 128 | — | 0.20529 | 0.8986 | 0.03997 | 0 | — | 0 | 0 | — | — |

DAM ends **13.8x** worse than IASBS on the gate quantity and, more tellingly,
only **1.17x** better than doing nothing at all: the uncontrolled reference chain
already sits at KS_occ 0.20529, so 18.8 h of training and **12.9 billion**
simulated CTMC jumps buy a 14% reduction where IASBS buys 16.2x in 25 minutes at
zero adjoint rollouts.

**The mechanism is estimator collapse, not a bad optimum.**  ESS is
**1.69 of 64** (p10 exactly 1.00), i.e. essentially every adjoint label is
carried by a single rollout out of 64, and **451,665** labels hit the
`LOG_M_CLIP` truncation.  On m=32 the same box held ESS at 31.39/64 with 0
clipping.  The importance ratio that DAM's `m_hat` is built from has a right tail
that widens with the state space, and at m=128 the truncation that keeps the run
numerically alive is also what destroys the gradient signal.  The trajectory is
consistent with that reading: KS_occ moves 0.2071 (it 25) -> 0.2086 (125) ->
0.2027 (250) -> 0.1834 (400) -> **0.1731** (475) -> 0.1755 (500), a slow crawl
that is still 3.5x above the gate after 500 iterations and has an eval-to-eval
noise of the same order as its per-100-iteration progress.

Energy agreement makes the same point: `E_ours = 65.89` against
`E_exact = 52.11` (26% high; the uncontrolled reference is 67.86, so DAM closed
13% of a 30% error), while the E-histogram KS is 0.9983 — the controlled law and
the target overlap almost nowhere.  Constraints, as everywhere in this repo, are
exact: 0 violations in 10,000 samples, because they are a property of the state
space and the reference chain rather than of the loss.

This is the honest ceiling for the baseline.  DAM is not broken by a tuning
mistake here; it is broken by the variance of the Monte-Carlo adjoint on a large
constrained space, which is precisely the quantity the intertwining identity
removes.

**Repro:** `bash dam/run_occs128_K64.sh` -> `json/results_dam_occs128_K64_500.json`
+ `ckpt/dam_occs128_K64_500.pt`.

**Repro:** `bash dam/run_dam.sh` runs all eight legs above serially (~23 h on one A100); `bash dam/run_dam.sh occ4` | `ising` | `occs32` | `isingL5` runs one group. Artifacts: `json/results_dam_{occupation_m4_K16,occ4_K16_long,occ4_K64,ising_L4_K32_5000,ising_L4_K64_2500,ising_L4_K32_7000,occs32_K64_1000,ising_L5_K32_5000}.json` + the identically-named `ckpt/*.pt`. The exact-control table is read off `fixed_ising.ExactControl`, which needs no training.

#### 6.3 Cost of the comparison, head to head

Same benchmark (occupation m=4), same hardware, best setting of each method:

Both rows are the **Dirac** source, m=N=4, 128 steps (see the source-matching
note in §5.3).

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

**Repro:** IASBS side `json/results_occ_full.json` + `ckpt/occ4_full.pt`; DAM side `json/results_dam_occ4_K16_long.json` + `ckpt/dam_occ4_K16_long.pt` (`bash dam/run_dam.sh occ4`).

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
the O(1) importance-weight variance of the adjoint estimator. So the column measures
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

**Repro:** the counter is `stats["f1_evals"]` in `dam/core.py:estimate_log_adjoint`; `python -m dam.tests_math` checks the identity it accumulates.

---

### 7. Reproduction

Every result section above ends with a **Repro:** line naming the script that
made it, the `json/results_*.json` it wrote and the `ckpt/*.pt` it left behind.
The index below is the same information grouped by script.

| script | produces | checkpoints |
|---|---|---|
| `structured_asbs/scripts/rerun_ckpt.sh` | `results_ising_{poisson,poisson256,mse}`, `results_occ_{full,uniform,occupancy,mse}`, `results_occ_s{32,128,1000}` | `ising_*.pt`, `occ4_*.pt`, `occ_s*.pt` |
| `structured_asbs/scripts/run_occ.sh` | `results_occ_{occ1,unif1}` (1-sample estimator ablation) | — |
| `structured_asbs/scripts/run_scale.sh` | `results_occ_s{32,128,1000}`, `results_occ_var{128,1000}` | — |
| `structured_asbs/scripts/rerun_sphere.sh` | `results_sphere_exact`, `results_sphere_train_{plain,anti,sym}` | `sphere_{plain,anti,sym}_seed{0..4}.pt` |
| `structured_asbs/scripts/run_scalefix.sh` | `results_stiefel_{scalefix,fill}` | `stiefel_{scalefix,fill}_b*_seed0.pt` (+ `_mcmc.pt`) |
| `structured_asbs/scripts/run_anneal*.sh` | `results_stiefel_anneal_b{50,100,100_fine}`, `results_chain_b{65,80,100}` | `stiefel_anneal*_b*_seed0.pt`, `chain_b*_seed0.pt` |
| `structured_asbs/remeasure.py {ising,sphere}` | re-derives those two sections' tables | reads only |
| `structured_asbs/_stiefel_extra.py` | `results_stiefel_extra` | reads only |
| `structured_asbs/_rasbs_extra.py` | printed table in §4.3 | reads only |
| `structured_asbs/figures.py`, `gallery.py` | everything in `fig/` | reads only |
| `rasbs/rasbs_steps.sh`, `rasbs_steps_highbeta.sh` | `results_rasbs_steps_*`, `results_rasbs_highbeta_steps_*` | `rasbs_*.pt` |
| `rasbs/run_fidelity_audit.sh` | `results_rasbs_sphere_matlabinit_s{0..4}`, `results_rasbs_audit_{uniform,vmf}_mi` | matching `.pt` |
| `rasbs/regen_f64.sh` | `results_{rasbs_b2,stiefel_b2}_f64` (float64, for the fig-7 residual panel) | float64 `rasbs_b2.pt`, `stiefel_grid_b2_seed0.pt` |
| `dam/run_dam.sh` | all seven §6.2 legs: `results_dam_*` | `ckpt/dam_*.pt`, same names |
| `dam/run_occs128_K64.sh` | the §6.2.1 leg: `results_dam_occs128_K64_500` | `ckpt/dam_occs128_K64_500.pt` |
| `dam/run_dam_a2.sh` | the two A.2 legs of §2.6: `results_dam_fs_*` | `ckpt/dam_fs_*.pt`, same names |

The runs that have no wrapper script are single commands:

```bash
# IASBS, Ising non-Dirac (§2.1) -> ckpt/ising_nd_L4.pt
python structured_asbs/fixed_ising.py train-nondirac --L 4 --steps 256 --iters 3000 \
    --n-samples 200000 --tag ising_nd_L4 --out json/results_ising_nd_L4.json

# IASBS, occupation non-Dirac scaling (§2.3) -> ckpt/occ_nd_s1000.pt
python structured_asbs/occupation.py scale-nondirac --m 1000 --steps 256 --iters 1500 \
    --mb 512 --n-samples 4000 --tag occ_nd_s1000 --out json/results_occ_nd_s1000.json

# IASBS, sphere non-Dirac (Haar) (§3.2) -> ckpt/sphere_nd_seed{0..4}.pt
python structured_asbs/sphere.py train-nondirac --steps 128 --iters 4000 --inner 16 \
    --inner-h 4 --batch 8192 --mb 16384 --mb-h 4096 --hidden 256 --lr 1e-3 \
    --ema 0.9995 --antithetic --seeds 5 --eval-every 200 --n-samples 200000 \
    --tag sphere_nd --out json/results_sphere_nd.json

# IASBS, Stiefel step ablation (§"The step-count ablation") -> ckpt/stiefel_d3_sweep_steps*.pt
python structured_asbs/stiefel.py sweep --out json/results_stiefel_sweep.json

# IASBS, earthquakes (§Earthquakes) -> ckpt/earthquake_k600.pt
python structured_asbs/earthquake.py train --tag earthquake_k600 \
    --out json/results_earthquake_k600.json

# R-ASBS Stiefel baseline (§Stiefel) -> ckpt/rasbs_b*.pt
python rasbs/rasbs_port.py --out json/results_rasbs_stiefel.json

# R-ASBS sphere baseline (§4.1) -> ckpt/rasbs_sphere_bimodal*.pt
python rasbs/rasbs_sphere_port.py --problem bimodal --seed 0 \
    --tag rasbs_sphere_bimodal --out json/results_rasbs_sphere_bimodal.json

# Correctness tests -- no artifacts, no GPU time worth mentioning
python structured_asbs/tests_math.py     # 27 gates, incl. the Appendix A.2 space
python -m dam.tests_math                 # 11 gates for the DAM baseline
```

Each `json/results_*.json` carries the full `config` block it was produced with,
so any flag not spelled out above can be read back off the artifact itself.

---

### 8. Still running / still to run

**Running:** nothing. Every job listed in previous revisions of this section has finished.

**Not run, and why**

| job | reason |
|---|---|
| DAM occupation-scale m=32, K=256 | the K axis saturates around K=32-64 (§6.2), so more rollouts per label is not where accuracy comes from |
| DAM occupation-scale m=1000 | per-iteration cost scales superlinearly in m — 16.6 s at m=32 and 131 s at m=128, both at K=64 — which puts a 200-iteration leg at m=1000 near 190 h on one A100. Not a useful spend |
| IASBS_600 Stiefel at beta = 0.1, 0.5, 7, 10, 20 | matched-budget ablation not performed at those temperatures; ~28 min per beta if wanted |

**Text only**

- README correction of the R-ASBS mode-collapse description (§4.2).

**Done since the last update.** DAM occupation-scale m=128, K=64, 500 iterations (§6.2.1): KS_occ 0.1755 after 18.8 h and 12.9 B simulated jumps, against IASBS's 0.01270 in 1,523 s -- the adjoint estimator collapses to ESS 1.69 / 64 with 451,665 clipped labels. Appendix A.2 fixed-support toy, both sources (§2.4): exact-law TV 0.00774 / 0.00776, A1 and A2 both PASS. DAM on the same two A.2 spaces (§2.6), on matched schedules: TV 0.08285 on the toy (A1 FAIL) and 0.26472 on annealed GB1, for 20.9 M rollout-endpoint f1 evals and 119 / 127 M simulated jumps each, against 0 for IASBS. Appendix A.2 GB1 k=3, both sources (§2.5): exact-law TV 0.21465 / 0.19501 at 3000 iterations, A1 FAIL and A2 PASS. The temperature-annealed schedule of §2.5.2, now the GB1 default at the same 3000-iteration budget, brings those to 0.13568 / 0.13226.

**One honest negative, recorded rather than buried.** IASBS Ising L=5 misses its
accuracy gate at TV 0.07228 (§2.1). Its A2 constraint gate passes, 0 / 200,000.
Fixed-support GB1 k=3 also misses A1, TV 0.13568 (Dirac) and 0.13226
(non-Dirac) under the annealed default, 0.21465 and 0.19501 flat, with the
constraint gate passing 0 / 20,000 on all four runs (§2.5).

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
