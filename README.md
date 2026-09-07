# Adjoint Schrödinger Bridge Sampling on Structured State Spaces via Markov Semigroup Intertwining

Experimental package for the paper. One standalone script per experiment, one
small shared utility file, no framework machinery — the layout deliberately
follows the R-ASBS repository's style.

Our method and the baseline live in separate directories, so that no reader has
to take on trust which code produced which column.

```
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
experiments concurrently), PyTorch 2.5.1, CUDA 12.6, NumPy 2.4.6, conda
environment `SML_env` (Python 3.11). Manifold state and all metrics are
`float64`; the networks are `float32`.

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

Checkpoints under `ckpt/` are not committed (1.4 GB); the `json/results_*.json`
files are, and every table here is built from them.

All commands below are run from the repository root.

```bash
python structured_asbs/tests_math.py   # mathematical unit tests
python structured_asbs/figures.py      # all figures -> fig/
python structured_asbs/gallery.py      # sample gallery -> fig/
```

`figures.py` plots metrics. `gallery.py` plots the samples themselves, as the
objects they actually are, beside the exact or MCMC reference drawn the same
way. The claim those panels support is "you cannot tell the two columns
apart", and that is a claim the eye should adjudicate. Each script also exposes
`verify` (correctness gates), `train`, and a sweep mode.

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
beside 50 drawn by exact enumeration; `fig2_discrete_correctness` carries the
metric panel.

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
python structured_asbs/fixed_ising.py verify         # gate A0
python structured_asbs/fixed_ising.py exact          # gate A0/A1/A2 with the exact control
python structured_asbs/fixed_ising.py train
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
| TV vs exact law | 0 | **0.0199** (iid floor 0.0167) |
| constraint violations | 0 | **0** |

As with the Ising lattice, the TV value sits essentially at the iid sampling
floor, so the residual is finite-sample error rather than sampler bias.

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
python structured_asbs/occupation.py train
python structured_asbs/occupation.py scale --m 1000 --N 1000
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
not better — ⟨E⟩ goes 1.267 (100 epochs) → 1.172 (300) → 1.040 (600), passing
through the exact 1.154 on its way past it. The sampler spends its training
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
are independent defects. `fig4_sphere` plots the comparison.

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
python structured_asbs/sphere.py train --antithetic
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
- **The exact-control baseline is missing at κ = 600**, so the failure is not
  yet attributed to the learned control as opposed to the integrator.
- **R-ASBS beats us on mode coverage here**, and we have not shown that our
  Dirac source is the reason — the argument above is a mechanism, not a
  measurement. The experiment that would settle it (rerun ours from a Haar
  source and see whether coverage rises to ≈0.94) has not been run.
- **The R-ASBS column is a port, not their binary.** MATLAB's Deep Learning
  Toolbox is not available here, so their script cannot be executed for a
  cross-check; see the porting audit in the Sphere section.
- **Still S².** Real data, but a 2-dimensional manifold.

```bash
python structured_asbs/earthquake.py verify                    # sampler + log Z identities
python structured_asbs/earthquake.py exact --kappa 20          # quadrature control, step sweep
python structured_asbs/earthquake.py train --kappa 600 --anneal 150 300 450 600 \
    --steps 512 --iters 6000 --tag earthquake_k600 \
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
| parameters | **139,528** | **140,560** (0.99× ours) |
| integration steps | 199 | 199 |
| iters × inner | 1500 × 8 | 1000 epochs |
| batch / minibatch | 2048 / 16384 | 600 |
| lr | 1e-3 | 1e-3 |
| EMA / buffer | 0.9995 / 4 | — |
| eval samples | 100,000 | 100,000 |
| reference | MCMC, 2e5 chains × 3000 sweeps | same |

### Results

`N` is the number of integration steps. β ≤ 20 is the shared 199-step grid both
methods were originally run on; at β = 50 and 100 each column is quoted at the
finest grid we ran for it, for the reason given in the ablation below. R-ASBS
column from `rasbs/rasbs_port.py` → `json/results_rasbs_stiefel.json`.

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
| 50 | 3.0418 | 3.2897 | +0.248 | 1024 | 3.0534 | **+0.010** | 1592 |
| 100 | 3.0279 | 3.2456 | +0.218 | 512 | 3.0496 | **+0.019** | 3184 |
| 200 | 3 (exact) | 3.2482 | +0.248 | 199 | not trained | — | — |
| 10⁶ | 3 (exact) | 3.1847 | +0.185 | 199 | not trained | — | — |

Worst finite-β error: **0.547** (theirs, β = 2) against **0.166** (ours, β = 1.3
at matched budget). Orthogonality residual: **3.8e-14** against 3.4e-07, because
the update is an exact geodesic step rather than a retraction. Two things the
curve alone does not show:

**The error changes sign at β ≈ 0.05.** At β → 0 the target *is* Haar, so their
source is exactly right and the −0.042 residual is pure QR discretisation. As β
grows the target leaves Haar, the source-tilting bias switches on and dominates,
peaking at +0.547 near β = 2. The correct initial law for the h-transform is
Haar·φ₀ / ⟨Haar, φ₀⟩ and φ₀ is not constant — the Dirac-vs-Haar distinction from
the top of this README, measured.

**The β → ∞ limit is never reached.** Their own paper states the limit is 3; the
rerun plateaus at 3.185 and stays there from β = 10³ to β = 10⁶.

**Budget matching.** PLAN §9.2 requires matching terminal energy/gradient
evaluations, and counting them exactly revealed that our default uses
**3,072,000** oracle calls against their **600,000**, a 5.1× advantage. The
budget-matched run (`--batch 400 --iters 1500`, exactly 600,000 calls, same 199
steps) gives +0.166 / +0.161 / +0.111 at β = 1.3 / 2 / 5 against their +0.451 /
+0.547 / +0.479 — removing the advantage costs 0.01–0.03 and changes no
conclusion. Parameters (0.99×), steps, and sample counts are matched throughout.
**Wall-clock**, single A100: ~1130 s per β at 199 steps, ~2180 s at 398, ~4380 s
at 796 for us; ~780 s per β at 512 steps and ~1560 s at 1024 for them. Every
`results_*.json` records `train_s` (ours) or `wall_s` (theirs).

### The step-count ablation

Our error is discretisation and falls with `N`; theirs is not, and does not.
PLAN §9.3 predicts exactly this but warns *do not claim such a floor before
measuring it*. Measured at β = 2, where their bias peaks, everything except the
step count held fixed:

R-ASBS row from `rasbs/rasbs_steps.sh` → `json/results_rasbs_steps_*.json`; our row
from `stiefel.py`'s `--sweep` → `json/results_stiefel_sweep.json`.

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
`json/results_rasbs_highbeta_steps_*.json`; our rows from
`json/results_stiefel_anneal_b{50,100}.json` and
`results_stiefel_anneal_b100_fine.json`.

| β | 199 | 398 | 512 | 796 | 1024 | 1592 | 3184 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| R-ASBS, 50 | +0.272 | — | +0.255 | — | **+0.248** | — | — |
| ours, 50 | +4.391 | +0.0685 | — | +0.0223 | — | **+0.0101** | — |
| R-ASBS, 100 | +0.235 | — | **+0.218** | — | +0.221 | — | — |
| ours, 100 | +5.290 | +4.063 | — | +3.535 | — | +0.750 | **+0.0187** |

Ours falls by 400× and 100× over that refinement; theirs moves by 0.024 and by
−0.003 (run-to-run noise on a quantity that has stopped moving). The refined
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
  (0.10–0.17 at 199 steps). Refining the same trained control drops β = 1 to
  0.050 at 1024 steps, so about half the residual is discretisation. The gate is
  stricter than anything R-ASBS achieves at any β, but it is reported as failed
  rather than relaxed.
- **The MCMC reference stops converging for β ≳ 1000**, where acceptance goes to
  zero and the chain freezes at 3.0233; for those β we quote the exact value 3.
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
python structured_asbs/stiefel.py ref --betas "0.001,0.01,0.1,0.5,1.3,2,5,7,10,20,50,100"
python structured_asbs/stiefel.py verify                       # gate D0, incl. spin-clock test
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
