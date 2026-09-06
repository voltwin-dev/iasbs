# Adjoint Schrödinger Bridge Sampling on Structured State Spaces via Markov Semigroup Intertwining

Experimental package for the paper. One standalone script per experiment, one
small shared utility file, no framework machinery — the layout deliberately
follows the R-ASBS repository's style.

```
common.py        shared kernels, quadrature, Stiefel/S^3 helpers, checkpoint IO
fixed_ising.py   Exp A  fixed-magnetisation Ising      (bijective discrete)
occupation.py    Exp B  occupation process             (non-bijective discrete)
sphere.py        Exp C  S^2                            (scalar Killing readout)
stiefel.py       Exp D  St(4,2)                        (matrix Killing readout)
rasbs_port.py    faithful PyTorch port of R-ASBS alg2_stiefel.m
tests_math.py    standalone mathematical unit tests
figures.py       all paper figures + the comparison table
gallery.py       sample gallery -- the samples themselves, not their metrics
json/            every results_*.json the figures and tables are built from
```

Four benchmarks, in order of increasing structure: two discrete state spaces
where the answer can be enumerated exactly, then two manifolds where it cannot.
Each section below carries its own configuration, results, figures, limitations
and reproduction commands. Every number is what the committed
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

```bash
python tests_math.py                 # mathematical unit tests
python figures.py                    # all figures -> fig/
python gallery.py                    # sample gallery -> fig/
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
| integration steps | 128 |
| iters × inner | 300 × 40 |
| batch / minibatch | 2048 / 1024 |
| lr | 3e-4 |
| buffer | 8 |
| eval samples | 20,000 |
| seeds | 1 |

### Results

| metric | exact | ours |
|---|---:|---:|
| TV vs exact law | 0 | **0.0517** (iid floor 0.0511) |
| constraint violations in 20,000 samples | 0 | **0** |

The TV number sits essentially at the iid sampling floor: draw 20,000 genuinely
exact samples and the empirical law is already 0.0511 away from the truth, so
the residual is finite-sample error rather than a defect of the sampler. The
combinatorial constraint is preserved exactly — zero violations across every
run.

`fig8_ising_configs` draws 50 raw 4×4 spin configurations from our sampler
beside 50 drawn by exact enumeration; `fig2_discrete_correctness` carries the
metric panel.

**Limitations.**

- **Only the 4×4 lattice.** This is the size at which the constrained
  distribution can be enumerated exactly, which is the entire reason for the
  experiment — but it does mean we have no evidence about larger lattices.
- **No R-ASBS counterpart exists**, in either direction: their method is
  formulated for embedded Riemannian manifolds and does not apply to discrete
  state spaces. The comparison here is against exact enumeration instead.

```bash
python fixed_ising.py verify         # gate A0
python fixed_ising.py exact          # gate A0/A1/A2 with the exact control
python fixed_ising.py train
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
state space grows. `python occupation.py scale --m M --N M`:

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
python occupation.py verify                  # gate B0
python occupation.py train
python occupation.py scale --m 1000 --N 1000
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

The R-ASBS column is the number printed in their paper. We did not rerun their
sphere sampler — see the limitation below.

| metric | exact | R-ASBS (their paper) | ours |
|---|---:|---:|---:|
| north mass | 0.500 | 0.438 | **0.4993 ± 0.0003** |
| absolute north error | 0 | 0.062 | **0.0007 ± 0.0003** |
| KS(x₃) | 0 | not reported | **0.0217 ± 0.0006** |
| max norm residual | 0 | not reported | **2.2e-16** |

Five seeds. The norm residual is 2.2e-16 because the update is an exact
geodesic step on S², not a projection: the constraint is preserved by
construction rather than repaired after the fact.

`sphere.py` also carries an ablation of the symmetry handling — plain,
antithetic and symmetrised variants — since the target is invariant under
x₃ ↦ −x₃ and exploiting that exactly is cheaper than learning it.
`fig4_sphere` plots the comparison.

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
- **The R-ASBS column is quoted from their paper, not measured.** We cannot run
  the MATLAB original, so we could not separate their algorithm's behaviour
  from a porting error of our own, and a claim we cannot check does not belong
  here. The Stiefel section is where their algorithm is actually rerun.

```bash
python sphere.py verify              # gate C0
python sphere.py train --antithetic
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
finest grid we ran for it, for the reason given in the ablation below.

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
python rasbs_port.py --check-retraction        # GS == sign-corrected QR
python rasbs_port.py --out json/results_rasbs_stiefel.json
python stiefel.py ref --betas "0.001,0.01,0.1,0.5,1.3,2,5,7,10,20,50,100"
python stiefel.py verify                       # gate D0, incl. spin-clock test
```
