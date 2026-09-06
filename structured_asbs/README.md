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
Each has its own section below, carrying its own configuration, its own results
and its own limitations.

---

## Benchmarks

Everything in this README is what the committed `json/results_*.json` files
were produced with; each of those files also carries its own full `config`
block, so any number here can be traced back to the flags that made it.

**Hardware and software.** Single NVIDIA A100 80GB per run (two available, used
only to run independent experiments concurrently). PyTorch 2.5.1, CUDA 12.6,
NumPy 2.4.6, conda environment `SML_env` (Python 3.11). Manifold state and all
metrics are `float64`; the networks are `float32`.

**The source is a Dirac in every experiment.** This is the structural point of
the method and it is what the whole R-ASBS comparison turns on, so it is worth
stating plainly: we start from a *fixed point* `x_0` and learn a control that
transports the reference process to the target, following Adjoint Sampling. We
never sample the source from a guessed distribution. R-ASBS instead starts from
**Haar**, which is only correct in the β → 0 limit — the correct initial law
for the h-transform is `Haar · φ₀ / ⟨Haar, φ₀⟩`, and `φ₀` is not constant. That
mismatch is the source-tilting bias measured in the Stiefel section.

| | A: Ising | B: occupation | C: S² | D: St(4,2) |
|---|---|---|---|---|
| script | `fixed_ising.py` | `occupation.py` | `sphere.py` | `stiefel.py` |
| state space | 4×4 periodic lattice, 16 spins, magnetisation fixed | 4 particles on 4 sites | unit sphere in R³ | 4×2 orthonormal frames |
| target | `E = −J Σ_⟨ij⟩ s_i s_j`, `J = 1`, `τ = 2` | `π(η) ∝ Π_i Γ(η_i+d)/(η_i! Γ(d))`, `d = 0.5` | `E(x) = 6(1 − x₃²)`, `π ∝ exp(6x₃²)` | `E(X) = tr(XᵀHX)`, `H = diag(1,2,5,8)` |
| temperature | `τ = 2` | `τ = 1` | `τ = 1` | `τ = 1/β`, β swept 1e-3 → 1e6 |
| source `x_0` | fixed spin configuration | fixed occupancy | `(1, 0, 0)` | `E₀ = [e₁, e₂]` |
| reference | exact enumeration | exact (matrix exponential) | exact inverse-CDF | MCMC, 2e5 chains × 3000 sweeps |
| network | `SwapController`, hidden 512 | `OccController`, hidden 256 | `ScoreNet`, hidden 256 | `ScoreNet`, hidden 256 |
| parameters | 670,464 | 139,536 | 136,963 | 139,528 |
| integration steps | 128 | 128 | 128 | 199 |
| iters × inner | 300 × 40 | 1500 × 4 | 4000 × 16 | 1500 × 8 |
| batch / minibatch | 2048 / 1024 | 512 / 1024 | 8192 / 16384 | 2048 / 16384 |
| lr | 3e-4 | 1e-3 | 1e-3 | 1e-3 |
| EMA / buffer | — / 8 | — / 8 | 0.9995 / 4 | 0.9995 / 4 |
| eval samples | 20,000 | 20,000 | 200,000 | 100,000 |
| seeds | 1 | 1 | 5 | 1 |

### Scale, stated plainly

Most of these state spaces are small; one is not.

| | state space | size | scaled? |
|---|---|---|---|
| A: Ising | 4×4 lattice, magnetisation 0 | C(16, 8) = 12,870 | **no** — 4×4 only |
| B: occupation | m particles on N sites | C(m+N−1, N−1) | **yes** — up to 10⁶⁰⁰ |
| C: S² | 2-manifold in R³ | continuous, dim 2 | n/a |
| D: St(4,2) | 4×2 orthonormal frames | continuous, dim 5 | n/a |

S² is 2-dimensional and St(4,2) is 5-dimensional. Apart from the occupation
sweep, nothing here is a high-dimensional benchmark by the standards of the
sampling literature. The occupation process is the one real scale test and it
is run properly: same network, same code, only the state space grows.

### Which R-ASBS experiments we reproduced: two of six

| their script | problem | ours |
|---|---|---|
| `asbs_sphere_sampler.m` | bimodal S², `E = 6(1 − x₃²)` | ✔ `sphere.py`, same energy (their sampler itself is not re-run on this problem) |
| `alg2_stiefel.m` | St(4,2), `E = tr(XᵀHX)`, spec(H) = {1,2,5,8} | ✔ `stiefel.py`, same H and same β grid, and their algorithm re-run by `rasbs_port.py` |
| `earthquake_sphere_ex.m` | earthquake epicentres on S² | ✘ not attempted |
| `cosmic_ray_ex.m` | arrival directions on the hemisphere S²₊ | ✘ not attempted |
| `alg2_robotics.m` | 10-DOF planar IK, loop closure + obstacles | ✘ not attempted |
| `asbs_m_wahba_so3.m` | robust Wahba problem on SO(3) | ✘ not attempted |

The two we did reproduce are the two synthetic benchmarks with a known target,
which are the only ones where an error against ground truth can be computed at
all. The four we did not are their applied and higher-dimensional cases, and
`alg2_robotics.m` in particular is a harder problem than anything in this
repository. Our Ising and occupation experiments have no R-ASBS counterpart in
either direction: their method is formulated for embedded Riemannian manifolds
and does not apply to discrete state spaces.

**R-ASBS baseline** (`rasbs_port.py`, a port of `alg2_stiefel.m` from commit
`bb71d14`): two networks totalling **140,560** parameters (`netU` 70,408 +
`netH` 70,152), 199 integration steps, batch 600, 1000 epochs, lr 1e-3, Haar
source, ambient Euler step followed by a QR retraction. Parameter counts are
within 1% of ours (139,528 vs 140,560) and the budget-matched comparison
equalises oracle calls at 600,000 exactly.

### Comparison scoreboard

Reference column is exact where an exact value exists, otherwise an
independent MCMC ground truth. The R-ASBS column is **measured**, never read
off a published plot: the sphere numbers are the ones printed in their paper,
the Stiefel numbers come from rerunning their algorithm ourselves.

| Benchmark | Metric | Exact/reference | R-ASBS | Ours |
|---|---|---:|---:|---:|
| S² analytic | north mass | 0.500 | 0.438 | **0.4993 ± 0.0003** |
| S² analytic | absolute north error | 0 | 0.062 | **0.0007 ± 0.0003** |
| S² analytic | KS(x₃) | 0 | not reported | **0.0217 ± 0.0006** |
| S² analytic | max norm residual | 0 | not reported | **2.2e-16** |
| St(4,2) | E at β = 0.1 | 7.6671 | 7.7262 | **7.6893** |
| St(4,2) | error at β = 0.001 | 0 | 0.042 | **0.0007** |
| St(4,2) | error at β = 50, 199 steps | 0 | **0.270** | 4.391 (not converged) |
| St(4,2) | error at β = 50, refined grid | 0 | 0.246 (1024 steps, floor) | **0.010** (1592 steps) |
| St(4,2) | error at β = 100, refined grid | 0 | 0.215 (512 steps, floor) | **0.019** (3184 steps) |
| St(4,2) | KS(E) at β = 100, 3184 steps | 0 | not reported | 0.258 (mean fixed, law not) |
| St(4,2) | E as β → ∞ | 3 | **3.185** | 3.310 (β=10) |
| St(4,2) | worst finite-β error | 0 | **0.547** (β=2) | **0.166** (β=1.3) |
| St(4,2) | error at β=1 | 0 | ≈0.30 (interp.) | **0.100** |
| St(4,2) | error at β=2, 512 steps | 0 | 0.493 | **0.072** |
| St(4,2) | surrogate floor as N→∞ | 0 | **≈0.46** | none measured |
| St(4,2) | orthogonality residual | 0 | 3.4e-07 | **3.8e-14** |
| frame-sensitive St(4,2) | reference discrepancy | 0 | not applicable | **0.120** (0.049 refined) |

Discrete experiments have no R-ASBS counterpart — they are compared against
exact enumeration:

| Benchmark | Metric | Exact | Ours |
|---|---|---:|---:|
| fixed-magnetisation Ising | TV vs exact | 0 | **0.0517** (iid floor 0.0511) |
| fixed-magnetisation Ising | constraint violations | 0 | **0** |
| occupation, small | TV vs exact | 0 | **0.0199** (iid floor 0.0167) |
| occupation, m = N = 32 | KS(occupancy) | 0 | **0.0043** (source 0.2054) |
| occupation, m = N = 128 | KS(occupancy) | 0 | **0.0084** (source 0.2054) |
| occupation, m = N = 1000 | KS(occupancy) | 0 | **0.0102** (source 0.2035) |
| occupation, m = N = 1000 | KS(max occupancy) | 0 | 0.2215 (our worst statistic) |
| occupation, m = N = 1000 | W₁(max) / m | 0 | **0.0014** |
| occupation, all scales | constraint violations | 0 | **0** |

Both discrete TV values sit essentially **at the iid sampling floor** — the
residual is the finite-sample error of drawing that many exact samples, not a
defect of the sampler.

### Wall-clock

Single A100, for the numbers quoted in this README: Experiment D trains in
~1130 s per β at 199 steps, ~2180 s at 398 and ~4380 s at 796, plus sampling
and the MCMC reference. R-ASBS costs ~780 s per β at 512 steps and ~1560 s at
1024. Every `results_*.json` records `train_s` (ours) or `wall_s` (theirs)
per β.

### Running

```bash
python tests_math.py                 # mathematical unit tests
python fixed_ising.py verify         # gate A0
python occupation.py verify          # gate B0
python sphere.py verify              # gate C0
python stiefel.py verify             # gate D0
python figures.py                    # all figures -> fig/
python gallery.py                    # sample gallery -> fig/
```

Each script exposes `verify` (correctness gates), `train`, and a sweep mode.
`stiefel.py` additionally has `ref` for the independent MCMC ground truth.

`figures.py` plots metrics. `gallery.py` plots the samples themselves, as the
objects they actually are, beside the exact or MCMC reference drawn the same
way. The claim those panels support is "you cannot tell the two columns
apart", and that is a claim the eye should adjudicate. Per-benchmark figures
are described in their own sections.

Two conventions that hold across every experiment:

- All manifold experiments preserve their constraint to machine precision by
  construction (2e-16 on S², 4e-14 on St(4,2)) because the update is an exact
  geodesic/group step, not a projection or retraction.
- The discrete experiments preserve their combinatorial constraint exactly:
  zero violations across every run at every scale.

**Checkpoints store samples in float64.** They used to be float32, whose
epsilon is 1.2e-07. Recomputing an orthogonality residual from such a file
returns storage round-off rather than the 3.8e-14 `stiefel.py` measures at
generation time — and that round-off happens to look like R-ASBS's 3.4e-07
retraction error, which is exactly the number the residual exists to
distinguish itself from. `common.py:save_ckpt` promotes any floating-point
sample tensor to float64; integer state indices are left alone. Figures draw
the residual only when the checkpoint on disk is genuinely float64 and print a
notice otherwise, so an old file degrades to a missing panel rather than to a
wrong one. `regen_f64.sh` re-runs the β = 2 pair to refresh the pre-change
files; everything else refreshes the next time its experiment is run.

---

## Ising lattice

**Experiment A — bijective discrete.** A 4×4 periodic Ising lattice with the
magnetisation held fixed at zero, so the state space is the constraint set
Ω = {s ∈ {±1}¹⁶ : Σᵢ sᵢ = 0}, of size C(16, 8) = 12,870. The target is the
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

| metric | exact | ours |
|---|---:|---:|
| TV vs exact law | 0 | **0.0517** (iid floor 0.0511) |
| constraint violations in 20,000 samples | 0 | **0** |

The TV number sits essentially at the iid sampling floor: draw 20,000 genuinely
exact samples and the empirical law is already 0.0511 away from the truth, so
the residual is finite-sample error rather than a defect of the sampler.

`fig8_ising_configs` draws 50 raw 4×4 spin configurations from our sampler
beside 50 drawn by exact enumeration, and `fig2_discrete_correctness` carries
the metric panel.

**Limitations.**

- **Only the 4×4 lattice.** This is the size at which the constrained
  distribution can be enumerated exactly, which is the entire reason for the
  experiment — but it does mean we have no evidence about larger lattices.
- No R-ASBS counterpart exists in either direction: their method is formulated
  for embedded Riemannian manifolds and does not apply to discrete state
  spaces.

```bash
python fixed_ising.py verify         # gate A0, exact-control propagation
python fixed_ising.py exact          # gate A0/A1/A2 with the exact control
python fixed_ising.py train
```

---

## Occupation

**Experiment B — non-bijective discrete, and the scale test.** m indistinguishable
particles distributed over N sites; a state is the occupancy vector η ∈ ℕᴺ with
Σᵢ ηᵢ = m, and the constraint set has size C(m+N−1, N−1). The target is
`π(η) ∝ Πᵢ Γ(ηᵢ+d) / (ηᵢ! Γ(d))` at `d = 0.5`. Unlike the Ising swap chain the
moves here are **not** bijective — particles are indistinguishable, so many
microscopic moves collapse onto the same occupancy transition, and the
intertwining has to carry that degeneracy.

At m = N = 4 the state space is small enough to enumerate, giving an exact TV:

| metric | exact | ours |
|---|---:|---:|
| TV vs exact law | 0 | **0.0199** (iid floor 0.0167) |
| constraint violations | 0 | **0** |

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
  the one we do worst on at m = 1000. It is reported here and in the scoreboard
  rather than left out.
- **The three rows are not budget-matched.** m = 32 and 128 use 3000 iterations,
  128 steps, batch 512; m = 1000 uses 1500 iterations, 256 steps, batch 128. So
  the row-to-row comparison should not be read as a clean scaling law.

```bash
python occupation.py verify                  # gate B0
python occupation.py train
python occupation.py scale --m 1000 --N 1000
```

---

## Sphere

**Experiment C — S², scalar Killing readout.** The bimodal target
`π ∝ exp(6 x₃²)` on the unit sphere, i.e. `E(x) = 6(1 − x₃²)`, which is the
same energy and the same scale as R-ASBS's `asbs_sphere_sampler.m`. Mass
concentrates on the two poles, and the diagnostic that matters is whether both
poles get half of it: a sampler that collapses onto one pole still gets the
radial statistics right, so the north mass and the KS distance of the `x₃`
marginal are what separate a correct sampler from a plausible-looking one.

The reference is exact — the density is ∝ exp(6x₃²) with uniform azimuth, so
inverse-CDF sampling draws genuinely iid target points.

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

- S² is 2-dimensional. This is a correctness benchmark, not a hard one.
- The R-ASBS column is quoted from their paper. We did not rerun their sphere
  sampler: we cannot run the MATLAB original, so we could not separate their
  algorithm's behaviour from a porting error of our own, and a claim we cannot
  check does not belong here.

```bash
python sphere.py verify              # gate C0
python sphere.py train --antithetic
```

---

## Stiefel

**Experiment D — St(4,2), matrix Killing readout, and the head-to-head against
R-ASBS.** The state space is the 4×2 orthonormal frames, dimension 5, with
`E(X) = tr(XᵀHX)` and `H = diag(1, 2, 5, 8)` — R-ASBS's own `H` up to a change
of basis, and their own β grid. Temperature is `τ = 1/β` with β swept from
1e-3 to 1e6. The reference is an independent MCMC ground truth: 2×10⁵ chains
× 3000 sweeps.

Experiment D additionally uses `σ = √2`, `nq = 64` fibre quadrature nodes, and
`--antithetic` (the exact 16-fold symmetry `T_s(X) = SXD`, automatically
disabled with `--frame`, since the frame term `−λ tr(CᵀX)` breaks it). Its
`ScoreNet` also carries one non-trainable scalar buffer, `out_scale`, described
under "High β: diagnostics".

Gate `D0` includes the mandatory first-moment test for the spin clock. On
St(4,2) each SU(2) ≅ S³ factor of Spin(4) runs at heat time s = r/2, not r; the
test checks `E[X_r | X_0] = e^{−3r} X_0` and separates the two conventions by a
factor of ~160 at r = 0.25, so a wrong clock cannot pass silently.

### The R-ASBS comparison

Their paper reports the expected-energy curve as a plot rather than a table,
so every number below was produced by rerunning their algorithm.

Frozen upstream commit:

```
https://github.com/mattiamosso/Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-via-Adjoint-Schrodinger-Bridges
bb71d1496468658f4ff1c9995773968fd12706a8   (2026-08-28)
```

`alg2_stiefel.m` depends on the MATLAB Deep Learning Toolbox (`dlnetwork`,
`dlfeval`, `adamupdate`), which GNU Octave does not implement, so
`rasbs_port.py` is a line-for-line PyTorch port. It keeps every structural
choice of the original: their 4×4 Z₂×Z₂ group matrix `H` (spectrum
{1, 2, 5, 8}), the **Haar** source, the ambient Euler step followed by a QR
retraction, the backward sequential projection, and σ = 1, N = 199, B = 600,
1000 epochs, lr = 1e-3.

`N` is the number of integration steps. β ≤ 20 is the shared 199-step grid both
methods were originally run on; at β = 50 and 100 each method is quoted at the
finest grid we ran for it, for the reason given directly below the table.

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

**Why β ≥ 50 uses a larger N, and why the two columns use different ones.**
The score the target implies is O(β), so the Euler increment h·score is O(1)
per step at β = 100 on a 199-point grid — the integrator is asked to take
order-one jumps on a manifold of diameter order one, and our sampler falls
apart there (+4.391 and +5.290 at N = 199; that is the honest matched-grid
number and R-ASBS beats us at it by an order of magnitude). Refining the grid
removes that error entirely. It does not remove theirs, because theirs is not
discretisation: it is the source-tilting bias of the Haar start plus the QR
retraction, neither of which is a function of step size. Measured, not assumed:

| N | 199 | 512 | 1024 | 1592 | 3184 |
|---|---:|---:|---:|---:|---:|
| R-ASBS, β = 50 | +0.272 | +0.255 | **+0.248** | — | — |
| ours, β = 50 | +4.391 | — | — | **+0.010** | — |
| R-ASBS, β = 100 | +0.235 | **+0.218** | +0.221 | — | — |
| ours, β = 100 | +5.290 | — | — | +0.750 | **+0.019** |

Their β = 100 row is the cleanest statement of the point: 512 → 1024 makes it
very slightly *worse* (+0.218 → +0.221), which is run-to-run noise on a
quantity that has stopped moving. Their β = 50 row still creeps down, by 0.024
over a 5× refinement, and Richardson extrapolation puts its limit near +0.24.
Ours falls by 400× and 100× over the same kind of refinement, because ours is
discretisation and theirs is not.

So each column above is quoted at the finest useful grid we ran for it. The
cost is real and is stated rather than hidden: our β ≥ 50 cells run 8–16× more
integration steps than theirs and are warm-started from the β below, a recipe
the rest of the column does not use.

Two observations that the curve alone does not show.

**The error changes sign at β ≈ 0.05.** At β → 0 the target *is* Haar, so
their Haar source is exactly right there and the residual −0.042 is pure
discretisation from the QR retraction. As β grows the target moves away from
Haar, the source-tilting bias switches on and dominates, peaking at +0.547
near β = 2. The two error sources are therefore separable, and the second one
is structural: the correct initial law for the h-transform is Haar·φ₀ /
⟨Haar, φ₀⟩, and φ₀ is not constant.

**The β → ∞ limit is never reached.** Their own paper states the limit is 3;
the rerun plateaus at 3.185 and stays there from β = 10³ to β = 10⁶.

### The geometric-surrogate floor

PLAN §9.3 predicts "ours: error decreases as step size decreases; R-ASBS:
possible nonzero surrogate floor", and explicitly warns *do not claim such a
floor before measuring it*. Measured at β = 2, where their bias peaks, with
everything except the step count held fixed:

| N_steps | R-ASBS error | our error |
|---:|---:|---:|
| 32 | 0.805 | 0.621 |
| 64 | 0.661 | 0.323 |
| 128 | 0.601 | 0.186 |
| 256 | 0.526 | 0.121 |
| 512 | 0.493 | **0.072** |

At the finest grid the gap is 6.9×. Per doubling of the step count our error
falls by ~1.7×; theirs by ~1.12×. Richardson extrapolation in 1/N on their two
finest grids gives a floor of **≈0.46**, i.e. roughly 93% of their remaining
error at N = 512 is not discretisation at all. Their orthogonality residual
likewise stays pinned near 3.8e-07 at every step count: refining the grid
cannot repair a retraction. Spending 16× the steps buys them 0.31; the same
factor buys us 400×.

### Fair-comparison protocol

PLAN §9.2 requires matching the number of terminal energy/gradient
evaluations. `stiefel.py` counts these exactly, inside the single function
that touches the energy gradient, rather than estimating them — and the count
revealed that our default configuration uses **3,072,000** oracle calls
against R-ASBS's **600,000**, a 5.1× advantage. The headline table above is
therefore *not* by itself a fair comparison.

The budget-matched run (`--batch 400 --iters 1500`, exactly 600,000 oracle
calls, same 199 integration steps) settles it:

| β | reference | R-ASBS (600k) | ours (600k) | ours (3.07M) | gain at matched budget |
|---:|---:|---:|---:|---:|---:|
| 1.3 | 4.7503 | +0.451 | **+0.166** | +0.151 | 2.7× |
| 2 | 4.0976 | +0.547 | **+0.161** | +0.136 | 3.4× |
| 5 | 3.4104 | +0.479 | **+0.111** | +0.111 | 4.3× |

Removing the 5.1× budget advantage costs us 0.01–0.03 in error and changes no
conclusion. Everything below is matched:

| | R-ASBS | ours |
|---|---:|---:|
| parameters | 140,560 | 139,528 (0.99×) |
| integration steps | 199 | 199 (matched) |
| oracle calls | 600,000 | 600,000 (matched) |
| generated samples | 100,000 | 100,000 (matched) |
| constraint enforcement | QR retraction | exact geodesic step |

### High β: diagnostics

Four things worth recording about the β ≥ 50 cells, including the two
hypotheses that turned out to be wrong.

**1. It is not the loss scale.** The regression loss really does grow like β²
(0.055 at β = 0.001, ~10¹ at β = 2, 7.1e4 at β = 50, 2.8e5 at β = 100) and at
β ≥ 50 it never descends: 71015 → 69980 → 71889 across 1500 iterations. But
Adam is scale-invariant, so magnitude alone cannot be the cause. We removed the
scale anyway — `ScoreNet` now factorises `score = out_scale · raw(t, X)` with
`out_scale` a running RMS of the label, so the loss, the gradient clip and the
output range are O(1) at every β. It works (normalised loss 7.43 at β = 100
against 7.5 at β = 2) and changes the answer by nothing: +4.463 / +5.271
against the original +4.391 / +5.290. Clean negative result. The factorisation
and `--init-from` were kept because they are what make the refined cells
reachable.

**2. The refined rows are not one sweep.** The 199 column is the original
cold-start run. For β = 50 the rest comes from one control trained on 398 steps
(warm-started from β = 20). For β = 100 the 398 cell is its own control, and
the 796 / 1592 / 3184 cells are the best of *two* controls that were both
trained on 796 steps: one warm-started directly from β = 50, and one from the
50 → 65 → 80 → 100 chain. Refinement always reuses the trained control and
never retrains. Raw records in `json/results_stiefel_anneal_*.json` and
`json/results_chain_b100.json`.

The chain control wins that cell, and this is worth stating explicitly because
we very nearly reported the worse number. Both controls look equally broken on
their own 796-step training grid (E = 6.50 and 6.57 against a target of 3.03,
both gates FAIL), so the chain was written up here as a failure. Evaluated on
3184 steps they separate:

| β = 100 control at 3184 steps | ΔE | KS(E) | E spread vs reference | train |
|---|---:|---:|---:|---:|
| warm start from β = 50 | +0.0461 | 0.312 | 16× too broad | 4393 s |
| **50 → 65 → 80 → 100 chain** | **+0.0187** | **0.258** | **2.1× too broad** | 2602 s |

The chain is better on error, on KS, on spread and on wall-clock at once, so
there is no metric on which quoting it is a favourable choice. The lesson is
about the gate rather than about annealing: our gate evaluates on the training
grid, and at β = 100 that grid is too coarse to tell a good control from a bad
one. Two controls that both "fail" identically differ by 2.5× once integrated
properly.

| β | 199 | 398 | 796 | 1592 | 3184 |
|---:|---:|---:|---:|---:|---:|
| 50 | +4.391 | +0.0685 | +0.0223 | **+0.0101** | — |
| 100 | +5.290 | +4.063 | +3.535 | +0.750 | **+0.0187** |

**3. At β = 100 refinement fixes the mean and not the law.** |ΔE| is the
flattering summary here:

| | steps | E | reference | ΔE | KS(E) |
|---|---:|---|---|---:|---:|
| β = 50 | 1592 | 3.0534 ± 0.0377 | 3.0433 ± 0.0286 | +0.0101 | 0.118 |
| β = 100 | 3184 | 3.0496 ± 0.0360 | 3.0310 ± 0.0175 | +0.0187 | 0.258 |

At β = 50 the agreement is real — mean 0.3% off, spread 30% too wide. At
β = 100 the mean lands within 0.019 and the spread is 2.1× the reference's,
which is much better than the 16× of the control we first reported, but
KS = 0.258 still says the two samples are plainly distinguishable. Both rows
fail the KS < 0.05 gate that every other β passes. So refinement gets the mean
and most of the spread at β = 100 and still does not get the law, and we did
not find a grid that does.

**4. We cannot predict which settings are stable.** `out_scale` sits near
0.55 β when the sampler is on the mode and near 2 β when it is not, and the
runs that fail spend their transient in the second regime. But increment size
does not control which happens: warm-starting β = 50 from β = 20 is a 2.5×
jump and stayed on-mode, while β = 65 from β = 50 is a 1.3× jump, on a *finer*
grid, and did not. Every leg of the 50 → 65 → 80 → 100 chain sat off-mode on
its own grid (`out_scale` 1.9β / 2.4β / 2.2β), and yet its final control is the
best β = 100 control we have. So "off-mode during training" does not predict
"bad control" either. The stability boundary is real and reproducible and we
have no account of it; the step counts above were found by measurement.

### What figure 7 shows

`fig5_stiefel` carries the energy curves and the step-refinement panels.
`fig7_stiefel_frames` draws 96 raw St(4,2) frames per sampler plus the second
moment E[XXᵀ].

**`stiefel.py` and `rasbs_port.py` do not use the same basis.** Ours works in
the eigenbasis of H (H = diag(1, 2, 5, 8)); theirs works in R-ASBS's ambient
Z₂×Z₂ basis. Figure 7 rotates their samples into the common eigenbasis before
comparing anything. Without that rotation the two second moments look like
each other's answers.

Once both are in the same basis, the diagonal of E[XXᵀ] at β = 2 reads:

| axis (eigenvalue of H) | 1 | 2 | 5 | 8 | ‖·−target‖_F | E |
|---|---:|---:|---:|---:|---:|---:|
| target (MCMC) | 0.893 | 0.859 | 0.166 | 0.081 | 0 | 4.093 |
| ours | 0.879 | 0.845 | 0.180 | 0.096 | **0.028** | 4.233 |
| R-ASBS | 0.834 | 0.793 | 0.254 | 0.119 | 0.132 | 4.644 |
| Haar (their source) | 0.500 | 0.500 | 0.500 | 0.500 | 0.755 | 8.000 |

Haar is the null hypothesis: it is what a sampler returns if it inherits its
source instead of transporting it. Measured along that axis, R-ASBS sits
**17% of the way back to Haar** and ours sits **4%** — a 4.7× difference in a
quantity that has nothing to do with the scalar energy we have been reporting.
This is the source-tilting bias, visible directly in the second moment rather
than inferred from an error curve.

Figure 7's bottom row draws the orthogonality residual from disk:
**3.0e-14 across all 100,000 of our samples against 2.9e-07 across theirs**, a
factor of 10⁷, recomputed rather than quoted.

**Limitations.**

- **At β ≥ 50 on the shared 199-step grid our sampler does not converge and
  R-ASBS is better than we are** (their +0.27 against our +4.39 at β = 50).
  The cause is discretisation, not learning: refining the grid and
  warm-starting from the β below recovers +0.010 at β = 50 (1592 steps) and
  +0.019 at β = 100 (3184 steps), both far below R-ASBS's floor, but at 8–16×
  their step budget and with a training recipe that differs from the rest of
  the sweep. We report both, and keep the 199-step number in the headline.
- **The β = 100 refined cell fixes the mean and not the law.** ΔE = +0.019 and
  the spread is 2.1× the reference's, but KS(E) = 0.258 still fails the gate.
  We did not find a grid fine enough to fix the distribution at β = 100.
- **Our gate cannot rank controls at β = 100.** It evaluates on the training
  grid, and at 796 steps that grid is too coarse: two controls that both fail
  it identically (E = 6.50 and 6.57) turn out to differ by 2.5× in error once
  integrated on 3184 steps. We reported the worse one until we checked.
- **We cannot predict which (β, steps, warm start) combinations are stable.**
  `out_scale` sits near 0.55 β on-mode and near 2 β off-mode, and the failing
  runs are the off-mode ones — but a 2.5× β jump succeeded where a 1.3× jump on
  a finer grid failed. The step counts above were found by measurement.
- Our self-imposed Stiefel gate is `|E − E_MCMC| < 0.05`. At β = 0.1 we pass
  it (0.022); at β ≥ 0.5 we do not (0.10–0.17 at 199 steps). Refining the
  integration grid on the *same* trained control drops β = 1 to 0.050 at 1024
  steps, so roughly half the residual is discretisation and half is score
  error. The gate is stricter than anything R-ASBS achieves at any β, but it
  is not yet met, and it is reported as failed rather than relaxed.
- **The MCMC reference stops converging for β ≳ 1000**, where a fixed proposal
  size gives near-zero acceptance and the chain freezes at 3.0233 (identical to
  four decimals at β = 10³, 10⁴ and 10⁶). For those β we quote the exact value
  3 instead. In the useful range β ≤ 100 two independent chains agree to
  0.0001–0.01, so the reference is trustworthy there and the comparison above
  is safe.
- KS(E) degrades at large β (0.23 at β = 10) even as the mean energy error
  improves, because the target concentrates and KS becomes very sensitive.
- The frame-sensitive target D2 has no analytic reference at all — MCMC is the
  only ground truth, so its error bar is the reference's own.

```bash
python rasbs_port.py --check-retraction        # GS == sign-corrected QR
python rasbs_port.py --out json/results_rasbs_stiefel.json
python stiefel.py ref --betas "0.001,0.01,0.1,0.5,1.3,2,5,7,10,20,50,100"
python stiefel.py verify                       # gate D0, incl. spin-clock test
```
