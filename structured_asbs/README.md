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

---

## Comparison scoreboard

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
| St(4,2) | error at β = 50 | 0 | **0.270** | 4.391 (not converged) |
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
| occupation, m = N = 1000 | KS(occupancy) | 0 | **0.0102** |
| occupation, all scales | constraint violations | 0 | **0** |

Both discrete TV values sit essentially **at the iid sampling floor** — the
residual is the finite-sample error of drawing that many exact samples, not a
defect of the sampler.

---

## The R-ASBS Stiefel comparison

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

| β | reference | R-ASBS rerun | R-ASBS error | ours | our error |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 7.9965 | 7.9546 | −0.042 | 7.9958 | **−0.001** |
| 0.01 | 7.9628 | 7.9407 | −0.022 | 7.9679 | **+0.005** |
| 0.1 | 7.6671 | 7.7262 | +0.059 | 7.6893 | **+0.022** |
| 0.5 | 6.4171 | 6.5426 | +0.126 | 6.5152 | **+0.098** |
| 1.3 | 4.7503 | 5.2008 | +0.451 | 4.9016 | **+0.151** |
| 2 | 4.0976 | 4.6444 | **+0.547** | 4.2335 | **+0.136** |
| 5 | 3.4104 | 3.8893 | +0.479 | 3.5212 | **+0.111** |
| 7 | 3.2905 | 3.6104 | +0.320 | 3.3995 | **+0.109** |
| 10 | 3.2025 | 3.5275 | +0.325 | 3.3102 | **+0.108** |
| 20 | 3.1005 | 3.3858 | +0.285 | 3.2170 | **+0.116** |
| 50 | 3.0418 | 3.3116 | **+0.270** | 7.4330 | +4.391 ✗ |
| 100 | 3.0279 | 3.2614 | **+0.234** | 8.3180 | +5.290 ✗ |
| 200 | 3 (exact) | 3.2482 | +0.248 | not trained | — |
| 10⁶ | 3 (exact) | 3.1847 | +0.185 | not trained | — |

Our column is trained one β at a time — each entry is a separate ~30 minute
run, which is the only reason the grid was ever partial. ✗ marks the two
points where our training does not converge; see below.

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

### Where our method fails

At β = 50 and β = 100 our training does not converge, and R-ASBS beats us
there by more than an order of magnitude. This is stated first because it is
the one place in this repository where the comparison goes the other way.

The mechanism is not subtle. The regression loss scales like β²:

| β | 0.001 | 2 | 50 | 100 |
|---|---:|---:|---:|---:|
| final loss | 0.055 | ~10¹ | 7.1e+04 | 2.8e+05 |
| converged? | yes | yes | no, flat for 1500 iters | no, flat for 1500 iters |

At β = 50 the loss reads 71015 → 69980 → 71889 across the run: it never
descends at all. The score magnitude that the target implies also destabilises
the 199-step Euler grid. Refining that grid on the *same* trained control
separates the two effects:

| β | 199 steps | 398 steps | 796 steps |
|---:|---:|---:|---:|
| 50 | +4.391 | +3.333 | **+0.415** |
| 100 | +5.290 | +4.202 | +3.509 |

So at β = 50 most of the failure is integration, not learning — 796 steps
recovers to +0.415, still bad but no longer catastrophic. At β = 100 both
effects are present and refinement recovers little. Either way the fixed budget
(1500 iterations, lr 1e-3, 199 steps) is simply the wrong budget at large β,
and we have not fitted a β-dependent one. R-ASBS's Haar source, whose bias
saturates at ≈+0.23, degrades far more gracefully.

The two points are plotted and tabulated like every other β, marked but not
removed.

### Reproducing

```bash
python rasbs_port.py --check-retraction        # GS == sign-corrected QR
python rasbs_port.py --out json/results_rasbs_stiefel.json
python stiefel.py ref --betas "0.001,0.01,0.1,0.5,1.3,2,5,7,10,20,50,100"
```

Caveat, stated because it matters: our MCMC reference itself stops converging
for β ≳ 1000, where a fixed proposal size gives near-zero acceptance and the
chain freezes at 3.0233 (identical to four decimals at β = 10³, 10⁴ and 10⁶).
For those β we quote the exact value 3 instead. In the useful range β ≤ 100
two independent chains agree to 0.0001–0.01, so the reference is trustworthy
there and the comparison above is safe.

---

## Fair-comparison protocol

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
falls by ~1.7×; theirs by ~1.12×.
Richardson extrapolation in 1/N on their two finest grids gives a floor of
**≈0.46**, i.e. roughly 93% of their remaining error at N = 512 is not
discretisation at all. Their orthogonality residual likewise stays pinned near
3.8e-07 at every step count: refining the grid cannot repair a retraction.

---

## Running

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

Gate `D0` includes the mandatory first-moment test for the spin clock. On
St(4,2) each SU(2) ≅ S³ factor of Spin(4) runs at heat time s = r/2, not r;
the test checks `E[X_r | X_0] = e^{−3r} X_0` and separates the two conventions
by a factor of ~160 at r = 0.25, so a wrong clock cannot pass silently.

---

## The sample gallery

`figures.py` plots metrics. `gallery.py` plots the samples themselves, as the
objects they actually are, beside the exact or MCMC reference drawn the same
way. The claim these panels support is "you cannot tell the two columns
apart", and that is a claim the eye should adjudicate.

| | contents |
|---|---|
| `fig6_sphere_cloud` | S² clouds: the uncontrolled source, ours, exact iid target, and the residual |
| `fig7_stiefel_frames` | 96 raw St(4,2) frames per sampler, plus the second moment E[XXᵀ] |
| `fig8_ising_configs` | 50 raw 4×4 spin configurations, ours beside exact enumeration |
| `fig9_occupation_raster` | raw occupancy vectors at m = N = 1000, plus per-box marginals |

Three things the gallery had to get right, because getting them wrong would
have flattered us:

**The S² reference is not `sphere_reference`.** That checkpoint holds the
*uncontrolled* process — the bridge's source, KS(x₃) = 0.324 — not the target.
Comparing against it measures the transport, not the error. Figure 6 draws
exact iid target samples by inverse-CDF instead (the density is ∝ exp(6x₃²)
with uniform azimuth, so this is exact), and against that reference 99% of the
1800 equal-area bins fall within ±3σ with max |z| = 4.3: at 200,000 samples a
side there is no resolvable structure left in the residual.

**Checkpoints now store samples in float64.** They used to be float32, whose
epsilon is 1.2e-07. Recomputing the orthogonality residual from such a file
returns storage round-off rather than the 3.8e-14 `stiefel.py` measures at
generation time — and that round-off happens to look like R-ASBS's 3.4e-07
retraction error, which is exactly the number the residual exists to
distinguish itself from. `common.py:save_ckpt` promotes any floating-point
sample tensor to float64; integer state indices are left alone. Figure 7 draws
the residual only when the checkpoint on disk is genuinely float64 and prints a
notice otherwise, so an old file degrades to a missing panel rather than to a
wrong one.

Checkpoints written before this change are float32. `regen_f64.sh` re-runs the
β = 2 pair (ours and the R-ASBS port) at the same configuration and seed to
refresh them; everything else refreshes the next time its experiment is run.
With those refreshed, figure 7's bottom row now draws the residual from disk:
**3.0e-14 across all 100,000 of our samples against 2.9e-07 across theirs**, a
factor of 10⁷, recomputed rather than quoted.

**`stiefel.py` and `rasbs_port.py` do not use the same basis.** Ours works in
the eigenbasis of H (H = diag(1, 2, 5, 8)); theirs works in R-ASBS's ambient
Z₂×Z₂ basis. Figure 7 rotates their samples into the common eigenbasis before
comparing anything. Without that rotation the two second moments look like
each other's answers.

### What figure 7 shows

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
This is the source-tilting bias of §"The R-ASBS Stiefel comparison", visible
directly in the second moment rather than inferred from an error curve.

---

## Notes

- All manifold experiments preserve their constraint to machine precision by
  construction (2e-16 on S², 4e-14 on St(4,2)) because the update is an exact
  geodesic/group step, not a projection or retraction.
- The discrete experiments preserve their combinatorial constraint exactly:
  zero violations across every run at every scale.
- `--antithetic` on `stiefel.py` exploits an exact 16-fold sign symmetry
  `X ↦ SXD`. It is automatically disabled with `--frame`, since the frame term
  `−λ tr(CᵀX)` breaks that symmetry.

---

## Honest limitations

- **At β ≥ 50 our training does not converge and R-ASBS is better than we
  are** (their +0.27 against our +4.39 at β = 50). The loss grows like β² and
  never descends at the fixed budget; grid refinement recovers β = 50 to +0.415
  but not β = 100. A β-dependent budget is the obvious fix and we have not done
  it.
- Our self-imposed Stiefel gate is `|E − E_MCMC| < 0.05`. At β = 0.1 we pass
  it (0.022); at β ≥ 0.5 we do not (0.10–0.17 at 199 steps). Refining the
  integration grid on the *same* trained control drops β = 1 to 0.050 at 1024
  steps, so roughly half the residual is discretisation and half is score
  error. The gate is stricter than anything R-ASBS achieves at any β, but it
  is not yet met, and it is reported as failed rather than relaxed.
- The MCMC reference stops converging for β ≳ 1000; the exact value 3 is used
  there instead.
- KS(E) degrades at large β (0.23 at β = 10) even as the mean energy error
  improves, because the target concentrates and KS becomes very sensitive.
- The frame-sensitive target D2 has no analytic reference at all — MCMC is the
  only ground truth, so its error bar is the reference's own.
