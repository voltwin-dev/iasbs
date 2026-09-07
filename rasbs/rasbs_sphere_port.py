"""Faithful PyTorch port of the public R-ASBS sphere scripts (their Alg. 1).

Upstream:  https://github.com/mattiamosso/
           Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-
           via-Adjoint-Schrodinger-Bridges
Commit:    bb71d1496468658f4ff1c9995773968fd12706a8   (2026-08-28)

Two of their scripts run the same algorithm on S^2 with different targets and
different hyperparameters, so both are ported here behind `--problem`:

  bimodal   asbs_sphere_sampler.m    E(x) = 6 (1 - x_3^2)
  quake     earthquake_sphere_ex.m   E(x) = -log (1/N) sum_i exp(kappa <m_i, x>)

A port is unavoidable for the same reason as `rasbs_port.py`: the scripts need
MATLAB's Deep Learning Toolbox (`dlnetwork`, `dlfeval`, `adamupdate`), which
GNU Octave does not implement.

WHAT IS DELIBERATELY NOT FIXED
------------------------------
Two of their choices differ from ours in ways that matter for the comparison,
and both are reproduced exactly rather than corrected.  Silently repairing a
baseline makes the comparison meaningless in the flattering direction, and
repairing it *loudly* still leaves us reporting numbers their paper does not
claim.  So:

  1. THE SOURCE IS UNIFORM, NOT A DIRAC.  `simulate_trajectories` draws
     `X0 = randn(3,B)` and normalises, i.e. Haar on S^2.  Adjoint Sampling's
     terminal identity assumes a *known* source; starting from a guessed one
     tilts the terminal law by the ratio of the two, which is the source-tilting
     bias we measure on Stiefel.  Ours starts from a fixed point.

  2. THE BRIDGE IS A GAUSSIAN SURROGATE, NOT THE BROWNIAN BRIDGE ON S^2.
     `compute_adjoint_targets` places `mu_t` on the geodesic, adds isotropic
     Gaussian noise in the ambient tangent space at `mu_t` with standard
     deviation `sigma sqrt(t(1-t))` (bimodal) or `sqrt(V_t (V_1 - V_t)/V_1)`
     (quake), and renormalises.  On a curved manifold that is not the law of
     the reference bridge; it is the Euclidean bridge retracted.  The
     discrepancy is O(curvature x variance) and does not vanish as the step
     count grows, which is what makes their error a floor rather than a
     discretisation term.

Everything else is copied line for line:

  * geodesic random walk: ambient Euler-Maruyama on the tangent-projected
    drift and noise, followed by a normalise-to-the-sphere retraction;
  * `a_1 = P_x(grad E) + h_phi(x)`, with h NOT projected before being added
    (it is projected inside the corrector loss instead), matching the source;
  * geodesic parallel transport of a_1 from x_1 back to x_t;
  * controller loss ||P_x(u) + sigma a_t||^2, corrector loss ||P_x(h) - b||^2;
  * bimodal: B=500, epochs=600, N=500, sigma=1, tanh MLPs 64/64 (u) and 48/48
    (h), lr 2e-3 for both, no drift clipping, constant sigma;
  * quake: B=500, epochs=2000, N=250, drift clip 30, sigma(t) = 0.03 +
    0.32 (1-t)^2 with the exact variance integral V(t), a staircase kappa
    anneal 150/300/450/600 at 25/50/75% of training, 256 random Fourier
    features (bandwidth 3 on x, 1 on t), 3 x 256 LayerNorm+GELU MLPs,
    lr_u 1e-3, lr_h 2e-3;
  * quake corrector target carries their volume-derivative term
    1/2 (cot theta - 1/theta), expanded as -theta/6 - theta^3/90 below 0.05.

Two departures, both mechanical rather than semantic:

  * their per-particle `for i = 1:B` loops (bimodal only; the quake script is
    already vectorised) are batched here.  Same arithmetic, different RNG
    stream, so individual numbers differ but the estimator does not.
  * `params.B` is used for buffer sizing in a couple of their helpers; we use
    the actual batch length, which is the same value in every call they make.

METRICS
-------
Scored with *our* measurement code, imported from `structured_asbs/`, so that
the R-ASBS column and our column in the README are produced by the same
estimator.  Only the algorithm is theirs; the ruler is shared.
"""

import argparse
import json
import math
import time

import numpy as np
import torch

# common.py and the shared json/ ckpt/ fig/ directories live at the repository
# root; the measurement helpers live one directory over, in structured_asbs/.
# Nothing algorithmic is imported from either -- see MEASUREMENT above.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "structured_asbs"))
import common as C
import earthquake as Q
import remeasure as R

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
# Everything -- state, networks, metrics -- is float64, matching the rest of
# this repository.  Their MATLAB `dlarray`s are single precision; running the
# port in double is a departure, but one that can only help the baseline, and
# it removes a whole class of "is this a port bug or a float32 artefact?"
# ambiguity from the numbers we are about to publish against them.
torch.set_default_dtype(DT)


# ============================================================================
# geometry primitives, verbatim from the .m files
# ============================================================================
def proj(x, v):
    """Tangent projection at x on the unit sphere: v - <v,x> x."""
    return v - (v * x).sum(-1, keepdim=True) * x


def retract(x):
    """Their `X ./ vecnorm(X, 2, 1)` -- normalise back onto S^2."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def haar(n, device, generator=None, dtype=DT):
    """`X0 = randn(3,n); X0 ./ vecnorm(X0)` -- uniform on S^2.

    This is their source.  It is NOT a Dirac; see the module docstring.
    """
    return retract(torch.randn(n, 3, device=device, dtype=dtype,
                               generator=generator))


def parallel_transport(a, x_from, x_to):
    """Geodesic parallel transport of a in T_{x_from} to T_{x_to}.

    Their formula:  a_ortho + a_par cos(phi) - <a, u> x_from sin(phi),
    with u the unit tangent at x_from pointing to x_to and phi the arc length.
    """
    d = (x_from * x_to).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    phi = torch.arccos(d)
    u_un = x_to - x_from * d
    nrm = u_un.norm(dim=-1, keepdim=True)
    u = u_un / nrm.clamp_min(1e-30)
    mag = (a * u).sum(-1, keepdim=True)
    a_par, a_ortho = u * mag, a - u * mag
    out = a_ortho + a_par * torch.cos(phi) - x_from * mag * torch.sin(phi)
    # their `if phi > 1e-5 ... else a_1` branch
    return torch.where(phi > 1e-5, out, a)


# ============================================================================
# networks
# ============================================================================
INIT = "matlab"     # set from --init; "torch" restores the pre-audit behaviour


def matlab_init(net):
    """MATLAB `fullyConnectedLayer` defaults: Glorot weights, ZERO bias.

    torch.nn.Linear instead draws both weight and bias from U(-1/sqrt(fan_in),
    +1/sqrt(fan_in)).  The nonzero bias is the part that matters here: on a
    target symmetric under x_3 -> -x_3 it is a fixed directional preference at
    initialisation, which is exactly what a winner-take-all training loop can
    amplify into one-pole collapse.  Removing it is a prerequisite for reading
    anything into this port's collapse behaviour.
    """
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return net


def mlp_tanh(din, dout, hidden):
    """bimodal: fullyConnected -> tanh -> fullyConnected -> tanh -> out."""
    net = torch.nn.Sequential(
        torch.nn.Linear(din, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, dout))
    return matlab_init(net) if INIT == "matlab" else net


def mlp_gelu_ln(din, dout, hidden=256, blocks=3):
    """quake: 3 x (fullyConnected -> layerNormalization -> gelu) -> out."""
    layers, d = [], din
    for _ in range(blocks):
        layers += [torch.nn.Linear(d, hidden), torch.nn.LayerNorm(hidden),
                   torch.nn.GELU()]
        d = hidden
    layers += [torch.nn.Linear(hidden, dout)]
    net = torch.nn.Sequential(*layers)
    return matlab_init(net) if INIT == "matlab" else net


class RFF:
    """`apply_rff(x, B) = [cos(2 pi B x); sin(2 pi B x)]`, fixed B."""

    def __init__(self, B):
        self.B = B

    def __call__(self, x):
        p = 2.0 * math.pi * (x @ self.B.T)
        return torch.cat([torch.cos(p), torch.sin(p)], -1)


# ============================================================================
# targets
# ============================================================================
class Bimodal:
    """E(x) = 6 (1 - x_3^2); grad_ambient = [0, 0, -12 x_3]."""

    name = "bimodal"

    def grad_riem(self, x, kappa=None):
        g = torch.zeros_like(x)
        g[:, 2] = -12.0 * x[:, 2]
        return proj(x, g)

    def energy(self, x):
        return 6.0 * (1.0 - x[:, 2] ** 2)


class QuakeTarget:
    """E(x) = -log (1/N) sum_i exp(kappa <m_i, x>), their vectorised form.

    kappa is passed per call because their training loop anneals it on a
    staircase while the mode set M stays fixed.
    """

    name = "quake"

    def __init__(self, M):
        self.M = M

    def grad_riem(self, x, kappa, chunk=4096):
        out = torch.empty_like(x)
        for a in range(0, x.shape[0], chunk):
            xb = x[a:a + chunk]
            d = xb @ self.M.T
            md = d.max(dim=1, keepdim=True).values
            w = torch.exp(kappa * (d - md))
            g = -(kappa * w) @ self.M / w.sum(1, keepdim=True)
            out[a:a + chunk] = proj(xb, g)
        return out

    def energy(self, x, kappa, chunk=4096):
        out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        for a in range(0, x.shape[0], chunk):
            d = x[a:a + chunk] @ self.M.T
            md = d.max(dim=1, keepdim=True).values
            out[a:a + chunk] = -(torch.log(torch.exp(kappa * (d - md)).mean(1))
                                 + kappa * md.squeeze(1))
        return out


# ============================================================================
# sigma schedules
# ============================================================================
class ConstSigma:
    """bimodal: sigma_t = 1, bridge std = sigma sqrt(t(1-t)), frac_t = t."""

    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.V1 = sigma ** 2

    def at(self, t):
        return torch.full_like(t, self.sigma)

    def bridge(self, t):
        std = self.sigma * torch.sqrt((t * (1.0 - t)).clamp_min(0.0))
        return t, std


class QuakeSigma:
    """quake: sigma(t) = a + b (1-t)^2 with the exact variance integral.

    V(t) = a^2 t + (2ab/3)(1 - (1-t)^3) + (b^2/5)(1 - (1-t)^5).  The bridge
    interpolates in V rather than in t, and its std is sqrt(V_t (V_1-V_t)/V_1).
    """

    def __init__(self, a=0.03, b=0.32):
        self.a, self.b = a, b
        self.V1 = self._V(torch.tensor(1.0, dtype=DT)).item()

    def _V(self, t):
        return (self.a ** 2) * t \
            + (2.0 * self.a * self.b / 3.0) * (1.0 - (1.0 - t) ** 3) \
            + (self.b ** 2 / 5.0) * (1.0 - (1.0 - t) ** 5)

    def at(self, t):
        return self.a + self.b * (1.0 - t) ** 2

    def bridge(self, t):
        Vt = self._V(t)
        std = torch.sqrt((Vt * (self.V1 - Vt) / self.V1).clamp_min(0.0))
        return Vt / self.V1, std


# ============================================================================
# Algorithm 1
# ============================================================================
def simulate(unet, rff, sched, steps, n, device, generator=None,
             max_drift=0.0, dtype=DT):
    """Their geodesic random walk.  Returns (X0, X1).

    bimodal:  X <- retract(X + u dt + sigma sqrt(dt) P_x(noise))
    quake:    X <- retract(X + sigma_t u_clipped dt + sigma_t P_x(dW))

    The two differ in where sigma multiplies the drift, and the quake script
    clips the projected drift norm to `max_drift`.  Both are reproduced.
    """
    dt = 1.0 / steps
    x0 = haar(n, device, generator, dtype)
    x = x0
    for s in range(steps):
        t = s * dt
        tt = torch.full((n, 1), t, device=device, dtype=dtype)
        with torch.no_grad():
            inp = torch.cat([x, tt], 1)
            u = unet(rff(inp) if rff is not None else inp).to(dtype)
        u = proj(x, u)
        if max_drift > 0.0:
            nrm = u.norm(dim=-1, keepdim=True)
            u = u * torch.clamp(max_drift / (nrm + 1e-8), max=1.0)
        noise = torch.randn(x.shape, device=device, dtype=dtype,
                            generator=generator)
        sig = float(sched.at(torch.tensor(t, dtype=dtype)))
        if max_drift > 0.0:                      # quake convention
            x = x + sig * u * dt + sig * proj(x, noise) * math.sqrt(dt)
        else:                                    # bimodal convention
            x = x + u * dt + sig * proj(x, noise) * math.sqrt(dt)
        x = retract(x)
    return x0, x


def adjoint_targets(x0, x1, hnet, rff_h, sched, target, kappa, generator=None):
    """Their `compute_adjoint_targets`: geodesic bridge + parallel transport."""
    n, device = x0.shape[0], x0.device
    t = torch.rand(n, 1, device=device, dtype=x0.dtype, generator=generator)
    with torch.no_grad():
        h = hnet(rff_h(x1) if rff_h is not None else x1).to(x0.dtype)
    h = proj(x1, h)

    gE = target.grad_riem(x1) if kappa is None else target.grad_riem(x1, kappa)
    a1 = gE + h

    d01 = (x0 * x1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(d01)
    u_un = x1 - x0 * d01
    # their `+ 1e-8` in the quake script, `/norm` in the bimodal one
    u_dir = u_un / (u_un.norm(dim=-1, keepdim=True) + 1e-8)

    frac, std = sched.bridge(t)
    mu = x0 * torch.cos(frac * theta) + u_dir * torch.sin(frac * theta)
    noise = torch.randn(mu.shape, device=device, dtype=mu.dtype,
                        generator=generator)
    # Gaussian in the ambient tangent space at mu, then retracted.  This is
    # the surrogate bridge; see the module docstring.
    xt = retract(mu + proj(mu, noise) * std)

    at = parallel_transport(a1, x1, xt)
    # their `if theta < 1e-5: Xt = x1; a_t = a_1` branch
    deg = (theta < 1e-5)
    xt = torch.where(deg, x1, xt)
    at = torch.where(deg, a1, at)
    return xt, t, at, sched.at(t)


def corrector_targets_bimodal(x0, x1, sched):
    """`log_map_{x1}(x0) / sigma^2`."""
    d01 = (x1 * x0).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(d01)
    u_un = x0 - x1 * d01
    u_rev = u_un / u_un.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    b = u_rev * theta / sched.V1
    return torch.where(theta > 1e-5, b, torch.zeros_like(b))


def corrector_targets_quake(x0, x1, sched):
    """`u_rev (theta/V1 + 1/2 (cot theta - 1/theta))`.

    The second term is the derivative of the log volume element on S^2; their
    small-angle branch is the series -theta/6 - theta^3/90, which is exactly
    the expansion of the first, so the two agree to O(theta^5).
    """
    d01 = (x1 * x0).sum(-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.arccos(d01)
    valid = theta > 1e-5
    th = theta.clamp(max=math.pi - 0.1)
    u_un = x0 - x1 * d01
    u_rev = u_un / (u_un.norm(dim=-1, keepdim=True) + 1e-8)
    small = th < 0.05
    vol = torch.where(
        small,
        -th / 6.0 - th ** 3 / 90.0,
        0.5 * (torch.cos(th) / torch.sin(th).clamp_min(1e-30) - 1.0 / th))
    b = u_rev * (th / sched.V1 + vol)
    return torch.where(valid, b, torch.zeros_like(b))


def train(args, target, device=DEV):
    """Their training loop: adjoint update, then corrector update, per epoch."""
    torch.manual_seed(args.seed)
    quake = (target.name == "quake")
    gen = torch.Generator(device=device).manual_seed(args.seed)

    if quake:
        sched = QuakeSigma()
        bw = torch.randn(args.n_freq, 4, device=device) * 3.0
        bw[:, 3] = torch.randn(args.n_freq, device=device) * 1.0
        rff_u = RFF(bw)
        rff_h = RFF(torch.randn(args.n_freq, 3, device=device) * 3.0)
        unet = mlp_gelu_ln(2 * args.n_freq, 3).to(device)
        hnet = mlp_gelu_ln(2 * args.n_freq, 3).to(device)
        lr_u, lr_h = 1e-3, 2e-3
        corrector = corrector_targets_quake
    else:
        sched = ConstSigma(args.sigma)
        rff_u = rff_h = None
        unet = mlp_tanh(4, 3, 64).to(device)
        hnet = mlp_tanh(3, 3, 48).to(device)
        lr_u = lr_h = 2e-3
        corrector = corrector_targets_bimodal

    optU = torch.optim.Adam(unet.parameters(), lr=lr_u)
    optH = torch.optim.Adam(hnet.parameters(), lr=lr_h)
    n_par = sum(p.numel() for p in unet.parameters()) \
        + sum(p.numel() for p in hnet.parameters())
    t0, hist = time.time(), []

    for ep in range(1, args.epochs + 1):
        # their staircase: 150 / 300 / 450 / 600 at 25 / 50 / 75% of training
        if quake:
            f = ep / args.epochs
            kap = 150.0 if f <= 0.25 else 300.0 if f <= 0.50 \
                else 450.0 if f <= 0.75 else args.kappa
        else:
            kap = None

        # --- STEP 1: adjoint update -------------------------------------
        x0, x1 = simulate(unet, rff_u, sched, args.steps, args.batch, device,
                          gen, args.max_drift)
        xt, tv, at, sig_t = adjoint_targets(x0, x1, hnet, rff_h, sched,
                                            target, kap, gen)
        inp = torch.cat([xt, tv], 1)
        up = unet(rff_u(inp) if rff_u is not None else inp).to(DT)
        err = proj(xt, up) + sig_t * at
        lossU = (err ** 2).sum(-1).mean()
        optU.zero_grad(set_to_none=True)
        lossU.backward()
        optU.step()

        # --- STEP 2: corrector update -----------------------------------
        x0n, x1n = simulate(unet, rff_u, sched, args.steps, args.batch, device,
                            gen, args.max_drift)
        b = corrector(x0n, x1n, sched)
        hp = hnet(rff_h(x1n) if rff_h is not None else x1n).to(DT)
        lossH = ((proj(x1n, hp) - b) ** 2).sum(-1).mean()
        optH.zero_grad(set_to_none=True)
        lossH.backward()
        optH.step()

        if ep % args.log_every == 0 or ep == 1:
            hist.append({"epoch": ep, "kappa": kap,
                         "loss_u": float(lossU), "loss_h": float(lossH),
                         "wall": time.time() - t0})
            k = "" if kap is None else f" | kappa {kap:6.1f}"
            print(f"  epoch {ep:5d}/{args.epochs}{k} | loss U {float(lossU):.5f}"
                  f" | loss H {float(lossH):.5f} | {time.time()-t0:.0f}s",
                  flush=True)

    return unet, rff_u, sched, hist, n_par


# ============================================================================
# internal consistency checks (not a claim about their algorithm, just a
# guarantee that this file implements what the .m file says)
# ============================================================================
def check(device=DEV):
    torch.manual_seed(0)
    ok = True

    x = haar(20000, device)
    d = float((x.norm(dim=-1) - 1.0).abs().max())
    north = float((x[:, 2] > 0).double().mean())
    p1 = d < 1e-12 and abs(north - 0.5) < 0.02
    print(f"  haar source on S^2            : |‖x‖-1|={d:.1e}, "
          f"north={north:.4f}  {'PASS' if p1 else 'FAIL'}")
    ok &= p1

    # parallel transport must be an isometry that lands in the target tangent
    a = proj(x[:100], torch.randn(100, 3, device=device, dtype=DT))
    y = haar(100, device)
    b = parallel_transport(a, x[:100], y)
    iso = float((a.norm(dim=-1) - b.norm(dim=-1)).abs().max())
    tang = float((b * y).sum(-1).abs().max())
    p2 = iso < 1e-10 and tang < 1e-10
    print(f"  parallel transport isometry   : d|a|={iso:.1e}, "
          f"tangency={tang:.1e}  {'PASS' if p2 else 'FAIL'}")
    ok &= p2

    # their bimodal energy gradient against a finite difference
    tgt = Bimodal()
    g = tgt.grad_riem(x[:200])
    eps = 1e-6
    fd = torch.zeros_like(g)
    for k in range(3):
        e = torch.zeros(3, device=device, dtype=DT)
        e[k] = eps
        fd[:, k] = (tgt.energy(retract(x[:200] + e))
                    - tgt.energy(retract(x[:200] - e))) / (2 * eps)
    err = float((g - proj(x[:200], fd)).abs().max())
    p3 = err < 1e-5
    print(f"  bimodal grad E vs central FD  : {err:.2e}  "
          f"{'PASS' if p3 else 'FAIL'}")
    ok &= p3

    # their variance integral against quadrature
    s = QuakeSigma()
    tg = torch.linspace(0, 1, 200001, dtype=DT)
    quad = float(np.trapezoid((s.at(tg) ** 2).numpy(), tg.numpy()))
    err = abs(quad - s.V1)
    p4 = err < 1e-9
    print(f"  quake V(1) vs quadrature      : |{s.V1:.8f}-{quad:.8f}|"
          f"={err:.1e}  {'PASS' if p4 else 'FAIL'}")
    ok &= p4

    # their small-angle branch against the closed form at the switch point
    th = torch.tensor([0.05, 0.049999], dtype=DT)
    exact = 0.5 * (torch.cos(th) / torch.sin(th) - 1.0 / th)
    series = -th / 6.0 - th ** 3 / 90.0
    err = float((exact - series).abs().max())
    p5 = err < 1e-8
    print(f"  cot series at the 0.05 switch : {err:.2e}  "
          f"{'PASS' if p5 else 'FAIL'}")
    ok &= p5

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ============================================================================
# runs
# ============================================================================
def run_bimodal(args):
    tgt = Bimodal()
    unet, rff_u, sched, hist, n_par = train(args, tgt)
    print(f"\n  sampling {args.n_samples:,} terminal points ...", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    _, x = simulate(unet, rff_u, sched, args.steps, args.n_samples, DEV, gen,
                    args.max_drift)
    xc = x.cpu()

    # measured with remeasure.py, the same code that produces our column
    grid, cdf, _ = R.z_grid()
    m2e, m4e, mEe, _ = R.z_moments()
    m = R.sphere_metrics(xc, grid, cdf, m2e, m4e, mEe)
    m["constraint"] = float((x.norm(dim=-1) - 1.0).abs().max())
    g2 = torch.Generator().manual_seed(0)
    floor = R.sphere_metrics(R.sample_exact(args.n_samples, g2), grid, cdf,
                             m2e, m4e, mEe)

    print(f"\n  north mass   {m['north']:.4f}   (exact 0.5, "
          f"iid {floor['north']:.4f})")
    print(f"  KS(x_3)      {m['KS_z']:.4f}   (iid floor {floor['KS_z']:.4f})")
    print(f"  W1(x_3)      {m['W1_z']:.5f}  (iid floor {floor['W1_z']:.5f})")
    print(f"  KS(azimuth)  {m['KS_phi']:.4f}   (iid floor "
          f"{floor['KS_phi']:.4f})")
    print(f"  <x_3^2>      {m['m2']:.5f}  (exact {m2e:.5f})")
    print(f"  <E>          {m['mE']:.5f}  (exact {mEe:.5f})")
    print(f"  |‖x‖-1|      {m['constraint']:.2e}")

    if args.ckpt_dir:
        C.save_ckpt(args.ckpt_dir, args.tag, net=unet, samples=xc,
                    extra={"metrics": m, "floor": floor, "params": n_par,
                           "commit": COMMIT})
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "final": m, "floor": floor,
                   "history": hist, "params": n_par, "commit": COMMIT}, f,
                  indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


def run_quake(args):
    tr, te, path = Q.load_quakes(args.data, args.train_frac, args.seed)
    print(f"  data {path}: {tr.shape[0]} train modes, {te.shape[0]} held out")
    tgt = QuakeTarget(tr.to(DEV).to(DT))
    unet, rff_u, sched, hist, n_par = train(args, tgt)
    print(f"\n  sampling {args.n_samples:,} terminal points ...", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    _, x = simulate(unet, rff_u, sched, args.steps, args.n_samples, DEV, gen,
                    args.max_drift)

    # scored against exact iid draws from the same vMF mixture, with the same
    # metric code our own earthquake run is scored with
    mix = Q.VMFMixture(tr.to(DEV).to(DT), args.kappa)
    g = torch.Generator(device=DEV).manual_seed(12345)
    ref = mix.sample(args.n_samples, generator=g)
    m = Q.quake_metrics(x, mix, ref, gen=g)
    floor = Q.quake_metrics(mix.sample(args.n_samples, generator=g), mix, ref,
                            gen=g)
    mix_te = Q.VMFMixture(te.to(DEV).to(DT), args.kappa)
    ref_te = mix_te.sample(args.n_samples, generator=g)
    m["test"] = Q.quake_metrics(x, mix_te, ref_te, gen=g)

    Q.report("rasbs ", m)
    Q.report("iid   ", floor)
    Q.report("heldout", m["test"])

    if args.ckpt_dir:
        C.save_ckpt(args.ckpt_dir, args.tag, net=unet, samples=x.cpu(),
                    extra={"metrics": m, "floor": floor, "params": n_par,
                           "commit": COMMIT})
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "final": m, "floor": floor,
                   "history": hist, "params": n_par, "log_Z": mix.log_Z(),
                   "data": path, "commit": COMMIT}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


COMMIT = "bb71d1496468658f4ff1c9995773968fd12706a8"


def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["bimodal", "quake"],
                    default="bimodal")
    ap.add_argument("--check", action="store_true",
                    help="internal consistency checks, no training")
    # their bimodal defaults; --problem quake overrides the ones that differ
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--max-drift", dest="max_drift", type=float, default=0.0)
    ap.add_argument("--kappa", type=float, default=600.0)
    ap.add_argument("--n-freq", dest="n_freq", type=int, default=256)
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=0.7)
    # their scripts plot 1500 / 700 points; we sample 100k so the metrics are
    # comparable with our own runs rather than dominated by sampling noise
    ap.add_argument("--n-samples", dest="n_samples", type=int, default=100000)
    ap.add_argument("--log-every", dest="log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    # Default "ckpt", never "": a run whose checkpoint is not
    # written cannot be re-measured later and must be repeated in full.
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    # "matlab" mirrors fullyConnectedLayer (Glorot, zero bias); "torch" is the
    # nn.Linear default that every results_rasbs_sphere_* JSON predating the
    # fidelity audit was produced with, kept so those runs stay reproducible.
    ap.add_argument("--init", choices=["matlab", "torch"], default="matlab")
    args = ap.parse_args()

    global INIT
    INIT = args.init

    if args.check:
        return check()

    if args.problem == "quake":
        # their earthquake_sphere_ex.m hyperparameters
        for k, v in [("epochs", 2000), ("steps", 250), ("max_drift", 30.0)]:
            if ap.get_default(k) == getattr(args, k):
                setattr(args, k, v)
    args.tag = args.tag or f"rasbs_sphere_{args.problem}"
    args.out = args.out or f"json/results_rasbs_sphere_{args.problem}.json"

    print(f"R-ASBS Alg. 1 port, problem={args.problem}, "
          f"upstream {COMMIT[:7]}", flush=True)
    return run_quake(args) if args.problem == "quake" else run_bimodal(args)


if __name__ == "__main__":
    raise SystemExit(main())
