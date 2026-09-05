"""
Experiment C -- sphere S^2  (PLAN.md section 6).

Reference process: Brownian motion on S^2 with generator (1/2) sigma^2 Delta,
started from a Dirac source x_0.  Integrated heat clock

    r_{t,1} = (1/2) sigma^2 (1 - t).

Target (the R-ASBS analytic benchmark):

    E(x) = 6 (1 - x_3^2),      pi(x) \propto exp(6 x_3^2),

which is symmetric in x_3 -> -x_3, so Pr_pi(x_3 > 0) = 1/2 exactly, and the
x_3 marginal has the closed form  p(z) \propto exp(6 z^2)  on [-1, 1].

Adjoint-Sampling terminal function for a Dirac source:

    f_1(y) \propto exp(-E(y)/tau) / p_base(y | x_0),
    G(y)   = grad_y log f_1(y)
           = -(1/tau) grad E(y) - grad_y log p_{r_{0,1}}(x_0 . y).

Killing-field collapsed readout (PLAN 6.4) -- the intertwining identity for
the rotation group acting on the sphere:

    grad_{S^2} log phi_t(x) = E[ (x.Y) G(Y) - Y (x.G(Y)) | X_t = x ],

expectation under the CONTROLLED law.  No Ricci damping factor.

What makes this script sharp
----------------------------
Everything is compared against an EXACT control, not just against samples.
phi_t is a two-dimensional integral against the spherical heat kernel, so it
is computed by product Gauss-Legendre quadrature on the sphere:

    phi_t(x)        = sum_q w_q p_r(x.Y_q) f1(Y_q)
    grad_S log phi  = sum_q w_q dp_r(x.Y_q) (Y_q - (x.Y_q) x) f1(Y_q) / phi

This gives (a) a ground-truth controller to sample from, isolating pure
discretisation error, and (b) a pointwise score-error metric for the learned
network -- the continuous analogue of the exact-law TV used in Experiments
A and B.

The heat kernel itself uses the truncated spectral series

    p_r(cos th) = (1/4pi) sum_l (2l+1) exp(-l(l+1) r) P_l(cos th)

evaluated once per time-grid point on a theta grid and then interpolated.
The interpolant stores log p (smooth) and the RESCALED score

    S(r, th) := r * (dp/p) * sin(th) / th   ->  1 + O(th^2, r)

which is O(1) and smooth even as r -> 0, where dp/p itself blows up like 1/r.

Subcommands
-----------
  verify   C0 -- readout tangency, kernel convergence, quadrature vs MC
  exact    C3 -- exact control, step-size sweep (discretisation floor)
  train    C1/C2 -- learned control, 5 seeds, hemisphere + KS(x3)
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_legendre

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# spherical heat kernel tables
# ----------------------------------------------------------------------------
def _lmax_for(r, decay=36.0, cap=400):
    """Smallest l with l(l+1) r > decay  (exp(-36) ~ 2e-16)."""
    l = 0.5 * (math.sqrt(1.0 + 4.0 * decay / max(r, 1e-12)) - 1.0)
    return int(min(cap, max(8, math.ceil(l) + 4)))


def heat_p_dp(theta, r, Lmax=None):
    """p_r(cos th) and d p_r / d c, numpy, shape of theta."""
    c = np.cos(theta)
    if Lmax is None:
        Lmax = _lmax_for(r)
    p = np.zeros_like(c)
    dp = np.zeros_like(c)
    Pm1 = np.zeros_like(c)
    P = np.ones_like(c)                        # P_0
    den = np.maximum(1.0 - c * c, 1e-13)
    for ell in range(Lmax + 1):
        coeff = (2 * ell + 1) * math.exp(-ell * (ell + 1) * r)
        p += coeff * P
        if ell >= 1:
            dp += coeff * ell * (Pm1 - c * P) / den
        # advance the Legendre recurrence
        Pp1 = ((2 * ell + 1) * c * P - ell * Pm1) / (ell + 1)
        Pm1, P = P, Pp1
    return p / (4.0 * math.pi), dp / (4.0 * math.pi)


def heat_asymptotic(theta, r):
    """Small-r Minakshisundaram-Pleijel expansion on S^2.

        p_r(th) ~ (4 pi r)^{-1} exp(-th^2/(4r)) (th/sin th)^{1/2} exp(r/6)

    (the Van Vleck determinant on the unit 2-sphere is (sin th / th), and the
    first heat coefficient is a_1 = R/12 = 1/6).  Returns (log p, dlog p/dth).
    This is used ONLY where the spectral sum has lost all significant digits
    to cancellation, i.e. where p is below ~1e-14 of its peak and the point
    contributes nothing to any expectation.
    """
    th = np.clip(theta, 1e-12, math.pi - 1e-9)
    s = np.sin(th)
    vv = 0.5 * np.log(th / np.maximum(s, 1e-14))
    logp = -math.log(4.0 * math.pi * r) - th * th / (4.0 * r) + vv + r / 6.0
    dth = -th / (2.0 * r) + 0.5 * (1.0 / th - np.cos(th) / np.maximum(s, 1e-14))
    return logp, dth


class HeatTable:
    """Interpolant of  log p_r  and  Q := r * dlog p_r/dth,  one row per r.

    The theta-parametrised score is the right object: dlog p/dth vanishes
    linearly at BOTH th = 0 and th = pi (p is a smooth function of cos th),
    and r * dlog p/dth -> -th/2 as r -> 0, so Q is O(1) and nearly linear.
    The c-parametrised score dlog p/dc = -(dlog p/dth)/sin th is the one that
    blows up, and it is only ever needed multiplied by a vector of norm sin th.
    """

    def __init__(self, rs, ntheta=8193, device=DEV):
        self.rs = np.asarray(rs, dtype=np.float64)
        self.device = device
        th = np.linspace(0.0, math.pi, ntheta)
        self.dth = float(th[1] - th[0])
        logp = np.empty((len(rs), ntheta))
        Q = np.empty((len(rs), ntheta))
        for a, r in enumerate(self.rs):
            r = float(r)
            p, dp = heat_p_dp(th, r)
            sn = np.sin(th)
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.log(np.maximum(p, 1e-300))
                q = r * (-sn * dp / np.maximum(p, 1e-300))
            # spectral cancellation floor: (2L+1) * eps relative to the peak
            good = np.isfinite(lp) & np.isfinite(q) & (p > 1e-11 * p.max())
            la, da = heat_asymptotic(th, r)
            lp = np.where(good, lp, la)
            q = np.where(good, q, r * da)
            logp[a], Q[a] = lp, q
        self.logp = torch.tensor(logp, device=device)
        self.Q = torch.tensor(Q, device=device)
        self.rt = torch.tensor(self.rs, device=device)

    def _interp(self, tab, s_idx, theta):
        """Linear interpolation in theta.  Flat indexing -- do NOT write
        tab[s_idx], which materialises a (..., ntheta) tensor."""
        nth = tab.shape[1]
        x = (theta / self.dth).clamp(0.0, nth - 1.0000001)
        i0 = x.floor().long()
        w = x - i0.to(x.dtype)
        flat = tab.reshape(-1)
        base = s_idx * nth + i0
        a = flat[base]
        b = flat[base + 1]
        return a + w * (b - a)

    def logp_at(self, s_idx, c):
        th = torch.arccos(c.clamp(-1.0, 1.0))
        return self._interp(self.logp, s_idx, th)

    def dlogp_dtheta(self, s_idx, c):
        th = torch.arccos(c.clamp(-1.0, 1.0))
        return self._interp(self.Q, s_idx, th) / self.rt[s_idx]

    def dlogp_dc(self, s_idx, c):
        """d log p_r / dc = -(d log p/dth)/sin th.

        Safe at both poles: the numerator vanishes linearly there, and the
        clamp on sin th only ever bites where sin th ~ th ~ 0, where the
        ratio is already the correct finite limit.
        """
        d = self.dlogp_dtheta(s_idx, c)
        sn = torch.sqrt((1.0 - c.clamp(-1.0, 1.0) ** 2).clamp(min=0.0))
        return -d / sn.clamp(min=1e-8)


# ----------------------------------------------------------------------------
# problem
# ----------------------------------------------------------------------------
def sphere_quadrature(nu=96, nphi=192, device=DEV):
    u, wu = np.polynomial.legendre.leggauss(nu)
    ph = (np.arange(nphi) + 0.5) * (2.0 * math.pi / nphi)
    U, PH = np.meshgrid(u, ph, indexing="ij")
    WU, _ = np.meshgrid(wu, ph, indexing="ij")
    s = np.sqrt(np.maximum(1.0 - U * U, 0.0))
    Y = np.stack([s * np.cos(PH), s * np.sin(PH), U], axis=-1).reshape(-1, 3)
    w = (WU * (2.0 * math.pi / nphi)).reshape(-1)
    return (torch.tensor(Y, device=device),
            torch.tensor(w, device=device))


class SphereProblem:
    def __init__(self, sigma=math.sqrt(2.0), tau=1.0, steps=128,
                 nu=96, nphi=192, device=DEV, x0=None):
        self.sigma, self.tau, self.steps, self.device = sigma, tau, steps, device
        self.x0 = torch.tensor([1.0, 0.0, 0.0], device=device,
                               dtype=torch.float64) if x0 is None else x0
        self.ts = np.arange(steps) / steps
        # r_{t,1} on the grid  (+ the terminal clock r_{0,1})
        self.r_t1 = np.array([C.heat_clock(t, 1.0, sigma) for t in self.ts])
        self.r01 = float(C.heat_clock(0.0, 1.0, sigma))
        self.tab = HeatTable(self.r_t1, device=device)
        self.Y, self.w = sphere_quadrature(nu, nphi, device=device)
        # f1 on the quadrature nodes
        c0 = (self.Y @ self.x0).clamp(-1.0, 1.0)
        si0 = torch.zeros(c0.shape, device=device, dtype=torch.long)
        self.logp0_nodes = self.tab.logp_at(si0, c0)
        self.logf1_nodes = (-C.sphere_energy(self.Y) / tau) - self.logp0_nodes
        self.logf1_nodes = self.logf1_nodes - self.logf1_nodes.max()
        self.logf1w = torch.log(self.w) + self.logf1_nodes       # (Q,)
        self.f1w = torch.exp(self.logf1w)

    # -- terminal gradient ---------------------------------------------------
    def G(self, y):
        """Ambient representative of grad_y log f1(y) = -gradE/tau - grad log p0."""
        g = C.sphere_project_tangent(y, -C.sphere_energy_grad(y) / self.tau)
        c = (y @ self.x0).clamp(-1.0, 1.0)
        si = torch.zeros(c.shape, device=y.device, dtype=torch.long)   # r_t1[0]=r01
        k = self.tab.dlogp_dc(si, c)
        return g - k[:, None] * (self.x0[None, :] - c[:, None] * y)

    # -- exact control by quadrature ----------------------------------------
    def exact_score(self, s_idx, x, chunk=2048):
        """grad_S log phi_t(x) at grid index s_idx (int or (B,) tensor)."""
        B = x.shape[0]
        if not torch.is_tensor(s_idx):
            s_idx = torch.full((B,), int(s_idx), device=x.device,
                               dtype=torch.long)
        out = torch.empty_like(x)
        for a in range(0, B, chunk):
            xb = x[a:a + chunk]
            sb = s_idx[a:a + chunk]
            c = (xb @ self.Y.T).clamp(-1.0, 1.0)                  # (b, Q)
            si = sb[:, None].expand(-1, c.shape[1])
            lw = self.tab.logp_at(si, c) + self.logf1w[None, :]
            lw = lw - lw.max(dim=1, keepdim=True).values          # log-domain
            k = self.tab.dlogp_dc(si, c)
            wgt = torch.exp(lw)                                   # (b, Q)
            phi = wgt.sum(dim=1, keepdim=True)
            num = (wgt * k) @ self.Y - ((wgt * k * c).sum(dim=1, keepdim=True)
                                        * xb)
            out[a:a + chunk] = num / phi.clamp(min=1e-300)
        return C.sphere_project_tangent(x, out)

    def log_phi(self, s_idx, x):
        """log phi_t(x) up to the (t-independent) f1 normalisation."""
        B = x.shape[0]
        if not torch.is_tensor(s_idx):
            s_idx = torch.full((B,), int(s_idx), device=x.device,
                               dtype=torch.long)
        c = (x @ self.Y.T).clamp(-1.0, 1.0)
        si = s_idx[:, None].expand(-1, c.shape[1])
        lw = self.tab.logp_at(si, c) + self.logf1w[None, :]
        return torch.logsumexp(lw, dim=1)

    def sample_terminal_given_x(self, s_idx, x, generator=None, chunk=4096):
        """Y ~ controlled law  p_r(x.y) f1(y) w  on the quadrature grid."""
        B = x.shape[0]
        if not torch.is_tensor(s_idx):
            s_idx = torch.full((B,), int(s_idx), device=x.device,
                               dtype=torch.long)
        out = torch.empty_like(x)
        for a in range(0, B, chunk):
            xb = x[a:a + chunk]
            c = (xb @ self.Y.T).clamp(-1.0, 1.0)
            si = s_idx[a:a + chunk, None].expand(-1, c.shape[1])
            lw = self.tab.logp_at(si, c) + self.logf1w[None, :]
            lw = lw - lw.max(dim=1, keepdim=True).values
            j = torch.multinomial(torch.exp(lw), 1, generator=generator)[:, 0]
            out[a:a + chunk] = self.Y[j]
        return out


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------
def simulate(prob, score_fn, batch, steps, generator=None, keep_path=False):
    """Geodesic random walk for the controlled BM on S^2."""
    dt = 1.0 / steps
    sig = prob.sigma
    x = prob.x0[None, :].expand(batch, 3).contiguous().clone()
    path = [] if keep_path else None
    for s in range(steps):
        if keep_path:
            path.append(x.clone())
        drift = sig * sig * score_fn(s, x) * dt if score_fn is not None else 0.0
        xi = torch.randn(batch, 3, dtype=x.dtype, device=x.device,
                         generator=generator)
        noise = sig * math.sqrt(dt) * C.sphere_project_tangent(x, xi)
        v = C.sphere_project_tangent(x, drift + noise) if score_fn is not None \
            else noise
        x = C.sphere_exp(x, v)
        x = x / x.norm(dim=-1, keepdim=True)
    if keep_path:
        return x, torch.stack(path)
    return x


def bridge_path(prob, y1, steps, generator=None):
    """Reference Brownian bridge x_0 -> y1 on S^2, all grid points.

    Doob h-transform of the reference with h(x, t) = p_{r_{t,1}}(x . y1),
    whose score is exact from the spectral table.
    """
    sig = prob.sigma
    dt = 1.0 / steps
    B = y1.shape[0]
    x = prob.x0[None, :].expand(B, 3).contiguous().clone()
    out = []
    for s in range(steps):
        out.append(x.clone())
        c = (x * y1).sum(dim=-1)
        si = torch.full((B,), s, device=x.device, dtype=torch.long)
        k = prob.tab.dlogp_dc(si, c)
        sc = k[:, None] * (y1 - c[:, None] * x)                   # grad_S log h
        xi = torch.randn(B, 3, dtype=x.dtype, device=x.device,
                         generator=generator)
        v = sig * sig * sc * dt + sig * math.sqrt(dt) * \
            C.sphere_project_tangent(x, xi)
        x = C.sphere_exp(x, C.sphere_project_tangent(x, v))
        x = x / x.norm(dim=-1, keepdim=True)
    return torch.stack(out)                                       # (steps,B,3)


# ----------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------
class ScoreNet(torch.nn.Module):
    """MLP score.  With symmetric=True the z -> -z symmetry of the problem
    (both E and x_0 are invariant under R = diag(1,1,-1)) is imposed as a hard
    equivariance  s(Rx) = R s(x).  That is an ABLATION, not the headline
    result: it makes the hemisphere mass exactly 1/2 by construction, so the
    honest comparison against R-ASBS uses symmetric=False."""

    def __init__(self, hidden=256, n_freq=6, symmetric=False):
        super().__init__()
        self.n_freq = n_freq
        self.symmetric = symmetric
        din = 3 + 2 + 2 * n_freq
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 3),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.to(torch.float32)

    def _raw(self, xf, tf):
        k = torch.arange(1, self.n_freq + 1, device=xf.device,
                         dtype=xf.dtype)[None, :]
        f = torch.cat([xf, tf, 1.0 - tf, torch.sin(math.pi * k * tf),
                       torch.cos(math.pi * k * tf)], dim=-1)
        return self.net(f)

    def forward(self, t, x):
        xf = x.to(torch.float32)
        tf = t.to(torch.float32)[:, None]
        v = self._raw(xf, tf)
        if self.symmetric:
            R = torch.tensor([1.0, 1.0, -1.0], device=xf.device,
                             dtype=xf.dtype)[None, :]
            v = 0.5 * (v + R * self._raw(R * xf, tf))
        return C.sphere_project_tangent(x, v.to(x.dtype))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def z_metrics(x):
    z = x[:, 2].cpu().numpy()
    grid, cdf = C.exact_s2_z_cdf_grid(num=40001)
    ks = C.ks_against_grid_cdf(z, grid, cdf)
    north = float((z > 0).mean())
    return {"north_mass": north, "north_err": abs(north - 0.5), "KS_z": ks,
            "constraint": float((x.norm(dim=-1) - 1.0).abs().max())}


def report(tag, m):
    print(f"    {tag}  north={m['north_mass']:.4f} "
          f"(err {m['north_err']:.4f})  KS(x3)={m['KS_z']:.4f}  "
          f"|‖x‖-1|={m['constraint']:.2e}")


# ----------------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------------
def run_verify(args):
    prob = SphereProblem(sigma=args.sigma, tau=args.tau, steps=args.steps)
    torch.manual_seed(args.seed)
    print(f"sigma={prob.sigma:.4f}  r01={prob.r01:.4f}  steps={prob.steps}  "
          f"quad={prob.Y.shape[0]} nodes")
    ok = True

    # (a) kernel normalisation and spectral convergence
    worst_n, worst_c = 0.0, 0.0
    for r in [0.01, 0.05, 0.2, 1.0]:
        # mass = 2 pi int_{-1}^{1} p_r(u) du -- Gauss-Legendre in u = cos th
        # is exact for the (polynomial) truncated spectral sum, so any
        # deviation is the kernel's, not the quadrature's.
        u, wu = np.polynomial.legendre.leggauss(2 * _lmax_for(r) + 8)
        pu, _ = heat_p_dp(np.arccos(np.clip(u, -1, 1)), r)
        worst_n = max(worst_n, abs(2 * math.pi * float((wu * pu).sum()) - 1.0))
        th = np.linspace(0, math.pi, 20001)
        p2, dp2 = heat_p_dp(th, r, Lmax=_lmax_for(r) * 2)
        p1, dp1 = heat_p_dp(th, r)
        worst_c = max(worst_c, float(np.abs(p2 - p1).max() / p1.max()))
    print(f"  |kernel mass - 1|            = {worst_n:.3e}")
    print(f"  rel. change on doubling Lmax = {worst_c:.3e}")
    ok &= worst_n < 1e-9 and worst_c < 1e-10

    # (b) table interpolation vs direct spectral evaluation, on the region
    #     where the spectral sum itself is trustworthy (p > 1e-13 of peak)
    th = torch.rand(4000, device=prob.device, dtype=torch.float64) * math.pi
    for s in [0, prob.steps // 4, prob.steps // 2, prob.steps - 1]:
        r = float(prob.r_t1[s])
        si = torch.full((4000,), s, device=prob.device, dtype=torch.long)
        c = torch.cos(th)
        thn = th.cpu().numpy()
        pr, dpr = heat_p_dp(thn, r)
        m = pr > 1e-9 * pr.max()
        mt = torch.tensor(m, device=prob.device)
        lp = prob.tab.logp_at(si, c)[mt]
        d = prob.tab.dlogp_dtheta(si, c)[mt]
        ref_lp = torch.tensor(np.log(pr[m]), device=prob.device)
        ref_d = torch.tensor(-np.sin(thn[m]) * dpr[m] / pr[m], device=prob.device)
        e1 = float((lp - ref_lp).abs().max())
        e2 = float((d - ref_d).abs().max() * r)   # compare  r*dlogp/dth = O(1)
        print(f"  s={s:4d} r={r:.5f}  n={int(m.sum())}/4000  "
              f"max|logp| err = {e1:.3e}   max|r dlogp/dth| err = {e2:.3e}")
        ok &= e1 < 1e-4 and e2 < 1e-4

    # (c) Killing readout is tangent at x
    x = C.random_stiefel(64, 3, 1, device=prob.device)[:, :, 0]
    y = C.random_stiefel(64, 3, 1, device=prob.device)[:, :, 0]
    G = torch.randn(64, 3, dtype=x.dtype, device=x.device)
    out = C.sphere_terminal_readout(x, y, G)
    tang = float((x * out).sum(-1).abs().max())
    print(f"  |x . readout|                = {tang:.3e}")
    ok &= tang < 1e-12

    # (d) THE test: MC average of the Killing readout == quadrature exact
    #     score.  Judged by the MC standard error (the score is genuinely
    #     tiny at small t, so a relative test there is meaningless).
    worst = 0.0
    for s in [0, prob.steps // 4, prob.steps // 2, 3 * prob.steps // 4]:
        npts = 8
        xs = C.random_stiefel(npts, 3, 1, device=prob.device)[:, :, 0]
        ex = prob.exact_score(s, xs)
        nrep = args.mc
        xr = xs[None, :, :].expand(nrep, npts, 3).reshape(-1, 3).contiguous()
        yy = prob.sample_terminal_given_x(s, xr)
        rd = C.sphere_terminal_readout(xr, yy, prob.G(yy)).view(nrep, npts, 3)
        mu = rd.mean(dim=0)
        se = rd.std(dim=0) / math.sqrt(nrep)
        z = float(((mu - ex).abs() / se.clamp(min=1e-300)).max())
        worst = max(worst, z)
        print(f"  s={s:4d}  MC readout vs quadrature score: max z = {z:.2f}"
              f"   (|score|~{float(ex.norm(dim=-1).mean()):.3f},"
              f" SE~{float(se.mean()):.4f})")
    ok &= worst < 5.0

    print(f"\nC0 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def run_exact(args):
    print(f"exact control (quadrature ground truth), sigma={args.sigma:.4f}")
    out = {}
    for steps in [32, 64, 128, 256, 512]:
        prob = SphereProblem(sigma=args.sigma, tau=args.tau, steps=steps)
        torch.manual_seed(args.seed)
        x = simulate(prob, lambda s, xx: prob.exact_score(s, xx),
                     args.n_samples, steps)
        m = z_metrics(x)
        out[steps] = m
        report(f"steps={steps:4d}", m)
        C.save_ckpt(args.ckpt_dir, f"{args.tag}_exact_steps{steps}",
                    samples=x.to(torch.float32), extra={"metrics": m})
    prob = SphereProblem(sigma=args.sigma, tau=args.tau, steps=128)
    torch.manual_seed(args.seed)
    xr = simulate(prob, None, args.n_samples, 128)
    report("reference   ", z_metrics(xr))
    C.save_ckpt(args.ckpt_dir, f"{args.tag}_reference",
                samples=xr.to(torch.float32),
                extra={"metrics": z_metrics(xr)})
    # finite-sample KS floor
    grid, cdf = C.exact_s2_z_cdf_grid(num=40001)
    u = np.random.rand(args.n_samples)
    zz = np.interp(u, cdf, grid)
    floor = C.ks_against_grid_cdf(zz, grid, cdf)
    print(f"  iid finite-sample KS floor at N={args.n_samples}: {floor:.5f}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "sweep": out, "ks_floor": floor},
                  f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


class EMA:
    """Polyak averaging of the weights.

    The regression labels here have a standard deviation of ~3 while the
    score they estimate is ~0.01-6, so the SGD iterate never settles -- the
    loss floor IS the label variance.  Averaging the iterate is what turns a
    correct-in-expectation estimator into a usable controller.
    """

    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net):
        d = self.decay
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, net):
        net.load_state_dict(self.shadow)


def _score_fn(net, steps):
    def f(s, xx):
        t = torch.full((xx.shape[0],), s / steps, device=xx.device,
                       dtype=xx.dtype)
        return net(t, xx)
    return f


def run_train(args):
    prob = SphereProblem(sigma=args.sigma, tau=args.tau, steps=args.steps)
    steps = args.steps
    all_m, hist = [], []
    for seed in range(args.seeds):
        torch.manual_seed(1000 + seed)
        net = ScoreNet(hidden=args.hidden, symmetric=args.symmetrize
                       ).to(prob.device)
        ema_net = ScoreNet(hidden=args.hidden, symmetric=args.symmetrize
                           ).to(prob.device)
        n_par = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.iters * args.inner, eta_min=args.lr * 1e-2)
        ema = EMA(net, decay=args.ema)
        # replay buffer of the last `nbuf` outer iterations (stored fp32)
        buf = []
        t0 = time.time()
        for it in range(1, args.iters + 1):
            with torch.no_grad():
                y1 = simulate(prob, _score_fn(net, steps), args.batch, steps)
                path = bridge_path(prob, y1, steps)               # (S,B,3)
                Gy = prob.G(y1)
                buf.append((path.to(torch.float32), y1.to(torch.float32),
                            Gy.to(torch.float32)))
                if len(buf) > args.nbuf:
                    buf.pop(0)
                P = torch.cat([b[0] for b in buf], dim=1)
                Y = torch.cat([b[1] for b in buf], dim=0)
                GG = torch.cat([b[2] for b in buf], dim=0)
                B = Y.shape[0]
            for _ in range(args.inner):
                b = torch.randint(B, (args.mb,), device=prob.device)
                s = torch.randint(steps, (args.mb,), device=prob.device)
                xt = P[s, b].to(torch.float64)
                yb, gb = Y[b].to(torch.float64), GG[b].to(torch.float64)
                if args.antithetic:
                    # R = diag(1,1,-1) is an isometry fixing x_0 and E, so
                    # (R path, R y1, R G) is an exact sample from the same
                    # law.  Pure data augmentation -- no constraint on the
                    # network, unlike --symmetrize.
                    f = (torch.rand(xt.shape[0], 1, device=xt.device) < 0.5)
                    sgn = torch.where(f, -1.0, 1.0).to(xt.dtype)
                    xt = xt.clone(); yb = yb.clone(); gb = gb.clone()
                    xt[:, 2] *= sgn[:, 0]
                    yb[:, 2] *= sgn[:, 0]
                    gb[:, 2] *= sgn[:, 0]
                lab = C.sphere_terminal_readout(xt, yb, gb)
                pred = net(s.to(xt.dtype) / steps, xt)
                loss = ((pred - lab) ** 2).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                sched.step()
                ema.update(net)
            if it % args.eval_every == 0 or it == args.iters:
                ema.copy_to(ema_net)
                with torch.no_grad():
                    x = simulate(prob, _score_fn(ema_net, steps),
                                 args.n_samples, steps)
                m = z_metrics(x)
                m.update({"seed": seed, "iter": it, "loss": float(loss)})
                hist.append(m)
                print(f"  seed {seed} it {it:4d}  loss {float(loss):8.4f}  "
                      f"north {m['north_mass']:.4f}  KS(x3) {m['KS_z']:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        ema.copy_to(ema_net)
        # score error against the exact control
        with torch.no_grad():
            errs = []
            for s in [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]:
                xs = C.random_stiefel(2048, 3, 1, device=prob.device)[:, :, 0]
                ex = prob.exact_score(s, xs)
                le = ema_net(torch.full((2048,), s / steps, device=prob.device,
                                        dtype=xs.dtype), xs)
                # normalised by the RMS score magnitude, not pointwise:
                # at small t the exact score is ~0.01 and a pointwise
                # relative error says nothing.
                rel = float((le - ex).norm(dim=-1).pow(2).mean().sqrt()
                            / ex.norm(dim=-1).pow(2).mean().sqrt().clamp(min=1e-12))
                errs.append((s / steps, rel))
        m = hist[-1]
        m["score_rel_err"] = errs
        m["params"] = n_par
        all_m.append(m)
        if args.ckpt_dir:
            with torch.no_grad():
                xs_fin = simulate(prob, _score_fn(ema_net, steps),
                                  args.n_samples, steps)
            p = C.save_ckpt(args.ckpt_dir, f"{args.tag}_seed{seed}",
                            net=ema_net, samples=xs_fin.to(torch.float32),
                            extra={"config": vars(args), "metrics": m,
                                   "score_rel_err": errs})
            print(f"    ckpt -> {p}", flush=True)
        print("    score rel-err vs exact: " +
              "  ".join(f"t={t:.2f}:{e:.3f}" for t, e in errs), flush=True)

    north_err = float(np.mean([m["north_err"] for m in all_m]))
    ks = float(np.mean([m["KS_z"] for m in all_m]))
    cons = float(np.max([m["constraint"] for m in all_m]))
    print(f"\n  mean over {args.seeds} seeds: hemisphere err {north_err:.4f}   "
          f"KS(x3) {ks:.4f}   max |‖x‖-1| {cons:.2e}")
    print("\n=== GATES ===")
    print(f"C1  mean hemisphere err < 0.03 (R-ASBS 0.062): "
          f"{'PASS' if north_err < 0.03 else 'FAIL'}  ({north_err:.4f})")
    print(f"C2  KS(x3) < 0.05                            : "
          f"{'PASS' if ks < 0.05 else 'FAIL'}  ({ks:.4f})")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "seeds": all_m, "history": hist,
                   "north_err": north_err, "KS_z": ks}, f, indent=2,
                  default=str)
    print(f"  wrote {args.out}")
    return 0 if (north_err < 0.03 and ks < 0.05) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "exact", "train"])
    ap.add_argument("--sigma", type=float, default=math.sqrt(2.0))
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=100000)
    ap.add_argument("--mc", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--nbuf", type=int, default=4)
    ap.add_argument("--symmetrize", action="store_true")
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--ckpt-dir", type=str, default="")
    ap.add_argument("--tag", type=str, default="sphere")
    ap.add_argument("--out", type=str, default="results_sphere.json")
    args = ap.parse_args()
    return {"verify": run_verify, "exact": run_exact,
            "train": run_train}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
