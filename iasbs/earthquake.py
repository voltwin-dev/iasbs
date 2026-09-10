"""
Experiment E -- earthquake epicentres on S^2 (the R-ASBS application benchmark).

R-ASBS's `earthquake_sphere_ex.m` fits the USGS M4.5+ catalogue (4776 events,
70% train = 3343 modes) with the energy

    E(x) = -log[ (1/N) sum_i exp(kappa <m_i, x>) ],       kappa = 600,

and reports *a picture*: no KS, no TV, no mean energy, no reference sample.
That is not an accident -- for a general energy on S^2 there is nothing to
compare against.

But this particular energy is special, and the paper missed it:

    pi(x) \propto exp(-E(x)) = (1/N) sum_i exp(kappa <m_i, x>)

and  int_{S^2} exp(kappa <m, x>) dx = 4 pi sinh(kappa)/kappa  is INDEPENDENT
of m.  Therefore

    (i)  pi is EXACTLY the equal-weight von Mises-Fisher mixture
             pi = (1/N) sum_i vMF(m_i, kappa),
         so exact iid reference samples are available in closed form
         (the vMF radial CDF on S^2 inverts analytically), and
    (ii) the normalising constant is known exactly,
             log Z = log(4 pi sinh(kappa)/kappa).

So the one experiment R-ASBS could only draw, we can score.  Every metric in
this script is against an exact iid reference draw, with the finite-sample
iid-vs-iid floor reported next to it.

Source law
----------
R-ASBS starts from Haar (uniform) on S^2 and only diffuses locally
(sigma(t) = 0.03 + 0.32(1-t)^2, total heat time r_{0,1} ~ 0.014, i.e. a
9-degree spread).  A uniform source hides the source-tilting bias that the
paper's own construction incurs.  We start from a DIRAC source x_0 and take
sigma = sqrt(2) so that r_{0,1} = 1 and the reference reaches the whole
sphere.  The terminal function then carries the base-density correction

    f_1(y) \propto exp(-E(y)/tau) / p_{r_{0,1}}(x_0 . y),
    G(y) = grad_y log f_1(y)
         = -(1/tau) grad E(y) - grad_y log p_{r_{0,1}}(x_0 . y),

which is exactly the Dirac-source object used in `sphere.py`.  The heat
kernel, its tables, the reference bridge and the collapsed Killing readout
are imported from `sphere.py` / `common.py` unchanged -- this experiment adds
only the target.

Difficulty
----------
kappa = 600 puts the modes at ~1/sqrt(kappa) = 0.041 rad = 2.3 degrees and
makes the terminal score O(kappa).  This is the Stiefel high-beta regime
again: the discretisation error is O(sigma^2 kappa / N), so N must grow with
kappa.  We use the same staircase anneal as R-ASBS (150 -> 300 -> 450 -> 600)
with warm starts, plus a drift clip (R-ASBS clips at 30).

Subcommands
-----------
  verify   E0 -- data, energy/gradient, exact sampler, exact log Z, readout
  exact    E1 -- exact quadrature control at reduced kappa, step-size sweep
  train    E2 -- learned control at full kappa, metrics vs the exact mixture
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

# common.py and the shared json/ ckpt/ fig/ directories live at the
# repository root, one level up from this script.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C
import sphere as S

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

_DATA_CANDIDATES = ["data/query.csv", "../rasbs_ref/query.csv",
                    "rasbs_ref/query.csv"]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def find_data(path=""):
    if path:
        return path
    for p in _DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("query.csv not found; pass --data")


def load_quakes(path="", train_frac=0.7, seed=0, device=DEV):
    """USGS catalogue -> unit vectors, deterministic 70/30 split.

    Columns 1 and 2 of the USGS query export are latitude and longitude in
    degrees.  The split is a seeded permutation so that train and test cover
    the globe alike; R-ASBS takes the first 70% of the file, which is a
    *chronological* split and mixes a coverage change into the comparison.
    """
    path = find_data(path)
    lat, lon = [], []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            if len(row) < 3 or not row[1]:
                continue
            lat.append(float(row[1]))
            lon.append(float(row[2]))
    la = np.radians(np.asarray(lat))
    lo = np.radians(np.asarray(lon))
    M = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                  np.sin(la)], axis=1)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    n = M.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    ntr = int(n * train_frac)
    tr = torch.tensor(M[perm[:ntr]], device=device)
    te = torch.tensor(M[perm[ntr:]], device=device)
    return tr, te, path


# ----------------------------------------------------------------------------
# target:  an exactly-samplable vMF mixture
# ----------------------------------------------------------------------------
class VMFMixture:
    """pi(x) \\propto (1/N) sum_i exp(kappa <m_i, x>) on S^2.

    Exact because the vMF normaliser 4 pi sinh(kappa)/kappa does not depend on
    the mean direction, so the equal-weight energy mixture IS an equal-weight
    density mixture.
    """

    def __init__(self, M, kappa):
        self.M = M
        self.kappa = float(kappa)
        self.n = M.shape[0]
        self.device = M.device

    # -- energy ------------------------------------------------------------
    def energy(self, x, chunk=8192):
        """E(x) = -logsumexp_i(kappa <m_i,x>) + log N."""
        out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        for a in range(0, x.shape[0], chunk):
            d = x[a:a + chunk] @ self.M.T
            out[a:a + chunk] = -torch.logsumexp(self.kappa * d, dim=1)
        return out + math.log(self.n)

    def grad(self, x, chunk=8192):
        """Ambient grad E; the caller projects.  Softmax-weighted mean mode."""
        out = torch.empty_like(x)
        for a in range(0, x.shape[0], chunk):
            d = x[a:a + chunk] @ self.M.T
            w = torch.softmax(self.kappa * d, dim=1)
            out[a:a + chunk] = -self.kappa * (w @ self.M)
        return out

    # -- exact normalisation and iid sampling ------------------------------
    def log_Z(self):
        """log int exp(-E) dx = log(4 pi sinh(k)/k), stable for large k."""
        k = self.kappa
        return math.log(2.0 * math.pi / k) + k + math.log1p(-math.exp(-2.0 * k))

    def sample(self, n, generator=None):
        """Exact iid draw.  vMF on S^2 has cos-angle density
        (k/(2 sinh k)) e^{k w} on [-1,1], whose CDF inverts in closed form:
            w = 1 + log(u + (1-u) e^{-2k}) / k.
        """
        k, dev = self.kappa, self.device
        idx = torch.randint(self.n, (n,), device=dev, generator=generator)
        mu = self.M[idx]
        u = torch.rand(n, device=dev, generator=generator)
        w = 1.0 + torch.log(u + (1.0 - u) * math.exp(-2.0 * k)) / k
        # an orthonormal tangent frame at mu, built from the least-aligned axis
        e = torch.zeros_like(mu)
        e.scatter_(1, mu.abs().argmin(dim=1, keepdim=True), 1.0)
        a = C.sphere_project_tangent(mu, e)
        a = a / a.norm(dim=1, keepdim=True)
        b = torch.linalg.cross(mu, a)
        ph = 2.0 * math.pi * torch.rand(n, device=dev, generator=generator)
        s = torch.sqrt((1.0 - w * w).clamp(min=0.0))
        x = w[:, None] * mu + s[:, None] * (torch.cos(ph)[:, None] * a
                                            + torch.sin(ph)[:, None] * b)
        return x / x.norm(dim=1, keepdim=True)


# ----------------------------------------------------------------------------
# problem
# ----------------------------------------------------------------------------
class QuakeProblem:
    """Dirac-source adjoint-sampling problem for the vMF-mixture target.

    Identical plumbing to `sphere.SphereProblem` (same heat tables, same
    reference bridge, same collapsed readout); only f_1 changes.  The
    quadrature grid is built lazily because at kappa = 600 an exact-control
    run is hopeless anyway (the modes are 2.3 degrees wide) -- quadrature is
    used only for the reduced-kappa verification.
    """

    def __init__(self, mix, sigma=math.sqrt(2.0), tau=1.0, steps=256,
                 device=DEV, x0=None, nu=0, nphi=0):
        self.mix, self.sigma, self.tau = mix, sigma, tau
        self.steps, self.device = steps, device
        self.x0 = (torch.tensor([1.0, 0.0, 0.0], device=device)
                   if x0 is None else x0.to(device))
        self.ts = np.arange(steps) / steps
        self.r_t1 = np.array([C.heat_clock(t, 1.0, sigma) for t in self.ts])
        self.r01 = float(C.heat_clock(0.0, 1.0, sigma))
        self.tab = S.HeatTable(self.r_t1, device=device)
        self.Y = self.w = None
        if nu and nphi:
            self.build_quadrature(nu, nphi)

    # -- terminal gradient --------------------------------------------------
    def G(self, y):
        g = C.sphere_project_tangent(y, -self.mix.grad(y) / self.tau)
        c = (y @ self.x0).clamp(-1.0, 1.0)
        si = torch.zeros(c.shape, device=y.device, dtype=torch.long)
        k = self.tab.dlogp_dc(si, c)
        return g - k[:, None] * (self.x0[None, :] - c[:, None] * y)

    def log_f1(self, y):
        c = (y @ self.x0).clamp(-1.0, 1.0)
        si = torch.zeros(c.shape, device=y.device, dtype=torch.long)
        return -self.mix.energy(y) / self.tau - self.tab.logp_at(si, c)

    # -- exact control by quadrature (reduced kappa only) -------------------
    def build_quadrature(self, nu, nphi):
        self.Y, self.w = S.sphere_quadrature(nu, nphi, device=self.device)
        lf = self.log_f1(self.Y)
        self.logf1w = torch.log(self.w) + (lf - lf.max())
        return self

    def exact_score(self, s_idx, x, chunk=512):
        B = x.shape[0]
        if not torch.is_tensor(s_idx):
            s_idx = torch.full((B,), int(s_idx), device=x.device,
                               dtype=torch.long)
        out = torch.empty_like(x)
        for a in range(0, B, chunk):
            xb, sb = x[a:a + chunk], s_idx[a:a + chunk]
            c = (xb @ self.Y.T).clamp(-1.0, 1.0)
            si = sb[:, None].expand(-1, c.shape[1])
            lw = self.tab.logp_at(si, c) + self.logf1w[None, :]
            lw = lw - lw.max(dim=1, keepdim=True).values
            k = self.tab.dlogp_dc(si, c)
            wg = torch.exp(lw)
            phi = wg.sum(dim=1, keepdim=True)
            num = (wg * k) @ self.Y - ((wg * k * c).sum(dim=1, keepdim=True)
                                       * xb)
            out[a:a + chunk] = num / phi.clamp(min=1e-300)
        return C.sphere_project_tangent(x, out)


# ----------------------------------------------------------------------------
# simulation  (drift-clipped; R-ASBS clips at max_drift = 30)
# ----------------------------------------------------------------------------
def simulate(prob, score_fn, batch, steps, generator=None, max_drift=0.0):
    dt, sig = 1.0 / steps, prob.sigma
    x = prob.x0[None, :].expand(batch, 3).contiguous().clone()
    for s in range(steps):
        if score_fn is None:
            drift = torch.zeros_like(x)
        else:
            drift = sig * sig * score_fn(s, x)
            if max_drift > 0:
                nn = drift.norm(dim=-1, keepdim=True)
                drift = drift * (max_drift / nn.clamp(min=max_drift))
        xi = torch.randn(batch, 3, dtype=x.dtype, device=x.device,
                         generator=generator)
        v = C.sphere_project_tangent(
            x, drift * dt + sig * math.sqrt(dt) * xi)
        x = C.sphere_exp(x, v)
        x = x / x.norm(dim=-1, keepdim=True)
    return x


# ----------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------
class QuakeScoreNet(torch.nn.Module):
    """MLP on multi-scale Fourier features of x.

    The score field has structure at 1/sqrt(kappa) ~ 0.04 rad, so a plain
    coordinate MLP cannot resolve it.  We use random directions on a
    geometric bandwidth ladder (R-ASBS uses a single bandwidth of 3.0, which
    is 20x too coarse for kappa = 600).  Output is multiplied by `scale`
    ~ kappa so the trunk works on O(1) targets.
    """

    def __init__(self, hidden=384, n_dir=256, n_time=8, scale=1.0,
                 bw_lo=1.0, bw_hi=64.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(n_dir, 3, generator=g)
        d = d / d.norm(dim=1, keepdim=True)
        lad = torch.exp(torch.linspace(math.log(bw_lo), math.log(bw_hi),
                                       n_dir))
        self.register_buffer("B", (d * lad[:, None]).to(torch.float32))
        self.n_time, self.scale = n_time, float(scale)
        din = 3 + 2 + 2 * n_time + 2 * n_dir
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 3),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.to(torch.float32)

    def forward(self, t, x):
        xf, tf = x.to(torch.float32), t.to(torch.float32)[:, None]
        k = torch.arange(1, self.n_time + 1, device=xf.device,
                         dtype=xf.dtype)[None, :]
        pr = xf @ self.B.T
        f = torch.cat([xf, tf, 1.0 - tf,
                       torch.sin(math.pi * k * tf),
                       torch.cos(math.pi * k * tf),
                       torch.sin(pr), torch.cos(pr)], dim=-1)
        v = self.scale * self.net(f)
        return C.sphere_project_tangent(x, v.to(x.dtype))


def score_fn_of(net, steps):
    def f(s, xx):
        t = torch.full((xx.shape[0],), s / steps, device=xx.device,
                       dtype=xx.dtype)
        return net(t, xx)
    return f


# ----------------------------------------------------------------------------
# metrics -- everything against an exact iid draw from pi
# ----------------------------------------------------------------------------
def ks_two_sample(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def nearest_mode(x, M, chunk=4096):
    """(index, angle) of the closest catalogue mode for each sample."""
    idx = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    ang = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    for a in range(0, x.shape[0], chunk):
        d = (x[a:a + chunk] @ M.T).clamp(-1.0, 1.0)
        v, j = d.max(dim=1)
        idx[a:a + chunk], ang[a:a + chunk] = j, torch.arccos(v)
    return idx, ang


def energy_distance(a, b, n=4000, generator=None):
    """Szekely energy distance with geodesic ground metric, subsampled."""
    ia = torch.randperm(a.shape[0], device=a.device, generator=generator)[:n]
    ib = torch.randperm(b.shape[0], device=b.device, generator=generator)[:n]
    A, Bb = a[ia], b[ib]
    gd = lambda u, v: torch.arccos((u @ v.T).clamp(-1.0, 1.0))
    return float(2.0 * gd(A, Bb).mean() - gd(A, A).mean() - gd(Bb, Bb).mean())


def quake_metrics(x, mix, ref, gen=None):
    Ex = mix.energy(x)
    Er = mix.energy(ref)
    _, ax = nearest_mode(x, mix.M)
    _, ar = nearest_mode(ref, mix.M)
    hx = torch.bincount(nearest_mode(x, mix.M)[0], minlength=mix.n).double()
    hr = torch.bincount(nearest_mode(ref, mix.M)[0], minlength=mix.n).double()
    tv = 0.5 * float((hx / hx.sum() - hr / hr.sum()).abs().sum())
    return {
        "mean_E": float(Ex.mean()), "mean_E_ref": float(Er.mean()),
        "dE": float(Ex.mean() - Er.mean()),
        "KS_E": ks_two_sample(Ex.cpu().numpy(), Er.cpu().numpy()),
        "KS_theta": ks_two_sample(ax.cpu().numpy(), ar.cpu().numpy()),
        "mode_TV": tv,
        "coverage": float((hx > 0).double().mean()),
        "coverage_ref": float((hr > 0).double().mean()),
        "ED": energy_distance(x, ref, generator=gen),
        "constraint": float((x.norm(dim=-1) - 1.0).abs().max()),
    }


def report(tag, m):
    print(f"    {tag}  dE={m['dE']:+.4f}  KS(E)={m['KS_E']:.4f}  "
          f"KS(th)={m['KS_theta']:.4f}  modeTV={m['mode_TV']:.4f}  "
          f"cov={m['coverage']:.3f}/{m['coverage_ref']:.3f}  "
          f"ED={m['ED']:.2e}  |‖x‖-1|={m['constraint']:.1e}", flush=True)


# ----------------------------------------------------------------------------
# E0 -- verify
# ----------------------------------------------------------------------------
def run_verify(args):
    tr, te, path = load_quakes(args.data, args.train_frac, args.seed)
    print(f"data {path}: {tr.shape[0]} train + {te.shape[0]} test quakes")
    torch.manual_seed(args.seed)
    ok = True

    # (1) energy gradient vs autograd
    mix = VMFMixture(tr, args.kappa)
    x = torch.randn(64, 3, device=DEV)
    x = x / x.norm(dim=1, keepdim=True)
    xg = x.clone().requires_grad_(True)
    mix.energy(xg).sum().backward()
    e = float((mix.grad(x) - xg.grad).abs().max())
    print(f"  grad E vs autograd            : {e:.3e}")
    ok &= e < 1e-8

    # (2) exact log Z against quadrature (small kappa, where it converges)
    Yq, wq = S.sphere_quadrature(args.nu, args.nphi, device=DEV)
    for kap in [1.0, 5.0, 20.0]:
        m2 = VMFMixture(tr, kap)
        lq = float(torch.logsumexp(torch.log(wq) - m2.energy(Yq), dim=0))
        e = abs(lq - m2.log_Z())
        print(f"  log Z quad vs closed form k={kap:5.1f}: {e:.3e}")
        ok &= e < 1e-8

    # (3) exact sampler: mean resultant length of one vMF vs Langevin A_3
    for kap in [2.0, 50.0, args.kappa]:
        one = VMFMixture(tr[:1], kap)
        s = one.sample(400000)
        got = float((s @ tr[0]).mean())
        want = 1.0 / math.tanh(kap) - 1.0 / kap
        se = 3.0 / math.sqrt(400000) * float((s @ tr[0]).std())
        print(f"  vMF <cos> k={kap:6.1f}: {got:.6f} vs {want:.6f} "
              f"(3se {se:.1e})")
        ok &= abs(got - want) < max(se, 1e-6)

    # (4) mixture sampler vs quadrature CDF of E, at a kappa quadrature can see
    m2 = VMFMixture(tr, 20.0)
    s = m2.sample(200000)
    lw = torch.log(wq) - m2.energy(Yq)
    lw = lw - torch.logsumexp(lw, dim=0)
    Eq = m2.energy(Yq).cpu().numpy()
    o = np.argsort(Eq)
    cdf = np.cumsum(torch.exp(lw).cpu().numpy()[o])
    ks = C.ks_against_grid_cdf(m2.energy(s).cpu().numpy(), Eq[o], cdf)
    print(f"  mixture sampler KS(E) vs quadrature : {ks:.5f}")
    ok &= ks < 0.01

    # (5) readout tangency + f1 identity
    prob = QuakeProblem(mix, sigma=args.sigma, tau=args.tau, steps=64)
    y = mix.sample(4096)
    lab = C.sphere_terminal_readout(x[:1].expand(4096, 3).contiguous(), y,
                                    prob.G(y))
    e = float((lab * x[:1]).sum(-1).abs().max())
    print(f"  readout tangency              : {e:.3e}")
    ok &= e < 1e-9
    e = float((prob.G(y) * y).sum(-1).abs().max())
    print(f"  G tangency                    : {e:.3e}")
    ok &= e < 1e-9

    print(f"\n=== E0 {'PASS' if ok else 'FAIL'} ===")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "n_train": tr.shape[0],
                   "n_test": te.shape[0], "pass": bool(ok)}, f, indent=2,
                  default=str)
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# E1 -- exact quadrature control at reduced kappa
# ----------------------------------------------------------------------------
def run_exact(args):
    tr, te, path = load_quakes(args.data, args.train_frac, args.seed)
    mix = VMFMixture(tr, args.kappa)
    print(f"exact control, kappa={args.kappa:g}, quad {args.nu}x{args.nphi} "
          f"= {args.nu*args.nphi} nodes")
    torch.manual_seed(args.seed)
    ref = mix.sample(args.n_samples)
    ref2 = mix.sample(args.n_samples)
    fl = quake_metrics(ref2, mix, ref)
    report("iid floor  ", fl)
    out = {}
    for steps in args.sweep:
        prob = QuakeProblem(mix, sigma=args.sigma, tau=args.tau, steps=steps,
                            nu=args.nu, nphi=args.nphi)
        torch.manual_seed(args.seed)
        t0 = time.time()
        x = simulate(prob, lambda s, xx: prob.exact_score(s, xx),
                     args.n_samples, steps, max_drift=args.max_drift)
        m = quake_metrics(x, mix, ref)
        m["wall"] = time.time() - t0
        out[steps] = m
        report(f"steps={steps:4d}", m)
        C.save_ckpt(args.ckpt_dir, f"{args.tag}_exact_steps{steps}",
                    samples=x, extra={"metrics": m})
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "sweep": out, "floor": fl,
                   "data": path}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


# ----------------------------------------------------------------------------
# E2 -- learned control
# ----------------------------------------------------------------------------
def run_train(args):
    tr, te, path = load_quakes(args.data, args.train_frac, args.seed)
    steps = args.steps
    mix_f = VMFMixture(tr, args.kappa)
    mix_te = VMFMixture(te, args.kappa)
    torch.manual_seed(args.seed)
    ref = mix_f.sample(args.n_samples)
    ref2 = mix_f.sample(args.n_samples)
    floor = quake_metrics(ref2, mix_f, ref)
    print(f"data {path}: {tr.shape[0]} train modes, kappa={args.kappa:g}, "
          f"steps={steps}")
    report("iid floor  ", floor)
    print(f"  exact log Z = {mix_f.log_Z():.6f}")

    stages = args.anneal if args.anneal else [args.kappa]
    net = QuakeScoreNet(hidden=args.hidden, n_dir=args.n_dir,
                        scale=args.kappa, bw_hi=args.bw_hi,
                        seed=args.seed).to(DEV)
    ema_net = QuakeScoreNet(hidden=args.hidden, n_dir=args.n_dir,
                            scale=args.kappa, bw_hi=args.bw_hi,
                            seed=args.seed).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  params {n_par}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iters * args.inner), eta_min=args.lr * 1e-2)
    ema = S.EMA(net, decay=args.ema)
    hist, t0 = [], time.time()
    per = max(1, args.iters // len(stages))

    for it in range(1, args.iters + 1):
        kap = stages[min((it - 1) // per, len(stages) - 1)]
        mix = VMFMixture(tr, kap)
        prob = QuakeProblem(mix, sigma=args.sigma, tau=args.tau, steps=steps)
        with torch.no_grad():
            y1 = simulate(prob, score_fn_of(net, steps), args.batch, steps,
                          max_drift=args.max_drift)
            path_t = S.bridge_path(prob, y1, steps)              # (S,B,3)
            Gy = prob.G(y1)
            P = path_t.to(torch.float32)
            Y = y1.to(torch.float32)
            GG = Gy.to(torch.float32)
            B = Y.shape[0]
        for _ in range(args.inner):
            b = torch.randint(B, (args.mb,), device=DEV)
            s = torch.randint(steps, (args.mb,), device=DEV)
            xt = P[s, b].to(torch.float64)
            lab = C.sphere_terminal_readout(xt, Y[b].to(torch.float64),
                                            GG[b].to(torch.float64))
            pred = net(s.to(xt.dtype) / steps, xt)
            loss = (((pred - lab) / args.kappa) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(net)
        if it % args.eval_every == 0 or it == args.iters:
            ema.copy_to(ema_net)
            pf = QuakeProblem(mix_f, sigma=args.sigma, tau=args.tau,
                              steps=steps)
            with torch.no_grad():
                x = simulate(pf, score_fn_of(ema_net, steps), args.n_samples,
                             steps, max_drift=args.max_drift)
            m = quake_metrics(x, mix_f, ref)
            m.update({"iter": it, "kappa_stage": kap, "loss": float(loss),
                      "wall": time.time() - t0})
            hist.append(m)
            report(f"it {it:5d} k={kap:5.0f}", m)

    ema.copy_to(ema_net)
    pf = QuakeProblem(mix_f, sigma=args.sigma, tau=args.tau, steps=steps)
    with torch.no_grad():
        x = simulate(pf, score_fn_of(ema_net, steps), args.n_samples, steps,
                     max_drift=args.max_drift)
    m = quake_metrics(x, mix_f, ref)
    # generalisation: the same sampler scored against the held-out 30%
    ref_te = mix_te.sample(args.n_samples)
    m_te = quake_metrics(x, mix_te, ref_te)
    m["test"] = {k: m_te[k] for k in ("dE", "KS_E", "KS_theta", "mode_TV",
                                      "coverage", "ED")}
    m["params"] = n_par
    if args.ckpt_dir:
        p = C.save_ckpt(args.ckpt_dir, args.tag, net=ema_net, samples=x,
                        extra={"config": vars(args), "metrics": m,
                               "floor": floor})
        print(f"    ckpt -> {p}")
    print("\n=== GATES ===")
    g1 = m["KS_E"] < 5.0 * max(floor["KS_E"], 1e-9)
    g2 = m["mode_TV"] < 1.5 * floor["mode_TV"]
    g3 = m["constraint"] < 1e-12
    print(f"E1  KS(E) within 5x the iid floor : "
          f"{'PASS' if g1 else 'FAIL'}  ({m['KS_E']:.5f} vs "
          f"{floor['KS_E']:.5f})")
    print(f"E2  mode TV within 1.5x the floor : "
          f"{'PASS' if g2 else 'FAIL'}  ({m['mode_TV']:.5f} vs "
          f"{floor['mode_TV']:.5f})")
    print(f"E3  |‖x‖-1| < 1e-12               : "
          f"{'PASS' if g3 else 'FAIL'}  ({m['constraint']:.2e})")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "final": m, "history": hist,
                   "floor": floor, "log_Z": mix_f.log_Z(), "data": path},
                  f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (g1 and g2 and g3) else 1


# ----------------------------------------------------------------------------
def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "exact", "train"])
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--kappa", type=float, default=600.0)
    ap.add_argument("--anneal", type=float, nargs="*", default=None)
    ap.add_argument("--sigma", type=float, default=math.sqrt(2.0))
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--sweep", type=int, nargs="*",
                    default=[64, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--inner", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--n-dir", type=int, default=256)
    ap.add_argument("--bw-hi", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--max-drift", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=100000)
    ap.add_argument("--nu", type=int, default=256)
    ap.add_argument("--nphi", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="earthquake")
    ap.add_argument("--out", type=str, default="json/results_earthquake.json")
    args = ap.parse_args()
    return {"verify": run_verify, "exact": run_exact,
            "train": run_train}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
