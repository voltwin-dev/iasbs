"""Experiment D -- Adjoint Schrodinger Bridge sampling on the Stiefel
manifold St(4,2), via the canonical Killing readout and the *corrected*
spin-factor clock  s = r/2  (PLAN.md sections 7.1-7.10).

Structure mirrors sphere.py:

    verify   gate D0   -- all numerical unit tests, incl. the mandatory
                          first-moment test  E[X_r|X_0] = exp(-3 r) X_0
    train    gates D1/D2 -- learned control vs MCMC ground truth
    sweep    gate D3   -- step-count discretisation sweep

Everything below works in the quaternion double cover  Spin(4) = SU(2)xSU(2),
    R(a,b) x = a x conj(b),          x in H = R^4,
so the SO(4) heat kernel factorises into TWO S^3 heat kernels, each run at
spin time  s = r/2.  The Stiefel kernel is the SO(2)-fibre integral of that
product over the stabiliser of E0 = [1, i].
"""

import argparse
import json
import math
import time

import numpy as np
import torch

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


# ============================================================================
# quaternion algebra   q = (w, x, y, z),  tensors of shape (..., 4)
# ============================================================================
def qmul(p, q):
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    ], dim=-1)


def qconj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _q(vals, device=DEV):
    return torch.tensor(vals, dtype=torch.float64, device=device)


def quat_consts(device=DEV):
    return (_q([1., 0., 0., 0.], device), _q([0., 1., 0., 0.], device),
            _q([0., 0., 1., 0.], device), _q([0., 0., 0., 1.], device))


def quat_to_matrix(a, b):
    """The SO(4) matrix of  x -> a x conj(b),  built column by column."""
    n = a.shape[:-1]
    I4 = torch.eye(4, dtype=a.dtype, device=a.device)
    cols = []
    for m in range(4):
        e = I4[m].expand(*n, 4)
        cols.append(qmul(qmul(a, e), qconj(b)))
    return torch.stack(cols, dim=-1)                    # (..., 4, 4)


def stiefel_from_quat(a, b):
    """R(a,b) E0  where E0 = [e1, e2] = [1, i]  (columns as quaternions)."""
    qi = _q([0., 1., 0., 0.], a.device)
    bc = qconj(b)
    y1 = qmul(a, bc)
    y2 = qmul(qmul(a, qi.expand_as(a)), bc)
    return torch.stack([y1, y2], dim=-1)                # (..., 4, 2)


def section_lift(Y):
    """A measurable section  Y -> (a_Y, b_Y)  with  R(a_Y,b_Y) E0 = Y.

    y1 = a conj(b),  y2 = a i conj(b)   ==>   conj(y1) y2 = b i conj(b) =: w.
    So b is any unit quaternion whose conjugation carries i to w -- a Hopf
    section, which cannot be globally continuous, so we use two charts.
    """
    dev = Y.device
    q1, qi, qj, _ = quat_consts(dev)
    y1, y2 = Y[..., 0], Y[..., 1]
    w = qmul(qconj(y1), y2)                              # unit pure imaginary

    one = q1.expand_as(y1)
    ii = qi.expand_as(y1)
    jj = qj.expand_as(y1)

    c1 = one - qmul(w, ii)                               # chart 1: fails at w=-i
    n1 = c1.norm(dim=-1, keepdim=True)

    wp = qmul(qmul(qconj(jj), w), jj)                    # chart 2 via w' = j^-1 w j
    c2 = one - qmul(wp, ii)
    c2 = qmul(jj, c2 / c2.norm(dim=-1, keepdim=True).clamp(min=1e-30))

    use1 = (n1 > 1e-3)
    b = torch.where(use1, c1 / n1.clamp(min=1e-30), c2)
    b = b / b.norm(dim=-1, keepdim=True)
    a = qmul(y1, b)
    return a, b


# ============================================================================
# S^3 heat kernel      p_s(theta),  spectral (stable) + exact-k0 fallback
# ============================================================================
def _nmax_for(s, decay=42.0, cap=800):
    return int(min(cap, max(8, math.ceil(math.sqrt(decay / max(s, 1e-9) + 1.0)))))


def s3_spectral_np(u, s, Nmax=None):
    """p(u) and dp/du,   p = (2 pi^2)^-1 sum_{n>=1} n e^{-(n^2-1)s} U_{n-1}(u).

    Chebyshev-U recurrence: polynomial in u = cos(theta), hence perfectly
    stable at both poles (unlike sin(n th)/sin(th) evaluated directly).
    """
    N = Nmax or _nmax_for(s)
    u = np.asarray(u, dtype=np.float64)
    p = np.zeros_like(u)
    dp = np.zeros_like(u)
    Uprev = np.zeros_like(u)          # U_{-1}
    Ucur = np.ones_like(u)            # U_0
    dUprev = np.zeros_like(u)
    dUcur = np.zeros_like(u)
    for n in range(1, N + 1):
        cf = n * math.exp(-(n * n - 1.0) * s)
        p += cf * Ucur
        dp += cf * dUcur
        Un = 2.0 * u * Ucur - Uprev
        dUn = 2.0 * Ucur + 2.0 * u * dUcur - dUprev
        Uprev, Ucur = Ucur, Un
        dUprev, dUcur = dUcur, dUn
    c = 1.0 / (2.0 * math.pi ** 2)
    return c * p, c * dp


def s3_asymptotic_np(u, s):
    """Winding term k=0 only:  the exact small-s branch on S^3.

        log p = s - 3/2 log(4 pi s) - th^2/(4s) + log(th / sin th)
    """
    th = np.arccos(np.clip(u, -1.0, 1.0))
    th = np.clip(th, 1e-12, math.pi - 1e-9)
    sn = np.sin(th)
    logp = s - 1.5 * math.log(4.0 * math.pi * s) - th * th / (4.0 * s) \
        + np.log(th / np.maximum(sn, 1e-300))
    # 1/th - cot th  (series near 0 to avoid catastrophic cancellation)
    small = th < 0.1
    cot_part = np.where(small,
                        th / 3.0 + th ** 3 / 45.0 + 2.0 * th ** 5 / 945.0,
                        1.0 / th - np.cos(th) / np.maximum(sn, 1e-300))
    dth = -th / (2.0 * s) + cot_part
    dlogp_du = -dth / np.maximum(sn, 1e-300)
    return logp, dlogp_du


class S3Table:
    """logp and dlogp/du on a theta grid, for every spin time in `ss`."""

    def __init__(self, ss, ntheta=16385, device=DEV):
        self.ss = np.asarray(ss, dtype=np.float64)
        self.ntheta = ntheta
        th = np.linspace(0.0, math.pi, ntheta)
        self.dth = math.pi / (ntheta - 1)
        u = np.cos(th)
        LP = np.empty((len(self.ss), ntheta))
        DU = np.empty_like(LP)
        for a, s in enumerate(self.ss):
            s = float(s)
            p, dp = s3_spectral_np(u, s)
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.log(np.maximum(p, 1e-300))
                du = dp / np.maximum(p, 1e-300)
            good = np.isfinite(lp) & np.isfinite(du) & (p > 1e-11 * p.max())
            la, da = s3_asymptotic_np(u, s)
            LP[a] = np.where(good, lp, la)
            DU[a] = np.where(good, du, da)
        self.logp = torch.tensor(LP, device=device)
        self.dlogpdu = torch.tensor(DU, device=device)

    def _interp(self, tab, s_idx, theta):
        nth = tab.shape[1]
        x = (theta / self.dth).clamp(0.0, nth - 1.0000001)
        i0 = x.floor().long()
        w = x - i0.to(x.dtype)
        flat = tab.reshape(-1)
        base = s_idx * nth + i0
        return flat[base] + w * (flat[base + 1] - flat[base])

    def logp_at(self, s_idx, u):
        th = torch.arccos(u.clamp(-1.0, 1.0))
        return self._interp(self.logp, s_idx, th)

    def dlogp_du(self, s_idx, u):
        th = torch.arccos(u.clamp(-1.0, 1.0))
        return self._interp(self.dlogpdu, s_idx, th)


def s3_logp_torch(u, s, Nmax=None):
    """Autograd-safe spectral log p_s(u).  Used only at the terminal clock
    s = r01/2 = O(1), where ~10 terms suffice and there is no cancellation."""
    N = Nmax or _nmax_for(s)
    p = torch.zeros_like(u)
    Uprev = torch.zeros_like(u)
    Ucur = torch.ones_like(u)
    for n in range(1, N + 1):
        p = p + (n * math.exp(-(n * n - 1.0) * s)) * Ucur
        Uprev, Ucur = Ucur, 2.0 * u * Ucur - Uprev
    return torch.log(p.clamp(min=1e-300)) - math.log(2.0 * math.pi ** 2)


# ============================================================================
# the problem
# ============================================================================
def killing_basis(n=4, device=DEV):
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    O = torch.zeros(len(idx), n, n, dtype=torch.float64, device=device)
    for k, (i, j) in enumerate(idx):
        O[k, i, j] = 1.0
        O[k, j, i] = -1.0
    return O


def skew_lift(X, D):
    """The unique skew Omega with Omega X = D, for D tangent at X."""
    XT = X.transpose(-1, -2)
    return D @ XT - X @ D.transpose(-1, -2) - X @ (XT @ D) @ XT


def proj_tangent(X, D):
    S = X.transpose(-1, -2) @ D
    return D - X @ (0.5 * (S + S.transpose(-1, -2)))


def energy_quad(X, H):
    return torch.einsum("...np,nm,...mp->...", X, H, X)


def energy_quad_grad(X, H):
    return 2.0 * (H @ X)


class StiefelProblem:
    """St(4,2) with a Dirac source at E0 and target  pi ~ exp(-beta E)."""

    def __init__(self, sigma=math.sqrt(2.0), beta=1.0, steps=128, nq=64,
                 nfib=512, frame=False, lam=1.0, device=DEV):
        self.sigma, self.beta, self.steps, self.device = sigma, beta, steps, device
        self.tau = 1.0 / beta
        self.frame = frame
        self.lam = lam
        self.n_oracle = 0           # terminal energy-gradient evaluations
        self.H = torch.diag(_q([1., 2., 5., 8.], device))
        self.Cmat = torch.tensor([[0.7, -0.2], [0.1, 0.8],
                                  [-0.4, 0.3], [0.2, -0.5]],
                                 dtype=torch.float64, device=device)
        self.E0 = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
                               dtype=torch.float64, device=device)
        self.Om = killing_basis(4, device)

        self.ts = np.arange(steps) / steps
        self.r_t1 = np.array([C.heat_clock(t, 1.0, sigma) for t in self.ts])
        self.r01 = float(C.heat_clock(0.0, 1.0, sigma))
        # PLAN 7.5: EVERY spin-factor evaluation runs at r/2, never r.
        self.spin_ss = np.array([C.stiefel_spin_factor_time(r)
                                 for r in self.r_t1])
        self.spin_r01 = float(C.stiefel_spin_factor_time(self.r01))
        self.tab = S3Table(self.spin_ss, device=device)

        # SO(2) fibre quadrature (PLAN 7.8) in the double-cover angle psi;
        # psi in [0, 2pi) sweeps BOTH lifts of the SO(2) stabiliser element.
        psi, wq = C.periodic_legendre_nodes(nq, device=device)
        self.psi, self.wq = psi, wq
        self.logwq = torch.log(wq)
        # a finer uniform grid used only to SAMPLE the fibre for the bridge
        pf = torch.arange(nfib, device=device, dtype=torch.float64) \
            * (2.0 * math.pi / nfib)
        self.psi_f = pf

    # -- energy ----------------------------------------------------------
    def energy(self, X):
        e = energy_quad(X, self.H)
        if self.frame:
            e = e - self.lam * torch.einsum("np,...np->...", self.Cmat, X)
        return e

    def energy_grad(self, X):
        g = energy_quad_grad(X, self.H)
        if self.frame:
            g = g - self.lam * self.Cmat
        return g

    # -- Stiefel heat kernel at the terminal clock ------------------------
    def _fibre_u(self, Y, psi):
        a, b = section_lift(Y)
        cs, sn = torch.cos(psi)[None, :], torch.sin(psi)[None, :]
        ua = a[:, 0:1] * cs - a[:, 1:2] * sn
        ub = b[:, 0:1] * cs - b[:, 1:2] * sn
        return ua, ub, a, b

    def log_pst(self, Y):
        """log p^{St}_{r01}(Y | E0), up to an additive constant.

        = log int_0^{2pi} p^{S3}_{r01/2}(u_a(psi)) p^{S3}_{r01/2}(u_b(psi)) dpsi
        Section-independent, hence differentiable through `section_lift`.
        """
        ua, ub, _, _ = self._fibre_u(Y, self.psi)
        s = self.spin_r01
        lk = s3_logp_torch(ua.clamp(-1., 1.), s) \
            + s3_logp_torch(ub.clamp(-1., 1.), s) + self.logwq[None, :]
        return torch.logsumexp(lk, dim=1)

    def G(self, Y):
        """Ambient grad_Y log f1(Y),  f1 = exp(-E/tau) / p^{St}_{r01}(.|E0)."""
        # PLAN 9.2 requires reporting terminal energy/gradient evaluations.
        # This is the ONLY place the energy gradient is touched during
        # training, so counting here is exact rather than estimated.
        self.n_oracle += int(Y.shape[0])
        Yr = Y.detach().requires_grad_(True)
        lp = self.log_pst(Yr).sum()
        gp = torch.autograd.grad(lp, Yr)[0]
        return -self.energy_grad(Y) / self.tau - gp

    # -- fibre sampling for the bridge ------------------------------------
    def sample_fibre_lift(self, Y, generator=None, chunk=8192):
        outa, outb = [], []
        s = self.spin_r01
        for k in range(0, Y.shape[0], chunk):
            Yb = Y[k:k + chunk]
            ua, ub, a, b = self._fibre_u(Yb, self.psi_f)
            lk = s3_logp_torch(ua.clamp(-1., 1.), s) \
                + s3_logp_torch(ub.clamp(-1., 1.), s)
            lk = lk - lk.max(dim=1, keepdim=True).values
            j = torch.multinomial(torch.exp(lk), 1, generator=generator)[:, 0]
            ps = self.psi_f[j]
            c = torch.stack([torch.cos(ps), torch.sin(ps),
                             torch.zeros_like(ps), torch.zeros_like(ps)], -1)
            outa.append(qmul(a, c))
            outb.append(qmul(b, c))
        return torch.cat(outa), torch.cat(outb)


# ============================================================================
# simulation
# ============================================================================
def simulate(prob, score_fn, batch, steps, generator=None, keep_path=False):
    """Controlled Brownian motion on St(4,2):
           X <- exp( sigma sqrt(dt) A  +  dt sigma^2 Omega(score) ) X,
       A = sum_{i<j} xi_ij Omega_ij.  Left multiplication by an orthogonal
       matrix keeps X^T X = I at machine precision."""
    dt = 1.0 / steps
    sig = prob.sigma
    X = prob.E0[None].expand(batch, 4, 2).contiguous().clone()
    path = [] if keep_path else None
    for s in range(steps):
        if keep_path:
            path.append(X.clone())
        xi = torch.randn(batch, 6, dtype=X.dtype, device=X.device,
                         generator=generator)
        M = sig * math.sqrt(dt) * torch.einsum("bk,kij->bij", xi, prob.Om)
        if score_fn is not None:
            D = proj_tangent(X, score_fn(s, X))
            M = M + (sig * sig * dt) * skew_lift(X, D)
        X = torch.matrix_exp(M) @ X
    if keep_path:
        return X, torch.stack(path)
    return X


def _s3_step(q, score, c, dt, generator=None):
    """One step of an S^3 Brownian motion with generator c*Laplacian,
    Doob-tilted by `score` (an ambient R^4 vector)."""
    xi = torch.randn(q.shape, dtype=q.dtype, device=q.device,
                     generator=generator)
    v = 2.0 * c * score * dt + math.sqrt(2.0 * c * dt) * xi
    v = v - (v * q).sum(-1, keepdim=True) * q
    qn = C.sphere_exp(q, v)
    return qn / qn.norm(dim=-1, keepdim=True)


def spin_bridge_path(prob, a1, b1, steps, generator=None):
    """Two independent S^3 Doob bridges 1 -> a1, 1 -> b1, each on the spin
    clock r/2, projected to St(4,2).  Returns (steps, B, 4, 2)."""
    B = a1.shape[0]
    q1, _, _, _ = quat_consts(a1.device)
    a = q1[None].expand(B, 4).contiguous().clone()
    b = q1[None].expand(B, 4).contiguous().clone()
    c = prob.spin_r01                      # total spin time over t in [0,1]
    dt = 1.0 / steps
    out = []
    for s in range(steps):
        out.append(stiefel_from_quat(a, b))
        si = torch.full((B,), s, device=a.device, dtype=torch.long)
        ua = (a * a1).sum(-1).clamp(-1., 1.)
        ub = (b * b1).sum(-1).clamp(-1., 1.)
        sa = prob.tab.dlogp_du(si, ua)[:, None] * a1
        sb = prob.tab.dlogp_du(si, ub)[:, None] * b1
        a = _s3_step(a, sa, c, dt, generator)
        b = _s3_step(b, sb, c, dt, generator)
    return torch.stack(out)


def bridge_path(prob, Y, steps, generator=None):
    a1, b1 = prob.sample_fibre_lift(Y, generator=generator)
    return spin_bridge_path(prob, a1, b1, steps, generator=generator)


# ============================================================================
# network
# ============================================================================
class ScoreNet(torch.nn.Module):
    def __init__(self, hidden=256, n_freq=6):
        super().__init__()
        self.n_freq = n_freq
        din = 8 + 2 + 2 * n_freq
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 8),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.to(torch.float32)

    def forward(self, t, X):
        xf = X.to(torch.float32).reshape(X.shape[0], 8)
        tf = t.to(torch.float32)[:, None]
        k = torch.arange(1, self.n_freq + 1, device=xf.device,
                         dtype=xf.dtype)[None, :]
        f = torch.cat([xf, tf, 1.0 - tf, torch.sin(math.pi * k * tf),
                       torch.cos(math.pi * k * tf)], dim=-1)
        v = self.net(f).reshape(X.shape[0], 4, 2).to(X.dtype)
        return proj_tangent(X, v)


class EMA:
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
    def f(s, XX):
        t = torch.full((XX.shape[0],), s / steps, device=XX.device,
                       dtype=XX.dtype)
        return net(t, XX)
    return f


# ============================================================================
# ground truth: random-walk Metropolis on St(4,2)  (5-dimensional, so many
# parallel chains from Haar starts mix quickly and are trivially cheap)
# ============================================================================
def mcmc_stiefel(prob, nchain=200000, nsweep=4000, eps=0.35, burn=0.5,
                 generator=None, verbose=False):
    dev = prob.device
    X = C.random_stiefel(nchain, 4, 2, device=dev, generator=generator)
    E = prob.energy(X)
    beta = prob.beta
    nacc = 0
    for k in range(nsweep):
        xi = torch.randn(nchain, 6, dtype=X.dtype, device=dev,
                         generator=generator) * eps
        Xp = torch.matrix_exp(torch.einsum("bk,kij->bij", xi, prob.Om)) @ X
        Ep = prob.energy(Xp)
        u = torch.rand(nchain, dtype=X.dtype, device=dev, generator=generator)
        acc = u < torch.exp(-beta * (Ep - E)).clamp(max=1.0)
        X = torch.where(acc[:, None, None], Xp, X)
        E = torch.where(acc, Ep, E)
        nacc += float(acc.to(X.dtype).mean())
        if k == int(burn * nsweep):
            pass
    if verbose:
        print(f"    MCMC accept {nacc / nsweep:.3f}  eps={eps}")
    return X


# ============================================================================
# metrics
# ============================================================================
def energy_metrics(prob, X, ref_E=None):
    E = prob.energy(X)
    m = {"E_mean": float(E.mean()), "E_std": float(E.std()),
         "constraint": float(C.stiefel_constraint_error(X).max()),
         "trCX": float(torch.einsum("np,bnp->b", prob.Cmat, X).mean())}
    if ref_E is not None:
        g = np.sort(ref_E.cpu().numpy())
        cdf = (np.arange(len(g)) + 1.0) / len(g)
        m["KS_E"] = C.ks_against_grid_cdf(E.cpu().numpy(), g, cdf)
        m["E_err"] = abs(m["E_mean"] - float(ref_E.mean()))
    return m


def report(tag, m):
    s = (f"    {tag}  E={m['E_mean']:.4f}+-{m['E_std']:.4f}"
         f"  tr(C^T X)={m['trCX']:.4f}  |X^TX-I|={m['constraint']:.2e}")
    if "KS_E" in m:
        s += f"  KS(E)={m['KS_E']:.4f}  dE={m['E_err']:.4f}"
    return s


# ============================================================================
# gate D0
# ============================================================================
def run_verify(args):
    prob = StiefelProblem(sigma=args.sigma, beta=args.beta, steps=args.steps,
                          nq=args.nq, frame=args.frame)
    torch.manual_seed(args.seed)
    dev = prob.device
    print(f"sigma={prob.sigma:.4f}  r01={prob.r01:.4f}  spin s01={prob.spin_r01:.4f}"
          f"  steps={prob.steps}  fibre nodes={args.nq}")
    ok = True

    # (a) explicit Killing sum == collapsed readout            (PLAN 7.4)
    worst = 0.0
    for _ in range(8):
        X = C.random_stiefel(1, 4, 2, device=dev)[0]
        Y = C.random_stiefel(1, 4, 2, device=dev)[0]
        Gm = torch.randn(4, 2, dtype=X.dtype, device=dev)
        worst = max(worst, float((C.stiefel_readout_explicit(X, Y, Gm)
                                  - C.stiefel_readout_collapsed(X, Y, Gm)).abs().max()))
    print(f"  |explicit - collapsed| readout        = {worst:.3e}")
    ok &= worst < 1e-10

    # (b) readout is tangent at X                              (PLAN 7.3)
    X = C.random_stiefel(256, 4, 2, device=dev)
    Y = C.random_stiefel(256, 4, 2, device=dev)
    Gm = torch.randn(256, 4, 2, dtype=X.dtype, device=dev)
    Rd = C.stiefel_terminal_readout(X, Y, Gm)
    sk = X.transpose(-1, -2) @ Rd + Rd.transpose(-1, -2) @ X
    print(f"  |X^T R + R^T X|                       = {float(sk.abs().max()):.3e}")
    ok &= float(sk.abs().max()) < 1e-12
    # and skew_lift really inverts it
    Om = skew_lift(X, Rd)
    e1 = float((Om @ X - Rd).abs().max())
    e2 = float((Om + Om.transpose(-1, -2)).abs().max())
    print(f"  |Omega X - D| / |Omega + Omega^T|     = {e1:.3e} / {e2:.3e}")
    ok &= e1 < 1e-10 and e2 < 1e-12

    # (c) S^3 winding-sum truncation K -> K+2                  (PLAN 7.7)
    dl, ds = 0.0, 0.0
    for s in [float(prob.spin_ss.min()), 0.05, float(prob.spin_r01)]:
        th = torch.linspace(1e-6, math.pi - 1e-6, 4001, dtype=torch.float64)
        k8 = C.s3_heat_kernel(th, s, K=8)
        k10 = C.s3_heat_kernel(th, s, K=10)
        m = k8 > 1e-9 * k8.max()
        dl = max(dl, float((torch.log(k10[m]) - torch.log(k8[m])).abs().max()))
        g8 = torch.gradient(torch.log(k8[m]), spacing=(th[m],))[0]
        g10 = torch.gradient(torch.log(k10[m]), spacing=(th[m],))[0]
        ds = max(ds, float((g10 - g8).abs().max()))
    print(f"  winding K->K+2: dlog={dl:.3e}  dscore={ds:.3e}")
    ok &= dl < 1e-10 and ds < 1e-8

    # (d) spectral kernel: mass, and table interpolation accuracy
    mworst, tworst, dworst = 0.0, 0.0, 0.0
    for a, s in enumerate([0, prob.steps // 4, prob.steps // 2, prob.steps - 1]):
        sv = float(prob.spin_ss[s])
        thn = np.linspace(1e-12, math.pi - 1e-12, 200001)
        pn, _ = s3_spectral_np(np.cos(thn), sv)
        mworst = max(mworst, abs(float(np.trapezoid(
            pn * 4 * math.pi * np.sin(thn) ** 2, thn)) - 1.0))
        # interpolation vs direct, on the numerically trustworthy support
        thr = np.random.rand(4000) * math.pi
        pr, dpr = s3_spectral_np(np.cos(thr), sv)
        msk = pr > 1e-9 * pr.max()
        si = torch.full((4000,), s, device=dev, dtype=torch.long)
        uu = torch.tensor(np.cos(thr), device=dev)
        lp = prob.tab.logp_at(si, uu).cpu().numpy()[msk]
        du = prob.tab.dlogp_du(si, uu).cpu().numpy()[msk]
        ref_lp = np.log(pr[msk])
        ref_du = dpr[msk] / pr[msk]
        tworst = max(tworst, float(np.abs(lp - ref_lp).max()))
        dworst = max(dworst, float((np.abs(du - ref_du)
                                    / np.maximum(np.abs(ref_du), 1.0)).max()))
    print(f"  |S^3 mass - 1|                        = {mworst:.3e}")
    print(f"  table |logp| err / rel dlogp/du err   = {tworst:.3e} / {dworst:.3e}")
    ok &= mworst < 1e-10 and tworst < 1e-4 and dworst < 1e-3

    # (e) MANDATORY first moment  E[X_r|X_0] = exp(-3 r) X_0   (PLAN 7.6)
    print("  first-moment test  E[X_r|E0] = exp(-3r) E0   [MANDATORY]")
    fm_ok = True
    for r in [0.25, 1.0]:
        tgt = C.expected_st42_mean_factor(r)
        nb, nst = args.mc_moment, 512
        # (i) SO(4) matrix simulator
        Rm = C.simulate_so_brownian(nb, 4, r, nst, device=dev)
        Xm = Rm @ prob.E0
        mu = Xm.mean(dim=0)
        se = float((Xm.std(dim=0) / math.sqrt(nb)).max())
        em = float((mu - tgt * prob.E0).abs().max())
        # (ii) quaternion spin simulator at s = r/2  (the corrected clock)
        c = C.stiefel_spin_factor_time(r)
        q1, _, _, _ = quat_consts(dev)
        aa = q1[None].expand(nb, 4).contiguous().clone()
        bb = q1[None].expand(nb, 4).contiguous().clone()
        z = torch.zeros(nb, 4, dtype=torch.float64, device=dev)
        for _ in range(nst):
            aa = _s3_step(aa, z, c, 1.0 / nst)
            bb = _s3_step(bb, z, c, 1.0 / nst)
        Xq = stiefel_from_quat(aa, bb)
        eq = float((Xq.mean(dim=0) - tgt * prob.E0).abs().max())
        # (iii) the WRONG convention s = r would give exp(-6r)
        wrong = abs(math.exp(-6.0 * r) - tgt)
        print(f"    r={r:.2f}  target={tgt:.6f}  matrix err={em:.2e}"
              f"  spin(r/2) err={eq:.2e}   (5*SE={5*se:.2e},"
              f"  |exp(-6r)-exp(-3r)|={wrong:.3f})")
        fm_ok &= em < 5 * se + 2e-3 and eq < 5 * se + 2e-3
    ok &= fm_ok

    # (f) spin lift vs matrix simulator: same law of  u = <x_1, e1>
    r = prob.r01
    nb = args.mc_moment
    Rm = C.simulate_so_brownian(nb, 4, r, 512, device=dev)
    um = (Rm @ prob.E0)[:, 0, 0].cpu().numpy()
    c = C.stiefel_spin_factor_time(r)
    q1, _, _, _ = quat_consts(dev)
    aa = q1[None].expand(nb, 4).contiguous().clone()
    bb = q1[None].expand(nb, 4).contiguous().clone()
    z = torch.zeros(nb, 4, dtype=torch.float64, device=dev)
    for _ in range(512):
        aa = _s3_step(aa, z, c, 1.0 / 512)
        bb = _s3_step(bb, z, c, 1.0 / 512)
    uq = stiefel_from_quat(aa, bb)[:, 0, 0].cpu().numpy()
    g = np.sort(um)
    ksq = C.ks_against_grid_cdf(uq, g, (np.arange(nb) + 1.0) / nb)
    print(f"  KS(spin sim vs SO(4) matrix sim)      = {ksq:.4f}"
          f"   (2-sample 1% crit ~ {1.63*math.sqrt(2.0/nb):.4f})")
    ok &= ksq < 1.63 * math.sqrt(2.0 / nb)
    print(f"  |X^T X - I| after {prob.steps} sim steps       = "
          f"{float(C.stiefel_constraint_error(simulate(prob, None, 4096, prob.steps)).max()):.2e}")

    # (g) fibre quadrature convergence and section independence
    Y = C.random_stiefel(512, 4, 2, device=dev)
    lp1 = prob.log_pst(Y)
    p2 = StiefelProblem(sigma=args.sigma, beta=args.beta, steps=args.steps,
                        nq=2 * args.nq, frame=args.frame)
    lp2 = p2.log_pst(Y)
    print(f"  fibre nq -> 2nq change in log p^St    = "
          f"{float((lp2 - lp1).abs().max()):.3e}")
    ok &= float((lp2 - lp1).abs().max()) < 1e-10
    # rotate the section by a random stabiliser element: value must not move
    ps = torch.rand(512, dtype=torch.float64, device=dev) * 2 * math.pi
    cq = torch.stack([torch.cos(ps), torch.sin(ps),
                      torch.zeros_like(ps), torch.zeros_like(ps)], -1)
    aY, bY = section_lift(Y)
    Yr = stiefel_from_quat(qmul(aY, cq), qmul(bY, cq))
    print(f"  |p^St| section-shift invariance       = "
          f"{float((prob.log_pst(Yr) - lp1).abs().max()):.3e}")
    ok &= float((prob.log_pst(Yr) - lp1).abs().max()) < 1e-9

    # (h) left invariance under the stabiliser of E0
    phi = float(np.random.rand() * 2 * math.pi)
    Sm = torch.eye(4, dtype=torch.float64, device=dev)
    Sm[2, 2] = Sm[3, 3] = math.cos(phi)
    Sm[2, 3] = -math.sin(phi)
    Sm[3, 2] = math.sin(phi)
    inv = float((prob.log_pst(Sm @ Y) - lp1).abs().max())
    print(f"  p^St(S Y) = p^St(Y), S in Stab(E0)    = {inv:.3e}")
    ok &= inv < 1e-9

    # (i) terminal limit of the readout: as t -> 1 the conditional law
    #     collapses onto Y = X, so the Killing readout -> G - X G^T X.
    s = prob.steps - 1
    Xs = C.random_stiefel(256, 4, 2, device=dev)
    Gs = prob.G(Xs)
    lim = C.stiefel_terminal_readout(Xs, Xs, Gs)
    print(f"  |readout(X,X,G) - (G - X G^T X)|      = "
          f"{float((lim - (Gs - Xs @ (Gs.transpose(-1,-2) @ Xs))).abs().max()):.3e}")
    ok &= float((lim - (Gs - Xs @ (Gs.transpose(-1, -2) @ Xs))).abs().max()) < 1e-12

    # (j) autograd G against a finite difference of log p^St
    Y = C.random_stiefel(64, 4, 2, device=dev)
    gp = prob.G(Y) + prob.energy_grad(Y) / prob.tau       # = -grad log p^St
    h = 1e-6
    V = torch.randn(64, 4, 2, dtype=torch.float64, device=dev)
    V = V / V.norm(dim=(-2, -1), keepdim=True)
    Yp = Y + h * V
    Yp = Yp @ torch.linalg.inv(torch.linalg.cholesky(
        Yp.transpose(-1, -2) @ Yp).transpose(-1, -2))
    Ym = Y - h * V
    Ym = Ym @ torch.linalg.inv(torch.linalg.cholesky(
        Ym.transpose(-1, -2) @ Ym).transpose(-1, -2))
    fd = (prob.log_pst(Yp) - prob.log_pst(Ym)) / (2 * h)
    an = (-gp * (Yp - Ym) / (2 * h)).sum(dim=(-2, -1))
    rel = float(((fd - an).abs() / fd.abs().clamp(min=1e-3)).max())
    print(f"  autograd grad log p^St vs finite diff = {rel:.3e}")
    ok &= rel < 1e-4

    # (k) THE end-to-end test of the terminal function.  Reference samples
    #     Y ~ p_{r01}(.|E0) reweighted by  f1 = exp(-beta E) / p_{r01}(.|E0)
    #     must reproduce the target exp(-beta E).  Validates the fibre
    #     quadrature, the r/2 clock and f1 at once, with no learning involved.
    torch.manual_seed(args.seed)
    Yr = simulate(prob, None, args.mc_moment, 512)
    lw = -prob.beta * prob.energy(Yr) - prob.log_pst(Yr)
    lw = lw - lw.max()
    w = torch.exp(lw)
    w = w / w.sum()
    ess = float(1.0 / (w ** 2).sum())
    Eis = float((w * prob.energy(Yr)).sum())
    Xr = mcmc_stiefel(prob, nchain=args.mcmc_chains // 2,
                      nsweep=args.mcmc_sweeps // 2, eps=args.mcmc_eps)
    Em = float(prob.energy(Xr).mean())
    tol = 3.0 * float(prob.energy(Yr).std()) / math.sqrt(ess) + 0.01
    print(f"  IS(reference x f1) E={Eis:.4f}  vs MCMC E={Em:.4f}"
          f"  |d|={abs(Eis-Em):.4f} (tol {tol:.4f})  ESS={ess:.0f}/{args.mc_moment}")
    ok &= abs(Eis - Em) < tol

    print(f"\nD0 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ============================================================================
# training  (Reciprocal Adjoint Matching with the canonical Killing readout)
# ============================================================================
def train_one(prob, args, seed, verbose=True):
    steps = prob.steps
    torch.manual_seed(1000 + seed)
    net = ScoreNet(hidden=args.hidden).to(prob.device)
    ema_net = ScoreNet(hidden=args.hidden).to(prob.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 1e-2)
    ema = EMA(net, decay=args.ema)
    buf = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            Y = simulate(prob, _score_fn(net, steps), args.batch, steps)
        Gy = prob.G(Y)
        with torch.no_grad():
            path = bridge_path(prob, Y, steps)                # (S,B,4,2)
            buf.append((path.to(torch.float32), Y.to(torch.float32),
                        Gy.to(torch.float32)))
            if len(buf) > args.nbuf:
                buf.pop(0)
            P = torch.cat([b[0] for b in buf], dim=1)
            YY = torch.cat([b[1] for b in buf], dim=0)
            GG = torch.cat([b[2] for b in buf], dim=0)
            B = YY.shape[0]
        for _ in range(args.inner):
            b = torch.randint(B, (args.mb,), device=prob.device)
            s = torch.randint(steps, (args.mb,), device=prob.device)
            xt = P[s, b].to(torch.float64)
            yb, gb = YY[b].to(torch.float64), GG[b].to(torch.float64)
            if args.antithetic:
                # T_s(X) = S X D,  S = diag(s1..s4), D = diag(s1,s2), s_i = +-1.
                # T_s fixes E0, preserves the quadratic energy (H diagonal) and
                # conjugates the SO(4) Brownian motion by S, so the whole path
                # law is invariant -- and the Killing readout is equivariant,
                # readout(SXD, SYD, SGD) = S readout(X,Y,G) D.  16 exact
                # copies of every sample, at zero modelling cost.
                s4 = (torch.randint(0, 2, (xt.shape[0], 4), device=xt.device,
                                    dtype=torch.long) * 2 - 1).to(xt.dtype)
                Sm = s4[:, :, None]
                Dm = s4[:, None, :2]
                xt = Sm * xt * Dm
                yb = Sm * yb * Dm
                gb = Sm * gb * Dm
            lab = C.stiefel_terminal_readout(xt, yb, gb)
            pred = net(s.to(xt.dtype) / steps, xt)
            loss = ((pred - lab) ** 2).sum(dim=(-2, -1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            ema.update(net)
        if verbose and (it % args.eval_every == 0 or it == args.iters):
            ema.copy_to(ema_net)
            with torch.no_grad():
                Xs = simulate(prob, _score_fn(ema_net, steps), 20000, steps)
            print(f"    it {it:4d}  loss {float(loss):9.3f}  "
                  f"E={float(prob.energy(Xs).mean()):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    ema.copy_to(ema_net)
    return ema_net


def run_train(args):
    betas = [float(b) for b in args.betas.split(",")]
    dev = DEV
    out, gate_dE, gate_ks = {}, [], []
    print(f"Stiefel St(4,2)  {'FRAME-SENSITIVE' if args.frame else 'R-ASBS quadratic'}"
          f" target,  steps={args.steps}, seeds={args.seeds}")
    if args.frame:
        # tr(X^T H X) is invariant under X -> X Q, the frame term is not, so
        # the 8 / 3 limits of the quadratic problem simply do not apply here.
        print("  asymptotes:  N/A -- the frame term -lam tr(C^T X) breaks the "
              "right O(p) invariance;\n               MCMC is the only "
              "reference for this target.")
    else:
        print(f"  asymptotes:  beta->0  E=8.0000    beta->inf  E=3.0000"
              f"   (quadratic target)")
    for beta in betas:
        prob = StiefelProblem(sigma=args.sigma, beta=beta, steps=args.steps,
                              nq=args.nq, frame=args.frame, device=dev)
        torch.manual_seed(args.seed)
        Xref = mcmc_stiefel(prob, nchain=args.mcmc_chains,
                            nsweep=args.mcmc_sweeps, eps=args.mcmc_eps)
        Eref = prob.energy(Xref)
        mref = energy_metrics(prob, Xref)
        print(f"\n  beta={beta:g}   MCMC ref: E={mref['E_mean']:.4f}"
              f"+-{mref['E_std']:.4f}  tr(C^T X)={mref['trCX']:.4f}")
        ms = []
        for seed in range(args.seeds):
            prob.n_oracle = 0
            t_seed = time.time()
            net = train_one(prob, args, seed, verbose=args.verbose)
            with torch.no_grad():
                Xs = simulate(prob, _score_fn(net, args.steps),
                              args.n_samples, args.steps)
            m = energy_metrics(prob, Xs, ref_E=Eref)
            m["oracle_calls"] = prob.n_oracle
            m["train_s"] = time.time() - t_seed
            m["steps"] = args.steps
            # discretisation diagnostic: the SAME continuous control, only
            # integrated on a finer grid.  Any change is pure discretisation
            # bias, not score error.
            m["refined"] = {}
            for rf in [int(v) for v in str(args.refine).split(",") if int(v) > 1]:
                with torch.no_grad():
                    sf = rf * args.steps
                    Xf = simulate(prob, _score_fn(net, sf), args.n_samples, sf)
                mf = energy_metrics(prob, Xf, ref_E=Eref)
                m["refined"][sf] = mf
                print(report(f"seed {seed} steps={sf:5d}", mf), flush=True)
            ms.append(m)
            print(report(f"seed {seed}", m), flush=True)
            C.save_ckpt(args.ckpt_dir, f"{args.tag}_b{beta:g}_seed{seed}",
                        net=net, samples=Xs,
                        extra={"config": vars(args), "metrics": m,
                               "mcmc": mref, "beta": beta})
        dE = float(np.mean([m["E_err"] for m in ms]))
        ks = float(np.mean([m["KS_E"] for m in ms]))
        gate_dE.append(dE)
        gate_ks.append(ks)
        out[beta] = {"mcmc": mref, "seeds": ms, "dE": dE, "KS_E": ks}
        print(f"    beta={beta:g}  mean |dE|={dE:.4f}   mean KS(E)={ks:.4f}")
        C.save_ckpt(args.ckpt_dir, f"{args.tag}_b{beta:g}_mcmc",
                    samples=Xref[:args.n_samples],
                    extra={"metrics": mref, "beta": beta})

    mdE, mks = float(np.mean(gate_dE)), float(np.mean(gate_ks))
    tagD = "D2" if args.frame else "D1"
    print("\n=== GATES ===")
    print(f"{tagD}a mean |E_ours - E_MCMC| < 0.05      : "
          f"{'PASS' if mdE < 0.05 else 'FAIL'}  ({mdE:.4f})")
    print(f"{tagD}b mean KS(E) < 0.05                  : "
          f"{'PASS' if mks < 0.05 else 'FAIL'}  ({mks:.4f})")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "betas": out,
                   "mean_dE": mdE, "mean_KS": mks}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (mdE < 0.05 and mks < 0.05) else 1


def run_ref(args):
    """Independent MCMC ground truth on an arbitrary beta grid.

    This fills the 'exact/reference MCMC' column of PLAN 8.2, which the R-ASBS
    paper leaves as TBD.  Two chains with different seeds are run at every beta
    so the Monte-Carlo error of the reference itself is reported rather than
    assumed: the spread between them bounds how tight a gate can meaningfully
    be set.
    """
    dev = DEV
    betas = [float(b) for b in args.betas.split(",")]
    print(f"Stiefel St(4,2) MCMC reference  "
          f"{'frame' if args.frame else 'quadratic'} target")
    print(f"  chains={args.mcmc_chains}  sweeps={args.mcmc_sweeps}  "
          f"eps={args.mcmc_eps}")
    print(f"  asymptotes:  beta->0  E=8    beta->inf  E=3")
    out = {}
    for beta in betas:
        prob = StiefelProblem(sigma=args.sigma, beta=beta, steps=args.steps,
                              nq=args.nq, frame=args.frame, device=dev)
        es = []
        for rep in range(2):
            torch.manual_seed(args.seed + 1000 * rep)
            X = mcmc_stiefel(prob, nchain=args.mcmc_chains,
                             nsweep=args.mcmc_sweeps, eps=args.mcmc_eps)
            es.append(energy_metrics(prob, X))
        gap = abs(es[0]["E_mean"] - es[1]["E_mean"])
        out[beta] = {"E_mean": 0.5 * (es[0]["E_mean"] + es[1]["E_mean"]),
                     "E_std": es[0]["E_std"], "seed_gap": gap,
                     "trCX": es[0]["trCX"], "constraint": es[0]["constraint"]}
        print(f"  beta={beta:<10g} E={out[beta]['E_mean']:.4f}"
              f"+-{out[beta]['E_std']:.4f}   seed-gap={gap:.4f}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "betas": out}, f, indent=2,
                  default=str)
    print(f"  wrote {args.out}")
    return 0


def run_sweep(args):
    """Gate D3: step-count sweep at a single beta, against MCMC."""
    print(f"step sweep, beta={args.beta}, "
          f"{'frame' if args.frame else 'quadratic'} target")
    p0 = StiefelProblem(sigma=args.sigma, beta=args.beta, steps=128,
                        nq=args.nq, frame=args.frame)
    torch.manual_seed(args.seed)
    Xref = mcmc_stiefel(p0, nchain=args.mcmc_chains, nsweep=args.mcmc_sweeps,
                        eps=args.mcmc_eps)
    Eref = p0.energy(Xref)
    print(report("MCMC ref  ", energy_metrics(p0, Xref)))
    out = {}
    for steps in [32, 64, 128, 256, 512]:
        prob = StiefelProblem(sigma=args.sigma, beta=args.beta, steps=steps,
                              nq=args.nq, frame=args.frame)
        prob.n_oracle = 0
        t_s = time.time()
        net = train_one(prob, args, 0, verbose=args.verbose)
        with torch.no_grad():
            Xs = simulate(prob, _score_fn(net, steps), args.n_samples, steps)
        m = energy_metrics(prob, Xs, ref_E=Eref)
        m["oracle_calls"] = prob.n_oracle
        m["train_s"] = time.time() - t_s
        out[steps] = m
        print(report(f"steps={steps:4d}", m), flush=True)
        if args.ckpt_dir:
            C.save_ckpt(args.ckpt_dir, f"{args.tag}_sweep_steps{steps}",
                        net=net, samples=Xs,
                        extra={"metrics": m, "steps": steps})
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "sweep": out}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "train", "ref", "sweep"])
    ap.add_argument("--sigma", type=float, default=math.sqrt(2.0))
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--betas", type=str, default="0.01,0.1,0.5,1.3,2,5,10")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--nq", type=int, default=64)
    ap.add_argument("--frame", action="store_true")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--mb", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--nbuf", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=100000)
    ap.add_argument("--mc-moment", type=int, default=200000)
    ap.add_argument("--mcmc-chains", type=int, default=200000)
    ap.add_argument("--mcmc-sweeps", type=int, default=3000)
    ap.add_argument("--mcmc-eps", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--refine", type=str, default="2,4,8,16")
    ap.add_argument("--ckpt-dir", type=str, default="")
    ap.add_argument("--tag", type=str, default="stiefel")
    ap.add_argument("--out", type=str, default="json/results_stiefel.json")
    args = ap.parse_args()
    if args.antithetic and args.frame:
        # The frame term -lambda tr(C^T X) is NOT invariant under
        # T_s(X) = S X D (it picks up sign flips on every entry of C), so the
        # 16-fold sign symmetry is broken and the augmentation would be wrong.
        print("[guard] --frame breaks the sign symmetry; disabling --antithetic")
        args.antithetic = False
    return {"verify": run_verify, "train": run_train, "ref": run_ref,
            "sweep": run_sweep}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
