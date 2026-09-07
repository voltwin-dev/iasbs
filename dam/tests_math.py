"""
Hard mathematical gate for the DAM implementation (tests T0--T10).

No Section-6 DAM experiment may be run until every test here passes.  The
tests are ordered so that a failure localises the bug: T0/T8 check the terminal
factor, T1/T2/T10 check the path Radon-Nikodym derivative, T3/T4 check the
adjoint estimator, T5 checks the loss, T6/T7 check the simulator itself, and T9
checks the O(m) factorised rate algebra.

    python -m dam.tests_math
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                              # noqa: E402
from structured_asbs import fixed_ising as FI                    # noqa: E402
from structured_asbs import occupation as OC                     # noqa: E402
from dam.core import (rollout_ctmc, estimate_log_adjoint, gkl_loss,
                      gkl_literal, rates_ns, TINY)               # noqa: E402
from dam.discrete import (IsingAdapter, OccAdapter, ScaleAdapter,
                          scale_logf1)                           # noqa: E402

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def perturb(net, scale=0.3, seed=1):
    """Give a zero-initialised controller a deliberately nonzero output."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    last = net.net[-1]
    with torch.no_grad():
        last.weight.copy_(scale * torch.randn(last.weight.shape, generator=g)
                          .to(last.weight.device, last.weight.dtype))
        last.bias.copy_(scale * torch.randn(last.bias.shape, generator=g)
                        .to(last.bias.device, last.bias.dtype))
    return net


# ----------------------------------------------------------------------------
# T0 / T8 -- the terminal factor
# ----------------------------------------------------------------------------
def T0(ising, occ):
    """p_base(x | x_0) exp(-g(x)), normalised, must equal the exact target.

    This is the single most dangerous place to be wrong: using the energy alone
    instead of the density ratio silently changes the problem DAM solves.
    """
    pb = torch.exp(ising.log_kappa_full[ising.dist0])
    w = pb * torch.exp(ising.logf1)
    e1 = float((w / w.sum() - ising.pi).abs().max())
    w = occ.pb0 * torch.exp(occ.logf1)
    e2 = float((w / w.sum() - occ.pi).abs().max())
    return check("T0 terminal-loss identity", max(e1, e2) < 1e-12,
                 f"ising {e1:.3e}  occupation {e2:.3e}  (tol 1e-12)")


def T8(occ):
    """scale_logf1 must reproduce OccupationSpace.logf1 up to one constant."""
    sp = OC.ScaleOccupation(m=occ.m, N=occ.N, d=occ.d, tau=occ.tau,
                            gamma=occ.gamma, source=occ.source, device=DEV)
    a = scale_logf1(sp, occ.S.to(torch.float64))
    b = occ.logf1
    d = (a - a.mean()) - (b - b.mean())
    err = float(d.abs().max())
    return check("T8 scale logf1 identity", err < 1e-12,
                 f"max centered diff {err:.3e}  (tol 1e-12)")


# ----------------------------------------------------------------------------
# T1 / T2 -- the path Radon-Nikodym derivative
# ----------------------------------------------------------------------------
def T1(cases):
    worst = 0.0
    for tag, ad, net, x0, steps in cases:
        t0 = torch.rand(x0.shape[0], device=DEV) * 0.5
        _, logw, _ = rollout_ctmc(ad, net, x0, t0, steps)
        worst = max(worst, float(logw.abs().max()))
    return check("T1 model == base => logW == 0", worst < 1e-10,
                 f"max |logW| {worst:.3e}  (tol 1e-10)")


def T2(cases, n=4096):
    """E_model[W] = 1 for a deliberately nonzero frozen controller."""
    ok = True
    msgs = []
    for tag, ad, net, x0, steps in cases:
        xs = torch.repeat_interleave(x0[:1], n, dim=0)
        for t in (0.0, 0.5):
            t0 = torch.full((n,), t, device=DEV)
            _, logw, _ = rollout_ctmc(ad, net, xs, t0, steps)
            w = torch.exp(logw)
            mu = float(w.mean())
            se = float(w.std() / math.sqrt(n))
            z = abs(mu - 1.0) / max(se, 1e-12)
            ok = ok and z < 5.0
            msgs.append(f"{tag}@t={t}: {mu:.4f}+-{2*se:.4f} (z={z:.1f})")
    return check("T2 E[W] == 1", ok, "  ".join(msgs))


# ----------------------------------------------------------------------------
# T3 / T4 -- the adjoint estimator
# ----------------------------------------------------------------------------
def T3(cases_phi, n=8192):
    """E_model[W f1(X_1)] = phi_t(x), with phi from the exact tables."""
    ok = True
    msgs = []
    for tag, ad, net, steps, states, logphi in cases_phi:
        for si, s in enumerate([0, steps // 2]):
            t = s / steps
            for x in states:
                xs = torch.full((n,), int(x), device=DEV, dtype=torch.long)
                t0 = torch.full((n,), t, device=DEV)
                z, logw, _ = rollout_ctmc(ad, net, xs, t0, steps)
                v = torch.exp(logw + ad.log_f1(z))
                mu, se = float(v.mean()), float(v.std() / math.sqrt(n))
                ref = float(torch.exp(logphi[s][int(x)]))
                z_sc = abs(mu - ref) / max(se, 1e-12)
                ok = ok and z_sc < 5.0
                msgs.append(f"{tag}[x={int(x)},t={t:.2f}] "
                            f"{mu:.4e} vs {ref:.4e} (z={z_sc:.1f})")
    return check("T3 E[W f1] == phi_t", ok, "  ".join(msgs))


def T4(ad, net, steps, logphi, Ks=(1, 4, 16, 64, 256), B=256):
    """DAM estimator against the exact adjoint.

    A finite-K ratio estimator is biased and its single numerator rollout keeps
    an O(1) variance no matter how large K is, so the thing that must converge
    is the DENOMINATOR, which estimates phi_t(x) directly.  Its relative error
    should fall like 1/sqrt(K).  The mean multiplier is reported alongside.
    """
    sp = ad.sp
    g = torch.Generator(device=DEV).manual_seed(7)
    ti = torch.randint(steps, (B,), device=DEV, generator=g)
    x = torch.randint(sp.M, (B,), device=DEV, generator=g)
    t = ti.to(torch.float64) / steps
    with torch.no_grad():
        rt = ad.rate_state(net, t, x)
        e, _, _ = ad.sample_edge(rt, g)
        y = ad.apply_edge(x, e, torch.ones_like(e, dtype=torch.bool))
    ex = logphi[ti, y] - logphi[ti, x]
    rows = []
    for K in Ks:
        with torch.no_grad():
            log_m, st = estimate_log_adjoint(ad, net, t, x, y, K, steps,
                                             generator=g)
        den_err = float(((torch.exp(st["log_den"]) - torch.exp(logphi[ti, x]))
                         / torch.exp(logphi[ti, x])).abs().mean())
        rows.append((K, den_err,
                     float((torch.exp(log_m) - torch.exp(ex)).mean().abs())))
    ok = rows[-1][1] < rows[0][1] / 4.0
    return check("T4 estimator vs exact, K sweep", ok,
                 "  ".join(f"K={K}: den relerr {d:.4f} mult bias {b:.3e}"
                           for K, d, b in rows))


# ----------------------------------------------------------------------------
# T5 -- the loss
# ----------------------------------------------------------------------------
def T5(B=1024):
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(B, generator=g) * 0.7).to(DEV).requires_grad_(True)
    a2 = a.detach().clone().requires_grad_(True)
    log_m = (torch.randn(B, generator=g) * 0.7).to(DEV)
    r = torch.rand(B, generator=g).to(DEV) + 0.1
    log_q = torch.log(torch.rand(B, generator=g).to(DEV) + 0.1)
    gkl_loss(a, log_m, r, log_q).backward()
    gkl_literal(a2, log_m, r, log_q).backward()
    rel = float((a.grad - a2.grad).abs().max()
                / a2.grad.abs().max().clamp_min(1e-300))
    return check("T5 gKL gradient identity", rel < 1e-12,
                 f"relative gradient error {rel:.3e}  (tol 1e-12)")


# ----------------------------------------------------------------------------
# T6 / T7 -- the simulator and the constraints
# ----------------------------------------------------------------------------
def T6(cases_ref, n=200000):
    ok = True
    msgs = []
    for tag, ad, steps, p_ref in cases_ref:
        net = ad_zero_net(ad)
        idx, _, _ = rollout_ctmc(ad, net, ad.source_state(n),
                                 torch.zeros(n, device=DEV), steps)
        emp = torch.bincount(idx, minlength=p_ref.shape[0]).double()
        emp = emp / emp.sum()
        tv = 0.5 * float((emp - p_ref).abs().sum())
        iid = torch.multinomial(p_ref, n, replacement=True)
        e2 = torch.bincount(iid, minlength=p_ref.shape[0]).double()
        e2 = e2 / e2.sum()
        floor = 0.5 * float((e2 - p_ref).abs().sum())
        ok = ok and tv < 3.0 * floor + 1e-3
        msgs.append(f"{tag}: TV {tv:.5f} vs iid floor {floor:.5f}")
    return check("T6 uncontrolled endpoint law", ok, "  ".join(msgs))


def T7(cases, n=20000):
    ok = True
    msgs = []
    for tag, ad, net, _x0, steps in cases:
        x1, _, _ = rollout_ctmc(ad, net, ad.source_state(n),
                                torch.zeros(n, device=DEV), steps)
        v = ad.violations(x1)
        ok = ok and v == 0
        msgs.append(f"{tag}: {v}/{n}")
    return check("T7 constraint preservation", ok, "  ".join(msgs))


_ZERO_CACHE = {}


def ad_zero_net(ad):
    """A zero-initialised controller of the architecture the adapter expects."""
    key = id(ad)
    if key in _ZERO_CACHE:
        return _ZERO_CACHE[key]
    if isinstance(ad, IsingAdapter):
        net = FI.SwapController(ad.sp.n, hidden=64).to(DEV)
    elif isinstance(ad, OccAdapter):
        net = OC.OccController(ad.sp.m, ad.sp.N, hidden=64).to(DEV)
    else:
        net = OC.ScaleController(ad.sp.m, ad.sp.N, hidden=32).to(DEV)
    _ZERO_CACHE[key] = net
    return net


# ----------------------------------------------------------------------------
# T9 -- the O(m) factorised rate algebra
# ----------------------------------------------------------------------------
def T9(m=5, N=7, B=64):
    sp = OC.ScaleOccupation(m=m, N=N, d=0.5, tau=1.0, gamma=4.0, device=DEV)
    ad = ScaleAdapter(sp)
    net = perturb(OC.ScaleController(m, N, hidden=32).to(DEV), 0.5, seed=5)
    g = torch.Generator(device=DEV).manual_seed(11)
    eta = torch.distributions.Multinomial(
        total_count=N, probs=torch.ones(m, device=DEV)).sample((B,)).double()
    t = torch.rand(B, device=DEV, generator=g)
    with torch.no_grad():
        rt = ad.rate_state(net, t, eta)
        al, be = rt.al, rt.be
    # explicit m(m-1) construction
    U = (sp.gamma / (m - 1.0)) * eta[:, :, None] * torch.exp(
        al[:, :, None] + be[:, None, :])                        # (B, i, j)
    off = ~torch.eye(m, dtype=torch.bool, device=DEV)
    Ufull = U * off
    e_R = float((Ufull.sum(dim=(1, 2)) - rt.R_model).abs().max()
                / rt.R_model.abs().max())
    qi_full = Ufull.sum(dim=2) / Ufull.sum(dim=(1, 2), keepdim=True)[:, :, 0]
    qi_fact = rt.wi / rt.Z[:, None]
    e_i = float((qi_full - qi_fact).abs().max())
    # conditional destination law for a fixed departure mode.  Only rows with
    # eta_{i0} > 0 have a departure law at all; the rest are 0/0 by definition.
    i0 = 0
    live = eta[:, i0] > 0
    qj_full = (Ufull[live, i0]
               / Ufull[live, i0].sum(dim=1, keepdim=True))
    Bm = rt.Bj[live].clone()
    Bm[:, i0] = 0.0
    qj_fact = Bm / Bm.sum(dim=1, keepdim=True)
    e_j = float((qj_full - qj_fact).abs().max()) if int(live.sum()) else 0.0
    assert all(np.isfinite([e_R, e_i, e_j])), "non-finite T9 residual"
    return check("T9 O(m) factorised rates", max(e_R, e_i, e_j) < 1e-12,
                 f"R {e_R:.3e}  q(i) {e_i:.3e}  q(j|i) {e_j:.3e}  (tol 1e-12)")


# ----------------------------------------------------------------------------
# T10 -- a tiny 3-state CTMC checked against a direct path-density calculation
# ----------------------------------------------------------------------------
class ToyAdapter:
    """3-state fully connected CTMC with a hand-written piecewise-constant
    control table.  Nothing about it touches the benchmark code, so a sign or
    index error in dam/core.py cannot hide behind a compensating error in an
    adapter."""

    name = "toy"

    def __init__(self, R, A, steps, logf1, device=DEV):
        self.n = R.shape[0]
        tgt, red = [], []
        for x in range(self.n):
            row = [y for y in range(self.n) if y != x]
            tgt.append(row)
            red.append([float(R[y, x]) for y in row])
        self.tgt = torch.tensor(tgt, device=device)
        self.redge = torch.tensor(red, device=device, dtype=torch.float64)
        self.A = A                                    # (steps, n, n-1)
        self.steps = steps
        self.logf1v = logf1

    def _s(self, t):
        return torch.clamp((t * self.steps).floor().long(), 0, self.steps - 1)

    def rate_state(self, net, t, idx):
        av = self.A[self._s(t), idx]                  # (B, n-1)
        rb = self.redge[idx]
        u = rb * torch.exp(av)
        return rates_ns(u.sum(1), rb.sum(1), u=u, av=av)

    def sample_edge(self, rt, generator=None):
        ar = torch.arange(rt.u.shape[0], device=rt.u.device)
        e = torch.multinomial(rt.u, 1, generator=generator)[:, 0]
        return (e, rt.av[ar, e],
                torch.log(rt.u[ar, e]) - torch.log(rt.R_model))

    def apply_edge(self, idx, e, mask):
        return torch.where(mask, self.tgt[idx, e], idx)

    def log_f1(self, idx):
        return self.logf1v[idx]

    def source_state(self, batch):
        return torch.zeros(batch, device=DEV, dtype=torch.long)

    def violations(self, idx):
        return 0


def T10(steps=8, B=512, n_mc=200000):
    g = torch.Generator(device=DEV).manual_seed(23)
    Rm = torch.tensor([[0.0, 1.3, 0.4],
                       [2.1, 0.0, 0.9],
                       [0.7, 1.7, 0.0]], device=DEV, dtype=torch.float64)
    A = 0.8 * torch.randn(steps, 3, 2, device=DEV, generator=g,
                          dtype=torch.float64)
    logf1 = torch.tensor([0.0, -0.6, 0.9], device=DEV, dtype=torch.float64)
    ad = ToyAdapter(Rm, A, steps, logf1)

    # (a) trace-replay: recompute logW from raw tables, independently of core
    x0 = torch.randint(3, (B,), device=DEV, generator=g)
    t0 = torch.rand(B, device=DEV, generator=g) * 0.6
    tr = []
    _, logw, _ = rollout_ctmc(ad, None, x0, t0, steps, generator=g, trace=tr)
    man = torch.zeros(B, device=DEV, dtype=torch.float64)
    for rec in tr:
        s = torch.clamp((rec["tau"] * steps).floor().long(), 0, steps - 1)
        x = rec["x"]
        rb = ad.redge[x]
        a = A[s, x]
        man = man + (rb * torch.exp(a)).sum(1) * rec["h"] - rb.sum(1) * rec["h"]
        e = rec["edge"]
        ar = torch.arange(B, device=DEV)
        man = man + torch.where(rec["jump"], -a[ar, e],
                                torch.zeros_like(man))
    e_replay = float((man - logw).abs().max())

    # (b) the base-measure identity, against an exact matrix exponential
    Qn = Rm.cpu().numpy().copy()
    np.fill_diagonal(Qn, 0.0)
    np.fill_diagonal(Qn, -Qn.sum(axis=0))             # generator, columns sum 0
    t_test = 2.0 / steps
    P = torch.tensor(scipy.linalg.expm(Qn * (1.0 - t_test)), device=DEV)
    phi = (torch.exp(logf1)[None, :] @ P).reshape(-1)  # phi(x) = sum_z P[z,x] f1
    xs = torch.zeros(n_mc, device=DEV, dtype=torch.long)
    errs = []
    for x in range(3):
        z, lw, _ = rollout_ctmc(ad, None, xs + x,
                                torch.full((n_mc,), t_test, device=DEV), steps,
                                generator=g)
        v = torch.exp(lw + ad.log_f1(z))
        mu, se = float(v.mean()), float(v.std() / math.sqrt(n_mc))
        errs.append((float(phi[x]), mu, abs(mu - float(phi[x])) / max(se, 1e-12)))
    ok_mc = all(e[2] < 5.0 for e in errs)
    return check("T10 tiny CTMC path likelihood",
                 e_replay < 1e-10 and ok_mc,
                 f"trace replay {e_replay:.3e} (tol 1e-10); "
                 + " ".join(f"phi {a:.4f} vs {b:.4f} (z={z:.1f})"
                            for a, b, z in errs))


# ----------------------------------------------------------------------------
def main():
    C.use_repo_root()
    torch.manual_seed(0)
    print("building spaces ...", flush=True)
    ising = FI.FixedIsingSpace(L=4, J=1.0, tau=2.0, gamma=10.0, device=DEV)
    occ = OC.OccupationSpace(m=4, N=4, d=0.5, tau=1.0, gamma=4.0, device=DEV)
    steps = 32
    ad_i, ad_o = IsingAdapter(ising), OccAdapter(occ)

    net_i0 = FI.SwapController(ising.n, hidden=64).to(DEV)
    net_o0 = OC.OccController(occ.m, occ.N, hidden=64).to(DEV)
    sc = OC.ScaleOccupation(m=6, N=6, d=0.5, tau=1.0, gamma=4.0, device=DEV)
    ad_s = ScaleAdapter(sc)
    net_s0 = OC.ScaleController(sc.m, sc.N, hidden=32).to(DEV)

    zero_cases = [
        ("ising", ad_i, net_i0, ad_i.source_state(256), steps),
        ("occ", ad_o, net_o0, ad_o.source_state(256), steps),
        ("scale", ad_s, net_s0, ad_s.source_state(256), steps),
    ]
    net_i = perturb(FI.SwapController(ising.n, hidden=64).to(DEV), 0.25, 1)
    net_o = perturb(OC.OccController(occ.m, occ.N, hidden=64).to(DEV), 0.25, 2)
    net_s = perturb(OC.ScaleController(sc.m, sc.N, hidden=32).to(DEV), 0.25, 3)
    live_cases = [
        ("ising", ad_i, net_i, ad_i.source_state(256), steps),
        ("occ", ad_o, net_o, ad_o.source_state(256), steps),
        ("scale", ad_s, net_s, ad_s.source_state(256), steps),
    ]

    ts = np.arange(steps) / steps
    logphi_i = torch.log(ising.phi_grid(ts).clamp_min(1e-300))     # (T, M)
    logphi_o = occ.phi_grid(ts)                                    # (T, M)

    print("\n-- terminal factor --", flush=True)
    T0(ising, occ)
    T8(occ)
    print("\n-- path weight --", flush=True)
    T1(zero_cases)
    T2(live_cases)
    T10()
    print("\n-- adjoint estimator --", flush=True)
    T3([("ising", ad_i, net_i, steps,
         torch.tensor([ising.i0, 5, 100], device=DEV), logphi_i),
        ("occ", ad_o, net_o, steps,
         torch.tensor([occ.i0, 7, 20], device=DEV), logphi_o)])
    T4(ad_o, net_o, steps, logphi_o)
    print("\n-- loss --", flush=True)
    T5()
    print("\n-- simulator --", flush=True)
    T6([("ising", ad_i, steps, torch.exp(ising.log_kappa_full[ising.dist0])),
        ("occ", ad_o, steps, occ.pb0 / occ.pb0.sum())])
    T7(live_cases)
    print("\n-- factorised rates --", flush=True)
    T9()

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n=== {n_ok}/{len(RESULTS)} DAM gates passed ===")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED  {name}: {detail}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
