"""
Discrete Adjoint Matching on the IASBS discrete benchmarks.

Three adapters (fixed-composition Ising, enumerable occupation, scalable
occupation) plus one training loop and one CLI.  Everything benchmark-specific
-- state spaces, reference bridges, controller architectures, target samplers,
metrics, checkpoint helpers -- is imported from the existing experiment
modules; nothing is duplicated here.

DAM uses the SAME Dirac source as the Dirac IASBS runs and the SAME terminal
factor

    g(z) = -log f1(z),        f1 propto mu(z) / p_base_{1|0}(z | x_0),

so it targets exactly the same terminal law.  It is NOT given the energy alone:
`g = E/tau` would be a different problem whenever the reference terminal law is
non-uniform, which it is here.

Usage
-----
    python -m dam.discrete ising            --L 4  --K 16
    python -m dam.discrete occupation       --m 4  --K 16
    python -m dam.discrete occupation-scale --m 32 --K 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                              # noqa: E402
from structured_asbs import fixed_ising as FI                    # noqa: E402
from structured_asbs import occupation as OC                     # noqa: E402
from dam.core import (rollout_ctmc, estimate_log_adjoint, gkl_loss,
                      rates_ns, TINY, LOG_M_CLIP)                # noqa: E402

torch.set_default_dtype(torch.float64)


# ============================================================================
# adapters
# ============================================================================
class IsingAdapter:
    """Fixed-composition Ising swap chain.

    Every legal occupied->empty swap has base rate gamma / (k (n-k)), so the
    base escape rate is exactly gamma and the controlled rate is
    (gamma / n_edges) exp(a_theta).  States are carried as enumeration indices;
    the (M, n_edges) target table does the jumping.
    """

    name = "ising"

    def __init__(self, space, clamp=20.0):
        self.sp = space
        self.clamp = clamp
        self.base = space.gamma / space.n_edges

    def rate_state(self, net, t, idx):
        sp = self.sp
        B = idx.shape[0]
        a = net(t, sp.S[idx]).to(torch.float64)
        b = torch.arange(B, device=sp.device).unsqueeze(1)
        av = a[b, sp.edge_i[idx], sp.edge_j[idx]].clamp(-self.clamp, self.clamp)
        u = self.base * torch.exp(av)
        R = u.sum(dim=1)
        Rb = torch.full_like(R, float(sp.gamma))
        return rates_ns(R, Rb, u=u, av=av)

    def sample_edge(self, rt, generator=None):
        B = rt.u.shape[0]
        ar = torch.arange(B, device=rt.u.device)
        e = torch.multinomial(rt.u, 1, generator=generator)[:, 0]
        log_a = rt.av[ar, e]
        log_q = (torch.log(rt.u[ar, e].clamp_min(TINY))
                 - torch.log(rt.R_model.clamp_min(TINY)))
        return e, log_a, log_q

    def apply_edge(self, idx, e, mask):
        return torch.where(mask, self.sp.tgt[idx, e], idx)

    def a_on_edge(self, net, t, idx, e):
        sp = self.sp
        ar = torch.arange(idx.shape[0], device=sp.device)
        a = net(t, sp.S[idx]).to(torch.float64)
        i = sp.edge_i[idx][ar, e]
        j = sp.edge_j[idx][ar, e]
        return a[ar, i, j].clamp(-self.clamp, self.clamp)

    def base_rate_on_edge(self, idx, e):
        return torch.full((idx.shape[0],), self.base, device=self.sp.device,
                          dtype=torch.float64)

    def log_f1(self, idx):
        return self.sp.logf1[idx]

    def source_state(self, batch):
        return torch.full((batch,), self.sp.i0, device=self.sp.device,
                          dtype=torch.long)

    def violations(self, idx):
        return int((self.sp.S[idx].sum(dim=1) != self.sp.k).sum())


class OccAdapter:
    """Enumerable occupation process.

    Transfer i -> j (i != j) has base rate gamma/(m-1) * eta_i, so the base
    escape rate is gamma N.  Edges with eta_i = 0 carry zero rate in both the
    base and the model, which preserves common support.
    """

    name = "occupation"

    def __init__(self, space, clamp=20.0):
        self.sp = space
        self.clamp = clamp
        self.c = space.gamma / (space.m - 1.0)

    def rate_state(self, net, t, idx):
        sp = self.sp
        a = net(t, sp.S[idx]).to(torch.float64)[:, sp.edge_i, sp.edge_j]
        av = a.clamp(-self.clamp, self.clamp)
        rate = self.c * sp.eocc[idx] * torch.exp(av)
        rate = torch.where(sp.emask[idx], rate, torch.zeros_like(rate))
        R = rate.sum(dim=1)
        Rb = torch.full_like(R, float(sp.gamma * sp.N))
        return rates_ns(R, Rb, u=rate, av=av)

    def sample_edge(self, rt, generator=None):
        B = rt.u.shape[0]
        ar = torch.arange(B, device=rt.u.device)
        e = torch.multinomial(rt.u, 1, generator=generator)[:, 0]
        log_a = rt.av[ar, e]
        log_q = (torch.log(rt.u[ar, e].clamp_min(TINY))
                 - torch.log(rt.R_model.clamp_min(TINY)))
        return e, log_a, log_q

    def apply_edge(self, idx, e, mask):
        return torch.where(mask, self.sp.etgt[idx, e], idx)

    def a_on_edge(self, net, t, idx, e):
        sp = self.sp
        ar = torch.arange(idx.shape[0], device=sp.device)
        a = net(t, sp.S[idx]).to(torch.float64)
        return a[ar, sp.edge_i[e], sp.edge_j[e]].clamp(-self.clamp, self.clamp)

    def base_rate_on_edge(self, idx, e):
        ar = torch.arange(idx.shape[0], device=self.sp.device)
        return self.c * self.sp.eocc[idx][ar, e]

    def log_f1(self, idx):
        return self.sp.logf1[idx]

    def source_state(self, batch):
        return torch.full((batch,), self.sp.i0, device=self.sp.device,
                          dtype=torch.long)

    def violations(self, idx):
        return int((self.sp.S[idx].sum(dim=1) != self.sp.N).sum())


def scale_logf1(sp, eta, logq=None):
    """log f1(eta) for the scalable occupation benchmark, up to a constant.

    mu(eta) propto prod_i Gamma(eta_i + d) / (eta_i! Gamma(d)) and the Dirac
    reference terminal law is Multinomial(N, q), so the log(eta_i!) terms
    cancel exactly and

        log f1(eta) = sum_i lgamma(eta_i + d) - sum_i eta_i log q_i + C.

    No enumeration and no normalisation constant are needed.
    """
    lq = torch.log(sp.q1()) if logq is None else logq
    e = eta.to(torch.float64)
    return torch.lgamma(e + sp.d).sum(-1) - (e * lq).sum(-1)


class ScaleAdapter:
    """Scalable occupation with the rank-1 DeepSets controller.

    The model rate is u_{ji} = gamma/(m-1) eta_i exp(alpha_i + beta_j), so with
    A_i = eta_i e^{alpha_i} and B_j = e^{beta_j},

        R_theta = gamma/(m-1) sum_i A_i (B_tot - B_i),
        q(i)    propto A_i (B_tot - B_i),
        q(j|i)  = B_j / (B_tot - B_i),   j != i,

    all O(m).  Note that (B_tot - B_i) cancels in the joint, so
    q(i, j) = A_i B_j / Z with Z = sum_i A_i (B_tot - B_i).

    Unlike `scale_step`, j = i is forbidden: a self-transfer is not a genuine
    off-diagonal CTMC jump and admitting one would corrupt the path density.
    """

    name = "occupation_scale"

    def __init__(self, sp, clamp=10.0):
        self.sp = sp
        self.clamp = clamp
        self.c = sp.gamma / (sp.m - 1.0)
        self.logq = torch.log(sp.q1())

    def rate_state(self, net, t, eta):
        sp = self.sp
        if net is None:
            al = torch.zeros_like(eta)
            be = torch.zeros_like(eta)
        else:
            al, be = net(t, eta, sp.source)
            al = al.to(torch.float64).clamp(-self.clamp, self.clamp)
            be = be.to(torch.float64).clamp(-self.clamp, self.clamp)
        A = eta.to(torch.float64) * torch.exp(al)
        Bj = torch.exp(be)
        Btot = Bj.sum(dim=1, keepdim=True)
        wi = A * (Btot - Bj)
        Z = wi.sum(dim=1)
        R = self.c * Z
        Rb = torch.full_like(R, float(sp.gamma * sp.N))
        return rates_ns(R, Rb, A=A, Bj=Bj, wi=wi, Z=Z, al=al, be=be)

    def sample_edge(self, rt, generator=None):
        B = rt.wi.shape[0]
        ar = torch.arange(B, device=rt.wi.device)
        i = torch.multinomial(rt.wi.clamp_min(0.0), 1,
                              generator=generator)[:, 0]
        Bm = rt.Bj.clone()
        Bm[ar, i] = 0.0
        j = torch.multinomial(Bm, 1, generator=generator)[:, 0]
        log_a = rt.al[ar, i] + rt.be[ar, j]
        log_q = (torch.log(rt.A[ar, i].clamp_min(TINY))
                 + torch.log(rt.Bj[ar, j].clamp_min(TINY))
                 - torch.log(rt.Z.clamp_min(TINY)))
        return (i, j), log_a, log_q

    def apply_edge(self, eta, edge, mask):
        i, j = edge
        ar = torch.arange(eta.shape[0], device=eta.device)
        mm = mask.to(eta.dtype)
        out = eta.clone()
        out[ar, i] = out[ar, i] - mm
        out[ar, j] = out[ar, j] + mm
        return out

    def a_on_edge(self, net, t, eta, edge):
        i, j = edge
        ar = torch.arange(eta.shape[0], device=eta.device)
        al, be = net(t, eta, self.sp.source)
        al = al.to(torch.float64).clamp(-self.clamp, self.clamp)
        be = be.to(torch.float64).clamp(-self.clamp, self.clamp)
        return al[ar, i] + be[ar, j]

    def base_rate_on_edge(self, eta, edge):
        i, _ = edge
        ar = torch.arange(eta.shape[0], device=eta.device)
        return self.c * eta.to(torch.float64)[ar, i]

    def log_f1(self, eta):
        return scale_logf1(self.sp, eta, self.logq)

    def source_state(self, batch):
        eta = torch.zeros(batch, self.sp.m, device=self.sp.device,
                          dtype=torch.float64)
        eta[:, self.sp.source] = float(self.sp.N)
        return eta

    def violations(self, eta):
        return int((eta.sum(dim=1).round().long() != self.sp.N).sum())


# ============================================================================
# shared DAM training step
# ============================================================================
def dam_step(ad, net, opt, sched, Xt, ti, steps, K, K_num, clip=10.0):
    """One generalized-KL matching update at the reciprocal states Xt.

    Returns (loss, stats).  The proposal q and the adjoint label are produced
    entirely under no_grad -- differentiating through the proposal is one of
    the explicit failure modes listed in the plan.
    """
    t = ti.to(torch.float64) / steps
    with torch.no_grad():
        rt = ad.rate_state(net, t, Xt)
        # A non-finite rate here would only surface later as an opaque
        # device-side assert inside multinomial, so fail loudly instead.
        if not bool(torch.isfinite(rt.R_model).all()):
            raise RuntimeError("dam_step: non-finite model escape rate -- the "
                               "controller has diverged")
        edge, _, log_q = ad.sample_edge(rt)
        Y = ad.apply_edge(Xt, edge, torch.ones_like(log_q, dtype=torch.bool))
        log_m, st = estimate_log_adjoint(ad, net, t, Xt, Y, K, steps,
                                         K_num=K_num)
        r = ad.base_rate_on_edge(Xt, edge)
        st["clipped"] = int((log_m.abs() > LOG_M_CLIP).sum())
    a = ad.a_on_edge(net, t, Xt, edge)
    loss = gkl_loss(a, log_m, r, log_q)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    if sched is not None:
        sched.step()
    return float(loss), st


def _acc(store, st):
    store["ess"].append(float(st["ess"].mean()))
    store["ess_all"].append(st["ess"].detach().cpu())
    store["jumps"] += int(st["jumps"])
    store["f1"] += int(st["f1_evals"])
    store["clipped"] += int(st.get("clipped", 0))


def _weight_report(store):
    ess = torch.cat(store["ess_all"]) if store["ess_all"] else torch.zeros(1)
    return {"ess_mean": float(ess.mean()),
            "ess_p10": float(torch.quantile(ess, 0.10)),
            "rollout_jumps": store["jumps"],
            "f1_evals": store["f1"],
            "clipped_labels": store["clipped"],
            "nonfinite_weights": 0}


def _new_store():
    return {"ess": [], "ess_all": [], "jumps": 0, "f1": 0,
            "clipped": 0}


# ============================================================================
# drivers
# ============================================================================
def run_ising(args):
    sp = FI.FixedIsingSpace(L=args.L, J=args.J, tau=args.tau, gamma=args.gamma)
    torch.manual_seed(args.seed)
    net = FI.SwapController(sp.n, hidden=args.hidden).to(sp.device)
    ad = IsingAdapter(sp)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iters * args.inner), eta_min=args.lr * 0.05)
    print(f"[DAM ising] L={args.L} |Omega|={sp.M}  params={n_par}  "
          f"steps={steps}  K={args.K}  device={sp.device}", flush=True)

    ts = np.arange(steps) / steps
    lk0 = sp.kappa_table(np.maximum(sp.gamma * ts, 1e-12))
    lk1 = sp.kappa_table(np.maximum(sp.gamma * (1 - ts), 1e-12))

    buf, hist, store = [], [], _new_store()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x0 = ad.source_state(args.batch)
        X1, _, jj = rollout_ctmc(ad, net, x0,
                                 torch.zeros(args.batch, device=sp.device),
                                 steps)
        store["jumps"] += int(jj.sum())
        buf.append(X1)
        if len(buf) > args.buffer:
            buf.pop(0)
        pool = torch.cat(buf)

        for _ in range(args.inner):
            sel = torch.randint(len(pool), (args.mb,), device=sp.device)
            ti = torch.randint(steps, (args.mb,), device=sp.device)
            with torch.no_grad():
                Xt = FI.sample_bridge(sp, pool[sel], ti, lk0, lk1)
            loss, st = dam_step(ad, net, opt, sched, Xt, ti, steps, args.K,
                                args.K_num)
            _acc(store, st)

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = FI.propagate_exact(
                    sp, lambda t: FI.net_mult_all_states(sp, net, t), steps)
            tv = 0.5 * float((p - sp.pi).abs().sum())
            eh = FI.energy_hist_tv(sp, p)
            hist.append({"iter": it, "TV": tv, "energy_hist_TV": eh,
                         "loss": loss, "ess": store["ess"][-1]})
            print(f"  it {it:4d}  loss {loss:11.4f}  TV {tv:.5f}  "
                  f"E-hist TV {eh:.5f}  ESS {store['ess'][-1]:.2f}/{args.K}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        p = FI.propagate_exact(
            sp, lambda t: FI.net_mult_all_states(sp, net, t), steps)
        idx, _, _ = rollout_ctmc(ad, net, ad.source_state(args.n_samples),
                                 torch.zeros(args.n_samples, device=sp.device),
                                 steps)
    tv = 0.5 * float((p - sp.pi).abs().sum())
    viol = ad.violations(idx)
    rmse = _mult_rmse_ising(sp, net, steps)
    return _finish(args, ad, net, sp, p=p, samples=idx, tv=tv, viol=viol,
                   hist=hist, store=store, n_par=n_par, wall=time.time() - t0,
                   headline={"TV": tv}, extra_diag={"log_multiplier_RMSE": rmse})


def _mult_rmse_ising(sp, net, steps):
    ctrl = FI.ExactControl(sp, steps)
    out = []
    with torch.no_grad():
        for s in [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]:
            t = s / steps
            d = (FI.net_mult_all_states(sp, net, t) - ctrl.all_states(t))
            out.append((t, float((d ** 2).mean().sqrt()), float(d.abs().max())))
    return out


def run_occupation(args):
    sp = OC.OccupationSpace(m=args.m, N=args.N, d=args.d, tau=args.tau,
                            gamma=args.gamma)
    torch.manual_seed(args.seed)
    net = OC.OccController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    ad = OccAdapter(sp)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iters * args.inner), eta_min=args.lr * 0.05)
    print(f"[DAM occupation] m={sp.m} N={sp.N} |X|={sp.M}  params={n_par}  "
          f"steps={steps}  K={args.K}  device={sp.device}", flush=True)

    ts = np.arange(steps) / steps
    K0 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * t)[sp.i0] for t in ts]), 1e-300)),
        device=sp.device)
    K1 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * (1 - t)) for t in ts]), 1e-300)),
        device=sp.device)

    buf, hist, store = [], [], _new_store()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x0 = ad.source_state(args.batch)
        X1, _, jj = rollout_ctmc(ad, net, x0,
                                 torch.zeros(args.batch, device=sp.device),
                                 steps)
        store["jumps"] += int(jj.sum())
        buf.append(X1)
        if len(buf) > args.buffer:
            buf.pop(0)
        pool = torch.cat(buf)

        for _ in range(args.inner):
            sel = torch.randint(len(pool), (args.mb,), device=sp.device)
            ti = torch.randint(steps, (args.mb,), device=sp.device)
            with torch.no_grad():
                Xt = OC.sample_bridge(sp, pool[sel], ti, K0, K1)
            loss, st = dam_step(ad, net, opt, sched, Xt, ti, steps, args.K,
                                args.K_num)
            _acc(store, st)

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = OC.propagate_exact(
                    sp, lambda t: OC.net_mult_all(sp, net, t), steps)
            tv = 0.5 * float((p - sp.pi).abs().sum())
            ho = 0.5 * float((OC.occ_hist(sp, p)
                              - OC.occ_hist(sp, sp.pi)).abs().sum())
            hist.append({"iter": it, "TV": tv, "occ_hist_TV": ho,
                         "loss": loss, "ess": store["ess"][-1]})
            print(f"  it {it:4d}  loss {loss:11.4f}  TV {tv:.5f}  "
                  f"occ-hist TV {ho:.5f}  ESS {store['ess'][-1]:.2f}/{args.K}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        p = OC.propagate_exact(sp, lambda t: OC.net_mult_all(sp, net, t), steps)
        idx, _, _ = rollout_ctmc(ad, net, ad.source_state(args.n_samples),
                                 torch.zeros(args.n_samples, device=sp.device),
                                 steps)
    tv = 0.5 * float((p - sp.pi).abs().sum())
    viol = ad.violations(idx)

    ctrl = OC.ExactControl(sp, steps)
    rmse = []
    with torch.no_grad():
        for s in [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]:
            t = s / steps
            d = (OC.net_mult_all(sp, net, t) - ctrl.all_states(t))
            d = torch.where(sp.emask, d, torch.zeros_like(d))
            rmse.append((t, float((d ** 2).sum().div(sp.emask.sum()).sqrt()),
                         float(d.abs().max())))
    return _finish(args, ad, net, sp, p=p, samples=idx, tv=tv, viol=viol,
                   hist=hist, store=store, n_par=n_par, wall=time.time() - t0,
                   headline={"TV": tv}, extra_diag={"log_multiplier_RMSE": rmse})


def run_scale(args):
    sp = OC.ScaleOccupation(m=args.m, N=args.N if args.N > 0 else args.m,
                            d=args.d, tau=args.tau, gamma=args.gamma)
    torch.manual_seed(args.seed)
    net = OC.ScaleController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    ad = ScaleAdapter(sp)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iters * args.inner), eta_min=args.lr * 0.05)
    print(f"[DAM occupation-scale] m={sp.m} N={sp.N}  params={n_par}  "
          f"steps={steps}  K={args.K}  device={sp.device}", flush=True)

    exact = C.sample_inclusion_exact(args.n_samples, sp.m, sp.N, sp.d,
                                     device=sp.device).to(torch.float64)
    with torch.no_grad():
        ref, _, _ = rollout_ctmc(ad, None, ad.source_state(args.n_samples),
                                 torch.zeros(args.n_samples, device=sp.device),
                                 steps)
    mref = OC.scale_metrics(sp, ref, exact)
    print(f"  reference (no control): KS_occ={mref['KS_occ']:.4f}  "
          f"KS_max={mref['KS_max']:.4f}  W1max/N={mref['W1_max_frac']:.4f}",
          flush=True)

    buf, hist, store = [], [], _new_store()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        X1, _, jj = rollout_ctmc(ad, net, ad.source_state(args.batch),
                                 torch.zeros(args.batch, device=sp.device),
                                 steps)
        store["jumps"] += int(jj.sum())
        buf.append(X1)
        if len(buf) > args.buffer:
            buf.pop(0)
        pool = torch.cat(buf)

        for _ in range(args.inner):
            sel = torch.randint(len(pool), (args.mb,), device=sp.device)
            ti = torch.randint(steps, (args.mb,), device=sp.device)
            with torch.no_grad():
                Xt = OC.bridge_scale(sp, pool[sel],
                                     ti.to(torch.float64) / steps)
            loss, st = dam_step(ad, net, opt, sched, Xt, ti, steps, args.K,
                                args.K_num)
            _acc(store, st)

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                ours, _, _ = rollout_ctmc(
                    ad, net, ad.source_state(args.n_samples),
                    torch.zeros(args.n_samples, device=sp.device), steps)
            mm = OC.scale_metrics(sp, ours, exact)
            mm.update({"iter": it, "loss": loss, "ess": store["ess"][-1]})
            hist.append(mm)
            print(f"  it {it:4d}  loss {loss:11.3f}  KS_occ {mm['KS_occ']:.4f} "
                  f" KS_max {mm['KS_max']:.4f}  W1max/N "
                  f"{mm['W1_max_frac']:.4f}  viol {mm['violations']}  "
                  f"ESS {store['ess'][-1]:.2f}/{args.K}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    mm = hist[-1]
    return _finish(args, ad, net, sp, p=None, samples=ours, tv=None,
                   viol=mm["violations"], hist=hist, store=store, n_par=n_par,
                   wall=time.time() - t0,
                   headline={"KS_occ": mm["KS_occ"], "KS_max": mm["KS_max"]},
                   extra_diag={"reference": mref, "metrics": mm},
                   exact_samples=exact)


def _finish(args, ad, net, sp, p, samples, tv, viol, hist, store, n_par, wall,
            headline, extra_diag, exact_samples=None):
    diag = _weight_report(store)
    diag["wall_sec"] = round(wall, 1)
    diag["K"] = args.K
    diag["K_num"] = args.K_num

    print("\n=== GATES ===")
    ok = viol == 0
    if tv is not None:
        print(f"  TV <= 0.05     : {'PASS' if tv <= 0.05 else 'FAIL'}  "
              f"({tv:.5f})")
        ok = ok and tv <= 0.05
    else:
        print(f"  KS_occ <= 0.05 : "
              f"{'PASS' if headline['KS_occ'] <= 0.05 else 'FAIL'}  "
              f"({headline['KS_occ']:.4f})")
        ok = ok and headline["KS_occ"] <= 0.05
    print(f"  constraint     : {'PASS' if viol == 0 else 'FAIL'}  ({viol})")
    print(f"  ESS mean {diag['ess_mean']:.2f} / {args.K}   p10 "
          f"{diag['ess_p10']:.2f}   f1 evals {diag['f1_evals']:,}   "
          f"jumps {diag['rollout_jumps']:,}   {diag['wall_sec']}s")

    if args.ckpt_dir:
        extra = {"config": vars(args), "headline": headline, "dam": diag,
                 "violations": viol}
        if p is not None:
            extra["exact_law"] = p.detach().cpu()
            extra["pi"] = sp.pi.detach().cpu()
            extra["states"] = sp.S.detach().cpu()
        if exact_samples is not None:
            extra["exact_samples"] = exact_samples.to(torch.int32).cpu()
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          samples=samples.to(torch.int32), extra=extra)
        print(f"  ckpt -> {pth}")

    res = {"method": "DAM", "benchmark": ad.name, "config": vars(args),
           "params": n_par, "headline": headline, "dam": diag,
           "violations": viol, "history": hist}
    res.update(extra_diag)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if ok else 1


# ============================================================================
def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ising", "occupation", "occupation-scale"])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--N", type=int, default=0, help="0 means N = m")
    ap.add_argument("--d", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--K-num", dest="K_num", type=int, default=1,
                    help="numerator rollouts; ablation only, keep at 1")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=0, help="0 = benchmark default")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=50)
    ap.add_argument("--n-samples", dest="n_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    if args.N <= 0:
        args.N = args.m
    stem = args.cmd.replace("-", "_")
    if args.cmd == "ising":
        args.tau = 2.0 if args.tau is None else args.tau
        args.gamma = 10.0 if args.gamma is None else args.gamma
        args.hidden = args.hidden or 512
        key = f"dam_ising_L{args.L}_K{args.K}"
    else:
        args.tau = 1.0 if args.tau is None else args.tau
        args.gamma = 4.0 if args.gamma is None else args.gamma
        args.hidden = args.hidden or (256 if args.cmd == "occupation" else 128)
        key = f"dam_{stem}_m{args.m}_K{args.K}"
    args.out = args.out or f"json/results_{key}.json"
    args.tag = args.tag or key

    return {"ising": run_ising, "occupation": run_occupation,
            "occupation-scale": run_scale}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
