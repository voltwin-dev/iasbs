"""Reviewer weakness 3, part 1: the analytic (symmetry-derived) corrector.

Claim under test.  For the non-Dirac occupation experiments the source is
uniform over the m pure states N e_c, the reference generator is the complete
graph, and the target pi(eta) prop prod_i Gamma(eta_i+d)/(eta_i! Gamma(d)).
All three objects are S_m-invariant, so the static Schrodinger potential
f_0 must be constant on supp nu_0 and therefore

    fhat_1(xi)  prop  (1/m) sum_c p^base_{1|0}(xi | N e_c)  =:  pbar(xi).

With q^{(c)}_j = o + r [j = c],  o = (1-r)/m,  r = exp(-m*Gamma/(m-1)),

    pbar(xi) = (N! / prod_j xi_j!) o^N S(xi),   S(xi) = (1/m) sum_c rho^{xi_c},
    rho = (o + r)/o,

so the corrector log-ratio the training loop has to learn is closed form

    h_{b<-a}(xi) = log pbar(xi - e_a + e_b) - log pbar(xi)
                 = log xi_a - log(xi_b + 1) + log S(xi') - log S(xi),

and the terminal ratio that enters the label is

    log R_{b<-a}(xi) = -log(xi_a - 1 + d) + log(xi_b + d)
                       + log S(xi) - log S(xi').

S(xi') is O(1) from S(xi):  m S(xi') = m S(xi) + (rho-1)(s_b - s_a/rho),
s_a = rho^{xi_a}.

Three independent checks are run here.

A1  m = N = 4, full enumeration.  Solve the static bridge exactly with
    Sinkhorn and verify (i) the scaling u_c = f_0(N e_c) is constant across
    the m source atoms, (ii) log fhat_1 from Sinkhorn equals log pbar up to
    an additive constant, where pbar is assembled from the enumerated
    semigroup rows that Sinkhorn also used.
A2  m = N = 4.  Verify the closed form o/r/rho/S against the same enumerated
    semigroup rows -- this tests the large-m formula, which never enumerates.
A3  Verify that the Bregman regression target used by the training loop,
    Q_{ba}(c, xi) = (xi_a/(xi_b+1)) q^{(c)}_b / q^{(c)}_a, has conditional
    mean exp(h_{b<-a}(xi)) given xi, by exact enumeration over c.

Then the learned correctors shipped in ckpt/occ_nd_s{32,128,1000}*.pt are
scored against the analytic h on held-out exact target samples.
"""

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as C                                               # noqa: E402
import occupation as O                                           # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = torch.float64


# ---------------------------------------------------------------- analytic
class AnalyticCorrector:
    """Closed-form  log S,  h_{b<-a},  log R_{b<-a}  for the uniform source."""

    def __init__(self, sp):
        self.sp = sp
        self.o, self.r = sp.q_atoms(sp.gamma)
        self.rho = (self.o + self.r) / self.o
        self.lrho = math.log(self.rho)
        self.m, self.d = sp.m, sp.d

    def logS(self, xi):
        """log (1/m) sum_c rho^{xi_c}.  Stable via logsumexp.  xi: (B, m)."""
        return torch.logsumexp(self.lrho * xi, dim=1) - math.log(self.m)

    def logS_shift(self, xi, a, b):
        """log S(xi - e_a + e_b), stable, O(1) per row."""
        ar = torch.arange(len(xi), device=xi.device)
        z = self.lrho * xi
        mx = z.max(dim=1, keepdim=True).values
        e = torch.exp(z - mx)
        tot = e.sum(dim=1)
        sa, sb = e[ar, a], e[ar, b]
        # replace rho^{xi_a} -> rho^{xi_a - 1} and rho^{xi_b} -> rho^{xi_b + 1}
        new = tot - sa - sb + sa / self.rho + sb * self.rho
        # a == b leaves the state unchanged
        new = torch.where(a == b, tot, new)
        return torch.log(new.clamp_min(1e-300)) + mx[:, 0] - math.log(self.m)

    def h(self, xi, a, b):
        """log pbar(xi - e_a + e_b) - log pbar(xi)."""
        ar = torch.arange(len(xi), device=xi.device)
        xa, xb = xi[ar, a], xi[ar, b]
        return (torch.log(xa.clamp_min(1e-300)) - torch.log(xb + 1.0)
                + self.logS_shift(xi, a, b) - self.logS(xi))

    def logR(self, xi, a, b):
        """log f_1(xi - e_a + e_b) / f_1(xi),  f_1 = mu / pbar."""
        ar = torch.arange(len(xi), device=xi.device)
        xa, xb = xi[ar, a], xi[ar, b]
        return (-torch.log((xa - 1.0 + self.d).clamp_min(1e-300))
                + torch.log(xb + self.d)
                + self.logS(xi) - self.logS_shift(xi, a, b))


# ------------------------------------------------------------------- A1/A2
def check_small(out, m=4, N=4, d=0.5, tau=1.0, gamma=4.0):
    sp = O.OccupationSpace(m=m, N=N, d=d, tau=tau, gamma=gamma, device=DEV)
    sp.set_nondirac_source(np.ones(m))
    u, logfhat, err = O.exact_sinkhorn(sp)

    # A1(i): f_0 constant on the source orbit
    f0 = u / sp.nu0 * sp.nu0            # u IS f_0 (row scaling of Gamma)
    f0 = u
    rel = float((f0.max() - f0.min()) / f0.mean().abs())

    # A1(ii): Sinkhorn corrector vs the symmetric mixture, up to a constant
    logpbar = torch.logsumexp(sp.logP01_src - math.log(m), dim=0)   # (M,)
    dlt = logfhat - logpbar
    a1_spread = float(dlt.max() - dlt.min())

    # A2: closed form against the enumerated semigroup rows
    an = AnalyticCorrector(O.ScaleOccupation(m=m, N=N, d=d, tau=tau,
                                             gamma=gamma, device=DEV))
    xi = sp.Sf
    lfact = torch.lgamma(xi + 1.0).sum(1)
    logpbar_cf = (math.lgamma(N + 1.0) - lfact + N * math.log(an.o)
                  + an.logS(xi))
    a2_max = float((logpbar_cf - logpbar).abs().max())

    # A2b: the two single-particle routes agree
    q0 = C.occupation_single_particle_probs(m, gamma, 0)
    a2b = float(np.abs(np.array([an.o + an.r if j == 0 else an.o
                                 for j in range(m)]) - np.asarray(q0)).max())

    # A3: conditional mean of the training target equals exp(h)
    g = torch.Generator(device=DEV); g.manual_seed(31337)
    idx = torch.randint(sp.M, (4096,), device=DEV, generator=g)
    x = sp.Sf[idx]
    a = torch.multinomial(x / N, 1, generator=g)[:, 0]
    b = torch.randint(m, (4096,), device=DEV, generator=g)
    ar = torch.arange(4096, device=DEV)
    # posterior over the source atom c given xi
    lpc = sp.logP01_src[:, idx].T - math.log(m)                    # (B, m)
    w = torch.softmax(lpc, dim=1)
    qa = an.o + an.r * (torch.arange(m, device=DEV)[None, :] == a[:, None]).to(DT)
    qb = an.o + an.r * (torch.arange(m, device=DEV)[None, :] == b[:, None]).to(DT)
    Q = (x[ar, a] / (x[ar, b] + 1.0))[:, None] * (qb / qa)
    EQ = (w * Q).sum(1)
    keep = (a != b) & (x[ar, a] > 0)
    a3 = float((torch.log(EQ[keep]) - an.h(x, a, b)[keep]).abs().max())

    # A4: exact R table from the Sinkhorn corrector vs the analytic logR
    sp.rebuild_R(logfhat)
    Rtab = sp.R[idx][ar, a, b]
    la = an.logR(x, a, b)
    keep2 = keep & (Rtab > 0)
    a4 = float((torch.log(Rtab[keep2]) - la[keep2]).abs().max())

    # A5: the O(m) analytic label routine against the enumerated full sum
    sps = O.ScaleOccupation(m=m, N=N, d=d, tau=tau, gamma=gamma, device=DEV)
    a5 = 0.0
    for t in (0.0, 0.25, 0.5, 0.75, 0.9):
        ct = C.occupation_c_t(m, gamma * (1.0 - t))
        lam_ex = sp.labels_full(ct)                       # (M, n_edges)
        ii = sp.edge_i.repeat(sp.M)
        jj = sp.edge_j.repeat(sp.M)
        xx = sp.Sf.repeat_interleave(sp.n_edges, dim=0)
        lam_an = sps.labels_analytic(xx, ct, ii, jj).view(sp.M, sp.n_edges)
        a5 = max(a5, float((lam_an - lam_ex).abs().max()))

    out["A_symmetry"] = {
        "m": m, "N": N, "sinkhorn_marginal_err": err,
        "f0_relative_spread": rel,
        "f0_values": [float(v) for v in f0],
        "A1_logfhat_minus_logpbar_spread": a1_spread,
        "A2_closed_form_logpbar_max_abs_err": a2_max,
        "A2b_q_atoms_vs_common_max_abs_err": a2b,
        "A3_log_condmean_target_vs_h_max_abs_err": a3,
        "A4_log_R_exact_vs_analytic_max_abs_err": a4,
        "A5_labels_analytic_vs_enumerated_max_abs_err": a5,
        "o": an.o, "r": an.r, "rho": an.rho}
    print(f"[A] m=N={m}  sinkhorn_err={err:.3e}  f0 spread={rel:.3e}  "
          f"A1={a1_spread:.3e}  A2={a2_max:.3e}  A2b={a2b:.3e}  "
          f"A3={a3:.3e}  A4={a4:.3e}  A5={a5:.3e}", flush=True)
    return sp, an


# -------------------------------------------------------- learned vs exact
CKPTS = {
    "m32_seed0": ("ckpt/occ_nd_s32_seed0.pt", 32, 32),
    "m32_seed1": ("ckpt/occ_nd_s32_seed1.pt", 32, 32),
    "m32_s2": ("ckpt/occ_nd_s32_s2.pt", 32, 32),
    "m128": ("ckpt/occ_nd_s128.pt", 128, 128),
    "m128_s1": ("ckpt/occ_nd_s128_s1.pt", 128, 128),
    "m128_s2": ("ckpt/occ_nd_s128_s2.pt", 128, 128),
    "m1000": ("ckpt/occ_nd_s1000.pt", 1000, 1000),
    "m1000_s1": ("ckpt/occ_nd_s1000_s1.pt", 1000, 1000),
    "m1000_s2": ("ckpt/occ_nd_s1000_s2.pt", 1000, 1000),
}


def corrector_error(out, nsamp=4000, seed=909):
    res = {}
    for name, (path, m, N) in CKPTS.items():
        if not os.path.exists(path):
            print(f"  skip {name}: missing {path}", flush=True)
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["extra"]["config"]
        sp = O.ScaleOccupation(m=m, N=N, d=cfg["d"], tau=cfg["tau"],
                               gamma=cfg["gamma"], device=DEV)
        an = AnalyticCorrector(sp)
        net_h = O.ScaleController(m, N, hidden=cfg["hidden"]).to(DEV)
        net_h.load_state_dict(blob["state_dicts"]["corrector"])
        net_h.eval()
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        # held-out states: exact target draws, never seen in training
        x = C.sample_inclusion_exact(nsamp, m, N, cfg["d"],
                                     device=DEV).to(DT)
        a = torch.multinomial(x / N, 1, generator=g)[:, 0]
        b = torch.randint(m, (nsamp,), device=DEV, generator=g)
        ar = torch.arange(nsamp, device=DEV)
        with torch.no_grad():
            ones = torch.ones(nsamp, device=DEV, dtype=DT)
            ah, bh = net_h(ones, x, None)
            hl = (ah.to(DT)[ar, a] + bh.to(DT)[ar, b])
        ha = an.h(x, a, b)
        keep = (a != b) & (x[ar, a] > 0)
        e = (hl - ha)[keep]
        # the corrector only enters R through h, and an additive constant in h
        # is NOT free (h is a ratio of pbar at two states), so the raw error
        # is the meaningful one; the centred error is reported alongside.
        res[name] = {
            "n": int(keep.sum()), "m": m, "N": N,
            "rmse": float((e ** 2).mean().sqrt()),
            "max_abs": float(e.abs().max()),
            "bias": float(e.mean()),
            "rmse_centred": float(((e - e.mean()) ** 2).mean().sqrt()),
            "h_analytic_sd": float(ha[keep].std()),
            "h_analytic_range": [float(ha[keep].min()), float(ha[keep].max())],
            "rho": an.rho}
        print(f"  {name:11s} m={m:4d}  h_rmse={res[name]['rmse']:.4f}  "
              f"max={res[name]['max_abs']:.4f}  bias={res[name]['bias']:+.4f}  "
              f"centred={res[name]['rmse_centred']:.4f}  "
              f"sd(h_exact)={res[name]['h_analytic_sd']:.4f}", flush=True)
        # also the induced error on the terminal ratio log R
        with torch.no_grad():
            uvl = sp.uv_nondirac(x, ah.to(DT), bh.to(DT))
        lR_learn = uvl[0][ar, a] + uvl[1][ar, b]
        lR_ex = an.logR(x, a, b)
        er = (lR_learn - lR_ex)[keep]
        res[name]["logR_rmse"] = float((er ** 2).mean().sqrt())
        res[name]["logR_max_abs"] = float(er.abs().max())
        res[name]["logR_exact_sd"] = float(lR_ex[keep].std())
        del net_h, blob
        torch.cuda.empty_cache()
    out["corrector_error"] = res


def main():
    out = {}
    check_small(out)
    print("[D] learned corrector vs analytic, held-out target states",
          flush=True)
    corrector_error(out)
    p = "/home/RESEARCH/iasbs/json/results_weakness3_corrector.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
