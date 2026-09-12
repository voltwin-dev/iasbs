"""Reviewer weakness 3, part 6: why the EXACT corrector can score worse.

Measured fact: at m = 128 the analytic-corrector runs are reproducibly worse
on KS_max (0.082 over 2 seeds) than the learned rank-1 corrector runs (0.039
over 3 seeds), even though the analytic corrector is exact to machine
precision and the learned one carries 0.04-0.15 nats of label error.

Hypothesis.  The controller is rank-1 by construction: ScaleController emits
(alpha_i, beta_j) and the fitted log-intensity is a_{ji} = alpha_i + beta_j.
The Bregman optimum it chases is

    a*_{ji}(xi_t) = log( xi_i + Lambda_{ji} ) - log xi_i.

If a* is itself rank-1 the controller can hit it exactly; whatever part of a*
is NOT rank-1 is unrepresentable and is projected away.  A rank-1 corrector
keeps log R_{b<-a} = u_a + v_b, so it pushes a* toward the rank-1 family; the
exact corrector does not, because S(xi - e_a + e_b) couples a and b.  If that
is the mechanism, the exact labels must have a strictly larger non-rank-1
residual than the learned ones -- and the extra accuracy in the target is
spent on a component the network is structurally unable to fit.

Test.  For held-out states xi_t, form the full matrix a*_{ji} over a K x K
grid of (i, j), remove the best additive rank-1 part by two-way ANOVA
(a*_{ji} ~ mu + row_i + col_j), and report the residual as a fraction of the
total spread.  Done for the learned R and the analytic R on the SAME states
and the SAME (i, j) grid, so the only difference is the corrector.
"""

import json
import os
import sys

import numpy as np
import torch


import _paths                                    # noqa: F401

import common as C                                               # noqa: E402
import occupation as O                                           # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = torch.float64

CKPTS = {
    "m32_seed0": ("ckpt/occ_nd_s32_seed0.pt", 32),
    "m128": ("ckpt/occ_nd_s128.pt", 128),
    "m128_s1": ("ckpt/occ_nd_s128_s1.pt", 128),
    "m1000": ("ckpt/occ_nd_s1000.pt", 1000),
}
K = 96          # size of the (i, j) grid
NST = 48        # held-out states
TVALS = [0.25, 0.5, 0.75, 0.95]


def anova_residual(A):
    """Fraction of variance of A (S, K, K) left after removing mu+row+col."""
    mu = A.mean(dim=(1, 2), keepdim=True)
    ri = A.mean(dim=2, keepdim=True) - mu
    cj = A.mean(dim=1, keepdim=True) - mu
    R = A - (mu + ri + cj)
    tot = (A - mu).pow(2).mean(dim=(1, 2))
    res = R.pow(2).mean(dim=(1, 2))
    return res, tot


def main():
    out = {}
    for name, (path, m) in CKPTS.items():
        if not os.path.exists(path):
            print("skip", name, flush=True)
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["extra"]["config"]
        N = cfg["N"]
        sp = O.ScaleOccupation(m=m, N=N, d=cfg["d"], tau=cfg["tau"],
                               gamma=cfg["gamma"], device=DEV)
        net_h = O.ScaleController(m, N, hidden=cfg["hidden"]).to(DEV)
        net_h.load_state_dict(blob["state_dicts"]["corrector"])
        net_h.eval()
        g = torch.Generator(device=DEV); g.manual_seed(1234)
        x1 = C.sample_inclusion_exact(NST, m, N, cfg["d"], device=DEV).to(DT)
        with torch.no_grad():
            ones = torch.ones(NST, device=DEV, dtype=DT)
            ah, bh = net_h(ones, x1, None)
            uv_all = sp.uv_nondirac(x1, ah.to(DT), bh.to(DT))
        rec = {}
        for t in TVALS:
            ct = C.occupation_c_t(m, cfg["gamma"] * (1.0 - t))
            tv = torch.full((NST,), t, device=DEV, dtype=DT)
            c0 = torch.randint(m, (NST,), device=DEV, generator=g)
            xt = O.bridge_scale(sp, x1, tv, generator=g, c0=c0)
            accL, accA, nA = [], [], 0
            for s in range(NST):
                # i must have xi_t[i] > 0 for the label to be defined
                occ_nz = torch.nonzero(xt[s] > 0)[:, 0]
                if len(occ_nz) < 4:
                    continue
                ii = occ_nz[torch.randperm(len(occ_nz), generator=g,
                                           device=DEV)[:K]]
                jj = torch.randperm(m, generator=g, device=DEV)[:K]
                Ki, Kj = len(ii), len(jj)
                I = ii[:, None].expand(Ki, Kj).reshape(-1)
                J = jj[None, :].expand(Ki, Kj).reshape(-1)
                xr = x1[s][None].expand(Ki * Kj, m)
                uvr = (uv_all[0][s][None].expand(Ki * Kj, m),
                       uv_all[1][s][None].expand(Ki * Kj, m))
                with torch.no_grad():
                    lamL = sp.labels_full(xr, None, ct, I, J, uv=uvr)
                    lamA = sp.labels_analytic(xr, ct, I, J)
                occ = xt[s][I]
                yL, yA = occ + lamL, occ + lamA
                ok = (yL > 0) & (yA > 0) & (I != J)
                if float(ok.to(DT).mean()) < 0.95:
                    continue
                aL = (torch.log(yL.clamp_min(1e-300))
                      - torch.log(occ)).view(Ki, Kj)
                aA = (torch.log(yA.clamp_min(1e-300))
                      - torch.log(occ)).view(Ki, Kj)
                # the number of occupied modes varies per state, so the grids
                # are ragged and each one is reduced on its own
                accL.append(anova_residual(aL[None]))
                accA.append(anova_residual(aA[None]))
                nA += 1
            if nA == 0:
                continue
            rL = torch.cat([a for a, _ in accL])
            tL = torch.cat([b for _, b in accL])
            rA = torch.cat([a for a, _ in accA])
            tA = torch.cat([b for _, b in accA])
            rec[str(t)] = {
                "n_states": nA, "K": K,
                "learned_nonrank1_frac": float((rL / tL).mean()),
                "analytic_nonrank1_frac": float((rA / tA).mean()),
                "learned_resid_sd": float(rL.sqrt().mean()),
                "analytic_resid_sd": float(rA.sqrt().mean()),
                "learned_total_sd": float(tL.sqrt().mean()),
                "analytic_total_sd": float(tA.sqrt().mean())}
            r = rec[str(t)]
            print(f"  {name:11s} t={t:4.2f}  nonrank1 frac: learned="
                  f"{r['learned_nonrank1_frac']:.4f} analytic="
                  f"{r['analytic_nonrank1_frac']:.4f}   resid sd: "
                  f"{r['learned_resid_sd']:.4f} vs "
                  f"{r['analytic_resid_sd']:.4f}  (tot "
                  f"{r['learned_total_sd']:.4f}/{r['analytic_total_sd']:.4f})",
                  flush=True)
        out[name] = {"m": m, "by_t": rec}
        del net_h, blob
        torch.cuda.empty_cache()
    p = os.path.join(_paths.ROOT, "json", "results_weakness3_rank1.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
