"""KS(E), constraint error and energy dispersion for the Stiefel St(4,2) sweep.

`json/results_rasbs_stiefel.json` stores `E_std` and the constraint error for the
R-ASBS port but leaves `KS_E` and `E_err` as NaN, so the published comparison
rested on the mean energy alone.  Both missing statistics are functions of the
stored samples, so they are recovered here from `ckpt/rasbs_b*.pt` scored against
the *same* MCMC reference used for IASBS (`ckpt/stiefel_grid_b*_mcmc.pt`), with
identical definitions on both sides.

The two methods were sampled at different sizes -- 5,000 for the R-ASBS port,
100,000 for IASBS -- and KS has an O(1/sqrt(n)) null floor, so a raw comparison
would flatter the smaller sample.  Every KS below is therefore reported twice:
at each method's own n, and with IASBS subsampled to the R-ASBS n, together with
the finite-sample floor measured by scoring exact MCMC draws of that size
against the reference CDF.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

BETAS = ["1.3", "2", "5"]
DT = torch.float64


def ks_vs_ref(E, ref_sorted, cdf):
    return C.ks_against_grid_cdf(np.asarray(E, dtype=np.float64),
                                 ref_sorted, cdf)


def energy_of(X, beta, H):
    """E(X) = tr(X^T H X) + beta * <quadratic penalty>, as StiefelProblem does."""
    raise SystemExit("unused")


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stiefel", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "stiefel.py"))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)

    matched = json.load(open("json/results_stiefel_matched.json"))
    grid = json.load(open("json/results_stiefel_grid.json"))
    ref_tab = json.load(open("json/results_stiefel_ref.json"))["betas"]

    rows = []
    for b in BETAS:
        beta = float(b)
        prob = S.StiefelProblem(sigma=2.0 ** 0.5, beta=beta, steps=199,
                                nq=64, frame=False, device="cpu")

        # shared MCMC reference: the same draws the IASBS numbers were scored on
        mc = torch.load(f"ckpt/stiefel_grid_b{beta:g}_mcmc.pt",
                        map_location="cpu", weights_only=False)["samples"]
        Eref = prob.energy(mc.to(DT)).cpu().numpy()
        g = np.sort(Eref)
        cdf = (np.arange(len(g)) + 1.0) / len(g)
        Eref_mean = float(np.mean(Eref))
        Eref_std = float(np.std(Eref, ddof=1))

        # R-ASBS port
        # The port scores its samples with the R-ASBS paper's circulant H,
        # whose eigenbasis differs from the IASBS diag(1,2,5,8) even though the
        # spectrum is identical.  St(4,2) is invariant under left O(4), so the
        # LAW of E under the Gibbs target is the same for both and the scalars
        # are comparable -- but each set of samples must be scored with the H it
        # was trained against.
        H_RASBS = torch.tensor([[4.0, 0.5, 2.5, 1.0],
                                [0.5, 4.0, 1.0, 2.5],
                                [2.5, 1.0, 4.0, 0.5],
                                [1.0, 2.5, 0.5, 4.0]], dtype=DT)
        ra = torch.load(f"ckpt/rasbs_b{beta:g}.pt", map_location="cpu",
                        weights_only=False)["samples"].to(DT)
        Era = torch.einsum("bnp,nm,bmp->b", ra, H_RASBS, ra).cpu().numpy()
        n_ra = len(Era)

        # IASBS, full sweep leg
        ia = torch.load(f"ckpt/stiefel_grid_b{beta:g}_seed0.pt",
                        map_location="cpu", weights_only=False)["samples"].to(DT)
        Eia = prob.energy(ia).cpu().numpy()

        # finite-sample KS floor at the R-ASBS sample size
        rng = np.random.default_rng(0)
        floor = float(np.mean([
            ks_vs_ref(rng.choice(Eref, n_ra, replace=False), g, cdf)
            for _ in range(20)]))
        # IASBS subsampled to the R-ASBS n, averaged over draws
        ks_ia_sub = float(np.mean([
            ks_vs_ref(rng.choice(Eia, n_ra, replace=False), g, cdf)
            for _ in range(20)]))

        key = [k for k in matched["betas"] if float(k) == beta][0]
        mm = matched["betas"][key]["seeds"][0]
        gkey = [k for k in grid["betas"] if float(k) == beta][0]
        gg = grid["betas"][gkey]["seeds"][0]

        rows.append(dict(
            beta=b,
            E_ref_table=ref_tab[[k for k in ref_tab if float(k)==beta][0]]["E_mean"],
            E_ref_used=Eref_mean, E_ref_std=Eref_std,
            n_ra=n_ra, n_ia=len(Eia),
            ra_dE=abs(float(np.mean(Era)) - Eref_mean),
            ra_ks=ks_vs_ref(Era, g, cdf),
            ra_std=float(np.std(Era, ddof=1)),
            ra_con=float(C.stiefel_constraint_error(ra).max()),
            ia_dE=gg["E_err"], ia_ks=gg["KS_E"], ia_std=gg["E_std"],
            ia_con=gg["constraint"], ia_ks_sub=ks_ia_sub,
            m6_dE=mm["E_err"], m6_ks=mm["KS_E"], m6_std=mm["E_std"],
            m6_con=mm["constraint"],
            ks_floor=floor,
        ))

    hdr = ("beta  n_ra  n_ia | R-ASBS dE / KS / std / con | IASBS dE / KS / "
           "KS@n_ra / std / con | IASBS600 dE / KS / std / con | KS floor@n_ra"
           " | E_ref")
    print(hdr)
    for r in rows:
        print(f"{r['beta']:>4} {r['n_ra']:5d} {r['n_ia']:6d} | "
              f"{r['ra_dE']:.4f} {r['ra_ks']:.4f} {r['ra_std']:.4f} "
              f"{r['ra_con']:.1e} | "
              f"{r['ia_dE']:.4f} {r['ia_ks']:.4f} {r['ia_ks_sub']:.4f} "
              f"{r['ia_std']:.4f} {r['ia_con']:.1e} | "
              f"{r['m6_dE']:.4f} {r['m6_ks']:.4f} {r['m6_std']:.4f} "
              f"{r['m6_con']:.1e} | {r['ks_floor']:.4f} | "
              f"{r['E_ref_used']:.4f} (table {r['E_ref_table']:.4f}) "
              f"std {r['E_ref_std']:.4f}")

    with open("json/results_stiefel_extra.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("  wrote json/results_stiefel_extra.json")


if __name__ == "__main__":
    main()
