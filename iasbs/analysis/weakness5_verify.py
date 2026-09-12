"""Reviewer weakness 5, part 1b: verify the *exact* reference sampler itself.

The fixed-endpoint validation is only as good as the exact conditional sampler
it compares against, so that sampler is checked two independent ways.

1. Chapman-Kolmogorov / normalising constant.  Rejection sampling with
   proposal p_small(centre, .) and envelope M has acceptance probability

       E[accept] = (1 / M) int p_small(c, z) p_large(c', z) dz
                 = p_{r01}(X0, X1) / M,

   because the two heat kernels compose to the terminal one.  Multiplying the
   measured acceptance rate by M must therefore reproduce the terminal
   transition density, evaluated from a table that the rejection sampler never
   touches.  This tests the semigroup property, the table normalisation and
   the sampler in one number.

2. Deterministic quadrature (S^2 only, where the space is two dimensional).
   The conditional first moments E[z.X0], E[z.X1] are recomputed by Gauss-
   Legendre x uniform-azimuth quadrature of the closed-form conditional
   density, with no random numbers at all, and compared with the rejection
   estimates.
"""

import json
import math
import os
import sys

import numpy as np
import torch


import _paths                                    # noqa: F401

import common as C                                              # noqa: E402
import sphere as SPH                                            # noqa: E402
import weakness5_bridge as W                                   # noqa: E402

DEV = W.DEV
DT = torch.float64


def main():
    gen = torch.Generator(device=DEV)
    gen.manual_seed(424242)
    W.N_REF = 20000
    out = {"sphere": {}, "stiefel": {}}

    # ---------------------------------------------------------------- S^2
    tab01 = SPH.HeatTable([W.R01], ntheta=16385, device=DEV)
    Yq, wq = SPH.sphere_quadrature(nu=400, nphi=800, device=DEV)
    e3 = torch.tensor([0., 0., 1.], device=DEV, dtype=DT)

    def at_angle(a):
        return torch.tensor([math.sin(a), 0.0, math.cos(a)], device=DEV, dtype=DT)

    g0 = torch.tensor([0.4, -0.7, 0.5], device=DEV, dtype=DT); g0 /= g0.norm()
    g1 = torch.tensor([-0.8, 0.1, -0.6], device=DEV, dtype=DT); g1 /= g1.norm()
    pairs = {"P1_close_0.35": (e3, at_angle(0.35)),
             "P2_orthogonal_pi/2": (e3, at_angle(math.pi / 2)),
             "P3_near_antipodal_2.90": (e3, at_angle(2.90)),
             "P4_generic_x0": (g0, g1)}

    for pn, (x0, x1) in pairs.items():
        si0 = torch.zeros(1, device=DEV, dtype=torch.long)
        p01 = float(torch.exp(tab01.logp_at(si0, (x0 @ x1).reshape(1)
                                            .clamp(-1., 1.)))[0])
        for t in W.TIMES:
            ex = W.S2Exact(t, DEV)
            Z, info = ex.sample(x0, x1, W.N_REF, gen)
            # envelope M: recompute exactly as sample() did is not possible
            # (pilot is consumed), so redo the pilot with the same generator
            # state is unnecessary -- instead use the CK relation in the form
            # acc * M = p01 with M recovered from the recorded acceptance.
            # We therefore recompute M directly here.
            c_prop = x0 if ex.k_prop == 0 else x1
            c_wgt = x1 if ex.k_prop == 0 else x0
            zp = ex._propose(W.PILOT, c_prop, gen)
            lw = ex.logp(ex.k_wgt, (zp @ c_wgt).clamp(-1., 1.))
            logM = float(lw.max()) + math.log(W.SAFETY)
            # measure the acceptance rate at THIS M
            z = ex._propose(400000, c_prop, gen)
            lw = ex.logp(ex.k_wgt, (z @ c_wgt).clamp(-1., 1.))
            u = torch.rand(z.shape[0], generator=gen, device=DEV, dtype=DT)
            acc = float((torch.log(u.clamp(min=1e-300)) < (lw - logM))
                        .to(DT).mean())
            ck = acc * math.exp(logM)

            # deterministic quadrature of the conditional density
            c0 = (Yq @ x0).clamp(-1., 1.)
            c1 = (Yq @ x1).clamp(-1., 1.)
            sA = torch.zeros(c0.shape, device=DEV, dtype=torch.long)
            sB = torch.ones(c0.shape, device=DEV, dtype=torch.long)
            lg = ex.tab.logp_at(sA, c0) + ex.tab.logp_at(sB, c1)
            ww = wq * torch.exp(lg - lg.max())
            Zq = float(ww.sum())
            q_x0 = float((ww * c0).sum() / Zq)
            q_x1 = float((ww * c1).sum() / Zq)
            # the same normalising constant, by quadrature
            ck_q = float((wq * torch.exp(lg)).sum())

            out["sphere"][f"{pn}@t={t}"] = {
                "p01_table": p01, "CK_acc_times_M": ck,
                "CK_rel_err": ck / p01 - 1.0,
                "CK_quadrature": ck_q, "CK_quad_rel_err": ck_q / p01 - 1.0,
                "quad_E_z_x0": q_x0, "quad_E_z_x1": q_x1,
                "rej_E_z_x0": W.obs_stats(Z @ x0),
                "rej_E_z_x1": W.obs_stats(Z @ x1),
                "accept_rate": acc}
            e = out["sphere"][f"{pn}@t={t}"]
            print(f"  S2 {pn:24s} t={t}  p01={p01:.6f}  CK={ck:.6f} "
                  f"({e['CK_rel_err']:+.2e})  quadCK={ck_q:.6f} "
                  f"({e['CK_quad_rel_err']:+.2e})  "
                  f"E[z.x0] quad={q_x0:+.5f} rej={e['rej_E_z_x0'][0]:+.5f}"
                  f"+-{e['rej_E_z_x0'][1]:.5f}", flush=True)

    # ------------------------------------------------------------ St(4,2)
    # Same CK test.  The fibre-quadrature kernel is a density with respect to
    # a fixed measure on St whose normalisation is set by the psi-measure
    # convention, so the test is that acc * M / p^{St}_{r01}(X0,X1) is the
    # SAME constant for every pair and every time (it comes out as 1).
    E0 = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
                      device=DEV, dtype=DT)

    def rot(ang, i, j):
        M = torch.zeros(4, 4, device=DEV, dtype=DT)
        M[i, j], M[j, i] = ang, -ang
        return torch.matrix_exp(M)

    st_pairs = {
        "Q1_close": (E0, rot(0.25, 0, 2) @ E0),
        "Q2_mid": (E0, rot(math.pi / 4, 0, 2) @ rot(math.pi / 5, 1, 3) @ E0),
        "Q3_orthogonal_complement": (
            E0, torch.tensor([[0., 0.], [0., 0.], [1., 0.], [0., 1.]],
                             device=DEV, dtype=DT)),
        "Q4_generic_X0_equivariance": (rot(0.9, 0, 3) @ rot(-0.6, 1, 2) @ E0,
                                       rot(2.1, 0, 2) @ rot(1.3, 1, 3) @ E0),
    }
    ex01 = W.StExact(0.5, device=DEV)          # only used for its psi nodes
    import stiefel as ST                                          # noqa: E402
    tab01 = ST.S3Table(np.array([C.stiefel_spin_factor_time(W.R01),
                                 C.stiefel_spin_factor_time(W.R01)]),
                       ntheta=16385, device=DEV)
    ex01.tab = tab01                            # both rows are the r01 clock
    for pn, (X0, X1) in st_pairs.items():
        _, R0i = W.R_of(X0)
        p01 = float(torch.exp(ex01.logk(0, R0i, X1[None]))[0])
        for t in W.TIMES:
            ex = W.StExact(t, device=DEV)
            R0, R0i = W.R_of(X0)
            R1, R1i = W.R_of(X1)
            Rp = R0 if ex.k_prop == 0 else R1
            Rwi = R1i if ex.k_prop == 0 else R0i
            Zp = ex._propose(W.PILOT, Rp, gen)
            logM = float(ex.logk(ex.k_wgt, Rwi, Zp).max()) + math.log(W.SAFETY)
            Z = ex._propose(400000, Rp, gen)
            lw = ex.logk(ex.k_wgt, Rwi, Z)
            u = torch.rand(Z.shape[0], generator=gen, device=DEV, dtype=DT)
            acc = float((torch.log(u.clamp(min=1e-300)) < (lw - logM))
                        .to(DT).mean())
            ck = acc * math.exp(logM)
            out["stiefel"][f"{pn}@t={t}"] = {
                "p01_quadrature": p01, "CK_acc_times_M": ck,
                "CK_ratio": ck / p01, "accept_rate": acc}
            print(f"  St {pn:28s} t={t}  p01={p01:.6f}  CK={ck:.6f}  "
                  f"ratio={ck / p01:.5f}  acc={acc:.4f}", flush=True)

    with open(os.path.join(_paths.ROOT, "json", "results_weakness5_verify.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote json/results_weakness5_verify.json")


if __name__ == "__main__":
    main()
