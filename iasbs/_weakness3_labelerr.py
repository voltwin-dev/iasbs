"""Reviewer weakness 3, part 5: corrector error propagated into the label.

KS_max at m = 1000 turns out to be a very noisy statistic -- across the three
shipped learned-corrector seeds it wanders over 0.07 .. 0.79 from one
evaluation to the next -- so comparing two training runs on it cannot
separate corrector error from controller training variance.

This script removes training from the question.  The controller is fitted by
Bregman regression onto

    y_{ji}(xi) = xi_i + Lambda_{ji}(xi),

whose minimiser is exp(a_{ji}) = y_{ji} / xi_i.  Lambda is built from the
terminal ratio R, and R is the only place the corrector enters.  So feeding
the exact analytic R and the learned rank-1 R through the SAME label routine
and comparing gives the error in the regression target itself, in nats of
control, with no optimiser, no seed and no tau-leap involved.  If that error
is small compared with the spread of the control the controller actually has
to represent, corrector learning cannot be what limits accuracy.

Reported per system size:
  dlog_y_rmse   RMSE of log y_learned - log y_analytic  (nats of control)
  dlog_y_p99    99th percentile of |.|
  sign_flips    fraction of labels where the two disagree on sign of y
  lam_rel_rmse  RMSE of (Lam_learned - Lam_analytic) / |Lam_analytic|
  ctrl_sd       sd of log(y_analytic / xi_i), the signal the net must fit
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

CKPTS = {
    "m32_seed0": ("ckpt/occ_nd_s32_seed0.pt", 32),
    "m32_seed1": ("ckpt/occ_nd_s32_seed1.pt", 32),
    "m32_s2": ("ckpt/occ_nd_s32_s2.pt", 32),
    "m128": ("ckpt/occ_nd_s128.pt", 128),
    "m128_s1": ("ckpt/occ_nd_s128_s1.pt", 128),
    "m128_s2": ("ckpt/occ_nd_s128_s2.pt", 128),
    "m1000": ("ckpt/occ_nd_s1000.pt", 1000),
    "m1000_s1": ("ckpt/occ_nd_s1000_s1.pt", 1000),
    "m1000_s2": ("ckpt/occ_nd_s1000_s2.pt", 1000),
}
NS = 4000
TVALS = [0.0, 0.25, 0.5, 0.75, 0.95]


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
        g = torch.Generator(device=DEV); g.manual_seed(7717)
        # held-out terminal states: exact target draws
        x1 = C.sample_inclusion_exact(NS, m, N, cfg["d"], device=DEV).to(DT)
        with torch.no_grad():
            ones = torch.ones(NS, device=DEV, dtype=DT)
            ah, bh = net_h(ones, x1, None)
            uv = sp.uv_nondirac(x1, ah.to(DT), bh.to(DT))
        rec = {}
        for t in TVALS:
            ct = C.occupation_c_t(m, cfg["gamma"] * (1.0 - t))
            # the bridge state the label is attached to, as in training
            tv = torch.full((NS,), t, device=DEV, dtype=DT)
            c0 = torch.randint(m, (NS,), device=DEV, generator=g)
            xt = O.bridge_scale(sp, x1, tv, generator=g, c0=c0)
            i_idx = torch.multinomial(xt / N, 1, generator=g)[:, 0]
            j_idx = torch.randint(m, (NS,), device=DEV, generator=g)
            ar = torch.arange(NS, device=DEV)
            occ = xt[ar, i_idx]
            with torch.no_grad():
                lam_l = sp.labels_full(x1, None, ct, i_idx, j_idx, uv=uv)
                lam_a = sp.labels_analytic(x1, ct, i_idx, j_idx)
            yl, ya = occ + lam_l, occ + lam_a
            keep = (i_idx != j_idx) & (occ > 0)
            pos = keep & (yl > 0) & (ya > 0)
            dl = torch.log(yl[pos]) - torch.log(ya[pos])
            rel = ((lam_l - lam_a).abs()
                   / lam_a.abs().clamp_min(1e-12))[keep]
            ctrl = torch.log(ya[pos] / occ[pos])
            rec[str(t)] = {
                "n": int(keep.sum()), "n_pos": int(pos.sum()),
                "dlog_y_rmse": float((dl ** 2).mean().sqrt()),
                "dlog_y_p99": float(dl.abs().quantile(0.99)),
                "dlog_y_max": float(dl.abs().max()),
                "sign_flips": float((((yl > 0) != (ya > 0)) & keep)
                                    .to(DT).mean()),
                "lam_rel_rmse": float((rel ** 2).mean().sqrt().clamp(max=1e6)),
                "lam_rel_median": float(rel.median()),
                "ctrl_sd": float(ctrl.std()),
                "ctrl_iqr": float(ctrl.quantile(0.75) - ctrl.quantile(0.25))}
            r = rec[str(t)]
            print(f"  {name:11s} t={t:4.2f}  dlog_y rmse={r['dlog_y_rmse']:.4f}"
                  f" p99={r['dlog_y_p99']:.4f}  flips={r['sign_flips']:.4f}  "
                  f"lam_rel_med={r['lam_rel_median']:.4f}  "
                  f"ctrl_sd={r['ctrl_sd']:.4f}", flush=True)
        out[name] = {"m": m, "by_t": rec}
        del net_h, blob
        torch.cuda.empty_cache()
    p = "/home/RESEARCH/iasbs/json/results_weakness3_labelerr.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
