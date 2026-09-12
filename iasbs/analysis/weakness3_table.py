"""Reviewer weakness 3, part 4: the learned-vs-analytic corrector table.

Assembles one table per system size from the checkpoints, so every row is
measured the same way on the same exact-target sample:

  reference   uncontrolled reference process from the uniform source
  learned     IASBS with the trained rank-1 corrector  (shipped run)
  analytic    IASBS with the exact S_m-symmetric corrector, controller
              retrained with identical flags and --inner-h 0
  exact       independent draws from the inclusion target (self-comparison,
              i.e. the finite-sample floor of every metric)

Metrics: KS_occ, KS_max, W1(max) unnormalised and /N, mean energy,
E[max_i eta_i] and its exact counterpart, constraint violations.
"""

import json
import math
import os
import sys

import numpy as np
import torch


import _paths                                    # noqa: F401

import common as C                                               # noqa: E402
import occupation as O                                           # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = torch.float64

SIZES = {
    32: {"learned": "ckpt/occ_nd_s32_seed0.pt",
         "analytic": "ckpt/occ_nd_s32_analytic.pt", "steps": 128, "n": 10000},
    128: {"learned": "ckpt/occ_nd_s128.pt",
          "analytic": "ckpt/occ_nd_s128_analytic.pt", "steps": 128,
          "n": 10000},
    1000: {"learned": "ckpt/occ_nd_s1000.pt",
           "analytic": "ckpt/occ_nd_s1000_analytic.pt", "steps": 256,
           "n": 4000},
}


def row(sp, X, exact, gen=None):
    d = O.scale_metrics(sp, X, exact)
    mo = X.max(dim=1).values
    d["E_max_occ"] = float(mo.mean())
    d["E_max_occ_se"] = float(mo.std(unbiased=True) / math.sqrt(len(mo)))
    d["E_max_occ_exact"] = float(exact.max(dim=1).values.mean())
    E = C.inclusion_energy(X, sp.tau, sp.d)
    d["E_ours_se"] = float(E.std(unbiased=True) / math.sqrt(len(E)))
    d["n"] = int(len(X))
    return d


def main():
    out = {}
    for m, cfgs in SIZES.items():
        N = m
        blob = torch.load(cfgs["learned"], map_location="cpu",
                          weights_only=False)
        c = blob["extra"]["config"]
        sp = O.ScaleOccupation(m=m, N=N, d=c["d"], tau=c["tau"],
                               gamma=c["gamma"], device=DEV)
        torch.manual_seed(20260911)
        ns = cfgs["n"]
        exact = C.sample_inclusion_exact(ns, m, N, c["d"],
                                         device=DEV).to(DT)
        rec = {"m": m, "N": N, "steps": cfgs["steps"], "n_samples": ns,
               "rows": {}}

        # uncontrolled reference and a second exact draw (metric floor)
        with torch.no_grad():
            c0 = torch.randint(m, (ns,), device=DEV)
            ref = O.simulate_scale(sp, None, ns, cfgs["steps"], c0=c0)
        rec["rows"]["reference"] = row(sp, ref, exact)
        ex2 = C.sample_inclusion_exact(ns, m, N, c["d"], device=DEV).to(DT)
        rec["rows"]["exact_floor"] = row(sp, ex2, exact)

        for tag in ("learned", "analytic"):
            p = cfgs[tag]
            if not os.path.exists(p):
                print(f"  m={m} {tag}: MISSING {p}", flush=True)
                continue
            b = torch.load(p, map_location="cpu", weights_only=False)
            cc = b["extra"]["config"]
            net = O.ScaleController(m, N, hidden=cc["hidden"]).to(DEV)
            net.load_state_dict(b["state_dicts"]["control"])
            net.eval()
            with torch.no_grad():
                c0 = torch.randint(m, (ns,), device=DEV)
                X = O.simulate_scale(sp, net, ns, cfgs["steps"], c0=c0)
            r = row(sp, X, exact)
            r["iters"] = cc["iters"]
            r["params"] = (136450 if tag == "analytic" else 2 * 136450)
            rec["rows"][tag] = r
            del net, b
            torch.cuda.empty_cache()

        out[str(m)] = rec
        for k, v in rec["rows"].items():
            print(f"  m={m:4d} {k:12s} KS_occ={v['KS_occ']:.4f} "
                  f"KS_max={v['KS_max']:.4f} W1max={v['W1_max']:.4f} "
                  f"({v['W1_max_frac']:.4f}) E={v['E_ours']:.4f} "
                  f"Emax={v['E_max_occ']:.4f} viol={v['violations']}",
                  flush=True)

    p = os.path.join(_paths.ROOT, "json", "results_weakness3_table.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
