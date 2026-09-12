"""Reviewer weakness 3, part 3: is the large-m residual tau-leap error?

The shipped non-Dirac controllers are re-evaluated with NO retraining at the
production step count and at refinements of it.  The controller is a function
of (t, eta) only, so evaluating it on a finer grid is legitimate: the learned
object is the jump intensity, and the tau-leap discretisation of it is a
separate approximation.  If the metrics move when only `steps` changes, the
residual is integration error; if they do not, it is the controller (or the
corrector that trained it).

Also reports the metrics of the *stored* terminal samples so the fresh
evaluation can be checked against the number in the paper.
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

RUNS = {
    "m32": ("ckpt/occ_nd_s32_seed0.pt", [128, 256, 512, 1024], 10000),
    "m128": ("ckpt/occ_nd_s128.pt", [128, 256, 512, 1024], 10000),
    "m1000": ("ckpt/occ_nd_s1000.pt", [256, 512, 1024], 4000),
    # analytic-corrector controllers, same refinement ladder
    "m32_analytic": ("ckpt/occ_nd_s32_analytic.pt",
                     [128, 256, 512, 1024], 10000),
    "m32_analytic_s1": ("ckpt/occ_nd_s32_analytic_s1.pt",
                        [128, 512, 1024], 10000),
    "m128_analytic": ("ckpt/occ_nd_s128_analytic.pt",
                      [128, 256, 512, 1024], 10000),
    "m128_analytic_s1": ("ckpt/occ_nd_s128_analytic_s1.pt",
                         [128, 512, 1024], 10000),
}


def main():
    out = {}
    for name, (path, steplist, ns) in RUNS.items():
        if not os.path.exists(path):
            print("skip", name, path, flush=True)
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["extra"]["config"]
        m, N = cfg["m"], cfg["N"]
        sp = O.ScaleOccupation(m=m, N=N, d=cfg["d"], tau=cfg["tau"],
                               gamma=cfg["gamma"], device=DEV)
        net = O.ScaleController(m, N, hidden=cfg["hidden"]).to(DEV)
        net.load_state_dict(blob["state_dicts"]["control"])
        net.eval()
        torch.manual_seed(4242)
        exact = C.sample_inclusion_exact(ns, m, N, cfg["d"],
                                         device=DEV).to(DT)
        rec = {"config_steps": cfg["steps"], "n_samples": ns, "by_steps": {}}
        # stored terminal samples from the original run, for cross-checking
        st = blob["samples"].to(DEV).to(DT)
        rec["stored"] = O.scale_metrics(sp, st, exact)
        for s in steplist:
            with torch.no_grad():
                c0 = torch.randint(m, (ns,), device=DEV)
                ours = O.simulate_scale(sp, net, ns, s, c0=c0)
            mm = O.scale_metrics(sp, ours, exact)
            rec["by_steps"][str(s)] = mm
            print(f"  {name:6s} steps={s:5d}  KS_occ={mm['KS_occ']:.4f}  "
                  f"KS_max={mm['KS_max']:.4f}  W1max={mm['W1_max']:.4f} "
                  f"({mm['W1_max_frac']:.4f})  E={mm['E_ours']:.4f} vs "
                  f"{mm['E_exact']:.4f}  E_KS={mm['E_KS']:.4f}  "
                  f"viol={mm['violations']}", flush=True)
        out[name] = rec
        del net, blob
        torch.cuda.empty_cache()
    p = os.path.join(_paths.ROOT, "json", "results_weakness3_tauleap.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
