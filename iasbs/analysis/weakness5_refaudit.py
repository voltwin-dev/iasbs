"""Reviewer weakness 5, part 2: independent-reference audit for the frame target.

Section 5.7 scored IASBS and R-ASBS against a single MCMC reference and used a
"cross reference" that turned out to be a byte-identical copy of it, so the
reported noise floor was a split-half of one sample and could not see MCMC
bias.  Here three references are used:

  refA  ckpt/stiefel_frame_s5_b1_mcmc.pt          seed 0,        200k x 3000, eps 0.35
  refB  ckpt/stiefel_frame_b1_mcmc_refB.pt        seed 20260911, 200k x 3000, eps 0.35
  refC  ckpt/stiefel_frame_b1_mcmc_refC.pt        seed 777,      100k x 6000, eps 0.20

refA vs refB changes only the seed: their MMD^2 is pure Monte-Carlo
variability at fixed sampler.  refC additionally changes the proposal scale
and the chain length, so any *systematic* residual of the random-walk
Metropolis sampler shows up as an excess of A/B-vs-C over A-vs-B.  Every
method is then scored against all three, and the spread across references
bounds how much of its MMD^2 can be blamed on the reference.
"""

import json
import math
import os
import sys

import numpy as np
import torch


import _paths                                    # noqa: F401

import bridge_audit as A                                       # noqa: E402
import rasbs_port as RP                                         # noqa: E402

DEV = "cuda:0"
DT = torch.float64
MMD_N = 4000
REPS = 8
Cmat = torch.tensor([[0.7, -0.2], [0.1, 0.8], [-0.4, 0.3], [0.2, -0.5]],
                    dtype=DT, device=DEV)
U, _ = RP.basis_rotation(DEV)
U = U.to(DT)


def load(f, key=None, rot=False):
    c = torch.load(f, map_location="cpu", weights_only=False)
    s = c["samples"] if key is None else c["extra"]["samples_refined"][key]
    s = s.to(DEV).to(DT)
    if rot:                       # R-ASBS stores X in its own basis: X_r = U X
        s = U.transpose(0, 1) @ s
    return s


def obs(X):
    n = X.shape[0]
    tr = (X * Cmat[None]).sum((-1, -2))
    d = {"n": n,
         "E_trCX": float(tr.mean()),
         "E_trCX_se": float(tr.std(unbiased=True) / math.sqrt(n))}
    m = X.mean(0)
    d["EX"] = [[float(v) for v in r] for r in m]
    V = X.reshape(n, -1)
    Cv = torch.cov(V.T)
    d["cov"] = [[float(v) for v in r] for r in Cv]
    d["var_diag"] = [float(v) for v in torch.diagonal(Cv)]
    return d, V


def mmd_pair(Va, Vb, s2, gen, reps=REPS):
    vals = [A.mmd2_unbiased(
        Va[torch.randperm(Va.shape[0], generator=gen, device=DEV)[:MMD_N]],
        Vb[torch.randperm(Vb.shape[0], generator=gen, device=DEV)[:MMD_N]], s2)
        for _ in range(reps)]
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(reps))


def main():
    gen = torch.Generator(device=DEV)
    gen.manual_seed(5170)
    refs = {
        "refA": load("ckpt/stiefel_frame_s5_b1_mcmc.pt"),
        "refB": load("ckpt/stiefel_frame_b1_mcmc_refB.pt"),
        "refC": load("ckpt/stiefel_frame_b1_mcmc_refC.pt"),
    }
    O, V = {}, {}
    for k, X in refs.items():
        O[k], V[k] = obs(X)
    s2 = A.median_bandwidth(V["refA"], cap=4000)

    res = {"bandwidth_sigma2": s2, "mmd_n": MMD_N, "reps": REPS,
           "refs": O, "ref_floor": {}, "ref_cross": {}, "ref_moment": {},
           "methods": {}}

    for k in refs:
        m, sd = A.split_half_floor(V[k], MMD_N, s2, 16, gen)
        res["ref_floor"][k] = {"mean": m, "sd": sd}

    for a, b in [("refA", "refB"), ("refA", "refC"), ("refB", "refC")]:
        m, se = mmd_pair(V[a], V[b], s2, gen)
        res["ref_cross"][f"{a}_vs_{b}"] = {"mean": m, "se": se}
        EA, EB = np.array(O[a]["EX"]), np.array(O[b]["EX"])
        CA, CB = np.array(O[a]["cov"]), np.array(O[b]["cov"])
        res["ref_moment"][f"{a}_vs_{b}"] = {
            "d_trCX": O[a]["E_trCX"] - O[b]["E_trCX"],
            "d_trCX_se": math.hypot(O[a]["E_trCX_se"], O[b]["E_trCX_se"]),
            "EX_max_abs_diff": float(np.abs(EA - EB).max()),
            "cov_frob_diff": float(np.linalg.norm(CA - CB))}

    sets = {
        "IASBS_199": [("ckpt/stiefel_frame_s5_b1_seed%d.pt" % i, None, False)
                      for i in range(5)],
        "IASBS600_199": [("ckpt/stiefel_frame600_b1_seed%d.pt" % i, None, False)
                         for i in range(5)],
        "RASBS_199": [("ckpt/rasbs_frame_b1_seed%d.pt" % i, None, True)
                      for i in range(5)],
        "RASBS_398": [("ckpt/rasbs_frame_b1_seed%d.pt" % i, 398, True)
                      for i in range(5)],
        "RASBS_796": [("ckpt/rasbs_frame_b1_seed%d.pt" % i, 796, True)
                      for i in range(5)],
    }
    for name, fs in sets.items():
        per = []
        for f, key, rot in fs:
            X = load(f, key, rot)
            o, Vx = obs(X)
            for rk in refs:
                m, se = mmd_pair(Vx, V[rk], s2, gen)
                o[f"MMD2_{rk}"] = m
                o[f"MMD2_{rk}_se"] = se
            per.append(o)
        agg = {}
        for rk in refs:
            v = [p[f"MMD2_{rk}"] for p in per]
            agg[f"MMD2_{rk}"] = [float(np.mean(v)), float(np.std(v, ddof=1))]
        tv = [p["E_trCX"] for p in per]
        agg["E_trCX"] = [float(np.mean(tv)), float(np.std(tv, ddof=1))]
        EX = np.array([p["EX"] for p in per]).mean(0)
        agg["EX_err_vs_refA"] = float(np.abs(EX - np.array(O["refA"]["EX"])).max())
        CVm = np.array([p["cov"] for p in per]).mean(0)
        agg["cov_frob_err_vs_refA"] = float(
            np.linalg.norm(CVm - np.array(O["refA"]["cov"])))
        res["methods"][name] = {"per_seed": per, "agg": agg}
        print(f"{name:14s} trCX={agg['E_trCX'][0]:.5f}+-{agg['E_trCX'][1]:.5f} "
              f"A={agg['MMD2_refA'][0]:+.3e} B={agg['MMD2_refB'][0]:+.3e} "
              f"C={agg['MMD2_refC'][0]:+.3e} "
              f"EXerr={agg['EX_err_vs_refA']:.4f} "
              f"covF={agg['cov_frob_err_vs_refA']:.4f}", flush=True)

    out = os.path.join(_paths.ROOT, "json", "results_weakness5_refaudit.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("floors", {k: v["mean"] for k, v in res["ref_floor"].items()})
    print("cross", {k: v["mean"] for k, v in res["ref_cross"].items()})
    print("trCX", {k: (O[k]["E_trCX"], O[k]["E_trCX_se"]) for k in refs})
    print("wrote", out)


if __name__ == "__main__":
    main()
