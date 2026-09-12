"""Weakness 2, Stiefel side: one table joining the energy-law metrics, the
frame-law MMD^2, feasibility, oracle calls and wall time for every IASBS and
R-ASBS leg that exists on disk.

Aggregator only -- it trains nothing.  Energy-law numbers are read back out of
the training JSONs (they were produced by the same evaluation path on both
sides); MMD^2 is recomputed here from `ckpt/` so that every leg, including ones
that post-date `frame_law_audit.py`, is scored against the same MCMC reference
with the same bandwidth, the same 4000-vs-4000 draw and the same generator seed.

Rows are grouped into `equal oracle budget` and `equal wall time`, which are
different controls and must not be read as one ranking.

  python iasbs/analysis/weakness2_stiefel.py
"""
import json, os, sys
import numpy as np
import torch

import _paths                                    # noqa: F401

ROOT = _paths.ROOT

import bridge_audit as A            # noqa: E402
import rasbs_port as RP              # noqa: E402

DEV = "cuda:0"
DT = torch.float64
MN, REPS = 4000, 8
REF_CKPT = os.path.join(ROOT, "ckpt", "stiefel_frame_s5_b1_mcmc.pt")

U, _ = RP.basis_rotation(DEV)
U = U.to(DT)


def load_samples(path, key=None, rot=False):
    c = torch.load(path, map_location="cpu", weights_only=False)
    s = c["samples"] if key is None else c["extra"]["samples_refined"][str(key)]
    s = s.to(DEV).to(DT)
    if rot:                          # R-ASBS stores X in its own basis: X_r = U X
        s = U.transpose(0, 1) @ s
    return s


# ---------------------------------------------------------------- legs
# (label, group, training json, ckpt stem, rotate?, refine key, n_seeds)
LEGS = [
    ("IASBS native",  "oracle", "json/results_stiefel_frame_s5.json",
     "stiefel_frame_s5_b1_seed%d.pt", False, None, 5),
    ("IASBS600",      "oracle", "json/results_stiefel_frame600.json",
     "stiefel_frame600_b1_seed%d.pt", False, None, 5),
    ("R-ASBS",        "oracle", "json/results_rasbs_frame.json",
     "rasbs_frame_b1_seed%d.pt", True, None, 5),
    ("R-ASBS 6.1x",   "budget", "json/results_w2_rasbs_frame_big.json",
     "rasbs_w2_big_b1_seed%d.pt", True, None, 3),
    ("IASBS wall310", "wall",   "json/results_w2_stiefel_wall310.json",
     "stiefel_w2_wall310_b1_seed%d.pt", False, None, 5),
]

# step-refinement ladder, energy law only (no extra MMD draws needed beyond
# what `frame_law_audit.py` already reports for R-ASBS)
LADDER = [
    ("IASBS native",  "json/results_stiefel_frame_s5.json"),
    ("IASBS600",      "json/results_stiefel_frame600.json"),
    ("R-ASBS",        "json/results_rasbs_frame.json"),
    ("R-ASBS 6.1x",   "json/results_w2_rasbs_frame_big.json"),
]


def energy_rows(jpath, steps_key=None):
    """-> dict of aggregated energy-law metrics, or None if the file is absent."""
    p = os.path.join(ROOT, jpath)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    b = d.get("betas", {}).get("1.0")
    if b is None:
        return None
    seeds = b["seeds"]
    if steps_key is None:
        get = lambda s, k: s[k]                                    # noqa: E731
    else:
        get = lambda s, k: s["refined"][str(steps_key)][k]          # noqa: E731
    dE = [abs(get(s, "E_err")) for s in seeds]
    ks = [get(s, "KS_E") for s in seeds]
    con = [get(s, "constraint") for s in seeds]
    wall = [s.get("wall_s", s.get("train_s")) for s in seeds]
    orc = [s.get("oracle_calls") for s in seeds]
    f = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1))] if len(v) > 1 \
        else [float(v[0]), 0.0]                                    # noqa: E731
    return {
        "n_seeds": len(seeds),
        "dE": f(dE), "KS_E": f(ks),
        "constraint": float(np.max(con)),
        "wall_s": float(np.mean([w for w in wall if w is not None])),
        "oracle_calls": int(orc[0]) if orc[0] is not None else None,
        "per_seed_dE": dE, "per_seed_KS_E": ks,
    }


def main():
    ref = load_samples(REF_CKPT)
    Vref = ref.reshape(ref.shape[0], -1)
    s2 = A.median_bandwidth(Vref, cap=MN)
    gg = torch.Generator(device=DEV)
    gg.manual_seed(99)
    floor_m, floor_sd = A.split_half_floor(Vref, MN, s2, 16, gg)

    def mmd(V):
        vals = [A.mmd2_unbiased(
            V[torch.randperm(V.shape[0], generator=gg, device=DEV)[:MN]],
            Vref[torch.randperm(Vref.shape[0], generator=gg, device=DEV)[:MN]],
            s2) for _ in range(REPS)]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    out = {"bandwidth_sigma2": float(s2), "mmd_n": MN, "reps": REPS,
           "mmd_floor_splithalf": [float(floor_m), float(floor_sd)],
           "ref_ckpt": os.path.relpath(REF_CKPT, ROOT), "legs": {}}

    for label, group, jpath, stem, rot, rk, ns in LEGS:
        e = energy_rows(jpath, rk)
        if e is None:
            print("skip %-14s (%s missing)" % (label, jpath), flush=True)
            continue
        per = []
        for i in range(e["n_seeds"]):
            p = os.path.join(ROOT, "ckpt", stem % i)
            if not os.path.exists(p):
                continue
            X = load_samples(p, rk, rot=rot)
            m, _ = mmd(X.reshape(X.shape[0], -1))
            per.append(m)
        e["group"] = group
        e["MMD2"] = ([float(np.mean(per)), float(np.std(per, ddof=1))]
                     if len(per) > 1 else ([float(per[0]), 0.0] if per else None))
        e["MMD2_n_ckpt"] = len(per)
        out["legs"][label] = e
        print("%-14s dE %.4f+-%.4f  KS %.4f+-%.4f  MMD2 %s  |XtX-I| %.1e  "
              "orc %s  wall %.0fs  (%d seeds)"
              % (label, e["dE"][0], e["dE"][1], e["KS_E"][0], e["KS_E"][1],
                 ("%.3e" % e["MMD2"][0]) if e["MMD2"] else "--",
                 e["constraint"], e["oracle_calls"], e["wall_s"], e["n_seeds"]),
              flush=True)

    out["ladder"] = {}
    for label, jpath in LADDER:
        row = {}
        for k in (None, 398, 796):
            e = energy_rows(jpath, k)
            if e is None:
                continue
            row["199" if k is None else str(k)] = {
                "dE": e["dE"], "KS_E": e["KS_E"], "constraint": e["constraint"]}
        if row:
            out["ladder"][label] = row
            print("ladder %-14s " % label + "  ".join(
                "%s:%.4f" % (s, v["dE"][0]) for s, v in row.items()), flush=True)

    dst = os.path.join(ROOT, "json", "results_weakness2_stiefel.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("\nMMD2 split-half floor %.3e +- %.3e" % (floor_m, floor_sd))
    print("wrote", os.path.relpath(dst, ROOT))


if __name__ == "__main__":
    main()
