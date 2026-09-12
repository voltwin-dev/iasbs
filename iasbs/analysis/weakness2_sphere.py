"""Reviewer weakness 2, part 1: controlled S^2 ablation, IASBS vs R-ASBS.

Section 4.4/4.5 compares the two samplers as they are shipped, which leaves
several nuisance factors unaligned at once: our headline sphere run uses a
Dirac source, sigma = sqrt(2), 128 integrator steps and the reflection
augmentation `--antithetic`, while the R-ASBS port uses a Haar source,
sigma = 1, 500 steps and no augmentation.  The reviewer's objection is that
the gap could be any of those rather than the supervision.

This script only *aggregates*; the runs it reads were all produced at one
matched configuration -- Haar source, sigma = 1, 500 steps, float64,
300,000 terminal oracle calls, 100,000 evaluation samples, 5 seeds -- with
the reflection augmentation turned on or off on BOTH sides (`--antithetic`
was added to the R-ASBS port for exactly this purpose; it is off by default
so no pre-existing number moves).

Every row is scored by `remeasure.sphere_metrics`, the same estimator that
produces the section 4.2/4.3/4.5 tables, so the ruler is shared.
"""

import json
import os
import re
import sys

import numpy as np
import torch


import _paths                                    # noqa: F401

import common as C                                               # noqa: E402
import remeasure as R                                            # noqa: E402

KEYS = ["north", "north_err", "KS_z", "W1_z", "KS_E", "KS_phi", "m2", "mE"]

# ckpt stem -> label, for the legs we trained
IASBS = [("sphere_w2_nd_anti", "IASBS Haar, aug"),
         ("sphere_w2_nd_plain", "IASBS Haar, no aug"),
         ("sphere_w2_nd_inner1", "IASBS Haar, aug, inner 1")]
# json stem -> label, for the R-ASBS legs (their port already scores itself
# with remeasure's estimator, so we read the numbers rather than the ckpt)
RASBS = [("results_rasbs_sphere_matlabinit_s%d", "R-ASBS, no aug"),
         ("results_w2_rasbs_anti_s%d", "R-ASBS, aug")]


def walls_from_log(path):
    if not os.path.exists(path):
        return []
    txt = open(path).read()
    return [float(x) for x in re.findall(r"\((\d+)s\)", txt)]


def agg(rows, k):
    v = np.array([r[k] for r in rows], dtype=float)
    return float(v.mean()), float(v.std())


def main():
    grid, cdf, _ = R.z_grid()
    m2e, m4e, mEe, _ = R.z_moments()
    out, table = {}, []

    for stem, label in IASBS:
        p0 = os.path.join("ckpt", f"{stem}_seed0.pt")
        if not os.path.exists(p0):
            print("skip", label, flush=True)
            continue
        rows, consts, n_used = R.load_variant(stem, grid, cdf, m2e, m4e, mEe)
        w = walls_from_log(f"iasbs/logs/{stem.replace('sphere_w2_nd_','w2_iasbs_nd_')}.log")
        rec = {k: agg(rows, k) for k in KEYS}
        rec["constraint"] = float(max(consts))
        rec["wall_s"] = float(np.mean(w)) if w else None
        rec["n_eval"] = int(n_used)
        rec["per_seed"] = rows
        out[label] = rec
        table.append((label, rec))

    for pat, label in RASBS:
        rows, walls = [], []
        for s in range(5):
            p = f"json/{pat % s}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            rows.append(d["final"])
            walls.append(d["history"][-1]["wall"])
        if not rows:
            print("skip", label, flush=True)
            continue
        rec = {k: agg(rows, k) for k in KEYS}
        rec["constraint"] = float(max(r["constraint"] for r in rows))
        rec["wall_s"] = float(np.mean(walls))
        rec["n_eval"] = 100000
        rec["per_seed"] = rows
        out[label] = rec
        table.append((label, rec))

    # finite-sample floor at the shared evaluation size
    gen = torch.Generator().manual_seed(0)
    fl = [R.sphere_metrics(R.sample_exact(100000, gen), grid, cdf,
                           m2e, m4e, mEe) for _ in range(5)]
    out["iid floor"] = {k: agg(fl, k) for k in KEYS}
    table.append(("iid floor (n=100,000)", out["iid floor"]))

    hdr = ("| leg | north_err | KS(x_3) | W1(x_3) | KS(E) | KS(phi) | "
           "<E> | wall/seed |")
    print(hdr)
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, r in table:
        w = r.get("wall_s")
        print(f"| {label} | {r['north_err'][0]:.5f} +- {r['north_err'][1]:.5f} "
              f"| {r['KS_z'][0]:.5f} +- {r['KS_z'][1]:.5f} "
              f"| {r['W1_z'][0]:.5f} +- {r['W1_z'][1]:.5f} "
              f"| {r['KS_E'][0]:.5f} +- {r['KS_E'][1]:.5f} "
              f"| {r['KS_phi'][0]:.5f} +- {r['KS_phi'][1]:.5f} "
              f"| {r['mE'][0]:.5f} +- {r['mE'][1]:.5f} "
              f"| {'—' if w is None else f'{w:.0f} s'} |")
    print(f"\nexact <E> = {mEe:.5f}")
    p = "json/results_weakness2_sphere.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    C.use_repo_root()
    main()
