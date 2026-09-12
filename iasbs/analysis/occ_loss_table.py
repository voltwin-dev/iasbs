"""Bregman vs MSE occupation loss, side by side.

Reads the `results_occ_*` artifacts written by `occupation.py` and prints the
three sweep rows (m = 4 enumerated, m = 128 and m = 1000 tau-leap) with the
shipped Bregman objective and the bounded-below MSE objective, matched on every
other setting.  Writes `json/results_occ_loss_table.json`.
"""
import json
import os

import numpy as np

import _paths                                    # noqa: F401

J = os.path.join(_paths.ROOT, "json")
OUT = os.path.join(J, "results_occ_loss_table.json")

GROUPS = [
    ("m=128", "bregman", ["results_occ_s128.json", "results_occ_s128_s1.json",
                          "results_occ_s128_s2.json"]),
    ("m=128", "mse", ["results_occ_s128_mse.json",
                      "results_occ_s128_mse_s1.json",
                      "results_occ_s128_mse_s2.json"]),
    ("m=1000", "bregman", ["results_occ_s1000.json",
                           "results_occ_s1000_s1.json",
                           "results_occ_s1000_s2.json"]),
    ("m=1000", "mse", ["results_occ_s1000_mse.json",
                       "results_occ_s1000_mse_s1.json",
                       "results_occ_s1000_mse_s2.json"]),
]
KEYS = ["KS_occ", "KS_max", "W1_max_frac", "dE"]


def load(fn):
    p = os.path.join(J, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    h = d["history"][-1]
    h = dict(h)
    h["dE"] = abs(h["E_ours"] - h["E_exact"])
    h["seed"] = d["config"]["seed"]
    h["loss"] = d["config"]["loss"]
    h["viol"] = h.get("violations", 0)
    return h


def main():
    res = {"rows": [], "m4": {}}

    # m = 4: exact full law, single seed each
    for loss, fn in (("bregman", "results_occ_full.json"),
                     ("mse", "results_occ_mse.json")):
        d = json.load(open(os.path.join(J, fn)))
        res["m4"][loss] = dict(TV=d["final_TV"], violations=d["violations"],
                               file=fn)
    print("m = 4 (enumerated, exact-law TV, 20,000 samples)")
    for loss in ("bregman", "mse"):
        r = res["m4"][loss]
        print(f"  {loss:8s} TV {r['TV']:.5f}   violations {r['violations']}")
    print("  iid floor 0.0167,  gate B1 TV <= 0.05")

    print("\ntau-leap sweeps, mean +- sd over 3 seeds")
    hdr = (f"{'m':>6} {'loss':>8} {'seeds':>6} | " +
           " | ".join(f"{k:>17}" for k in KEYS) + " | viol")
    print(hdr)
    for m, loss, files in GROUPS:
        hs = [h for h in (load(f) for f in files) if h is not None]
        if not hs:
            continue
        row = dict(m=m, loss=loss, n_seeds=len(hs),
                   seeds=[h["seed"] for h in hs],
                   violations=int(sum(h["viol"] for h in hs)))
        cells = []
        for k in KEYS:
            v = np.array([h[k] for h in hs], dtype=float)
            row[k] = dict(mean=float(v.mean()),
                          sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          per_seed=[float(x) for x in v])
            cells.append(f"{v.mean():8.4f} +- {v.std(ddof=1) if len(v)>1 else 0:6.4f}")
        res["rows"].append(row)
        print(f"{m:>6} {loss:>8} {len(hs):>6} | " + " | ".join(cells)
              + f" | {row['violations']}")

    print("\nper-seed detail")
    for r in res["rows"]:
        for k in KEYS:
            print(f"  {r['m']:>6} {r['loss']:>8} {k:>12} "
                  + "  ".join(f"{x:9.4f}" for x in r[k]["per_seed"]))

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
