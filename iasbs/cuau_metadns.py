"""Score the released MetaDNS CuAu samples on the fixed-composition benchmark.

MetaDNS samples the GRAND CANONICAL ensemble: single-site flips under
``H_eff = E(x) - h M(x)``, so the Au count is free.  Our benchmark is the
canonical fixed-composition sector.  The two are related exactly:

    p_grand(x | sum x = k)  =  e^{-E(x)/tau} / Z_k  =  pi(x),

because the ``h M`` term is constant on the sector and cancels.  So a perfect
grand-canonical sampler has conditional TV zero against our exact law, and the
comparison below is well posed rather than an ensemble mismatch.

What is reported:

* constraint violation rate -- the fraction of draws outside the sector.  The
  combinatorial floor for ANY unconstrained sampler with the right mean
  composition is ``1 - C(N,N/2)/2^N`` (0.804 at N=16, 0.901 at N=64), so a high
  rate here is a property of the ensemble, not a defect of their model.  It is
  reported because it is the cost our method does not pay.
* conditional TV against the exact law, on the draws that do land in the
  sector, with their own importance weights applied.
* the iid TV floor AT THE MATCHED EFFECTIVE SAMPLE COUNT.  This is the number
  that makes the comparison fair: a conditional TV estimated from a few
  thousand weighted draws cannot beat its own sampling floor, and quoting it
  against our 200k-sample number would be a rigged fight.

Their released NESS and free energy are carried through unchanged so the
method is also visible on its own terms.

CLI
---
    python -m iasbs.cuau_metadns --size 2 2 4 --run metadns --temp-key low
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch

from iasbs import cuau as CU


def load_samples(root, method, size, temp_key):
    sz = "x".join(str(int(v)) for v in size)
    path = os.path.join(root, "cuau", method, f"{sz}_{temp_key}", "samples.pkl")
    cfg_path = os.path.join(root, "cuau", method, f"{sz}_{temp_key}",
                            "config.json")
    with open(path, "rb") as f:
        d = pickle.load(f)
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    keys = list(d["configs"].keys())
    if len(keys) != 1:
        raise RuntimeError(f"expected one temperature key, got {keys}")
    k = keys[0]
    out = {
        "key": k, "path": path, "config": cfg,
        "configs": np.asarray(d["configs"][k]),
        "energies": np.asarray(d["energies"][k], dtype=np.float64),
        "log_rw": np.asarray(d["log_rw"][k], dtype=np.float64),
        "log_rnd": np.asarray(d["log_rnd"][k], dtype=np.float64),
        "ness": float(d["ness"][k]),
        "free_energy": float(d["free_energies"][k]),
    }
    return out


def weighted_ess(w):
    """Kish effective sample size of a set of non-negative weights."""
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (w * w).sum())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="external/metadns/checkpoints")
    p.add_argument("--method", default="metadns", choices=["metadns", "mdns"])
    p.add_argument("--size", type=int, nargs=3, default=[2, 2, 4])
    p.add_argument("--temp-key", default="low")
    p.add_argument("--temp", type=float, default=500.0)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--ref-json", default="",
                   help="PT reference JSON, used when the sector is too big "
                        "to enumerate")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    size = tuple(a.size)
    d = load_samples(a.root, a.method, size, a.temp_key)
    X = d["configs"]
    n = X.shape[1]
    k = n // 2
    if int(np.prod(size)) != n:
        raise RuntimeError(f"sample width {n} != prod(size) {np.prod(size)}")

    tables = CU.build_ce_tables(size, verbose=False)
    energy = CU.TorchCuAuEnergy(tables, device=a.device)
    Xt = torch.as_tensor(X, dtype=torch.int64, device=energy.device)

    # Cross-check their released energies against our CE on their own configs.
    # They do NOT match termwise, and should not: their stored quantity is the
    # grand-canonical effective energy ``E - h_eff M + c``.  The right test is
    # therefore whether the difference is AFFINE IN THE COMPOSITION, because
    # such a term is constant on the fixed-composition sector and cancels out
    # of the Boltzmann law there.  A residual beyond float32 noise would mean a
    # genuinely different Hamiltonian and would invalidate everything below.
    e_ours = energy.energy_torch(Xt).cpu().numpy()
    e_theirs = d["energies"]
    m_cnt = X.sum(1).astype(np.float64)
    A = np.stack([e_ours, m_cnt, np.ones_like(e_ours)], 1)
    coef, *_ = np.linalg.lstsq(A, e_theirs, rcond=None)
    resid = float(np.abs(A @ coef - e_theirs).max())
    scale = max(np.abs(e_ours).max(), 1e-12)
    energy_match = bool(abs(coef[0] - 1.0) < 1e-5 and resid / scale < 1e-5)

    cnt = X.sum(1)
    in_sector = cnt == k
    viol_rate = float(1.0 - in_sector.mean())

    w = np.exp(d["log_rw"] - d["log_rw"].max())
    ess_all = weighted_ess(w)
    ws = w[in_sector]
    ess_sec = weighted_ess(ws) if ws.size else 0.0
    # The reference for the violation rate is NOT a uniform-site combinatorial
    # count -- their target is a Boltzmann law that already concentrates on
    # near-equiatomic compositions, so a uniform-site number would be a
    # meaningless yardstick.  The right reference is the target's own sector
    # mass, estimated from their importance weights.
    p_sector = float(w[in_sector].sum() / w.sum()) if w.sum() > 0 else 0.0
    viol_rate_weighted = 1.0 - p_sector

    res = {
        "run": "cuau_metadns_scored",
        "method": a.method, "size": list(size), "N": n, "k": k,
        "temp_K": a.temp, "source": d["path"],
        "their_ness": d["ness"], "their_free_energy": d["free_energy"],
        "their_field_h": d["config"].get("field"),
        "their_cv_type": d["config"].get("cv_type"),
        "their_n_steps": d["config"].get("n_steps"),
        "n_samples": int(X.shape[0]),
        "energy_affine_fit_slope": float(coef[0]),
        "energy_affine_fit_h_eff_eV": float(-coef[1]),
        "energy_affine_fit_const_eV": float(coef[2]),
        "energy_affine_residual_rel": float(resid / scale),
        "energy_crosscheck_pass": energy_match,
        "violation_rate": viol_rate,
        "violation_rate_weighted": viol_rate_weighted,
        "target_sector_mass": p_sector,
        "rejection_overhead": (1.0 / p_sector) if p_sector > 0 else float("inf"),
        "n_in_sector": int(in_sector.sum()),
        "ess_all_weighted": ess_all,
        "ess_in_sector_weighted": ess_sec,
    }

    if not energy_match:
        print(f"!! energy cross-check FAILED: affine residual "
              f"{resid / scale:.3e}, slope {coef[0]:.6f} -- their target is "
              f"not our Hamiltonian and nothing below is comparable")

    Xs = Xt[torch.as_tensor(in_sector, device=energy.device)]
    wsec = torch.as_tensor(ws / max(ws.sum(), 1e-300),
                           dtype=torch.float64, device=energy.device)

    if Xs.shape[0]:
        Es = energy.energy_torch(Xs).to(torch.float64)
        q = CU.l10_q_vectors(float(energy.a))
        ph = np.asarray(energy.positions, dtype=np.float64) @ q.T
        cosp = torch.as_tensor(np.cos(ph), dtype=torch.float64,
                               device=energy.device)
        sinp = torch.as_tensor(np.sin(ph), dtype=torch.float64,
                               device=energy.device)
        Qs = CU.l10_order_parameters_torch(Xs, cosp, sinp).max(-1).values
        res["cond_E_per_N_meV"] = float((wsec * Es).sum()) / n * 1000.0
        res["cond_Qmax"] = float((wsec * Qs).sum())
        res["cond_E_per_N_meV_unweighted"] = float(Es.mean()) / n * 1000.0
        res["cond_Qmax_unweighted"] = float(Qs.mean())

    # Exact-law comparison where the sector is enumerable.
    if n <= 20:
        sp = CU.CuAuExact(size=size, temp_k=a.temp, gamma=a.gamma,
                          device=a.device, energy=energy, verbose=False)
        # map each in-sector config to its enumerated index
        pw = torch.tensor([1 << i for i in range(n)], dtype=torch.int64,
                          device=sp.device)
        codes = (sp.S * pw).sum(1)
        order = torch.argsort(codes)
        sc = codes[order]
        mine = (Xs * pw).sum(1)
        pos = torch.searchsorted(sc, mine)
        idx = order[pos.clamp(max=sc.numel() - 1)]
        if not bool((codes[idx] == mine).all()):
            raise RuntimeError("in-sector config not found in enumeration")
        emp = torch.zeros(sp.M, dtype=torch.float64, device=sp.device)
        emp.index_add_(0, idx, wsec)
        res["cond_TV_exact"] = 0.5 * float((emp - sp.pi).abs().sum())
        empu = torch.zeros(sp.M, dtype=torch.float64, device=sp.device)
        empu.index_add_(0, idx, torch.full_like(wsec, 1.0 / Xs.shape[0]))
        res["cond_TV_exact_unweighted"] = 0.5 * float((empu - sp.pi).abs().sum())
        # The floor that makes the comparison fair: what an ORACLE iid sampler
        # would score from the same number of effective draws.
        nfl = max(int(round(ess_sec)), 1)
        fl, fl_sd = sp.iid_tv_floor(nfl)
        res["iid_TV_floor_at_matched_ESS"] = fl
        res["iid_TV_floor_at_matched_ESS_sd"] = fl_sd
        res["matched_ESS_used"] = nfl
        rep = sp.exact_report(emp, "metadns")
        res["cond_energy_hist_TV"] = rep["energy_hist_TV"]
        res["exact_E_per_N_meV"] = rep.get("exact_mean_E_per_atom_meV")
        res["exact_Qmax"] = rep.get("exact_mean_Qmax")
    elif a.ref_json and os.path.exists(a.ref_json):
        ref = json.load(open(a.ref_json))
        res["reference"] = {kk: ref[kk] for kk in
                            ("E_per_N_meV", "E_per_N_meV_sd", "Qmax",
                             "Qmax_sd") if kk in ref}
        res["dE_per_N_meV"] = abs(res["cond_E_per_N_meV"]
                                  - ref["E_per_N_meV"])
        res["dQmax"] = abs(res["cond_Qmax"] - ref["Qmax"])

    print(f"\n[MetaDNS scored]  {a.method}  size={size}  N={n}  "
          f"T={a.temp} K  h={res['their_field_h']}  "
          f"CV={res['their_cv_type']}")
    print(f"  energy affine check     slope {res['energy_affine_fit_slope']:.6f}"
          f"  h_eff {res['energy_affine_fit_h_eff_eV']:.6f} eV"
          f"  resid {res['energy_affine_residual_rel']:.2e}  "
          f"{'PASS' if energy_match else 'FAIL'}")
    print(f"  samples                 {res['n_samples']:,}")
    print(f"  constraint violations   {viol_rate:.4f} raw   "
          f"{viol_rate_weighted:.4f} weighted")
    print(f"  target sector mass      {p_sector:.4f}   "
          f"rejection overhead {res['rejection_overhead']:.2f}x")
    print(f"  in sector               {res['n_in_sector']:,}  "
          f"weighted ESS {ess_sec:.1f}")
    print(f"  their NESS              {res['their_ness']:.4f}")
    if "cond_E_per_N_meV" in res:
        print(f"  conditional <E>/N       {res['cond_E_per_N_meV']:.4f} meV")
        print(f"  conditional <Qmax>      {res['cond_Qmax']:.4f}")
    if "cond_TV_exact" in res:
        print(f"  conditional TV (exact)  {res['cond_TV_exact']:.5f}   "
              f"unweighted {res['cond_TV_exact_unweighted']:.5f}")
        print(f"  iid TV floor @ ESS {res['matched_ESS_used']}   "
              f"{res['iid_TV_floor_at_matched_ESS']:.5f} +/- "
              f"{res['iid_TV_floor_at_matched_ESS_sd']:.5f}")
        print(f"  exact <E>/N {res['exact_E_per_N_meV']:.4f} meV   "
              f"exact <Qmax> {res['exact_Qmax']:.4f}")
    if "dE_per_N_meV" in res:
        print(f"  vs PT reference         dE {res['dE_per_N_meV']:.4f} meV   "
              f"dQ {res['dQmax']:.4f}")

    out = a.out or (f"json/results_cuau_{a.method}_"
                    f"{'x'.join(str(v) for v in size)}_{int(a.temp)}K.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
