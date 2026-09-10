"""Re-measure the Ising and sphere experiments from their saved checkpoints.

Both experiments print only a couple of headline numbers while they train, but
their checkpoints carry enough to answer far more than that without retraining:

  * `ckpt/ising_*.pt` stores `exact_law` -- the law of the *discretised
    controlled chain*, propagated exactly by enumeration rather than estimated
    from samples -- together with the target `pi` and the full state table.  So
    every divergence between the sampler and the truth is available in closed
    form, at machine precision, with no Monte-Carlo error at all.

  * `ckpt/sphere_*.pt` stores 200,000 samples.  The S^2 target
    `pi ∝ exp(6 x_3^2)` has an exactly integrable z-marginal, so moments,
    Wasserstein-1 and azimuthal uniformity can all be checked against closed
    form, and the finite-sample floor can be measured by drawing exact iid
    samples of the same size.

This script computes those metrics and emits the two markdown tables that go
into the README.  It runs on CPU on purpose: the GPUs may be busy with a live
training run, and nothing here needs one.

    python remeasure.py ising
    python remeasure.py sphere
    python remeasure.py all
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

# common.py and the shared json/ ckpt/ fig/ directories live at the
# repository root, one level up from this script.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C

DEV = "cpu"
DT = torch.float64


# ============================================================================
# generic divergences between two probability vectors
# ============================================================================
def divergences(p, q):
    """p = sampler law, q = target.  Both (M,), normalised, positive."""
    p = p.to(DT).clamp_min(0.0)
    q = q.to(DT).clamp_min(0.0)
    p = p / p.sum()
    q = q / q.sum()
    eps = 1e-300
    tv = 0.5 * float((p - q).abs().sum())
    kl_pq = float((p * (torch.log(p + eps) - torch.log(q + eps))).sum())
    kl_qp = float((q * (torch.log(q + eps) - torch.log(p + eps))).sum())
    hell = float(torch.sqrt(0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum()))
    chi2 = float((((p - q) ** 2) / (q + eps)).sum())
    # Renyi-2 of p against q, and the effective sample size an importance
    # reweighting from p to q would retain: ESS/n = 1 / (1 + chi2).
    renyi2 = math.log1p(chi2)
    ess = 1.0 / (1.0 + chi2)
    ratio = float(((p / (q + eps)) - 1.0).abs().max())
    return {"TV": tv, "KL(p||pi)": kl_pq, "KL(pi||p)": kl_qp,
            "Hellinger": hell, "chi2": chi2, "Renyi-2": renyi2,
            "ESS_frac": ess, "max|p/pi-1|": ratio}


# ============================================================================
# Ising
# ============================================================================
def ising_report(tag):
    path = os.path.join("ckpt", f"{tag}.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    ex = d["extra"]
    cfg = ex["config"]
    L, J, tau = int(cfg["L"]), float(cfg["J"]), float(cfg["tau"])

    p = ex["exact_law"].to(DT)
    pi = ex["pi"].to(DT)
    S = ex["states"]
    idx = d["samples"].long()
    M = pi.numel()

    edges = C.periodic_lattice_edges(L)
    s = 2.0 * S.to(DT) - 1.0
    E = torch.zeros(M, dtype=DT)
    for i, j in edges:
        E = E + s[:, i] * s[:, j]
    E = -J * E

    out = divergences(p, pi)

    # energy-resolved: the law of E under the sampler vs under the target
    lv, inv = torch.unique(E, return_inverse=True)
    hp = torch.zeros(len(lv), dtype=DT).index_add(0, inv, p)
    hq = torch.zeros(len(lv), dtype=DT).index_add(0, inv, pi)
    out["E-hist TV"] = 0.5 * float((hp - hq).abs().sum())

    mE_p, mE_q = float((p * E).sum()), float((pi * E).sum())
    v_p = float((p * E * E).sum()) - mE_p ** 2
    v_q = float((pi * E * E).sum()) - mE_q ** 2
    out["<E> ours"], out["<E> exact"] = mE_p, mE_q
    out["C ours"], out["C exact"] = v_p / tau ** 2, v_q / tau ** 2

    n_edges = len(edges)
    out["<s_i s_j> ours"] = -mE_p / (J * n_edges)
    out["<s_i s_j> exact"] = -mE_q / (J * n_edges)

    # exact free energy of the constrained ensemble (a property of the target,
    # not of the sampler -- it is what the sampler is implicitly integrating)
    logZ = float(torch.logsumexp(-E / tau, dim=0))
    out["F exact"] = -tau * logZ

    # the 20,000 stored samples, and the iid floor at that sample size
    n = idx.numel()
    emp = torch.bincount(idx, minlength=M).to(DT)
    emp = emp / emp.sum()
    out["TV empirical"] = 0.5 * float((emp - pi).abs().sum())
    g = torch.Generator().manual_seed(0)
    floors = []
    for _ in range(20):
        draw = torch.multinomial(pi.to(torch.float32), n, replacement=True,
                                 generator=g)
        e2 = torch.bincount(draw, minlength=M).to(DT)
        floors.append(0.5 * float((e2 / e2.sum() - pi).abs().sum()))
    out["TV iid floor"] = float(np.mean(floors))
    out["TV iid floor sd"] = float(np.std(floors))
    out["n_samples"] = n
    out["violations"] = ex["violations"]
    out["M"] = M
    out["mult_err"] = ex["mult_err"]
    return out


def print_ising(tag="ising_poisson256", exact_json="results_ising_exact.json",
                variants=("ising_mse", "ising_poisson", "ising_poisson256")):
    import json
    m = ising_report(tag)

    # the exact-control run is a separate artifact: same algorithm, but with the
    # multiplier taken from enumeration instead of learned.  It is the ceiling
    # the learned controller is trying to reach, and it must be labelled as
    # such rather than folded into the learned column.
    with open(os.path.join("json", exact_json)) as f:
        ex = json.load(f)
    print("\n### Ising, exact control vs integration steps\n")
    print("| steps | TV vs pi | E-hist TV | <E> | mass leak |")
    print("|---:|---:|---:|---:|---:|")
    for r in ex["rows"]:
        print(f"| {r['steps']} | {r['TV']:.5f} | {r['energy_hist_TV']:.5f} | "
              f"{r['mean_energy']:.4f} | {r['mass_leak']:.1e} |")
    print(f"\nexact <E> = {ex['rows'][0]['exact_mean_energy']:.4f}.  "
          f"Sampling {ex['config']['n_samples']:,} points from the "
          f"{ex['config']['steps_sweep'][-1]}-step exact control gives "
          f"empirical TV {ex['empirical_TV']:.4f} against an iid floor of "
          f"{ex['iid_TV_floor']:.4f}, with {ex['violations']} constraint "
          f"violations.")

    print(f"\n### Ising ({tag}, |Omega| = {m['M']})\n")
    print("| metric | exact | ours |")
    print("|---|---:|---:|")
    print(f"| TV of the exact chain law vs pi | 0 | **{m['TV']:.4f}** |")
    print(f"| KL(ours ‖ pi) | 0 | **{m['KL(p||pi)']:.4f}** |")
    print(f"| KL(pi ‖ ours) | 0 | **{m['KL(pi||p)']:.4f}** |")
    print(f"| Hellinger distance | 0 | **{m['Hellinger']:.4f}** |")
    print(f"| chi^2(ours ‖ pi) | 0 | **{m['chi2']:.4f}** |")
    print(f"| Renyi-2 divergence (nats) | 0 | **{m['Renyi-2']:.4f}** |")
    print(f"| importance-reweighting ESS fraction | 1 | "
          f"**{m['ESS_frac']:.4f}** |")
    print(f"| max_x |p(x)/pi(x) - 1| | 0 | **{m['max|p/pi-1|']:.4f}** |")
    print(f"| TV of the energy histogram | 0 | **{m['E-hist TV']:.4f}** |")
    print(f"| mean energy <E> | {m['<E> exact']:.4f} | "
          f"**{m['<E> ours']:.4f}** |")
    print(f"| heat capacity Var(E)/tau^2 | {m['C exact']:.4f} | "
          f"**{m['C ours']:.4f}** |")
    print(f"| nearest-neighbour correlation <s_i s_j> | "
          f"{m['<s_i s_j> exact']:.4f} | **{m['<s_i s_j> ours']:.4f}** |")
    print(f"| free energy -tau log Z | {m['F exact']:.4f} | (target property) |")
    print(f"| TV of {m['n_samples']:,} drawn samples vs pi | "
          f"{m['TV iid floor']:.4f} +- {m['TV iid floor sd']:.4f} (iid floor) | "
          f"**{m['TV empirical']:.4f}** |")
    print(f"| constraint violations in {m['n_samples']:,} samples | 0 | "
          f"**{m['violations']}** |")
    print("\n### Ising, learned controller: loss and step count\n")
    print("| loss | steps | iters | TV | KL(ours ‖ pi) | Hellinger "
          "| E-hist TV | <E> |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in variants:
        if not os.path.exists(os.path.join("ckpt", f"{t}.pt")):
            continue
        r = ising_report(t)
        c = torch.load(os.path.join("ckpt", f"{t}.pt"), map_location="cpu",
                       weights_only=False)["extra"]["config"]
        print(f"| {c['loss']} | {c['steps']} | {c['iters']} | {r['TV']:.4f} | "
              f"{r['KL(p||pi)']:.4f} | {r['Hellinger']:.4f} | "
              f"{r['E-hist TV']:.4f} | {r['<E> ours']:.4f} |")
    print(f"| exact | - | - | 0 | 0 | 0 | 0 | {m['<E> exact']:.4f} |")

    print("\nlearned vs exact log-multiplier, over all states x edges:\n")
    print("| t | mean abs error | max abs error |")
    print("|---:|---:|---:|")
    for t, me, mx in m["mult_err"]:
        print(f"| {float(t):.3f} | {float(me):.4f} | {float(mx):.4f} |")
    return m


# ============================================================================
# sphere
# ============================================================================
def z_grid(num=200001):
    """Exact z-marginal of pi ∝ exp(6 x_3^2) on S^2, on a fine grid."""
    z = np.linspace(-1.0, 1.0, num)
    w = np.exp(6.0 * (z ** 2 - 1.0))            # shifted for stability
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(z))])
    cdf /= cdf[-1]
    return z, cdf, w / np.trapezoid(w, z)


def z_moments(num=200001):
    z, _, dens = z_grid(num)
    m2 = float(np.trapezoid(dens * z ** 2, z))
    m4 = float(np.trapezoid(dens * z ** 4, z))
    mE = 6.0 * (1.0 - m2)
    vE = 36.0 * (m4 - m2 ** 2)
    return m2, m4, mE, vE


def sample_exact(n, gen, num=200001):
    z, cdf, _ = z_grid(num)
    u = torch.rand(n, generator=gen, dtype=torch.float64).numpy()
    zz = np.interp(u, cdf, z)
    phi = torch.rand(n, generator=gen, dtype=torch.float64).numpy() * 2 * np.pi
    r = np.sqrt(np.clip(1.0 - zz ** 2, 0.0, None))
    x = np.stack([r * np.cos(phi), r * np.sin(phi), zz], 1)
    return torch.tensor(x, dtype=DT)


def w1_against_cdf(samp, grid, cdf):
    """Wasserstein-1 between an empirical law and a grid CDF, on the line."""
    s = np.sort(np.asarray(samp, dtype=np.float64))
    f_emp = np.searchsorted(s, grid, side="right") / len(s)
    return float(np.trapezoid(np.abs(f_emp - cdf), grid))


def sphere_metrics(x, grid, cdf, m2e, m4e, mEe):
    z = x[:, 2].double().numpy()
    phi = np.arctan2(x[:, 1].double().numpy(), x[:, 0].double().numpy())
    out = {}
    out["north"] = float((z > 0).mean())
    out["north_err"] = abs(out["north"] - 0.5)
    out["KS_z"] = C.ks_against_grid_cdf(z, grid, cdf)
    out["W1_z"] = w1_against_cdf(z, grid, cdf)
    out["m2"] = float((z ** 2).mean())
    out["m4"] = float((z ** 4).mean())
    out["mE"] = 6.0 * (1.0 - out["m2"])
    out["d_m2"] = out["m2"] - m2e
    out["d_m4"] = out["m4"] - m4e
    out["d_mE"] = out["mE"] - mEe
    # E = 6(1 - z^2) is a monotone function of |z|, so KS(E) = KS(|z|)
    a = np.abs(z)
    gh = grid[grid >= 0.0]
    ch = np.interp(gh, grid, cdf)
    cdf_abs = ch - np.interp(-gh, grid, cdf)
    out["KS_E"] = C.ks_against_grid_cdf(a, gh, cdf_abs)
    # azimuth must be exactly uniform on [-pi, pi]
    u = (phi + np.pi) / (2 * np.pi)
    gu = np.linspace(0.0, 1.0, 20001)
    out["KS_phi"] = C.ks_against_grid_cdf(u, gu, gu)
    return out


def load_variant(prefix, grid, cdf, m2e, m4e, mEe):
    rows, consts, n_used = [], [], None
    for s in range(5):
        d = torch.load(os.path.join("ckpt", f"{prefix}_seed{s}.pt"),
                       map_location="cpu", weights_only=False)
        x = d["samples"]
        n_used = x.shape[0]
        rows.append(sphere_metrics(x, grid, cdf, m2e, m4e, mEe))
        consts.append(d["extra"]["metrics"]["constraint"])
    return rows, consts, n_used


def print_sphere():
    grid, cdf, _ = z_grid()
    m2e, m4e, mEe, vEe = z_moments()

    # the headline configuration in the README is the antithetic one
    rows, consts, n_used = load_variant("sphere_anti", grid, cdf,
                                        m2e, m4e, mEe)

    gen = torch.Generator().manual_seed(0)
    floors = [sphere_metrics(sample_exact(n_used, gen), grid, cdf,
                             m2e, m4e, mEe) for _ in range(5)]

    def agg(rs, k):
        v = np.array([r[k] for r in rs])
        return v.mean(), v.std()

    def cell(rs, k, fmt="{:.4f}"):
        m, s = agg(rs, k)
        return f"{fmt.format(m)} ± {fmt.format(s)}"

    print(f"\n### Sphere (5 seeds x {n_used:,} samples)\n")
    print("| metric | exact | ours | iid floor at the same n |")
    print("|---|---:|---:|---:|")
    print(f"| north mass | 0.5 | **{cell(rows,'north')}** | "
          f"{cell(floors,'north')} |")
    print(f"| absolute north error | 0 | **{cell(rows,'north_err')}** | "
          f"{cell(floors,'north_err')} |")
    print(f"| KS(x_3) | 0 | **{cell(rows,'KS_z')}** | "
          f"{cell(floors,'KS_z')} |")
    print(f"| Wasserstein-1 on x_3 | 0 | **{cell(rows,'W1_z','{:.5f}')}** | "
          f"{cell(floors,'W1_z','{:.5f}')} |")
    print(f"| KS(E) | 0 | **{cell(rows,'KS_E')}** | "
          f"{cell(floors,'KS_E')} |")
    print(f"| KS(azimuth) | 0 | **{cell(rows,'KS_phi')}** | "
          f"{cell(floors,'KS_phi')} |")
    print(f"| <x_3^2> | {m2e:.5f} | **{cell(rows,'m2','{:.5f}')}** | "
          f"{cell(floors,'m2','{:.5f}')} |")
    print(f"| <x_3^4> | {m4e:.5f} | **{cell(rows,'m4','{:.5f}')}** | "
          f"{cell(floors,'m4','{:.5f}')} |")
    print(f"| mean energy <E> | {mEe:.5f} | **{cell(rows,'mE','{:.5f}')}** | "
          f"{cell(floors,'mE','{:.5f}')} |")
    print(f"| max norm residual | 0 | **{max(consts):.1e}** | - |")

    # the symmetry ablation: the target is invariant under x_3 -> -x_3, and
    # how that invariance is handled is what decides whether the two poles
    # share the mass evenly.
    print("\n### Sphere, symmetry handling\n")
    print("| variant | north mass | abs north error | KS(x_3) | W1(x_3) "
          "| <E> |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, pre in [("plain", "sphere_plain"), ("antithetic", "sphere_anti"),
                      ("symmetrised", "sphere_sym")]:
        rs, _, _ = load_variant(pre, grid, cdf, m2e, m4e, mEe)
        print(f"| {name} | {cell(rs,'north')} | {cell(rs,'north_err')} | "
              f"{cell(rs,'KS_z')} | {cell(rs,'W1_z','{:.5f}')} | "
              f"{cell(rs,'mE','{:.5f}')} |")
    print(f"| iid at n = {n_used:,} | {cell(floors,'north')} | "
          f"{cell(floors,'north_err')} | {cell(floors,'KS_z')} | "
          f"{cell(floors,'W1_z','{:.5f}')} | {cell(floors,'mE','{:.5f}')} |")
    print(f"| exact | 0.5 | 0 | 0 | 0 | {mEe:.5f} |")

    # step sweep of the exact control, which has no learning in it at all
    print("\n### Sphere, exact control vs integration steps\n")
    print("| steps | north mass | KS(x_3) | W1(x_3) | <E> |")
    print("|---:|---:|---:|---:|---:|")
    for s in [32, 64, 128, 256, 512]:
        p = os.path.join("ckpt", f"sphere_exact_steps{s}.pt")
        if not os.path.exists(p):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = sphere_metrics(d["samples"], grid, cdf, m2e, m4e, mEe)
        print(f"| {s} | {m['north']:.4f} | {m['KS_z']:.4f} | "
              f"{m['W1_z']:.5f} | {m['mE']:.5f} |")
    f = floors[0]
    print(f"| iid | {f['north']:.4f} | {f['KS_z']:.4f} | {f['W1_z']:.5f} | "
          f"{f['mE']:.5f} |")
    print(f"\nexact <E> = {mEe:.5f}, exact <x_3^2> = {m2e:.5f}")


def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ising", "ising5", "sphere", "all"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd in ("ising", "all"):
        print_ising(a.tag or "ising_poisson256")
    if a.cmd in ("ising5", "all"):
        # L = 5: same code path, 5,200,300 states instead of 12,870.  The two
        # runs differ only in the integration grid (256 vs 512 steps), so the
        # "loss and step count" table degenerates to a step-count table.
        print_ising(a.tag or "ising_t1_L5_s512",
                    exact_json="results_ising_t1_L5_exact.json",
                    variants=("ising_t1_L5", "ising_t1_L5_s512"))
    if a.cmd in ("sphere", "all"):
        print_sphere()
    return 0


if __name__ == "__main__":
    sys.exit(main())
