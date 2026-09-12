"""Well-posedness of the occupation regression loss.

The shipped objective is the Bregman form

    l(a, y) = eta_i * exp(a) - a * y,        y = eta_i + Lambda_ji ,

whose stationary point is eta_i * exp(a*) = E[y | X_t].  When that conditional
mean is negative the objective has no minimiser: a -> -inf sends it to -inf.
The MSE form

    l(a, y) = (eta_i * exp(a) - y)^2

has the *same* stationary point, is bounded below by 0, and when E[y|X_t] < 0
its infimum E[y|X_t]^2 is approached but never attained -- still well defined.

This script does three things and writes `json/results_occ_loss_wellposed.json`:

1.  The m = 3, N = 1 case in closed form.  With N = 1 the state is a pure mode
    e_k, the only active edges are (k -> j), and the exact full-sum label gives

        y = 1 + (R_kj - 1) - c_t (R_kj - R_kk) = (1 - c_t) R_kj + c_t R_kk ,

    which is a convex combination of two nonnegative rate ratios, hence >= 0
    for every c_t in [0, 1].  The negative-coefficient counterexample therefore
    cannot be realised at m = 3, N = 1 with the exact label; it needs either a
    label estimator that can undershoot or a y supplied from outside.  Both
    losses are then evaluated on an *externally imposed* negative y to show the
    divergence and the bound.

2.  The empirical range of y over the enumerated grid (m = 3, 4) and over
    tau-leap draws at m = 32, 128, 1000, uncontrolled and (where a checkpoint
    exists) controlled, counting how often y < 0 actually occurs.

3.  Nothing is trained here; `--loss mse` already exists in `occupation.py`.
"""
import json
import os

import numpy as np
import torch

import _paths                                    # noqa: F401

import common as C
from occupation import (OccupationSpace, ScaleOccupation, ScaleController,
                        c_of_t, bridge_scale, simulate_scale)

DT = torch.float64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(_paths.ROOT, "json", "results_occ_loss_wellposed.json")


def enumerated_y(m, N, d=0.5, gamma=4.0, steps=128):
    sp = OccupationSpace(m=m, N=N, d=d, tau=1.0, gamma=gamma, device="cpu")
    cts = [c_of_t(sp, t) for t in np.arange(steps) / steps]
    lam = torch.stack([sp.labels_full(float(c)) for c in cts])
    y = sp.eocc.to(DT).unsqueeze(0) + lam
    ym = y[:, sp.emask]
    return sp, ym


def enumerated_y_bridge(m, N, d=0.5, gamma=4.0, steps=128, mb=200000,
                        pool="pi", seed=0):
    """y exactly as `run_train` builds it: occ from X_t, Lambda from X_1.

    The grid scan above pairs both at the same state, which is the diagonal
    X_1 = X_t and not what training sees.  This draws X_1 from the pool, X_t
    from the reference bridge, and reads Lambda at X_1 -- the same two lines as
    the trainer, `lam = lam_tab[ti, X1]` and `occ = sp.eocc[Xt]`.
    """
    from occupation import sample_bridge

    sp = OccupationSpace(m=m, N=N, d=d, tau=1.0, gamma=gamma, device="cpu")
    ts = np.arange(steps) / steps
    K0 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * t)[sp.i0] for t in ts]), 1e-300)))
    K1 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * (1 - t)) for t in ts]), 1e-300)))
    cts = [c_of_t(sp, float(t)) for t in ts]
    lam_tab = torch.stack([sp.labels_full(float(c)) for c in cts])
    w = sp.pi.to(DT) if pool == "pi" else torch.full((sp.M,), 1.0 / sp.M,
                                                     dtype=DT)
    g = torch.Generator().manual_seed(seed)
    out = []
    got = 0
    while got < mb:
        b = min(65536, mb - got)
        X1 = torch.multinomial(w, b, replacement=True, generator=g)
        ti = torch.randint(steps, (b,), generator=g)
        Xt = sample_bridge(sp, X1, ti, K0, K1, generator=g)
        y = sp.eocc[Xt].to(DT) + lam_tab[ti, X1]
        out.append(y[sp.emask[Xt]])
        got += b
    return torch.cat(out)


def scale_y(m, N, ckpt=None, mb=200000, d=0.5, gamma=4.0, steps=128,
            batch=4096, seed=0):
    """Replicate the label pipeline of run_scale and return y."""
    torch.manual_seed(seed)
    sp = ScaleOccupation(m=m, N=N, d=d, tau=1.0, gamma=gamma)
    q = sp.q1()
    net = None
    if ckpt is not None and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=sp.device, weights_only=False)
        net = ScaleController(sp.m, sp.N,
                              hidden=st.get("hidden", 256)).to(sp.device)
        net.load_state_dict(st["state_dict"])
        net.eval()
    with torch.no_grad():
        X1 = simulate_scale(sp, net, batch, steps)
        got, chunks = 0, []
        while got < mb:
            b = min(65536, mb - got)
            sel = torch.randint(len(X1), (b,), device=sp.device)
            x1 = X1[sel]
            tv = torch.rand(b, device=sp.device, dtype=DT)
            xt = bridge_scale(sp, x1, tv)
            ct = torch.tensor([C.occupation_c_t(sp.m, sp.gamma * (1 - float(z)))
                               for z in tv.cpu()], device=sp.device)
            i_idx = torch.multinomial(xt / sp.N, 1)[:, 0]
            j_idx = torch.randint(sp.m, (b,), device=sp.device)
            lam = sp.labels_full(x1, q, ct, i_idx, j_idx)
            ar = torch.arange(b, device=sp.device)
            occ = xt[ar, i_idx]
            chunks.append((occ + lam).cpu())
            got += b
    return torch.cat(chunks)


def population_coefficient(m=4, N=4, d=0.5, gamma=4.0, steps=128):
    """Exact E[y | X_t] on the m=4 grid, and the exact control it must equal.

    Training draws X_1 from the pool, then X_t from the reference bridge, and
    regresses on y = eta_i(X_t) + Lambda_t(X_1).  Conditionally on X_t the only
    random object left is X_1, so

        E[y | X_t = z] = eta_i(z) + sum_xi p(xi | z, t) Lambda_t(xi),

    which is enumerable at m = 4.  Under the exact control the pool is pi and
    the population coefficient must equal eta_i(z) exp(a*(t, z)), i.e. the Doob
    intensity ratio -- strictly positive.  That is what makes the *population*
    Bregman objective convex with a finite minimiser; the negative coefficients
    measured above are per-sample Monte-Carlo noise around a positive mean.
    """
    from occupation import ExactControl

    sp = OccupationSpace(m=m, N=N, d=d, tau=1.0, gamma=gamma, device="cpu")
    ts = np.arange(steps) / steps
    K0 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * t)[sp.i0] for t in ts]), 1e-300)))
    K1 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * (1 - t)) for t in ts]), 1e-300)))
    cts = [c_of_t(sp, float(t)) for t in ts]
    lam = torch.stack([sp.labels_full(float(c)) for c in cts])   # (T, M, E)
    ec = ExactControl(sp, steps)

    out = {}
    for pool_name, pw in (("pi (on-policy)", sp.pi.to(DT)),
                          ("uniform (off-policy)",
                           torch.full((sp.M,), 1.0 / sp.M, dtype=DT))):
        worst, worst_rel = float("inf"), 0.0
        for s in range(0, steps, 4):
            # p(z | xi) propto exp(K0[s][z] + K1[s][z, xi]), normalised over z
            lw = K0[s][:, None] + K1[s]                          # (M_z, M_xi)
            lw = lw - lw.max(dim=0, keepdim=True).values
            pz_xi = torch.exp(lw)
            pz_xi = pz_xi / pz_xi.sum(dim=0, keepdim=True)
            joint = pz_xi * pw[None, :]                          # (z, xi)
            tot = joint.sum(dim=1, keepdim=True).clamp_min(1e-300)
            pxi_z = joint / tot                                  # p(xi | z)
            Ey = sp.eocc.to(DT) + pxi_z @ lam[s]                 # (M, E)
            act = sp.emask
            worst = min(worst, float(Ey[act].min()))
            if pool_name.startswith("pi"):
                a_ex = ec.all_states(float(ts[s])).to(DT)
                pred = sp.eocc.to(DT) * torch.exp(a_ex)
                rel = ((pred - Ey).abs() / Ey.abs().clamp_min(1e-12))[act]
                worst_rel = max(worst_rel, float(rel.max()))
        out[pool_name] = dict(min_cond_mean=worst,
                              max_rel_err_vs_exact_control=worst_rel or None)
        print(f"  pool {pool_name:22s} min E[y|X_t] = {worst:+.6f}"
              + (f"   max rel err vs eta_i e^(a*) = {worst_rel:.2e}"
                 if worst_rel else ""))
    return out


def stats(y):
    y = y.reshape(-1).to(DT)
    return dict(n=int(y.numel()), min=float(y.min()), max=float(y.max()),
                mean=float(y.mean()), n_neg=int((y < 0).sum()),
                frac_neg=float((y < 0).to(DT).mean()))


def main():
    res = {"loss_curves": [], "y_ranges": []}

    # ---- 1. m = 3, N = 1 -------------------------------------------------
    sp, ym = enumerated_y(3, 1)
    s = stats(ym)
    s.update(case="m=3 N=1 enumerated, exact full label")
    res["y_ranges"].append(s)
    print(f"m=3 N=1  y in [{s['min']:+.6f}, {s['max']:+.6f}]  "
          f"neg {s['n_neg']}/{s['n']}")
    print("  closed form: y = (1 - c_t) R_kj + c_t R_kk >= 0 for c_t in [0,1]")

    # verify the closed form against labels_full on every (t, k, j)
    steps = 128
    err = 0.0
    for t in np.arange(0, steps, 8) / steps:
        c = c_of_t(sp, float(t))
        lab = sp.labels_full(float(c))
        for e in range(sp.n_edges):
            i, j = int(sp.edge_i[e]), int(sp.edge_j[e])
            for r in range(sp.M):
                if not bool(sp.emask[r, e]):
                    continue
                k = int(torch.argmax(sp.Sf[r]))
                pred = ((1.0 - c) * float(sp.R[r, k, j])
                        + c * float(sp.R[r, k, k]))
                got = float(sp.eocc[r, e]) + float(lab[r, e])
                err = max(err, abs(pred - got))
    res["m3N1_closed_form_max_abs_err"] = err
    print(f"  closed form vs labels_full: max abs err {err:.3e}")

    # ---- 2. both losses at an externally imposed negative y --------------
    for yv in (-0.5, -2.0):
        rows = []
        for a in (0.0, -5.0, -10.0, -15.0, -20.0, -50.0, -1e3, -1e4):
            rows.append(dict(a=a, bregman=1.0 * np.exp(a) - a * yv,
                             mse=(1.0 * np.exp(a) - yv) ** 2))
        res["loss_curves"].append(dict(eta_i=1.0, y=yv, rows=rows,
                                       mse_inf=yv ** 2))
        print(f"\n  eta_i = 1, y = {yv:+.1f}")
        print("       a        bregman            mse")
        for r in rows:
            print(f"  {r['a']:10.1f}  {r['bregman']:+16.4f}  {r['mse']:14.6f}")
        print(f"  bregman -> -inf (slope {-yv:+.2f}/unit a); "
              f"mse -> {yv**2:.4f}, bounded below by 0")

    # ---- 3. empirical y ranges ------------------------------------------
    sp4, y4 = enumerated_y(4, 4)
    s = stats(y4)
    s.update(case="m=4 N=4 enumerated, exact full label")
    res["y_ranges"].append(s)
    print(f"\nm=4 N=4  y in [{s['min']:+.4f}, {s['max']:+.4f}]  "
          f"neg {s['n_neg']}/{s['n']}")

    # same-state pairing above is the diagonal; this is the trainer's pairing
    for mm, NN in ((3, 1), (4, 4)):
        for pool in ("pi", "uniform"):
            yb = enumerated_y_bridge(mm, NN, pool=pool)
            s = stats(yb)
            s.update(case=f"m={mm} N={NN} enumerated, bridge pairing, "
                          f"pool={pool}")
            res["y_ranges"].append(s)
            print(f"m={mm} N={NN} bridge pairing, pool {pool:7s} "
                  f"y in [{s['min']:+.4f}, {s['max']:+.4f}]  "
                  f"neg {s['n_neg']}/{s['n']}")

    for m in (32, 128, 1000):
        steps = 256 if m == 1000 else 128
        for tag, ck in (("uncontrolled", None),
                        ("bregman ckpt",
                         os.path.join(_paths.ROOT, "ckpt", f"occ_s{m}.pt")),
                        ("mse ckpt",
                         os.path.join(_paths.ROOT, "ckpt",
                                      f"occ_s{m}_mse.pt"))):
            if ck is not None and not os.path.exists(ck):
                continue
            try:
                y = scale_y(m, m, ckpt=ck, steps=steps,
                            batch=2048 if m == 1000 else 4096)
            except Exception as exc:                      # noqa: BLE001
                print(f"m={m} {tag}: skipped ({exc})")
                continue
            s = stats(y)
            s.update(case=f"m={m} N={m} tau-leap, {tag}")
            res["y_ranges"].append(s)
            print(f"m={m:5d} {tag:13s} y in [{s['min']:+.4f}, {s['max']:+.4f}]"
                  f"  mean {s['mean']:+.4f}  neg {s['n_neg']}/{s['n']}")

    # ---- 4. the population coefficient is the (positive) Doob ratio ------
    print("\npopulation coefficient E[y | X_t] at m = 4 (exact enumeration):")
    res["population_coefficient_m4"] = population_coefficient()

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
