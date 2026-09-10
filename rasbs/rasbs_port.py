"""Faithful PyTorch port of the public R-ASBS `alg2_stiefel.m`.

Upstream:  https://github.com/mattiamosso/
           Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-
           via-Adjoint-Schrodinger-Bridges
Commit:    bb71d1496468658f4ff1c9995773968fd12706a8   (2026-08-28)

A port is unavoidable: the script depends on MATLAB's Deep Learning Toolbox
(`dlnetwork`, `dlfeval`, `adamupdate`), which GNU Octave does not implement.
Every structural choice below is copied line-for-line from the .m file:

  * H is their 4x4 Z2xZ2 group matrix (spectrum {1,2,5,8}, so it is the same
    problem as diag(1,2,5,8) up to a fixed orthogonal change of basis);
  * source is uniform Haar, NOT a Dirac;
  * Euler step in the ambient space followed by a QR retraction, NOT a
    geodesic / orthogonal-group step;
  * sigma = 1, N_steps = 199, B = 600, K_epochs = 1000, lr = 1e-3;
  * netU: [X_flat; t] (9) -> 256 tanh -> 256 tanh -> 8
    netH: [X_flat]    (8) -> 256 tanh -> 256 tanh -> 8
  * terminal adjoint  v = P_X(2 beta H X) + h_phi(X)   (h is NOT projected,
    matching the MATLAB source);
  * backward sequential projection (their "PAT");
  * controller loss  || P_X(u) + sigma v ||^2 / B;
  * corrector target  -P_{X_1}( (X_1 - X_0)/sigma^2 ).

The reported energy is tr(X^T H X), i.e. WITHOUT the beta factor, exactly as
in the .m file.
"""

import argparse
import json
import math
import time

import numpy as np
import torch

# common.py and the shared json/ ckpt/ fig/ directories live at the
# repository root, one level up from this script.  Only save_ckpt and the
# root-resolution helpers are borrowed -- none of the R-ASBS maths below
# touches our own utilities.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"

H_RASBS = [[4.0, 0.5, 2.5, 1.0],
           [0.5, 4.0, 1.0, 2.5],
           [2.5, 1.0, 4.0, 0.5],
           [1.0, 2.5, 0.5, 4.0]]

# The frame-sensitive target of the IASBS side, written in R-ASBS's own basis.
#
#   E(X) = tr(X^T H X) - lambda tr(C^T X)
#
# The quadratic part is invariant under a left O(4) change of basis, so
# H_RASBS and diag(1,2,5,8) pose the same problem; the frame term is NOT, so C
# has to be carried by the same rotation.  With H_RASBS = U diag(1,2,5,8) U^T
# and X_rasbs = U X_iasbs, both tr(X^T H X) and tr(C^T X) are preserved exactly
# by taking C_rasbs = U C_iasbs.  Nothing else in their algorithm sees the
# basis, so this leaves their construction untouched while putting both methods
# on the identical target.  `--check-basis` asserts the agreement numerically.
C_IASBS = [[0.7, -0.2], [0.1, 0.8], [-0.4, 0.3], [0.2, -0.5]]
H_IASBS_DIAG = [1.0, 2.0, 5.0, 8.0]


def basis_rotation(device=DEV):
    """U with H_RASBS = U diag(1,2,5,8) U^T, computed in float64."""
    Hd = torch.tensor(H_RASBS, dtype=torch.float64, device=device)
    ev, U = torch.linalg.eigh(Hd)            # ascending -> 1, 2, 5, 8
    return U, ev


def frame_C(device=DEV, dtype=torch.float32):
    """C_iasbs pushed into the R-ASBS basis."""
    U, _ = basis_rotation(device)
    Ci = torch.tensor(C_IASBS, dtype=torch.float64, device=device)
    return (U @ Ci).to(dtype)

# Their grid verbatim from alg2_stiefel.m, with the duplicated 0.01 removed.
BETA_GRID = [0.001, 0.01, 0.1, 0.5, 1.3, 2, 5, 7, 10, 20, 50, 100, 200,
             1000, 10000, 1000000]


def retraction_qr(A):
    """MATLAB Retraction(): thin QR with the sign of diag(R) folded in."""
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return Q * d.unsqueeze(-2)


def retraction(A):
    """Same map as retraction_qr, computed by modified Gram-Schmidt.

    Sign-corrected thin QR *is* Gram-Schmidt: the GS construction produces the
    unique QR factorisation with diag(R) > 0, which is exactly what folding
    sign(diag(R)) into Q gives.  Equality is asserted numerically by
    `--check-retraction`.  We need this because cuSOLVER's batched QR on
    600 x 4 x 2 blocks costs ~ms per call and the algorithm issues ~400 of them
    per epoch, dominating runtime; GS is pure elementwise algebra.
    """
    a1 = A[..., 0]
    q1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    a2 = A[..., 1]
    a2 = a2 - (q1 * a2).sum(-1, keepdim=True) * q1
    q2 = a2 / a2.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    return torch.stack([q1, q2], dim=-1)


def batch_projector(X, Z):
    """MATLAB batchProjector(): Z - X sym(X^T Z)."""
    XTZ = X.transpose(-1, -2) @ Z
    ZTX = Z.transpose(-1, -2) @ X
    return Z - X @ (0.5 * (XTZ + ZTX))


def haar(B, n, p, device, generator=None):
    return retraction(torch.randn(B, n, p, device=device, generator=generator))


def mlp(din, dout, hidden=256):
    return torch.nn.Sequential(
        torch.nn.Linear(din, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, dout))


def _u(netU, X, t):
    B = X.shape[0]
    f = torch.cat([X.reshape(B, 8),
                   torch.full((B, 1), t, device=X.device, dtype=X.dtype)], 1)
    return netU(f).reshape(B, 4, 2)


def _h(netH, X):
    return netH(X.reshape(X.shape[0], 8)).reshape(X.shape[0], 4, 2)


def simulate_trajectories(netU, X0, sigma, dt, N, keep=True, generator=None):
    """MATLAB simulate_trajectories(): ambient Euler step + QR retraction."""
    X = X0
    traj = [X0] if keep else None
    for j in range(1, N + 1):
        t = (j - 1) * dt
        u = _u(netU, X, t)
        noise = torch.randn(X.shape, device=X.device, dtype=X.dtype,
                            generator=generator)
        drift = sigma * batch_projector(X, u) * dt
        diff = sigma * math.sqrt(dt) * batch_projector(X, noise)
        X = retraction(X + drift + diff)
        if keep:
            traj.append(X)
    return torch.stack(traj) if keep else X


def energy_rasbs(X, H, Cm=None, lam=1.0):
    """tr(X^T H X) - lam tr(C^T X), both terms in whatever basis is passed."""
    e = torch.einsum("bnp,nm,bmp->b", X, H, X)
    if Cm is not None:
        e = e - lam * torch.einsum("np,bnp->b", Cm, X)
    return e


def energy_grad_rasbs(X, H, Cm=None, lam=1.0):
    g = 2.0 * (H @ X)
    if Cm is not None:
        g = g - lam * Cm
    return g


def train_rasbs(beta, args, device=DEV, verbose=False, seed=None):
    torch.manual_seed(args.seed if seed is None else seed)
    n, p, B, N = 4, 2, args.batch, args.steps
    dt = 1.0 / N
    sigma = args.sigma
    H = torch.tensor(H_RASBS, dtype=torch.float32, device=device)
    # Frame term, if requested.  It enters exactly where the quadratic
    # gradient does -- the terminal adjoint v = P_X(beta grad E) + h -- so the
    # rest of their algorithm is untouched.
    Cm = frame_C(device) if args.frame else None
    netU = mlp(n * p + 1, n * p).to(device)
    netH = mlp(n * p, n * p).to(device)
    optU = torch.optim.Adam(netU.parameters(), lr=args.lr)
    optH = torch.optim.Adam(netH.parameters(), lr=args.lr)
    oracle = 0                      # terminal energy-gradient evaluations
    t0 = time.time()
    for k in range(1, args.epochs + 1):
        X0 = haar(B, n, p, device)
        with torch.no_grad():
            traj = simulate_trajectories(netU, X0, sigma, dt, N)   # (N+1,B,4,2)
            XN = traj[N]
            v = batch_projector(
                XN, beta * energy_grad_rasbs(XN, H, Cm, args.lam)) \
                + _h(netH, XN)
            oracle += B
            V = torch.empty(N, B, n, p, device=device)
            for j in range(N - 1, -1, -1):
                v = batch_projector(traj[j], v)
                V[j] = v
            Xf = traj[:N].reshape(N * B, n, p)
            Vf = V.reshape(N * B, n, p)
            tf = torch.arange(N, device=device, dtype=torch.float32
                              ).repeat_interleave(B)[:, None] * dt
        inp = torch.cat([Xf.reshape(N * B, 8), tf], 1)
        up = netU(inp).reshape(N * B, n, p)
        lossU = ((batch_projector(Xf, up) + sigma * Vf) ** 2).sum() / (N * B)
        optU.zero_grad(set_to_none=True)
        lossU.backward()
        optU.step()

        with torch.no_grad():
            X1 = simulate_trajectories(netU, X0, sigma, dt, N, keep=False)
            b_target = -batch_projector(X1, (X1 - X0) / (sigma ** 2))
        hp = batch_projector(X1, _h(netH, X1))
        lossH = ((hp - b_target) ** 2).sum() / B
        optH.zero_grad(set_to_none=True)
        lossH.backward()
        optH.step()
        if verbose and (k % 200 == 0 or k == 1):
            print(f"    epoch {k:04d} | loss U {float(lossU):.4f} | "
                  f"loss H {float(lossH):.4f} | {time.time()-t0:.0f}s",
                  flush=True)
    return netU, netH, H, oracle, Cm


def check_retraction(device=DEV):
    """Assert GS retraction == sign-corrected QR retraction, and time both."""
    torch.manual_seed(0)
    A = torch.randn(600, 4, 2, device=device)
    Q1, Q2 = retraction_qr(A), retraction(A)
    err = float((Q1 - Q2).abs().max())
    orth = float((Q2.transpose(-1, -2) @ Q2
                  - torch.eye(2, device=device)).abs().max())
    def _t(fn, n=200):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn(A)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t) / n * 1e3
    tq, tg = _t(retraction_qr), _t(retraction)
    print(f"retraction check: |GS - QR|_max = {err:.3e}   "
          f"|Q^TQ - I|_max = {orth:.3e}")
    print(f"  QR {tq:.3f} ms/call   GS {tg:.3f} ms/call   "
          f"speedup {tq / max(tg, 1e-9):.1f}x")
    ok = err < 1e-5 and orth < 1e-5
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


def evaluate(netU, args, Hm, Cm, steps, device, ref_E=None):
    """Terminal metrics for a trained control propagated on `steps` steps.

    Identical statistics to `iasbs/stiefel.py:energy_metrics`, so
    the two columns of the comparison are computed by the same formulas: mean
    and std of E, max |X^T X - I|, mean tr(C^T X), and -- when a reference
    sample is supplied -- the KS distance of the energy law against it plus
    |Delta E|.
    """
    with torch.no_grad():
        X0 = haar(args.n_samples, 4, 2, device)
        X1 = simulate_trajectories(netU, X0, args.sigma, 1.0 / steps,
                                   steps, keep=False)
        E = energy_rasbs(X1, Hm, Cm, args.lam)
        Cq = Cm if Cm is not None else frame_C(device)
        m = {"E_mean": float(E.mean()), "E_std": float(E.std()),
             "constraint": float(torch.linalg.norm(
                 X1.transpose(-1, -2) @ X1
                 - torch.eye(2, device=device), dim=(-2, -1)).max()),
             "trCX": float(torch.einsum("np,bnp->b", Cq, X1).mean()),
             "n_samples": int(args.n_samples), "steps": int(steps)}
    if ref_E is not None:
        g = np.sort(np.asarray(ref_E, dtype=np.float64))
        cdf = (np.arange(len(g)) + 1.0) / len(g)
        m["KS_E"] = C.ks_against_grid_cdf(E.cpu().numpy(), g, cdf)
        m["E_err"] = abs(m["E_mean"] - float(g.mean()))
    m["_samples"] = X1
    return m


def reference_energies(args, beta, device):
    """MCMC reference energies for this beta, in the R-ASBS basis.

    The reference checkpoints are IASBS's (`ckpt/<stem>_b<beta>_mcmc.pt`,
    100,000 exact MCMC frames in the diag(1,2,5,8) basis).  Rotating them by U
    puts them in the port's basis; the energy is then computed with the same
    formula the port's own samples go through, so both methods are scored
    against literally the same reference sample.
    """
    if not args.ref_stem:
        return None
    import os
    path = os.path.join(args.ckpt_dir or "ckpt",
                        f"{args.ref_stem}_b{beta:g}_mcmc.pt")
    if not os.path.exists(path):
        print(f"  [ref] {path} missing -- KS(E) and dE will be omitted")
        return None
    blob = torch.load(path, map_location=device, weights_only=False)
    Xi = blob["samples"].to(device=device, dtype=torch.float64)
    U, _ = basis_rotation(device)
    Xr = (U @ Xi).to(torch.float32)                    # IASBS basis -> theirs
    Hm = torch.tensor(H_RASBS, dtype=torch.float32, device=device)
    Cm = frame_C(device) if args.frame else None
    E = energy_rasbs(Xr, Hm, Cm, args.lam)
    print(f"  [ref] {path}: {len(E)} MCMC frames, E = {float(E.mean()):.4f} "
          f"+- {float(E.std()):.4f}")
    return E.cpu().numpy()


def check_basis(device=DEV):
    """Assert the rotated target equals the IASBS target frame by frame."""
    torch.manual_seed(0)
    U, ev = basis_rotation(device)
    Xi = retraction(torch.randn(4096, 4, 2, dtype=torch.float64,
                                device=device))
    Xr = U @ Xi
    Hd = torch.diag(torch.tensor(H_IASBS_DIAG, dtype=torch.float64,
                                 device=device))
    Hr = torch.tensor(H_RASBS, dtype=torch.float64, device=device)
    Ci = torch.tensor(C_IASBS, dtype=torch.float64, device=device)
    Cr = U @ Ci
    lam = 1.0
    Ei = torch.einsum("bnp,nm,bmp->b", Xi, Hd, Xi) \
        - lam * torch.einsum("np,bnp->b", Ci, Xi)
    Er = torch.einsum("bnp,nm,bmp->b", Xr, Hr, Xr) \
        - lam * torch.einsum("np,bnp->b", Cr, Xr)
    orth = float((Xr.transpose(-1, -2) @ Xr
                  - torch.eye(2, dtype=torch.float64, device=device)
                  ).abs().max())
    err = float((Ei - Er).abs().max())
    print(f"basis check: spec(H_RASBS) = "
          f"{np.round(ev.cpu().numpy(), 12).tolist()}")
    print(f"  max |E_iasbs(X) - E_rasbs(U X)| = {err:.3e}   "
          f"|X^T X - I| after rotation = {orth:.3e}")
    ok = err < 1e-10 and orth < 1e-10
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


def run(args):
    device = DEV
    betas = ([float(b) for b in args.betas.split(",")] if args.betas
             else BETA_GRID)
    H = torch.tensor(H_RASBS, dtype=torch.float32, device=device)
    ev = torch.linalg.eigvalsh(H).cpu().numpy()
    emin = float(np.sort(ev)[:2].sum())
    emax = 2.0 / 4.0 * float(ev.sum())
    print(f"R-ASBS port (commit bb71d14).  spec(H) = "
          f"{np.round(np.sort(ev), 6).tolist()}")
    print(f"  target_min_energy = {emin:.4f}   target_max_energy = {emax:.4f}")
    print(f"  sigma={args.sigma}  N_steps={args.steps}  B={args.batch}  "
          f"epochs={args.epochs}  lr={args.lr}")
    nparU = sum(q.numel() for q in mlp(9, 8).parameters())
    nparH = sum(q.numel() for q in mlp(8, 8).parameters())
    print(f"  params: netU {nparU}  netH {nparH}  total {nparU + nparH}")
    if args.frame:
        print(f"  FRAME-SENSITIVE target: E = tr(X^T H X) - {args.lam} "
              f"tr(C^T X), C = U C_iasbs (see --check-basis)")
    refine = [int(r) for r in args.refine.split(",") if r] if args.refine else []
    out = {}
    for beta in betas:
        ref_E = reference_energies(args, beta, device)
        ms = []
        for k in range(args.seeds):
            sd = args.seed + k
            t0 = time.time()
            netU, netH, Hm, oracle, Cm = train_rasbs(
                beta, args, device, verbose=args.verbose, seed=sd)
            m = evaluate(netU, args, Hm, Cm, args.steps, device, ref_E)
            m.update(oracle_calls=oracle, wall_s=time.time() - t0,
                     steps=args.steps, seed=sd)
            # Same trained, t-conditioned control, re-propagated on finer
            # grids.  This is the inference-resolution axis IASBS is reported
            # on (its `--refine`), not a retraining.
            m["refined"] = {}
            for r in refine:
                N = args.steps * r
                m["refined"][N] = evaluate(netU, args, Hm, Cm, N, device, ref_E)
            ms.append(m)
            print(f"  beta={beta:<8g} seed={sd}  E={m['E_mean']:.4f}"
                  f" +- {m['E_std']:.4f}  tr(C^T X)={m['trCX']:.4f}"
                  f"  |X^TX-I|={m['constraint']:.2e}"
                  + (f"  KS(E)={m['KS_E']:.4f}  dE={m['E_err']:.4f}"
                     if "KS_E" in m else "")
                  + f"  ({m['wall_s']:.0f}s, {oracle} oracle calls)",
                  flush=True)
            for N, mr in m["refined"].items():
                print(f"      refine N={N:<5d} E={mr['E_mean']:.4f}"
                      f"  tr(C^T X)={mr['trCX']:.4f}"
                      + (f"  KS(E)={mr['KS_E']:.4f}  dE={mr['E_err']:.4f}"
                         if "KS_E" in mr else ""), flush=True)
            # Terminal samples are kept for every grid, not just the base one,
            # so any later re-scoring (against a different MCMC reference, or
            # for a statistic not computed here) needs no retraining.
            samp_ref = {N: mr.pop("_samples").cpu()
                        for N, mr in m["refined"].items()}
            if args.ckpt_dir:
                tag = (f"{args.tag}_b{beta:g}" if args.seeds == 1
                       else f"{args.tag}_b{beta:g}_seed{sd}")
                C.save_ckpt(args.ckpt_dir, tag, net=netU,
                            samples=m["_samples"].cpu(),
                            extra={"metrics": {kk: vv for kk, vv in m.items()
                                               if kk != "_samples"},
                                   "samples_refined": samp_ref,
                                   "beta": beta, "config": vars(args),
                                   "commit": "bb71d1496468658f4ff1c9995773968fd12706a8"})
            m.pop("_samples", None)
        agg = {"seeds": ms,
               "E_mean": float(np.mean([m["E_mean"] for m in ms])),
               "wall_s": float(np.mean([m["wall_s"] for m in ms])),
               "oracle_calls": ms[0]["oracle_calls"]}
        if "E_err" in ms[0]:
            agg["dE"] = float(np.mean([m["E_err"] for m in ms]))
            agg["KS_E"] = float(np.mean([m["KS_E"] for m in ms]))
        out[beta] = agg
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "target_min": emin,
                   "target_max": emax, "spec_H": np.sort(ev).tolist(),
                   "commit": "bb71d1496468658f4ff1c9995773968fd12706a8",
                   "betas": out}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=str, default="")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=199)
    ap.add_argument("--batch", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    # Default "ckpt", never "": a run whose checkpoint is not
    # written cannot be re-measured later and must be repeated in full.
    ap.add_argument("--ckpt-dir", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="rasbs")
    ap.add_argument("--out", type=str, default="json/results_rasbs_stiefel.json")
    ap.add_argument("--check-retraction", action="store_true")
    # --- fair-comparison additions (section "Fair Stiefel comparison") ------
    # None of these touch the algorithm: --frame changes the target both
    # methods are trained against, --seeds/--refine/--ref-stem change what is
    # measured and how many times.
    ap.add_argument("--frame", action="store_true",
                    help="frame-sensitive target E = tr(X^T H X) "
                         "- lam tr(C^T X), C rotated into the R-ASBS basis")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--refine", type=str, default="",
                    help="comma-separated step multipliers to re-propagate "
                         "the trained control on, e.g. 2,4 -> 398 and 796")
    ap.add_argument("--ref-stem", type=str, default="",
                    help="IASBS MCMC reference stem, e.g. stiefel_frame; "
                         "reads ckpt/<stem>_b<beta>_mcmc.pt for KS(E) and dE")
    ap.add_argument("--check-basis", action="store_true")
    args = ap.parse_args()
    if args.check_retraction:
        return check_retraction()
    if args.check_basis:
        return check_basis()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
