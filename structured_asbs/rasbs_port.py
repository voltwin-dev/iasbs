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

DEV = "cuda" if torch.cuda.is_available() else "cpu"

H_RASBS = [[4.0, 0.5, 2.5, 1.0],
           [0.5, 4.0, 1.0, 2.5],
           [2.5, 1.0, 4.0, 0.5],
           [1.0, 2.5, 0.5, 4.0]]

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


def train_rasbs(beta, args, device=DEV, verbose=False):
    torch.manual_seed(args.seed)
    n, p, B, N = 4, 2, args.batch, args.steps
    dt = 1.0 / N
    sigma = args.sigma
    H = torch.tensor(H_RASBS, dtype=torch.float32, device=device)
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
            v = batch_projector(XN, 2.0 * beta * (H @ XN)) + _h(netH, XN)
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
    return netU, netH, H, oracle


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
    out = {}
    for beta in betas:
        t0 = time.time()
        netU, netH, Hm, oracle = train_rasbs(beta, args, device,
                                             verbose=args.verbose)
        with torch.no_grad():
            X0 = haar(args.n_samples, 4, 2, device)
            X1 = simulate_trajectories(netU, X0, args.sigma,
                                       1.0 / args.steps, args.steps,
                                       keep=False)
            E = torch.einsum("bnp,nm,bmp->b", X1, Hm, X1)
        m = {"E_mean": float(E.mean()), "E_std": float(E.std()),
             "constraint": float(torch.linalg.norm(
                 X1.transpose(-1, -2) @ X1
                 - torch.eye(2, device=device), dim=(-2, -1)).max()),
             "oracle_calls": oracle, "wall_s": time.time() - t0}
        out[beta] = m
        print(f"  beta={beta:<10g} E[tr(X^T H X)] = {m['E_mean']:.4f} "
              f"+- {m['E_std']:.4f}   |X^TX-I|={m['constraint']:.2e}   "
              f"({m['wall_s']:.0f}s, {oracle} oracle calls)", flush=True)
        if args.ckpt_dir:
            import common as C
            C.save_ckpt(args.ckpt_dir, f"{args.tag}_b{beta:g}",
                        net=netU, samples=X1.cpu(),
                        extra={"metrics": m, "beta": beta,
                               "commit": "bb71d1496468658f4ff1c9995773968fd12706a8"})
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "target_min": emin,
                   "target_max": emax, "spec_H": np.sort(ev).tolist(),
                   "commit": "bb71d1496468658f4ff1c9995773968fd12706a8",
                   "betas": out}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


def main():
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
    ap.add_argument("--ckpt-dir", type=str, default="")
    ap.add_argument("--tag", type=str, default="rasbs")
    ap.add_argument("--out", type=str, default="json/results_rasbs_stiefel.json")
    ap.add_argument("--check-retraction", action="store_true")
    args = ap.parse_args()
    if args.check_retraction:
        return check_retraction()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
