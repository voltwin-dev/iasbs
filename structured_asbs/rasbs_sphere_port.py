"""Faithful PyTorch port of the public R-ASBS `asbs_sphere_sampler.m` (Alg. 1).

Upstream:  https://github.com/mattiamosso/
           Hard-Constrained-Sampling-on-Embedded-Riemannian-Manifold-
           via-Adjoint-Schrodinger-Bridges
Commit:    bb71d1496468658f4ff1c9995773968fd12706a8   (2026-08-28)

Why a port.  The same reason as `rasbs_port.py`: the script needs MATLAB's
Deep Learning Toolbox (`dlnetwork`, `dlfeval`, `adamupdate`), which Octave does
not implement, so there is no way to run the original here.  This file exists
so that figure 10 can put their samples and ours in the *same* picture instead
of comparing our render against a screenshot of theirs.

Every structural choice is copied from the .m file:

  * target  E(x) = 6 (1 - x_3^2)  on S^2, ambient gradient [0, 0, -12 x_3];
  * source is uniform on S^2 (Gaussian, normalised), NOT a Dirac;
  * ambient Euler-Maruyama step with tangent-projected drift and noise,
    followed by a normalise-to-the-sphere retraction;
  * sigma = 1, N_steps = 500, B = 500, epochs = 600, lr_u = lr_h = 2e-3;
  * netU: [x1,x2,x3,t] (4) -> 64 tanh -> 64 tanh -> 3
    netH: [x1,x2,x3]   (3) -> 48 tanh -> 48 tanh -> 3
  * terminal adjoint  a_1 = P_x(grad E) + P_x(h_phi(x));
  * geodesic bridge with tangential noise of std sigma sqrt(t(1-t)),
    then normalise;
  * geodesic parallel transport of a_1 from x_1 back to x_t;
  * controller loss  || P_x(u) + sigma a_t ||^2  averaged over the batch;
  * corrector target  Log_{x1}(x0) / sigma^2.

One MATLAB detail worth flagging because it is easy to get wrong when porting:
their parallel-transport line is

    a_t = a1_ortho + a1_parallel*cos(phi) - dot(a_1,u_pt)*x1*sin(phi)

which is the standard transport formula written with `u_pt` pointing from x1
towards xt.  It is reproduced verbatim rather than rewritten.

Nothing here is tuned, improved or "fixed".  Deviating would make the
comparison meaningless.
"""

import argparse
import json
import math
import time

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(din, dout, hidden):
    """MATLAB fullyConnectedLayer/tanhLayer stack, two hidden layers."""
    return torch.nn.Sequential(
        torch.nn.Linear(din, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, dout))


def proj_tangent(x, v):
    """P_x(v) = v - <v, x> x.  Valid because |x| = 1 on the unit sphere."""
    return v - (v * x).sum(-1, keepdim=True) * x


def normalise(x):
    """Their retraction: divide by the norm."""
    return x / x.norm(dim=-1, keepdim=True)


def grad_E(x):
    """Ambient gradient of E(x) = 6 (1 - x_3^2), i.e. [0, 0, -12 x_3]."""
    g = torch.zeros_like(x)
    g[..., 2] = -12.0 * x[..., 2]
    return g


def uniform_sphere(n, device, generator=None):
    """Their source: N(0, I_3) normalised.  Uniform on S^2, not a Dirac."""
    z = torch.randn(n, 3, device=device, generator=generator)
    return normalise(z)


def simulate_trajectories(netU, n, sigma, dt, N_steps, device,
                          generator=None):
    """`simulate_trajectories` from the .m file: geodesic random walk with a
    normalise retraction.  Returns (X0, X1)."""
    X0 = uniform_sphere(n, device, generator)
    X = X0.clone()
    with torch.no_grad():
        for step in range(N_steps):
            t_val = step * dt                      # MATLAB (step-1)*dt, 1-based
            t = torch.full((n, 1), t_val, device=device)
            u = proj_tangent(X, netU(torch.cat([X, t], -1)))
            noise = torch.randn(n, 3, device=device, generator=generator)
            X = X + u * dt + sigma * math.sqrt(dt) * proj_tangent(X, noise)
            X = normalise(X)
    return X0, X


def adjoint_targets(X0, X1, netH, sigma, generator=None):
    """Vectorised `compute_adjoint_targets`.  The MATLAB loop is over the
    batch; the arithmetic per particle is unchanged."""
    dev = X0.device
    B = X0.shape[0]
    with torch.no_grad():
        h = proj_tangent(X1, netH(X1))
    a1 = proj_tangent(X1, grad_E(X1)) + h

    t = torch.rand(B, 1, device=dev, generator=generator)
    dot01 = (X0 * X1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot01)

    w = X1 - dot01 * X0
    wn = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u_dir = w / wn
    mu = X0 * torch.cos(t * theta) + u_dir * torch.sin(t * theta)

    noise = torch.randn(B, 3, device=dev, generator=generator)
    std = sigma * torch.sqrt((t * (1.0 - t)).clamp(min=0.0))
    Xt = normalise(mu + std * proj_tangent(mu, noise))

    # theta ~ 0: the bridge is degenerate, MATLAB sets Xt = x1, a_t = a_1.
    degen = (theta < 1e-5)
    Xt = torch.where(degen, X1, Xt)

    dot1t = (X1 * Xt).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    phi = torch.acos(dot1t)
    v = Xt - dot1t * X1
    vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u_pt = v / vn
    c = (a1 * u_pt).sum(-1, keepdim=True)
    a_par = c * u_pt
    a_t = (a1 - a_par) + a_par * torch.cos(phi) - c * X1 * torch.sin(phi)
    a_t = torch.where((phi > 1e-5) & (~degen), a_t, a1)
    return Xt, t, a_t


def corrector_targets(X0, X1, sigma):
    """`compute_corrector_targets`: Log_{x1}(x0) / int_0^1 sigma_t^2 dt."""
    dot01 = (X1 * X0).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot01)
    w = X0 - dot01 * X1
    wn = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    b = (w / wn) * theta / (sigma ** 2)
    return torch.where(theta > 1e-5, b, torch.zeros_like(b))


def train_rasbs_sphere(args, device=DEV, verbose=False):
    torch.manual_seed(args.seed)
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    netU = mlp(4, 3, args.hidden_u).to(device)
    netH = mlp(3, 3, args.hidden_h).to(device)
    optU = torch.optim.Adam(netU.parameters(), lr=args.lr_u)
    optH = torch.optim.Adam(netH.parameters(), lr=args.lr_h)

    dt = 1.0 / args.steps
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # STEP 1: adjoint / controller update.
        X0, X1 = simulate_trajectories(netU, args.batch, args.sigma, dt,
                                       args.steps, device, g)
        Xt, tv, a_t = adjoint_targets(X0, X1, netH, args.sigma, g)
        u = proj_tangent(Xt, netU(torch.cat([Xt, tv], -1)))
        loss_u = ((u + args.sigma * a_t) ** 2).sum(-1).mean()
        optU.zero_grad(set_to_none=True)
        loss_u.backward()
        optU.step()

        # STEP 2: corrector update, on freshly collected pairs.
        X0n, X1n = simulate_trajectories(netU, args.batch, args.sigma, dt,
                                         args.steps, device, g)
        b = corrector_targets(X0n, X1n, args.sigma)
        h = proj_tangent(X1n, netH(X1n))
        loss_h = ((h - b) ** 2).sum(-1).mean()
        optH.zero_grad(set_to_none=True)
        loss_h.backward()
        optH.step()

        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{args.epochs} | controller "
                  f"{loss_u.item():8.5f} | corrector {loss_h.item():8.5f}",
                  flush=True)
    return netU, netH, time.time() - t0


def ks_uniform_z(z):
    """KS distance of x_3 against the exact target law, which for
    pi ~ exp(6 x_3^2) on S^2 has density proportional to exp(6 z^2) on
    z in [-1, 1] (the sphere's z-marginal is uniform before tilting)."""
    zs = np.sort(np.asarray(z, dtype=np.float64))
    grid = np.linspace(-1.0, 1.0, 20001)
    w = np.exp(6.0 * grid ** 2)
    cdf = np.cumsum(w)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    F = np.interp(zs, grid, cdf)
    n = zs.size
    emp_lo = np.arange(n) / n
    emp_hi = np.arange(1, n + 1) / n
    return float(max(np.abs(F - emp_lo).max(), np.abs(F - emp_hi).max()))


def run(args):
    dev = torch.device(DEV)
    print("R-ASBS sphere port (commit bb71d14).  E(x) = 6 (1 - x_3^2)")
    print(f"  sigma={args.sigma}  N_steps={args.steps}  B={args.batch}  "
          f"epochs={args.epochs}  lr={args.lr_u}")
    pu = sum(p.numel() for p in mlp(4, 3, args.hidden_u).parameters())
    ph = sum(p.numel() for p in mlp(3, 3, args.hidden_h).parameters())
    print(f"  params: netU {pu}  netH {ph}  total {pu + ph}", flush=True)

    netU, netH, wall = train_rasbs_sphere(args, dev, args.verbose)

    g = torch.Generator(device=dev)
    g.manual_seed(args.seed + 12345)
    _, X = simulate_trajectories(netU, args.n_samples, args.sigma,
                                 1.0 / args.steps, args.steps, dev, g)
    X = X.double().cpu().numpy()
    z = X[:, 2]
    north = float((z > 0).mean())
    ks = ks_uniform_z(z)
    resid = float(np.abs((X ** 2).sum(-1) - 1.0).max())
    E = float((6.0 * (1.0 - z ** 2)).mean())
    print(f"  north mass {north:.4f} (err {abs(north - 0.5):.4f})  "
          f"KS(x3) {ks:.4f}  |x|-1 {resid:.2e}  E {E:.4f}  ({wall:.0f}s)")

    out = {"config": vars(args),
           "commit": "bb71d1496468658f4ff1c9995773968fd12706a8",
           "params": {"netU": pu, "netH": ph, "total": pu + ph},
           "north_mass": north, "north_err": abs(north - 0.5),
           "KS_z": ks, "max_norm_resid": resid, "E_mean": E,
           "wall_s": wall}
    if args.samples_out:
        np.save(args.samples_out, X.astype(np.float64))
        out["samples"] = args.samples_out
        print(f"  wrote {args.samples_out}  ({X.shape[0]} samples)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr-u", type=float, default=2e-3)
    ap.add_argument("--lr-h", type=float, default=2e-3)
    ap.add_argument("--hidden-u", type=int, default=64)
    ap.add_argument("--hidden-h", type=int, default=48)
    ap.add_argument("--n-samples", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--samples-out", type=str,
                    default="json/samples_rasbs_sphere.npy")
    ap.add_argument("--out", type=str,
                    default="json/results_rasbs_sphere.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
