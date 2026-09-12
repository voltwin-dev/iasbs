"""Reviewer concern 2: reference-bridge audit for S^2 and St(4,2).

Three questions, answered with the shipped code and no retraining:

  1. what is the tangent-to-skew map used by the controlled Stiefel
     integrator, and does Omega_X(S) X = S hold for tangent S,
  2. do the reference bridge samplers actually target the reference bridge,
  3. are the production bridge / fibre / table resolutions converged.

Question 2 is answered by the mixture identity: if the bridge kernel is
correct, then drawing Y from the terminal law of the *unconditioned* reference
process and running the bridge to Y must reproduce the unconditioned marginal
at every intermediate time.  Both sides come from the same discretisation, so
any disagreement is a bridge error and not a step-size artefact.  The
discrepancy is reported as an unbiased Gaussian-kernel MMD^2 against the
split-half noise floor of the unconditioned marginal itself.
"""

import math

import numpy as np
import torch


import _paths                                    # noqa: F401

import common as C                                             # noqa: E402
import sphere as SPH                                           # noqa: E402
import stiefel as ST                                           # noqa: E402


def mmd2_unbiased(X, Y, sigma2):
    """Unbiased MMD^2 with kernel exp(-|a-b|^2 / (2 sigma2))."""
    X, Y = X.to(torch.float64), Y.to(torch.float64)
    n, m = X.shape[0], Y.shape[0]

    def K(A, B):
        d = torch.cdist(A, B).pow(2)
        return torch.exp(-d / (2.0 * sigma2))

    Kxx, Kyy, Kxy = K(X, X), K(Y, Y), K(X, Y)
    Kxx.fill_diagonal_(0.0)
    Kyy.fill_diagonal_(0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                 - 2.0 * Kxy.mean())


def median_bandwidth(X, cap=2000):
    X = X[:cap].to(torch.float64)
    d = torch.cdist(X, X)
    return float(d[d > 0].median() ** 2)


def split_half_floor(Z, n, sigma2, reps, gen):
    """Reference-vs-reference MMD^2 on disjoint halves: the noise floor."""
    vals = []
    for _ in range(reps):
        p = torch.randperm(Z.shape[0], generator=gen, device=Z.device)
        vals.append(mmd2_unbiased(Z[p[:n]], Z[p[n:2 * n]], sigma2))
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------- concern 2.1
def audit_omega(dev, n_trial=512):
    """Omega_X(S) = S X^T - X S^T - X (X^T S) X^T  (``skew_lift``).

    For S tangent at X, i.e. X^T S + S^T X = 0 and X^T X = I_p:

        Omega X = S X^T X - X S^T X - X (X^T S) X^T X
                = S - X (S^T X + X^T S) = S,

    so the identity is algebraic and uses tangency exactly once.  Skewness is
    likewise equivalent to tangency.  Both are checked numerically here, and
    the non-tangent case is checked to FAIL, so the test cannot pass vacuously.
    """
    X = C.random_stiefel(n_trial, 4, 2, device=dev)
    D = torch.randn(n_trial, 4, 2, dtype=X.dtype, device=dev)
    S = ST.proj_tangent(X, D)
    tang = float((X.transpose(-1, -2) @ S
                  + S.transpose(-1, -2) @ X).abs().max())
    Om = ST.skew_lift(X, S)
    e_id = float((Om @ X - S).abs().max())
    e_sk = float((Om + Om.transpose(-1, -2)).abs().max())
    # control: the same map on a NON-tangent D must not satisfy Omega X = D
    Om_bad = ST.skew_lift(X, D)
    e_bad = float((Om_bad @ X - D).abs().max())
    return {"tangency_residual": tang, "max|Omega X - S|": e_id,
            "max|Omega + Omega^T|": e_sk,
            "control max|Omega X - D|, D non-tangent": e_bad}


# ---------------------------------------------------------------- concern 2.2
@torch.no_grad()
def audit_stiefel_bridge(dev, batch=8000, steps=128, nfib=512, nq=64,
                         times=(0.25, 0.5, 0.75), mmd_n=2000, reps=8,
                         hi_steps=0, seed=0):
    prob = ST.StiefelProblem(steps=steps, nq=nq, nfib=nfib, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    # unconditioned reference process, keeping every grid point
    Y, path = ST.simulate(prob, None, batch, steps, generator=g,
                          keep_path=True)
    bp = ST.bridge_path(prob, Y, steps, generator=g)
    out = []
    for t in times:
        s = int(round(t * steps))
        A = path[s].reshape(batch, -1)
        B = bp[s].reshape(batch, -1)
        s2 = median_bandwidth(A)
        gg = torch.Generator(device=dev)
        gg.manual_seed(1234)
        mmd = np.mean([mmd2_unbiased(A[torch.randperm(batch, generator=gg,
                                                      device=dev)[:mmd_n]],
                                     B[torch.randperm(batch, generator=gg,
                                                      device=dev)[:mmd_n]], s2)
                       for _ in range(reps)])
        fl, fsd = split_half_floor(A, mmd_n, s2, reps, gg)
        out.append({"t": t, "MMD2": float(mmd), "floor": fl, "floor_sd": fsd,
                    "ratio": float(mmd) / max(abs(fl), 1e-30)})
    endpoint = float((bp[0] - prob.E0[None]).abs().max())
    return out, endpoint


@torch.no_grad()
def audit_sphere_bridge(dev, batch=8000, steps=128, times=(0.25, 0.5, 0.75),
                        mmd_n=2000, reps=8, seed=0):
    prob = SPH.SphereProblem(steps=steps, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    Y, path = SPH.simulate(prob, None, batch, steps, generator=g,
                           keep_path=True)
    bp = SPH.bridge_path(prob, Y, steps, generator=g)
    out = []
    for t in times:
        s = int(round(t * steps))
        A, B = path[s], bp[s]
        s2 = median_bandwidth(A)
        gg = torch.Generator(device=dev)
        gg.manual_seed(1234)
        mmd = np.mean([mmd2_unbiased(A[torch.randperm(batch, generator=gg,
                                                      device=dev)[:mmd_n]],
                                     B[torch.randperm(batch, generator=gg,
                                                      device=dev)[:mmd_n]], s2)
                       for _ in range(reps)])
        fl, fsd = split_half_floor(A, mmd_n, s2, reps, gg)
        out.append({"t": t, "MMD2": float(mmd), "floor": fl, "floor_sd": fsd,
                    "ratio": float(mmd) / max(abs(fl), 1e-30)})
    return out


# ---------------------------------------------------------------- concern 2.3
@torch.no_grad()
def audit_resolution(dev, batch=4000, steps=128, seed=0):
    """Sensitivity of the quotient kernel and the fibre sampler.

    ``nq`` is the Gauss-Legendre order of the SO(2) fibre quadrature that
    defines ``log_pst`` (hence the terminal label), ``nfib`` is the uniform
    grid the bridge SAMPLES the fibre on, and ``ntheta`` is the S^3 table
    resolution.  Production is nq=64, nfib=512, ntheta=16385.
    """
    torch.manual_seed(seed)
    Y = C.random_stiefel(batch, 4, 2, device=dev)
    rows = {}
    ref = ST.StiefelProblem(steps=steps, nq=512, nfib=4096, device=dev)
    lref = ref.log_pst(Y)
    for nq in (16, 32, 64, 128, 256):
        p = ST.StiefelProblem(steps=steps, nq=nq, device=dev)
        d = (p.log_pst(Y) - lref).abs()
        rows[f"nq={nq}"] = {"max|dlog_pst|": float(d.max()),
                            "mean|dlog_pst|": float(d.mean())}
    # fibre sampler: compare the law of the lifted pair against nfib=4096
    g = torch.Generator(device=dev)
    fib = {}
    g.manual_seed(7)
    a_ref, b_ref = ref.sample_fibre_lift(Y, generator=g)
    Zref = torch.cat([a_ref, b_ref], dim=-1)
    s2 = median_bandwidth(Zref)
    gg = torch.Generator(device=dev)
    gg.manual_seed(11)
    fl, _ = split_half_floor(Zref, batch // 2, s2, 6, gg)
    for nfib in (64, 128, 512, 2048):
        p = ST.StiefelProblem(steps=steps, nq=64, nfib=nfib, device=dev)
        g.manual_seed(7)
        a, b = p.sample_fibre_lift(Y, generator=g)
        Z = torch.cat([a, b], dim=-1)
        m = mmd2_unbiased(Z[:batch // 2], Zref[batch // 2:], s2)
        fib[f"nfib={nfib}"] = {"MMD2_vs_4096": m, "floor": fl,
                               "ratio": m / max(abs(fl), 1e-30)}
    # S^3 table resolution
    ss = ref.spin_ss
    t_hi = ST.S3Table(ss, ntheta=65537, device=dev)
    tabs = {}
    u = torch.linspace(-0.999, 0.999, 4001, dtype=torch.float64, device=dev)
    si = torch.full((u.shape[0],), len(ss) // 2, device=dev, dtype=torch.long)
    for nth in (4097, 16385, 32769):
        t_lo = ST.S3Table(ss, ntheta=nth, device=dev)
        d1 = (t_lo.logp_at(si, u) - t_hi.logp_at(si, u)).abs().max()
        d2 = (t_lo.dlogp_du(si, u) - t_hi.dlogp_du(si, u)).abs().max()
        tabs[f"ntheta={nth}"] = {"max|dlogp|": float(d1),
                                 "max|d dlogp/du|": float(d2)}
    return rows, fib, tabs


@torch.no_grad()
def audit_bridge_steps(dev, batch=8000, prod=128, hi=1024, times=(0.5,),
                       mmd_n=2000, reps=8, seed=0):
    """Production bridge resolution against a 8x finer bridge, matched times.

    Both legs condition on the SAME terminal points, so the comparison isolates
    the bridge discretisation rather than the terminal law.
    """
    p_lo = ST.StiefelProblem(steps=prod, device=dev)
    p_hi = ST.StiefelProblem(steps=hi, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    Y = ST.simulate(p_lo, None, batch, prod, generator=g)
    g.manual_seed(seed + 1)
    lo = ST.bridge_path(p_lo, Y, prod, generator=g)
    g.manual_seed(seed + 1)
    hi_p = ST.bridge_path(p_hi, Y, hi, generator=g)
    out = []
    for t in times:
        A = lo[int(round(t * prod))].reshape(batch, -1)
        B = hi_p[int(round(t * hi))].reshape(batch, -1)
        s2 = median_bandwidth(A)
        gg = torch.Generator(device=dev)
        gg.manual_seed(5)
        mmd = np.mean([mmd2_unbiased(
            A[torch.randperm(batch, generator=gg, device=dev)[:mmd_n]],
            B[torch.randperm(batch, generator=gg, device=dev)[:mmd_n]], s2)
            for _ in range(reps)])
        fl, _ = split_half_floor(A, mmd_n, s2, reps, gg)
        out.append({"t": t, "MMD2_128_vs_1024": float(mmd), "floor": fl,
                    "ratio": float(mmd) / max(abs(fl), 1e-30)})
    return out
