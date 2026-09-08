"""
tests_math.py -- Phase 1.  Deterministic mathematical unit tests.

No neural network is trained here.  Every mathematically important branch of
the structured-ASBS construction is validated before any experiment starts.

Run:
    python tests_math.py
    python tests_math.py --fast      # skip the two Monte-Carlo tests
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback

import numpy as np
import torch
from scipy.linalg import expm

# common.py and the shared json/ ckpt/ fig/ directories live at the
# repository root, one level up from this script.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C

torch.set_default_dtype(torch.float64)

_REGISTRY = []


def test(name):
    def deco(fn):
        _REGISTRY.append((name, fn))
        return fn
    return deco


# ============================================================================
# [1] Binary single-swap generator rows sum to zero
# ============================================================================


@test("[1] binary swap generator rows sum to zero")
def test_binary_generator_rows():
    for (n, k) in [(4, 2), (6, 3), (6, 2), (8, 4)]:
        states = C.enumerate_fixed_count(n, k)
        M, _ = C.binary_full_generator(states)
        rs = np.abs(M.sum(axis=1)).max()
        assert rs < 1e-12, f"n={n} k={k} row sum {rs}"
        # total escape rate must be exactly gamma = 1
        esc = np.abs(-np.diag(M) - 1.0).max()
        assert esc < 1e-12, f"n={n} k={k} escape rate {esc}"
        # off-diagonals non-negative
        off = M - np.diag(np.diag(M))
        assert off.min() >= 0.0
    return "rowsum<1e-12, escape rate == 1 exactly, 4 lattices"


# ============================================================================
# [2] Binary reference preserves particle number
# ============================================================================


@test("[2] binary reference preserves particle number")
def test_binary_particle_number():
    g = torch.Generator().manual_seed(0)
    n, k = 16, 8
    x = torch.zeros(n, dtype=torch.long)
    x[:k] = 1
    for _ in range(5000):
        i, j = C.sample_uniform_swap(x, generator=g)
        assert x[i] == 1 and x[j] == 0
        x = C.swap_state(x, i, j)
        assert int(x.sum()) == k
    C.assert_fixed_count(x.unsqueeze(0), k)

    # the enumerated generator never leaves the orbit either
    states = C.enumerate_fixed_count(8, 4)
    M, _ = C.binary_full_generator(states)
    assert states.sum(axis=1).min() == states.sum(axis=1).max() == 4
    assert M.shape == (70, 70)
    return "5000 swaps, zero violations; orbit closed under generator"


# ============================================================================
# [3] Small binary orbit kernel equals full matrix exponential
# ============================================================================


@test("[3] binary orbit kernel == full expm")
def test_binary_orbit_kernel():
    worst = 0.0
    for (n, k) in [(4, 2), (6, 3), (6, 2), (8, 3), (8, 4)]:
        states = C.enumerate_fixed_count(n, k)
        M, index = C.binary_full_generator(states)
        for Gamma in [0.05, 0.5, 2.0, 10.0]:
            P = expm(Gamma * M)
            kappa = C.binary_orbit_kernel(n, k, Gamma)
            x0 = states[0]
            row = P[0]
            pred = np.array([kappa[C.binary_orbit_distance(x0, s)] for s in states])
            err = np.abs(row - pred).max()
            worst = max(worst, err)
            assert err < 1e-10, f"n={n} k={k} Gamma={Gamma} err={err:.3e}"
            assert abs(row.sum() - 1.0) < 1e-12
    return f"max abs error {worst:.2e} over 5 orbits x 4 clocks (tol 1e-10)"


# ============================================================================
# [4] Occupation commutator identity
# ============================================================================


@test("[4] occupation commutator identities")
def test_occupation_commutator():
    worst = 0.0
    for (m, N) in [(3, 3), (4, 3), (3, 4), (4, 4)]:
        states = C.enumerate_occupation(m, N)
        index = C.occupation_index(states)
        L = C.occupation_L(states, index, m)

        # L is a proper generator
        assert np.abs(L.sum(axis=1)).max() < 1e-11
        # total escape rate is gamma * N
        assert np.abs(-np.diag(L) - N).max() < 1e-11

        for (i, j) in [(0, 1), (1, 0), (0, m - 1)]:
            if i == j:
                continue
            F = C.occupation_F(states, index, j, i)
            S = C.occupation_S(states, index, m, j, i)

            lhs1 = L @ F - F @ L
            rhs1 = S / (m - 1.0)
            e1 = np.abs(lhs1 - rhs1).max()

            lhs2 = L @ S - S @ L
            rhs2 = (m / (m - 1.0)) * S
            e2 = np.abs(lhs2 - rhs2).max()

            worst = max(worst, e1, e2)
            assert e1 < 1e-10, f"[L,F] m={m} N={N} err={e1:.3e}"
            assert e2 < 1e-10, f"[L,S] m={m} N={N} err={e2:.3e}"
    return f"[L,F]=S/(m-1) and [L,S]=m/(m-1) S, max err {worst:.2e}"


# ============================================================================
# [5] Occupation terminal intertwining F P = P A
# ============================================================================


@test("[5] occupation intertwining  F P = P A")
def test_occupation_intertwining():
    worst = 0.0
    for (m, N) in [(3, 3), (4, 3), (4, 4)]:
        states = C.enumerate_occupation(m, N)
        index = C.occupation_index(states)
        L = C.occupation_L(states, index, m)
        for Gamma in [0.1, 0.7, 3.0]:
            P = expm(Gamma * L)
            for (i, j) in [(0, 1), (1, 2 % m), (m - 1, 0)]:
                if i == j:
                    continue
                F = C.occupation_F(states, index, j, i)
                A = C.occupation_A(states, index, m, j, i, Gamma)
                err = np.abs(F @ P - P @ A).max()
                scale = max(np.abs(F @ P).max(), 1.0)
                worst = max(worst, err / scale)
                assert err / scale < 1e-10, (
                    f"m={m} N={N} Gamma={Gamma} (i,j)=({i},{j}) err={err:.3e}")
    return f"max relative error {worst:.2e} (tol 1e-10)"


# ============================================================================
# [6] Concentrated-source multinomial kernel ratio
# ============================================================================


@test("[6] concentrated-source multinomial kernel + ratio")
def test_occupation_multinomial_ratio():
    from scipy.stats import multinomial as spmulti

    worst_kernel = 0.0
    worst_ratio = 0.0
    for (m, N) in [(3, 4), (4, 3), (4, 5)]:
        states = C.enumerate_occupation(m, N)
        index = C.occupation_index(states)
        L = C.occupation_L(states, index, m)
        for Gamma in [0.15, 0.9, 4.0]:
            c = 0
            eta0 = np.zeros(m, dtype=np.int64)
            eta0[c] = N
            row = expm(Gamma * L)[index[tuple(eta0)]]

            q = C.occupation_single_particle_probs(m, Gamma, source_mode=c)
            assert abs(q.sum() - 1.0) < 1e-12
            pred = np.array([spmulti.pmf(s, N, q) for s in states])
            e = np.abs(row - pred).max()
            worst_kernel = max(worst_kernel, e)
            assert e < 1e-10, f"m={m} N={N} Gamma={Gamma} kernel err={e:.3e}"

            # neighbour ratio p(xi) / p(xi - e_b + e_a)
            for xi in states:
                for a in range(m):
                    for b in range(m):
                        if a == b or xi[b] == 0:
                            continue
                        zeta = xi.copy()
                        zeta[b] -= 1
                        zeta[a] += 1
                        direct = spmulti.pmf(xi, N, q) / spmulti.pmf(zeta, N, q)
                        got = C.occupation_neighbor_kernel_ratio(xi, a, b, q)
                        rel = abs(direct - got) / max(abs(direct), 1e-300)
                        worst_ratio = max(worst_ratio, rel)
                        assert rel < 1e-9
    return (f"kernel err {worst_kernel:.2e}, ratio rel err {worst_ratio:.2e}")


# ============================================================================
# [6b] Occupation terminal label -- full sum vs stochastic estimators
# ============================================================================


@test("[6b] occupation terminal label estimators unbiased")
def test_occupation_label_estimators():
    m, N, d, tau = 4, 5, 0.6, 1.0
    Gamma_0t = 0.8
    Gamma_t1 = 0.9
    q = C.occupation_single_particle_probs(m, Gamma_0t, source_mode=0)
    c_t = C.occupation_c_t(m, Gamma_t1)
    rho_t1 = C.occupation_rho(m, Gamma_t1)
    assert abs(m * c_t - (1.0 - rho_t1)) < 1e-12, "m c_t = 1 - rho identity"

    def energy_fn(eta):
        return C.inclusion_energy(eta, tau, d)

    xi = torch.tensor([2, 1, 1, 1], dtype=torch.long)
    i, j = 1, 2
    full = C.occupation_terminal_label_full(xi, i, j, q, energy_fn, tau, c_t)

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    n_mc = 40000
    u = np.mean([C.occupation_terminal_label_uniform_one_sample(
        xi, i, j, q, energy_fn, tau, rho_t1, rng=rng) for _ in range(n_mc)])
    o = np.mean([C.occupation_terminal_label_occupancy_sample(
        xi, i, j, q, energy_fn, tau, c_t) for _ in range(n_mc)])

    assert abs(u - full) < 5e-3 * max(abs(full), 1.0), (u, full)
    assert abs(o - full) < 5e-3 * max(abs(full), 1.0), (o, full)
    return (f"full={full:.6f}  uniform-1={u:.6f}  occupancy-1={o:.6f} "
            f"(n_mc={n_mc})")


# ============================================================================
# [7] Sphere Killing-frame resolution of identity
# ============================================================================


@test("[7] sphere Killing-frame resolution of identity")
def test_sphere_killing_identity():
    torch.manual_seed(0)
    worst_gram = 0.0
    worst_read = 0.0
    for d in [3, 4, 5]:
        for _ in range(5):
            x = torch.randn(d)
            x = x / x.norm()
            y = torch.randn(d)
            y = y / y.norm()
            G = torch.randn(d)

            gram = C.sphere_killing_gram(x, y)
            ref = (x @ y) * torch.eye(d) - torch.outer(y, x)
            worst_gram = max(worst_gram, float((gram - ref).abs().max()))

            expl = C.sphere_readout_explicit(x, y, G)
            coll = C.sphere_terminal_readout(x.unsqueeze(0), y.unsqueeze(0),
                                             G.unsqueeze(0))[0]
            worst_read = max(worst_read, float((expl - coll).abs().max()))

            # resolution of identity at y = x  ->  tangent projector
            proj = C.sphere_killing_gram(x, x)
            assert float((proj - (torch.eye(d) - torch.outer(x, x))).abs().max()) < 1e-12

    assert worst_gram < 1e-12, worst_gram
    assert worst_read < 1e-12, worst_read
    return (f"sum V(x)V(y)^T = (x.y)I - y x^T err {worst_gram:.2e}; "
            f"explicit vs collapsed {worst_read:.2e}; P_x at y=x exact")


# ============================================================================
# [8] Sphere tangent controller
# ============================================================================


@test("[8] sphere tangency / exp map / heat-kernel score")
def test_sphere_tangent_and_score():
    torch.manual_seed(1)
    B, d = 64, 3
    x = torch.randn(B, d)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(B, d)
    y = y / y.norm(dim=-1, keepdim=True)
    G = torch.randn(B, d)

    v = C.sphere_project_tangent(x, torch.randn(B, d))
    assert float((x * v).sum(-1).abs().max()) < 1e-12

    out = C.sphere_terminal_readout(x, y, G)
    tang = float((x * out).sum(-1).abs().max())
    assert tang < 1e-12, tang

    xe = C.sphere_exp(x, 0.3 * v)
    nrm = float((xe.norm(dim=-1) - 1.0).abs().max())
    assert nrm < 1e-12, nrm
    xe0 = C.sphere_exp(x, 1e-9 * v)
    assert float((xe0 - x).abs().max()) < 1e-8

    # AS terminal gradient pieces
    xg = torch.randn(4, 3)
    xg = xg / xg.norm(dim=-1, keepdim=True)
    xg.requires_grad_(True)
    E = C.sphere_energy(xg).sum()
    (gauto,) = torch.autograd.grad(E, xg)
    gman = C.sphere_energy_grad(xg.detach())
    assert float((gauto - gman).abs().max()) < 1e-12

    # S^2 heat-kernel spectral convergence
    x0 = np.array([0.0, 0.0, 1.0])
    conv = []
    for r in [0.05, 0.2, 0.8]:
        cs = np.cos(np.linspace(0.15, math.pi - 0.15, 41))
        p1, _ = C.s2_heat_kernel_and_dc(cs, r, Lmax=120)
        p2, _ = C.s2_heat_kernel_and_dc(cs, r, Lmax=240)
        conv.append(float(np.max(np.abs(p2 - p1) / np.abs(p2))))
    assert max(conv) < 1e-10, conv

    # normalisation  int p_r dOmega = 1
    th = np.linspace(1e-6, math.pi - 1e-6, 40001)
    for r in [0.05, 0.2, 0.8]:
        p, _ = C.s2_heat_kernel_and_dc(np.cos(th), r, Lmax=240)
        mass = np.trapezoid(p * 2.0 * np.pi * np.sin(th), th)
        assert abs(mass - 1.0) < 1e-8, (r, mass)

    # score vs finite difference of log kernel along a tangent geodesic
    rng = np.random.RandomState(3)
    worst = 0.0
    for r in [0.1, 0.4]:
        for _ in range(6):
            yv = rng.randn(3)
            yv /= np.linalg.norm(yv)
            sc = C.s2_reference_score(yv, x0, r, Lmax=240)
            tv = rng.randn(3)
            tv -= yv * (yv @ tv)
            tv /= np.linalg.norm(tv)
            eps = 1e-5
            yp = math.cos(eps) * yv + math.sin(eps) * tv
            ym = math.cos(eps) * yv - math.sin(eps) * tv
            fd = (C.s2_log_kernel(yp, x0, r, Lmax=240)
                  - C.s2_log_kernel(ym, x0, r, Lmax=240)) / (2 * eps)
            worst = max(worst, abs(fd - float(sc @ tv)))
    assert worst < 1e-6, worst
    return (f"tangency<1e-12, |x|=1 exact, spectral conv {max(conv):.1e}, "
            f"score vs FD {worst:.2e}")


# ============================================================================
# [9] Stiefel canonical Killing-frame resolution
# ============================================================================


@test("[9] Stiefel canonical Killing resolution / tangency")
def test_stiefel_killing_identity():
    torch.manual_seed(2)
    n, p, B = 4, 2, 32
    X = C.random_stiefel(B, n, p)
    Y = C.random_stiefel(B, n, p)
    G = torch.randn(B, n, p)

    assert float(C.stiefel_constraint_error(X).max()) < 1e-12

    R = C.stiefel_terminal_readout(X, Y, G)
    skew = X.transpose(-1, -2) @ R + R.transpose(-1, -2) @ X
    assert float(skew.abs().max()) < 1e-12, float(skew.abs().max())

    # resolution of identity at Y = X:  sum_a V_a <V_a, G> = G - X G^T X,
    # which is exactly the canonical-metric dual of the ambient covector G,
    # i.e.  g_c(Delta, T) = <G, T>_F  for every tangent T.
    Delta = C.stiefel_terminal_readout(X, X, G)
    ref = G - X @ (G.transpose(-1, -2) @ X)
    assert float((Delta - ref).abs().max()) < 1e-12

    skew2 = X.transpose(-1, -2) @ Delta + Delta.transpose(-1, -2) @ X
    assert float(skew2.abs().max()) < 1e-12

    worst_dual = 0.0
    for _ in range(8):
        T = C.stiefel_random_tangent(X)
        sk = X.transpose(-1, -2) @ T + T.transpose(-1, -2) @ X
        assert float(sk.abs().max()) < 1e-12, "random tangent not tangent"
        lhs = C.stiefel_canonical_inner(X, Delta, T)
        rhs = (G * T).sum(dim=(-2, -1))
        worst_dual = max(worst_dual, float((lhs - rhs).abs().max()))
    assert worst_dual < 1e-11, worst_dual

    # Casimir:  sum_{i<j} Omega_ij^2 = -(n-1) I
    for nn in [3, 4, 5, 6]:
        cas = C.stiefel_killing_casimir(nn)
        assert float((cas + (nn - 1) * torch.eye(nn)).abs().max()) < 1e-12
    return (f"skew test <1e-12, canonical duality g_c(Delta,T)=<G,T> err "
            f"{worst_dual:.2e}, Casimir = -(n-1) I")


# ============================================================================
# [10] Stiefel collapsed readout equals explicit generator sum
# ============================================================================


@test("[10] Stiefel collapsed readout == explicit Killing sum")
def test_stiefel_collapsed_readout():
    torch.manual_seed(3)
    worst = 0.0
    for (n, p) in [(4, 2), (4, 1), (5, 2), (6, 3)]:
        for _ in range(6):
            X = C.random_stiefel(1, n, p)[0]
            Y = C.random_stiefel(1, n, p)[0]
            G = torch.randn(n, p)
            e = C.stiefel_readout_explicit(X, Y, G)
            c = C.stiefel_readout_collapsed(X, Y, G)
            worst = max(worst, float((e - c).abs().max()))
    assert worst < 1e-10, worst
    return f"max |explicit - collapsed| = {worst:.2e} in float64 (tol 1e-10)"


# ============================================================================
# [11] Stiefel reference first moment uses exp(-3 r)
# ============================================================================


@test("[11] St(4,2) reference first moment = exp(-3 r) X0  (NOT exp(-6 r))")
def test_stiefel_first_moment(fast=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)

    # (a) deterministic operator statement:  L X = (sigma^2/2) sum Omega^2 X
    cas = C.stiefel_killing_casimir(4)
    assert float((cas + 3.0 * torch.eye(4)).abs().max()) < 1e-12

    # (b) Monte-Carlo on the actual simulator
    X0 = torch.zeros(4, 2, dtype=torch.float64, device=device)
    X0[0, 0] = 1.0
    X0[1, 1] = 1.0

    batch = 20000 if fast else 120000
    steps = 100 if fast else 300
    lines = []
    ok = True
    for r in [0.05, 0.15]:
        R = C.simulate_so_brownian(batch, 4, r, steps, device=device)
        Xt = R @ X0
        mean = Xt.mean(dim=0)
        pred3 = C.expected_st42_mean_factor(r) * X0
        pred6 = math.exp(-6.0 * r) * X0
        e3 = float((mean - pred3).norm())
        e6 = float((mean - pred6).norm())
        se = float(Xt.std(dim=0).norm()) / math.sqrt(batch)
        tol = 6.0 * se + 0.02 * abs(math.exp(-3.0 * r))
        lines.append(f"r={r}: |mean-e^-3r X0|={e3:.5f} (tol {tol:.5f}), "
                     f"|mean-e^-6r X0|={e6:.5f}, se={se:.5f}")
        ok = ok and (e3 < tol) and (e6 > 5 * e3)
        # constraint preserved exactly
        assert float(C.stiefel_constraint_error(Xt).max()) < 1e-10
    assert ok, " | ".join(lines)
    return " | ".join(lines) + f"  [batch={batch}, steps={steps}]"


# ============================================================================
# [12] Stiefel heat-kernel helper calls S^3 kernel at r/2
# ============================================================================


@test("[12] St(4,2) spin factors run at r/2 (S^3 direct still uses r)")
def test_stiefel_spin_clock(fast=False):
    # (a) helper contract
    for r in [0.01, 0.3, 1.0, 3.0]:
        assert C.stiefel_spin_factor_time(r) == 0.5 * r
        sig = math.sqrt(2 * r)
        assert abs(C.sphere_factor_clock_for_stiefel(0.0, 1.0, sig) - 0.5 * r) < 1e-14
        assert abs(C.heat_clock(0.0, 1.0, sig) - r) < 1e-14
        tp = torch.tensor([0.4, 1.1])
        tq = torch.tensor([0.9, 2.0])
        kp, kq = C.stiefel_st42_kernel_factors(tp, tq, r)
        assert torch.allclose(kp, C.s3_heat_kernel(tp, 0.5 * r))
        assert torch.allclose(kq, C.s3_heat_kernel(tq, 0.5 * r))
        assert not torch.allclose(kp, C.s3_heat_kernel(tp, r))

    # (b) S^3 kernel normalisation and winding convergence
    for s in [0.05, 0.2, 0.9]:
        _, _, tot = C.s3_radial_cdf_grid(s, num=60001, K=10)
        assert abs(tot - 1.0) < 1e-6, (s, tot)
        th = torch.linspace(0.05, math.pi - 0.05, 200, dtype=torch.float64)
        a = C.s3_heat_kernel(th, s, K=6)
        b = C.s3_heat_kernel(th, s, K=8)
        rel = float(((a - b).abs() / b.abs()).max())
        assert rel < 1e-10, (s, rel)

    # (c) the real statement: SO(4) BM at heat time r has spin radii ~ p_{r/2}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(12)
    batch = 8000 if fast else 30000
    steps = 200 if fast else 600
    r = 0.30
    R = C.simulate_so_brownian(batch, 4, r, steps, device=device)
    tp, tq = C.so4_spin_angles(R)
    pooled = np.concatenate([tp, tq])

    gx_half, gc_half, _ = C.s3_radial_cdf_grid(0.5 * r, num=60001, K=10)
    gx_full, gc_full, _ = C.s3_radial_cdf_grid(r, num=60001, K=10)
    ks_half = C.ks_against_grid_cdf(pooled, gx_half, gc_half)
    ks_full = C.ks_against_grid_cdf(pooled, gx_full, gc_full)

    assert ks_half < 0.02, f"KS vs p_(r/2) = {ks_half:.4f} (should be small)"
    assert ks_full > 0.10, f"KS vs p_r = {ks_full:.4f} (should be rejected)"
    return (f"KS(spin radii, p_r/2) = {ks_half:.4f}   "
            f"KS(spin radii, p_r) = {ks_full:.4f}   "
            f"[batch={batch}, steps={steps}, r={r}]")


# ============================================================================
# runner
# ============================================================================


# ============================================================================
# [A2.*] Appendix A.2 fixed-support tests (structured_asbs/_a2_tests.py)
# ============================================================================

from structured_asbs._a2_tests import A2_TESTS  # noqa: E402

for _name, _fn in A2_TESTS:
    test(_name)(_fn)


def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("structured-ASBS  Phase 1  --  mathematical unit tests")
    print(f"torch {torch.__version__}   device "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}   "
          f"fast={args.fast}")
    print("=" * 78)

    n_pass = n_fail = 0
    t_all = time.time()
    for name, fn in _REGISTRY:
        t0 = time.time()
        try:
            if "fast" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                msg = fn(fast=args.fast)
            else:
                msg = fn()
            dt = time.time() - t0
            print(f"PASS  {name}   ({dt:.2f}s)")
            if msg:
                print(f"      {msg}")
            n_pass += 1
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"FAIL  {name}   ({dt:.2f}s)")
            traceback.print_exc()
            print(f"      {exc}")
            n_fail += 1

    print("=" * 78)
    print(f"{n_pass} passed, {n_fail} failed   in {time.time() - t_all:.1f}s")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
