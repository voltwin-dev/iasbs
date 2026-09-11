"""Mandatory mathematical tests for the radial Johnson reference.

Implements Tests A-N of ``CUAU_RADIAL_JOHNSON_IASBS_PLAYBOOK.md`` section 40 on
brute-force enumerable systems, using the section 41 helpers.  Nothing in the
CuAu pipeline is allowed to run until this module exits ``0``.

    python -m iasbs.tests_radial

The tests are deliberately independent of the CuAu energy: every object that
the production code needs from :class:`iasbs.cuau.CuAuSpace` and that is not
reference specific (``distance``, the four-block bridge tables) is borrowed
directly from that class by a tiny stand-in, so what is tested is the same code
that runs at ``N = 64``.
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import torch
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as C  # noqa: E402
from iasbs.cuau import CuAuSpace  # noqa: E402
from iasbs.cuau_radial import (PiecewiseShellSchedule, RadialController,  # noqa: E402
                               ShellCorrector,
                               apply_masks, apply_pairs,
                               corrector_label_shell, esp_sample,
                               log_esp_prefix, log_esp_select, log_q_shell,
                               named_schedule,
                               occupied_empty, propagate_exact_radial,
                               radial_rate_rows, sample_uniform_shell,
                               sel_mask, shell_head_value, shell_logits,
                               shell_move, simulate_radial)

FAIL = []
PASS = []


def report(name, ok, detail):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<46s} {detail}")


# ============================================================================
# section 41 brute-force helpers
# ============================================================================


def enumerate_fixed(n, k):
    out = []
    for occ in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=np.int8)
        x[list(occ)] = 1
        out.append(x)
    return np.stack(out)


def johnson_distance(x, y):
    k = int(x.sum())
    return k - int(np.dot(x, y))


def full_shell_matrix(states, j):
    M = len(states)
    K = np.zeros((M, M), dtype=np.float64)
    index = {tuple(x.tolist()): i for i, x in enumerate(states)}
    n = states.shape[1]
    k = int(states[0].sum())
    vj = C.shell_size(n, k, j)
    for a, x in enumerate(states):
        occ = np.flatnonzero(x == 1)
        emp = np.flatnonzero(x == 0)
        for R in itertools.combinations(occ, j):
            for A in itertools.combinations(emp, j):
                y = x.copy()
                y[list(R)] = 0
                y[list(A)] = 1
                K[a, index[tuple(y.tolist())]] += 1.0 / vj
    return K


def full_generator(states, clocks):
    M = len(states)
    Q = np.zeros((M, M), dtype=np.float64)
    for j, g in clocks.items():
        Q += float(g) * (full_shell_matrix(states, int(j)) - np.eye(M))
    return Q


def reduced_generator(n, k, clocks):
    D = min(k, n - k)
    G = np.zeros((D + 1, D + 1), dtype=np.float64)
    for j, g in clocks.items():
        G += float(g) * (C.shell_orbit_matrix_cached(n, k, int(j))
                         - np.eye(D + 1))
    return G


class TinySpace:
    """Reference-agnostic half of :class:`CuAuSpace` on an enumerable system."""

    _bridge_tables = CuAuSpace._bridge_tables
    distance = CuAuSpace.distance

    def __init__(self, n, k, x0=None, tau=0.05, schedule=None, device="cpu"):
        self.n, self.k, self.device = int(n), int(k), device
        if x0 is None:
            x0 = np.concatenate([np.ones(k, np.int64), np.zeros(n - k, np.int64)])
        self.x0 = torch.as_tensor(np.asarray(x0, dtype=np.int64), device=device)
        self.tau = float(tau)
        self.schedule = schedule
        self._bridge_tables()


# ============================================================================
# Tests A-E: the reference itself
# ============================================================================


def test_A():
    worst = 0
    for n, k in ((6, 3), (8, 4), (7, 3), (10, 5)):
        st = enumerate_fixed(n, k)
        D = min(k, n - k)
        for x in st:
            cnt = np.zeros(D + 1, dtype=np.int64)
            for y in st:
                cnt[johnson_distance(x, y)] += 1
            want = np.array([C.shell_size(n, k, j) for j in range(D + 1)])
            worst = max(worst, int(np.abs(cnt - want).max()))
    report("A shell cardinality", worst == 0, f"max |count - v_j| = {worst}")


def test_B():
    worst = 0.0
    for n, k in ((6, 3), (8, 4), (7, 3), (10, 5)):
        st = enumerate_fixed(n, k)
        D = min(k, n - k)
        S0 = st[0]
        dist = np.array([johnson_distance(S0, y) for y in st])
        for j in range(1, D + 1):
            K = full_shell_matrix(st, j)
            B = C.shell_orbit_matrix_cached(n, k, j)
            agg = np.zeros((D + 1, D + 1))
            seen = np.zeros(D + 1, dtype=bool)
            for i in range(len(st)):
                a = dist[i]
                row = np.bincount(dist, weights=K[i], minlength=D + 1)
                if seen[a]:
                    worst = max(worst, float(np.abs(agg[a] - row).max()))
                agg[a], seen[a] = row, True
            worst = max(worst, float(np.abs(agg - B).max()))
    report("B shell orbit matrix vs brute force", worst < 1e-13,
           f"max abs err = {worst:.3e}")


def test_C():
    rng = np.random.default_rng(0)
    worst_full, worst_red = 0.0, 0.0
    for n, k in ((6, 3), (8, 4), (10, 5)):
        st = enumerate_fixed(n, k)
        idx = {tuple(x.tolist()): i for i, x in enumerate(st)}
        D = min(k, n - k)
        Ks = {j: full_shell_matrix(st, j) for j in range(1, D + 1)}
        for _ in range(3):
            g = rng.permutation(n)
            P = np.zeros((len(st), len(st)))
            for i, x in enumerate(st):
                P[i, idx[tuple(x[g].tolist())]] = 1.0
            for K in Ks.values():
                worst_full = max(worst_full, float(np.abs(P @ K - K @ P).max()))
        Bs = [C.shell_orbit_matrix_cached(n, k, j) for j in range(1, D + 1)]
        for Bi in Bs:
            for Bj in Bs:
                worst_red = max(worst_red, float(np.abs(Bi @ Bj - Bj @ Bi).max()))
    report("C commutation (D_g K_j, B_i B_j)",
           worst_full < 1e-12 and worst_red < 1e-12,
           f"full {worst_full:.3e}  reduced {worst_red:.3e}")


def test_D():
    n, k = 6, 3
    clocks = {1: 0.7, 2: 1.1, 3: 0.4}
    st = enumerate_fixed(n, k)
    P = expm(full_generator(st, clocks))
    _, kap = C.radial_orbit_kernel(n, k, clocks)
    x0 = st[0]
    want = np.array([kap[johnson_distance(x0, y)] for y in st])
    err = float(np.abs(P[0] - want).max())
    report("D mixed radial kernel vs full generator", err < 1e-13,
           f"max abs err = {err:.3e}  (row sums {P[0].sum():.15f})")


def test_E():
    n, k = 6, 3
    ca, cb = {1: 0.5, 2: 0.3}, {2: 0.9, 3: 0.2}
    cc = {j: ca.get(j, 0.0) + cb.get(j, 0.0) for j in (1, 2, 3)}
    st = enumerate_fixed(n, k)
    Pa, Pb = expm(full_generator(st, ca)), expm(full_generator(st, cb))
    Pc = expm(full_generator(st, cc))
    e_full = float(np.abs(Pa @ Pb - Pc).max())
    Ma = expm(reduced_generator(n, k, ca))
    Mb = expm(reduced_generator(n, k, cb))
    Mc = expm(reduced_generator(n, k, cc))
    e_red = float(np.abs(Ma @ Mb - Mc).max())
    q, _ = C.radial_orbit_kernel(n, k, cc)
    e_q = float(np.abs(Mc[0] - q).max())
    report("E Chapman-Kolmogorov", max(e_full, e_red, e_q) < 1e-12,
           f"full {e_full:.3e}  reduced {e_red:.3e}  orbit {e_q:.3e}")


# ============================================================================
# Test F: bridge
# ============================================================================


def test_F():
    n, k, t = 6, 3, 0.4
    sch = PiecewiseShellSchedule((1, 2, 3), [0.0, 0.5, 1.0],
                                 [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]], 3.0)
    c0, c1 = sch.integrated(0.0, t), sch.integrated(t, 1.0)
    st = enumerate_fixed(n, k)
    P0, P1 = expm(full_generator(st, c0)), expm(full_generator(st, c1))
    _, kap0 = C.radial_orbit_kernel(n, k, c0)
    _, kap1 = C.radial_orbit_kernel(n, k, c1)

    sp = TinySpace(n, k, x0=st[0], schedule=sch)
    iS, iY = 0, 17
    S, Y = st[iS], st[iY]
    exact = P0[iS] * P1[:, iY]
    exact = exact / exact.sum()

    # class-table construction, expanded uniformly inside each class
    m = int(johnson_distance(S, Y))
    lk0 = torch.as_tensor(np.log(kap0), dtype=torch.float64)[None, :]
    lk1 = torch.as_tensor(np.log(kap1), dtype=torch.float64)[None, :]
    lw = (sp.br_lmult[m]
          + lk0[0][sp.br_d0[m]] + lk1[0][sp.br_d1[m]])
    w = torch.exp(lw - lw.max())
    w = (w / w.sum()).numpy()
    sizes = np.exp(sp.br_lmult[m].numpy())
    cls = np.zeros(len(st))
    for i, Z in enumerate(st):
        a = int(np.dot(S, Y) * 0)  # placeholder, computed below
        a = int(((S == 1) & (Y == 1) & (Z == 1)).sum())
        b = int(((S == 1) & (Y == 0) & (Z == 1)).sum())
        c = int(((S == 0) & (Y == 1) & (Z == 1)).sum())
        d = int(((S == 0) & (Y == 0) & (Z == 1)).sum())
        hit = np.flatnonzero((sp.br_abcd[m].numpy() == (a, b, c, d)).all(1))
        cls[i] = w[hit[0]] / sizes[hit[0]]
    e_exact = float(np.abs(cls - exact).max())

    # empirical
    NS = 1_000_000
    gen = torch.Generator().manual_seed(7)
    from iasbs.cuau import sample_bridge_direct
    Y1 = torch.as_tensor(np.tile(Y.astype(np.int64), (NS, 1)))
    tidx = torch.zeros(NS, dtype=torch.long)
    Z = sample_bridge_direct(sp, Y1, tidx, torch.log(
        torch.as_tensor(kap0, dtype=torch.float64))[None, :], torch.log(
        torch.as_tensor(kap1, dtype=torch.float64))[None, :], generator=gen)
    comp_ok = bool((Z.sum(1) == k).all())
    idx = {tuple(x.tolist()): i for i, x in enumerate(st)}
    emp = np.zeros(len(st))
    for row in Z.numpy():
        emp[idx[tuple(row.astype(np.int8).tolist())]] += 1
    emp /= NS
    se = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / NS)
    z = float(np.abs(emp - exact).max() / se.max())
    report("F exact bridge distribution",
           e_exact < 1e-13 and z < 5.0 and comp_ok,
           f"exact {e_exact:.3e}  empirical z = {z:.2f}  comp {comp_ok}")


# ============================================================================
# Tests G, H: the terminal identity
# ============================================================================


def _build_g(R, A, n):
    """Site permutation array for the involution swapping R_m <-> A_m."""
    g = np.arange(n)
    for r, a in zip(R, A):
        g[r], g[a] = a, r
    return g


def test_G():
    rng = np.random.default_rng(3)
    worst = 0.0
    ncase = 0
    for n, k in ((6, 3), (8, 4)):
        st = enumerate_fixed(n, k)
        idx = {tuple(x.tolist()): i for i, x in enumerate(st)}
        D = min(k, n - k)
        for trial in range(4):
            clocks = {j: float(rng.uniform(0.1, 1.2)) for j in range(1, D + 1)}
            P = expm(full_generator(st, clocks))
            f1 = np.exp(rng.normal(0.0, 1.5, size=len(st)))
            phi = P @ f1
            for _ in range(6):
                ix = int(rng.integers(len(st)))
                x = st[ix]
                j = int(rng.integers(1, D + 1))
                occ = np.flatnonzero(x == 1)
                emp = np.flatnonzero(x == 0)
                R = rng.choice(occ, j, replace=False)
                A = rng.choice(emp, j, replace=False)
                y = x.copy()
                y[R], y[A] = 0, 1
                g = _build_g(R, A, n)
                lhs = phi[idx[tuple(y.tolist())]] / phi[ix]
                pstar = P[ix] * f1 / phi[ix]
                gz = np.array([f1[idx[tuple(z[g].tolist())]] for z in st])
                rhs = float((pstar * gz / f1).sum())
                worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1.0))
                ncase += 1
    report("G full IASBS terminal identity", worst < 1e-12,
           f"max |LHS-RHS| = {worst:.3e} over {ncase} edges")


def test_H():
    rng = np.random.default_rng(11)
    n, k, j = 8, 4, 3
    st = enumerate_fixed(n, k)
    idx = {tuple(x.tolist()): i for i, x in enumerate(st)}
    clocks = {1: 0.6, 2: 0.4, 3: 0.5, 4: 0.2}
    P = expm(full_generator(st, clocks))
    f1 = np.exp(rng.normal(0.0, 1.2, size=len(st)))
    phi = P @ f1
    ix = 5
    x = st[ix]
    occ, emp = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
    R = occ[:j]
    A = emp[:j]
    y = x.copy()
    y[R], y[A] = 0, 1
    lhs = phi[idx[tuple(y.tolist())]] / phi[ix]
    pstar = P[ix] * f1 / phi[ix]
    rhss, spreads = [], []
    for perm in itertools.permutations(range(j)):
        g = _build_g(R, A[list(perm)], n)
        lam = np.array([f1[idx[tuple(z[g].tolist())]] / f1[idx[tuple(z.tolist())]]
                        for z in st])
        rhss.append(float((pstar * lam).sum()))
        spreads.append(float(lam.std()))
    err = max(abs(r - lhs) for r in rhss)
    differ = float(np.std(spreads)) > 1e-6
    report("H pairing invariance of conditional mean",
           err < 1e-12 and differ,
           f"max err {err:.3e}  samplewise labels differ = {differ}")


def test_G2():
    """The shipped ``terminal_log_label_shell`` equals ``log f1(gz)/f1(z)``."""
    from iasbs.cuau_radial import terminal_log_label_shell
    rng = np.random.default_rng(5)
    n, k, tau = 8, 4, 0.05
    st = enumerate_fixed(n, k)
    idx = {tuple(x.tolist()): i for i, x in enumerate(st)}
    Etab = rng.normal(0.0, 0.02, size=len(st))

    class FakeE:
        def energy_torch(self, X):
            key = [idx[tuple(r.tolist())] for r in X.to(torch.int8).numpy()]
            return torch.as_tensor(Etab[key], dtype=torch.float64)

    sp = TinySpace(n, k, x0=st[0], tau=tau)
    sp.energy = FakeE()
    clocks = {1: 0.6, 2: 0.4, 3: 0.5, 4: 0.2}
    _, kap = C.radial_orbit_kernel(n, k, clocks)
    sp.log_kappa_full = torch.as_tensor(np.log(kap), dtype=torch.float64)

    B, j = 64, 2
    ix = rng.integers(len(st), size=B)
    X1 = torch.as_tensor(st[ix].astype(np.int64))
    E1 = torch.as_tensor(Etab[ix])
    jrow = torch.full((B,), j, dtype=torch.long)
    gen = torch.Generator().manual_seed(1)
    R, A, valid = sample_uniform_shell(X1, jrow, generator=gen)
    out, Xg = terminal_log_label_shell(sp, X1, E1, R, A, valid)
    want = []
    for b in range(B):
        g = _build_g(R[b].numpy()[:j], A[b].numpy()[:j], n)
        z = st[ix[b]]
        zg = z[g]
        d1 = johnson_distance(st[0], z)
        dg = johnson_distance(st[0], zg)
        want.append(-(Etab[idx[tuple(zg.tolist())]] - Etab[ix[b]]) / tau
                    + np.log(kap[d1]) - np.log(kap[dg]))
    err = float(np.abs(out.numpy() - np.array(want)).max())
    comp = bool((Xg.sum(1) == k).all())
    report("G2 shipped terminal label vs formula", err < 1e-12 and comp,
           f"max err {err:.3e}  comp {comp}")


# ============================================================================
# Tests I, J: elementary symmetric polynomial machinery
# ============================================================================


def test_I():
    rng = np.random.default_rng(2)
    worst = 0.0
    for m in (3, 5, 8):
        logw = torch.as_tensor(rng.normal(0, 1.5, size=(4, m)))
        jmax = min(3, m)
        E = log_esp_prefix(logw, jmax)
        for j in range(jmax + 1):
            for b in range(4):
                tot = torch.logsumexp(torch.as_tensor(
                    [logw[b, list(s)].sum() for s in
                     itertools.combinations(range(m), j)] or [0.0]), 0)
                worst = max(worst, abs(float(E[b, m, j]) - float(tot)))
    report("I ESP normalizer vs enumeration", worst < 1e-12,
           f"max abs log err = {worst:.3e}")


def test_J():
    rng = np.random.default_rng(4)
    m, j, NS = 6, 3, 400_000
    logw = torch.as_tensor(rng.normal(0, 1.0, size=(1, m))).expand(NS, m).contiguous()
    jrow = torch.full((NS,), j, dtype=torch.long)
    gen = torch.Generator().manual_seed(9)
    sel = esp_sample(logw, jrow, generator=gen, E=log_esp_prefix(logw, j))
    ok_card = bool((sel.sum(1) == j).all())
    subs = list(itertools.combinations(range(m), j))
    key = {s: i for i, s in enumerate(subs)}
    lp = np.array([float(logw[0, list(s)].sum()) for s in subs])
    exact = np.exp(lp - lp.max())
    exact /= exact.sum()
    emp = np.zeros(len(subs))
    for row in sel.numpy():
        emp[key[tuple(np.flatnonzero(row).tolist())]] += 1
    emp /= NS
    # exact per-subset log prob from log_esp_select must match too
    lsel = log_esp_select(logw[:len(subs)], torch.as_tensor(
        np.stack([np.isin(np.arange(m), s) for s in subs])),
        torch.full((len(subs),), j, dtype=torch.long))
    e_sel = float(np.abs(np.exp(lsel.numpy()) - exact).max())
    se = np.sqrt(exact * (1 - exact) / NS)
    z = float(np.abs(emp - exact).max() / se.max())
    report("J ESP sampler", ok_card and z < 5.0 and e_sel < 1e-12,
           f"card {ok_card}  log-prob err {e_sel:.3e}  empirical z {z:.2f}")


# ============================================================================
# Tests K, L, M, N: controller, loss, simulator
# ============================================================================


@torch.no_grad()
def _controlled_rates(sp, net, x_np, shells, gam, clamp=10.0):
    """``{(j, dest_index): u_j}`` by explicit enumeration of every shell."""
    n, k = sp.n, sp.k
    x = torch.as_tensor(x_np.astype(np.int64))[None, :]
    t = torch.zeros(1, dtype=torch.float64)
    b, lw_rem, lw_add, occ, emp = shell_logits(net, t, x, clamp=clamp)
    lam = (torch.as_tensor(gam, dtype=torch.float64)[None, :] * torch.exp(b))[0]
    occ_np, emp_np = occ[0].numpy(), emp[0].numpy()
    out = {}
    for si, j in enumerate(shells):
        wr, wa = lw_rem[0, si], lw_add[0, si]
        Er = log_esp_prefix(wr[None, :], j)
        Ea = log_esp_prefix(wa[None, :], j)
        jr = torch.tensor([j])
        for Ri in itertools.combinations(range(k), j):
            selr = torch.zeros(1, k, dtype=torch.bool)
            selr[0, list(Ri)] = True
            lqr = log_esp_select(wr[None, :], selr, jr, E=Er)
            for Ai in itertools.combinations(range(n - k), j):
                sela = torch.zeros(1, n - k, dtype=torch.bool)
                sela[0, list(Ai)] = True
                lqa = log_esp_select(wa[None, :], sela, jr, E=Ea)
                y = x_np.copy()
                y[occ_np[list(Ri)]] = 0
                y[emp_np[list(Ai)]] = 1
                out[(j, tuple(y.tolist()))] = float(
                    lam[si] * torch.exp(lqr + lqa))
    return out, lam.numpy()


def _random_net(n, shells, seed=0, scale=0.35):
    torch.manual_seed(seed)
    net = RadialController(n, shells, hidden=64, n_freq=2).to(torch.float32)
    with torch.no_grad():
        net.net[-1].weight.normal_(0.0, scale)
        net.net[-1].bias.normal_(0.0, scale)
    return net


def test_K():
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    gam = np.array([1.0, 0.7, 0.4, 0.2])
    sp = TinySpace(n, k)
    net = _random_net(n, shells, seed=0)
    st = enumerate_fixed(n, k)
    worst = 0.0
    for ix in (0, 13, 41):
        u, lam = _controlled_rates(sp, net, st[ix].astype(np.int64), shells, gam)
        for si, j in enumerate(shells):
            tot = sum(v for (jj, _), v in u.items() if jj == j)
            worst = max(worst, abs(tot - lam[si]) / lam[si])
            assert len(shells) == 4
            nd = sum(1 for (jj, _) in u if jj == j)
            if nd != C.shell_size(n, k, j):
                worst = float("inf")
    report("K controller normalization", worst < 1e-12,
           f"max rel |sum_y u_j - lambda_j| = {worst:.3e}")


def test_L():
    rng = np.random.default_rng(6)
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    gam = np.array([1.0, 0.7, 0.4, 0.2])
    gtot = float(gam.sum())
    sp = TinySpace(n, k)
    net = _random_net(n, shells, seed=1)
    st = enumerate_fixed(n, k)
    x = st[7].astype(np.int64)
    u, lam = _controlled_rates(sp, net, x, shells, gam)
    keys = list(u.keys())
    lab = {kk: float(np.exp(rng.normal(0.0, 0.4))) for kk in keys}
    exact = float(lam.sum())
    for (j, y), uu in u.items():
        r = gam[shells.index(j)] / C.shell_size(n, k, j)
        exact -= r * lab[(j, y)] * math.log(uu / r)
    by_j = {j: [kk for kk in keys if kk[0] == j] for j in shells}
    NS = 400_000
    pj = gam / gtot
    acc = 0.0
    js = rng.choice(len(shells), NS, p=pj)
    for si in js:
        j = shells[si]
        kk = by_j[j][int(rng.integers(len(by_j[j])))]
        r = gam[si] / C.shell_size(n, k, j)
        acc += lam.sum() - gtot * lab[kk] * math.log(u[kk] / r)
    mc = acc / NS
    rel = abs(mc - exact) / max(abs(exact), 1.0)
    report("L sampled Bregman unbiasedness", rel < 5e-3,
           f"exact {exact:.6f}  MC {mc:.6f}  rel {rel:.2e}")


def test_M():
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    gtot, dt, NS = 1.0, 0.02, 400_000
    sch = PiecewiseShellSchedule(shells, [0.0, 1.0],
                                 [[0.45, 0.30, 0.17, 0.08]], gtot)
    gam = np.array(sch.rates(0.0))
    sp = TinySpace(n, k, schedule=sch)
    net = _random_net(n, shells, seed=2, scale=0.25)
    st = enumerate_fixed(n, k)
    idx = {tuple(x.tolist()): i for i, x in enumerate(st)}
    ix = 0
    x = st[ix].astype(np.int64)

    # full controlled generator by explicit enumeration
    Qth = np.zeros((len(st), len(st)))
    for a, xa in enumerate(st):
        ua, _ = _controlled_rates(sp, net, xa.astype(np.int64), shells, gam)
        for (_, y), uu in ua.items():
            Qth[a, idx[tuple(y)]] += uu
        Qth[a, a] -= Qth[a].sum()
    exact = expm(dt * Qth)[ix]

    # one bin of length dt: a single-step schedule at the same instantaneous
    # rates, so ``simulate_radial`` runs exactly one frozen-rate bin
    sp1 = TinySpace(n, k, schedule=PiecewiseShellSchedule(
        shells, [0.0, 1.0], [list(gam / gtot)], gtot * dt))
    gen = torch.Generator().manual_seed(13)
    x0 = torch.as_tensor(x)[None, :].expand(NS, n).contiguous()
    stats = {}
    Z = simulate_radial(sp1, net, NS, 1, generator=gen, x0=x0, stats=stats)
    comp = bool((Z.sum(1) == k).all())
    emp = np.zeros(len(st))
    for r in Z.numpy():
        emp[idx[tuple(r.astype(np.int8).tolist())]] += 1
    emp /= NS
    se = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / NS)
    dev = float(np.abs(emp - exact).max())
    tol = float(5 * se.max()) + stats["max_two_event_prob"]
    report("M exact controlled simulation vs expm",
           dev < tol and comp,
           f"max dev {dev:.2e} < tol {tol:.2e}  P(>=2 ev) "
           f"{stats['max_two_event_prob']:.2e}  comp {comp}")


def test_N():
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    # coarse-to-fine 4 -> 3 -> 2 -> 1 with a 5% j=1 floor, the S2 pattern
    # rescaled to the shells that exist at (n, k) = (8, 4)
    sch = PiecewiseShellSchedule(
        shells, [0.0, 0.25, 0.5, 0.75, 1.0],
        [[0.05, 0.0, 0.0, 0.95], [0.05, 0.0, 0.95, 0.0],
         [0.05, 0.95, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], 4.0)
    sp = TinySpace(n, k, schedule=sch)
    net = _random_net(n, shells, seed=3)
    st = enumerate_fixed(n, k)
    rng = np.random.default_rng(8)
    B = 512
    X = torch.as_tensor(st[rng.integers(len(st), size=B)].astype(np.int64))
    ok = [bool((X.sum(1) == k).all())]
    gen = torch.Generator().manual_seed(21)
    for j in shells:
        jrow = torch.full((B,), j, dtype=torch.long)
        R, A, valid = sample_uniform_shell(X, jrow, generator=gen)
        Y = shell_move(X, R, A, valid)
        Yp = apply_pairs(X, R, A, valid)
        ok.append(bool((Y.sum(1) == k).all()))
        ok.append(bool((Y == Yp).all()))
        ok.append(bool((sp.distance(X, Y) == j).all()))
        occ, emp = occupied_empty(X)
        selr = torch.zeros(B, k, dtype=torch.bool)
        sela = torch.zeros(B, n - k, dtype=torch.bool)
        selr[:, :j] = True
        sela[:, :j] = True
        Ym = apply_masks(X, occ, emp, selr, sela)
        ok.append(bool((Ym.sum(1) == k).all()))
    Zc = simulate_radial(sp, net, B, 32, generator=gen, x0=X, stats={})
    ok.append(bool((Zc.sum(1) == k).all()))
    report("N composition preserved everywhere", all(ok),
           f"{sum(ok)}/{len(ok)} checks; shell_move == apply_pairs")


def test_O():
    """The training code path itself, against the enumerated Bregman loss.

    Test L validated the *formula*; this validates the *implementation* used by
    ``cmd_train``: ``sel_mask`` -> ``log_q_shell`` -> ``log m = b + log v + log q``
    and the sampled estimator assembled exactly as in the inner loop.
    """
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    gtot = 2.3
    sch = PiecewiseShellSchedule(shells, [0.0, 1.0],
                                 [[0.4, 0.3, 0.2, 0.1]], gtot)
    dev = "cpu"
    torch.manual_seed(11)
    net = RadialController(n, shells, hidden=32).to(dev)
    with torch.no_grad():
        for p in net.net[-1].parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    t = 0.37
    st = enumerate_fixed(n, k)
    x = torch.as_tensor(st[9].astype(np.int64), device=dev)[None, :]
    tt = torch.full((1,), t, dtype=torch.float64, device=dev)
    gam = torch.as_tensor(sch.rates(t), dtype=torch.float64, device=dev)[None, :]
    b, lw_rem, lw_add, occ, emp = shell_logits(net, tt, x)
    lam = (gam * torch.exp(b)).squeeze(0)
    logv = torch.as_tensor([math.log(C.shell_size(n, k, j)) for j in shells],
                           dtype=torch.float64, device=dev)

    # exact enumeration of q_j and of the loss
    xs = x[0].numpy()
    lab = {}
    rng = np.random.default_rng(3)
    exact = float(lam.sum())
    qsum = {}
    for si, j in enumerate(shells):
        tot = 0.0
        acc = 0.0
        for y in st:
            if int(k - (xs * y).sum()) != j:
                continue
            yt = torch.as_tensor(y.astype(np.int64), device=dev)[None, :]
            R = torch.nonzero((x == 1) & (yt == 0))[:, 1][None, :]
            A = torch.nonzero((x == 0) & (yt == 1))[:, 1][None, :]
            v = torch.ones_like(R, dtype=torch.bool)
            jr = torch.full((1,), j, dtype=torch.long, device=dev)
            sr = sel_mask(R, v, occ, n)
            sa = sel_mask(A, v, emp, n)
            lq = float(log_q_shell(lw_rem, lw_add, sr, sa, jr,
                                   torch.full((1,), si, dtype=torch.long)))
            tot += math.exp(lq)
            key = (j, tuple(y.tolist()))
            lab[key] = float(np.exp(rng.normal(0.0, 0.3)))
            log_m = float(b[0, si]) + float(logv[si]) + lq
            r = float(gam[0, si]) / C.shell_size(n, k, j)
            acc += r * lab[key] * log_m
        qsum[j] = tot
        exact -= acc
    qerr = max(abs(v - 1.0) for v in qsum.values())

    # sampled estimator, assembled as in cmd_train
    NS = 300_000
    pj = (gam[0] / gam[0].sum()).numpy()
    sis = rng.choice(len(shells), NS, p=pj)
    by_j = {j: [y for y in st if int(k - (xs * y).sum()) == j] for j in shells}
    acc = 0.0
    lam_sum = float(lam.sum())
    for si in sis:
        j = shells[si]
        y = by_j[j][int(rng.integers(len(by_j[j])))]
        yt = torch.as_tensor(y.astype(np.int64), device=dev)[None, :]
        R = torch.nonzero((x == 1) & (yt == 0))[:, 1][None, :]
        A = torch.nonzero((x == 0) & (yt == 1))[:, 1][None, :]
        v = torch.ones_like(R, dtype=torch.bool)
        jr = torch.full((1,), j, dtype=torch.long, device=dev)
        lq = float(log_q_shell(lw_rem, lw_add, sel_mask(R, v, occ, n),
                               sel_mask(A, v, emp, n), jr,
                               torch.full((1,), si, dtype=torch.long)))
        log_m = float(b[0, si]) + float(logv[si]) + lq
        acc += lam_sum - gtot * lab[(j, tuple(y.tolist()))] * log_m
    mc = acc / NS
    rel = abs(mc - exact) / max(abs(exact), 1.0)
    report("O train-path loss vs enumeration",
           qerr < 1e-12 and rel < 5e-3,
           f"sum_y q_j err {qerr:.3e}  exact {exact:.6f}  MC {mc:.6f}  "
           f"rel {rel:.2e}")


def test_P():
    """``radial_rate_rows`` against the enumerated controlled generator.

    The dense evaluation row is a different algorithm from the sampler: it gets
    ``sum_{i in R} w^-_i`` from a matrix product against the whole enumeration
    instead of a gather, and it reads the shell off the Johnson distance instead
    of drawing it.  It must agree with the explicit shell enumeration of
    ``_controlled_rates`` to machine precision, entry by entry.
    """
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    gtot = 1.7
    sch = PiecewiseShellSchedule(shells, [0.0, 1.0],
                                 [[0.4, 0.3, 0.2, 0.1]], gtot)
    sp = TinySpace(n, k, schedule=sch)
    st = enumerate_fixed(n, k)
    sp.S = torch.as_tensor(st.astype(np.int64))
    sp.M = len(st)
    net = _random_net(n, shells, seed=5)
    gam = torch.as_tensor(np.array(sch.rates(0.0)))
    imap = {tuple(x.tolist()): i for i, x in enumerate(st)}
    U = radial_rate_rows(sp, net, 0.0, torch.arange(sp.M))
    diag = float(U[torch.arange(sp.M), torch.arange(sp.M)].abs().max())

    # the reference enumeration must read the SAME controller forward pass: the
    # float32 head is only reproducible at a fixed batch shape, so re-invoking
    # it per state would compare GEMM blocking noise (~1e-7) rather than the two
    # rate algorithms.
    tz = torch.zeros(sp.M, dtype=torch.float64)
    b, lw_rem, lw_add, occ, emp = shell_logits(net, tz, sp.S)
    lam = gam[None, :] * torch.exp(b)
    worst = 0.0
    for a in (0, 7, 33, 69):
        occ_np, emp_np = occ[a].numpy(), emp[a].numpy()
        ref = np.zeros(sp.M)
        for si, j in enumerate(shells):
            wr, wa = lw_rem[a, si][None, :], lw_add[a, si][None, :]
            Er, Ea = log_esp_prefix(wr, j), log_esp_prefix(wa, j)
            jr = torch.tensor([j])
            for Ri in itertools.combinations(range(k), j):
                selr = torch.zeros(1, k, dtype=torch.bool)
                selr[0, list(Ri)] = True
                lqr = log_esp_select(wr, selr, jr, E=Er)
                for Ai in itertools.combinations(range(n - k), j):
                    sela = torch.zeros(1, n - k, dtype=torch.bool)
                    sela[0, list(Ai)] = True
                    lqa = log_esp_select(wa, sela, jr, E=Ea)
                    y = st[a].astype(np.int64).copy()
                    y[occ_np[list(Ri)]] = 0
                    y[emp_np[list(Ai)]] = 1
                    ref[imap[tuple(y.tolist())]] += float(
                        lam[a, si] * torch.exp(lqr + lqa))
        worst = max(worst, float(np.abs(U[a].numpy() - ref).max() / ref.max()))
    tot = float((U.sum(dim=1) - lam.sum(dim=1)).abs().max())
    report("P dense rate rows vs enumeration",
           worst < 1e-12 and diag == 0.0 and tot < 1e-12,
           f"max rel dev {worst:.3e}  self-rate {diag:.1e}  "
           f"max |sum_y u - sum_j lambda_j| {tot:.3e}")


def test_Q():
    """``propagate_exact_radial`` against ``simulate_radial``.

    Two independent implementations of the same one-jump-per-bin chain: the
    propagator moves a law with dense rate rows, the sampler draws a shell and
    then an exact fixed-cardinality subset.  Agreement is checked at 5 binomial
    sigma on every state of the sector.
    """
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    steps, NS = 8, 400_000
    sch = PiecewiseShellSchedule(shells, [0.0, 0.5, 1.0],
                                 [[0.1, 0.2, 0.3, 0.4],
                                  [0.7, 0.2, 0.1, 0.0]], 2.5)
    sp = TinySpace(n, k, schedule=sch)
    st = enumerate_fixed(n, k)
    sp.S = torch.as_tensor(st.astype(np.int64))
    sp.M = M = len(st)
    net = _random_net(n, shells, seed=9, scale=0.3)
    p0 = torch.full((M,), 1.0 / M, dtype=torch.float64)
    p = propagate_exact_radial(sp, net, steps, p0=p0, chunk=13).numpy()

    gen = torch.Generator().manual_seed(21)
    i0 = torch.randint(M, (NS,), generator=gen)
    x0 = sp.S[i0]
    stats = {}
    Z = simulate_radial(sp, net, NS, steps, generator=gen, x0=x0, stats=stats)
    imap = {tuple(x.tolist()): i for i, x in enumerate(st)}
    emp = np.zeros(M)
    for r in Z.numpy():
        emp[imap[tuple(r.tolist())]] += 1
    emp /= NS
    se = np.sqrt(np.maximum(p * (1 - p), 1e-12) / NS)
    z = float(np.abs(emp - p).max() / se.max())
    report("Q exact propagation vs sampler",
           z < 5.0 and abs(p.sum() - 1.0) < 1e-12,
           f"max |emp - exact| {np.abs(emp - p).max():.2e}  z {z:.2f}  "
           f"sum p - 1 = {p.sum() - 1.0:.1e}")


def test_R():
    """The non-Dirac corrector identity, by enumeration (playbook 13).

    With a tilted source ``nu_0 ~ w`` the exact ratio is
    ``fhat_1(z) = sum_{x0} kappa(d(x0,z)) w(x0)`` evaluated at ``g X_1`` over
    ``X_1``, and the single-sample estimator ``Q`` must reproduce it when ``X_0``
    is drawn from the endpoint conditional ``p*(x0 | X1) ~ kappa(d) w``.  The
    expectation is taken by summing over the whole sector, so this is an exact
    identity check, not a Monte Carlo one.
    """
    n, k = 8, 4
    shells = (1, 2, 3, 4)
    sch = PiecewiseShellSchedule(shells, [0.0, 1.0],
                                 [[0.4, 0.3, 0.2, 0.1]], 1.9)
    sp = TinySpace(n, k, schedule=sch)
    st = enumerate_fixed(n, k)
    M = len(st)
    _, kap = C.radial_orbit_kernel(n, k, sch.integrated(0.0, 1.0))
    sp.log_kappa_full = torch.as_tensor(np.log(kap), dtype=torch.float64)
    S = torch.as_tensor(st.astype(np.int64))
    Dm = (k - S.to(torch.float64) @ S.to(torch.float64).T).round().long()
    rng = np.random.default_rng(17)
    w = torch.as_tensor(np.exp(rng.normal(0.0, 0.8, M)))
    Kap = torch.as_tensor(kap)[Dm]
    fhat = Kap @ w
    gen = torch.Generator().manual_seed(31)
    imap = {tuple(x.tolist()): i for i, x in enumerate(st)}
    worst = 0.0
    for i1 in (3, 22, 55):
        X1 = S[i1][None, :]
        pw = Kap[:, i1] * w
        pw = pw / pw.sum()
        for j in shells:
            R, A, valid = sample_uniform_shell(
                X1, torch.tensor([j]), generator=gen)
            lq, Xg = corrector_label_shell(
                sp, S, X1.expand(M, -1), R.expand(M, -1), A.expand(M, -1),
                valid.expand(M, -1))
            got = float((pw * torch.exp(lq)).sum())
            want = float(fhat[imap[tuple(Xg[0].tolist())]] / fhat[i1])
            worst = max(worst, abs(got - want) / want)
    # zero-initialised head is the exact IPF-1 answer h = 0
    hnet = ShellCorrector(n, shells, hidden=16)
    R, A, valid = sample_uniform_shell(S[:4], torch.tensor([1, 2, 3, 4]),
                                       generator=gen)
    h0 = float(shell_head_value(hnet, torch.zeros(4, dtype=torch.float64),
                                S[:4], R, A, valid,
                                torch.tensor([0, 1, 2, 3])).abs().max())
    report("R corrector estimator identity", worst < 1e-12 and h0 == 0.0,
           f"max rel dev {worst:.3e}  |h| at init {h0:.1e}")


def main():
    for f in (test_A, test_B, test_C, test_D, test_E, test_F, test_G, test_G2,
              test_H, test_I, test_J, test_K, test_L, test_M, test_N, test_O,
              test_P, test_Q, test_R):
        try:
            f()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            report(f.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
