"""Appendix A.2 fixed-support deterministic tests (A2.1 - A2.14).

Imported by structured_asbs/tests_math.py; every test here must pass before any
fixed-support training run is started (Section 14/19 of the experiment spec).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.linalg import expm

from structured_asbs import fixed_support as FS

MICRO = dict(n=5, k=2, r=4)
GB1_M = "data/gb1/elife-16965-supp1-v4.xlsx"
GB1_I = "data/gb1/elife-16965-supp2-v4.xlsx"

_CACHE = {}


def space(tag):
    if tag in _CACHE:
        return _CACHE[tag]
    if tag == "micro":
        sp = FS.make_space(5, 2, 4, device="cpu")
    elif tag == "toy":
        sp = FS.make_space(14, 4, 4, device="cpu")
    elif tag == "gb1":
        sp = FS.make_space(4, 3, 20, device="cpu")
    else:
        raise KeyError(tag)
    _CACHE[tag] = sp
    return sp


def full_generator(sp):
    Q = np.zeros((sp.M, sp.M))
    rl = sp.alpha / sp.E_lab
    rs = sp.beta / sp.E_sup
    for x in range(sp.M):
        for e in range(sp.E):
            Q[x, sp.tgt[x, e]] += rl if e < sp.E_lab else rs
        Q[x, x] -= Q[x].sum()
    return Q


def a2_1_enumeration():
    sp = space("micro")
    assert sp.M == 90, sp.M
    S = sp.states
    assert ((S != 0).sum(1) == sp.k).all()
    assert len({tuple(v) for v in S}) == sp.M
    assert (sp.state_to_index_np(S) == np.arange(sp.M)).all()
    assert (sp.index_to_state(torch.arange(sp.M)).numpy() == S).all()
    assert sp.E == 22, sp.E
    assert sp.tgt.min() >= 0 and sp.tgt.max() < sp.M
    nb = sp.states[sp.tgt]
    assert ((nb != 0).sum(2) == sp.k).all()
    for x in range(sp.M):
        assert len(set(sp.tgt[x].tolist())) == sp.E, x
        assert x not in set(sp.tgt[x].tolist())
    return f"M={sp.M} edges/state={sp.E}"


def a2_2_full_generator():
    sp = space("micro")
    Q = full_generator(sp)
    off = Q - np.diag(np.diag(Q))
    assert np.abs(Q.sum(1)).max() < 1e-12
    assert np.abs(-np.diag(Q) - sp.gamma).max() < 1e-12
    assert off.min() >= 0.0
    return f"escape rate = {-Q[0,0]:.6f}"


def a2_3_orbit_valency():
    msg = []
    for tag in ("micro", "toy", "gb1"):
        sp = space(tag)
        assert abs(sp.N_ij.sum() - sp.M) < 1e-6
        cnt = np.bincount(sp.orbit_x0, minlength=sp.C).astype(float)
        assert np.allclose(cnt, sp.N_ij), tag
        msg.append(f"{tag}:C={sp.C}")
    return " ".join(msg)


def a2_4_commutation():
    msg = []
    cfgs = [("micro", 5, 2, 4), ("toy", 14, 4, 4), ("gb1", 4, 3, 20),
            ("product2", 8, 4, 20)]
    for tag, n, k, r in cfgs:
        _, _, N, Bl, Bs = FS.orbit_generators(n, k, r)
        d = np.max(np.abs(Bl @ Bs - Bs @ Bl))
        assert d < 1e-12, (tag, d)
        from math import comb
        assert abs(N.sum() - comb(n, k) * (r - 1) ** k) < 1e-3, tag
        msg.append(f"{tag}:{d:.1e}")
    return " ".join(msg)


def a2_5_kernel_vs_expm():
    sp = space("micro")
    Q = full_generator(sp)
    sig = sp.sigma_np(np.broadcast_to(sp.x0, (sp.M, sp.n)), sp.states)
    worst = 0.0
    for d in (0.05, 0.5, 1.0):
        P = expm(d * Q)
        row = P[sp.x0_idx]
        assert abs(row.sum() - 1.0) < 1e-12
        err = np.abs(row - sp.kappa(d)[sig]).max()
        worst = max(worst, err)
        assert err < 1e-10, (d, err)
    return f"max err {worst:.2e}"


def a2_6_local_edge_action():
    sp = space("micro")
    idx = torch.arange(sp.M)
    g = sp.edge_action_t[idx]
    y = sp.apply_action(sp.states_t[idx], g)
    assert (sp.state_to_index(y).numpy() == sp.tgt).all()
    # inverse round trip on every action
    for a in range(sp.A):
        perm = sp.act_perm[a]
        pinv = np.argsort(perm)
        z = sp.apply_action_np(sp.states, a)
        w = z[:, pinv]
        sa, sb = sp.act_sa[a][pinv], sp.act_sb[a][pinv]
        w = np.where(w == sa, sb, np.where(w == sb, sa, w))
        assert (w == sp.states).all(), a
    return f"{sp.M} states x {sp.E} edges, {sp.A} actions"


def a2_7_equivariance():
    sp = space("micro")
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.integers(0, sp.M, 4000))
    z = torch.as_tensor(rng.integers(0, sp.M, 4000))
    g = torch.as_tensor(rng.integers(0, sp.A, 4000))
    sx = sp.states_t[x]
    sz = sp.states_t[z]
    s0 = sp.sigma(sx, sz)
    s1 = sp.sigma(sp.apply_action(sx, g), sp.apply_action(sz, g))
    assert (s0 == s1).all()
    return "4000 random (x,z,g)"


def a2_8_canonical_actions():
    msg = []
    for tag, sel in (("micro", None), ("toy", 5000), ("gb1", None)):
        sp = space(tag)
        if sel is None:
            idx = torch.arange(sp.M)
        else:
            idx = torch.as_tensor(
                np.random.default_rng(1).integers(0, sp.M, sel))
        y = sp.states_t[idx]
        perm, sa, sb = sp.canon_action(y)
        x0 = torch.as_tensor(sp.x0.astype(np.int64))[None].expand_as(y)
        assert (sp.apply_pls(x0, perm, sa, sb) == x0).all(), tag
        yc = torch.as_tensor(sp.y_canon.astype(np.int64))[sp.orbit_x0_t[idx]]
        assert (sp.apply_pls(yc, perm, sa, sb) == y).all(), tag
        msg.append(f"{tag}:{idx.numel()}")
    return " ".join(msg)


def a2_9_chapman_kolmogorov():
    msg = []
    for tag, tol in (("micro", 1e-11), ("toy", 1e-10), ("gb1", 1e-10)):
        sp = space(tag)
        ka = np.repeat(np.arange(sp.C), sp.C)
        kb = np.tile(np.arange(sp.C), sp.C)
        worst = 0.0
        for t in (0.1, 0.37, 0.8):
            kt, k1 = sp.kappa(t), sp.kappa(1 - t)
            S = (sp.bridge_counts * kt[ka][None, :] * k1[kb][None, :]).sum(1)
            worst = max(worst, np.abs(S - sp.kappa(1.0)).max())
        assert worst < tol, (tag, worst)
        msg.append(f"{tag}:{worst:.1e}")
    return " ".join(msg)


def a2_10_empirical_bridge(fast=False):
    sp = space("micro")
    Q = full_generator(sp)
    t = 0.37
    Pt = expm(t * Q)
    P1t = expm((1 - t) * Q)
    P1 = expm(Q)
    nS = 20000 if fast else 100000
    steps = 100
    ti = torch.full((nS,), int(round(t * steps)))
    sp.build_kappa_grid(steps)
    ends, worst = [], 0.0
    seen = set()
    for y in range(sp.M):
        c = int(sp.orbit_x0[y])
        if c in seen:
            continue
        seen.add(c)
        ends.append(y)
        if len(ends) >= 4:
            break
    for y in ends:
        ex = Pt[sp.x0_idx] * P1t[:, y] / P1[sp.x0_idx, y]
        z = sp.sample_bridge(torch.full((nS,), y), ti)
        emp = np.bincount(z.numpy(), minlength=sp.M) / nS
        tv = 0.5 * np.abs(emp - ex).sum()
        worst = max(worst, tv)
        assert tv < 0.02, (y, tv)
    return f"{len(ends)} endpoints, max TV {worst:.4f} ({nS} samples)"


def a2_11_terminal_identity():
    sp = space("micro")
    Q = full_generator(sp)
    rng = np.random.default_rng(3)
    f1 = np.exp(rng.normal(0, 0.8, sp.M))
    t = 0.43
    P = expm((1 - t) * Q)
    phi = P @ f1
    worst = 0.0
    for x in rng.integers(0, sp.M, 12):
        for e in rng.integers(0, sp.E, 6):
            g = int(sp.edge_action[x, e])
            gx = int(sp.tgt[x, e])
            gy = sp.state_to_index_np(sp.apply_action_np(sp.states, g))
            lhs = phi[gx] / phi[x]
            rhs = float((P[x] * f1[gy]).sum() / phi[x])
            worst = max(worst, abs(lhs - rhs) / abs(lhs))
    assert worst < 1e-10, worst
    return f"max rel err {worst:.2e}"


def a2_12_nondirac_corrector():
    sp = space("micro")
    Q = full_generator(sp)
    P1 = expm(Q)
    src = sp.source_idx
    pb = P1[src]                                  # (4, M)
    fhat = pb.mean(0)
    rng = np.random.default_rng(4)
    worst = 0.0
    if not hasattr(sp, "pi"):
        FS.set_energy(sp, np.zeros(sp.M), 1.0)
    for _ in range(200):
        y = int(rng.integers(sp.M))
        g = int(rng.integers(sp.A))
        gy = int(sp.state_to_index_np(
            sp.apply_action_np(sp.states[y][None], g))[0])
        post = pb[:, y] / pb[:, y].sum()
        lhs = float((post * pb[:, gy] / pb[:, y]).sum())
        rhs = fhat[gy] / fhat[y]
        worst = max(worst, abs(lhs - rhs) / abs(rhs))
    # module-level exact diagnostic must be finite everywhere
    lc = FS.exact_corrector_ratio(
        sp, torch.arange(sp.M), torch.zeros(sp.M, 1, dtype=torch.long))
    assert torch.isfinite(lc).all()
    assert worst < 1e-10, worst
    return f"max rel err {worst:.2e}"


def a2_13_exact_propagation():
    sp = space("micro")
    if not hasattr(sp, "pi"):
        FS.set_energy(sp, np.zeros(sp.M), 1.0)
    for steps in (1, 8, 64):
        p = FS.propagate_exact(
            sp, lambda t: torch.zeros(sp.M, sp.E, dtype=torch.float64), steps)
        assert abs(float(p.sum()) - 1.0) < 1e-12, steps
        assert float(p.min()) >= -1e-15
        assert p.numel() == sp.M
    # one-jump-per-step discretisation converges to expm as steps grows
    Q = full_generator(sp)
    ex = expm(Q)[sp.x0_idx]
    d = []
    for steps in (16, 256):
        p = FS.propagate_exact(
            sp, lambda t: torch.zeros(sp.M, sp.E, dtype=torch.float64), steps)
        d.append(float(np.abs(p.numpy() - ex).sum()))
    assert d[1] < d[0], d
    return f"L1 to expm: {d[0]:.4f} (16 steps) -> {d[1]:.4f} (256 steps)"


def a2_14_gb1_loader():
    if not (os.path.exists(GB1_M) and os.path.exists(GB1_I)):
        return "SKIP (workbooks absent)"
    merged, nm, ni = FS.load_gb1(GB1_M, GB1_I, verbose=False)
    assert nm == 149361 and ni == 10639 and len(merged) == 160000
    alpha = FS.gb1_site_alphabets()
    for s in merged:
        assert FS.gb1_labels_to_seq(FS.gb1_seq_to_labels(s, alpha), alpha) == s
    sp = space("gb1")
    sector = [s for s in merged
              if sum(s[p] != FS.GB1_WT[p] for p in range(4)) == 3]
    assert len(sector) == 27436, len(sector)
    labs = np.array([FS.gb1_seq_to_labels(s, alpha) for s in sector],
                    dtype=np.int16)
    idx = sp.state_to_index_np(labs)
    assert sorted(idx.tolist()) == list(range(sp.M))
    back = [FS.gb1_labels_to_seq(v, alpha) for v in sp.states[idx]]
    assert back == sector
    return f"measured={nm} imputed={ni} merged={len(merged)} HD3={len(sector)}"


A2_TESTS = [
    ("[A2.1] fixed-support enumeration and edge closure", a2_1_enumeration),
    ("[A2.2] fixed-support full generator", a2_2_full_generator),
    ("[A2.3] orbit valency", a2_3_orbit_valency),
    ("[A2.4] orbit-generator commutation", a2_4_commutation),
    ("[A2.5] orbit kernel equals matrix exponential", a2_5_kernel_vs_expm),
    ("[A2.6] local edge action and inverses", a2_6_local_edge_action),
    ("[A2.7] action equivariance of sigma", a2_7_equivariance),
    ("[A2.8] canonical endpoint actions h_y", a2_8_canonical_actions),
    ("[A2.9] Chapman-Kolmogorov bridge classes", a2_9_chapman_kolmogorov),
    ("[A2.10] empirical bridge sampler", a2_10_empirical_bridge),
    ("[A2.11] terminal IASBS identity", a2_11_terminal_identity),
    ("[A2.12] non-Dirac corrector identity", a2_12_nondirac_corrector),
    ("[A2.13] exact controlled propagation", a2_13_exact_propagation),
    ("[A2.14] GB1 loader", a2_14_gb1_loader),
]
