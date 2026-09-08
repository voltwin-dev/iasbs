"""Appendix A.2: fixed-support discrete state space.

State space
    X^(r)_{n,k} = { x in {0,...,r-1}^n : #{i : x_i != 0} = k },   |X| = C(n,k)(r-1)^k

Label 0 means "inactive" (wild type).  Two legal move families:

    relabel : change the label of one active site,   E_lab = k(r-2)
    support : move activity from one site to another, E_sup = k(n-k)(r-1)

Reference rates are chosen as alpha = gamma E_lab / E, beta = gamma E_sup / E so
that every legal edge carries the identical rate gamma / E and every state has
escape rate exactly gamma.

This module mirrors structured_asbs/fixed_ising.py in structure: an exact space
object with dense (M, E) neighbour tables, an orbit kernel built by matrix
exponential on the small orbit lattice, a canonical-source bridge sampler that
never materialises an M x M matrix, and exact forward propagation for
evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import combinations, product
from math import comb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C  # noqa: E402

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TINY = 1e-300


def orbit_generators(n, k, r):
    """Orbit lattice, valencies and commuting generators for X^(r)_{n,k}.

    Pure combinatorics: never enumerates the state space, so it is usable for
    configurations far too large to enumerate (Section 27 product targets).
    """
    orbits = [(i, j) for j in range(min(k, n - k) + 1)
              for i in range(k - j + 1)]
    Cn = len(orbits)
    lut = np.full((k + 1, k + 1), -1, dtype=np.int64)
    for c, (i, j) in enumerate(orbits):
        lut[i, j] = c
    N_ij = np.array([comb(k, j) * comb(n - k, j) * (r - 1) ** j
                     * comb(k - j, i) * (r - 2) ** i for (i, j) in orbits],
                    dtype=np.float64)
    B_lab = np.zeros((Cn, Cn))
    B_sup = np.zeros((Cn, Cn))
    D = k * (n - k) * (r - 1)

    def add(Mx, c, i2, j2, val):
        if i2 < 0 or j2 < 0 or j2 > min(k, n - k) or i2 > k - j2:
            return
        Mx[c, lut[i2, j2]] += val

    for c, (i, j) in enumerate(orbits):
        s = k - i - j
        z = n - k - j
        add(B_lab, c, i + 1, j, s / k)
        add(B_lab, c, i - 1, j, i / (k * (r - 2)))
        add(B_sup, c, i + 1, j, s * j * (r - 2) / D)
        add(B_sup, c, i - 1, j, i * j / D)
        add(B_sup, c, i, j + 1, s * z / (k * (n - k)))
        add(B_sup, c, i - 1, j + 1, i * z / (k * (n - k)))
        add(B_sup, c, i, j - 1, j * j / D)
        add(B_sup, c, i + 1, j - 1, j * j * (r - 2) / D)
    for Mx in (B_lab, B_sup):
        np.fill_diagonal(Mx, 0.0)
        np.fill_diagonal(Mx, -Mx.sum(1))
    assert np.max(np.abs(B_lab.sum(1))) < 1e-12
    assert np.max(np.abs(B_sup.sum(1))) < 1e-12
    assert (B_lab - np.diag(np.diag(B_lab))).min() >= 0
    assert (B_sup - np.diag(np.diag(B_sup))).min() >= 0
    assert np.max(np.abs(B_lab @ B_sup - B_sup @ B_lab)) < 1e-12
    return orbits, lut, N_ij, B_lab, B_sup



# ----------------------------------------------------------------------------
# state space
# ----------------------------------------------------------------------------
class FixedSupportSpace:
    """Enumerated fixed-support space with exact rank map and edge tables."""

    def __init__(self, n: int, k: int, r: int, gamma: float = 10.0,
                 device: str = "cpu", build_edges: bool = True):
        assert 0 < k < n and r >= 3
        self.n, self.k, self.r = n, k, r
        self.gamma = float(gamma)
        self.device = torch.device(device)

        self.supports = list(combinations(range(n), k))
        self.n_supports = len(self.supports)
        self.L = (r - 1) ** k
        self.M = self.n_supports * self.L
        assert self.M == comb(n, k) * (r - 1) ** k

        # ---- enumerate states -------------------------------------------
        labels = np.array(list(product(range(1, r), repeat=k)), dtype=np.int16)
        assert labels.shape == (self.L, k)
        states = np.zeros((self.M, n), dtype=np.int16)
        for si, sup in enumerate(self.supports):
            states[si * self.L:(si + 1) * self.L, list(sup)] = labels
        self.states = states

        # ---- exact rank map ---------------------------------------------
        self.pow2 = (1 << np.arange(n)).astype(np.int64)
        self.lab_w = ((r - 1) ** (k - 1 - np.arange(k))).astype(np.int64)
        lut = np.full(1 << n, -1, dtype=np.int64)
        for si, sup in enumerate(self.supports):
            lut[int(sum(1 << p for p in sup))] = si
        self.support_rank_lut = lut

        # ---- edge bookkeeping -------------------------------------------
        self.E_lab = k * (r - 2)
        self.E_sup = k * (n - k) * (r - 1)
        self.E = self.E_lab + self.E_sup
        self.alpha = gamma * self.E_lab / self.E
        self.beta = gamma * self.E_sup / self.E
        self.edge_rate = gamma / self.E

        # ---- global action ids -------------------------------------------
        self.P = comb(r - 1, 2)
        pair_lut = np.full((r, r), -1, dtype=np.int64)
        cnt = 0
        for u in range(1, r):
            for v in range(u + 1, r):
                pair_lut[u, v] = cnt
                cnt += 1
        assert cnt == self.P
        self.pair_lut = pair_lut
        self.A_lab = n * self.P
        self.A_sup = n * (n - 1) * (1 + self.P)
        self.A = self.A_lab + self.A_sup
        op = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                op[i, j] = i * (n - 1) + (j - (1 if j > i else 0))
        self.op_idx = op

        self.out_dim = n * (r - 1) + n * n * (r - 1)

        # torch mirrors
        dev = self.device
        self.pow2_t = torch.as_tensor(self.pow2, device=dev)
        self.lab_w_t = torch.as_tensor(self.lab_w, device=dev)
        self.support_lut_t = torch.as_tensor(lut, device=dev)
        self.states_t = torch.as_tensor(states.astype(np.int64), device=dev)
        # adapter aliases (Section 17)
        self.S = self.states_t
        self.n_edges = self.E

        self._build_actions()
        if build_edges:
            self._build_edges()

    # -- rank map ---------------------------------------------------------
    def state_to_index_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        B = x.shape[0]
        mask = ((x != 0).astype(np.int64) * self.pow2).sum(1)
        srank = self.support_rank_lut[mask]
        nz = np.nonzero(x)[1].reshape(B, self.k)
        vals = x[np.arange(B)[:, None], nz].astype(np.int64)
        lrank = ((vals - 1) * self.lab_w).sum(1)
        return srank * self.L + lrank

    def state_to_index(self, y: torch.Tensor) -> torch.Tensor:
        """y : (..., n) long tensor with exactly k nonzeros per row."""
        shp = y.shape[:-1]
        y = y.reshape(-1, self.n)
        mask = ((y != 0).long() * self.pow2_t).sum(1)
        srank = self.support_lut_t[mask]
        vals = y[y != 0].reshape(-1, self.k)
        lrank = ((vals - 1) * self.lab_w_t).sum(1)
        return (srank * self.L + lrank).reshape(shp)

    def index_to_state(self, idx: torch.Tensor) -> torch.Tensor:
        return self.states_t[idx]

    # -- actions ----------------------------------------------------------
    def _build_actions(self):
        n, r, P = self.n, self.r, self.P
        A = self.A
        perm = np.tile(np.arange(n, dtype=np.int64), (A, 1))
        sa = np.full((A, n), -1, dtype=np.int64)
        sb = np.full((A, n), -1, dtype=np.int64)
        pairs = [(u, v) for u in range(1, r) for v in range(u + 1, r)]
        for s in range(n):
            for pi, (u, v) in enumerate(pairs):
                a = s * P + pi
                sa[a, s], sb[a, s] = u, v
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                base = self.A_lab + self.op_idx[i, j] * (1 + P)
                for la in range(1 + P):
                    a = base + la
                    perm[a, i], perm[a, j] = j, i
                    if la > 0:
                        u, v = pairs[la - 1]
                        sa[a, j], sb[a, j] = u, v
        self.act_perm = perm
        self.act_sa = sa
        self.act_sb = sb
        dev = self.device
        self.act_perm_t = torch.as_tensor(perm, device=dev)
        self.act_sa_t = torch.as_tensor(sa, device=dev)
        self.act_sb_t = torch.as_tensor(sb, device=dev)

    def apply_action(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x : (B, n) long, g : (B,) or (B, A2) long -> (B, n) or (B, A2, n)."""
        if g.dim() == 1:
            p = self.act_perm_t[g]
            y = torch.gather(x, 1, p)
            a = self.act_sa_t[g]
            b = self.act_sb_t[g]
        else:
            B, A2 = g.shape
            p = self.act_perm_t[g]                     # (B, A2, n)
            xe = x.unsqueeze(1).expand(B, A2, self.n)
            y = torch.gather(xe, 2, p)
            a = self.act_sa_t[g]
            b = self.act_sb_t[g]
        y = torch.where(y == a, b, torch.where(y == b, a, y))
        return y

    def apply_action_np(self, x: np.ndarray, g: int) -> np.ndarray:
        y = x[..., self.act_perm[g]]
        a, b = self.act_sa[g], self.act_sb[g]
        y = np.where(y == a, b, np.where(y == b, a, y))
        return y

    # -- edges ------------------------------------------------------------
    def _build_edges(self, chunk: int = 8192):
        n, k, r = self.n, self.k, self.r
        E_lab, E_sup, E, M, P = self.E_lab, self.E_sup, self.E, self.M, self.P
        tgt = np.zeros((M, E), dtype=np.int32)
        act = np.zeros((M, E), dtype=np.int32)
        out = np.zeros((M, E), dtype=np.int32)
        cvec = np.arange(r - 2, dtype=np.int64)
        bvec = np.arange(1, r, dtype=np.int64)
        for s0 in range(0, M, chunk):
            s1 = min(M, s0 + chunk)
            x = self.states[s0:s1].astype(np.int64)
            B = s1 - s0
            asite = np.nonzero(x)[1].reshape(B, k)
            isite = np.nonzero(x == 0)[1].reshape(B, n - k)
            alab = np.take_along_axis(x, asite, 1)              # (B, k)

            # relabel edges
            newl = cvec[None, None, :] + 1
            newl = newl + (newl >= alab[:, :, None])            # (B,k,r-2)
            yl = np.broadcast_to(x[:, None, None, :], (B, k, r - 2, n)).copy()
            si_l = np.broadcast_to(asite[:, :, None], (B, k, r - 2))
            np.put_along_axis(yl, si_l[..., None], newl[..., None], axis=3)
            u = np.minimum(alab[:, :, None], newl)
            v = np.maximum(alab[:, :, None], newl)
            act[s0:s1, :E_lab] = (si_l * P + self.pair_lut[u, v]).reshape(B, E_lab)
            out[s0:s1, :E_lab] = (si_l * (r - 1) + (newl - 1)).reshape(B, E_lab)

            # support edges
            si = asite[:, :, None, None]
            dj = isite[:, None, :, None]
            bl = bvec[None, None, None, :]
            si_b, dj_b, bl_b = np.broadcast_arrays(
                si, dj, bl)                                     # (B,k,n-k,r-1)
            ys = np.broadcast_to(
                x[:, None, None, None, :], (B, k, n - k, r - 1, n)).copy()
            np.put_along_axis(ys, si_b[..., None].copy(),
                              np.zeros(si_b.shape + (1,), dtype=np.int64), axis=4)
            np.put_along_axis(ys, dj_b[..., None].copy(),
                              bl_b[..., None].copy(), axis=4)
            ab = np.broadcast_to(alab[:, :, None, None], si_b.shape)
            uu = np.minimum(ab, bl_b)
            vv = np.maximum(ab, bl_b)
            la = np.where(ab == bl_b, 0, 1 + self.pair_lut[uu, vv])
            act[s0:s1, E_lab:] = (
                self.A_lab + self.op_idx[si_b, dj_b] * (1 + P) + la
            ).reshape(B, E_sup)
            out[s0:s1, E_lab:] = (
                n * (r - 1) + (si_b * n + dj_b) * (r - 1) + (bl_b - 1)
            ).reshape(B, E_sup)

            allst = np.concatenate(
                [yl.reshape(B * E_lab, n), ys.reshape(B * E_sup, n)], 0)
            gi = self.state_to_index_np(allst)
            assert gi.min() >= 0
            tgt[s0:s1, :E_lab] = gi[:B * E_lab].reshape(B, E_lab)
            tgt[s0:s1, E_lab:] = gi[B * E_lab:].reshape(B, E_sup)

        self.tgt = tgt
        self.edge_action = act
        self.edge_out = out
        dev = self.device
        self.tgt_t = torch.as_tensor(tgt.astype(np.int64), device=dev)
        self.edge_action_t = torch.as_tensor(act.astype(np.int64), device=dev)
        self.edge_out_t = torch.as_tensor(out.astype(np.int64), device=dev)


    # -- orbits ------------------------------------------------------------
    def build_orbits(self):
        n, k, r = self.n, self.k, self.r
        orbits, lut, N_ij, B_lab, B_sup = orbit_generators(n, k, r)
        self.orbits, self.C = orbits, len(orbits)
        C = len(orbits)  # local shadow of the module alias, on purpose
        self.orbit_lut = lut
        self.orbit_lut_t = torch.as_tensor(lut, device=self.device)
        self.N_ij = N_ij
        self.B_lab, self.B_sup = B_lab, B_sup
        assert abs(self.N_ij.sum() - self.M) < 1e-6, (self.N_ij.sum(), self.M)

        # canonical endpoints
        yc = np.zeros((C, n), dtype=np.int16)
        for c, (i, j) in enumerate(orbits):
            s = k - i - j
            yc[c, :s] = 1
            yc[c, s:s + i] = 2
            yc[c, k:k + j] = 1
        self.y_canon = yc
        self.x0 = np.zeros(n, dtype=np.int16)
        self.x0[:k] = 1
        self.x0_idx = int(self.state_to_index_np(self.x0[None])[0])
        self.y_canon_idx = self.state_to_index_np(yc)
        for c in range(C):
            assert self.sigma_np(self.x0[None], yc[c][None])[0] == c

        # orbit of every state relative to x0
        self.orbit_x0 = self.sigma_np(
            np.broadcast_to(self.x0, (self.M, n)), self.states)
        cnt = np.bincount(self.orbit_x0, minlength=C).astype(np.float64)
        assert np.allclose(cnt, self.N_ij), (cnt, self.N_ij)
        self.orbit_x0_t = torch.as_tensor(self.orbit_x0, device=self.device)

    def sigma_np(self, x, y):
        nzx, nzy = x != 0, y != 0
        i = (nzx & nzy & (x != y)).sum(-1)
        j = self.k - (nzx & nzy).sum(-1)
        return self.orbit_lut[i, j]

    def sigma(self, x, y):
        nzx, nzy = x != 0, y != 0
        i = (nzx & nzy & (x != y)).sum(-1)
        j = self.k - (nzx & nzy).sum(-1)
        return self.orbit_lut_t[i, j]

    # -- orbit kernel ------------------------------------------------------
    def kappa(self, delta: float) -> np.ndarray:
        if delta <= 0.0:
            q = np.zeros(self.C)
            q[0] = 1.0
        else:
            Mx = self.alpha * delta * self.B_lab + self.beta * delta * self.B_sup
            q = expm(Mx)[0]
        return q / self.N_ij

    def log_kappa(self, delta: float) -> np.ndarray:
        return np.log(np.maximum(self.kappa(delta), TINY))

    def kappa_grid(self, steps: int) -> np.ndarray:
        return np.stack([self.kappa(s / steps) for s in range(steps + 1)], 0)

    def log_kappa_grid(self, steps: int) -> np.ndarray:
        return np.log(np.maximum(self.kappa_grid(steps), TINY))

    # -- canonical endpoint action h_y -------------------------------------
    def canon_action(self, y: torch.Tensor):
        """y : (B,n) long -> perm,(sa,sb) with h_y(x0)=x0, h_y(y_orbit)=y."""
        n, k = self.n, self.k
        pos = torch.arange(n, device=y.device).expand_as(y)
        head = torch.where(y == 1, 0, torch.where(y == 0, 2, 1))
        tail = torch.where(y != 0, 3, 4)
        blk = torch.where(pos < k, head, tail)
        order = torch.argsort(blk, dim=1, stable=True)
        perm = torch.argsort(order, dim=1)
        sa = torch.where(blk == 1, 2, torch.where(blk == 3, 1, -1))
        sb = torch.where((blk == 1) | (blk == 3), y, torch.full_like(y, -1))
        return perm, sa, sb

    @staticmethod
    def apply_pls(x, perm, sa, sb):
        y = torch.gather(x, 1, perm)
        return torch.where(y == sa, sb, torch.where(y == sb, sa, y))

    # -- bridge class tables ------------------------------------------------
    def build_bridge(self):
        C, M = self.C, self.M
        a = self.orbit_x0
        members = np.zeros((C, M), dtype=np.int32)
        counts = np.zeros((C, C * C), dtype=np.int64)
        offsets = np.zeros((C, C * C + 1), dtype=np.int64)
        for c in range(C):
            b = self.sigma_np(self.states,
                              np.broadcast_to(self.y_canon[c], (M, self.n)))
            key = a * C + b
            order = np.argsort(key, kind="stable")
            members[c] = order.astype(np.int32)
            cc = np.bincount(key, minlength=C * C)
            counts[c] = cc
            offsets[c, 1:] = np.cumsum(cc)
        self.bridge_members = members
        self.bridge_counts = counts
        self.bridge_offsets = offsets
        dev = self.device
        self.bm_t = torch.as_tensor(members.astype(np.int64), device=dev)
        self.bc_t = torch.as_tensor(counts, device=dev)
        self.bo_t = torch.as_tensor(offsets, device=dev)
        ka = np.repeat(np.arange(C), C)
        kb = np.tile(np.arange(C), C)
        self.key_a_t = torch.as_tensor(ka, device=dev)
        self.key_b_t = torch.as_tensor(kb, device=dev)

    def build_kappa_grid(self, steps: int):
        self.grid_steps = steps
        g = self.kappa_grid(steps)
        self.kap_grid_t = torch.as_tensor(g, device=self.device,
                                          dtype=torch.float64)
        return self.kap_grid_t

    def sample_bridge_canonical(self, c: torch.Tensor, ti: torch.Tensor,
                                gen=None):
        """c : (B,) endpoint orbits, ti : (B,) time indices in 1..steps-1."""
        steps = self.grid_steps
        kap = self.kap_grid_t
        kt = kap[ti]                       # (B, C)
        k1 = kap[steps - ti]               # (B, C)
        wb = self.bc_t[c].to(torch.float64) * kt[:, self.key_a_t] \
            * k1[:, self.key_b_t]
        wb = wb / wb.sum(1, keepdim=True)
        key = torch.multinomial(wb, 1, generator=gen).squeeze(1)
        cnt = self.bc_t[c, key]
        off = self.bo_t[c, key]
        u = torch.rand(c.shape[0], device=c.device, generator=gen)
        pos = off + (u * cnt.to(torch.float64)).floor().long().clamp_max_(
            cnt - 1)
        return self.bm_t[c, pos]

    def sample_bridge(self, y_idx: torch.Tensor, ti: torch.Tensor, gen=None):
        """Dirac bridge from x0 to endpoints y_idx at time indices ti."""
        if not torch.is_tensor(ti):
            ti = torch.full_like(y_idx, int(ti))
        out = torch.full_like(y_idx, self.x0_idx)
        m = ti > 0
        if not bool(m.any()):
            return out
        y = self.states_t[y_idx[m]]
        c = self.orbit_x0_t[y_idx[m]]
        z = self.sample_bridge_canonical(c, ti[m], gen)
        perm, sa, sb = self.canon_action(y)
        zs = self.apply_pls(self.states_t[z], perm, sa, sb)
        out[m] = self.state_to_index(zs)
        return out

    # -- non-Dirac source ---------------------------------------------------
    def build_sources(self):
        n, k = self.n, self.k
        x0 = self.x0
        s = np.repeat(x0[None], 4, 0)
        s[1, 0] = 2
        s[2, k - 1] = 0
        s[2, k] = 1
        s[3, k - 1] = 0
        s[3, k] = 2
        assert len({tuple(v) for v in s}) == 4
        self.sources = s
        self.source_idx = self.state_to_index_np(s)
        acts = []
        for c in range(4):
            perm = np.arange(n)
            sa = np.full(n, -1)
            sb = np.full(n, -1)
            if c == 1:
                sa[0], sb[0] = 1, 2
            elif c in (2, 3):
                perm[k - 1], perm[k] = k, k - 1
                if c == 3:
                    sa[k], sb[k] = 1, 2
            acts.append((perm, sa, sb))
        self.src_act = acts
        inv = []
        for (perm, sa, sb) in acts:
            pinv = np.argsort(perm)
            inv.append((pinv, sa[pinv], sb[pinv]))
        self.src_act_inv = inv
        dev = self.device
        self.src_perm_t = torch.as_tensor(
            np.stack([a[0] for a in acts]), device=dev)
        self.src_sa_t = torch.as_tensor(
            np.stack([a[1] for a in acts]), device=dev)
        self.src_sb_t = torch.as_tensor(
            np.stack([a[2] for a in acts]), device=dev)
        self.srci_perm_t = torch.as_tensor(
            np.stack([a[0] for a in inv]), device=dev)
        self.srci_sa_t = torch.as_tensor(
            np.stack([a[1] for a in inv]), device=dev)
        self.srci_sb_t = torch.as_tensor(
            np.stack([a[2] for a in inv]), device=dev)
        for c in range(4):
            got = self.apply_pls(self.states_t[self.x0_idx][None],
                                 self.src_perm_t[c][None],
                                 self.src_sa_t[c][None],
                                 self.src_sb_t[c][None])[0]
            assert (got.cpu().numpy() == s[c]).all(), c
        self.source_idx_t = torch.as_tensor(self.source_idx, device=dev)

    def sample_bridge_nondirac(self, src: torch.Tensor, y_idx: torch.Tensor,
                               ti: torch.Tensor, gen=None):
        """src : (B,) in {0,1,2,3}; endpoints y_idx; time indices ti."""
        if not torch.is_tensor(ti):
            ti = torch.full_like(y_idx, int(ti))
        out = self.source_idx_t[src].clone()
        m = ti > 0
        if not bool(m.any()):
            return out
        s = src[m]
        y = self.states_t[y_idx[m]]
        yp = self.apply_pls(y, self.srci_perm_t[s], self.srci_sa_t[s],
                            self.srci_sb_t[s])
        c = self.sigma(self.states_t[self.x0_idx][None].expand_as(yp), yp)
        z = self.sample_bridge_canonical(c, ti[m], gen)
        perm, sa, sb = self.canon_action(yp)
        zp = self.apply_pls(self.states_t[z], perm, sa, sb)
        zs = self.apply_pls(zp, self.src_perm_t[s], self.src_sa_t[s],
                            self.src_sb_t[s])
        out[m] = self.state_to_index(zs)
        return out


def make_space(n, k, r, gamma=10.0, device="cpu", edges=True):
    sp = FixedSupportSpace(n, k, r, gamma=gamma, device=device,
                           build_edges=edges)
    sp.build_orbits()
    sp.build_bridge()
    sp.build_sources()
    return sp


AA = "ACDEFGHIKLMNPQRSTVWY"
GB1_WT = "VDGV"
GB1_FITNESS_EPS = 1e-4
GB1_DOI = "10.7554/eLife.16965"
TOY_SEED = 20260908


# ----------------------------------------------------------------------------
# toy target
# ----------------------------------------------------------------------------
def toy_energy(space: FixedSupportSpace, seed: int = TOY_SEED):
    n, r = space.n, space.r
    rng = np.random.default_rng(seed)
    H = rng.normal(0.0, 0.7, size=(n, r))
    J = rng.normal(0.0, 0.4, size=(n, r, r))
    H[:, 0] = 0.0
    J[:, 0, :] = 0.0
    J[:, :, 0] = 0.0
    x = space.states.astype(np.int64)
    nxt = np.roll(x, -1, axis=1)
    site = np.arange(n)[None, :]
    E = H[site, x].sum(1) + J[site, x, nxt].sum(1)
    return E, H, J


# ----------------------------------------------------------------------------
# GB1 loader
# ----------------------------------------------------------------------------
def _read_variant_sheet(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        rows = []
        for ri, row in enumerate(ws.iter_rows(values_only=True)):
            rows.append(row)
            if ri >= 24:
                break
        hdr_i = seq_c = fit_c = None
        for ri, row in enumerate(rows):
            if row is None:
                continue
            norm = [str(v).strip().lower() if v is not None else "" for v in row]
            sc = [i for i, v in enumerate(norm)
                  if v in ("variant", "variants", "sequence", "sequences")]
            fc = [i for i, v in enumerate(norm) if "fitness" in v]
            if sc and fc:
                hdr_i, seq_c, fit_c = ri, sc[0], fc[0]
                break
        if hdr_i is None:
            continue
        out = {}
        for ri, row in enumerate(ws.iter_rows(values_only=True)):
            if ri <= hdr_i:
                continue
            if row is None or row[seq_c] is None or row[fit_c] is None:
                continue
            s = str(row[seq_c]).strip().upper()
            if not s:
                continue
            out[s] = float(row[fit_c])
        wb.close()
        return out
    wb.close()
    raise RuntimeError(f"no variant/fitness header found in {path}")


def load_gb1(measured_path, imputed_path, verbose=True):
    meas = _read_variant_sheet(measured_path)
    imp = _read_variant_sheet(imputed_path)
    assert len(meas) == 149361, len(meas)
    assert len(imp) == 10639, len(imp)
    inter = set(meas) & set(imp)
    assert len(inter) == 0, len(inter)
    merged = dict(meas)
    merged.update(imp)
    assert len(merged) == 160000, len(merged)
    for s, f in merged.items():
        assert len(s) == 4 and all(ch in AA for ch in s), s
        assert math.isfinite(f) and f >= 0.0, (s, f)
    assert GB1_WT in meas
    assert abs(meas[GB1_WT] - 1.0) < 1e-6, meas[GB1_WT]
    if verbose:
        print(f"[gb1] measured={len(meas)} imputed={len(imp)} "
              f"merged={len(merged)} WT={GB1_WT} F(WT)={meas[GB1_WT]}")
    return merged, len(meas), len(imp)


def gb1_site_alphabets():
    """Per-site label -> amino acid; label 0 is that site's WT residue."""
    out = []
    for p in range(4):
        wt = GB1_WT[p]
        out.append(wt + "".join(c for c in AA if c != wt))
    return out


def gb1_seq_to_labels(seq, alpha):
    return [alpha[p].index(seq[p]) for p in range(4)]


def gb1_labels_to_seq(lab, alpha):
    return "".join(alpha[p][lab[p]] for p in range(4))


def gb1_energy(space: FixedSupportSpace, measured_path, imputed_path,
               verbose=True):
    merged, nmeas, nimp = load_gb1(measured_path, imputed_path, verbose)
    alpha = gb1_site_alphabets()
    # round-trip test on all variants
    for s in list(merged)[:2000]:
        assert gb1_labels_to_seq(gb1_seq_to_labels(s, alpha), alpha) == s
    sector = {s: f for s, f in merged.items()
              if sum(s[p] != GB1_WT[p] for p in range(4)) == 3}
    assert len(sector) == 27436, len(sector)
    F = np.zeros(space.M)
    seen = 0
    labs = space.states.astype(np.int64)
    for s, f in sector.items():
        lab = gb1_seq_to_labels(s, alpha)
        idx = int(space.state_to_index_np(np.array(lab, dtype=np.int16)[None])[0])
        F[idx] = f
        seen += 1
    assert seen == space.M == 27436
    E = -np.log(F + GB1_FITNESS_EPS)
    meta = dict(doi=GB1_DOI, measured=nmeas, imputed=nimp,
                sector=len(sector), wt=GB1_WT, eps=GB1_FITNESS_EPS,
                alphabet=AA)
    return E, meta


def target_law(E, tau=1.0):
    lw = -np.asarray(E) / tau
    lw = lw - lw.max()
    pi = np.exp(lw)
    return pi / pi.sum()


def energy_stats(E, pi):
    return dict(E_min=float(E.min()), E_max=float(E.max()),
                E_mean=float(E.mean()), E_std=float(E.std()),
                pi_max=float(pi.max()),
                entropy=float(-(pi * np.log(np.maximum(pi, TINY))).sum()))


def set_energy(space: FixedSupportSpace, E: np.ndarray, tau: float = 1.0):
    space.tau = float(tau)
    space.energy = np.asarray(E, dtype=np.float64)
    space.energy_t = torch.as_tensor(space.energy, device=space.device)
    space.pi_np = target_law(space.energy, tau)
    space.pi_t = torch.as_tensor(space.pi_np, device=space.device)
    space.pi = space.pi_t          # DAM _finish expects a tensor
    lk1 = space.log_kappa(1.0)
    lf = -space.energy / tau - lk1[space.orbit_x0]
    space.logf1 = lf - lf.max()
    space.logf1_t = torch.as_tensor(space.logf1, device=space.device)
    lk1_t = torch.as_tensor(lk1, device=space.device)
    space.log_kappa1_t = lk1_t
    # non-Dirac base terminal marginal (four-atom source)
    f = np.zeros(space.M)
    for c in range(4):
        a = space.sigma_np(np.broadcast_to(space.sources[c], (space.M, space.n)),
                           space.states)
        f += 0.25 * space.kappa(1.0)[a]
    space.log_f1_base_nd = np.log(np.maximum(f, TINY))
    space.log_f1_base_nd_t = torch.as_tensor(space.log_f1_base_nd,
                                             device=space.device)
    return space


# ----------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------
def time_feats(t: torch.Tensor) -> torch.Tensor:
    q = torch.arange(1, 5, device=t.device, dtype=t.dtype)
    a = math.pi * t[:, None] * q[None, :]
    return torch.cat([t[:, None], 1 - t[:, None], torch.sin(a), torch.cos(a)], 1)


class FixedSupportController(nn.Module):
    def __init__(self, n, r, out_dim, hidden=512):
        super().__init__()
        self.n, self.r = n, r
        self.net = nn.Sequential(
            nn.Linear(n * r + 10, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim)).float()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        oh = F.one_hot(x, self.r).flatten(1).float()
        tf = time_feats(t.float())
        return self.net(torch.cat([oh, tf], 1))


class FixedSupportCorrector(nn.Module):
    def __init__(self, n, r, n_actions, hidden=256):
        super().__init__()
        self.n, self.r, self.H = n, r, hidden
        self.enc = nn.Sequential(
            nn.Linear(n * r, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden)).float()
        self.action_embed = nn.Embedding(n_actions, hidden).float()
        self.action_bias = nn.Embedding(n_actions, 1).float()
        nn.init.normal_(self.action_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.action_bias.weight)

    def forward(self, y, g):
        """y : (B,n) long, g : (B,A2) long -> (B,A2)."""
        oh = F.one_hot(y, self.r).flatten(1).float()
        h = self.enc(oh)                                   # (B,H)
        e = self.action_embed(g)                           # (B,A2,H)
        b = self.action_bias(g).squeeze(-1)                # (B,A2)
        return (e * h[:, None, :]).sum(-1) / math.sqrt(self.H) + b


def net_mult_on_index(space, net, t, idx):
    x = space.states_t[idx]
    tt = torch.full((idx.shape[0],), float(t), device=idx.device)
    out = net(x, tt).double()
    return torch.gather(out, 1, space.edge_out_t[idx])


@torch.no_grad()
def net_mult_all_states(space, net, t, chunk=4096):
    outs = []
    for s in range(0, space.M, chunk):
        idx = torch.arange(s, min(space.M, s + chunk), device=space.device)
        outs.append(net_mult_on_index(space, net, t, idx).double())
    return torch.cat(outs, 0)


# ----------------------------------------------------------------------------
# exact propagation and simulation
# ----------------------------------------------------------------------------
@torch.no_grad()
def propagate_exact(space, mult_fn, steps, p0=None, chunk=4096):
    dev = space.device
    if p0 is None:
        p = torch.zeros(space.M, dtype=torch.float64, device=dev)
        p[space.x0_idx] = 1.0
    else:
        p = p0.clone().to(dev).double()
    dt = 1.0 / steps
    base = space.gamma / space.E
    tgt = space.tgt_t.reshape(-1)
    for s in range(steps):
        t = s * dt
        a = mult_fn(t)
        u = base * torch.exp(a)
        R = u.sum(1)
        pstay = torch.exp(-R * dt)
        newp = p * pstay
        w = (p * (1.0 - pstay) / R)[:, None] * u
        newp = newp.index_add(0, tgt, w.reshape(-1))
        p = newp
    return p


@torch.no_grad()
def simulate(space, mult_fn_states, batch, steps, generator=None, idx0=None):
    dev = space.device
    if idx0 is None:
        idx = torch.full((batch,), space.x0_idx, device=dev, dtype=torch.long)
    else:
        idx = idx0.clone()
    dt = 1.0 / steps
    base = space.gamma / space.E
    for s in range(steps):
        t = s * dt
        a = mult_fn_states(t, idx).double()
        u = base * torch.exp(a)
        R = u.sum(1)
        pstay = torch.exp(-R * dt)
        r1 = torch.rand(idx.shape[0], device=dev, generator=generator,
                        dtype=torch.float64)
        jump = r1 > pstay
        if jump.any():
            pe = u[jump] / R[jump][:, None]
            e = torch.multinomial(pe, 1, generator=generator).squeeze(1)
            idx = idx.clone()
            idx[jump] = space.tgt_t[idx[jump], e]
    return idx


# ----------------------------------------------------------------------------
# training labels
# ----------------------------------------------------------------------------
def terminal_labels(space, y_idx, g):
    """Dirac log Lambda_g(Y). y_idx : (B,), g : (B,E) -> (B,E)."""
    y = space.states_t[y_idx]
    gy = space.apply_action(y, g)
    gyi = space.state_to_index(gy)
    dE = space.energy_t[gyi] - space.energy_t[y_idx][:, None]
    lk = space.log_kappa1_t
    return -dE / space.tau + lk[space.orbit_x0_t[y_idx]][:, None] \
        - lk[space.orbit_x0_t[gyi]]


def corrector_labels(space, src, y_idx, g):
    """log Q_g(X0,Y) with X0 = source atom src. g : (B,A2) -> (B,A2)."""
    y = space.states_t[y_idx]
    gy = space.apply_action(y, g)
    x0 = torch.as_tensor(space.sources.astype(np.int64),
                         device=space.device)[src]
    lk = space.log_kappa1_t
    a1 = space.sigma(x0[:, None, :].expand_as(gy), gy)
    a0 = space.sigma(x0, y)
    return lk[a1] - lk[a0][:, None]


def nondirac_terminal_labels(space, y_idx, g, hnet):
    y = space.states_t[y_idx]
    gy = space.apply_action(y, g)
    gyi = space.state_to_index(gy)
    dE = space.energy_t[gyi] - space.energy_t[y_idx][:, None]
    h = hnet(y, g).double()
    return -dE / space.tau - h


@torch.no_grad()
def exact_corrector_ratio(space, y_idx, g):
    """log C_g(y) = log f1_base(g y) - log f1_base(y), four-atom source."""
    y = space.states_t[y_idx]
    gy = space.apply_action(y, g)
    gyi = space.state_to_index(gy)
    lb = space.log_f1_base_nd_t
    return lb[gyi] - lb[y_idx][:, None]


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
@torch.no_grad()
def report(space, p, samples=None, fitness=None):
    pi = space.pi_t
    p = p.double()
    tv = 0.5 * (p - pi).abs().sum().item()
    pc = p.clamp_min(1e-300)
    kl = (pc * (pc.log() - pi.clamp_min(1e-300).log())).sum().item()
    hell = (0.5 * ((p.sqrt() - pi.sqrt()) ** 2).sum()).sqrt().item()
    out = dict(tv=tv, kl=kl, hellinger=hell,
               mean_energy=float((p * space.energy_t).sum().item()),
               target_mean_energy=float((pi * space.energy_t).sum().item()),
               mass_err=float(abs(p.sum().item() - 1.0)))
    am = int(p.argmax().item())
    out["argmax_state"] = am
    out["argmax_p"] = float(p[am].item())
    out["argmax_pi"] = float(pi[am].item())
    if fitness is not None:
        ft = torch.as_tensor(fitness, device=space.device)
        out["mean_fitness"] = float((p * ft).sum().item())
        out["target_mean_fitness"] = float((pi * ft).sum().item())
    if samples is not None:
        cnt = torch.bincount(samples, minlength=space.M).double()
        emp = cnt / cnt.sum()
        out["tv_emp"] = 0.5 * (emp - pi).abs().sum().item()
        st = space.states_t[samples]
        out["violations"] = int(((st != 0).sum(1) != space.k).sum().item())
    return out


def violations(space, idx):
    return ((space.S[idx] != 0).sum(1) != space.k)


TARGETS = {
    "toy": dict(name="fixed_support_toy_v2", n=14, k=4, r=4, tau=1.0,
                gamma=10.0, M=81081),
    "gb1": dict(name="fixed_support_gb1_k3", n=4, k=3, r=20, tau=1.0,
                gamma=10.0, M=27436),
}


def build_space(args, device=None):
    """Construct the space and attach the target energy for --target."""
    if args.target not in TARGETS:
        raise NotImplementedError(
            f"target {args.target} is not implemented (see Section 27)")
    cfg = TARGETS[args.target]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp = make_space(cfg["n"], cfg["k"], cfg["r"], gamma=cfg["gamma"],
                    device=dev)
    assert sp.M == cfg["M"], (sp.M, cfg["M"])
    meta = dict(target=args.target, **{q: cfg[q] for q in
                                       ("name", "n", "k", "r", "tau", "gamma")})
    if args.target == "toy":
        E, H, J = toy_energy(sp)
        meta.update(seed=TOY_SEED, H=H.tolist(), J=J.tolist())
        fit = None
    else:
        E, gmeta = gb1_energy(sp, args.gb1_measured, args.gb1_imputed)
        meta.update(gmeta)
        fit = np.exp(-E) - GB1_FITNESS_EPS
    set_energy(sp, E, cfg["tau"])
    st = energy_stats(sp.energy, sp.pi_np)
    meta.update(st)
    print(f"[{cfg['name']}] |X|={sp.M} E_edges={sp.E} "
          f"(lab {sp.E_lab} + sup {sp.E_sup})  A={sp.A}  C={sp.C}  "
          f"alpha={sp.alpha:.4f} beta={sp.beta:.4f} dev={dev}")
    print("  target stats: " + "  ".join(f"{q}={st[q]:.4f}" for q in
                                         ("E_min", "E_max", "E_mean", "E_std",
                                          "pi_max", "entropy")))
    return sp, meta, fit


def _final_eval(space, net, args, p0=None, fitness=None, idx0_fn=None):
    steps = args.steps
    with torch.no_grad():
        p = propagate_exact(
            space, lambda t: net_mult_all_states(space, net, t), steps, p0=p0)
        i0 = idx0_fn(args.n_samples) if idx0_fn is not None else None
        idx = simulate(space, lambda t, i: net_mult_on_index(space, net, t, i),
                       args.n_samples, steps, idx0=i0)
    rep = report(space, p, samples=idx, fitness=fitness)
    iid = torch.multinomial(space.pi_t, args.n_samples, replacement=True)
    e2 = torch.bincount(iid, minlength=space.M).double()
    rep["tv_emp_floor"] = 0.5 * float((e2 / e2.sum() - space.pi_t).abs().sum())
    return p, idx, rep


def _gates(rep, args):
    tv, viol = rep["tv"], rep["violations"]
    print("\n=== GATES ===")
    print(f"A1  TV <= 0.05 : {'PASS' if tv <= 0.05 else 'FAIL'}  "
          f"(exact-law TV {tv:.5f})")
    print(f"A2  constraint : {'PASS' if viol == 0 else 'FAIL'}  "
          f"({viol} violations in {args.n_samples} samples)")
    print(f"    empirical TV {rep['tv_emp']:.5f}   "
          f"iid floor {rep['tv_emp_floor']:.5f}")
    return 0 if (tv <= 0.05 and viol == 0) else 1


def run_train(args):
    space, meta, fit = build_space(args)
    torch.manual_seed(args.seed)
    net = FixedSupportController(space.n, space.r, space.out_dim,
                                 hidden=args.hidden).to(space.device)
    n_par = sum(p.numel() for p in net.parameters())
    if getattr(args, "init_from", ""):
        blob = torch.load(args.init_from, map_location=space.device)
        net.load_state_dict(blob["state_dict"])
        print(f"  warm start <- {args.init_from} (weights only)")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    space.build_kappa_grid(steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)
    print(f"  params={n_par}  steps={steps}  iters={args.iters}  "
          f"source=Dirac(x0)")

    buf, hist = [], []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            idx1 = simulate(
                space, lambda t, i: net_mult_on_index(space, net, t, i),
                args.batch, steps)
        buf.append(idx1)
        if len(buf) > args.buffer:
            buf.pop(0)
        pool = torch.cat(buf)

        for _ in range(args.inner):
            sel = torch.randint(len(pool), (args.mb,), device=space.device)
            X1 = pool[sel]
            ti = torch.randint(steps, (args.mb,), device=space.device)
            with torch.no_grad():
                Xt = space.sample_bridge(X1, ti)
                g = space.edge_action_t[Xt]
                log_lam = terminal_labels(space, X1, g)
                lam = torch.exp(log_lam.clamp(-20.0, 20.0))
            tfl = ti.double() / steps
            out = net(space.states_t[Xt], tfl).double()
            av = torch.gather(out, 1, space.edge_out_t[Xt]).clamp(-20.0, 20.0)
            if args.loss == "poisson":
                loss = (torch.exp(av) - av * lam).mean()
            else:
                loss = ((torch.exp(av) - lam) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = propagate_exact(
                    space, lambda t: net_mult_all_states(space, net, t), steps)
            tv = 0.5 * float((p - space.pi_t).abs().sum())
            hist.append({"iter": it, "TV": tv, "loss": float(loss)})
            print(f"  it {it:4d}  loss {float(loss):10.4f}   "
                  f"exact-law TV {tv:.5f}   ({time.time()-t0:.0f}s)")

    p, idx, rep = _final_eval(space, net, args, fitness=fit)
    rc = _gates(rep, args)
    if getattr(args, "ckpt_dir", ""):
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          samples=idx.to(torch.int32),
                          extra={"config": vars(args), "meta": meta,
                                 "report": rep, "history": hist,
                                 "exact_law": p.detach().cpu(),
                                 "pi": space.pi_t.detach().cpu()})
        print(f"  ckpt -> {pth}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "meta": meta, "params": n_par,
                   "history": hist, "report": rep}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return rc


def run_train_nondirac(args):
    space, meta, fit = build_space(args)
    torch.manual_seed(args.seed)
    net = FixedSupportController(space.n, space.r, space.out_dim,
                                 hidden=args.hidden).to(space.device)
    net_h = FixedSupportCorrector(space.n, space.r, space.A,
                                  hidden=args.corrector_hidden).to(space.device)
    n_par = sum(p.numel() for p in net.parameters())
    n_par_h = sum(p.numel() for p in net_h.parameters())
    if getattr(args, "init_from", ""):
        blob = torch.load(args.init_from, map_location=space.device)
        net.load_state_dict(blob["state_dicts"]["control"])
        net_h.load_state_dict(blob["state_dicts"]["corrector"])
        print(f"  warm start <- {args.init_from} (weights only)")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    opt_h = torch.optim.Adam(net_h.parameters(), lr=args.lr_h)
    steps = args.steps
    space.build_kappa_grid(steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)
    sched_h = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_h, T_max=args.iters * args.inner_h, eta_min=args.lr_h * 0.05)
    print(f"  params={n_par}+{n_par_h}  steps={steps}  "
          f"source=4-atom mixture {space.source_idx.tolist()}")

    p0 = torch.zeros(space.M, dtype=torch.float64, device=space.device)
    p0[space.source_idx_t] = 0.25

    def draw_src(bs):
        return torch.randint(4, (bs,), device=space.device)

    buf0, buf1, hist, it1 = [], [], [], {}
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            src = draw_src(args.batch)
            idx1 = simulate(
                space, lambda t, i: net_mult_on_index(space, net, t, i),
                args.batch, steps, idx0=space.source_idx_t[src])
        buf0.append(src)
        buf1.append(idx1)
        if len(buf1) > args.buffer:
            buf0.pop(0)
            buf1.pop(0)
        pool0, pool1 = torch.cat(buf0), torch.cat(buf1)

        for _ in range(args.inner_h):
            sel = torch.randint(len(pool1), (args.mb,), device=space.device)
            Y, S0 = pool1[sel], pool0[sel]
            g = torch.randint(space.A, (args.mb, args.corrector_actions),
                              device=space.device)
            with torch.no_grad():
                q = torch.exp(corrector_labels(space, S0, Y, g)
                              .clamp(-20.0, 20.0))
            hv = net_h(space.states_t[Y], g).double().clamp(-20.0, 20.0)
            loss_h = (torch.exp(hv) - hv * q).mean()
            opt_h.zero_grad(set_to_none=True)
            loss_h.backward()
            torch.nn.utils.clip_grad_norm_(net_h.parameters(), 10.0)
            opt_h.step()
            sched_h.step()

        for _ in range(args.inner):
            sel = torch.randint(len(pool1), (args.mb,), device=space.device)
            X1, S0 = pool1[sel], pool0[sel]
            ti = torch.randint(steps, (args.mb,), device=space.device)
            with torch.no_grad():
                Xt = space.sample_bridge_nondirac(S0, X1, ti)
                g = space.edge_action_t[Xt]
                log_lam = nondirac_terminal_labels(space, X1, g, net_h)
                lam = torch.exp(log_lam.clamp(-20.0, 20.0))
            tfl = ti.double() / steps
            out = net(space.states_t[Xt], tfl).double()
            av = torch.gather(out, 1, space.edge_out_t[Xt]).clamp(-20.0, 20.0)
            if args.loss == "poisson":
                loss = (torch.exp(av) - av * lam).mean()
            else:
                loss = ((torch.exp(av) - lam) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = propagate_exact(
                    space, lambda t: net_mult_all_states(space, net, t), steps,
                    p0=p0)
            tv = 0.5 * float((p - space.pi_t).abs().sum())
            hist.append({"iter": it, "TV": tv, "loss": float(loss),
                         "loss_h": float(loss_h)})
            print(f"  it {it:4d}  loss {float(loss):10.4f}  "
                  f"loss_h {float(loss_h):8.4f}   exact-law TV {tv:.5f}   "
                  f"({time.time()-t0:.0f}s)")

    # exact corrector diagnostic (evaluation only)
    with torch.no_grad():
        npair, chunk, errs = 100000, 2000, []
        gper = 10
        for s in range(0, npair, chunk * gper):
            y = torch.randint(space.M, (chunk,), device=space.device)
            g = torch.randint(space.A, (chunk, gper), device=space.device)
            hv = net_h(space.states_t[y], g).double()
            lc = exact_corrector_ratio(space, y, g)
            errs.append((hv - lc).flatten())
        e = torch.cat(errs)
        cdiag = dict(n=int(e.numel()),
                     rmse=float((e ** 2).mean().sqrt()),
                     mae=float(e.abs().mean()),
                     max_abs=float(e.abs().max()))
    print(f"  corrector vs exact log C: RMSE {cdiag['rmse']:.4f}  "
          f"MAE {cdiag['mae']:.4f}  max {cdiag['max_abs']:.4f}  "
          f"(n={cdiag['n']})")

    p, idx, rep = _final_eval(
        space, net, args, p0=p0, fitness=fit,
        idx0_fn=lambda b: space.source_idx_t[draw_src(b)])
    rc = _gates(rep, args)
    if getattr(args, "ckpt_dir", ""):
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          nets={"control": net, "corrector": net_h},
                          samples=idx.to(torch.int32),
                          extra={"config": vars(args), "meta": meta,
                                 "report": rep, "history": hist,
                                 "corrector_diag": cdiag,
                                 "source": "four-atom",
                                 "exact_law": p.detach().cpu(),
                                 "pi": space.pi_t.detach().cpu()})
        print(f"  ckpt -> {pth}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "meta": meta,
                   "params": n_par + n_par_h, "source": "four-atom",
                   "history": hist, "report": rep,
                   "corrector_diag": cdiag}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return rc


def run_build_cache(args):
    raise NotImplementedError(
        "build-cache is only required for --target gb1-product2 (Section 27), "
        "which is gated on both Table 1 rows completing first.")


def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["train", "train-nondirac", "build-cache"]:
        p = sub.add_parser(name)
        stem = name.replace("-", "_")
        p.add_argument("--target", type=str, default="toy",
                       choices=["toy", "gb1", "gb1-product2"])
        p.add_argument("--steps", type=int, default=256)
        p.add_argument("--iters", type=int, default=3000)
        p.add_argument("--batch", type=int, default=2048)
        p.add_argument("--buffer", type=int, default=8)
        p.add_argument("--inner", type=int, default=40)
        p.add_argument("--mb", type=int, default=1024)
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--eval-every", dest="eval_every", type=int, default=500)
        p.add_argument("--n-samples", dest="n_samples", type=int, default=20000)
        p.add_argument("--loss", type=str, default="poisson",
                       choices=["poisson", "mse"])
        p.add_argument("--out", type=str,
                       default=f"json/results_fixed_support_{stem}.json")
        p.add_argument("--ckpt-dir", dest="ckpt_dir", type=str, default="ckpt")
        # Warm start: network weights only.  Optimiser moments and the cosine
        # position are not in the checkpoint, so a continuation is a new run
        # initialised at the old weights, not a resumed single schedule.
        p.add_argument("--init-from", dest="init_from", type=str, default="")
        p.add_argument("--tag", type=str, default=f"fixed_support_{stem}")
        p.add_argument("--gb1-measured", dest="gb1_measured", type=str,
                       default="data/gb1/elife-16965-supp1-v4.xlsx")
        p.add_argument("--gb1-imputed", dest="gb1_imputed", type=str,
                       default="data/gb1/elife-16965-supp2-v4.xlsx")
        if name == "train-nondirac":
            p.add_argument("--inner-h", dest="inner_h", type=int, default=40)
            p.add_argument("--lr-h", dest="lr_h", type=float, default=3e-4)
            p.add_argument("--corrector-hidden", dest="corrector_hidden",
                           type=int, default=256)
            p.add_argument("--corrector-actions", dest="corrector_actions",
                           type=int, default=32)
    args = ap.parse_args()
    if args.cmd == "build-cache":
        return run_build_cache(args)
    return run_train_nondirac(args) if args.cmd == "train-nondirac" \
        else run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
