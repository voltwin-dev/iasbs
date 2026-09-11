"""Radial Johnson IASBS for fixed-composition CuAu.

This module implements ``CUAU_RADIAL_JOHNSON_IASBS_PLAYBOOK.md``.  The
single-swap reference used by the published experiments is the ``j = 1`` member
of a family indexed by Johnson shell,

    L_t = sum_j gamma_j(t) (K_j - I),      (K_j f)(S) = (1/v_j) sum_{d(S,Y)=j} f(Y),

with ``v_j = C(k,j) C(n-k,j)``.  Every ``K_j`` lies in the Bose-Mesner algebra of
the Johnson scheme, so the shells commute with each other and with every
coordinate permutation.  Two consequences make the family usable at ``N = 64``
where a single shell can hold ``3.6e17`` states:

* the reference is exactly reducible to the ``(D+1)``-state distance chain, so
  the kernel is a ``33 x 33`` matrix exponential no matter how large ``v_j`` is;
* the *controller* must also avoid the shell, which is why the controlled rate
  is factored as ``u_j = lambda_j(x,t) q_j(y | x,t)`` with ``q_j`` an exactly
  normalized, exactly sampleable fixed-cardinality product distribution.  Its
  normalizer is an elementary symmetric polynomial evaluated by an ``O(N j)``
  dynamic program.

Everything that does not depend on the reference is inherited from
:mod:`iasbs.cuau`: the cluster-expansion energy, the L1_0 order parameters, the
four-block bridge class table, the symmetry group and the sampled observables.
In particular the bridge sampler is reused *verbatim* -- its combinatorics
depends only on ``(n, k)``, and the only reference-dependent inputs are the two
``log kappa`` tables it is handed.

This module is additive.  It does not modify any published result path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as C  # noqa: E402
from iasbs.cuau import (KB_EV_PER_K, CuAuSpace, TorchCuAuEnergy,  # noqa: E402
                        arg_config, build_ce_tables, check_states,
                        provenance, random_sector_batch, sample_bridge_direct,
                        sampled_observables, symmetry_perms)

NEG = float("-inf")
LOG_ZERO = -1.0e30


# ============================================================================
# 1.  piecewise-constant shell schedules            (playbook 28, 43)
# ============================================================================


class PiecewiseShellSchedule:
    """``gamma_j(t) = gamma_total * w_j(segment(t))`` on a piecewise grid.

    ``breaks`` are the ``S+1`` segment boundaries with ``breaks[0] = 0`` and
    ``breaks[-1] = 1``; ``weights`` is ``(S, len(shells))`` and each row is
    normalized to sum to one, so the total event clock is ``gamma_total`` on
    every segment and schedules are compared at a matched reference event
    count (playbook 29).

    Because the shell clocks are piecewise constant, the integrated clocks
    ``Gamma_j(s,t) = int_s^t gamma_j`` are exact closed-form overlaps.
    """

    def __init__(self, shells, breaks, weights, gamma_total):
        self.shells = tuple(int(j) for j in shells)
        self.breaks = np.asarray(breaks, dtype=np.float64)
        W = np.asarray(weights, dtype=np.float64).reshape(-1, len(self.shells))
        if self.breaks.ndim != 1 or self.breaks.shape[0] != W.shape[0] + 1:
            raise ValueError("breaks must have one more entry than weight rows")
        if not (np.all(np.diff(self.breaks) > 0) and self.breaks[0] == 0.0
                and abs(self.breaks[-1] - 1.0) < 1e-15):
            raise ValueError("breaks must increase from 0 to 1")
        if W.min() < 0.0:
            raise ValueError("shell weights must be nonnegative")
        rs = W.sum(axis=1, keepdims=True)
        if float(rs.min()) <= 0.0:
            raise ValueError("every segment needs positive total weight")
        self.weights = W / rs
        self.gamma_total = float(gamma_total)
        if self.gamma_total <= 0.0:
            raise ValueError("gamma_total must be positive")
        if 1 not in self.shells:
            raise ValueError("a j=1 floor is required for irreducibility")

    # -- clocks --------------------------------------------------------------

    def rates(self, t):
        """``gamma_j(t)`` as a ``(len(shells),)`` array."""
        s = int(np.clip(np.searchsorted(self.breaks, float(t), side="right") - 1,
                        0, self.weights.shape[0] - 1))
        return self.gamma_total * self.weights[s]

    def integrated(self, s, t):
        """``{j: Gamma_j(s,t)}`` -- exact piecewise-constant time integral."""
        s, t = float(s), float(t)
        if t < s:
            raise ValueError(f"integrated clock needs s <= t, got {s} > {t}")
        acc = np.zeros(len(self.shells), dtype=np.float64)
        for m in range(self.weights.shape[0]):
            lo = max(s, self.breaks[m])
            hi = min(t, self.breaks[m + 1])
            if hi > lo:
                acc += (hi - lo) * self.gamma_total * self.weights[m]
        return {j: float(acc[a]) for a, j in enumerate(self.shells)}

    def describe(self):
        rows = []
        for m in range(self.weights.shape[0]):
            w = ", ".join(f"j{j}:{self.weights[m, a]:.2f}"
                          for a, j in enumerate(self.shells)
                          if self.weights[m, a] > 0)
            rows.append(f"[{self.breaks[m]:.2f},{self.breaks[m+1]:.2f}) {w}")
        return " | ".join(rows)

    def as_dict(self):
        return {"shells": list(self.shells), "breaks": self.breaks.tolist(),
                "weights": self.weights.tolist(),
                "gamma_total": self.gamma_total}


def named_schedule(name, gamma_total=10.0, shells=(1, 2, 4, 8, 16)):
    """The predeclared schedule family of playbook 28."""
    name = name.upper()
    sh = tuple(int(j) for j in shells)
    idx = {j: a for a, j in enumerate(sh)}

    if name == "S0":                                    # published baseline
        w = np.zeros((1, len(sh)))
        w[0, idx[1]] = 1.0
        return PiecewiseShellSchedule(sh, [0.0, 1.0], w, gamma_total)

    if name == "S1":                                    # fixed radial mixture
        tgt = {1: 0.10, 2: 0.15, 4: 0.20, 8: 0.25, 16: 0.30}
        w = np.array([[tgt.get(j, 0.0) for j in sh]], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("S1 needs shells from {1,2,4,8,16}")
        return PiecewiseShellSchedule(sh, [0.0, 1.0], w, gamma_total)

    if name in ("S2", "S3-8", "S3-4"):                  # coarse-to-fine
        ladder = {"S2": [16, 8, 4, 2, 1], "S3-8": [8, 4, 2, 1],
                  "S3-4": [4, 2, 1]}[name]
        for j in ladder:
            if j not in idx:
                raise ValueError(f"{name} needs shell {j} in --shells")
        nseg = len(ladder)
        w = np.zeros((nseg, len(sh)))
        for m, j in enumerate(ladder):
            if j == 1:
                w[m, idx[1]] = 1.0                      # final segment, all local
            else:
                w[m, idx[j]] = 0.95
                w[m, idx[1]] = 0.05                     # irreducibility floor
        br = np.linspace(0.0, 1.0, nseg + 1)
        return PiecewiseShellSchedule(sh, br, w, gamma_total)

    raise ValueError(f"unknown schedule {name}")


# ============================================================================
# 2.  radial space                                   (playbook 7, 14)
# ============================================================================


class RadialCuAuSpace(CuAuSpace):
    """:class:`~iasbs.cuau.CuAuSpace` with a radial Johnson reference.

    Only the reference kernel changes.  The four-block bridge class table built
    by ``CuAuSpace._bridge_tables`` is a function of ``(n, k)`` alone -- it is
    exactly the ``p_ab^q`` intersection count of playbook 14.2 -- so the bridge
    sampler is correct for any radial mixture once it is handed the matching
    ``log kappa`` tables.  That is the whole of the change.
    """

    def __init__(self, size=(4, 4, 4), temp_k=500.0, schedule=None, **kw):
        if schedule is None:
            schedule = named_schedule("S0", kw.pop("gamma", 10.0))
        self.schedule = schedule
        # the parent stores gamma and builds the j=1 kernel; we keep that as the
        # S0 comparison point and then install the radial full-horizon kernel.
        kw.pop("gamma", None)
        super().__init__(size=size, temp_k=temp_k,
                         gamma=schedule.gamma_total, **kw)
        self.log_kappa_swap = self.log_kappa_full.clone()
        _, kap = self.radial_kappa(0.0, 1.0)
        self.log_kappa_full = torch.as_tensor(
            np.log(kap), dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(self.log_kappa_full).all()):
            raise RuntimeError("non-finite full-horizon radial log kappa")
        D = min(self.k, self.n - self.k)
        self.log_vj = torch.as_tensor(
            [math.log(C.shell_size(self.n, self.k, a)) if a > 0 else 0.0
             for a in range(D + 1)], dtype=torch.float64, device=self.device)

    # -- kernels -------------------------------------------------------------

    def radial_kappa(self, s, t):
        """``(q, kappa)`` of the reference over ``[s, t]``."""
        clocks = self.schedule.integrated(s, t)
        tot = sum(clocks.values())
        if tot <= 0.0:                       # degenerate interval: keep the
            clocks = {1: 1e-12}              # convention of ``kappa_grid``
        return C.radial_orbit_kernel(self.n, self.k, clocks)

    def kappa_grid(self, ts):
        raise NotImplementedError(
            "radial clocks are time-inhomogeneous; use bridge_kernels/radial_kappa")

    def bridge_kernels(self, steps):
        """``(log kappa_{0,t_s}, log kappa_{t_s,1})`` on the training grid.

        These are exactly the two inputs ``sample_bridge_direct`` consumes:
        ``P(Z | x_0, X_1) ~ kappa_{0,t}(d(x_0,Z)) kappa_{t,1}(d(Z,X_1))``.
        """
        ts = np.arange(steps, dtype=np.float64) / steps
        k0 = np.stack([self.radial_kappa(0.0, float(t))[1] for t in ts])
        k1 = np.stack([self.radial_kappa(float(t), 1.0)[1] for t in ts])
        lk0 = torch.as_tensor(np.log(np.maximum(k0, 1e-300)),
                              dtype=torch.float64, device=self.device)
        lk1 = torch.as_tensor(np.log(np.maximum(k1, 1e-300)),
                              dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(lk0).all() and torch.isfinite(lk1).all()):
            raise RuntimeError("non-finite log kappa on the radial bridge grid")
        return lk0, lk1


# ============================================================================
# 3.  elementary symmetric polynomial DP             (playbook 19, 20)
# ============================================================================


def log_esp_prefix(logw, jmax):
    """``E[b, i, r] = log e_r(w_1..w_i)`` for ``r <= jmax``.  ``(B, m+1, jmax+1)``.

    One pass gives the normalizers of *every* shell ``r <= jmax`` at once, which
    is why a single DP serves the whole active shell set.
    """
    B, m = logw.shape
    if jmax < 0 or jmax > m:
        raise ValueError(f"jmax={jmax} outside 0..{m}")
    # written out-of-place: the same DP is differentiated through in training,
    # where in-place slice writes would break autograd versioning.
    # LOG_ZERO rather than -inf: cells with r > i are structurally zero and
    # ``logaddexp(-inf, -inf)`` has a NaN derivative, which would poison the
    # gradient of every site logit even though those cells carry no mass.
    # exp(LOG_ZERO - anything finite) underflows to 0 in float64, so the
    # forward values and the gradients of the reachable cells are unchanged.
    rows = [torch.cat([logw.new_zeros(B, 1),
                       logw.new_full((B, jmax), LOG_ZERO)], dim=1)]
    pad = logw.new_full((B, 1), LOG_ZERO)
    for i in range(1, m + 1):
        prev = rows[-1]
        take = torch.cat([pad, prev[:, :-1]], dim=1) + logw[:, i - 1:i]
        rows.append(torch.logaddexp(prev, take))
    return torch.stack(rows, dim=1)


def log_esp_select(logw, sel, jrow, E=None):
    """``log q(S) = sum_{i in S} logw_i - log e_j(w)`` for a chosen subset."""
    if E is None:
        E = log_esp_prefix(logw, int(jrow.max()))
    num = torch.where(sel, logw, torch.zeros_like(logw)).sum(dim=1)
    den = E[:, -1, :].gather(1, jrow[:, None]).squeeze(1)
    return num - den


def esp_sample(logw, jrow, generator=None, E=None):
    """Exact backward sample of ``P(S) ~ prod_{i in S} w_i`` with ``|S| = j``.

    Walking down from ``(i=m, r=j)``, item ``i`` is included with probability
    ``w_i E[i-1, r-1] / E[i, r]``.  ``r`` is per row, so different rows may be
    drawing from different shells in the same call.
    """
    B, m = logw.shape
    jmax = int(jrow.max()) if jrow.numel() else 0
    if E is None:
        E = log_esp_prefix(logw, jmax)
    r = jrow.clone().long()
    if bool((r < 0).any()) or bool((r > m).any()):
        raise ValueError("shell size outside the candidate set")
    sel = torch.zeros(B, m, dtype=torch.bool, device=logw.device)
    for i in range(m, 0, -1):
        rm1 = (r - 1).clamp(min=0)
        num = logw[:, i - 1] + E[:, i - 1, :].gather(1, rm1[:, None]).squeeze(1)
        den = E[:, i, :].gather(1, r[:, None]).squeeze(1)
        lp = num - den
        u = torch.rand(B, device=logw.device, dtype=logw.dtype,
                       generator=generator).clamp_min(1e-300).log()
        inc = (u < lp) & (r > 0)
        sel[:, i - 1] = inc
        r = r - inc.long()
    if not bool((r == 0).all()):
        raise RuntimeError("ESP sampler did not place the requested cardinality")
    return sel


# ============================================================================
# 4.  shell geometry on batched states               (playbook 12, 15)
# ============================================================================


def occupied_empty(x):
    """``(occ, emp)`` index tensors, ``(B, k)`` and ``(B, n-k)``."""
    B, n = x.shape
    k = int(x[0].sum())
    order = torch.argsort(x.to(torch.int8), dim=1, descending=True, stable=True)
    return order[:, :k].contiguous(), order[:, k:].contiguous()


def sample_uniform_shell(x, jrow, generator=None):
    """``(R, A, valid)`` for a uniform distance-``j`` move, padded to ``jmax``.

    Removes ``j`` occupied sites and fills ``j`` empty ones, both uniformly
    without replacement, so the composition is preserved exactly.
    """
    B, n = x.shape
    jmax = int(jrow.max())
    key = torch.rand(B, n, device=x.device, dtype=torch.float64,
                     generator=generator)
    big = torch.full_like(key, 2.0)
    occ = torch.where(x == 1, key, big).argsort(dim=1)[:, :jmax]
    emp = torch.where(x == 0, key, big).argsort(dim=1)[:, :jmax]
    valid = torch.arange(jmax, device=x.device)[None, :] < jrow[:, None]
    return occ, emp, valid


def pbc_dist2(positions, cell):
    """``(n, n)`` minimum-image squared distances under the supercell."""
    P = np.asarray(positions, dtype=np.float64)
    H = np.asarray(cell, dtype=np.float64)
    frac = P @ np.linalg.inv(H)
    df = frac[:, None, :] - frac[None, :, :]
    df -= np.round(df)
    d = df @ H
    return (d * d).sum(-1)


@torch.no_grad()
def ce_pair_cost(space, X, n_sub=4096, cache="", seed=0):
    """``(n, n)`` mean transposition penalty ``E[(E(g_ra X1) - E(X1))/tau]``.

    This is a *fixed* table: it is precomputed once from the target ensemble and
    is thereafter a constant function of the site pair, exactly like the lattice
    geometry.  It never inspects the ``X_1`` of the draw being labelled, so by
    playbook 12 every pairing built from it still has the exact conditional mean
    ``phi(y)/phi(x)`` and the choice remains a pure variance lever.
    """
    if cache and os.path.exists(cache):
        return np.load(cache)["cost"]
    n = space.n
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), size=min(n_sub, len(X)), replace=False)
    Xs = torch.as_tensor(X[sel].astype(np.int64), device=space.device)
    E0 = space.energy.energy_torch(Xs).to(torch.float64)
    cost = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for r in range(n):
        for a in range(r + 1, n):
            Y = Xs.clone()
            Y[:, [r, a]] = Y[:, [a, r]]
            dE = (space.energy.energy_torch(Y).to(torch.float64) - E0) / space.tau
            cost[r, a] = cost[a, r] = float(dE.mean())
    print(f"  CE pair-cost table {n}x{n} from {len(sel)} samples "
          f"({time.time() - t0:.0f}s)  range [{cost.min():.2f}, {cost.max():.2f}]")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, cost=cost, n_sub=len(sel))
    return cost


def pair_geometry(R, A, valid, dist2):
    """Reorder ``A`` so that ``sum_m d_PBC(R_m, A_m)^2`` is minimal (playbook 12.2).

    The assignment reads only the edge ``(R, A)`` and the fixed lattice, never
    ``X_1``, so every pairing has the same exact conditional mean label and the
    choice is a pure variance lever.  ``dist2`` may be any edge-measurable cost
    matrix -- PBC squared distance (playbook 12.2) or the CE pair-cost table.
    """
    from scipy.optimize import linear_sum_assignment
    Rc, Ac, vc = R.cpu().numpy(), A.cpu().numpy(), valid.cpu().numpy()
    out = Ac.copy()
    for b in range(Rc.shape[0]):
        m = vc[b]
        j = int(m.sum())
        if j < 2:
            continue
        r, a = Rc[b, m], Ac[b, m]
        _, col = linear_sum_assignment(dist2[np.ix_(r, a)])
        out[b, m] = a[col]
    return torch.as_tensor(out, device=A.device, dtype=A.dtype)


def apply_pairs(z, R, A, valid):
    """``g z`` for ``g = prod_m (R_m A_m)``, a product of disjoint transpositions.

    ``g`` is an involution, so ``(g z)[R] = z[A]``, ``(g z)[A] = z[R]`` and every
    other site is fixed.  Padded columns are routed to a trash slot.
    """
    B, n = z.shape
    perm = torch.arange(n + 1, device=z.device).expand(B, n + 1).clone()
    trash = torch.full_like(R, n)
    Rp = torch.where(valid, R, trash)
    Ap = torch.where(valid, A, trash)
    perm.scatter_(1, Rp, Ap)
    perm.scatter_(1, Ap, Rp)
    return z.gather(1, perm[:, :n])


def apply_masks(x, occ, emp, sel_r, sel_a):
    """Shell move given boolean selections over the ``occ`` / ``emp`` columns."""
    y = x.scatter(1, occ, torch.where(
        sel_r, torch.zeros_like(sel_r, dtype=x.dtype), x.gather(1, occ)))
    return y.scatter(1, emp, torch.where(
        sel_a, torch.ones_like(sel_a, dtype=x.dtype), y.gather(1, emp)))


def shell_move(x, R, A, valid):
    """Apply the shell move to the *current* state: drop ``R``, fill ``A``.

    ``R`` must index occupied sites of ``x`` and ``A`` empty ones, so this
    agrees with :func:`apply_pairs` on ``x`` itself; it is kept separate because
    it is cheaper and states the intent.  Padded columns are routed to a trash
    slot and discarded.
    """
    B, n = x.shape
    trash = torch.full_like(R, n)
    Rp = torch.where(valid, R, trash)
    Ap = torch.where(valid, A, trash)
    y = torch.cat([x, torch.zeros(B, 1, dtype=x.dtype, device=x.device)], dim=1)
    y.scatter_(1, Rp, torch.zeros_like(Rp, dtype=x.dtype))
    y.scatter_(1, Ap, torch.ones_like(Ap, dtype=x.dtype))
    return y[:, :n].contiguous()


# ============================================================================
# 5.  terminal label                                 (playbook 11, 12)
# ============================================================================


@torch.no_grad()
def terminal_log_label_shell(space, X1, E1, R, A, valid, check=False):
    """``log Lambda_g(X_1)`` for the shell edge represented by ``(R, A)``.

    ``g`` is built from the edge alone -- never from ``X_1`` -- so the exact
    regression identity ``E[Lambda_g(X_1) | X_t = x] = phi_t(y)/phi_t(x)`` holds
    for any bijection ``R <-> A`` (playbook 12).
    """
    Xg = apply_pairs(X1, R, A, valid)
    check_states(Xg, space.k, "terminal g X1")
    Eg = space.energy.energy_torch(Xg).to(torch.float64)
    if not bool(torch.isfinite(Eg).all()):
        raise RuntimeError("non-finite CE energy in radial terminal label")

    x0b = space.x0.unsqueeze(0)
    d1 = space.distance(x0b, X1).long()
    dg = space.distance(x0b, Xg).long()
    out = (-(Eg - E1.to(torch.float64)) / space.tau
           + space.log_kappa_full[d1] - space.log_kappa_full[dg])
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("non-finite radial terminal label")
    if check and bool((~valid).all(dim=1).any()):
        idle = (~valid).all(dim=1)
        if float(out[idle].abs().max() if int(idle.sum()) else 0.0) > 1e-9:
            raise RuntimeError("empty shell move did not give Lambda = 1")
    return out, Xg


# ============================================================================
# 6.  normalized shell controller                    (playbook 17, 21, 22)
# ============================================================================


class RadialController(torch.nn.Module):
    """``u_j(y,x,t) = lambda_j(x,t) q_j(y | x,t)`` with exact normalization.

    The head emits, per shell, one total-rate log multiplier ``b_j`` and two
    sets of ``N`` site logits.  For five shells at ``N = 64`` that is
    ``5 * (1 + 128) = 645`` outputs against the ``64^2 = 4096`` of the pairwise
    ``SwapController``.  Zero initialization gives ``b_j = 0`` and uniform
    subset logits, hence ``q_j = 1/v_j`` and ``u = r`` exactly at the reference.
    """

    def __init__(self, n, shells, hidden=512, n_freq=4):
        super().__init__()
        self.n = int(n)
        self.shells = tuple(int(j) for j in shells)
        self.S = len(self.shells)
        self.n_freq = int(n_freq)
        out = self.S * (1 + 2 * self.n)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n + 2 + 2 * n_freq, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, out),
        ).to(torch.float32)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, x):
        """``(b, s_minus, s_plus)`` of shapes ``(B,S)``, ``(B,S,n)``, ``(B,S,n)``."""
        s = 2.0 * x.to(torch.float32) - 1.0
        t = t.to(torch.float32).reshape(-1, 1)
        kk = torch.arange(1, self.n_freq + 1, device=t.device,
                          dtype=t.dtype)[None, :]
        feats = [s, t, 1.0 - t,
                 torch.sin(math.pi * kk * t), torch.cos(math.pi * kk * t)]
        o = self.net(torch.cat(feats, dim=-1)).view(-1, self.S, 1 + 2 * self.n)
        return o[:, :, 0], o[:, :, 1:1 + self.n], o[:, :, 1 + self.n:]


def shell_logits(net, t, x, clamp=10.0):
    """Controller outputs gathered onto occupied/empty sites."""
    b, sm, sp = net(t, x)
    occ, emp = occupied_empty(x)
    b = b.to(torch.float64).clamp(-clamp, clamp)
    lw_rem = sm.to(torch.float64).clamp(-clamp, clamp).gather(
        2, occ[:, None, :].expand(-1, len(net.shells), -1))
    lw_add = sp.to(torch.float64).clamp(-clamp, clamp).gather(
        2, emp[:, None, :].expand(-1, len(net.shells), -1))
    return b, lw_rem, lw_add, occ, emp


def log_q_shell(lw_rem, lw_add, sel_r, sel_a, jrow, si):
    """``log q_j(y|x,t) = log q^-(R) + log q^+(A)`` for the selected shell row."""
    B = lw_rem.shape[0]
    ar = torch.arange(B, device=lw_rem.device)
    wr, wa = lw_rem[ar, si], lw_add[ar, si]
    return (log_esp_select(wr, sel_r, jrow)
            + log_esp_select(wa, sel_a, jrow))


# ============================================================================
# 7.  controlled simulation                          (playbook 25)
# ============================================================================


@torch.no_grad()
def simulate_radial(space, net, batch, steps, generator=None, x0=None,
                    stats=None):
    """Frozen-rate controlled CTMC, at most one jump per grid bin.

    Within a bin the controller is piecewise constant, the total escape rate is
    ``R = sum_j lambda_j`` exactly (no shell enumeration), the shell is drawn
    proportional to ``lambda_j`` and the destination exactly from ``q_j``.  The
    per-bin multi-event probability is recorded in ``stats`` so that the
    single-jump discretization can be audited rather than assumed.
    """
    if x0 is None:
        x = space.x0.unsqueeze(0).expand(batch, -1).clone()
    else:
        x = x0.long().to(space.device).clone()
        if x.shape[0] != batch:
            x = x.expand(batch, -1).clone()
    check_states(x, space.k, "radial sim init")

    shells = torch.as_tensor(net.shells, device=space.device)
    dt = 1.0 / steps
    n_jump = 0
    miss = 0.0
    for s in range(steps):
        t = s * dt
        gam = torch.as_tensor(space.schedule.rates(t), dtype=torch.float64,
                              device=space.device)
        tt = torch.full((batch,), t, device=space.device, dtype=torch.float64)
        b, lw_rem, lw_add, occ, emp = shell_logits(net, tt, x)
        lam = gam[None, :] * torch.exp(b)                       # (B, S)
        R = lam.sum(dim=1)
        if not bool(torch.isfinite(R).all()) or bool((R < 0).any()):
            raise RuntimeError("non-finite or negative controlled total rate")
        p_stay = torch.exp(-R * dt)
        # P(>=2 events | bin) audit for the one-jump-per-bin discretization
        miss = max(miss, float((1.0 - p_stay * (1.0 + R * dt)).max()))

        jump = torch.rand(batch, device=space.device, dtype=torch.float64,
                          generator=generator) > p_stay
        if not bool(jump.any()):
            continue
        rows = torch.nonzero(jump, as_tuple=False).squeeze(1)
        si = torch.multinomial(lam[rows], 1, generator=generator).squeeze(1)
        jrow = shells[si]
        jmax = int(jrow.max())
        wr = lw_rem[rows, si][:, :space.k]
        wa = lw_add[rows, si][:, :space.n - space.k]
        sel_r = esp_sample(wr, jrow, generator=generator,
                           E=log_esp_prefix(wr, jmax))
        sel_a = esp_sample(wa, jrow, generator=generator,
                           E=log_esp_prefix(wa, jmax))
        xr = apply_masks(x[rows], occ[rows], emp[rows], sel_r, sel_a)
        x = x.clone()
        x[rows] = xr
        n_jump += int(rows.numel())

    check_states(x, space.k, "radial sim final")
    if stats is not None:
        stats["events"] = stats.get("events", 0) + n_jump
        stats["max_two_event_prob"] = max(stats.get("max_two_event_prob", 0.0),
                                          miss)
    return x


# ============================================================================
# 8.  CLI
# ============================================================================


def build_space(args, energy=None, tables=None):
    size = tuple(args.size)
    if tables is None:
        tables = build_ce_tables(size, verbose=True)
    if energy is None:
        energy = TorchCuAuEnergy(tables, device=args.device)
    sched = named_schedule(args.schedule, args.gamma_total, args.shells)
    sp = RadialCuAuSpace(size=size, temp_k=args.temp, schedule=sched,
                         device=args.device, energy=energy, verbose=True)
    sp.dist2 = pbc_dist2(sp.positions, tables["cell"])
    return sp, tables


def spectral_gap(n, k, weights, gamma_total):
    """Gap of the time-homogeneous reduced generator ``sum_j w_j gtot (B_j-I)``.

    The orbit chain is reversible with respect to the uniform canonical law, so
    every eigenvalue is real; the gap is ``-max Re lambda`` over the non-trivial
    spectrum and sets the reference mixing time on the distance coordinate.
    """
    D = min(k, n - k)
    G = np.zeros((D + 1, D + 1), dtype=np.float64)
    for j, w in weights.items():
        if w > 0.0:
            G += gamma_total * float(w) * (
                C.shell_orbit_matrix_cached(n, k, int(j)) - np.eye(D + 1))
    ev = np.sort(np.linalg.eigvals(G).real)[::-1]
    return float(-ev[1]), ev[:4].tolist()


def load_pool(path, k):
    """PT configuration pool; every row must sit in the exact ``k`` sector."""
    z = np.load(path)
    X = z["X"].astype(np.int64)
    bad = int((X.sum(1) != k).sum())
    if bad:
        raise RuntimeError(f"{bad} pool states leave the k={k} sector")
    return X, z["E"].astype(np.float64)


def cmd_ref_diag(args):
    """Reference-only diagnostics: no controller, no training (playbook 30)."""
    sp, _ = build_space(args)
    D = min(sp.k, sp.n - sp.k)
    v = np.array([C.shell_size(sp.n, sp.k, a) for a in range(D + 1)],
                 dtype=np.float64)
    unif = v / v.sum()

    out = {"size": list(sp.size), "N": sp.n, "k": sp.k,
           "temperature_K": sp.temp_k, "gamma_total": args.gamma_total,
           "shells": list(args.shells), "rows": []}
    print(f"CuAu radial ref-diagnostics  N={sp.n} k={sp.k}  "
          f"gamma_total={args.gamma_total}")

    # 5. where the PT target actually lives, as seen from the source
    if args.pool:
        X, _ = load_pool(args.pool, sp.k)
        Xt = torch.as_tensor(X, device=sp.device)
        d = sp.distance(sp.x0.unsqueeze(0), Xt).cpu().numpy()
        hist = np.bincount(d, minlength=D + 1) / len(d)
        out["pt_source_distance"] = {
            "pool": args.pool, "n_samples": int(len(d)),
            "hist": hist.tolist(), "mean": float(d.mean()),
            "min": int(d.min()), "max": int(d.max())}
        print(f"  PT source distance  <d> {d.mean():.3f}  "
              f"range [{d.min()},{d.max()}]  ({len(d)} samples)")

    for name in args.compare:
        sch = named_schedule(name, args.gamma_total, args.shells)
        sp.schedule = sch
        q, _ = sp.radial_kappa(0.0, 1.0)
        tv = 0.5 * float(np.abs(q - unif).sum())
        nz = q > 0
        kl = float((q[nz] * np.log(q[nz] / unif[nz])).sum())
        wbar = sch.integrated(0.0, 1.0)
        tot = sum(wbar.values())
        gap, ev = spectral_gap(sp.n, sp.k, {j: w / tot for j, w in wbar.items()},
                               args.gamma_total)
        row = {"schedule": name, "TV_to_uniform": tv, "KL_to_uniform": kl,
               "orbit_law": q.tolist(),
               "mean_distance": float((np.arange(D + 1) * q).sum()),
               "support_frac": float((q > 1e-12).mean()),
               "spectral_gap": gap, "top_eigs": ev,
               "time_avg_mixture": {str(j): w / tot for j, w in wbar.items()},
               "describe": sch.describe()}
        if "pt_source_distance" in out:
            h = np.asarray(out["pt_source_distance"]["hist"])
            row["TV_to_PT_source"] = 0.5 * float(np.abs(q - h).sum())
            m = q > 1e-300
            row["log_cover_PT"] = float(np.log(q[m & (h > 0)]).min()) \
                if bool((m & (h > 0)).any()) else float("-inf")
        out["rows"].append(row)
        print(f"  {name:6s}  TV(q,unif) {tv:.6f}  KL {kl:.6f}  "
              f"<d> {row['mean_distance']:.3f}  gap {gap:.4f}"
              + (f"  TV(q,PT) {row['TV_to_PT_source']:.4f}"
                 if "TV_to_PT_source" in row else ""))
    if args.gamma_sweep:
        out["gamma_sweep"] = []
        print("  gamma_tot sweep (coverage vs reference event budget)")
        print(f"    {'sched':>6s} {'gtot':>6s} {'<j>':>5s} {'activity':>8s} "
              f"{'<d>':>7s} {'TV_unif':>8s} {'TV_PT':>7s}")
        for name in args.compare:
            sch0 = named_schedule(name, 1.0, args.shells)
            wbar = sch0.integrated(0.0, 1.0)
            jbar = sum(j * w for j, w in wbar.items()) / sum(wbar.values())
            for g in args.gamma_sweep:
                sp.schedule = named_schedule(name, float(g), args.shells)
                q, _ = sp.radial_kappa(0.0, 1.0)
                row = {"schedule": name, "gamma_total": float(g),
                       "mean_shell": float(jbar), "activity": float(g * jbar),
                       "mean_distance": float((np.arange(D + 1) * q).sum()),
                       "TV_to_uniform": 0.5 * float(np.abs(q - unif).sum())}
                if "pt_source_distance" in out:
                    h = np.asarray(out["pt_source_distance"]["hist"])
                    row["TV_to_PT_source"] = 0.5 * float(np.abs(q - h).sum())
                out["gamma_sweep"].append(row)
                print(f"    {name:>6s} {g:6.1f} {jbar:5.2f} {g*jbar:8.1f} "
                      f"{row['mean_distance']:7.3f} {row['TV_to_uniform']:8.5f} "
                      + (f"{row['TV_to_PT_source']:7.4f}"
                         if 'TV_to_PT_source' in row else ""))

    os.makedirs("json", exist_ok=True)
    path = args.out or f"json/results_cuau_radial_refdiag_{sp.n}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")
    return 0


def cmd_label_diag(args):
    """Terminal-label distribution by shell and time (playbook 31)."""
    sp, _ = build_space(args)
    X, Epool = load_pool(args.pool, sp.k)
    if args.pairing == "ce":
        sp.ce_cost = ce_pair_cost(
            sp, X, n_sub=args.ce_sub, seed=args.seed,
            cache=f"data/cuau/pair_cost_{sp.n}_{int(sp.temp_k)}K.npz")
    Xt = torch.as_tensor(X, device=sp.device)
    Et = torch.as_tensor(Epool, device=sp.device)
    gen = torch.Generator(device=sp.device).manual_seed(args.seed)
    steps = args.steps
    log_kap0, log_kap1 = sp.bridge_kernels(steps)

    out = {"size": list(sp.size), "N": sp.n, "k": sp.k,
           "temperature_K": sp.temp_k, "schedule": args.schedule,
           "gamma_total": args.gamma_total, "pool": args.pool,
           "pairing": args.pairing,
           "n_draws": args.draws, "clamp": args.clamp, "rows": []}
    print(f"CuAu radial label-diagnostics  N={sp.n}  schedule {args.schedule}  "
          f"gamma_total={args.gamma_total}  draws={args.draws}")
    print(f"  {'j':>3s} {'t':>5s} {'mean':>9s} {'sd':>9s} {'p01':>9s} "
          f"{'p50':>9s} {'p99':>9s} {'|clamp|':>8s} {'sd_E':>8s} {'sd_K':>8s}")
    for j in args.shells:
        for t in args.times:
            t_idx = torch.full((args.draws,), min(int(t * steps), steps - 1),
                               dtype=torch.long, device=sp.device)
            sel = torch.randint(len(X), (args.draws,), generator=gen,
                                device=sp.device)
            X1, E1 = Xt[sel], Et[sel]
            Xb = sample_bridge_direct(sp, X1, t_idx, log_kap0, log_kap1,
                                      generator=gen)
            check_states(Xb, sp.k, "label-diag bridge")
            jrow = torch.full((args.draws,), int(j), dtype=torch.long,
                              device=sp.device)
            R, A, valid = sample_uniform_shell(Xb, jrow, generator=gen)
            if args.pairing == "geometry":
                A = pair_geometry(R, A, valid, sp.dist2)
            elif args.pairing == "ce":
                A = pair_geometry(R, A, valid, sp.ce_cost)
            elif args.pairing == "index":
                R = torch.where(valid, R, torch.full_like(R, sp.n)).sort(1).values
                A = torch.where(valid, A, torch.full_like(A, sp.n)).sort(1).values
                R = torch.where(valid, R, torch.zeros_like(R))
                A = torch.where(valid, A, torch.zeros_like(A))
            lab, Xg = terminal_log_label_shell(sp, X1, E1, R, A, valid)
            Eg = sp.energy.energy_torch(Xg).to(torch.float64)
            e_term = (-(Eg - E1.to(torch.float64)) / sp.tau)
            k_term = lab - e_term
            a = lab.cpu().numpy()
            qs = np.quantile(a, [0.01, 0.5, 0.99])
            row = {"j": int(j), "t": float(t), "mean": float(a.mean()),
                   "sd": float(a.std(ddof=1)),
                   "p01": float(qs[0]), "p50": float(qs[1]), "p99": float(qs[2]),
                   "min": float(a.min()), "max": float(a.max()),
                   "frac_clamped": float((np.abs(a) > args.clamp).mean()),
                   "frac_clamped_hi": float((a > args.clamp).mean()),
                   "frac_clamped_lo": float((a < -args.clamp).mean()),
                   "clamp_mass_bias": float(
                       np.exp(a[a > args.clamp] - a.max()).sum()
                       / max(np.exp(a - a.max()).sum(), 1e-300)),
                   "sd_energy_term": float(e_term.cpu().numpy().std(ddof=1)),
                   "sd_kernel_term": float(k_term.cpu().numpy().std(ddof=1)),
                   "ess_frac": float(np.exp(
                       2 * torch.logsumexp(torch.as_tensor(a), 0).item()
                       - torch.logsumexp(torch.as_tensor(2 * a), 0).item()
                       - math.log(len(a))))}
            out["rows"].append(row)
            print(f"  {j:3d} {t:5.2f} {row['mean']:9.3f} {row['sd']:9.3f} "
                  f"{row['p01']:9.3f} {row['p50']:9.3f} {row['p99']:9.3f} "
                  f"{row['frac_clamped']:8.4f} {row['sd_energy_term']:8.3f} "
                  f"{row['sd_kernel_term']:8.3f}")
    os.makedirs("json", exist_ok=True)
    path = args.out or (f"json/results_cuau_radial_labeldiag_{sp.n}_"
                        f"{args.schedule}_{args.pairing}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")
    return 0


def sel_mask(sites, valid, cols, n):
    """Boolean mask over ``cols`` marking the padded site set ``sites``."""
    B = cols.shape[0]
    ind = torch.zeros(B, n + 1, dtype=torch.bool, device=cols.device)
    s = torch.where(valid, sites, torch.full_like(sites, n))
    ind.scatter_(1, s, torch.ones_like(s, dtype=torch.bool))
    return ind[:, :n].gather(1, cols)


def cmd_train(args):
    """Radial IASBS with the sampled Poisson-Bregman loss (playbook 23, 48).

    Dirac AS at ``x0`` (playbook 13): the source is a point mass, so there is no
    corrector network -- ``net_h`` and its optimizer disappear entirely and the
    whole reference-side label is the ``log kappa`` ratio already carried by
    :func:`terminal_log_label_shell`.

    The loss is the one-edge unbiased estimator

        L(x) = sum_j lambda_j(x,t) - gamma_tot * Lambda_J(Y,x) log m_J(Y,x),

    with ``J ~ gamma_j(t)/gamma_tot`` and ``Y`` uniform on the reference shell
    ``N_J(x)``.  The positive term is exact -- it never touches ``v_j`` -- and
    only the label term is Monte Carlo, which is what makes ``j = 16`` (a shell
    of 3.6e17 states) representable at all.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sp, tables = build_space(args)
    n, k = sp.n, sp.k
    shells = tuple(int(j) for j in args.shells)
    steps = args.steps

    ref = None
    if args.ref_json and os.path.exists(args.ref_json):
        with open(args.ref_json) as f:
            ref = json.load(f)
        if list(ref.get("size", [])) != list(sp.size):
            raise RuntimeError(f"{args.ref_json} is for size {ref.get('size')}"
                               f", not {list(sp.size)}")
        print(f"  PT reference: <E>/N {ref['E_per_N_meV']:.4f} +/- "
              f"{ref['E_per_N_meV_sd']:.4f} meV   Qmax {ref['Qmax']:.4f} "
              f"+/- {ref['Qmax_sd']:.4f}")

    if args.pairing == "ce":
        if not args.pool:
            raise RuntimeError("--pairing ce needs --pool for the cost table")
        Xp, _ = load_pool(args.pool, k)
        sp.ce_cost = ce_pair_cost(
            sp, Xp, n_sub=args.ce_sub, seed=args.seed,
            cache=f"data/cuau/pair_cost_{n}_{int(sp.temp_k)}K.npz")

    net = RadialController(n, shells, hidden=args.hidden).to(sp.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)
    gen = torch.Generator(device=sp.device).manual_seed(args.seed + 1234)
    log_kap0, log_kap1 = sp.bridge_kernels(steps)

    dt = 1.0 / steps
    gam_grid = torch.as_tensor(
        np.array([sp.schedule.rates(s * dt) for s in range(steps)]),
        dtype=torch.float64, device=sp.device)                 # (steps, S)
    logv = torch.as_tensor([math.log(C.shell_size(n, k, j)) for j in shells],
                           dtype=torch.float64, device=sp.device)
    sh_t = torch.as_tensor(shells, device=sp.device)

    print(f"CuAu radial train  N={n} k={k} T={sp.temp_k}K  "
          f"schedule {args.schedule} gamma_tot={args.gamma_total} "
          f"shells {shells}  steps={steps}  params={n_par}  "
          f"source=Dirac(x0)  pairing={args.pairing}")

    buf, hist = [], []
    t_start = time.time()
    loss = torch.zeros(())
    n_skip = 0
    for it in range(1, args.iters + 1):
        sp.energy.tag = "train"
        st = {}
        X1n = simulate_radial(sp, net, args.batch, steps, generator=gen,
                              stats=st)
        with torch.no_grad():
            E1n = sp.energy.energy_torch(X1n).to(torch.float64)
        buf.append((X1n.detach(), E1n.detach()))
        if len(buf) > args.buffer:
            buf.pop(0)
        pool1 = torch.cat([b[0] for b in buf], 0)
        poole = torch.cat([b[1] for b in buf], 0)

        for _ in range(args.inner):
            sel = torch.randint(len(pool1), (args.mb,), device=sp.device,
                                generator=gen)
            X1, E1 = pool1[sel], poole[sel]
            ti = torch.randint(steps, (args.mb,), device=sp.device,
                               generator=gen)
            tt = ti.to(torch.float64) * dt
            with torch.no_grad():
                Xt = sample_bridge_direct(sp, X1, ti, log_kap0, log_kap1,
                                          generator=gen)
            gam = gam_grid[ti]                                  # (B, S)
            gtot = gam.sum(dim=1)
            with torch.no_grad():
                si = torch.multinomial(gam, 1, generator=gen).squeeze(1)
                jrow = sh_t[si]
                R, A, valid = sample_uniform_shell(Xt, jrow, generator=gen)
                Ag = A
                if args.pairing == "geometry":
                    Ag = pair_geometry(R, A, valid, sp.dist2)
                elif args.pairing == "ce":
                    Ag = pair_geometry(R, A, valid, sp.ce_cost)
                lab, _ = terminal_log_label_shell(sp, X1, E1, R, Ag, valid)
                Lam = torch.exp(lab.clamp(-args.clamp, args.clamp))

            b, lw_rem, lw_add, occ, emp = shell_logits(net, tt, Xt)
            lam = gam * torch.exp(b)                            # (B, S)
            sel_r = sel_mask(R, valid, occ, n)
            sel_a = sel_mask(A, valid, emp, n)
            if not bool((sel_r.sum(1) == jrow).all()
                        and (sel_a.sum(1) == jrow).all()):
                raise RuntimeError("shell selection mask lost cardinality")
            ar = torch.arange(args.mb, device=sp.device)
            log_m = (b[ar, si] + logv[si]
                     + log_q_shell(lw_rem, lw_add, sel_r, sel_a, jrow, si))
            loss = (lam.sum(dim=1) - gtot * Lam * log_m).mean()
            opt.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                n_skip += 1
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            if not bool(torch.isfinite(gn)):
                # numerical guard only: a non-finite gradient is discarded and
                # counted, never silently rescaled into a different objective.
                opt.zero_grad(set_to_none=True)
                n_skip += 1
                continue
            opt.step()
            sch.step()

        if it % args.eval_every == 0 or it == args.iters:
            sp.energy.tag = "eval"
            se = {}
            with torch.no_grad():
                Xe = simulate_radial(sp, net, args.n_eval, steps,
                                     generator=gen, stats=se)
                obs = sampled_observables(sp, Xe, sp.temp_k)
            obs["iter"] = it
            obs["loss"] = float(loss)
            obs["secs"] = round(time.time() - t_start, 1)
            obs["violations"] = int((Xe.sum(1) != k).sum())
            obs["events_per_path"] = se["events"] / args.n_eval
            obs["max_two_event_prob"] = se["max_two_event_prob"]
            obs["train_events_per_path"] = st["events"] / args.batch
            obs["skipped_steps"] = n_skip
            if ref is not None:
                obs["dE_per_N_meV"] = abs(obs["E_per_N_meV"]
                                          - ref["E_per_N_meV"])
                obs["dQmax"] = abs(obs["Qmax"] - ref["Qmax"])
            hist.append(obs)
            msg = (f"  it {it:5d}  loss {float(loss):12.4f}  "
                   f"<E>/N {obs['E_per_N_meV']:9.4f} meV  "
                   f"Qmax {obs['Qmax']:.4f}  "
                   f"ev/path {obs['events_per_path']:.1f}")
            if ref is not None:
                msg += (f"  dE {obs['dE_per_N_meV']:7.4f}  "
                        f"dQ {obs['dQmax']:.4f}")
            print(msg + f"  {obs['secs']:.0f}s", flush=True)
            sp.energy.tag = "train"

    sp.energy.tag = "final"
    sf = {}
    with torch.no_grad():
        Xs = simulate_radial(sp, net, args.n_samples, steps, generator=gen,
                             stats=sf)
        fin = sampled_observables(sp, Xs, sp.temp_k)
    viol = int((Xs.sum(1) != k).sum())
    res = {"provenance": provenance(), "config": arg_config(args),
           "params": n_par, "history": hist, "final": fin,
           "violations": viol, "reference": ref,
           "schedule_describe": sp.schedule.describe(),
           "schedule": sp.schedule.as_dict(),
           "events_per_path": sf["events"] / args.n_samples,
           "max_two_event_prob": sf["max_two_event_prob"],
           "skipped_steps": n_skip,
           "total_steps": args.iters * args.inner,
           "energy_calls": sp.energy.report()}
    if ref is not None:
        res["dE_per_N_meV"] = abs(fin["E_per_N_meV"] - ref["E_per_N_meV"])
        res["dQmax"] = abs(fin["Qmax"] - ref["Qmax"])
    os.makedirs("json", exist_ok=True)
    out = args.out or (f"json/results_cuau_radial_{n}_{int(sp.temp_k)}K_"
                       f"{args.schedule}_g{int(args.gamma_total)}_"
                       f"s{args.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n  final <E>/N {fin['E_per_N_meV']:.4f} +/- "
          f"{fin['E_per_N_meV_sem']:.4f} meV/atom   "
          f"Qmax {fin['Qmax']:.4f} +/- {fin['Qmax_sem']:.4f}")
    print(f"  wrote {out}")
    if args.ckpt_dir:
        pth = C.save_ckpt(args.ckpt_dir,
                          args.tag or f"cuaurad{n}_{args.schedule}_s{args.seed}",
                          nets={"net": net},
                          samples=Xs[:65536].to(torch.int8),
                          extra={"config": arg_config(args)})
        print(f"  ckpt -> {pth}")
    print("\n=== GATES ===")
    print(f"M1  constraint violations 0 : {'PASS' if viol == 0 else 'FAIL'} "
          f"({viol})")
    ok = viol == 0
    if ref is not None:
        e_ok = res["dE_per_N_meV"] <= args.e_gate
        q_ok = res["dQmax"] <= args.q_gate
        print(f"M2  |d<E>/N| <= {args.e_gate} meV : "
              f"{'PASS' if e_ok else 'FAIL'} ({res['dE_per_N_meV']:.4f})")
        print(f"M3  |dQmax|   <= {args.q_gate}   : "
              f"{'PASS' if q_ok else 'FAIL'} ({res['dQmax']:.4f})")
        ok = ok and e_ok and q_ok
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser("iasbs.cuau_radial")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_args(q):
        q.add_argument("--size", type=int, nargs=3, default=[4, 4, 4])
        q.add_argument("--temp", type=float, default=500.0)
        q.add_argument("--shells", type=int, nargs="+", default=[1, 2, 4, 8, 16])
        q.add_argument("--schedule", type=str, default="S2")
        q.add_argument("--gamma-total", type=float, default=10.0)
        q.add_argument("--device", type=str, default="cuda")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--out", type=str, default="")

    r = sub.add_parser("ref-diagnostics", help="playbook 30 reference tables")
    common_args(r)
    r.add_argument("--compare", type=str, nargs="+",
                   default=["S0", "S1", "S2", "S3-8", "S3-4"])
    r.add_argument("--pool", type=str, default="",
                   help="npz PT pool for the source-distance law")
    r.add_argument("--gamma-sweep", type=float, nargs="*", default=[],
                   help="reference event budgets to sweep (playbook 29)")
    r.set_defaults(func=cmd_ref_diag)

    d = sub.add_parser("label-diagnostics", help="playbook 31 label study")
    common_args(d)
    d.add_argument("--pool", type=str, required=True)
    d.add_argument("--draws", type=int, default=50000)
    d.add_argument("--steps", type=int, default=512)
    d.add_argument("--clamp", type=float, default=20.0)
    d.add_argument("--times", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9])
    d.add_argument("--pairing", choices=["random", "index", "geometry", "ce"],
                   default="random")
    d.add_argument("--ce-sub", type=int, default=4096,
                   help="pool subsample for the CE pair-cost table")
    d.set_defaults(func=cmd_label_diag)

    t = sub.add_parser("train", help="playbook 23/48 radial IASBS training")
    common_args(t)
    t.add_argument("--steps", type=int, default=512)
    t.add_argument("--iters", type=int, default=8000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--mb", type=int, default=512)
    t.add_argument("--inner", type=int, default=5)
    t.add_argument("--buffer", type=int, default=20)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--clamp", type=float, default=20.0)
    t.add_argument("--pairing", choices=["random", "index", "geometry", "ce"],
                   default="ce")
    t.add_argument("--pool", type=str, default="",
                   help="npz PT pool used only to build the CE pair-cost table")
    t.add_argument("--ce-sub", type=int, default=4096)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--n-eval", type=int, default=8192)
    t.add_argument("--n-samples", type=int, default=200000)
    t.add_argument("--ref-json", type=str, default="")
    t.add_argument("--e-gate", type=float, default=0.5)
    t.add_argument("--q-gate", type=float, default=0.02)
    t.add_argument("--ckpt-dir", type=str, default="ckpt")
    t.add_argument("--tag", type=str, default="")
    t.set_defaults(func=cmd_train)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
