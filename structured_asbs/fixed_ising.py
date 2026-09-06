"""
fixed_ising.py -- Experiment A, Phase 2.

Fixed-magnetization Ising model on an L x L periodic lattice, n = L^2 binary
sites with exactly k = n/2 occupied.  This validates the *bijective* structured
discrete construction: the reference process is a single occupied<->empty swap,
and the terminal adjoint label uses the swap permutation g_ij, which commutes
with the reference semigroup.

Mathematics actually implemented
-------------------------------
Target        pi(x)      propto exp(-E(x)/tau),  E(x) = -J sum_<ij> s_i s_j.
Reference     r_t(S^ij, S) = gamma_t / (k (n-k)),  total escape rate gamma_t.
Terminal fn   f_1(y) = exp(-E(y)/tau) / p_base_{1|0}(y | x_0).
Value fn      phi_t(x) = E_base[ f_1(X_1) | X_t = x ].
Optimal rate  u*_t(y, x) = r_t(y, x) phi_t(y) / phi_t(x).

Because g_ij is a bijection commuting with the reference generator,
    phi_t(g x) = E_base[ f_1(g X_1) | X_t = x ],
hence under the *optimally controlled* process

    E^{u*}[ Lambda_g(X_1) | X_t = x ] = phi_t(g x) / phi_t(x),
    Lambda_g(X_1) = f_1(g X_1) / f_1(X_1)
                  = exp(-(E(gX_1) - E(X_1))/tau)
                    * p_base(X_1 | X_0) / p_base(g X_1 | X_0).

So the controller multiplier exp(a_theta(t, x, i, j)) is regressed on the
terminal label Lambda_{g_ij}(X_1), with X_1 drawn from the current controlled
process and X_t drawn from the *reference bridge* p_base_{t|0,1}(. | x_0, X_1).
This is reciprocal adjoint matching; at the fixed point the regression target
is exactly the optimal multiplier.

At L = 4 every object above is computable exactly by enumeration, so we can
(a) run the exact optimal control, (b) propagate the law of the *discretised*
controlled chain with zero Monte-Carlo noise, and (c) compare the learned
multiplier against the exact one state by state.

Usage
-----
    python fixed_ising.py exact          # gate A0/A1/A2 with the exact control
    python fixed_ising.py train          # neural controller + all gates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import common as C

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# enumerated state space
# ============================================================================


class FixedIsingSpace:
    """Everything about Omega_{n,k} that can be precomputed by enumeration."""

    def __init__(self, L=4, J=1.0, tau=2.0, gamma=10.0, source=None, device=DEV):
        self.L, self.J, self.tau, self.gamma = L, J, tau, gamma
        self.n = n = L * L
        self.k = k = n // 2
        self.device = device
        self.edges = C.periodic_lattice_edges(L)

        S = C.enumerate_fixed_count(n, k).astype(np.int64)      # (M, n)
        self.M = M = S.shape[0]
        self.S_np = S
        self.S = torch.tensor(S, device=device)
        self.Sf = self.S.to(torch.float32)

        # integer bitmask <-> index
        pow2 = (1 << np.arange(n)).astype(np.int64)
        masks = (S * pow2).sum(axis=1)
        lut = np.full(1 << n, -1, dtype=np.int64)
        lut[masks] = np.arange(M)
        self.masks = masks

        # occupied / empty site indices per state
        occ = np.nonzero(S == 1)[1].reshape(M, k)
        emp = np.nonzero(S == 0)[1].reshape(M, n - k)
        self.occ = torch.tensor(occ, device=device)
        self.emp = torch.tensor(emp, device=device)

        # target index of every legal ordered swap (i occupied -> j empty)
        tgt_mask = (masks[:, None, None]
                    ^ pow2[occ][:, :, None]
                    ^ pow2[emp][:, None, :])
        self.n_edges = k * (n - k)
        tgt = lut[tgt_mask].reshape(M, self.n_edges)
        assert tgt.min() >= 0
        self.tgt = torch.tensor(tgt, device=device)

        # ordered edge -> unordered pair id  (g_ij == g_ji)
        pair_id = np.full((n, n), -1, dtype=np.int64)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_id[i, j] = pair_id[j, i] = len(pairs)
                pairs.append((i, j))
        self.pairs = torch.tensor(np.asarray(pairs), device=device)   # (Np, 2)
        self.n_pairs = len(pairs)
        self.pair_id = torch.tensor(pair_id, device=device)
        ei = np.repeat(occ, n - k, axis=1)                # (M, n_edges)
        ej = np.tile(emp, (1, k))
        self.edge_i = torch.tensor(ei, device=device)
        self.edge_j = torch.tensor(ej, device=device)
        self.edge_pair = torch.tensor(pair_id[ei, ej], device=device)

        # energies and the exact Gibbs target
        E = C.ising_energy_np(S, self.edges, J=J)
        self.E = torch.tensor(E, device=device)
        self.pi = torch.tensor(C.exact_boltzmann_prob(E, tau), device=device)

        # source state (Dirac)
        if source is None:
            src = np.zeros(n, dtype=np.int64)
            src[:k] = 1                                   # top half occupied
        else:
            src = np.asarray(source, dtype=np.int64)
        self.x0 = torch.tensor(src, device=device)
        self.i0 = int(lut[int((src * pow2).sum())])
        self.dist0 = k - (self.S * self.x0).sum(dim=1)    # (M,)  d(x0, .)

        # reference kernel over the full horizon
        self.Gamma_full = gamma
        kap = C.binary_orbit_kernel(n, k, self.Gamma_full)
        self.log_kappa_full = torch.tensor(np.log(kap), device=device)

        # f_1(y) = exp(-E/tau) / p_base(y | x_0)   (log domain, shifted)
        logf1 = -self.E / tau - self.log_kappa_full[self.dist0]
        self.logf1 = logf1 - logf1.max()
        self.f1 = torch.exp(self.logf1)

        # W[x, j] = sum_{x1 : d(x, x1) = j} f1(x1)   ->  phi_t(x) = sum_j k_j W
        self.W = self._distance_weight_matrix(self.f1)

    # -- helpers -------------------------------------------------------------

    def _distance_weight_matrix(self, f, chunk=2048):
        k, M = self.k, self.M
        J = min(k, self.n - k)
        # the (chunk, M) distance block is the largest temporary here; cap it so
        # that L = 5 (M = 5.2e6) does not ask for a 42 GB allocation.
        chunk = max(1, min(chunk, int(1e9) // max(M, 1)))
        W = torch.zeros(M, J + 1, device=self.device, dtype=f.dtype)
        Sf = self.Sf
        for a in range(0, M, chunk):
            b = min(a + chunk, M)
            d = k - torch.round(Sf[a:b] @ Sf.T).long()          # (c, M)
            W[a:b].scatter_add_(1, d, f.unsqueeze(0).expand(b - a, M))
        return W

    def kappa_table(self, gammas):
        """log kappa for a list of integrated rates Gamma."""
        rows = [np.maximum(C.binary_orbit_kernel(self.n, self.k, float(g)),
                           1e-300) for g in gammas]
        return torch.tensor(np.log(np.stack(rows)), device=self.device)

    def phi_grid(self, ts):
        """Exact phi_t(x) for every state and every t in ts.  (T, M)."""
        Gam = np.maximum(self.gamma * (1.0 - np.asarray(ts, dtype=np.float64)),
                         1e-12)
        kap = np.stack([np.maximum(C.binary_orbit_kernel(self.n, self.k, g),
                                   1e-300) for g in Gam])
        K = torch.tensor(kap, device=self.device)             # (T, J+1)
        return K @ self.W.T                                   # (T, M)

    def energies_of(self, X):
        """E for an explicit batch of binary states.  X: (..., n)."""
        s = 2.0 * X.to(torch.float64) - 1.0
        tot = torch.zeros(X.shape[:-1], device=X.device, dtype=torch.float64)
        for i, j in self.edges:
            tot = tot + s[..., i] * s[..., j]
        return -self.J * tot


# ============================================================================
# exact propagation and sampling of the discretised controlled chain
# ============================================================================


def step_probs(space, log_mult, dt):
    """One Euler step of the controlled CTMC (at most one jump per step).

    log_mult: (M, n_edges) log of the multiplier a_theta on each legal edge.
    Returns (p_stay (M,), p_edge (M, n_edges)).
    """
    base = space.gamma / space.n_edges
    u = base * torch.exp(log_mult.clamp(-20.0, 20.0))
    R = u.sum(dim=1)
    p_stay = torch.exp(-R * dt)
    p_edge = (1.0 - p_stay).unsqueeze(1) * u / R.clamp_min(1e-300).unsqueeze(1)
    return p_stay, p_edge


def propagate_exact(space, mult_fn, steps):
    """Exact terminal law of the discretised controlled chain.  No sampling."""
    p = torch.zeros(space.M, device=space.device)
    p[space.i0] = 1.0
    dt = 1.0 / steps
    for s in range(steps):
        t = s * dt
        log_mult = mult_fn(t)                                  # (M, n_edges)
        p_stay, p_edge = step_probs(space, log_mult, dt)
        flow = p.unsqueeze(1) * p_edge                         # (M, n_edges)
        p_new = p * p_stay
        p_new = p_new.index_add(0, space.tgt.reshape(-1), flow.reshape(-1))
        p = p_new
    return p


def simulate(space, mult_fn_states, batch, steps, generator=None):
    """Sample trajectories.  mult_fn_states(t, idx) -> (B, n_edges) log-mult."""
    idx = torch.full((batch,), space.i0, device=space.device, dtype=torch.long)
    dt = 1.0 / steps
    base = space.gamma / space.n_edges
    for s in range(steps):
        t = s * dt
        u = base * torch.exp(mult_fn_states(t, idx).clamp(-20.0, 20.0))
        R = u.sum(dim=1)
        p_stay = torch.exp(-R * dt)
        jump = torch.rand(batch, device=space.device, generator=generator) > p_stay
        if jump.any():
            sel = torch.multinomial(u[jump], 1, generator=generator).squeeze(1)
            idx = idx.clone()
            idx[jump] = space.tgt[idx[jump], sel]
    return idx


# ============================================================================
# exact optimal control
# ============================================================================


class ExactControl:
    def __init__(self, space, steps):
        self.space = space
        ts = np.arange(steps) / steps
        phi = space.phi_grid(ts)                               # (T, M)
        self.logphi = torch.log(phi.clamp_min(1e-300))
        self.ts = ts
        self.steps = steps

    def _row(self, t):
        s = min(int(round(t * self.steps)), self.steps - 1)
        return self.logphi[s]

    def all_states(self, t):
        lp = self._row(t)
        return lp[self.space.tgt] - lp.unsqueeze(1)

    def on_states(self, t, idx):
        lp = self._row(t)
        return lp[self.space.tgt[idx]] - lp[idx].unsqueeze(1)


# ============================================================================
# neural controller
# ============================================================================


class SwapController(torch.nn.Module):
    """a_theta(t, x) -> (n, n) log-multiplier matrix (one pass for all edges)."""

    def __init__(self, n, hidden=512, n_freq=4):
        super().__init__()
        self.n = n
        self.n_freq = n_freq
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n + 2 + 2 * n_freq, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, n * n),
        ).to(torch.float32)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, x):
        """t: (B,)  x: (B, n) in {0,1}.  Returns (B, n, n), float32."""
        s = 2.0 * x.to(torch.float32) - 1.0
        t = t.to(torch.float32)[:, None]
        k = torch.arange(1, self.n_freq + 1, device=t.device,
                         dtype=t.dtype)[None, :]
        feats = [s, t, 1.0 - t,
                 torch.sin(math.pi * k * t), torch.cos(math.pi * k * t)]
        return self.net(torch.cat(feats, dim=-1)).view(-1, self.n, self.n)


def net_mult_on_index(space, net, t, idx):
    x = space.S[idx]
    tt = torch.full((idx.shape[0],), float(t), device=space.device)
    a = net(tt, x)
    b = torch.arange(idx.shape[0], device=space.device).unsqueeze(1)
    return a[b, space.edge_i[idx], space.edge_j[idx]].to(torch.float64)


def net_mult_all_states(space, net, t, chunk=4096):
    out = []
    for a in range(0, space.M, chunk):
        b = min(a + chunk, space.M)
        idx = torch.arange(a, b, device=space.device)
        out.append(net_mult_on_index(space, net, t, idx))
    return torch.cat(out, dim=0)


# ============================================================================
# terminal labels and bridge sampling
# ============================================================================


def terminal_labels(space, X1_idx):
    """Lambda_{g}(X1) for every unordered pair g.  (B, n_pairs)."""
    X1 = space.S[X1_idx]                                       # (B, n)
    P = space.pairs                                            # (Np, 2)
    Xg = X1.unsqueeze(1).repeat(1, space.n_pairs, 1)           # (B, Np, n)
    pp = torch.arange(space.n_pairs, device=X1.device)
    Xg[:, pp, P[:, 0]] = X1[:, P[:, 1]]
    Xg[:, pp, P[:, 1]] = X1[:, P[:, 0]]

    Eg = space.energies_of(Xg)                                 # (B, Np)
    E1 = space.E[X1_idx].unsqueeze(1)
    dg = space.k - (Xg.to(torch.float64) @ space.x0.to(torch.float64))
    d1 = space.dist0[X1_idx].unsqueeze(1)

    log_lam = (-(Eg - E1) / space.tau
               + space.log_kappa_full[d1.long()]
               - space.log_kappa_full[dg.long()])
    return log_lam


def sample_bridge(space, X1_idx, t_idx, log_kap0, log_kap1, generator=None):
    """Exact reference-bridge sample X_t | x_0, X_1 by enumeration."""
    d1 = (space.k
          - torch.round(space.Sf @ space.Sf[X1_idx].T).long()).T   # (B, M)
    lw = log_kap0[t_idx][:, space.dist0]                       # (B, M)
    lw = lw + torch.gather(log_kap1[t_idx], 1, d1)
    lw = lw - lw.max(dim=1, keepdim=True).values
    w = torch.exp(lw)
    return torch.multinomial(w, 1, generator=generator).squeeze(1)


# ============================================================================
# metrics
# ============================================================================


def report(space, p_terminal, tag, extra=None):
    tv = 0.5 * float((p_terminal - space.pi).abs().sum())
    mean_E = float((p_terminal * space.E).sum())
    exact_E = float((space.pi * space.E).sum())
    out = {
        "tag": tag,
        "TV": tv,
        "mean_energy": mean_E,
        "exact_mean_energy": exact_E,
        "energy_err": abs(mean_E - exact_E),
    }
    if extra:
        out.update(extra)
    return out


def energy_hist_tv(space, p):
    lv, inv = torch.unique(space.E, return_inverse=True)
    a = torch.zeros(len(lv), device=space.device).index_add(0, inv, p)
    b = torch.zeros(len(lv), device=space.device).index_add(0, inv, space.pi)
    return 0.5 * float((a - b).abs().sum())


# ============================================================================
# drivers
# ============================================================================


def run_exact(args):
    space = FixedIsingSpace(L=args.L, J=args.J, tau=args.tau,
                            gamma=args.gamma)
    print(f"L={args.L}  n={space.n}  k={space.k}  |Omega|={space.M}")
    kap = np.exp(space.log_kappa_full.cpu().numpy())
    print(f"tau={args.tau}  J={args.J}  gamma={args.gamma}")
    print(f"p_base(.|x0) kappa range: {kap.min():.3e} .. {kap.max():.3e}  "
          f"(ratio {kap.max()/kap.min():.2f})")
    print(f"exact <E> = {float((space.pi*space.E).sum()):.6f}   "
          f"source E = {float(space.E[space.i0]):.1f}")
    print()

    rows = []
    for steps in args.steps_sweep:
        ctrl = ExactControl(space, steps)
        t0 = time.time()
        p = propagate_exact(space, ctrl.all_states, steps)
        r = report(space, p, f"exact-control steps={steps}",
                   {"steps": steps, "energy_hist_TV": energy_hist_tv(space, p),
                    "mass_leak": abs(float(p.sum()) - 1.0),
                    "sec": round(time.time() - t0, 2)})
        rows.append(r)
        print(f"  steps={steps:4d}   TV={r['TV']:.5f}   "
              f"E-hist TV={r['energy_hist_TV']:.5f}   "
              f"<E>={r['mean_energy']:.4f} (exact {r['exact_mean_energy']:.4f})"
              f"   leak={r['mass_leak']:.2e}   {r['sec']}s")

    # Gate A2: sampled trajectories must satisfy the constraint exactly
    steps = args.steps_sweep[-1]
    ctrl = ExactControl(space, steps)
    idx = simulate(space, ctrl.on_states, args.n_samples, steps)
    X = space.S[idx]
    viol = int((X.sum(dim=1) != space.k).sum())
    emp = torch.bincount(idx, minlength=space.M).to(torch.float64)
    emp = emp / emp.sum()
    tv_emp = 0.5 * float((emp - space.pi).abs().sum())
    iid = torch.multinomial(space.pi, args.n_samples, replacement=True)
    e2 = torch.bincount(iid, minlength=space.M).to(torch.float64)
    e2 = e2 / e2.sum()
    tv_floor = 0.5 * float((e2 - space.pi).abs().sum())
    print(f"\n  sampled N={args.n_samples}: constraint violations = {viol}")
    print(f"  empirical TV = {tv_emp:.5f}   iid finite-sample floor = "
          f"{tv_floor:.5f}")

    res = {"config": vars(args), "rows": rows, "violations": viol,
           "empirical_TV": tv_emp, "iid_TV_floor": tv_floor}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n  wrote {args.out}")

    best = min(r["TV"] for r in rows)
    print("\n=== GATES ===")
    print(f"A1  TV <= 0.05 : {'PASS' if best <= 0.05 else 'FAIL'}  "
          f"(best exact-law TV {best:.5f})")
    print(f"A2  constraint : {'PASS' if viol == 0 else 'FAIL'}  "
          f"({viol} violations in {args.n_samples} samples)")
    return 0 if (best <= 0.05 and viol == 0) else 1


def run_train(args):
    space = FixedIsingSpace(L=args.L, J=args.J, tau=args.tau, gamma=args.gamma)
    torch.manual_seed(args.seed)
    net = SwapController(space.n, hidden=args.hidden).to(space.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps

    print(f"L={args.L} |Omega|={space.M}  params={n_par}  steps={steps}  "
          f"device={space.device}")

    ts = np.arange(steps) / steps
    log_kap0 = space.kappa_table(np.maximum(space.gamma * ts, 1e-12))
    log_kap1 = space.kappa_table(np.maximum(space.gamma * (1 - ts), 1e-12))

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)

    buf = []
    hist = []
    best = {"TV": 1e9}
    t_start = time.time()
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
                Xt = sample_bridge(space, X1, ti, log_kap0, log_kap1)
                log_lam = terminal_labels(space, X1)            # (mb, n_pairs)
                lam = torch.exp(log_lam.clamp(-20.0, 20.0))
                tgt = torch.gather(lam, 1, space.edge_pair[Xt])  # (mb, n_edges)
            tt = ti.to(torch.float64) / steps
            a = net(tt, space.S[Xt])
            b = torch.arange(args.mb, device=space.device).unsqueeze(1)
            av = a[b, space.edge_i[Xt], space.edge_j[Xt]].clamp(-20.0, 20.0)
            if args.loss == "poisson":
                # Bregman divergence of the exponential family.  The minimiser
                # of E[exp(a) - a*Lambda] satisfies exp(a) = E[Lambda | X_t],
                # i.e. exactly the same fixed point as the L2 regression, but
                # the gradient exp(a) - Lambda is LINEAR in the heavy-tailed
                # label instead of being multiplied by exp(a).
                loss = (torch.exp(av) - av * tgt).mean()
            else:
                loss = ((torch.exp(av) - tgt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it == 1:
            with torch.no_grad():
                ls = torch.exp(terminal_labels(space, pool[:512])
                               .clamp(-20, 20))
            print(f"  label Lambda stats: mean {float(ls.mean()):.3f}  "
                  f"p50 {float(ls.median()):.3f}  max {float(ls.max()):.1f}  "
                  f"frac==1 {float((ls-1).abs().lt(1e-9).double().mean()):.3f}")

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = propagate_exact(
                    space, lambda t: net_mult_all_states(space, net, t), steps)
            tv = 0.5 * float((p - space.pi).abs().sum())
            eh = energy_hist_tv(space, p)
            hist.append({"iter": it, "TV": tv, "energy_hist_TV": eh,
                         "loss": float(loss)})
            print(f"  it {it:4d}  loss {float(loss):10.4f}   "
                  f"exact-law TV {tv:.5f}   E-hist TV {eh:.5f}   "
                  f"({time.time()-t_start:.0f}s)")

    # final gates + comparison against the exact control
    with torch.no_grad():
        p = propagate_exact(
            space, lambda t: net_mult_all_states(space, net, t), steps)
        idx = simulate(space, lambda t, i: net_mult_on_index(space, net, t, i),
                       args.n_samples, steps)
    viol = int((space.S[idx].sum(dim=1) != space.k).sum())
    tv = 0.5 * float((p - space.pi).abs().sum())

    ctrl = ExactControl(space, steps)
    errs = []
    with torch.no_grad():
        for s in [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]:
            t = s / steps
            ex = ctrl.all_states(t)
            le = net_mult_all_states(space, net, t)
            w = p if s == steps - 1 else None
            d = (le - ex).abs()
            errs.append((t, float(d.mean()), float(d.max())))

    print("\n  learned vs exact log-multiplier (all 12870 states x 64 edges):")
    for t, me, mx in errs:
        print(f"    t={t:.3f}   mean|a_learn - a_exact| = {me:.4f}   "
              f"max = {mx:.4f}")

    print("\n=== GATES ===")
    print(f"A1  TV <= 0.05 : {'PASS' if tv <= 0.05 else 'FAIL'}  "
          f"(exact-law TV {tv:.5f})")
    print(f"A2  constraint : {'PASS' if viol == 0 else 'FAIL'}  "
          f"({viol} violations in {args.n_samples} samples)")

    if getattr(args, "ckpt_dir", ""):
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          samples=idx.to(torch.int32),
                          extra={"config": vars(args), "final_TV": tv,
                                 "violations": viol, "mult_err": errs,
                                 "exact_law": p.detach().cpu(),
                                 "pi": space.pi.detach().cpu(),
                                 "states": space.S.detach().cpu()})
        print(f"  ckpt -> {pth}")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "params": n_par, "history": hist,
                   "final_TV": tv, "violations": viol,
                   "mult_err": errs}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (tv <= 0.05 and viol == 0) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ["exact", "train"]:
        p = sub.add_parser(name)
        p.add_argument("--L", type=int, default=4)
        p.add_argument("--J", type=float, default=1.0)
        p.add_argument("--tau", type=float, default=2.0)
        p.add_argument("--gamma", type=float, default=10.0)
        p.add_argument("--n-samples", dest="n_samples", type=int, default=200000)
        p.add_argument("--out", type=str, default=f"json/results_ising_{name}.json")
        p.add_argument("--ckpt-dir", dest="ckpt_dir", type=str, default="")
        p.add_argument("--tag", type=str, default=f"ising_{name}")
        if name == "exact":
            p.add_argument("--steps-sweep", dest="steps_sweep", type=int,
                           nargs="+", default=[32, 64, 128, 256, 512])
        else:
            p.add_argument("--steps", type=int, default=128)
            p.add_argument("--iters", type=int, default=300)
            p.add_argument("--batch", type=int, default=2048)
            p.add_argument("--buffer", type=int, default=8)
            p.add_argument("--inner", type=int, default=40)
            p.add_argument("--mb", type=int, default=1024)
            p.add_argument("--hidden", type=int, default=512)
            p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--eval-every", dest="eval_every", type=int,
                           default=25)
            p.add_argument("--loss", type=str, default="poisson",
                           choices=["poisson", "mse"])

    args = ap.parse_args()
    return run_exact(args) if args.cmd == "exact" else run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
