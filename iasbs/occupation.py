"""
Experiment B -- occupation / inclusion process.

This is the genuinely NON-BIJECTIVE structured discrete case.  Unlike the
fixed-composition Ising experiment (where the intertwiner came from a group of
swaps acting bijectively on the state space), here the intertwining operators
are the occupation operators

    (E_ab f)(eta) = eta_b f(eta - e_b + e_a),      (E_aa f)(eta) = eta_a f(eta)
    F_ji          = E_ji - E_ii
    (F_ji f)(eta) = eta_i [ f(eta - e_i + e_j) - f(eta) ]

which are NOT induced by any bijection of the state space: they carry the
occupancy prefactor eta_i, and E_ji maps several states onto one.

Reference generator (gamma = 1):

    L = 1/(m-1) sum_{i != j} F_ji ,     total escape rate  =  N.

Commutators (verified numerically in tests_math.py [4]):

    [L, F_ji]  =  1/(m-1) S_ji ,        S_ji = sum_a (E_ja - E_ia)
    [L, S_ji]  =  m/(m-1) S_ji

so the *time-dependent* intertwiner that satisfies  F_ji P_{t,1} = P_{t,1} A_ji
is

    A^{(t)}_ji = F_ji - c_t S_ji ,      c_t = (1 - exp(-m/(m-1) Gamma_{t,1}))/m.

The controlled (h-transformed) edge rate is

    u*_t(eta - e_i + e_j, eta) = gamma_t/(m-1) * eta_i * phi_t(eta-e_i+e_j)/phi_t(eta)
                               = gamma_t/(m-1) * ( eta_i + E[ A_ji f1(X_1)/f1(X_1) | X_t=eta ] )

and the bracket is exactly the terminal label

    Lambda_ji(xi) = xi_i (R_ji - 1) - c_t sum_{a: xi_a > 0} xi_a (R_ja - R_ia),

evaluated in full for small m and estimated by the samplers below otherwise.

Target: Dirichlet-multinomial (inclusion process) with exact iid sampler, so
ground truth needs no MCMC.  At m = N = 4 the state space has only
C(N+m-1, m-1) = 35 elements, so EVERYTHING is exact:

  * the reference semigroup is a 35x35 matrix exponential,
  * phi_t is an exact matrix-vector product,
  * the terminal law of the discretised controlled chain is propagated exactly
    (no Monte-Carlo noise at all in the TV gate),
  * the terminal labels are computed in closed form for every state.

Subcommands
-----------
  verify   B0 -- operator algebra / kernel-ratio / estimator identities
  exact    exact h-transform control, TV vs discretisation steps  (B1 ceiling)
  var      B3 -- estimator variance ablation (full / uniform / occupancy / k-sample)
  train    neural controller, gate B1 (TV <= 0.05) + constraint check
"""

import argparse
import json
import math
import time

import numpy as np
import scipy.linalg
import torch

# common.py and the shared json/ ckpt/ fig/ directories live at the
# repository root, one level up from this script.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# state space
# ----------------------------------------------------------------------------
class OccupationSpace:
    """Full enumeration of {eta in N_0^m : sum eta = N} with all exact tables."""

    def __init__(self, m=4, N=4, d=0.5, tau=1.0, gamma=4.0, source=0, device=DEV):
        self.m, self.N, self.d, self.tau, self.gamma = m, N, d, tau, gamma
        self.device = device
        S = C.enumerate_occupation(m, N)
        self.index = C.occupation_index(S)
        self.M = len(S)
        self.S = torch.tensor(S, device=device)                 # (M, m) long
        self.Sf = self.S.to(torch.float64)

        # reference generator at gamma = 1
        self.Lhat = C.occupation_L(S, self.index, m)            # (M, M) numpy

        # transfer table  trans[r, a, b] = index of  eta - e_a + e_b
        trans = np.zeros((self.M, m, m), dtype=np.int64)
        valid = np.zeros((self.M, m, m), dtype=bool)
        for r, eta in enumerate(S):
            for a in range(m):
                if eta[a] == 0:
                    trans[r, a, :] = r
                    continue
                for b in range(m):
                    if b == a:
                        trans[r, a, b] = r
                        valid[r, a, b] = True
                        continue
                    t = eta.copy()
                    t[a] -= 1
                    t[b] += 1
                    trans[r, a, b] = self.index[tuple(t)]
                    valid[r, a, b] = True
        self.trans = torch.tensor(trans, device=device)
        self.valid = torch.tensor(valid, device=device)

        # directed edges (i -> j), i != j
        ei, ej = [], []
        for i in range(m):
            for j in range(m):
                if i != j:
                    ei.append(i)
                    ej.append(j)
        self.edge_i = torch.tensor(ei, device=device)
        self.edge_j = torch.tensor(ej, device=device)
        self.n_edges = len(ei)
        self.etgt = self.trans[:, self.edge_i, self.edge_j]      # (M, n_edges)
        self.eocc = self.Sf[:, self.edge_i]                      # (M, n_edges) eta_i
        self.emask = self.eocc > 0

        # exact target: pi(eta) \propto prod_i Gamma(eta_i+d)/(eta_i! Gamma(d))
        self.E = C.inclusion_energy(self.Sf, tau, d)             # (M,)
        w = torch.exp(-self.E / tau)
        self.pi = w / w.sum()

        # Dirac source eta_0 = N e_c
        x0 = np.zeros(m, dtype=np.int64)
        x0[source] = N
        self.source = source
        self.i0 = self.index[tuple(x0)]

        # reference terminal law from the source, exact
        P01 = scipy.linalg.expm(gamma * self.Lhat)
        self.pb0 = torch.tensor(np.maximum(P01[self.i0], 1e-300), device=device)

        # AS terminal function  f1 = exp(-E/tau) / p_base(.|eta_0)
        self.logf1 = (-self.E / tau) - torch.log(self.pb0)
        self.logf1 = self.logf1 - self.logf1.max()

        # R table  R[r, a, b] = f1(eta - e_a + e_b) / f1(eta)
        self.R = torch.exp(self.logf1[self.trans] - self.logf1[:, None, None])
        self.R = torch.where(self.valid, self.R, torch.zeros_like(self.R))

    # -- non-Dirac source ---------------------------------------------------
    def set_nondirac_source(self, nu0):
        """nu_0 = sum_c nu0[c] delta_{N e_c}, supported on the m pure modes.

        The source has m atoms, so only m rows of the reference semigroup are
        ever needed: the whole non-Dirac branch costs one extra (m, |X|) table.
        """
        w = torch.as_tensor(nu0, device=self.device, dtype=torch.float64)
        self.nu0 = w / w.sum()
        src = []
        for c in range(self.m):
            x = np.zeros(self.m, dtype=np.int64)
            x[c] = self.N
            src.append(self.index[tuple(x)])
        self.src_idx = torch.tensor(src, device=self.device)
        P01 = self.semigroup(self.gamma)
        self.logP01_src = torch.tensor(
            np.log(np.maximum(P01[np.asarray(src)], 1e-300)),
            device=self.device)                                 # (m, M)
        self.p0_vec = torch.zeros(self.M, device=self.device,
                                  dtype=torch.float64)
        self.p0_vec[self.src_idx] = self.nu0

    def rebuild_R(self, log_fhat):
        """Recompute f_1 = exp(-E/tau)/fhat_1 and the ratio table R from it.

        The Dirac constructor sets fhat_1 = p_base(.|eta_0); with a source
        distribution that single row is replaced by the corrector.  Nothing
        downstream changes -- labels_full, labels_sampled and phi_grid all read
        R and logf1, so the entire label machinery is shared between the two.
        """
        lf1 = (-self.E / self.tau) - log_fhat
        self.logf1 = lf1 - lf1.max()
        self.R = torch.exp(self.logf1[self.trans] - self.logf1[:, None, None])
        self.R = torch.where(self.valid, self.R, torch.zeros_like(self.R))

    # -- semigroup ----------------------------------------------------------
    def semigroup(self, Gamma):
        return scipy.linalg.expm(float(Gamma) * self.Lhat)

    def phi_grid(self, ts):
        """logphi_t on the whole state space for each t in ts.  (T, M)."""
        f1 = torch.exp(self.logf1).cpu().numpy()
        out = np.empty((len(ts), self.M), dtype=np.float64)
        for a, t in enumerate(ts):
            P = self.semigroup(self.gamma * (1.0 - t))
            out[a] = np.maximum(P @ f1, 1e-300)
        return torch.tensor(np.log(out), device=self.device)

    # -- labels -------------------------------------------------------------
    def labels_full(self, c_t):
        """Lambda_ji(xi) for every state, exact full sum.  (M, n_edges)."""
        # T[r, b] = sum_a xi_a R[r, a, b]
        T = torch.einsum("ra,rab->rb", self.Sf, self.R)
        first = self.eocc * (self.R[:, self.edge_i, self.edge_j] - 1.0)
        first = torch.where(self.emask, first, torch.zeros_like(first))
        corr = T[:, self.edge_j] - T[:, self.edge_i]
        return first - c_t * corr

    def labels_sampled(self, idx, c_t, mode="occupancy", n_a=1, generator=None):
        """Stochastic terminal label at states idx.  (B, n_edges).

        c_t may be a scalar or a per-sample tensor of shape (B,).

        uniform    : Lam = xi_i(R_ji - 1) - (1-rho) xi_A (R_jA - R_iA), A~Unif
        occupancy  : Lam = xi_i(R_ji - 1) - c_t N (R_jA - R_iA),        A~xi/N
        Both are unbiased for the full sum sum_a xi_a (R_ja - R_ia).  Uniform
        mode sampling is unbiased because m c_t = 1 - rho_{t,1}, but its
        variance grows with m; drawing A ~ xi/N instead absorbs the xi_a weight
        into the proposal and leaves the smaller-variance estimator.
        """
        xi = self.Sf[idx]                                        # (B, m)
        Rr = self.R[idx]                                         # (B, m, m)
        B = len(idx)
        ct = torch.as_tensor(c_t, device=self.device, dtype=torch.float64)
        ct = ct.expand(B) if ct.dim() == 0 else ct
        first = xi[:, self.edge_i] * (Rr[:, self.edge_i, self.edge_j] - 1.0)
        first = torch.where(xi[:, self.edge_i] > 0, first,
                            torch.zeros_like(first))
        ar = torch.arange(B, device=self.device)
        acc = torch.zeros(B, self.n_edges, device=self.device, dtype=torch.float64)
        for _ in range(n_a):
            if mode == "uniform":
                a = torch.randint(self.m, (B,), device=self.device,
                                  generator=generator)
                scale = self.m * ct * xi[ar, a]
            else:
                a = torch.multinomial(xi / self.N, 1, generator=generator)[:, 0]
                scale = ct * float(self.N)
            Rja = Rr[ar[:, None], a[:, None], self.edge_j[None, :]]
            Ria = Rr[ar[:, None], a[:, None], self.edge_i[None, :]]
            acc = acc + scale[:, None] * (Rja - Ria)
        return first - acc / n_a


def c_of_t(space, t):
    return C.occupation_c_t(space.m, space.gamma * (1.0 - t))


# ----------------------------------------------------------------------------
# forward simulation / exact law propagation
# ----------------------------------------------------------------------------
def step_probs(space, log_mult, dt):
    """log_mult: (M, n_edges).  Returns jump probabilities (M, n_edges)."""
    rate = (space.gamma / (space.m - 1.0)) * space.eocc * torch.exp(
        log_mult.clamp(-20.0, 20.0))
    rate = torch.where(space.emask, rate, torch.zeros_like(rate))
    p = rate * dt
    tot = p.sum(dim=1, keepdim=True)
    scale = torch.clamp(0.98 / torch.clamp(tot, min=1e-30), max=1.0)
    return p * scale


def propagate_exact(space, mult_fn, steps, p0=None):
    """p0 is the initial law; the default is the Dirac at space.i0.  The
    non-Dirac branch must evaluate under the SAME source it trained on."""
    dt = 1.0 / steps
    if p0 is not None:
        p = p0.to(space.device).to(torch.float64).clone()
    else:
        p = torch.zeros(space.M, device=space.device, dtype=torch.float64)
        p[space.i0] = 1.0
    for s in range(steps):
        pj = step_probs(space, mult_fn(s / steps), dt)
        flow = p[:, None] * pj
        nxt = p - flow.sum(dim=1)
        nxt = nxt.index_add(0, space.etgt.reshape(-1), flow.reshape(-1))
        p = nxt
    return p


def simulate(space, mult_fn_states, batch, steps, generator=None, idx0=None):
    dt = 1.0 / steps
    if idx0 is not None:
        idx = idx0.to(space.device).long()
    else:
        idx = torch.full((batch,), space.i0, device=space.device,
                         dtype=torch.long)
    for s in range(steps):
        lm = mult_fn_states(s / steps, idx)                      # (B, n_edges)
        rate = (space.gamma / (space.m - 1.0)) * space.eocc[idx] * torch.exp(
            lm.clamp(-20.0, 20.0))
        rate = torch.where(space.emask[idx], rate, torch.zeros_like(rate))
        p = rate * dt
        tot = p.sum(dim=1, keepdim=True)
        p = p * torch.clamp(0.98 / torch.clamp(tot, min=1e-30), max=1.0)
        stay = 1.0 - p.sum(dim=1, keepdim=True)
        w = torch.cat([stay, p], dim=1)
        pick = torch.multinomial(w, 1, generator=generator)[:, 0]
        move = pick > 0
        e = (pick - 1).clamp(min=0)
        nxt = space.etgt[idx, e]
        idx = torch.where(move, nxt, idx)
    return idx


class ExactControl:
    def __init__(self, space, steps):
        self.space = space
        ts = np.arange(steps) / steps
        self.logphi = space.phi_grid(ts)                          # (steps, M)
        self.steps = steps

    def all_states(self, t):
        s = int(round(t * self.steps))
        s = min(max(s, 0), self.steps - 1)
        lp = self.logphi[s]
        return lp[self.space.etgt] - lp[:, None]

    def on_states(self, t, idx):
        return self.all_states(t)[idx]


# ----------------------------------------------------------------------------
# neural controller
# ----------------------------------------------------------------------------
class OccController(torch.nn.Module):
    """(t, eta) -> log-multiplier a(t, eta, i, j) for every directed edge."""

    def __init__(self, m, N, hidden=256, n_freq=4):
        super().__init__()
        self.m, self.N, self.n_freq = m, N, n_freq
        din = m + 2 + 2 * n_freq
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, m * m),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.to(torch.float32)

    def forward(self, t, eta):
        e = eta.to(torch.float32) / self.N
        t = t.to(torch.float32)[:, None]
        k = torch.arange(1, self.n_freq + 1, device=t.device,
                         dtype=t.dtype)[None, :]
        feats = [e, t, 1.0 - t, torch.sin(math.pi * k * t),
                 torch.cos(math.pi * k * t)]
        return self.net(torch.cat(feats, dim=-1)).view(-1, self.m, self.m)


def net_mult_all(space, net, t):
    tt = torch.full((space.M,), t, device=space.device, dtype=torch.float64)
    a = net(tt, space.S).to(torch.float64)
    return a[:, space.edge_i, space.edge_j]


def net_mult_on(space, net, t, idx):
    tt = torch.full((len(idx),), t, device=space.device, dtype=torch.float64)
    a = net(tt, space.S[idx]).to(torch.float64)
    return a[:, space.edge_i, space.edge_j]


class SourceCorrector(torch.nn.Module):
    """log fhat_1(z) = logsumexp_c ( logw_c + log p_base(z | N e_c) ).

    fhat_1(z) = sum_c f_0(N e_c) nu_0(c) p_base(z | N e_c) is a nonnegative
    combination of m fixed rows, so with a finite source the corrector lives in
    an m-dimensional cone and this parameterisation CONTAINS the exact answer
    rather than approximating it: only the m log-weights are free.  R uses
    fhat_1 through ratios alone, so the overall scale of w is unidentifiable
    and harmless.
    """

    def __init__(self, space):
        super().__init__()
        self.logP = space.logP01_src                             # (m, M)
        self.logw = torch.nn.Parameter(torch.log(space.nu0.clone()))

    def forward(self):
        return torch.logsumexp(self.logw[:, None] + self.logP, dim=0)


def exact_sinkhorn(space, iters=20000, tol=1e-15):
    """Static Schrodinger bridge between nu_0 and pi, solved exactly.

    Gamma(c, z) = u_c K(c, z) v_z with row marginal nu_0 and column marginal
    pi, K = p_base(. | N e_c).  There are only m = 4 rows, so alternating
    scaling converges to machine precision in milliseconds and returns the
    corrector fhat_1 = K^T u that the training loop has to discover -- computed
    without any reference to the learned control, so it is an independent
    reference rather than a self-consistency check.
    """
    K = torch.exp(space.logP01_src)                              # (m, M)
    u = torch.ones(space.m, device=space.device, dtype=torch.float64)
    v = torch.ones(space.M, device=space.device, dtype=torch.float64)
    for _ in range(iters):
        u_new = space.nu0 / (K @ v).clamp_min(1e-300)
        v_new = space.pi / (K.T @ u_new).clamp_min(1e-300)
        done = ((u_new - u).abs().max() < tol
                and (v_new - v).abs().max() < tol)
        u, v = u_new, v_new
        if done:
            break
    row = (u[:, None] * K * v[None, :]).sum(1)
    col = (u[:, None] * K * v[None, :]).sum(0)
    err = max(float((row - space.nu0).abs().max()),
              float((col - space.pi).abs().max()))
    return u, torch.log((K.T @ u).clamp_min(1e-300)), err


def sample_bridge_nd(space, c0, X1_idx, t_idx, K0, K1, generator=None):
    """Reference bridge with a per-trajectory source atom.  K0 is (T, m, M)."""
    lw = K0[t_idx, c0]                                                # (B, M)
    lw = lw + K1[t_idx].gather(2, X1_idx[:, None, None].expand(
        -1, space.M, 1))[:, :, 0]
    lw = lw - lw.max(dim=1, keepdim=True).values
    return torch.multinomial(torch.exp(lw), 1, generator=generator)[:, 0]


def labels_grid(sp, cts):
    """labels_full for the whole time grid at once.  (T, M, n_edges).

    Only the scalar c_t varies with time, so the two state-dependent pieces are
    built once.  The non-Dirac loop rebuilds this table every outer iteration
    (R moves when the corrector moves), which would otherwise be a Python loop
    over all time steps.
    """
    T = torch.einsum("ra,rab->rb", sp.Sf, sp.R)
    first = sp.eocc * (sp.R[:, sp.edge_i, sp.edge_j] - 1.0)
    first = torch.where(sp.emask, first, torch.zeros_like(first))
    corr = T[:, sp.edge_j] - T[:, sp.edge_i]
    return first[None] - cts[:, None, None] * corr[None]


def sample_bridge(space, X1_idx, t_idx, K0, K1, generator=None):
    """Exact reference bridge  p(X_t = y | X_0 = eta_0, X_1 = xi)
       proportional to  p_{0,t}(y | eta_0) p_{t,1}(xi | y),  by enumeration."""
    lw = K0[t_idx]                                                    # (B, M)
    lw = lw + K1[t_idx].gather(2, X1_idx[:, None, None].expand(
        -1, space.M, 1))[:, :, 0]
    lw = lw - lw.max(dim=1, keepdim=True).values
    return torch.multinomial(torch.exp(lw), 1, generator=generator)[:, 0]


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def occ_hist(space, p):
    """P(eta_i = n) marginal over sites, from a law p on states."""
    h = torch.zeros(space.N + 1, device=space.device, dtype=torch.float64)
    for n in range(space.N + 1):
        h[n] = (p[:, None] * (space.S == n).to(torch.float64)).sum() / space.m
    return h


def maxocc_hist(space, p):
    mx = space.S.max(dim=1).values
    h = torch.zeros(space.N + 1, device=space.device, dtype=torch.float64)
    h.index_add_(0, mx, p)
    return h


def report_law(space, p, tag):
    tv = 0.5 * float((p - space.pi).abs().sum())
    ho = 0.5 * float((occ_hist(space, p) - occ_hist(space, space.pi)).abs().sum())
    hm = 0.5 * float((maxocc_hist(space, p)
                      - maxocc_hist(space, space.pi)).abs().sum())
    ce = float((p * space.E).sum())
    print(f"    {tag}  TV={tv:.5f}  occ-hist TV={ho:.5f}  "
          f"maxocc TV={hm:.5f}  <E>={ce:.4f}  leak={abs(float(p.sum())-1):.2e}")
    return {"TV": tv, "occ_hist_TV": ho, "maxocc_TV": hm, "meanE": ce}


# ----------------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------------
def run_verify(args):
    sp = OccupationSpace(m=args.m, N=args.N, d=args.d, tau=args.tau,
                         gamma=args.gamma)
    m, N = sp.m, sp.N
    print(f"m={m} N={N} d={sp.d} |X|={sp.M}  gamma={sp.gamma}")
    ok = True

    # (a) generator rows sum to zero, escape rate = N
    rs = np.abs(sp.Lhat.sum(axis=1)).max()
    esc = np.abs(-np.diag(sp.Lhat) - N).max()
    print(f"  row-sum max |.|      = {rs:.3e}")
    print(f"  escape rate - N      = {esc:.3e}")
    ok &= rs < 1e-12 and esc < 1e-12

    # (b) reference endpoint from the Dirac source is multinomial with q
    worst = 0.0
    for G in [0.3, 1.0, 4.0]:
        P = sp.semigroup(G)
        q = C.occupation_single_particle_probs(m, G, source_mode=sp.source)
        Snp = sp.S.cpu().numpy()
        lg = (math.lgamma(N + 1)
              - np.array([sum(math.lgamma(int(v) + 1) for v in s) for s in Snp])
              + Snp @ np.log(q))
        worst = max(worst, float(np.abs(P[sp.i0] - np.exp(lg)).max()))
    print(f"  |expm row - multinomial(q)| = {worst:.3e}")
    ok &= worst < 1e-11

    # (c) m c_t = 1 - rho
    dd = max(abs(m * C.occupation_c_t(m, G) - (1 - C.occupation_rho(m, G)))
             for G in [0.1, 1.0, 4.0])
    print(f"  |m c_t - (1-rho)|    = {dd:.3e}")
    ok &= dd < 1e-14

    # (d) intertwining  F_ji P_{t,1} = P_{t,1} A_ji
    worst = 0.0
    for G in [0.4, 1.7]:
        P = sp.semigroup(G)
        for (i, j) in [(0, 1), (1, 3), (2, 0)]:
            F = C.occupation_F(sp.S.cpu().numpy(), sp.index, j, i)
            A = C.occupation_A(sp.S.cpu().numpy(), sp.index, m, j, i, G)
            worst = max(worst, float(np.abs(F @ P - P @ A).max()))
    print(f"  |F P - P A|          = {worst:.3e}")
    ok &= worst < 1e-9

    # (e) label estimators unbiased against the full sum
    torch.manual_seed(0)
    for t in [0.0, 0.5, 0.9]:
        ct = c_of_t(sp, t)
        full = sp.labels_full(ct)
        idx = torch.arange(sp.M, device=sp.device)
        for mode in ["uniform", "occupancy"]:
            acc = torch.zeros_like(full)
            n_rep = 4000
            for _ in range(n_rep):
                acc += sp.labels_sampled(idx, ct, mode=mode)
            acc /= n_rep
            err = float((acc - full).abs().max())
            print(f"  t={t:.1f} {mode:10s} MC-mean err (n={n_rep}) = {err:.4f}")
            ok &= err < 0.35

    # (f) the identity that the whole experiment rests on:
    #
    #     eta_i + E^{u*}[ Lambda_ji(X_1) | X_t = eta ]
    #         = eta_i phi_t(eta - e_i + e_j) / phi_t(eta)
    #
    # NOTE the expectation is under the CONTROLLED (h-transformed) law
    #   P^{u*}(xi | eta) = P_{t,1}(xi | eta) f1(xi) / phi_t(eta),
    # because Lambda = (A_ji f1)/f1 and  F_ji P = P A_ji  gives
    #   (F_ji phi)(eta) = (P A_ji f1)(eta) = phi_t(eta) E^{u*}[Lambda | X_t].
    worst = 0.0
    for t in [0.0, 0.3, 0.7, 0.95]:
        ct = c_of_t(sp, t)
        lam = sp.labels_full(ct)                                  # (M, n_edges)
        P = torch.tensor(sp.semigroup(sp.gamma * (1.0 - t)), device=sp.device)
        f1 = torch.exp(sp.logf1)
        lp = sp.phi_grid([t])[0]
        Pu = P * f1[None, :] / torch.exp(lp)[:, None]             # rows sum to 1
        Elam = Pu @ lam                                           # (M, n_edges)
        rhs = sp.eocc * torch.exp(lp[sp.etgt] - lp[:, None])
        lhs = sp.eocc + Elam
        dif = torch.where(sp.emask, (lhs - rhs).abs(),
                          torch.zeros_like(lhs))
        worst = max(worst, float(dif.max()))
    print(f"  |eta_i + E[Lam] - eta_i phi'/phi| = {worst:.3e}")
    ok &= worst < 1e-8

    print(f"\nB0 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def run_exact(args):
    sp = OccupationSpace(m=args.m, N=args.N, d=args.d, tau=args.tau,
                         gamma=args.gamma)
    print(f"m={sp.m} N={sp.N} d={sp.d} tau={sp.tau} gamma={sp.gamma} "
          f"|X|={sp.M}  source={tuple(sp.S[sp.i0].tolist())}")
    print(f"  exact <E> = {float((sp.pi*sp.E).sum()):.6f}   "
          f"source E = {float(sp.E[sp.i0]):.4f}")
    print(f"  pi range {float(sp.pi.min()):.3e} .. {float(sp.pi.max()):.3e}")

    # uncontrolled reference for contrast
    ref = torch.tensor(sp.semigroup(sp.gamma)[sp.i0], device=sp.device)
    report_law(sp, ref, "reference (no control)")

    out = {}
    for steps in [8, 16, 32, 64, 128, 256]:
        ctrl = ExactControl(sp, steps)
        p = propagate_exact(sp, ctrl.all_states, steps)
        out[steps] = report_law(sp, p, f"steps={steps:5d}")

    # sampled check
    ctrl = ExactControl(sp, 128)
    torch.manual_seed(args.seed)
    idx = simulate(sp, ctrl.on_states, args.n_samples, 128)
    viol = int((sp.S[idx].sum(dim=1) != sp.N).sum())
    cnt = torch.bincount(idx, minlength=sp.M).double()
    emp = cnt / cnt.sum()
    tv_emp = 0.5 * float((emp - sp.pi).abs().sum())
    floor = float(np.mean([
        0.5 * np.abs(np.bincount(
            np.random.choice(sp.M, args.n_samples, p=sp.pi.cpu().numpy()),
            minlength=sp.M) / args.n_samples - sp.pi.cpu().numpy()).sum()
        for _ in range(5)]))
    print(f"  sampled N={args.n_samples}: constraint violations = {viol}")
    print(f"  empirical TV = {tv_emp:.5f}   iid finite-sample floor = {floor:.5f}")

    best = min(v["TV"] for v in out.values())
    print(f"\nB1(exact) {'PASS' if best <= 0.05 else 'FAIL'}  "
          f"(best exact-law TV {best:.5f})")
    print(f"B2(constraint) {'PASS' if viol == 0 else 'FAIL'}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "sweep": out,
                   "empirical_TV": tv_emp, "iid_floor": floor,
                   "violations": viol}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


def run_var(args):
    """B3 -- estimator variance ablation."""
    sp = OccupationSpace(m=args.m, N=args.N, d=args.d, tau=args.tau,
                         gamma=args.gamma)
    torch.manual_seed(args.seed)
    print(f"m={sp.m} N={sp.N} |X|={sp.M}  variance of Lambda estimators")
    print(f"  (variance averaged over states ~ reference law at t, "
          f"over all {sp.n_edges} edges)")
    rows = []
    for t in [0.0, 0.25, 0.5, 0.75, 0.9]:
        ct = c_of_t(sp, t)
        full = sp.labels_full(ct)
        w = torch.tensor(sp.semigroup(sp.gamma * t)[sp.i0], device=sp.device)
        w = w / w.sum()
        idx = torch.arange(sp.M, device=sp.device)
        rec = {"t": t, "c_t": ct}
        for mode, na in [("uniform", 1), ("occupancy", 1), ("occupancy", 4)]:
            n_rep = args.reps
            s1 = torch.zeros_like(full)
            s2 = torch.zeros_like(full)
            for _ in range(n_rep):
                v = sp.labels_sampled(idx, ct, mode=mode, n_a=na)
                s1 += v
                s2 += v * v
            var = s2 / n_rep - (s1 / n_rep) ** 2
            vbar = float((w[:, None] * var).sum() / sp.n_edges)
            rec[f"{mode}{na}"] = vbar
            rows.append((t, f"{mode}-{na}", vbar))
        print(f"  t={t:.2f}  c_t={ct:.4f}   "
              f"uniform-1 {rec['uniform1']:.4f}   "
              f"occupancy-1 {rec['occupancy1']:.4f}   "
              f"occupancy-4 {rec['occupancy4']:.4f}   "
              f"full-sum 0.0000")
    # gate: occupancy-1 variance <= uniform-1 variance at every t
    vals = {}
    for t, k, v in rows:
        vals[(t, k)] = v
    ts = sorted(set(t for t, _, _ in rows))
    okB3 = all(vals[(t, "occupancy-1")] <= vals[(t, "uniform-1")] + 1e-12
               for t in ts)
    print(f"\nB3 {'PASS' if okB3 else 'FAIL'}  "
          f"(occupancy-weighted variance <= uniform at all t)")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args),
                   "rows": [{"t": t, "est": k, "var": v} for t, k, v in rows],
                   "B3": bool(okB3)}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if okB3 else 1


def run_train(args):
    sp = OccupationSpace(m=args.m, N=args.N, d=args.d, tau=args.tau,
                         gamma=args.gamma)
    torch.manual_seed(args.seed)
    net = OccController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    print(f"m={sp.m} N={sp.N} |X|={sp.M}  params={n_par}  steps={steps}  "
          f"est={args.estimator}  loss={args.loss}  device={sp.device}")

    # bridge kernels on the time grid: K0[s] = log p(eta | eta_0, Gamma_{0,t}),
    # K1[s][y, xi] = log p(xi | y, Gamma_{t,1})
    ts = np.arange(steps) / steps
    K0 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * t)[sp.i0] for t in ts]), 1e-300)),
        device=sp.device)
    K1 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * (1 - t)) for t in ts]), 1e-300)),
        device=sp.device)
    cts = torch.tensor([c_of_t(sp, t) for t in ts], device=sp.device)
    # exact full-sum labels on the whole (t, state) grid -- (steps, M, n_edges)
    lam_tab = torch.stack([sp.labels_full(float(c)) for c in cts])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)

    buf, hist = [], []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            idx1 = simulate(sp, lambda t, i: net_mult_on(sp, net, t, i),
                            args.batch, steps)
        buf.append(idx1)
        if len(buf) > args.buffer:
            buf.pop(0)
        pool = torch.cat(buf)

        for _ in range(args.inner):
            sel = torch.randint(len(pool), (args.mb,), device=sp.device)
            X1 = pool[sel]
            ti = torch.randint(steps, (args.mb,), device=sp.device)
            with torch.no_grad():
                Xt = sample_bridge(sp, X1, ti, K0, K1)
                if args.estimator == "full":
                    lam = lam_tab[ti, X1]
                else:
                    lam = sp.labels_sampled(
                        X1, cts[ti], mode=args.estimator, n_a=args.n_a)
                # occupancy of the *current* state at the source mode i
                occ = sp.eocc[Xt]                                  # (mb, n_edges)
                y = occ + lam
            tt = ti.to(torch.float64) / steps
            a = net(tt, sp.S[Xt]).to(torch.float64)
            av = a[:, sp.edge_i, sp.edge_j].clamp(-20.0, 20.0)
            mask = sp.emask[Xt].to(torch.float64)
            if args.loss == "bregman":
                per = occ * torch.exp(av) - av * y
            else:
                per = (occ * torch.exp(av) - y) ** 2
            loss = (per * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = propagate_exact(sp, lambda t: net_mult_all(sp, net, t), steps)
            tv = 0.5 * float((p - sp.pi).abs().sum())
            ho = 0.5 * float((occ_hist(sp, p)
                              - occ_hist(sp, sp.pi)).abs().sum())
            hist.append({"iter": it, "TV": tv, "occ_hist_TV": ho,
                         "loss": float(loss)})
            print(f"  it {it:4d}  loss {float(loss):10.4f}   exact-law TV "
                  f"{tv:.5f}   occ-hist TV {ho:.5f}   ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        p = propagate_exact(sp, lambda t: net_mult_all(sp, net, t), steps)
        idx = simulate(sp, lambda t, i: net_mult_on(sp, net, t, i),
                       args.n_samples, steps)
    viol = int((sp.S[idx].sum(dim=1) != sp.N).sum())
    tv = 0.5 * float((p - sp.pi).abs().sum())
    stats = report_law(sp, p, "final")

    ctrl = ExactControl(sp, steps)
    errs = []
    with torch.no_grad():
        for s in [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]:
            t = s / steps
            ex = ctrl.all_states(t)
            le = net_mult_all(sp, net, t)
            d = torch.where(sp.emask, (le - ex).abs(), torch.zeros_like(le))
            errs.append((t, float(d.sum() / sp.emask.sum()), float(d.max())))
    print("\n  learned vs exact log-multiplier (all states x legal edges):")
    for t, me, mx in errs:
        print(f"    t={t:.3f}   mean|a_learn - a_exact| = {me:.4f}   "
              f"max = {mx:.4f}")

    print("\n=== GATES ===")
    print(f"B1  TV <= 0.05 : {'PASS' if tv <= 0.05 else 'FAIL'}  "
          f"(exact-law TV {tv:.5f})")
    print(f"B2  constraint : {'PASS' if viol == 0 else 'FAIL'}  "
          f"({viol} violations in {args.n_samples} samples)")
    if args.ckpt_dir:
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          samples=idx.to(torch.int32),
                          extra={"config": vars(args), "final_TV": tv,
                                 "violations": viol, "mult_err": errs,
                                 "exact_law": p.detach().cpu(),
                                 "pi": sp.pi.detach().cpu(),
                                 "states": sp.S.detach().cpu()})
        print(f"  ckpt -> {pth}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "params": n_par, "history": hist,
                   "final_TV": tv, "violations": viol, "stats": stats,
                   "mult_err": errs}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (tv <= 0.05 and viol == 0) else 1


def run_train_nondirac(args):
    """Non-Dirac source on the occupation space: nu_0 over the m pure modes.

    Contrast with the Ising non-Dirac run, where the source is uniform over the
    whole of Omega and the corrector has to be a network.  Here the source is
    finitely supported, so the exact corrector is an m-dimensional object and
    the *same* endpoint-pair estimator fits it with m free numbers -- and the
    exact answer is available by Sinkhorn, so the learned corrector can be
    checked directly rather than only through the terminal law.
    """
    sp = OccupationSpace(m=args.m, N=args.N, d=args.d, tau=args.tau,
                         gamma=args.gamma)
    nu0 = torch.tensor([args.nu_skew ** c for c in range(sp.m)],
                       dtype=torch.float64)
    sp.set_nondirac_source(nu0)
    torch.manual_seed(args.seed)
    net = OccController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    corr = SourceCorrector(sp).to(sp.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(corr.parameters(), lr=args.lr_h)
    steps = args.steps

    u_ex, logfhat_ex, sink_err = exact_sinkhorn(sp)
    print(f"m={sp.m} N={sp.N} |X|={sp.M}  params={n_par}+{sp.m}  "
          f"steps={steps}  source=nu0{np.round(sp.nu0.cpu().numpy(), 4)}  "
          f"device={sp.device}")
    print(f"  exact Sinkhorn marginal error {sink_err:.3e}   "
          f"f_0 (up to scale) {np.round((u_ex / sp.nu0 / (u_ex[0] / sp.nu0[0])).cpu().numpy(), 5)}")

    ts = np.arange(steps) / steps
    # (T, m, M): the source enters the bridge only through which of the m rows
    K0 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * t)[sp.src_idx.cpu().numpy()] for t in ts]),
        1e-300)), device=sp.device)
    K1 = torch.tensor(np.log(np.maximum(np.stack(
        [sp.semigroup(sp.gamma * (1 - t)) for t in ts]), 1e-300)),
        device=sp.device)
    cts = torch.tensor([c_of_t(sp, t) for t in ts], device=sp.device)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)

    buf_c, buf_1, hist = [], [], []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            c0 = torch.multinomial(sp.nu0, args.batch, replacement=True)
            idx1 = simulate(sp, lambda t, i: net_mult_on(sp, net, t, i),
                            args.batch, steps, idx0=sp.src_idx[c0])
        buf_c.append(c0)
        buf_1.append(idx1)
        if len(buf_1) > args.buffer:
            buf_c.pop(0)
            buf_1.pop(0)
        pool_c, pool_1 = torch.cat(buf_c), torch.cat(buf_1)

        # --- corrector: exp(h_g) -> E[ p_base(gY|X_0)/p_base(Y|X_0) | Y ] ---
        for _ in range(args.inner_h):
            sel = torch.randint(len(pool_1), (args.mb,), device=sp.device)
            Y, c = pool_1[sel], pool_c[sel]
            tg = sp.etgt[Y]                                       # (mb, n_edges)
            with torch.no_grad():
                q = torch.exp((sp.logP01_src[c].gather(1, tg)
                               - sp.logP01_src[c, Y][:, None]).clamp(-20, 20))
            lf = corr()
            h = (lf[tg] - lf[Y][:, None]).clamp(-20.0, 20.0)
            mask = sp.emask[Y].to(torch.float64)
            loss_c = ((torch.exp(h) - h * q) * mask).sum() / mask.sum()
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

        with torch.no_grad():
            sp.rebuild_R(corr())
            lam_tab = labels_grid(sp, cts)

        # --- controller: identical to the Dirac loop --------------------
        for _ in range(args.inner):
            sel = torch.randint(len(pool_1), (args.mb,), device=sp.device)
            X1, c = pool_1[sel], pool_c[sel]
            ti = torch.randint(steps, (args.mb,), device=sp.device)
            with torch.no_grad():
                Xt = sample_bridge_nd(sp, c, X1, ti, K0, K1)
                if args.estimator == "full":
                    lam = lam_tab[ti, X1]
                else:
                    lam = sp.labels_sampled(X1, cts[ti], mode=args.estimator,
                                            n_a=args.n_a)
                occ = sp.eocc[Xt]
                y = occ + lam
            tt = ti.to(torch.float64) / steps
            a = net(tt, sp.S[Xt]).to(torch.float64)
            av = a[:, sp.edge_i, sp.edge_j].clamp(-20.0, 20.0)
            mask = sp.emask[Xt].to(torch.float64)
            if args.loss == "bregman":
                per = occ * torch.exp(av) - av * y
            else:
                per = (occ * torch.exp(av) - y) ** 2
            loss = (per * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = propagate_exact(sp, lambda t: net_mult_all(sp, net, t),
                                    steps, p0=sp.p0_vec)
                lf = corr()
                cerr = float(((lf - lf.mean())
                              - (logfhat_ex - logfhat_ex.mean())).abs().max())
            tv = 0.5 * float((p - sp.pi).abs().sum())
            ho = 0.5 * float((occ_hist(sp, p)
                              - occ_hist(sp, sp.pi)).abs().sum())
            hist.append({"iter": it, "TV": tv, "occ_hist_TV": ho,
                         "loss": float(loss), "loss_h": float(loss_c),
                         "corrector_max_err": cerr})
            print(f"  it {it:4d}  loss {float(loss):9.4f}  "
                  f"max|log fhat - exact| {cerr:.2e}   exact-law TV {tv:.5f}"
                  f"   occ-hist TV {ho:.5f}   ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        p = propagate_exact(sp, lambda t: net_mult_all(sp, net, t), steps,
                            p0=sp.p0_vec)
        c0 = torch.multinomial(sp.nu0, args.n_samples, replacement=True)
        idx = simulate(sp, lambda t, i: net_mult_on(sp, net, t, i),
                       args.n_samples, steps, idx0=sp.src_idx[c0])
    viol = int((sp.S[idx].sum(dim=1) != sp.N).sum())
    tv = 0.5 * float((p - sp.pi).abs().sum())
    stats = report_law(sp, p, "final")
    emp = torch.bincount(idx, minlength=sp.M).to(torch.float64)
    tv_emp = 0.5 * float((emp / emp.sum() - sp.pi).abs().sum())
    iid = torch.multinomial(sp.pi, args.n_samples, replacement=True)
    e2 = torch.bincount(iid, minlength=sp.M).to(torch.float64)
    tv_floor = 0.5 * float((e2 / e2.sum() - sp.pi).abs().sum())

    print("\n=== GATES ===")
    print(f"B1  TV <= 0.05 : {'PASS' if tv <= 0.05 else 'FAIL'}  "
          f"(exact-law TV {tv:.5f})")
    print(f"B2  constraint : {'PASS' if viol == 0 else 'FAIL'}  "
          f"({viol} violations in {args.n_samples} samples)")
    print(f"    empirical TV {tv_emp:.5f}   iid floor {tv_floor:.5f}   "
          f"corrector max err {cerr:.2e}")

    if args.ckpt_dir:
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          nets={"control": net, "corrector": corr},
                          samples=idx.to(torch.int32),
                          extra={"config": vars(args), "final_TV": tv,
                                 "empirical_TV": tv_emp,
                                 "iid_TV_floor": tv_floor, "violations": viol,
                                 "corrector_max_err": cerr,
                                 "sinkhorn_marginal_err": sink_err,
                                 "nu0": sp.nu0.cpu(),
                                 "log_fhat_exact": logfhat_ex.cpu(),
                                 "log_fhat_learned": corr().detach().cpu(),
                                 "exact_law": p.detach().cpu(),
                                 "pi": sp.pi.detach().cpu(),
                                 "states": sp.S.detach().cpu()})
        print(f"  ckpt -> {pth}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "params": n_par + sp.m,
                   "source": "nondirac", "nu0": sp.nu0.tolist(),
                   "history": hist, "final_TV": tv, "empirical_TV": tv_emp,
                   "iid_TV_floor": tv_floor, "violations": viol,
                   "corrector_max_err": cerr,
                   "sinkhorn_marginal_err": sink_err, "stats": stats},
                  f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (tv <= 0.05 and viol == 0) else 1


# ============================================================================
# SCALE  (m = N = 32 / 128 / 1000)  --  no enumeration anywhere
# ============================================================================
#
# Three exact factorisations make the large-m case O(m) per state instead of
# O(m^2) or O(|X|):
#
# (1) f1 is exactly product-form.  With  w(n) = Gamma(n+d)/(n! Gamma(d))  and
#     p_base(xi|eta_0) = N!/prod(xi_i!) prod q_i^{xi_i},
#
#         log f1(xi) = sum_i G(xi_i, q_i) - log N!,
#         G(n, q)    = lgamma(n+d) - lgamma(d) - n log q.
#
#     Hence the terminal ratio for "take one from a, put it on b" is
#
#         log R_{b<-a}(xi) = u_a + v_b,
#         u_a = log q_a - log(xi_a - 1 + d),
#         v_b = log(xi'_b + d) - log q_b     with  xi' = xi - e_a
#
#     which correctly gives R = 1 when b == a.
#
# (2) Therefore the FULL correction sum is available at any m:
#
#         T[b] := sum_a xi_a R_{b<-a} = xi_b + e^{v_b} ( U - xi_b e^{u_b} ),
#         U    := sum_a xi_a e^{u_a},
#         Lambda_ji(xi) = xi_i (e^{u_i + v_j} - 1) - c_t ( T[j] - T[i] ).
#
#     So at m = 1000 we can still compare the one-sample estimators against
#     the exact full sum -- B3 becomes a genuine scale sweep.
#
# (3) The controller is parameterised in rank-1 log form
#
#         a_theta(t, eta, i, j) = alpha_i + beta_j ,
#
#     which is not an approximation of convenience: phi_t is a semigroup
#     applied to a product-form f1, so log phi(eta - e_i + e_j) - log phi(eta)
#     is exactly of this shape in the product-form limit.  It also makes
#     simulation O(m):  the total rate factorises, and -- crucially -- if we
#     ALLOW the destination j to equal the source i (a no-op) then the
#     remaining j != i rates are exactly gamma/(m-1) eta_i e^{alpha_i+beta_j},
#     so the "self-loop" trick is exact, not an approximation.  Departures
#     from mode i are then Binomial(eta_i, p_i) and, because every departing
#     particle draws its destination from the SAME law softmax(beta), all
#     arrivals are a single Multinomial(K, softmax(beta)).
#
# Reference is recovered exactly at alpha = beta = 0.


class ScaleOccupation:
    """m = N large.  Only per-mode vectors, never the state space."""

    def __init__(self, m=32, N=None, d=0.5, tau=1.0, gamma=4.0, source=0,
                 device=DEV):
        self.m = m
        self.N = m if N is None else N
        self.d, self.tau, self.gamma = d, tau, gamma
        self.source, self.device = source, device

    def q_at(self, Gamma):
        q = C.occupation_single_particle_probs(self.m, Gamma, self.source)
        return torch.tensor(q, device=self.device, dtype=torch.float64)

    def q1(self):
        return self.q_at(self.gamma)

    def uv(self, xi, q):
        """u_a, v_b of the terminal-ratio factorisation.  xi: (B, m) float."""
        u = torch.log(q)[None, :] - torch.log(xi - 1.0 + self.d)
        u = torch.where(xi > 0, u, torch.full_like(u, -1e30))
        v = torch.log(xi + self.d) - torch.log(q)[None, :]
        return u, v

    def uv_nondirac(self, xi, ah, bh):
        """u_a, v_b when the closed-form reference ratio is replaced by a
        learned corrector.  ah, bh are the rank-1 corrector outputs (B, m).

            log R_{b<-a} = log[ mu(xi') / mu(xi) ] - h_{b<-a}
                         = [ log xi_a   - log(xi_a - 1 + d) - ah_a ]
                         + [ log(xi_b + d) - log(xi_b + 1)  - bh_b ],

        so a RANK-1 corrector preserves the exact u_a + v_b factorisation that
        makes labels_full O(m) rather than O(m^2).  That is the whole reason to
        measure the rank-1 family before replacing it: the Dirac q_a/q_b term
        is simply swapped for ah_a + bh_b and nothing downstream changes shape.
        The exact corrector need not be rank-1, so this is an approximation
        family -- if it fails, that is an architecture limit, not a failure of
        the intertwining identity.

        Unlike the Dirac branch, ah/bh are unbounded network outputs, so u and
        v MUST be range-limited here.  labels_full forms
        T = xi + e^v (U - xi e^u) and then T_j - T_i; if either exponential
        overflows to +inf that difference is inf - inf = NaN, which then
        poisons the controller weights and kills the run several hundred
        iterations later with an unrelated-looking error.  +-60 keeps every
        intermediate (including e^{u+v} ~ e^120 ~ 1e52) inside float64.
        """
        ah = ah.clamp(-20.0, 20.0)
        bh = bh.clamp(-20.0, 20.0)
        u = (torch.log(xi.clamp_min(1e-300)) - torch.log(xi - 1.0 + self.d)
             - ah).clamp(-60.0, 60.0)
        u = torch.where(xi > 0, u, torch.full_like(u, -1e30))
        v = (torch.log(xi + self.d) - torch.log(xi + 1.0)
             - bh).clamp(-60.0, 60.0)
        return u, v

    def q_atoms(self, Gamma):
        """q^{(c)}_j = o + r [j == c] for a source concentrated at mode c.

        The reference walk is the complete graph, so one particle started at c
        is at c with probability o + r and at any other mode with probability o.
        Two scalars describe every source row, which is what makes the
        corrector label O(1) per queried transfer even at m = 1000.
        """
        r = math.exp(-self.m * Gamma / (self.m - 1.0))
        return (1.0 - r) / self.m, r

    def labels_full(self, xi, q, c_t, i_idx, j_idx, uv=None):
        """Exact full-sum Lambda_ji(xi) for the queried edges.  O(m)."""
        u, v = self.uv(xi, q) if uv is None else uv
        eu = torch.exp(u.clamp(min=-60.0))
        U = (xi * eu).sum(dim=1, keepdim=True)                    # (B,1)
        ev = torch.exp(v)
        T = xi + ev * (U - xi * eu)                               # (B, m)
        ar = torch.arange(len(xi), device=xi.device)
        # careful: v_j is evaluated at xi' = xi - e_i, which only differs when
        # j == i, and in that case R_{i<-i} = 1 exactly.
        Rji = torch.exp((u[ar, i_idx] + v[ar, j_idx]).clamp(-60.0, 60.0))
        Rji = torch.where(i_idx == j_idx, torch.ones_like(Rji), Rji)
        first = xi[ar, i_idx] * (Rji - 1.0)
        first = torch.where(xi[ar, i_idx] > 0, first, torch.zeros_like(first))
        return first - c_t * (T[ar, j_idx] - T[ar, i_idx])

    def labels_sampled(self, xi, q, c_t, i_idx, j_idx, mode="occupancy", n_a=1,
                       uv=None):
        u, v = self.uv(xi, q) if uv is None else uv
        B = len(xi)
        ar = torch.arange(B, device=xi.device)
        Rji = torch.exp((u[ar, i_idx] + v[ar, j_idx]).clamp(-60.0, 60.0))
        Rji = torch.where(i_idx == j_idx, torch.ones_like(Rji), Rji)
        first = xi[ar, i_idx] * (Rji - 1.0)
        first = torch.where(xi[ar, i_idx] > 0, first, torch.zeros_like(first))
        acc = torch.zeros(B, device=xi.device, dtype=torch.float64)
        for _ in range(n_a):
            if mode == "uniform":
                a = torch.randint(self.m, (B,), device=xi.device)
                scale = self.m * c_t * xi[ar, a]
            else:
                a = torch.multinomial(xi / self.N, 1)[:, 0]
                scale = c_t * float(self.N) * torch.ones_like(first)
            ua = u[ar, a]
            Rja = torch.exp((ua + v[ar, j_idx]).clamp(-60.0, 60.0))
            Rja = torch.where(a == j_idx, torch.ones_like(Rja), Rja)
            Ria = torch.exp((ua + v[ar, i_idx]).clamp(-60.0, 60.0))
            Ria = torch.where(a == i_idx, torch.ones_like(Ria), Ria)
            acc = acc + scale * (Rja - Ria)
        return first - acc / n_a


class ScaleController(torch.nn.Module):
    """Permutation-equivariant DeepSets net:  eta -> (alpha_i, beta_i)."""

    def __init__(self, m, N, hidden=128, n_freq=4):
        super().__init__()
        self.m, self.N, self.n_freq = m, N, n_freq
        din = 3 + 2 + 2 * n_freq + 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.to(torch.float32)

    def forward(self, t, eta, source):
        """t: (B,), eta: (B, m) -> alpha, beta each (B, m)."""
        B, m = eta.shape
        e = eta.to(torch.float32)
        # source=None is the non-Dirac case: the controller must NOT be told
        # which component the trajectory came from, or it is sampling a
        # different, source-conditioned problem.  The feature is zeroed rather
        # than removed so the two branches share one architecture and one
        # parameter count.
        src = torch.zeros(B, m, device=e.device, dtype=e.dtype)
        if source is not None:
            src[:, source] = 1.0
        per = torch.stack([e / self.N, torch.log1p(e), src], dim=-1)
        t = t.to(torch.float32)[:, None]
        k = torch.arange(1, self.n_freq + 1, device=e.device,
                         dtype=e.dtype)[None, :]
        glob = torch.cat([t, 1.0 - t, torch.sin(math.pi * k * t),
                          torch.cos(math.pi * k * t),
                          e.max(dim=1, keepdim=True).values / self.N,
                          (e == 0).to(e.dtype).mean(dim=1, keepdim=True),
                          ((e / self.N) ** 2).sum(dim=1, keepdim=True)], dim=-1)
        x = torch.cat([per, glob[:, None, :].expand(B, m, glob.shape[-1])],
                      dim=-1)
        out = self.net(x)
        al, be = out[..., 0], out[..., 1]
        # gauge-fix the shift degeneracy  (alpha_i, beta_j) -> (alpha+s, beta-s)
        s = be.mean(dim=1, keepdim=True)
        return al + s, be - s


def leaky_clamp(x, lo, hi, leak=1e-2):
    """clamp(x, lo, hi) but with a small surviving slope outside the range.

    A hard clamp on the exponent of a Bregman loss is a trap: once the network
    saturates past the bound the gradient is exactly zero, so the corrector can
    never come back and the run is dead while still reporting finite numbers.
    The leak bounds the exponential just as effectively but keeps a restoring
    gradient.
    """
    c = torch.clamp(x, lo, hi)
    return c + leak * (x - c)


def scale_step(sp, eta, alpha, beta, dt, generator=None):
    """One exact-in-rate tau-leap step of the rank-1 controlled process."""
    a = alpha.clamp(-10.0, 10.0).to(torch.float64)
    b = beta.clamp(-10.0, 10.0).to(torch.float64)
    # The split a_ij = alpha_i + beta_j is shift-degenerate, so beta is
    # recentred for numerical stability -- and the shift MUST be handed back
    # to alpha, otherwise every rate is silently rescaled by e^{-max beta}.
    bmax = b.max(dim=1, keepdim=True).values
    b = b - bmax
    eb = torch.exp(b)
    Btot = eb.sum(dim=1, keepdim=True)
    # per-particle departure probability from mode i (self-loops allowed)
    p = (sp.gamma / (sp.m - 1.0)) * torch.exp(
        (a + bmax).clamp(max=20.0)) * Btot * dt
    p = p.clamp(0.0, 0.9)
    dep = torch.binomial(eta.to(torch.float64), p)                # (B, m)
    K = dep.sum(dim=1)                                            # (B,)
    if not bool(torch.isfinite(K).all()):
        raise RuntimeError(
            "scale_step: non-finite departure counts -- the controller "
            "weights have diverged (check for non-finite training labels)")
    # every departing particle draws its destination from the SAME law
    # softmax(beta), so all arrivals in a row are one Multinomial(K, p_beta).
    # Sample it as K iid categorical draws, padded to Kmax and masked.
    arr = torch.zeros_like(eta)
    Kmax = int(K.max().item())
    if Kmax > 0:
        probs = eb / Btot
        draw = torch.multinomial(probs, Kmax, replacement=True,
                                 generator=generator)             # (B, Kmax)
        live = (torch.arange(Kmax, device=eta.device)[None, :]
                < K[:, None]).to(torch.float64)
        arr.scatter_add_(1, draw, live)
    return eta - dep + arr


def simulate_scale(sp, net, batch, steps, generator=None, c0=None):
    """c0 is a per-trajectory source component; the default is the Dirac at
    sp.source.  The controller is never told c0 -- see ScaleController."""
    eta = torch.zeros(batch, sp.m, device=sp.device, dtype=torch.float64)
    if c0 is None:
        eta[:, sp.source] = sp.N
    else:
        eta.scatter_(1, c0[:, None], float(sp.N))
    src = sp.source if c0 is None else None
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((batch,), s / steps, device=sp.device,
                       dtype=torch.float64)
        if net is None:
            al = torch.zeros_like(eta)
            be = torch.zeros_like(eta)
        else:
            al, be = net(t, eta, src)
        eta = scale_step(sp, eta, al, be, dt, generator=generator)
    return eta


def bridge_scale(sp, X1, tvals, generator=None, c0=None):
    """Exact reference bridge  X_t | X_0 = N e_c, X_1 = xi,  in O(N).

    All particles start at c, so expand xi into N endpoint labels and bridge
    each particle independently:  w(y) = (om0 + rho0 [y=c])(om1 + rho1 [y=l]),
    a 4-component mixture that is sampled in O(1) per particle.
    """
    B, m = X1.shape
    N = sp.N
    modes = torch.arange(m, device=X1.device).repeat(B)
    ends = torch.repeat_interleave(modes, X1.reshape(-1).long()).view(B, N)
    r0 = torch.exp(-m * sp.gamma * tvals / (m - 1.0))              # (B,)
    r1 = torch.exp(-m * sp.gamma * (1.0 - tvals) / (m - 1.0))
    o0 = (1.0 - r0) / m
    o1 = (1.0 - r1) / m
    # c is the source component: a single mode for the Dirac branch, one per
    # trajectory for the non-Dirac branch.  Only these two lines and the
    # "snap back to c" line below depend on which it is.
    c = torch.full((B, 1), sp.source, device=X1.device, dtype=torch.long) \
        if c0 is None else c0[:, None]
    isc = (ends == c).to(torch.float64)                            # (B, N)
    w_u = (m * o0 * o1)[:, None].expand(B, N)
    w_c = (r0 * o1)[:, None].expand(B, N)
    w_l = (o0 * r1)[:, None].expand(B, N)
    w_b = (r0 * r1)[:, None] * isc
    W = torch.stack([w_u, w_c, w_l, w_b], dim=-1)
    br = torch.multinomial(W.reshape(-1, 4), 1, generator=generator).view(B, N)
    unif = torch.randint(m, (B, N), device=X1.device, generator=generator)
    y = torch.where(br == 0, unif, ends)
    y = torch.where((br == 1) | (br == 3), c.expand(B, N), y)
    out = torch.zeros(B, m, device=X1.device, dtype=torch.float64)
    out.scatter_add_(1, y, torch.ones_like(y, dtype=torch.float64))
    return out


# -- scale metrics -----------------------------------------------------------
def _ks_discrete(a, b, hi):
    ca = torch.bincount(a.long().reshape(-1), minlength=hi + 1).double()
    cb = torch.bincount(b.long().reshape(-1), minlength=hi + 1).double()
    ca = torch.cumsum(ca / ca.sum(), 0)
    cb = torch.cumsum(cb / cb.sum(), 0)
    return float((ca - cb).abs().max())


def _w1_discrete(a, b, hi):
    ca = torch.bincount(a.long().reshape(-1), minlength=hi + 1).double()
    cb = torch.bincount(b.long().reshape(-1), minlength=hi + 1).double()
    ca = torch.cumsum(ca / ca.sum(), 0)
    cb = torch.cumsum(cb / cb.sum(), 0)
    return float((ca - cb).abs().sum())


def scale_metrics(sp, ours, exact):
    hi = sp.N
    viol = int((ours.sum(dim=1).round().long() != sp.N).sum())
    ks_occ = _ks_discrete(ours, exact, hi)
    mo, me = ours.max(dim=1).values, exact.max(dim=1).values
    ks_max = _ks_discrete(mo, me, hi)
    w1_max = _w1_discrete(mo, me, hi)
    Eo = C.inclusion_energy(ours, sp.tau, sp.d)
    Ee = C.inclusion_energy(exact, sp.tau, sp.d)
    return {"violations": viol, "KS_occ": ks_occ, "KS_max": ks_max,
            "W1_max": w1_max, "W1_max_frac": w1_max / sp.N,
            "cond_ours": float((mo / sp.N).mean()),
            "cond_exact": float((me / sp.N).mean()),
            "E_ours": float(Eo.mean()), "E_exact": float(Ee.mean()),
            "E_KS": float(C.ks_against_grid_cdf(
                Eo.cpu().numpy(),
                np.sort(Ee.cpu().numpy()),
                np.arange(1, len(Ee) + 1) / len(Ee)))}


def run_scale(args):
    sp = ScaleOccupation(m=args.m, N=args.N if args.N > 0 else args.m,
                         d=args.d, tau=args.tau, gamma=args.gamma)
    torch.manual_seed(args.seed)
    q = sp.q1()
    net = ScaleController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    print(f"[scale] m={sp.m} N={sp.N} d={sp.d} gamma={sp.gamma}  "
          f"params={n_par}  steps={steps}  est={args.estimator}  "
          f"device={sp.device}")

    exact = C.sample_inclusion_exact(args.n_samples, sp.m, sp.N, sp.d,
                                     device=sp.device).to(torch.float64)
    with torch.no_grad():
        ref = simulate_scale(sp, None, args.n_samples, steps)
    mref = scale_metrics(sp, ref, exact)
    print(f"  reference (no control):  KS_occ={mref['KS_occ']:.4f}  "
          f"KS_max={mref['KS_max']:.4f}  W1max/N={mref['W1_max_frac']:.4f}  "
          f"cond {mref['cond_ours']:.4f} vs {mref['cond_exact']:.4f}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)
    buf, hist = [], []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            X1 = simulate_scale(sp, net, args.batch, steps)
        buf.append(X1)
        if len(buf) > args.buffer:
            buf.pop(0)
        pool = torch.cat(buf)

        for _ in range(args.inner):
            sel = torch.randint(len(pool), (args.mb,), device=sp.device)
            x1 = pool[sel]
            tv = torch.rand(args.mb, device=sp.device, dtype=torch.float64)
            with torch.no_grad():
                xt = bridge_scale(sp, x1, tv)
                ct = torch.tensor(
                    [C.occupation_c_t(sp.m, sp.gamma * (1 - float(z)))
                     for z in tv.cpu()], device=sp.device)
                i_idx = torch.multinomial(xt / sp.N, 1)[:, 0]
                j_idx = torch.randint(sp.m, (args.mb,), device=sp.device)
                if args.estimator == "full":
                    lam = sp.labels_full(x1, q, ct, i_idx, j_idx)
                else:
                    lam = sp.labels_sampled(x1, q, ct, i_idx, j_idx,
                                            mode=args.estimator, n_a=args.n_a)
                ar = torch.arange(args.mb, device=sp.device)
                occ = xt[ar, i_idx]
                y = occ + lam
            al, be = net(tv, xt, sp.source)
            av = (al.to(torch.float64)[ar, i_idx]
                  + be.to(torch.float64)[ar, j_idx]).clamp(-15.0, 15.0)
            if args.loss == "bregman":
                loss = (occ * torch.exp(av) - av * y).mean()
            else:
                loss = ((occ * torch.exp(av) - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                ours = simulate_scale(sp, net, args.n_samples, steps)
            mm = scale_metrics(sp, ours, exact)
            mm["iter"] = it
            mm["loss"] = float(loss)
            hist.append(mm)
            print(f"  it {it:4d}  loss {float(loss):11.3f}  "
                  f"KS_occ {mm['KS_occ']:.4f}  KS_max {mm['KS_max']:.4f}  "
                  f"W1max/N {mm['W1_max_frac']:.4f}  viol {mm['violations']}  "
                  f"({time.time()-t0:.0f}s)")

    mm = hist[-1]
    print(f"\n  condensate fraction  ours {mm['cond_ours']:.4f}   "
          f"exact {mm['cond_exact']:.4f}")
    print(f"  mean energy          ours {mm['E_ours']:.4f}   "
          f"exact {mm['E_exact']:.4f}   (KS {mm['E_KS']:.4f})")
    okKS = mm["KS_occ"] <= 0.05
    okW1 = mm["W1_max_frac"] <= 0.05
    okV = mm["violations"] == 0
    print("\n=== GATES ===")
    print(f"B2a occupancy KS <= 0.05     : {'PASS' if okKS else 'FAIL'}  "
          f"({mm['KS_occ']:.4f})")
    print(f"B2b max-occ W1 <= 5% of N    : {'PASS' if okW1 else 'FAIL'}  "
          f"({mm['W1_max_frac']:.4f})")
    print(f"B2c constraint violations = 0: {'PASS' if okV else 'FAIL'}  "
          f"({mm['violations']}/{args.n_samples})")
    if args.ckpt_dir:
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          samples=ours.to(torch.int32),
                          extra={"config": vars(args), "metrics": mm,
                                 "reference": mref,
                                 "exact_samples": exact.to(torch.int32).cpu()})
        print(f"  ckpt -> {pth}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "params": n_par,
                   "reference": mref, "history": hist,
                   "B2": bool(okKS and okW1 and okV)}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (okKS and okW1 and okV) else 1


def run_scale_nondirac(args):
    """Non-Dirac source at scale: nu_0 uniform over the m concentrated states.

    The first attempt deliberately keeps the existing rank-1 architecture for
    both networks.  A rank-1 corrector is not implied by any exactness argument
    here -- with a source mixture fhat_1 is a sum of m multinomial rows, not a
    product -- but it is the family the O(m) simulator and the O(m) label
    formula already support, so it must be measured before it is replaced.
    """
    sp = ScaleOccupation(m=args.m, N=args.N if args.N > 0 else args.m,
                         d=args.d, tau=args.tau, gamma=args.gamma)
    torch.manual_seed(args.seed)
    net = ScaleController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    net_h = ScaleController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    opt_h = torch.optim.Adam(net_h.parameters(), lr=args.lr_h)
    steps = args.steps
    o, r = sp.q_atoms(sp.gamma)
    print(f"[scale-nd] m={sp.m} N={sp.N} d={sp.d} gamma={sp.gamma}  "
          f"params={n_par}x2  steps={steps}  est={args.estimator}  "
          f"source=Uniform over {sp.m} modes  q=(o {o:.3e}, o+r {o+r:.3e})  "
          f"device={sp.device}")

    exact = C.sample_inclusion_exact(args.n_samples, sp.m, sp.N, sp.d,
                                     device=sp.device).to(torch.float64)
    with torch.no_grad():
        c0 = torch.randint(sp.m, (args.n_samples,), device=sp.device)
        ref = simulate_scale(sp, None, args.n_samples, steps, c0=c0)
    mref = scale_metrics(sp, ref, exact)
    print(f"  reference (no control):  KS_occ={mref['KS_occ']:.4f}  "
          f"KS_max={mref['KS_max']:.4f}  W1max/N={mref['W1_max_frac']:.4f}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)
    # The corrector needs its own decay: it is the component that ran away at
    # m = 32, and a constant high lr_h keeps injecting energy into a loop whose
    # own targets it moves.
    sched_h = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_h, T_max=max(1, args.iters * args.inner_h),
        eta_min=args.lr_h * 0.05)
    ones = torch.ones(args.mb, device=sp.device, dtype=torch.float64)
    ar = torch.arange(args.mb, device=sp.device)
    buf_c, buf_1, hist = [], [], []
    n_bad, n_skip = 0, 0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        with torch.no_grad():
            c0 = torch.randint(sp.m, (args.batch,), device=sp.device)
            X1 = simulate_scale(sp, net, args.batch, steps, c0=c0)
        buf_c.append(c0)
        buf_1.append(X1)
        if len(buf_1) > args.buffer:
            buf_c.pop(0)
            buf_1.pop(0)
        pool_c, pool_1 = torch.cat(buf_c), torch.cat(buf_1)

        # --- corrector: exp(h_{b<-a}) -> E[ Q_{ba}(c, xi) | xi ] ------------
        # Q_{ba}(c, xi) = (xi_a / (xi_b + 1)) (q_b^{(c)} / q_a^{(c)}), which is
        # O(1) because q^{(c)} takes only the two values o and o + r.
        for _ in range(args.inner_h):
            sel = torch.randint(len(pool_1), (args.mb,), device=sp.device)
            x1, c = pool_1[sel], pool_c[sel]
            a_i = torch.multinomial(x1 / sp.N, 1)[:, 0]
            b_i = torch.randint(sp.m, (args.mb,), device=sp.device)
            with torch.no_grad():
                qa = o + r * (a_i == c).to(torch.float64)
                qb = o + r * (b_i == c).to(torch.float64)
                q_lab = (x1[ar, a_i] / (x1[ar, b_i] + 1.0)) * (qb / qa)
            ah, bh = net_h(ones, x1, None)
            hv = leaky_clamp(ah.to(torch.float64)[ar, a_i]
                             + bh.to(torch.float64)[ar, b_i], -15.0, 15.0)
            keep = (a_i != b_i).to(torch.float64)
            loss_h = (((torch.exp(hv) - hv * q_lab) * keep).sum()
                      / keep.sum().clamp(min=1.0))
            if not bool(torch.isfinite(loss_h)):
                n_skip += 1
                opt_h.zero_grad(set_to_none=True)
                sched_h.step()
                continue
            opt_h.zero_grad(set_to_none=True)
            loss_h.backward()
            torch.nn.utils.clip_grad_norm_(net_h.parameters(), 1.0)
            opt_h.step()
            sched_h.step()

        # --- controller: the Dirac loop with the corrected u, v -------------
        for _ in range(args.inner):
            sel = torch.randint(len(pool_1), (args.mb,), device=sp.device)
            x1, c = pool_1[sel], pool_c[sel]
            tv = torch.rand(args.mb, device=sp.device, dtype=torch.float64)
            with torch.no_grad():
                xt = bridge_scale(sp, x1, tv, c0=c)
                ct = torch.tensor(
                    [C.occupation_c_t(sp.m, sp.gamma * (1 - float(z)))
                     for z in tv.cpu()], device=sp.device)
                i_idx = torch.multinomial(xt / sp.N, 1)[:, 0]
                j_idx = torch.randint(sp.m, (args.mb,), device=sp.device)
                ah, bh = net_h(ones, x1, None)
                uv = sp.uv_nondirac(x1, ah.to(torch.float64),
                                    bh.to(torch.float64))
                if args.estimator == "full":
                    lam = sp.labels_full(x1, None, ct, i_idx, j_idx, uv=uv)
                else:
                    lam = sp.labels_sampled(x1, None, ct, i_idx, j_idx,
                                            mode=args.estimator, n_a=args.n_a,
                                            uv=uv)
                occ = xt[ar, i_idx]
                y = occ + lam
                # A single non-finite label would silently NaN every weight
                # (and only surface ~1000 iterations later inside the
                # simulator).  Drop those rows instead: y = occ is the
                # zero-correction label, so they contribute no signal.
                bad = ~torch.isfinite(y)
                if bool(bad.any()):
                    n_bad += int(bad.sum())
                    y = torch.where(bad, occ, y)
            al, be = net(tv, xt, None)
            av = leaky_clamp(al.to(torch.float64)[ar, i_idx]
                             + be.to(torch.float64)[ar, j_idx], -15.0, 15.0)
            if args.loss == "bregman":
                loss = (occ * torch.exp(av) - av * y).mean()
            else:
                loss = ((occ * torch.exp(av) - y) ** 2).mean()
            if not bool(torch.isfinite(loss)):
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                c0 = torch.randint(sp.m, (args.n_samples,), device=sp.device)
                ours = simulate_scale(sp, net, args.n_samples, steps, c0=c0)
            mm = scale_metrics(sp, ours, exact)
            mm.update({"iter": it, "loss": float(loss),
                       "loss_h": float(loss_h), "bad_labels": n_bad,
                       "skipped_steps": n_skip})
            hist.append(mm)
            print(f"  it {it:4d}  loss {float(loss):11.3f}  "
                  f"loss_h {float(loss_h):8.4f}  KS_occ {mm['KS_occ']:.4f}  "
                  f"KS_max {mm['KS_max']:.4f}  W1max/N "
                  f"{mm['W1_max_frac']:.4f}  viol {mm['violations']}  "
                  f"bad {n_bad}/{n_skip}  ({time.time()-t0:.0f}s)")

    mm = hist[-1]
    okKS, okW1 = mm["KS_occ"] <= 0.05, mm["W1_max_frac"] <= 0.05
    okV = mm["violations"] == 0
    print("\n=== GATES ===")
    print(f"B2a occupancy KS <= 0.05     : {'PASS' if okKS else 'FAIL'}  "
          f"({mm['KS_occ']:.4f})")
    print(f"B2b max-occ W1 <= 5% of N    : {'PASS' if okW1 else 'FAIL'}  "
          f"({mm['W1_max_frac']:.4f})")
    print(f"B2c constraint violations = 0: {'PASS' if okV else 'FAIL'}  "
          f"({mm['violations']}/{args.n_samples})")
    if args.ckpt_dir:
        pth = C.save_ckpt(args.ckpt_dir, args.tag, net=net,
                          nets={"control": net, "corrector": net_h},
                          samples=ours.to(torch.int32),
                          extra={"config": vars(args), "metrics": mm,
                                 "reference": mref, "source": "nondirac",
                                 "exact_samples": exact.to(torch.int32).cpu()})
        print(f"  ckpt -> {pth}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "params": 2 * n_par,
                   "source": "nondirac", "reference": mref, "history": hist,
                   "B2": bool(okKS and okW1 and okV)}, f, indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if (okKS and okW1 and okV) else 1


def run_scalevar(args):
    """B3 at scale: one-sample estimator variance vs the exact full sum."""
    sp = ScaleOccupation(m=args.m, N=args.N if args.N > 0 else args.m,
                         d=args.d, tau=args.tau, gamma=args.gamma)
    torch.manual_seed(args.seed)
    q = sp.q1()
    print(f"[scalevar] m={sp.m} N={sp.N}   variance of Lambda estimators")
    rows = []
    for t in [0.25, 0.5, 0.75, 0.9]:
        ct = torch.full((args.mb,), C.occupation_c_t(sp.m, sp.gamma * (1 - t)),
                        device=sp.device, dtype=torch.float64)
        x1 = C.sample_inclusion_exact(args.mb, sp.m, sp.N, sp.d,
                                      device=sp.device).to(torch.float64)
        i_idx = torch.multinomial(x1 / sp.N, 1)[:, 0]
        j_idx = torch.randint(sp.m, (args.mb,), device=sp.device)
        full = sp.labels_full(x1, q, ct, i_idx, j_idx)
        rec = {"t": t}
        for mode, na in [("uniform", 1), ("occupancy", 1), ("occupancy", 4)]:
            s1 = torch.zeros_like(full)
            s2 = torch.zeros_like(full)
            for _ in range(args.reps):
                v = sp.labels_sampled(x1, q, ct, i_idx, j_idx,
                                      mode=mode, n_a=na)
                s1 += v
                s2 += v * v
            mean = s1 / args.reps
            var = (s2 / args.reps - mean ** 2).clamp(min=0.0)
            bias = float((mean - full).abs().mean())
            rec[f"{mode}{na}"] = float(var.mean())
            rec[f"{mode}{na}_bias"] = bias
        rows.append(rec)
        print(f"  t={t:.2f}  uniform-1 var {rec['uniform1']:.3e} "
              f"(bias {rec['uniform1_bias']:.3e})   "
              f"occupancy-1 var {rec['occupancy1']:.3e} "
              f"(bias {rec['occupancy1_bias']:.3e})   "
              f"occupancy-4 var {rec['occupancy4']:.3e}")
    ok = all(r["occupancy1"] <= r["uniform1"] for r in rows)
    print(f"\nB3(scale) {'PASS' if ok else 'FAIL'}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows, "B3": bool(ok)}, f,
                  indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if ok else 1


def run_scale_jumps(args):
    """Jump-count audit of the large-m simulator.

    The enumerated simulator :func:`simulate` moves at most one particle per
    grid interval; the large-m simulator :func:`scale_step` does not -- it is a
    tau-leap whose departures from mode i are Binomial(eta_i, p_i), so a bin
    can move many particles at once.  The two are different algorithms and the
    step count means a different thing in each, which is exactly the point this
    audit exists to make checkable.  It reports, for a trained checkpoint:

      * transfers per path, against the one-jump-per-bin budget of ``steps``,
      * the largest number of particles moved in any single bin,
      * max_i eta_i at t = 1, against the ``N - steps`` floor that a
        one-jump-per-bin chain started from ``N e_c`` would be forced to obey,
      * the peak per-particle departure probability and the number of times it
        hit the 0.9 clamp inside ``scale_step``, which is the bias term of the
        leap and is only negligible while the clamp stays inactive.
    """
    torch.manual_seed(args.seed)
    sp = ScaleOccupation(m=args.m, N=args.N, d=args.d, tau=args.tau,
                         gamma=args.gamma)
    ck = torch.load(f"{args.ckpt_dir}/{args.tag}.pt", map_location=sp.device,
                    weights_only=False)
    net = ScaleController(sp.m, sp.N, hidden=args.hidden).to(sp.device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    steps, B, dt = args.steps, args.batch, 1.0 / args.steps
    gen = torch.Generator(device=sp.device)
    gen.manual_seed(args.seed)
    eta = torch.zeros(B, sp.m, device=sp.device, dtype=torch.float64)
    eta[:, sp.source] = sp.N
    moves = torch.zeros(B, device=sp.device, dtype=torch.float64)
    step_max, p_max, n_clamp, n_live = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for s in range(steps):
            t = torch.full((B,), s / steps, device=sp.device,
                           dtype=torch.float64)
            al, be = net(t, eta, sp.source)
            # replay the rate that scale_step will use, to audit the clamp
            a = al.clamp(-10.0, 10.0).to(torch.float64)
            b = be.clamp(-10.0, 10.0).to(torch.float64)
            bmax = b.max(dim=1, keepdim=True).values
            eb = torch.exp(b - bmax)
            p = ((sp.gamma / (sp.m - 1.0))
                 * torch.exp((a + bmax).clamp(max=20.0))
                 * eb.sum(dim=1, keepdim=True) * dt)
            live = eta > 0
            p_max = max(p_max, float(p[live].max()))
            n_clamp += int((p[live] > 0.9).sum())
            n_live += int(live.sum())
            prev = eta
            eta = scale_step(sp, eta, al, be, dt, generator=gen)
            mv = (eta - prev).clamp(min=0).sum(dim=1)
            moves += mv
            step_max = max(step_max, float(mv.max()))
    mx = eta.max(dim=1).values
    floor = sp.N - steps
    out = {
        "m": sp.m, "N": sp.N, "steps": steps, "paths": B,
        "transfers_per_path_mean": float(moves.mean()),
        "transfers_per_path_min": float(moves.min()),
        "transfers_per_path_max": float(moves.max()),
        "one_jump_per_bin_budget": steps,
        "max_particles_moved_in_one_bin": step_max,
        "max_occ_mean": float(mx.mean()),
        "max_occ_min": float(mx.min()),
        "max_occ_max": float(mx.max()),
        "one_jump_per_bin_max_occ_floor": floor,
        "paths_below_that_floor": int((mx < floor).sum()),
        "peak_departure_prob": p_max,
        "departure_clamp_hits": n_clamp,
        "occupied_mode_steps": n_live,
        "violations": int((eta.sum(dim=1).round().long() != sp.N).sum()),
    }
    print(f"m={sp.m} N={sp.N} steps={steps} paths={B}")
    print(f"  transfers/path       mean {out['transfers_per_path_mean']:.1f}"
          f"  min {out['transfers_per_path_min']:.0f}"
          f"  max {out['transfers_per_path_max']:.0f}"
          f"   (one-jump-per-bin budget {steps})")
    print(f"  particles per bin    max {step_max:.0f}"
          f"   (one-jump-per-bin would force 1)")
    print(f"  max_i eta_i at t=1   mean {out['max_occ_mean']:.2f}"
          f"  min {out['max_occ_min']:.0f}  max {out['max_occ_max']:.0f}"
          f"   (one-jump-per-bin floor N-steps = {floor};"
          f" paths below it {out['paths_below_that_floor']}/{B})")
    print(f"  departure prob       peak {p_max:.4e}"
          f"  clamp hits {n_clamp}/{n_live}")
    print(f"  constraint violations {out['violations']}/{B}")
    with open(args.out, "w") as f:
        json.dump({"jump_audit": out}, f,
                  indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0


def main():
    C.use_repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "exact", "var", "train",
                                    "train-nondirac", "scale",
                                    "scale-nondirac", "scalevar",
                                    "scale-jumps"])
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--N", type=int, default=0, help="0 means N = m")
    ap.add_argument("--d", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["bregman", "mse"], default="bregman")
    ap.add_argument("--estimator",
                    choices=["full", "uniform", "occupancy"], default="full")
    ap.add_argument("--n-a", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="json/results_occupation.json")
    # Default "ckpt", never "": a run whose checkpoint is not
    # written cannot be re-measured later and must be repeated in full.
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="occ")
    # non-Dirac only: nu_0(c) propto nu_skew^c over the m pure modes.  1.0 is
    # the uniform source, whose exact f_0 is constant by mode symmetry; a skew
    # breaks that symmetry so the corrector has something non-trivial to find.
    ap.add_argument("--nu-skew", dest="nu_skew", type=float, default=1.0)
    ap.add_argument("--inner-h", dest="inner_h", type=int, default=4)
    ap.add_argument("--lr-h", dest="lr_h", type=float, default=1e-2)
    args = ap.parse_args()
    if args.N <= 0:
        args.N = args.m
    return {"verify": run_verify, "exact": run_exact,
            "var": run_var, "train": run_train,
            "train-nondirac": run_train_nondirac,
            "scale": run_scale, "scale-nondirac": run_scale_nondirac,
            "scalevar": run_scalevar,
            "scale-jumps": run_scale_jumps}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
