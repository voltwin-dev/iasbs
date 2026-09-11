"""
common.py -- shared mathematical utilities for the structured-ASBS experiments.

Nothing in here trains a neural network.  Every routine is a deterministic
mathematical primitive that is unit-tested in ``tests_math.py``.

Conventions
-----------
* Continuous time t in [0, 1].
* Heat clock          r(t, s) = 1/2 * int_t^s sigma_u^2 du.
* Discrete jump clock Gamma(t, s) = int_t^s gamma_u du.
* Sphere kernels use r.
* St(4,2) spin-cover S^3 factors use r/2.   <-- mandatory distinction
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import torch
from scipy.linalg import expm
from scipy.special import eval_legendre

# ----------------------------------------------------------------------------
# 3.0  Repository layout
# ----------------------------------------------------------------------------
# ``common.py`` sits at the repository root, next to the three shared data
# directories.  Every experiment writes to ``json/``, ``ckpt/`` and ``fig/``
# using bare relative paths, so a run only lands in the right place if the
# process is rooted here.  ``use_repo_root()`` makes that true regardless of
# where the script was invoked from.
ROOT = os.path.dirname(os.path.abspath(__file__))


def use_repo_root():
    """chdir to the repository root so ``json/``, ``ckpt/`` and ``fig/`` resolve.

    Call this once at the top of a script's ``main()`` -- never at import time,
    so that importing ``common`` from a notebook or a test runner has no side
    effect on the caller's working directory.
    """
    os.chdir(ROOT)


def add_root_to_path():
    """Put the repository root on ``sys.path`` so ``import common`` works."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


# ----------------------------------------------------------------------------
# 3.1  Time and noise clocks
# ----------------------------------------------------------------------------


def heat_clock(t0, t1, sigma):
    """r = 1/2 * int sigma^2 dt for constant sigma."""
    return 0.5 * sigma * sigma * (t1 - t0)


def sphere_factor_clock_for_stiefel(t0, t1, sigma):
    """Each SU(2) ~= S^3 factor runs at half the SO(4) quotient heat time."""
    return 0.5 * heat_clock(t0, t1, sigma)


def stiefel_spin_factor_time(r):
    """The ONLY sanctioned way to obtain the S^3 factor time from r."""
    return 0.5 * r


def jump_clock(t0, t1, gamma):
    """Gamma = int gamma_t dt for constant gamma."""
    return gamma * (t1 - t0)


# ============================================================================
# Experiment A -- fixed-composition binary states  (bijective / swap)
# ============================================================================


def swap_state(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """x: (..., n) binary.  Move a particle from occupied i to empty j."""
    y = x.clone()
    y[..., i] = 0
    y[..., j] = 1
    return y


def sample_uniform_swap(x: torch.Tensor, generator=None):
    """Uniform over the k(n-k) legal ordered occupied->empty swaps of one state."""
    occ = torch.nonzero(x > 0.5, as_tuple=False).flatten()
    emp = torch.nonzero(x < 0.5, as_tuple=False).flatten()
    ii = occ[torch.randint(len(occ), (), generator=generator)]
    jj = emp[torch.randint(len(emp), (), generator=generator)]
    return int(ii), int(jj)


def assert_fixed_count(x, k):
    assert torch.all(x.sum(dim=-1) == k)


def enumerate_fixed_count(n, k):
    """All binary vectors of length n with exactly k ones.  Shape (C(n,k), n)."""
    states = []
    for occ in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=np.int8)
        x[list(occ)] = 1
        states.append(x)
    return np.stack(states)


def exact_boltzmann_prob(energies, tau):
    logw = -np.asarray(energies, dtype=np.float64) / tau
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def periodic_lattice_edges(L):
    """Nearest-neighbour edge list of an L x L periodic square lattice."""
    edges = []
    for a in range(L):
        for b in range(L):
            s = a * L + b
            edges.append((s, a * L + (b + 1) % L))
            edges.append((s, ((a + 1) % L) * L + b))
    return edges


def ising_energy_np(x, edges, J=1.0):
    """E = -J sum_<ij> s_i s_j with s = 2x - 1.  x: (..., n) 0/1."""
    s = 2.0 * np.asarray(x, dtype=np.float64) - 1.0
    tot = np.zeros(s.shape[:-1], dtype=np.float64)
    for i, j in edges:
        tot += s[..., i] * s[..., j]
    return -J * tot


# ---- 4.4  binary orbit kernel ----------------------------------------------


def binary_orbit_generator(n, k):
    """Birth/death generator of the overlap-distance chain j = k - |S cap S0|."""
    J = min(k, n - k)
    B = np.zeros((J + 1, J + 1), dtype=np.float64)
    for j in range(J + 1):
        if j + 1 <= J:
            B[j, j + 1] = ((k - j) * (n - k - j)) / (k * (n - k))
        if j - 1 >= 0:
            B[j, j - 1] = (j * j) / (k * (n - k))
    B[np.diag_indices_from(B)] = -B.sum(axis=1)
    return B


def binary_orbit_kernel(n, k, Gamma):
    """kappa[j] = p_base(S | S0) for any S at overlap-distance j from S0."""
    B = binary_orbit_generator(n, k)
    q = np.zeros(len(B), dtype=np.float64)
    q[0] = 1.0
    q = q @ expm(Gamma * B)

    kappa = np.empty_like(q)
    for j in range(len(q)):
        Nj = math.comb(k, j) * math.comb(n - k, j)
        kappa[j] = q[j] / Nj
    return kappa


# ---- 4.5  radial Johnson kernel (arbitrary shell mixtures) ------------------
#
# The single-swap reference above is the j = 1 member of a family indexed by
# Johnson shell.  For each j the shell averaging operator is
#
#     (K_j f)(S) = (1 / v_j) sum_{d(S,Y) = j} f(Y),      v_j = C(k,j) C(n-k,j),
#
# and the reference generator is any nonnegative mixture L = sum_j g_j (K_j - I).
# Every K_j lies in the Bose-Mesner algebra of the Johnson scheme, so the K_j
# commute with each other and with every coordinate permutation; the time
# ordered exponential therefore collapses and the whole family is exactly
# reducible to the (D+1)-state distance chain regardless of how large v_j is.


def shell_size(n, k, j):
    """``v_j = C(k,j) C(n-k,j)``, the number of states at Johnson distance j."""
    if j < 0 or j > min(k, n - k):
        return 0
    return math.comb(k, j) * math.comb(n - k, j)


def _comb0(n, r):
    return math.comb(n, r) if 0 <= r <= n else 0


def shell_orbit_matrix(n, k, j):
    """``B_j``: one uniform distance-``j`` jump seen through ``d(S0, .)``.

    With ``a = d(S0, Y)`` the coordinates split into ``S0 & Y`` (size ``k-a``),
    ``Y \\ S0`` (``a``), ``S0 \\ Y`` (``a``) and the outside block (``n-k-a``).
    A distance-``j`` move drops ``r`` occupied sites out of ``S0 & Y`` and
    ``j-r`` out of ``Y \\ S0``, then fills ``s`` sites of ``S0 \\ Y`` and
    ``j-s`` outside ones, so the new distance is ``b = a + r - s``.
    """
    D = min(k, n - k)
    if not (1 <= j <= D):
        raise ValueError(f"shell {j} outside 1..{D} for (n,k)=({n},{k})")
    vj = shell_size(n, k, j)
    B = np.zeros((D + 1, D + 1), dtype=np.float64)
    for a in range(D + 1):
        for r in range(j + 1):
            c_rem = _comb0(k - a, r) * _comb0(a, j - r)
            if c_rem == 0:
                continue
            for s in range(j + 1):
                b = a + r - s
                if 0 <= b <= D:
                    B[a, b] += c_rem * _comb0(a, s) * _comb0(n - k - a, j - s) / vj
    if not np.allclose(B.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError(f"shell orbit matrix j={j} is not row stochastic")
    return B


_SHELL_ORBIT_CACHE = {}


def shell_orbit_matrix_cached(n, k, j):
    """Memoized :func:`shell_orbit_matrix`; the result must not be mutated."""
    key = (int(n), int(k), int(j))
    B = _SHELL_ORBIT_CACHE.get(key)
    if B is None:
        B = shell_orbit_matrix(*key)
        B.flags.writeable = False
        _SHELL_ORBIT_CACHE[key] = B
    return B


def radial_orbit_kernel(n, k, clocks):
    """``(q, kappa)`` for ``L = sum_j gamma_j (K_j - I)`` over one interval.

    ``clocks`` maps shell ``j`` to the *integrated* clock
    ``Gamma_j = int gamma_j(u) du`` over the interval.  ``q[a]`` is the
    probability that the endpoint lies in orbit ``a`` and ``kappa[a] = q[a]/v_a``
    is the per-state reference kernel, which is what the terminal label ratios
    and the bridge weights consume.

    Passing ``{1: Gamma}`` reproduces :func:`binary_orbit_kernel` exactly;
    ``tests_radial.py`` asserts this.
    """
    D = min(k, n - k)
    G = np.zeros((D + 1, D + 1), dtype=np.float64)
    eye = np.eye(D + 1)
    for j, Gam in clocks.items():
        Gam = float(Gam)
        if Gam < 0.0:
            raise ValueError(f"negative integrated clock for shell {j}: {Gam}")
        if Gam == 0.0:
            continue
        G += Gam * (shell_orbit_matrix_cached(n, k, int(j)) - eye)

    q = np.zeros(D + 1, dtype=np.float64)
    q[0] = 1.0
    q = q @ expm(G)

    if q.min() < -1e-12:
        raise FloatingPointError(f"negative orbit probability {q.min():.3e}")
    q = np.maximum(q, 0.0)
    tot = q.sum()
    if not np.isfinite(tot) or tot <= 0.0:
        raise FloatingPointError("radial orbit law did not normalize")
    q = q / tot

    v = np.array([shell_size(n, k, a) for a in range(D + 1)], dtype=np.float64)
    return q, q / v


def binary_orbit_distance(x0, x):
    k = int(np.sum(x0))
    overlap = int(np.sum(np.asarray(x0) * np.asarray(x)))
    return k - overlap


def binary_full_generator(states):
    """Dense CTMC generator of the single-swap reference on an enumerated orbit.

    M[a, b] = rate(state_a -> state_b) = 1 / (k (n-k)) for each legal swap.
    Total escape rate is exactly 1 (i.e. gamma_t = 1).
    """
    S = np.asarray(states, dtype=np.int64)
    ns, n = S.shape
    k = int(S[0].sum())
    rate = 1.0 / (k * (n - k))

    index = {tuple(s): a for a, s in enumerate(S)}
    M = np.zeros((ns, ns), dtype=np.float64)
    for a, s in enumerate(S):
        occ = np.nonzero(s == 1)[0]
        emp = np.nonzero(s == 0)[0]
        for i in occ:
            for j in emp:
                t = s.copy()
                t[i] = 0
                t[j] = 1
                M[a, index[tuple(t)]] += rate
    M[np.diag_indices_from(M)] = -M.sum(axis=1)
    return M, index


def binary_as_label(x1, x1_g, x0, energy_fn, tau, kappa):
    """Lambda_g(X1) for the bijective (swap-permutation) construction."""
    e0 = energy_fn(x1)
    e1 = energy_fn(x1_g)

    j0 = binary_orbit_distance(x0, x1)
    j1 = binary_orbit_distance(x0, x1_g)

    log_energy_ratio = -(e1 - e0) / tau
    kernel_ratio = kappa[j0] / kappa[j1]
    return np.exp(log_energy_ratio) * kernel_ratio


# ============================================================================
# Experiment B -- occupation / inclusion process  (non-bijective)
# ============================================================================


def enumerate_occupation(m, N):
    """All eta in N_0^m with sum eta = N.  Shape (C(N+m-1, m-1), m)."""
    states = []
    for cuts in itertools.combinations(range(N + m - 1), m - 1):
        eta = []
        prev = -1
        for c in cuts:
            eta.append(c - prev - 1)
            prev = c
        eta.append(N + m - 1 - prev - 1)
        states.append(eta)
    return np.asarray(states, dtype=np.int64)


def occupation_index(states):
    return {tuple(s): a for a, s in enumerate(np.asarray(states))}


def occupation_E(states, index, a, b):
    """(E_ab f)(eta) = eta_b f(eta - e_b + e_a);  (E_aa f)(eta) = eta_a f(eta)."""
    S = np.asarray(states)
    ns = len(S)
    M = np.zeros((ns, ns), dtype=np.float64)
    if a == b:
        for r, eta in enumerate(S):
            M[r, r] = eta[a]
        return M
    for r, eta in enumerate(S):
        nb = int(eta[b])
        if nb == 0:
            continue
        t = eta.copy()
        t[b] -= 1
        t[a] += 1
        M[r, index[tuple(t)]] += nb
    return M


def occupation_F(states, index, j, i):
    """F_ji = E_ji - E_ii ;  (F_ji f)(eta) = eta_i [f(eta - e_i + e_j) - f(eta)]."""
    return occupation_E(states, index, j, i) - occupation_E(states, index, i, i)


def occupation_L(states, index, m):
    """Reference generator L = 1/(m-1) sum_{i != j} F_ji  (gamma = 1)."""
    S = np.asarray(states)
    ns = len(S)
    M = np.zeros((ns, ns), dtype=np.float64)
    for r, eta in enumerate(S):
        for i in range(m):
            ni = int(eta[i])
            if ni == 0:
                continue
            for j in range(m):
                if j == i:
                    continue
                t = eta.copy()
                t[i] -= 1
                t[j] += 1
                M[r, index[tuple(t)]] += ni / (m - 1)
                M[r, r] -= ni / (m - 1)
    return M


def occupation_S(states, index, m, j, i):
    """S_{ji} = sum_a (E_ja - E_ia)."""
    ns = len(states)
    out = np.zeros((ns, ns), dtype=np.float64)
    for a in range(m):
        out += occupation_E(states, index, j, a) - occupation_E(states, index, i, a)
    return out


def occupation_c_t(m, Gamma):
    """c_t = (1 - exp(-m/(m-1) Gamma_{t,1})) / m."""
    return (1.0 - np.exp(-m * Gamma / (m - 1.0))) / m


def occupation_rho(m, Gamma):
    return float(np.exp(-m * Gamma / (m - 1.0)))


def occupation_A(states, index, m, j, i, Gamma):
    """A^{(t)}_{ji} = F_ji - c_t sum_a (E_ja - E_ia)."""
    c = occupation_c_t(m, Gamma)
    return occupation_F(states, index, j, i) - c * occupation_S(states, index, m, j, i)


def sample_inclusion_exact(batch, m, N, d, device="cpu", generator=None):
    """Exact iid samples from the Dirichlet-multinomial inclusion target."""
    alpha = torch.full((m,), float(d), device=device, dtype=torch.float64)
    p = torch.distributions.Dirichlet(alpha).sample((batch,))
    eta = torch.distributions.Multinomial(total_count=N, probs=p).sample()
    return eta.to(torch.long)


def sample_occupation_reference_jump(eta, generator=None):
    m = eta.numel()
    probs = eta.double() / eta.sum()
    i = torch.multinomial(probs, 1, generator=generator).item()
    r = torch.randint(m - 1, (), generator=generator).item()
    j = r if r < i else r + 1
    out = eta.clone()
    out[i] -= 1
    out[j] += 1
    return out, i, j


def occupation_single_particle_probs(m, Gamma, source_mode=0):
    rho = np.exp(-m * Gamma / (m - 1))
    omega = (1.0 - rho) / m
    q = np.full(m, omega, dtype=np.float64)
    q[source_mode] += rho
    return q


def occupation_neighbor_kernel_ratio(xi, a, b, q):
    """ratio p(xi) / p(xi - e_b + e_a).   Requires a != b and xi[b] > 0."""
    assert a != b
    assert xi[b] > 0
    return ((xi[a] + 1.0) / xi[b]) * (q[b] / q[a])


def inclusion_energy(eta, tau, d):
    x = torch.as_tensor(eta, dtype=torch.float64)
    logw = torch.lgamma(x + d) - torch.lgamma(x + 1.0) - torch.lgamma(
        torch.as_tensor(float(d), dtype=x.dtype)
    )
    return -tau * logw.sum(dim=-1)


def transfer_state(xi, a, b):
    """source a -> destination b."""
    assert a != b
    assert xi[a] > 0
    out = xi.clone()
    out[a] -= 1
    out[b] += 1
    return out


def occupation_f_ratio(xi, a, b, q, energy_fn, tau):
    """R_{ba}(xi) = f1(xi - e_a + e_b) / f1(xi)."""
    if a == b:
        return 1.0
    if int(xi[a]) <= 0:
        raise ValueError("illegal transfer")

    xi2 = transfer_state(xi, a, b)
    kr = occupation_neighbor_kernel_ratio(np.asarray(xi.cpu()), b, a, q)
    dE = float(energy_fn(xi2) - energy_fn(xi))
    return float(np.exp(-dE / tau) * kr)


def occupation_terminal_label_full(xi, i, j, q, energy_fn, tau, c_t):
    assert i != j
    if int(xi[i]) > 0:
        Rji = occupation_f_ratio(xi, i, j, q, energy_fn, tau)
        first = float(xi[i]) * (Rji - 1.0)
    else:
        first = 0.0

    corr = 0.0
    for a in range(len(xi)):
        na = int(xi[a])
        if na == 0:
            continue
        Rja = 1.0 if a == j else occupation_f_ratio(xi, a, j, q, energy_fn, tau)
        Ria = 1.0 if a == i else occupation_f_ratio(xi, a, i, q, energy_fn, tau)
        corr += na * (Rja - Ria)
    return first - c_t * corr


def occupation_terminal_label_uniform_one_sample(
    xi, i, j, q, energy_fn, tau, rho_t1, rng=np.random
):
    if int(xi[i]) > 0:
        Rji = occupation_f_ratio(xi, i, j, q, energy_fn, tau)
        first = float(xi[i]) * (Rji - 1.0)
    else:
        first = 0.0

    m = len(xi)
    a = int(rng.randint(m))
    na = int(xi[a])
    if na == 0:
        correction = 0.0
    else:
        Rja = 1.0 if a == j else occupation_f_ratio(xi, a, j, q, energy_fn, tau)
        Ria = 1.0 if a == i else occupation_f_ratio(xi, a, i, q, energy_fn, tau)
        correction = (1.0 - rho_t1) * na * (Rja - Ria)
    return first - correction


def occupation_terminal_label_occupancy_sample(xi, i, j, q, energy_fn, tau, c_t):
    if int(xi[i]) > 0:
        Rji = occupation_f_ratio(xi, i, j, q, energy_fn, tau)
        first = float(xi[i]) * (Rji - 1.0)
    else:
        first = 0.0

    probs = xi.double() / xi.sum()
    a = int(torch.multinomial(probs, 1).item())
    Rja = 1.0 if a == j else occupation_f_ratio(xi, a, j, q, energy_fn, tau)
    Ria = 1.0 if a == i else occupation_f_ratio(xi, a, i, q, energy_fn, tau)

    correction = c_t * float(xi.sum()) * (Rja - Ria)
    return first - correction


# ============================================================================
# Experiment C -- sphere S^{d-1}
# ============================================================================


def sphere_project_tangent(x, v):
    return v - x * (x * v).sum(dim=-1, keepdim=True)


def sphere_exp(x, v, eps=1e-12):
    nv = torch.linalg.norm(v, dim=-1, keepdim=True)
    direction = v / torch.clamp(nv, min=eps)
    out = torch.cos(nv) * x + torch.sin(nv) * direction

    small = nv < 1e-7
    approx = x + v
    approx = approx / torch.linalg.norm(approx, dim=-1, keepdim=True)
    return torch.where(small, approx, out)


def sphere_terminal_readout(x, y, G):
    """Collapsed Killing readout:  (x.y) G - y (x.G).  Tangent at x."""
    xy = (x * y).sum(dim=-1, keepdim=True)
    xG = (x * G).sum(dim=-1, keepdim=True)
    return xy * G - y * xG


def killing_omega(n, i, j, dtype=torch.float64, device="cpu"):
    O = torch.zeros((n, n), dtype=dtype, device=device)
    O[i, j] = 1.0
    O[j, i] = -1.0
    return O


def sphere_readout_explicit(x, y, G):
    """sum_{i<j} V_ij(x) <V_ij(y), G>  with V_ij(z) = Omega_ij z.  Single sample."""
    d = x.shape[-1]
    out = torch.zeros_like(x)
    for i in range(d):
        for j in range(i + 1, d):
            O = killing_omega(d, i, j, dtype=x.dtype, device=x.device)
            Vx = O @ x
            Vy = O @ y
            out = out + Vx * torch.dot(Vy, G)
    return out


def sphere_killing_gram(x, y):
    """sum_{i<j} V_ij(x) V_ij(y)^T  == (x.y) I - y x^T."""
    d = x.shape[-1]
    out = torch.zeros((d, d), dtype=x.dtype, device=x.device)
    for i in range(d):
        for j in range(i + 1, d):
            O = killing_omega(d, i, j, dtype=x.dtype, device=x.device)
            out = out + torch.outer(O @ x, O @ y)
    return out


def sphere_energy(x):
    """R-ASBS analytic benchmark: E(x) = 6 (1 - x_3^2)."""
    return 6.0 * (1.0 - x[..., 2] ** 2)


def sphere_energy_grad(x):
    g = torch.zeros_like(x)
    g[..., 2] = -12.0 * x[..., 2]
    return g


# ---- 6.6  S^2 heat kernel ---------------------------------------------------


def s2_heat_kernel_and_dc(c, r, Lmax=200):
    """Spectral S^2 heat kernel p_r(c) and its derivative in c = cos(theta)."""
    c = np.asarray(c, dtype=np.float64)
    p = np.zeros_like(c)
    dp = np.zeros_like(c)

    for ell in range(Lmax + 1):
        coeff = (2 * ell + 1) * np.exp(-ell * (ell + 1) * r)
        P = eval_legendre(ell, c)
        p = p + coeff * P
        if ell >= 1:
            Pm1 = eval_legendre(ell - 1, c)
            den = np.maximum(1.0 - c * c, 1e-14)
            dP = ell * (Pm1 - c * P) / den
            dp = dp + coeff * dP

    p = p / (4.0 * np.pi)
    dp = dp / (4.0 * np.pi)
    return p, dp


def s2_log_kernel(y, x0, r, Lmax=200):
    c = float(np.dot(np.asarray(x0, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    c = min(max(c, -1.0), 1.0)
    p, _ = s2_heat_kernel_and_dc(np.array(c), r, Lmax=Lmax)
    return float(np.log(p))


def s2_reference_score(y, x0, r, Lmax=200):
    """grad_y log p_r(x0 -> y), tangent at y.  numpy, shapes (3,)."""
    x0 = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c = float(np.dot(x0, y))
    c = min(max(c, -1.0), 1.0)
    p, dp = s2_heat_kernel_and_dc(np.array(c), r, Lmax=Lmax)
    coeff = float(dp / p)
    tangent = x0 - c * y
    return coeff * tangent


def exact_s2_z_cdf_grid(num=20001):
    from scipy.integrate import cumulative_trapezoid

    z = np.linspace(-1.0, 1.0, num)
    w = np.exp(6.0 * z * z)
    cdf = np.concatenate([[0.0], cumulative_trapezoid(w, z)])
    cdf /= cdf[-1]
    return z, cdf


# ============================================================================
# Experiment D -- Stiefel St(n, p)
# ============================================================================


def stiefel_constraint_error(X):
    p = X.shape[-1]
    I = torch.eye(p, dtype=X.dtype, device=X.device)
    gram = X.transpose(-1, -2) @ X
    return torch.linalg.norm(gram - I, dim=(-2, -1))


def stiefel_terminal_readout(X, Y, G):
    """Collapsed canonical Killing readout:  G (Y^T X) - Y (G^T X)."""
    return G @ (Y.transpose(-1, -2) @ X) - Y @ (G.transpose(-1, -2) @ X)


def stiefel_canonical_inner(X, A, B):
    """Canonical metric  g_c(A, B) = tr( A^T (I - 1/2 X X^T) B )  at X."""
    n = X.shape[-2]
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    M = I - 0.5 * (X @ X.transpose(-1, -2))
    return torch.einsum("...np,...nm,...mp->...", A, M, B)


def stiefel_random_tangent(X, generator=None):
    """A random canonical-metric tangent vector at X:  T = X A + X_perp B,
    A skew (p x p), B arbitrary ((n-p) x p).  Satisfies X^T T + T^T X = 0."""
    G = torch.randn(X.shape, dtype=X.dtype, device=X.device, generator=generator)
    S = X.transpose(-1, -2) @ G
    skew = 0.5 * (S - S.transpose(-1, -2))
    n = X.shape[-2]
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    perp = (I - X @ X.transpose(-1, -2)) @ G
    return X @ skew + perp


def stiefel_readout_explicit(X, Y, G):
    """sum_{i<j} Omega_ij X <Omega_ij Y, G>.   Single sample, shape (n, p)."""
    n = X.shape[0]
    out = torch.zeros_like(X)
    for i in range(n):
        for j in range(i + 1, n):
            O = killing_omega(n, i, j, dtype=X.dtype, device=X.device)
            VX = O @ X
            VY = O @ Y
            out = out + VX * torch.sum(VY * G)
    return out


def stiefel_readout_collapsed(X, Y, G):
    return G @ (Y.T @ X) - Y @ (G.T @ X)


def stiefel_killing_casimir(n, dtype=torch.float64):
    """sum_{i<j} Omega_ij^2 .   Equals -(n-1) I."""
    out = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(i + 1, n):
            O = killing_omega(n, i, j, dtype=dtype)
            out = out + O @ O
    return out


def expected_st42_mean_factor(r):
    """E[X_r | X_0] = exp(-3 r) X_0 for St(4, 2).   NOT exp(-6 r)."""
    return float(np.exp(-3.0 * r))


def random_stiefel(batch, n, p, dtype=torch.float64, device="cpu", generator=None):
    A = torch.randn(batch, n, p, dtype=dtype, device=device, generator=generator)
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q * sign.unsqueeze(-2)


def so_brownian_increment(batch, n, dt, sigma, dtype=torch.float64, device="cpu",
                          generator=None):
    """exp( sigma sqrt(dt) sum_{i<j} xi_ij Omega_ij ) -- geodesic random walk step."""
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    xi = torch.randn(batch, len(idx), dtype=dtype, device=device, generator=generator)
    A = torch.zeros(batch, n, n, dtype=dtype, device=device)
    for k, (i, j) in enumerate(idx):
        A[:, i, j] += xi[:, k]
        A[:, j, i] -= xi[:, k]
    return torch.matrix_exp(sigma * math.sqrt(dt) * A)


def simulate_so_brownian(batch, n, r, steps, dtype=torch.float64, device="cpu",
                         generator=None):
    """Brownian motion on SO(n) with generator (sigma^2/2) sum_{i<j} V_ij^2,
    run until heat time r = 1/2 int sigma^2 dt.  Returns R in SO(n)."""
    sigma = math.sqrt(2.0 * r)  # unit total time, r = sigma^2 / 2
    dt = 1.0 / steps
    R = torch.eye(n, dtype=dtype, device=device).expand(batch, n, n).contiguous()
    for _ in range(steps):
        R = so_brownian_increment(batch, n, dt, sigma, dtype=dtype, device=device,
                                  generator=generator) @ R
    return R


def so4_spin_angles(R):
    """Unordered pair (theta_p, theta_q) of S^3 geodesic radii of the spin lift.

    A rotation of SO(4) with invariant-plane angles (alpha, beta) lifts to the
    quaternion pair exp(i (alpha+beta)/2), exp(i (alpha-beta)/2), so the two
    S^3 geodesic radii are |alpha+beta|/2 and |alpha-beta|/2.
    """
    Rn = np.asarray(R.detach().cpu().numpy(), dtype=np.float64)
    ev = np.linalg.eigvals(Rn)
    ang = np.abs(np.angle(ev))            # (..., 4) in [0, pi]
    ang = np.sort(ang, axis=-1)
    # conjugate pairs -> take entries 0 and 2 after sorting (duplicates)
    alpha = ang[..., 0]
    beta = ang[..., 2]
    tp = 0.5 * (beta + alpha)
    tq = 0.5 * (beta - alpha)
    return tp, tq


# ---- 7.7  S^3 heat kernel ---------------------------------------------------


def s3_heat_kernel(theta, s, K=8):
    """p_s(theta) = e^s (4 pi s)^{-3/2} (1/sin theta) sum_k (theta + 2 pi k)
    exp[-(theta + 2 pi k)^2 / (4 s)].   theta a torch tensor in (0, pi)."""
    theta = torch.as_tensor(theta, dtype=torch.float64)
    ks = torch.arange(-K, K + 1, device=theta.device, dtype=theta.dtype)
    z = theta[..., None] + 2.0 * math.pi * ks
    terms = z * torch.exp(-(z * z) / (4.0 * s))
    series = terms.sum(dim=-1)

    denom = torch.sin(theta)
    denom = torch.where(denom.abs() < 1e-12,
                        torch.full_like(denom, 1e-12), denom)
    pref = math.exp(s) / ((4.0 * math.pi * s) ** 1.5)
    return pref * series / denom


def s3_radial_density(theta, s, K=8):
    """Density of the geodesic radius theta:  p_s(theta) * 4 pi sin^2(theta)."""
    theta = torch.as_tensor(theta, dtype=torch.float64)
    return s3_heat_kernel(theta, s, K=K) * 4.0 * math.pi * torch.sin(theta) ** 2


def s3_radial_cdf_grid(s, num=40001, K=8):
    from scipy.integrate import cumulative_trapezoid

    th = np.linspace(1e-9, math.pi - 1e-9, num)
    dens = s3_radial_density(torch.tensor(th), s, K=K).numpy()
    cdf = np.concatenate([[0.0], cumulative_trapezoid(dens, th)])
    total = cdf[-1]
    return th, cdf / total, total


def stiefel_st42_kernel_factors(theta_p, theta_q, r, K=8):
    """The ONLY sanctioned St(4,2) spin-factor evaluation: both at time r/2."""
    rs = stiefel_spin_factor_time(r)
    kp = s3_heat_kernel(theta_p, rs, K=K)
    kq = s3_heat_kernel(theta_q, rs, K=K)
    return kp, kq


def periodic_legendre_nodes(nq=64, device="cpu", dtype=torch.float64):
    from numpy.polynomial.legendre import leggauss

    z, w = leggauss(nq)
    phi = np.pi * (z + 1.0)
    weight = np.pi * w
    return (torch.tensor(phi, device=device, dtype=dtype),
            torch.tensor(weight, device=device, dtype=dtype))


def stiefel_frame_energy(X, H, C, lam=1.0):
    quad = torch.einsum("...np,nm,...mp->...", X, H, X)
    linear = torch.einsum("np,...np->...", C, X)
    return quad - lam * linear


# ============================================================================
# Misc statistics helpers
# ============================================================================


def ks_against_grid_cdf(samples, grid_x, grid_cdf):
    """One-sample Kolmogorov-Smirnov statistic against a tabulated CDF."""
    s = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(s)
    F = np.interp(s, grid_x, grid_cdf)
    d_plus = np.max(np.arange(1, n + 1) / n - F)
    d_minus = np.max(F - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def total_variation(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# ============================================================================
# checkpointing -- everything needed to regenerate a figure without retraining
# ============================================================================


def save_ckpt(ckpt_dir, tag, net=None, samples=None, extra=None, nets=None):
    """Write <ckpt_dir>/<tag>.pt with weights, final samples and metadata.

    Metric histories live in the results_*.json files; this is for the two
    things those cannot hold -- the learned control itself (for score-field
    plots) and the raw terminal samples (for histograms and bootstrap CIs).

    nets is an optional {name: module} map for runs that train more than one
    network.  A non-Dirac run learns a control AND a corrector, and the
    corrector is not reconstructible from the control: dropping it would make
    the checkpoint unable to reproduce the terminal law it reports.
    """
    import os

    if not ckpt_dir:
        return None
    os.makedirs(ckpt_dir, exist_ok=True)
    blob = {"tag": tag}
    if net is not None:
        # The network really is a float32 module; storing it as anything else
        # would be a lie about what was trained.
        blob["state_dict"] = {k: v.detach().cpu()
                              for k, v in net.state_dict().items()}
    if samples is not None:
        if torch.is_tensor(samples):
            x = samples.detach().cpu()
        else:
            x = torch.as_tensor(np.asarray(samples))
        # Manifold samples are stored in float64.  Our constraint residuals are
        # 2e-16 on S^2 and 3.8e-14 on St(4,2); float32 storage has an epsilon of
        # 1.2e-07, so downcasting would destroy the very result the checkpoint
        # exists to evidence -- and would leave a number that coincidentally
        # resembles R-ASBS's 3.4e-07 retraction error.  Integer state indices
        # are left alone.
        if x.is_floating_point():
            x = x.to(torch.float64)
        blob["samples"] = x
    if nets:
        blob["state_dicts"] = {
            name: {k: v.detach().cpu() for k, v in m.state_dict().items()}
            for name, m in nets.items()}
    if extra:
        blob["extra"] = extra
    path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save(blob, path)
    return path
