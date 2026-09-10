"""IASBS on a real DFT-fitted CuAu alloy energy at fixed equiatomic composition.

This module adds a materials benchmark to the fixed-composition IASBS
construction that `iasbs/fixed_ising.py` already validates on enumerable state
spaces.  **The IASBS mathematics is unchanged.**  Only two things differ:

1.  *Storage.*  States are direct ``(B, N)`` binary tensors instead of indices
    into an enumerated space.  ``C(64,32) = 1.83e18`` states cannot be
    enumerated, so there is no lookup table, no ``M``, and no ``tgt`` table.
2.  *Terminal labels.*  The controller loss is an average over the legal edges
    of the intermediate state, so one uniformly sampled legal edge per training
    example is an unbiased estimator.  This replaces ``C(N,2)`` target-energy
    evaluations per example with one.

Everything else -- ``SwapController``, the Johnson orbit kernel
``common.binary_orbit_kernel``, the exact occupancy-class reference bridge, and
the Poisson/Bregman objective -- is reused verbatim.

Energy oracle
-------------
The target is the canonical Boltzmann law of a published cluster expansion (CE)
fitted to DFT by the MetaDNS authors (ICML 2026).  Provenance, hashes and the
pinned upstream commit are in ``data/cuau/PROVENANCE.md``.

The CE is a *multilinear polynomial* in the spin variables ``s_i = 2 x_i - 1``:
a constant, a one-body term, and one symmetry orbit per ECI.  This module
extracts the orbit site-index tables from CLEASE (exact structure) and then
solves a small linear system against CLEASE energies for the effective orbit
coefficients (exact normalisation, no reverse engineering of CLEASE
conventions).  The resulting evaluator agrees with CLEASE to ~1e-15 eV and runs
batched on the GPU, which is what makes the training and reference budgets
affordable.

Run everything in the dedicated ``cuau_env`` environment; see
``requirements-cuau.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.special import gammaln

# common.py and the shared json/ ckpt/ directories live at the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402

KB_EV_PER_K = 8.617333262145e-5  # eV/K, CODATA; identical to ase.units.kB

DATA = "data/cuau"
ECI_JSON = f"{DATA}/CI_params_ECI_CuAu_Final_Submission.json"
VASP = {
    (2, 2, 4): f"{DATA}/cuau_fcc_2x2x4_supercell.vasp",
    (4, 4, 4): f"{DATA}/cuau_fcc_4x4x4_supercell.vasp",
}
METADNS = "external/metadns"

# ----------------------------------------------------------------------------
# provenance helpers
# ----------------------------------------------------------------------------


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def provenance() -> dict:
    """Hashes/commit of every external input, embedded in each result JSON."""
    out = {"kB_eV_per_K": KB_EV_PER_K, "files": {}}
    for p in [ECI_JSON] + list(VASP.values()):
        if os.path.exists(p):
            out["files"][p] = sha256(p)
    cm = f"{DATA}/METADNS_COMMIT.txt"
    if os.path.exists(cm):
        out["metadns_commit"] = open(cm).read().strip()
    out["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                       "numpy": np.__version__}
    return out


# ----------------------------------------------------------------------------
# CLEASE-backed reference energy (the parity authority)
# ----------------------------------------------------------------------------


class CleaseCuAuEnergy:
    """Ground-truth CE energy, evaluated one configuration at a time.

    Total configurational energy of the whole supercell in **eV**.  This is the
    raw CLEASE/icet cluster-expansion energy: no ``k_B T`` rescaling (upstream's
    ``forward`` does divide by ``k_B T``; we never use that path) and no
    formation-energy reference subtraction.  At fixed equiatomic composition a
    formation reference or a chemical-potential term is a constant across the
    whole canonical sector and cancels from every Boltzmann ratio, so the choice
    only shifts the zero of energy.

    Species mapping is explicit and matches upstream ``(x * 50) + 29``:
    ``x=1 -> Au (Z=79)``, ``x=0 -> Cu (Z=29)``.
    """

    def __init__(self, size=(4, 4, 4), vasp=None, eci_json=ECI_JSON, a=3.8,
                 max_cluster_dia=(6.0, 4.5, 4.5), scratch="external/probe"):
        import ase.io                                   # noqa: PLC0415
        from clease.calculator import attach_calculator  # noqa: PLC0415
        from clease.settings import CEBulk, Concentration  # noqa: PLC0415

        self.size = tuple(int(s) for s in size)
        vasp = vasp or VASP.get(self.size)
        self.eci = json.load(open(eci_json))
        cwd = os.getcwd()
        os.makedirs(scratch, exist_ok=True)
        try:
            os.chdir(scratch)                            # CEBulk writes a .db
            conc = Concentration(basis_elements=[["Au", "Cu"]])
            self.settings = CEBulk(crystalstructure="fcc", a=a, size=list(self.size),
                                   concentration=conc, db_name="aucu_dft.db",
                                   max_cluster_dia=list(max_cluster_dia))
        finally:
            os.chdir(cwd)
        if vasp is not None and os.path.exists(vasp):
            self.atoms_ref = ase.io.read(vasp)
            self.source = "vasp"
        else:
            # CuAu-L (4x4x8) is not shipped upstream; build it from the same
            # primitive cell and the same CE settings instead of inventing one.
            self.atoms_ref = self.settings.atoms.copy()
            self.source = "settings"
        self.N = len(self.atoms_ref)
        self.atoms = attach_calculator(self.settings, atoms=self.atoms_ref.copy(),
                                       eci=self.eci)
        self.positions = np.asarray(self.atoms.get_positions(), dtype=np.float64)
        self.cell = np.asarray(self.atoms.get_cell(), dtype=np.float64)
        self.a = float((4.0 * self.atoms.get_volume() / self.N) ** (1.0 / 3.0))
        self.n_calls = 0

    def energy_one(self, x) -> float:
        self.atoms.numbers = (np.asarray(x).astype(np.int64) * 50 + 29)
        self.n_calls += 1
        return float(self.atoms.get_potential_energy())

    def energy_np(self, x) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            return np.float64(self.energy_one(x))
        flat = x.reshape(-1, x.shape[-1])
        out = np.array([self.energy_one(r) for r in flat], dtype=np.float64)
        return out.reshape(x.shape[:-1])


# ----------------------------------------------------------------------------
# exact multilinear orbit expansion of the same CE
# ----------------------------------------------------------------------------


def build_ce_tables(size=(4, 4, 4), n_calib=400, seed=0, cache=True, verbose=True):
    """Return the exact multilinear spin representation of the CE.

    ``E(x) = c[0] + c[1] * sum_i s_i
             + sum_p c[2+p] * sum_{cluster in orbit p} mult * prod s``

    with ``s = 2x - 1``.  Cluster site tuples come from CLEASE's cluster list and
    translation matrix, so the *structure* is exact by construction.  The 2 + P
    coefficients are then obtained by an exactly determined least-squares solve
    against CLEASE energies on random (unconstrained) configurations, which
    sidesteps every CLEASE normalisation convention.  The solve residual is
    reported and gated: it is ~1e-15 eV when the structure is right and O(0.1 eV)
    when it is not, so this doubles as a strong structural test.

    Multiplicities matter.  In a small supercell the same *pair of sites* can be
    joined by several inequivalent lattice translations, and a cluster figure can
    even map two entries onto the same site.  CLEASE sums over all ``(reference
    site, figure)`` pairs, so this function accumulates a multiplicity per unique
    site tuple rather than deduplicating; collapsing them silently breaks the
    2x2x4 cell (residual 5e-2 eV) while leaving 4x4x4 exact.
    """
    size = tuple(int(s) for s in size)
    path = f"{DATA}/ce_tables_{size[0]}x{size[1]}x{size[2]}.npz"
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        n_orb = int(z["n_orbits"])
        tables = {
            "size": size,
            "N": int(z["N"]),
            "coef": z["coef"],
            "orbits": [z[f"orb{p}"] for p in range(n_orb)],
            "mults": [z[f"mul{p}"] for p in range(n_orb)],
            "names": [str(s) for s in z["names"]],
            "positions": z["positions"],
            "cell": z["cell"],
            "a": float(z["a"]),
            "fit_max_err": float(z["fit_max_err"]),
            "heldout_max_err": float(z["heldout_max_err"]),
        }
        return tables

    from collections import Counter                     # noqa: PLC0415

    ref = CleaseCuAuEnergy(size=size)
    tm = ref.settings.trans_matrix
    N = ref.N
    names, orbits, mults = [], [], []
    for cl in ref.settings.cluster_list:
        if cl.size < 2:
            continue
        cnt = Counter()
        for r in range(N):
            for fig in cl.indices:
                # ``fig`` already lists every site of the figure relative to the
                # reference site, so translating each entry gives the cluster.
                cnt[tuple(sorted(tm[r][i] for i in fig))] += 1
        keys = sorted(cnt)
        arr = np.array(keys, dtype=np.int64)
        mul = np.array([cnt[k] for k in keys], dtype=np.float64)
        names.append(cl.name)
        orbits.append(arr)
        mults.append(mul)
        if verbose:
            um = sorted(set(mul.astype(int)))
            print(f"  orbit {cl.name:<16s} size {cl.size}  unique {len(arr):5d}  "
                  f"total {int(mul.sum()):6d}  mult {um}")

    def feats(x):
        s = 2.0 * np.asarray(x, dtype=np.float64) - 1.0
        f = [1.0, s.sum()]
        for arr, mul in zip(orbits, mults):
            f.append(float((mul * np.prod(s[arr], axis=1)).sum()))
        return np.array(f, dtype=np.float64)

    rng = np.random.default_rng(seed)
    # Unconstrained random configurations: an equiatomic-only design matrix is
    # rank deficient because sum_i s_i is then identically zero.
    Xc = np.array([rng.integers(0, 2, N) for _ in range(n_calib)])
    A = np.stack([feats(x) for x in Xc])
    y = ref.energy_np(Xc)
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    fit_err = float(np.abs(A @ coef - y).max())
    if rank != A.shape[1]:
        raise RuntimeError(f"CE design matrix rank {rank} < {A.shape[1]}: "
                           "orbit construction is degenerate")

    # held-out check on the equiatomic sector we actually sample
    Xh = np.zeros((300, N), dtype=np.int64)
    for r in range(300):
        Xh[r, rng.permutation(N)[: N // 2]] = 1
    held = float(np.abs(np.stack([feats(x) for x in Xh]) @ coef - ref.energy_np(Xh)).max())
    if verbose:
        print(f"  lstsq rank {rank}/{A.shape[1]}  fit max err {fit_err:.3e} eV  "
              f"heldout equiatomic max err {held:.3e} eV")
    if max(fit_err, held) > 1e-10:
        raise RuntimeError("CE multilinear reconstruction failed the 1e-10 eV gate")

    tables = {"size": size, "N": N, "coef": coef, "orbits": orbits, "mults": mults,
              "names": names, "positions": ref.positions, "cell": ref.cell, "a": ref.a,
              "fit_max_err": fit_err, "heldout_max_err": held}
    if cache:
        extra = {f"orb{p}": o for p, o in enumerate(orbits)}
        extra.update({f"mul{p}": m for p, m in enumerate(mults)})
        np.savez(path, N=N, coef=coef, names=np.array(names), positions=ref.positions,
                 cell=ref.cell, a=ref.a, n_orbits=len(orbits), fit_max_err=fit_err,
                 heldout_max_err=held, **extra)
        if verbose:
            print(f"  cached -> {path}")
    return tables


class TorchCuAuEnergy:
    """Batched GPU evaluator for the same CE, total eV, float64.

    Exact to ~1e-15 eV against :class:`CleaseCuAuEnergy`; see the ``validate``
    subcommand, which is a hard gate.
    """

    def __init__(self, tables, device="cuda", dtype=torch.float64):
        self.N = int(tables["N"])
        self.size = tuple(tables["size"])
        self.device = torch.device(device)
        self.dtype = dtype
        c = np.asarray(tables["coef"], dtype=np.float64)
        self.c0 = float(c[0])
        self.c1 = float(c[1])
        self.orb = [torch.as_tensor(o, dtype=torch.long, device=self.device)
                    for o in tables["orbits"]]
        self.mul = [torch.as_tensor(np.asarray(m, dtype=np.float64), dtype=dtype,
                                    device=self.device) for m in tables["mults"]]
        self.w = [float(v) for v in c[2:]]
        self.names = list(tables["names"])
        self.positions = np.asarray(tables["positions"], dtype=np.float64)
        self.a = float(tables["a"])
        self.n_calls = 0

    @torch.no_grad()
    def energy_torch(self, x: torch.Tensor) -> torch.Tensor:
        s = (2.0 * x.to(self.dtype) - 1.0)
        self.n_calls += int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
        out = self.c0 + self.c1 * s.sum(-1)
        for arr, mul, w in zip(self.orb, self.mul, self.w):
            # s[..., arr] -> (..., n_clusters, cluster_size)
            out = out + w * (mul * s[..., arr].prod(-1)).sum(-1)
        return out

    def energy_np(self, x) -> np.ndarray:
        t = torch.as_tensor(np.asarray(x), device=self.device)
        return self.energy_torch(t).cpu().numpy()


class CountedEnergy:
    """Wrap an energy oracle and count target evaluations (configs, not batches).

    Train-time and eval-time counts are kept separate so matched-budget curves
    against MCMC/PT are honest.
    """

    def __init__(self, backend):
        self.backend = backend
        self.n_configs = 0
        self.n_calls = 0
        self.seconds = 0.0
        self.tag = "train"
        self.by_tag = {}

    def _bump(self, n, dt):
        self.n_configs += n
        self.n_calls += 1
        self.seconds += dt
        d = self.by_tag.setdefault(self.tag, {"configs": 0, "seconds": 0.0})
        d["configs"] += n
        d["seconds"] += dt

    def energy_torch(self, x):
        t0 = time.time()
        out = self.backend.energy_torch(x)
        self._bump(int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1, time.time() - t0)
        return out

    def energy_np(self, x):
        x = np.asarray(x)
        t0 = time.time()
        out = self.backend.energy_np(x)
        self._bump(int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1, time.time() - t0)
        return out

    def report(self):
        return {"configs": self.n_configs, "calls": self.n_calls,
                "seconds": round(self.seconds, 3),
                "by_tag": {k: {"configs": v["configs"], "seconds": round(v["seconds"], 3)}
                           for k, v in self.by_tag.items()}}

    def __getattr__(self, k):
        return getattr(self.backend, k)


# ----------------------------------------------------------------------------
# L10 order parameters
# ----------------------------------------------------------------------------


def l10_q_vectors(a: float) -> np.ndarray:
    """The three X-point ordering wavevectors of an FCC L10 structure."""
    return (2.0 * np.pi / a) * np.eye(3, dtype=np.float64)


def l10_order_parameters(x, positions, q) -> np.ndarray:
    """``(..., 3)`` absolute structure factors ``Q_alpha`` in ``[0, 1]``."""
    x = np.asarray(x)
    r = np.asarray(positions, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if x.shape[-1] != r.shape[0] or q.shape != (3, 3):
        raise ValueError("state/position/q shape mismatch")
    s = 2.0 * x.astype(np.float64) - 1.0
    phase = np.exp(1j * (r @ q.T))
    z = np.einsum("...n,nq->...q", s, phase) / x.shape[-1]
    return np.abs(z)


def l10_order_parameters_torch(x: torch.Tensor, cosp: torch.Tensor,
                               sinp: torch.Tensor) -> torch.Tensor:
    """Same quantity on GPU.  ``cosp``/``sinp`` are ``(N, 3)`` phase tables."""
    s = 2.0 * x.to(cosp.dtype) - 1.0
    re = s @ cosp
    im = s @ sinp
    return torch.sqrt(re * re + im * im) / x.shape[-1]


def ideal_l10_state(positions, q_vector) -> np.ndarray:
    """One ideal L10 variant; the binary complement is the equivalent partner."""
    c = np.cos(np.asarray(positions, dtype=np.float64) @ np.asarray(q_vector))
    x = (c < 0.0).astype(np.int64)
    if x.sum() * 2 != x.size:
        raise ValueError("q/positions do not generate an equiatomic L10 pattern")
    return x


# ----------------------------------------------------------------------------
# fixed-composition state utilities (direct tensors, never enumerated)
# ----------------------------------------------------------------------------


def transpose_batch(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Swap sites ``i`` and ``j`` of every row.  Works for any occupancy pattern."""
    y = x.clone()
    b = torch.arange(x.shape[0], device=x.device)
    xi = x[b, i].clone()
    xj = x[b, j].clone()
    y[b, i] = xj
    y[b, j] = xi
    return y


def legal_mask(x: torch.Tensor) -> torch.Tensor:
    """``(B, N, N)`` mask of ordered legal swaps: site ``i`` is Au, site ``j`` is Cu."""
    au = x.bool()
    return (au.unsqueeze(2) & (~au).unsqueeze(1))


def all_legal_edges(x: torch.Tensor):
    """``(B, k(N-k))`` ordered legal edges of every row, row-major order."""
    B, n = x.shape
    k = int(x[0].sum())
    e = torch.nonzero(legal_mask(x).reshape(B, -1), as_tuple=False)[:, 1]
    e = e.reshape(B, k * (n - k))
    return e // n, e % n


def sample_uniform_legal_edge(x: torch.Tensor, generator=None):
    """Uniform draw from ``E(x)``, the ``k(N-k)`` ordered legal swaps of ``x``."""
    B, n = x.shape
    m = legal_mask(x).reshape(B, -1).to(torch.float32)
    e = torch.multinomial(m, 1, generator=generator).squeeze(1)
    return e // n, e % n


def random_fixed_source(n, k, seed=1729) -> torch.Tensor:
    """Deterministic pseudorandom equiatomic source (playbook 13, headline)."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    x = torch.zeros(n, dtype=torch.int64)
    x[perm[:k]] = 1
    return x


def check_states(x, k, tag="state"):
    """Hard guard: binary values and exact fixed count (playbook 31)."""
    if x.dtype not in (torch.int64, torch.int32, torch.bool):
        raise RuntimeError(f"{tag}: non-integer state tensor {x.dtype}")
    xl = x.long()
    if int(xl.min()) < 0 or int(xl.max()) > 1:
        raise RuntimeError(f"{tag}: non-binary state values")
    if not bool((xl.sum(-1) == k).all()):
        raise RuntimeError(f"{tag}: fixed count violated (k={k})")
    return xl


# ----------------------------------------------------------------------------
# CuAuSpace: the fixed-composition state space without enumeration
# ----------------------------------------------------------------------------


class CuAuSpace:
    """``Omega_{N,k}`` for CuAu: direct ``(B, N)`` tensors, no enumeration.

    Everything the IASBS fixed-composition branch needs that does *not* depend
    on enumeration is reused verbatim from :mod:`iasbs.fixed_ising`: the Johnson
    orbit kernel ``kappa``, the occupancy-class bridge tables, the uniform
    legal-swap reference rate ``gamma / (k (N-k))``, and the Dirac terminal
    label.  The two differences are (i) states are stored as binary tensors and
    (ii) the energy is the pinned CuAu cluster expansion.
    """

    def __init__(self, size=(4, 4, 4), temp_k=500.0, gamma=10.0, source=None,
                 device="cuda", energy=None, count=True, verbose=False):
        self.size = tuple(int(v) for v in size)
        self.device = torch.device(device)
        self.temp_k = float(temp_k)
        self.tau = KB_EV_PER_K * self.temp_k          # eV; E is total eV
        self.gamma = float(gamma)

        if energy is None:
            tables = build_ce_tables(self.size, verbose=verbose)
            energy = TorchCuAuEnergy(tables, device=device)
        self.energy = CountedEnergy(energy) if count else energy
        self.n = self.n_sites = int(energy.N)
        if self.n != int(np.prod(self.size)):
            raise RuntimeError("CE structure site count != prod(size)")
        self.k = self.n // 2
        if 2 * self.k != self.n:
            raise RuntimeError("CuAu benchmark requires an even site count")
        self.n_edges = self.k * (self.n - self.k)
        self.positions = np.asarray(energy.positions, dtype=np.float64)
        self.a = float(energy.a)

        # L10 phase tables for the order parameters
        q = l10_q_vectors(self.a)
        ph = self.positions @ q.T
        self.cosp = torch.as_tensor(np.cos(ph), dtype=torch.float64,
                                    device=self.device)
        self.sinp = torch.as_tensor(np.sin(ph), dtype=torch.float64,
                                    device=self.device)

        # source state
        if source is None:
            src = random_fixed_source(self.n, self.k)
        elif isinstance(source, torch.Tensor):
            src = source.detach().cpu().long()
        else:
            src = torch.as_tensor(np.asarray(source, dtype=np.int64))
        check_states(src.unsqueeze(0), self.k, "source")
        self.x0 = src.to(self.device)

        # Johnson reference kernel over the full horizon
        kap = C.binary_orbit_kernel(self.n, self.k, self.gamma)
        if not np.all(np.isfinite(kap)) or np.any(kap <= 0.0):
            raise RuntimeError("invalid Johnson orbit kernel")
        self.log_kappa_full = torch.as_tensor(np.log(kap), dtype=torch.float64,
                                              device=self.device)
        self._bridge_tables()

    # -- bridge --------------------------------------------------------------

    def _bridge_tables(self):
        """Occupancy-class tables; copied from ``FixedIsingSpace`` unchanged.

        Depends only on ``(n, k)``: the four blocks A = both endpoints,
        B = source only, C = terminal only, D = neither have sizes
        ``(k-m, m, m, n-k-m)`` for Johnson distance ``m``, and the bridge weight
        is constant on a class of size ``C(|A|,a) C(|B|,b) C(|C|,c) C(|D|,d)``.
        """
        n, k = self.n, self.k
        J = min(k, n - k)
        lc = lambda N, r: (gammaln(N + 1.0) - gammaln(r + 1.0)
                           - gammaln(N - r + 1.0))
        rows = []
        for m in range(J + 1):
            sz = (k - m, m, m, n - k - m)
            cur = [(a, b, c, k - a - b - c,
                    sum(lc(sz[i], v) for i, v in
                        enumerate((a, b, c, k - a - b - c))))
                   for a in range(sz[0] + 1)
                   for b in range(sz[1] + 1)
                   for c in range(sz[2] + 1)
                   if 0 <= k - a - b - c <= sz[3]]
            rows.append(cur)
        Lm = max(len(r) for r in rows)
        A = np.zeros((J + 1, Lm, 4), dtype=np.int64)
        Wt = np.full((J + 1, Lm), -np.inf)
        D0 = np.zeros((J + 1, Lm), dtype=np.int64)
        D1 = np.zeros((J + 1, Lm), dtype=np.int64)
        for m, cur in enumerate(rows):
            for i, (a, b, c, d, w) in enumerate(cur):
                A[m, i] = (a, b, c, d)
                Wt[m, i] = w
                D0[m, i] = k - a - b
                D1[m, i] = k - a - c
        dev = self.device
        self.br_abcd = torch.as_tensor(A, device=dev)
        self.br_lmult = torch.as_tensor(Wt, dtype=torch.float64, device=dev)
        self.br_d0 = torch.as_tensor(D0, device=dev)
        self.br_d1 = torch.as_tensor(D1, device=dev)
        self.br_size = torch.as_tensor(
            np.array([[k - m, m, m, n - k - m] for m in range(J + 1)]),
            device=dev)

    # -- helpers -------------------------------------------------------------

    def distance(self, x0, x1):
        """Johnson distance ``d = k - <x0, x1>``; broadcasts over batches."""
        return self.k - (x0.long() * x1.long()).sum(-1)

    def kappa_grid(self, ts):
        """``kappa_{gamma (1-t)}[d]`` for every ``t`` in ``ts``.  ``(T, J+1)``."""
        Gam = np.maximum(self.gamma * (1.0 - np.asarray(ts, dtype=np.float64)),
                         1e-12)
        kap = np.stack([np.maximum(C.binary_orbit_kernel(self.n, self.k, float(g)),
                                   1e-300) for g in Gam])
        return torch.as_tensor(kap, dtype=torch.float64, device=self.device)

    def bridge_kernels(self, steps):
        """``(log_kap0, log_kap1)`` on the ``steps``-point training grid.

        ``log_kap0[s]`` is the kernel of the elapsed time ``t_s = s / steps``
        and ``log_kap1[s]`` that of the remaining time ``1 - t_s``, matching
        ``fixed_ising.run_train``.
        """
        ts = np.arange(steps, dtype=np.float64) / steps
        log_kap0 = torch.log(self.kappa_grid(1.0 - ts))
        log_kap1 = torch.log(self.kappa_grid(ts))
        if not bool(torch.isfinite(log_kap0).all() and torch.isfinite(log_kap1).all()):
            raise RuntimeError("non-finite log kappa on the bridge grid")
        return log_kap0, log_kap1

    def order_parameters(self, x):
        """``(..., 3)`` L10 structure factors ``Q_alpha``."""
        return l10_order_parameters_torch(x, self.cosp, self.sinp)

    def ideal_l10(self, axis=0):
        q = l10_q_vectors(self.a)[axis]
        return torch.as_tensor(ideal_l10_state(self.positions, q),
                               device=self.device)


# ----------------------------------------------------------------------------
# direct bridge / simulation / labels  (playbook 8-10)
# ----------------------------------------------------------------------------


@torch.no_grad()
def sample_bridge_direct(space, X1, t_idx, log_kap0, log_kap1, generator=None,
                         X0=None):
    """Exact reference bridge ``X_t | x_0, X_1`` on direct binary tensors.

    Identical construction to ``fixed_ising.sample_bridge``; the only changes
    are that ``X1`` arrives as a ``(B, N)`` tensor instead of an enumeration
    index and the result is returned as a tensor instead of an index.
    """
    B, n = X1.shape
    x0b = space.x0.unsqueeze(0).expand(B, -1) if X0 is None else X0.long()
    m = space.distance(x0b, X1)

    lw = (space.br_lmult[m]
          + torch.gather(log_kap0[t_idx], 1, space.br_d0[m])
          + torch.gather(log_kap1[t_idx], 1, space.br_d1[m]))
    lw = lw - lw.max(dim=1, keepdim=True).values
    sel = torch.multinomial(torch.exp(lw), 1, generator=generator).squeeze(1)

    cnt = space.br_abcd[m, sel]                                # (B, 4)
    sizes = space.br_size[m]                                   # (B, 4)

    blk = 2 * (1 - x0b) + (1 - X1.long())                      # A,B,C,D labels
    key = torch.rand(B, n, device=X1.device, generator=generator)
    order = torch.argsort(blk.to(key.dtype) * 2.0 + key, dim=1)
    bs = torch.gather(blk, 1, order)

    start = torch.cat([torch.zeros_like(sizes[:, :1]),
                       sizes.cumsum(1)[:, :3]], dim=1)
    pos = torch.arange(n, device=X1.device).expand(B, n)
    take = (pos - torch.gather(start, 1, bs)) < torch.gather(cnt, 1, bs)

    x = torch.zeros(B, n, dtype=torch.int64, device=X1.device)
    x.scatter_(1, order, take.long())
    return check_states(x, space.k, "bridge")


@torch.no_grad()
def simulate_direct(space, net, batch, steps, generator=None, x0=None,
                    return_path=False):
    """Frozen-rate, at most one jump per grid bin -- same scheme as the Ising branch."""
    if x0 is None:
        x = space.x0.unsqueeze(0).expand(batch, -1).clone()
    else:
        x = x0.long().to(space.device).clone()
        if x.shape[0] != batch:
            x = x.expand(batch, -1).clone()
    check_states(x, space.k, "sim init")

    dt = 1.0 / steps
    base = space.gamma / space.n_edges
    path = [x.clone()] if return_path else None

    for s in range(steps):
        tt = torch.full((batch,), s * dt, device=space.device, dtype=torch.float64)
        if net is None:
            a = torch.zeros(batch, space.n, space.n, device=space.device,
                            dtype=torch.float64)
        else:
            a = net(tt, x).to(torch.float64).clamp(-20.0, 20.0)
        u = base * torch.exp(a) * legal_mask(x)
        flat = u.reshape(batch, -1)
        R = flat.sum(dim=1)
        p_stay = torch.exp(-R * dt)

        jump = torch.rand(batch, device=space.device, generator=generator) > p_stay
        if bool(jump.any()):
            rows = torch.nonzero(jump, as_tuple=False).squeeze(1)
            e = torch.multinomial(flat[rows], 1, generator=generator).squeeze(1)
            i, j = e // space.n, e % space.n
            x = x.clone()
            x[rows, i] = 0
            x[rows, j] = 1
        if return_path:
            path.append(x.clone())

    check_states(x, space.k, "sim final")
    return (x, path) if return_path else x


@torch.no_grad()
def terminal_log_label_edge(space, X1, E1, i, j, check=False):
    """``log Lambda_{(i,j)}(X1)`` for one sampled edge per row.  ``(B,)``."""
    Xg = transpose_batch(X1, i, j)
    Eg = space.energy.energy_torch(Xg).to(torch.float64)
    if not bool(torch.isfinite(Eg).all()):
        raise RuntimeError("non-finite CE energy in terminal label")

    x0b = space.x0.unsqueeze(0)
    d1 = space.distance(x0b, X1).long()
    dg = space.distance(x0b, Xg).long()
    out = (-(Eg - E1.to(torch.float64)) / space.tau
           + space.log_kappa_full[d1] - space.log_kappa_full[dg])
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("non-finite terminal label before clipping")
    if check:
        b = torch.arange(X1.shape[0], device=X1.device)
        same = X1[b, i] == X1[b, j]
        if bool(same.any()) and float(out[same].abs().max()) > 1e-9:
            raise RuntimeError("equal-symbol transposition did not give Lambda=1")
    return out


@torch.no_grad()
def terminal_log_label_all_pairs(space, X1, E1, pairs):
    """``log Lambda_g`` for every unordered pair ``g``.  ``(B, Np)``; tests only."""
    B = X1.shape[0]
    out = torch.empty(B, pairs.shape[0], dtype=torch.float64, device=X1.device)
    for p in range(pairs.shape[0]):
        i = torch.full((B,), int(pairs[p, 0]), device=X1.device)
        j = torch.full((B,), int(pairs[p, 1]), device=X1.device)
        out[:, p] = terminal_log_label_edge(space, X1, E1, i, j)
    return out


def poisson_edge_loss(a_edge, lam):
    """Bregman/Poisson loss on the sampled edges; ``exp(a) - a * Lambda``."""
    return (torch.exp(a_edge) - a_edge * lam).mean()


def all_edge_poisson_loss(a, lam_full, mask):
    """Exact edge-averaged loss for the same quantities; reference for tests."""
    m = mask.to(a.dtype)
    per = (torch.exp(a) - a * lam_full) * m
    return (per.sum(dim=(1, 2)) / m.sum(dim=(1, 2))).mean()


# ----------------------------------------------------------------------------
# CuAu-S: the enumerated 16-site sector, for exact evaluation only
# ----------------------------------------------------------------------------


class CuAuExact(CuAuSpace):
    """``CuAuSpace`` plus the enumeration tables that only ``N=16`` affords.

    The canonical sector has ``C(16,8) = 12,870`` states, so the exact Gibbs
    law, the exact terminal law of the discretised controlled chain and the
    oracle (Doob) control are all computable.  Every attribute name matches
    :class:`iasbs.fixed_ising.FixedIsingSpace`, so ``step_probs``,
    ``propagate_exact``, ``simulate``, ``ExactControl``, ``net_mult_on_index``
    and ``net_mult_all_states`` are reused from that module verbatim -- the
    exact-evaluation numerics are literally the ones already validated on the
    Ising benchmark.
    """

    def __init__(self, size=(2, 2, 4), **kw):
        super().__init__(size=size, **kw)
        n, k, dev = self.n, self.k, self.device
        if n > 20:
            raise RuntimeError(f"refusing to enumerate {n} sites")

        S = C.enumerate_fixed_count(n, k).astype(np.int64)
        self.M = M = S.shape[0]
        self.S_np = S
        self.S = torch.as_tensor(S, device=dev)
        self.Sf = self.S.to(torch.float32)

        pow2 = (1 << np.arange(n)).astype(np.int64)
        masks = (S * pow2).sum(axis=1)
        lut = np.full(1 << n, -1, dtype=np.int64)
        lut[masks] = np.arange(M)
        self.masks = masks
        self.pow2 = torch.as_tensor(pow2, device=dev)
        self.lut_t = torch.as_tensor(lut, device=dev)

        occ = np.nonzero(S == 1)[1].reshape(M, k)
        emp = np.nonzero(S == 0)[1].reshape(M, n - k)
        self.occ = torch.as_tensor(occ, device=dev)
        self.emp = torch.as_tensor(emp, device=dev)

        tgt_mask = (masks[:, None, None] ^ pow2[occ][:, :, None]
                    ^ pow2[emp][:, None, :])
        tgt = lut[tgt_mask].reshape(M, self.n_edges)
        if tgt.min() < 0:
            raise RuntimeError("swap target left the fixed-count sector")
        self.tgt = torch.as_tensor(tgt, device=dev)

        pair_id = np.full((n, n), -1, dtype=np.int64)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_id[i, j] = pair_id[j, i] = len(pairs)
                pairs.append((i, j))
        self.pairs = torch.as_tensor(np.asarray(pairs), device=dev)
        self.n_pairs = len(pairs)
        self.pair_id = torch.as_tensor(pair_id, device=dev)
        ei = np.repeat(occ, n - k, axis=1)
        ej = np.tile(emp, (1, k))
        self.edge_i = torch.as_tensor(ei, device=dev)
        self.edge_j = torch.as_tensor(ej, device=dev)
        self.edge_pair = torch.as_tensor(pair_id[ei, ej], device=dev)

        # exact canonical law under the pinned CE (total eV)
        self.E = self.energy.energy_torch(self.S).to(torch.float64)
        if not bool(torch.isfinite(self.E).all()):
            raise RuntimeError("non-finite CE energy on the enumerated sector")
        logw = -self.E / self.tau
        logw = logw - logw.max()
        w = torch.exp(logw)
        self.pi = w / w.sum()

        self.i0 = int(lut[int((self.x0.cpu().numpy() * pow2).sum())])
        self.dist0 = self.k - (self.S * self.x0).sum(dim=1)

        # f1 and the distance-weight matrix for the oracle control
        logf1 = -self.E / self.tau - self.log_kappa_full[self.dist0]
        self.logf1 = logf1 - logf1.max()
        self.f1 = torch.exp(self.logf1)
        ov = self.Sf @ self.Sf.T                                # (M, M)
        J = min(k, n - k)
        self.W = torch.stack([((ov == (k - jj)).to(self.f1.dtype)
                               * self.f1).sum(1) for jj in range(J + 1)], 1)
        del ov

        # order parameters of every state, for exact Q laws
        self.Q = self.order_parameters(self.S)                  # (M, 3)
        self.Qmax = self.Q.max(dim=1).values

    # -- exact metrics -------------------------------------------------------

    def heat_capacity(self, p):
        """``Cv/N`` in eV/K per atom for a law ``p`` over the sector."""
        m1 = float((p * self.E).sum())
        m2 = float((p * self.E * self.E).sum())
        var = max(m2 - m1 * m1, 0.0)
        return var / (self.n * KB_EV_PER_K * self.temp_k ** 2)

    def exact_report(self, p, tag="", extra=None):
        pi = self.pi
        p = p / p.sum()
        tv = 0.5 * float((p - pi).abs().sum())
        kl = float((p * (torch.log(p.clamp_min(1e-300))
                         - torch.log(pi.clamp_min(1e-300)))).sum())
        hell = float(torch.sqrt(0.5 * ((p.sqrt() - pi.sqrt()) ** 2).sum()))
        # CE energies are generically all distinct in float64, so quantise to
        # 1 ueV before binning; otherwise this metric degenerates to full TV.
        lv, inv = torch.unique(torch.round(self.E * 1e6), return_inverse=True)
        a = torch.zeros(len(lv), device=self.device, dtype=p.dtype).index_add(0, inv, p)
        b = torch.zeros(len(lv), device=self.device, dtype=pi.dtype).index_add(0, inv, pi)
        e_hist_tv = 0.5 * float((a - b).abs().sum())
        mE = float((p * self.E).sum())
        mE_ex = float((pi * self.E).sum())
        cv, cv_ex = self.heat_capacity(p), self.heat_capacity(pi)
        # Qmax law on bins fixed by the exact reference
        edges = torch.linspace(0.0, 1.0, 21, device=self.device,
                               dtype=torch.float64)
        bin_idx = torch.clamp(torch.bucketize(self.Qmax, edges) - 1, 0, 19)
        qa = torch.zeros(20, device=self.device, dtype=p.dtype).index_add(0, bin_idx, p)
        qb = torch.zeros(20, device=self.device, dtype=pi.dtype).index_add(0, bin_idx, pi)
        out = {
            "tag": tag,
            "TV": tv,
            "KL": kl,
            "Hellinger": hell,
            "energy_hist_TV": e_hist_tv,
            "Qmax_hist_TV": 0.5 * float((qa - qb).abs().sum()),
            "mean_E_per_atom_meV": 1000.0 * mE / self.n,
            "exact_mean_E_per_atom_meV": 1000.0 * mE_ex / self.n,
            "mean_E_err_meV_per_atom": 1000.0 * abs(mE - mE_ex) / self.n,
            "Cv_per_atom": cv,
            "exact_Cv_per_atom": cv_ex,
            "Cv_rel_err": abs(cv - cv_ex) / max(cv_ex, 1e-300),
            "mean_Qmax": float((p * self.Qmax).sum()),
            "exact_mean_Qmax": float((pi * self.Qmax).sum()),
            "mass_leak": abs(float(p.sum()) - 1.0),
        }
        if extra:
            out.update(extra)
        return out

    def empirical_law(self, x):
        """Index a batch of direct states and return its empirical law."""
        check_states(x, self.k, "empirical")
        idx = self.lut_t[(x * self.pow2).sum(dim=1)]
        if int(idx.min()) < 0:
            raise RuntimeError("state outside the enumerated sector")
        c = torch.bincount(idx, minlength=self.M).to(torch.float64)
        return c / c.sum()

    def iid_tv_floor(self, n_samples, reps=5, seed=0):
        """Finite-sample TV floor of ``n_samples`` exact iid draws."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        vals = []
        for _ in range(reps):
            idx = torch.multinomial(self.pi, n_samples, replacement=True,
                                    generator=g)
            e = torch.bincount(idx, minlength=self.M).to(torch.float64)
            e = e / e.sum()
            vals.append(0.5 * float((e - self.pi).abs().sum()))
        return float(np.mean(vals)), float(np.std(vals))


# ----------------------------------------------------------------------------
# validate: Phase 1 hard gates
# ----------------------------------------------------------------------------


def cmd_validate(args):
    dev = args.device
    size = tuple(args.size)
    print(f"[cuau validate] size={size}  device={dev}")
    print("provenance:", json.dumps(provenance(), indent=2)[:400], "...")

    print("\n-- building exact multilinear CE tables from CLEASE --")
    tab = build_ce_tables(size=size, n_calib=args.n_calib, cache=not args.no_cache)
    N = tab["N"]
    ref = CleaseCuAuEnergy(size=size)
    tor = TorchCuAuEnergy(tab, device=dev)
    rng = np.random.default_rng(args.seed)
    fails = []

    # [G1] site ordering / geometry parity between the VASP file and CLEASE
    dpos = np.abs(ref.positions - tab["positions"]).max()
    ok = dpos < 1e-9
    print(f"[G1] site ordering/positions match          max|dr| = {dpos:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["G1"]

    # [G2] N equals the requested supercell
    ok = N == int(np.prod(size))
    print(f"[G2] site count = prod(size)               N = {N}                {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["G2"]

    # [G3] 100+ equiatomic configuration energy parity, torch vs CLEASE
    n = args.n_parity
    Xe = np.zeros((n, N), dtype=np.int64)
    for r in range(n):
        Xe[r, rng.permutation(N)[: N // 2]] = 1
    e_ref = ref.energy_np(Xe)
    e_tor = tor.energy_np(Xe)
    err = np.abs(e_ref - e_tor).max()
    ok = err < args.tol
    print(f"[G3] {n} equiatomic energy parity         max|dE| = {err:.3e} eV   "
          f"{'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["G3"]

    # [G4] swap Delta-E parity
    dref, dtor = [], []
    for r in range(n):
        x = Xe[r].copy()
        au = np.flatnonzero(x == 1)
        cu = np.flatnonzero(x == 0)
        i = int(rng.choice(au))
        j = int(rng.choice(cu))
        y = x.copy()
        y[i], y[j] = x[j], x[i]
        dref.append(ref.energy_one(y) - e_ref[r])
        dtor.append(float(tor.energy_np(y[None])[0]) - e_tor[r])
    derr = np.abs(np.array(dref) - np.array(dtor)).max()
    ok = derr < args.tol
    print(f"[G4] {n} swap Delta-E parity              max|ddE| = {derr:.3e} eV  "
          f"{'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["G4"]

    # [G5] species mapping: 1 = Au.  Pure Au must be Z=79 everywhere.
    ref.atoms.numbers = (np.ones(N, dtype=np.int64) * 50 + 29)
    ok = bool((ref.atoms.numbers == 79).all())
    ref.atoms.numbers = (np.zeros(N, dtype=np.int64) * 50 + 29)
    ok = ok and bool((ref.atoms.numbers == 29).all())
    print(f"[G5] species map x=1 -> Au(79), x=0 -> Cu(29)                      "
          f"{'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["G5"]

    # [G6] unit gate: energy is total eV for the supercell, not eV/atom and not
    # k_B T.  Doubling the cell must double the energy of the replicated state.
    print(f"[G6] unit check  E(random)/N = {e_ref[0] / N:+.6f} eV/atom, "
          f"E = {e_ref[0]:+.6f} eV total   (kB*T at 500K = {KB_EV_PER_K * 500:.6f} eV)")

    # [G7] ideal L10 states are equiatomic and degenerate under x/y/z
    q = l10_q_vectors(tab["a"])
    eL, QL = [], []
    for al in range(3):
        xi = ideal_l10_state(tab["positions"], q[al])
        eL.append(ref.energy_one(xi))
        QL.append(l10_order_parameters(xi, tab["positions"], q))
    eL = np.array(eL)
    QL = np.array(QL)
    cubic = (size[0] == size[1] == size[2])
    spread = float(eL.max() - eL.min())
    ok = (spread < args.tol) if cubic else True
    print(f"[G7] ideal L10 x/y/z energies              E/N = "
          f"{', '.join('%+.6f' % (v / N) for v in eL)} eV/atom, spread {spread:.2e}  "
          f"{'PASS' if ok else 'FAIL'}{'' if cubic else '  (non-cubic cell, not gated)'}")
    fails += [] if ok else ["G7"]

    # [G8] each ideal variant maximises its own Q and only its own
    okQ = True
    for al in range(3):
        okQ = okQ and int(np.argmax(QL[al])) == al and QL[al, al] > 0.99
        okQ = okQ and QL[al][np.arange(3) != al].max() < 1e-9
    print(f"[G8] Q_alpha selectivity                   diag = "
          f"{', '.join('%.4f' % QL[a, a] for a in range(3))}, offdiag max "
          f"{max(QL[a][np.arange(3) != a].max() for a in range(3)):.2e}   "
          f"{'PASS' if okQ else 'FAIL'}")
    fails += [] if okQ else ["G8"]

    # [G9] random equiatomic states show no deterministic orientation preference
    Qr = l10_order_parameters(Xe, tab["positions"], q)
    frac = np.bincount(Qr.argmax(1), minlength=3) / len(Qr)
    ok = frac.max() < 0.60
    print(f"[G9] random-state argmax fractions         {np.round(frac, 3)}          "
          f"{'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["G9"]

    # [G10] cubic Hamiltonian symmetry: the site permutation induced by the
    # x<->y<->z rotation must leave the CE energy invariant.
    if cubic:
        pos = tab["positions"]
        L = tab["a"] * size[0]
        key = {tuple(np.round(p / tab["a"] * 2).astype(int) % (2 * size[0])): m
               for m, p in enumerate(pos)}
        perm = np.full(N, -1, dtype=np.int64)
        for m, p in enumerate(pos):
            rolled = np.array([p[2], p[0], p[1]])   # cyclic x->y->z->x
            k = tuple(np.round(rolled / tab["a"] * 2).astype(int) % (2 * size[0]))
            perm[m] = key.get(k, -1)
        if (perm < 0).any():
            print("[G10] cubic rotation permutation                                    "
                  "FAIL (site map incomplete)")
            fails += ["G10"]
        else:
            e_rot = ref.energy_np(Xe[:, perm])
            rerr = np.abs(e_rot - e_ref).max()
            ok = rerr < args.tol
            print(f"[G10] cubic x->y->z Hamiltonian symmetry   max|dE| = {rerr:.3e} eV   "
                  f"{'PASS' if ok else 'FAIL'}")
            fails += [] if ok else ["G10"]
            Qp = l10_order_parameters(Xe[:1, perm], tab["positions"], q)
            print(f"      Q permutes: {np.round(Qr[0], 4)} -> {np.round(Qp[0], 4)}")
        np.save(f"{DATA}/cubic_perm_{size[0]}x{size[1]}x{size[2]}.npy", perm)
    else:
        print("[G10] cubic symmetry gate skipped (non-cubic supercell: no exact "
              "1/3 orientation mass may be claimed)")

    # [G11] throughput
    B = args.bench
    Xb = np.zeros((B, N), dtype=np.int64)
    for r in range(B):
        Xb[r, rng.permutation(N)[: N // 2]] = 1
    t = torch.as_tensor(Xb, device=dev)
    tor.energy_torch(t)
    torch.cuda.synchronize() if dev.startswith("cuda") else None
    t0 = time.time()
    for _ in range(10):
        tor.energy_torch(t)
    torch.cuda.synchronize() if dev.startswith("cuda") else None
    rate_t = 10 * B / (time.time() - t0)
    t0 = time.time()
    ref.energy_np(Xb[: min(B, 500)])
    rate_c = min(B, 500) / (time.time() - t0)
    print(f"[G11] throughput   torch {rate_t:,.0f} configs/s   CLEASE {rate_c:,.0f} "
          f"configs/s   speedup {rate_t / rate_c:,.0f}x")

    out = {
        "size": list(size), "N": N, "provenance": provenance(),
        "ce_tables": {"names": tab["names"], "n_clusters": [int(len(o)) for o in tab["orbits"]],
                      "coef": [float(v) for v in tab["coef"]],
                      "fit_max_err": tab["fit_max_err"], "heldout_max_err": tab["heldout_max_err"]},
        "gates": {"G3_energy_parity": float(err), "G4_dE_parity": float(derr),
                  "G7_l10_energy_spread": spread,
                  "G7_l10_E_per_atom": [float(v / N) for v in eL],
                  "G9_argmax_fracs": [float(v) for v in frac]},
        "throughput": {"torch_configs_per_s": rate_t, "clease_configs_per_s": rate_c},
        "failed_gates": fails,
    }
    os.makedirs("json", exist_ok=True)
    p = f"json/results_cuau_validate_{size[0]}x{size[1]}x{size[2]}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    if fails:
        print(f"FAILED GATES: {fails}  -- do not proceed")
        return 1
    print("all gates PASS")
    return 0


def cmd_exact(args):
    """Phase 3: exact CuAu-S law + oracle-control discretization sweep."""
    from iasbs.fixed_ising import ExactControl, propagate_exact, simulate

    sp = CuAuExact(size=tuple(args.size), temp_k=args.temp, gamma=args.gamma,
                   device=args.device, verbose=True)
    kap = np.exp(sp.log_kappa_full.cpu().numpy())
    print(f"CuAu-S  size={sp.size}  N={sp.n}  k={sp.k}  |Omega|={sp.M}")
    print(f"T={sp.temp_k} K  tau={sp.tau:.6f} eV  gamma={sp.gamma}")
    print(f"kappa(.|x0) range {kap.min():.3e} .. {kap.max():.3e}  "
          f"(ratio {kap.max()/kap.min():.3e})")
    print(f"exact <E>/N = {1000*float((sp.pi*sp.E).sum())/sp.n:.4f} meV/atom   "
          f"source E/N = {1000*float(sp.E[sp.i0])/sp.n:.4f} meV/atom")
    print(f"exact <Qmax> = {float((sp.pi*sp.Qmax).sum()):.4f}   "
          f"Cv/N = {sp.heat_capacity(sp.pi):.6e} eV/K")
    print(f"pi: max {float(sp.pi.max()):.3e}  min {float(sp.pi.min()):.3e}  "
          f"top-1 state E/N {1000*float(sp.E[int(sp.pi.argmax())])/sp.n:.4f} meV/atom")
    print()

    rows = []
    for steps in args.steps_sweep:
        ctrl = ExactControl(sp, steps)
        t0 = time.time()
        p = propagate_exact(sp, ctrl.all_states, steps)
        r = sp.exact_report(p, f"oracle steps={steps}",
                            {"steps": steps, "sec": round(time.time() - t0, 2)})
        rows.append(r)
        print(f"  steps={steps:4d}  TV={r['TV']:.5f}  KL={r['KL']:.3e}  "
              f"Hell={r['Hellinger']:.5f}  E-hist TV={r['energy_hist_TV']:.5f}  "
              f"dE={r['mean_E_err_meV_per_atom']:.4f} meV/atom  "
              f"Cv rel {r['Cv_rel_err']:.4f}  leak={r['mass_leak']:.1e}  "
              f"{r['sec']}s")

    steps = args.steps_sweep[-1]
    ctrl = ExactControl(sp, steps)
    idx = simulate(sp, ctrl.on_states, args.n_samples, steps,
                   generator=torch.Generator(device=sp.device).manual_seed(0))
    X = sp.S[idx]
    viol = int((X.sum(dim=1) != sp.k).sum())
    emp = torch.bincount(idx, minlength=sp.M).to(torch.float64)
    emp = emp / emp.sum()
    tv_emp = 0.5 * float((emp - sp.pi).abs().sum())
    floor, floor_sd = sp.iid_tv_floor(args.n_samples)
    print(f"\n  oracle sampled N={args.n_samples} at steps={steps}: "
          f"violations={viol}  empirical TV={tv_emp:.5f}  "
          f"iid floor={floor:.5f} +- {floor_sd:.5f}")

    best = min(r["TV"] for r in rows)
    res = {"provenance": provenance(), "config": vars(args),
           "N": sp.n, "M": sp.M, "temp_k": sp.temp_k, "tau_eV": sp.tau,
           "exact": {"mean_E_per_atom_meV": 1000 * float((sp.pi * sp.E).sum()) / sp.n,
                     "Cv_per_atom": sp.heat_capacity(sp.pi),
                     "mean_Qmax": float((sp.pi * sp.Qmax).sum())},
           "rows": rows, "violations": viol, "empirical_TV": tv_emp,
           "iid_TV_floor": floor, "iid_TV_floor_sd": floor_sd,
           "oracle_floor_TV": best,
           "energy_calls": sp.energy.report()}
    out = args.out or f"json/results_cuau_exact_{sp.n}_{int(sp.temp_k)}K.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  wrote {out}")

    print("\n=== GATES ===")
    ok_tv = best <= 0.05
    print(f"S1  oracle floor TV <= 0.05 : {'PASS' if ok_tv else 'FAIL'} "
          f"({best:.5f})")
    print(f"S2  constraint violations 0 : {'PASS' if viol == 0 else 'FAIL'} "
          f"({viol})")
    return 0 if (ok_tv and viol == 0) else 1


def cmd_train(args):
    """IASBS training on CuAu with direct tensors and edge minibatching."""
    from iasbs.fixed_ising import (ExactControl, SwapController,
                                   net_mult_all_states, propagate_exact)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = tuple(args.size)
    exact = int(np.prod(size)) <= 16 and not args.no_exact
    cls = CuAuExact if exact else CuAuSpace

    tables = build_ce_tables(size, verbose=True)
    energy = TorchCuAuEnergy(tables, device=args.device)
    if args.source == "l10x":                  # ordered-source stress test
        src = ideal_l10_state(energy.positions, l10_q_vectors(energy.a)[0])
    elif args.source == "random":
        src = None                             # deterministic seed-1729 default
    else:
        raise RuntimeError(f"unknown source policy {args.source!r}")
    sp = cls(size=size, temp_k=args.temp, gamma=args.gamma, device=args.device,
             source=src, energy=energy, verbose=True)

    net = SwapController(sp.n, hidden=args.hidden).to(sp.device)
    if args.init_ckpt:
        # Temperature curriculum: warm start from a net trained at a higher
        # temperature, where pi is broad and every basin is reachable, then
        # continue in the spiky regime.  Only the controller is carried over;
        # the space, kappa tables and labels are rebuilt at the new tau.
        sd = torch.load(args.init_ckpt, map_location=sp.device,
                        weights_only=False)
        sd = sd.get("net", sd)
        net.load_state_dict(sd)
        print(f"  warm start from {args.init_ckpt}")
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters * args.inner, eta_min=args.lr * 0.05)
    log_kap0, log_kap1 = sp.bridge_kernels(steps)
    gen = torch.Generator(device=sp.device).manual_seed(args.seed + 1234)

    print(f"CuAu train  size={size}  N={sp.n}  k={sp.k}  T={sp.temp_k} K  "
          f"gamma={sp.gamma}  steps={steps}  params={n_par}  exact={exact}")
    if exact:
        print(f"  |Omega|={sp.M}  exact <E>/N = "
              f"{1000*float((sp.pi*sp.E).sum())/sp.n:.4f} meV/atom")

    replay, hist = [], []
    t_start = time.time()
    best = {"TV": float("inf")}
    for it in range(1, args.iters + 1):
        sp.energy.tag = "train"
        with torch.no_grad():
            # Exploration mixing.  The buffer is otherwise filled purely by the
            # current controller, so once it favours one of the six degenerate
            # L1_0 ground states the other five receive no label signal at all
            # and the loss has no term that can recover them.  A fraction of
            # each push is therefore drawn from the zero controller, i.e. the
            # base swap process, which is basin-agnostic by construction.  The
            # Poisson loss is an average over terminal states, so broadening
            # the terminal pool changes which states are visited, not the
            # per-state fixed point.
            n_exp = int(round(args.explore_frac * args.batch))
            n_pol = args.batch - n_exp
            parts = []
            if n_pol > 0:
                parts.append(simulate_direct(sp, net, n_pol, steps,
                                             generator=gen))
            if n_exp > 0:
                parts.append(simulate_direct(sp, None, n_exp, steps,
                                             generator=gen))
            X1n = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
            E1n = sp.energy.energy_torch(X1n).to(torch.float64)
        replay.append((X1n.detach(), E1n.detach()))
        if len(replay) > args.buffer:
            replay.pop(0)
        pool_x = torch.cat([p[0] for p in replay], 0)
        pool_e = torch.cat([p[1] for p in replay], 0)

        for _ in range(args.inner):
            sel = torch.randint(len(pool_x), (args.mb,), device=sp.device,
                                generator=gen)
            X1, E1 = pool_x[sel], pool_e[sel]
            ti = torch.randint(steps, (args.mb,), device=sp.device, generator=gen)
            with torch.no_grad():
                Xt = sample_bridge_direct(sp, X1, ti, log_kap0, log_kap1,
                                          generator=gen)
                if args.all_edges:
                    # exact edge average: every legal edge of X_t.  Only
                    # affordable because the CE evaluator is batched on GPU;
                    # used as the variance control for edge minibatching.
                    ei, ej = all_legal_edges(Xt)
                    rep = ei.shape[1]
                    X1r = X1.repeat_interleave(rep, 0)
                    E1r = E1.repeat_interleave(rep, 0)
                    i, j = ei.reshape(-1), ej.reshape(-1)
                    lam = torch.exp(terminal_log_label_edge(
                        sp, X1r, E1r, i, j).clamp(-20.0, 20.0))
                else:
                    rep = args.edge_samples
                    lam_l, i_l, j_l = [], [], []
                    for _ in range(rep):
                        ei, ej = sample_uniform_legal_edge(Xt, generator=gen)
                        lam_l.append(terminal_log_label_edge(sp, X1, E1, ei, ej))
                        i_l.append(ei)
                        j_l.append(ej)
                    i, j = torch.cat(i_l), torch.cat(j_l)
                    lam = torch.exp(torch.cat(lam_l).clamp(-20.0, 20.0))

            if args.all_edges:
                tt = (ti.to(torch.float64) / steps).repeat_interleave(rep)
                a = net(tt, Xt.repeat_interleave(rep, 0))
            else:
                tt = (ti.to(torch.float64) / steps).repeat(rep)
                a = net(tt, Xt.repeat(rep, 1))
            b = torch.arange(a.shape[0], device=sp.device)
            av = a[b, i, j].clamp(-20.0, 20.0).to(torch.float64)
            loss = (poisson_edge_loss(av, lam) if args.loss == "poisson"
                    else ((torch.exp(av) - lam) ** 2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()

        if it == 1:
            print(f"  label Lambda: mean {float(lam.mean()):.3f}  "
                  f"p50 {float(lam.median()):.3f}  max {float(lam.max()):.2f}  "
                  f"frac==1 {float((lam-1).abs().lt(1e-9).double().mean()):.3f}")

        if it % args.eval_every == 0 or it == args.iters:
            row = {"iter": it, "loss": float(loss),
                   "train_ce_configs": sp.energy.by_tag.get("train", {}).get("configs", 0),
                   "sec": round(time.time() - t_start, 1)}
            if exact:
                sp.energy.tag = "eval"
                with torch.no_grad():
                    p = propagate_exact(
                        sp, lambda t: net_mult_all_states(sp, net, t), steps)
                row.update(sp.exact_report(p, f"it={it}"))
            hist.append(row)
            msg = (f"  it {it:5d}  loss {float(loss):11.4f}  "
                   f"CE {row['train_ce_configs']:>10,}  {row['sec']:.0f}s")
            if exact:
                msg += (f"  TV {row['TV']:.5f}  E-hist {row['energy_hist_TV']:.5f}"
                        f"  dE {row['mean_E_err_meV_per_atom']:.4f} meV/at")
                if row["TV"] < best["TV"]:
                    best = dict(row)
            print(msg, flush=True)

    # final sampling + gates
    sp.energy.tag = "eval"
    with torch.no_grad():
        Xs = simulate_direct(sp, net, args.n_samples, steps, generator=gen)
    viol = int((Xs.sum(1) != sp.k).sum())
    res = {"provenance": provenance(), "config": vars(args), "params": n_par,
           "history": hist, "violations": viol,
           "energy_calls": sp.energy.report()}
    if exact:
        with torch.no_grad():
            p = propagate_exact(
                sp, lambda t: net_mult_all_states(sp, net, t), steps)
        fin = sp.exact_report(p, "final")
        emp = sp.empirical_law(Xs)
        floor, floor_sd = sp.iid_tv_floor(args.n_samples)
        ctrl = ExactControl(sp, steps)
        errs = []
        with torch.no_grad():
            for s in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
                d = (net_mult_all_states(sp, net, s / steps)
                     - ctrl.all_states(s / steps)).abs()
                errs.append((s / steps, float(d.mean()), float(d.max())))
        res.update({"final": fin, "best": best, "mult_err": errs,
                    "empirical_TV": 0.5 * float((emp - sp.pi).abs().sum()),
                    "iid_TV_floor": floor, "iid_TV_floor_sd": floor_sd})
        print(f"\n  final exact-law TV {fin['TV']:.5f}  (best {best['TV']:.5f})"
              f"  empirical TV {res['empirical_TV']:.5f}  "
              f"iid floor {floor:.5f}")
        print("  learned vs oracle log-multiplier:")
        for t, me, mx in errs:
            print(f"    t={t:.3f}  mean|da|={me:.4f}  max={mx:.4f}")
    else:
        qs = sp.order_parameters(Xs)
        Es = sp.energy.energy_torch(Xs)
        res.update({"sample_mean_E_per_atom_meV": 1000 * float(Es.mean()) / sp.n,
                    "sample_mean_Qmax": float(qs.max(dim=1).values.mean())})

    out = args.out or (f"json/results_cuau_train_{sp.n}_{int(sp.temp_k)}K_"
                       f"g{int(sp.gamma)}_s{args.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  wrote {out}")
    if args.ckpt_dir:
        pth = C.save_ckpt(args.ckpt_dir, args.tag or f"cuau{sp.n}_s{args.seed}",
                          net=net, samples=Xs.to(torch.int8),
                          extra={"config": vars(args),
                                 "result": {k: v for k, v in res.items()
                                            if k != "provenance"}})
        print(f"  ckpt -> {pth}")
    print("\n=== GATES ===")
    print(f"T1  constraint violations 0 : {'PASS' if viol == 0 else 'FAIL'} ({viol})")
    if exact:
        ok = res["final"]["TV"] <= args.tv_gate
        print(f"T2  exact-law TV <= {args.tv_gate} : {'PASS' if ok else 'FAIL'} "
              f"({res['final']['TV']:.5f})")
        return 0 if (ok and viol == 0) else 1
    return 0 if viol == 0 else 1


def cmd_distill(args):
    """Capacity control: supervised regression of the net onto the exact Doob
    control, then the same exact-law evaluation used by ``train``.

    This isolates approximation capacity from the training objective.  The
    oracle log-multiplier ``a*(t, x, edge)`` is a closed form on CuAu-S, so a
    plain L2 fit over all ``|Omega| x steps`` pairs measures what the chosen
    architecture can represent at all.  If this reaches the oracle TV then the
    plateau seen by ``train`` is a property of the loss and its on-policy data,
    not of the parameter count.
    """
    from iasbs.fixed_ising import (ExactControl, SwapController,
                                   net_mult_all_states, propagate_exact)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = tuple(args.size)
    tables = build_ce_tables(size, verbose=True)
    energy = TorchCuAuEnergy(tables, device=args.device)
    sp = CuAuExact(size=size, temp_k=args.temp, gamma=args.gamma,
                   device=args.device, energy=energy, verbose=True)
    steps = args.steps
    ctrl = ExactControl(sp, steps)
    net = SwapController(sp.n, hidden=args.hidden).to(sp.device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=args.lr * 0.05)
    gen = torch.Generator(device=sp.device).manual_seed(args.seed + 99)

    print(f"CuAu distill  N={sp.n}  |Omega|={sp.M}  T={sp.temp_k} K  "
          f"gamma={sp.gamma}  steps={steps}  params={n_par}")

    hist, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        s_idx = int(torch.randint(steps, (1,), generator=None).item())
        t = s_idx / steps
        idx = torch.randint(sp.M, (args.mb,), device=sp.device, generator=gen)
        with torch.no_grad():
            tgt = ctrl.on_states(t, idx)                       # (mb, n_edges)
        x = sp.S[idx]
        tt = torch.full((idx.shape[0],), float(t), device=sp.device)
        a = net(tt, x)
        b = torch.arange(idx.shape[0], device=sp.device).unsqueeze(1)
        pred = a[b, sp.edge_i[idx], sp.edge_j[idx]].to(torch.float64)
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % args.eval_every == 0 or it == args.iters:
            with torch.no_grad():
                p = propagate_exact(
                    sp, lambda tq: net_mult_all_states(sp, net, tq), steps)
                rep = sp.exact_report(p, "distill")
                errs = []
                for s in (0, steps // 2, steps - 1):
                    d = (net_mult_all_states(sp, net, s / steps)
                         - ctrl.all_states(s / steps)).abs()
                    errs.append((s / steps, float(d.mean()), float(d.max())))
            row = {"it": it, "mse": float(loss), "TV": rep["TV"],
                   "mult_err": errs, "secs": round(time.time() - t0, 1)}
            hist.append(row)
            print(f"  it {it:6d}  mse {float(loss):10.5f}  "
                  f"TV {rep['TV']:.5f}  "
                  f"mean|da| t=0 {errs[0][1]:.4f} t=1 {errs[2][1]:.4f}  "
                  f"{row['secs']:.0f}s")

    with torch.no_grad():
        p = propagate_exact(sp, lambda tq: net_mult_all_states(sp, net, tq),
                            steps)
        fin = sp.exact_report(p, "final")
        p_or = propagate_exact(sp, ctrl.all_states, steps)
        orc = sp.exact_report(p_or, "oracle")
    res = {"provenance": provenance(), "config": vars(args), "params": n_par,
           "history": hist, "final": fin, "oracle": orc,
           "energy_calls": sp.energy.report()}
    out = args.out or (f"json/results_cuau_distill_{sp.n}_{int(sp.temp_k)}K_"
                       f"g{int(sp.gamma)}_s{args.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n  distilled TV {fin['TV']:.5f}   oracle TV {orc['TV']:.5f}")
    print(f"  wrote {out}")
    print("\n=== GATES ===")
    ok = fin["TV"] <= args.tv_gate
    print(f"D1  distilled TV <= {args.tv_gate} : {'PASS' if ok else 'FAIL'} "
          f"({fin['TV']:.5f})  [capacity {'sufficient' if ok else 'suspect'}]")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Phase 1 energy/geometry/symmetry gates")
    v.add_argument("--size", type=int, nargs=3, default=[4, 4, 4])
    v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    v.add_argument("--n-parity", type=int, default=200)
    v.add_argument("--n-calib", type=int, default=400)
    v.add_argument("--bench", type=int, default=4096)
    v.add_argument("--tol", type=float, default=1e-10)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--no-cache", action="store_true")
    v.set_defaults(fn=cmd_validate)

    e = sub.add_parser("exact", help="Phase 3 exact CuAu-S law + oracle sweep")
    e.add_argument("--size", type=int, nargs=3, default=[2, 2, 4])
    e.add_argument("--temp", type=float, default=500.0)
    e.add_argument("--gamma", type=float, default=10.0)
    e.add_argument("--steps-sweep", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512])
    e.add_argument("--n-samples", type=int, default=200_000)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--out", default="")
    e.set_defaults(fn=cmd_exact)

    d = sub.add_parser("distill", help="capacity control: fit the exact Doob control")
    d.add_argument("--size", type=int, nargs=3, default=[2, 2, 4])
    d.add_argument("--temp", type=float, default=500.0)
    d.add_argument("--gamma", type=float, default=10.0)
    d.add_argument("--steps", type=int, default=512)
    d.add_argument("--iters", type=int, default=4000)
    d.add_argument("--mb", type=int, default=512)
    d.add_argument("--hidden", type=int, default=512)
    d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--eval-every", type=int, default=250)
    d.add_argument("--tv-gate", type=float, default=0.05)
    d.add_argument("--device", default="cuda")
    d.add_argument("--out", default=None)
    d.set_defaults(fn=cmd_distill)

    t = sub.add_parser("train", help="IASBS training on CuAu")
    t.add_argument("--size", type=int, nargs=3, default=[2, 2, 4])
    t.add_argument("--temp", type=float, default=500.0)
    t.add_argument("--gamma", type=float, default=10.0)
    t.add_argument("--steps", type=int, default=128)
    t.add_argument("--iters", type=int, default=1000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--mb", type=int, default=256)
    t.add_argument("--inner", type=int, default=4)
    t.add_argument("--buffer", type=int, default=8)
    t.add_argument("--edge-samples", type=int, default=1)
    t.add_argument("--explore-frac", type=float, default=0.0,
                   help="fraction of each buffer push drawn from the zero "
                        "controller (base process) instead of the net")
    t.add_argument("--init-ckpt", default=None,
                   help="warm-start the controller from this checkpoint")
    t.add_argument("--all-edges", action="store_true",
                   help="exact edge average instead of sampled edges (control)")
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--loss", choices=["poisson", "l2"], default="poisson")
    t.add_argument("--source", choices=["random", "l10x"], default="random")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--n-samples", type=int, default=65_536)
    t.add_argument("--tv-gate", type=float, default=0.05)
    t.add_argument("--no-exact", action="store_true")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--ckpt-dir", default="ckpt")
    t.add_argument("--tag", default="")
    t.add_argument("--out", default="")
    t.set_defaults(fn=cmd_train)

    a = ap.parse_args(argv)
    C.use_repo_root()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
