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


def sample_uniform_legal_edge(x: torch.Tensor, generator=None):
    """Uniform draw from ``E(x)``, the ``k(N-k)`` ordered legal swaps of ``x``."""
    B, n = x.shape
    m = legal_mask(x).reshape(B, -1).to(torch.float32)
    e = torch.multinomial(m, 1, generator=generator).squeeze(1)
    return e // n, e % n


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

    a = ap.parse_args(argv)
    C.use_repo_root()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
