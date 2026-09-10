"""Phase-2 test suite for the CuAu fixed-composition branch.

Groups follow section 30 of ``CUAU_IASBS_EXPERIMENT_PLAYBOOK.md``:

```text
A representation      B Johnson kernel   C bridge
D terminal label      E edge minibatch   F controlled simulation
J cost counter
```

Groups G (energy backend) and I (order parameter) are the hard gates of
``python -m iasbs.cuau validate`` and are not duplicated here; group H (exact
N=16 target) belongs to Phase 3.

Run with

```bash
/root/miniconda3/envs/cuau_env/bin/python -u -m unittest iasbs.tests_cuau -v
```
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402
from iasbs import cuau  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CE_SIZE = (2, 2, 4)          # N = 16, the cheap CE cell


class TinySpace:
    """Minimal space shim for the ``n <= 16`` bridge/label tests.

    Reuses :class:`cuau.CuAuSpace` methods that depend only on ``(n, k)`` so
    the tested code path is literally the production one.
    """

    _bridge_tables = cuau.CuAuSpace._bridge_tables
    distance = cuau.CuAuSpace.distance
    kappa_grid = cuau.CuAuSpace.kappa_grid
    bridge_kernels = cuau.CuAuSpace.bridge_kernels

    def __init__(self, n, k, gamma=10.0, tau=1.0, x0=None, energy=None,
                 device="cpu"):
        self.n, self.k = n, k
        self.gamma, self.tau = float(gamma), float(tau)
        self.device = torch.device(device)
        self.n_edges = k * (n - k)
        self.energy = energy
        kap = C.binary_orbit_kernel(n, k, self.gamma)
        self.kappa_full_np = kap
        self.log_kappa_full = torch.as_tensor(np.log(kap), dtype=torch.float64,
                                              device=self.device)
        if x0 is None:
            x0 = cuau.random_fixed_source(n, k, seed=7)
        self.x0 = torch.as_tensor(x0, device=self.device).long()
        self._bridge_tables()


def _ce_space(gamma=10.0, temp_k=500.0, source=None):
    tables = cuau.build_ce_tables(CE_SIZE, verbose=False)
    en = cuau.TorchCuAuEnergy(tables, device=DEV)
    return cuau.CuAuSpace(size=CE_SIZE, temp_k=temp_k, gamma=gamma,
                          source=source, device=DEV, energy=en)


# ============================================================================
# A. representation
# ============================================================================


class TestRepresentation(unittest.TestCase):
    def test_source_has_exactly_k_ones(self):
        for n, k in ((16, 8), (64, 32), (128, 64)):
            x = cuau.random_fixed_source(n, k)
            self.assertEqual(int(x.sum()), k)
            self.assertEqual(set(np.unique(x.numpy()).tolist()), {0, 1})
        # deterministic given the seed
        self.assertTrue(torch.equal(cuau.random_fixed_source(64, 32),
                                    cuau.random_fixed_source(64, 32)))

    def test_sampled_edges_are_legal_and_count_preserving(self):
        g = torch.Generator(device=DEV).manual_seed(0)
        n, k, B = 24, 12, 4096
        x = torch.stack([cuau.random_fixed_source(n, k, seed=s)
                         for s in range(B)]).to(DEV)
        i, j = cuau.sample_uniform_legal_edge(x, generator=g)
        b = torch.arange(B, device=DEV)
        self.assertTrue(bool((x[b, i] == 1).all()))
        self.assertTrue(bool((x[b, j] == 0).all()))
        y = cuau.transpose_batch(x, i, j)
        self.assertTrue(bool((y.sum(1) == k).all()))
        self.assertTrue(bool((y[b, i] == 0).all()))
        self.assertTrue(bool((y[b, j] == 1).all()))

    def test_legal_mask_has_k_times_nk_entries(self):
        n, k = 16, 8
        x = torch.stack([cuau.random_fixed_source(n, k, seed=s)
                         for s in range(32)]).to(DEV)
        m = cuau.legal_mask(x)
        self.assertTrue(bool((m.sum(dim=(1, 2)) == k * (n - k)).all()))

    def test_transpose_is_involutive(self):
        g = torch.Generator(device=DEV).manual_seed(1)
        B, n = 512, 20
        x = (torch.rand(B, n, device=DEV, generator=g) < 0.5).long()
        i = torch.randint(n, (B,), device=DEV, generator=g)
        j = torch.randint(n, (B,), device=DEV, generator=g)
        self.assertTrue(torch.equal(
            cuau.transpose_batch(cuau.transpose_batch(x, i, j), i, j), x))

    def test_transpose_is_identity_on_equal_symbols(self):
        B, n = 256, 12
        x = torch.zeros(B, n, dtype=torch.long, device=DEV)
        x[:, :6] = 1
        i = torch.zeros(B, dtype=torch.long, device=DEV)
        j = torch.ones(B, dtype=torch.long, device=DEV)      # both are 1
        self.assertTrue(torch.equal(cuau.transpose_batch(x, i, j), x))

    def test_check_states_raises(self):
        with self.assertRaises(RuntimeError):
            cuau.check_states(torch.tensor([[0, 1, 2, 0]]), 1)
        with self.assertRaises(RuntimeError):
            cuau.check_states(torch.tensor([[0, 1, 1, 0]]), 1)


# ============================================================================
# B. Johnson kernel
# ============================================================================


class TestJohnsonKernel(unittest.TestCase):
    def test_orbit_masses_sum_to_one(self):
        for n, k in ((10, 5), (16, 8), (64, 32)):
            for gam in (0.5, 10.0, 40.0):
                kap = C.binary_orbit_kernel(n, k, gam)
                tot = sum(math.comb(k, j) * math.comb(n - k, j) * kap[j]
                          for j in range(len(kap)))
                self.assertAlmostEqual(tot, 1.0, places=10)
                self.assertTrue(np.all(kap > 0.0))

    def test_distance_matches_common(self):
        rng = np.random.default_rng(0)
        n, k = 16, 8
        sp = TinySpace(n, k)
        x0 = sp.x0.numpy()
        for _ in range(200):
            p = rng.permutation(n)
            x = np.zeros(n, dtype=np.int64)
            x[p[:k]] = 1
            want = C.binary_orbit_distance(x0, x)
            got = int(sp.distance(sp.x0, torch.as_tensor(x)))
            self.assertEqual(want, got)


# ============================================================================
# C. bridge
# ============================================================================


def _exact_bridge_law(n, k, gamma, x0, x1, t):
    """Explicit ``p(x_t = x | x_0, x_1)`` over every enumerated state."""
    S = C.enumerate_fixed_count(n, k).astype(np.int64)
    k0 = np.maximum(C.binary_orbit_kernel(n, k, gamma * t), 1e-300)
    k1 = np.maximum(C.binary_orbit_kernel(n, k, gamma * (1.0 - t)), 1e-300)
    d0 = k - S @ x0
    d1 = k - S @ x1
    w = k0[d0] * k1[d1]
    return S, w / w.sum()


class TestBridge(unittest.TestCase):
    def test_direct_bridge_matches_explicit_formula(self):
        n, k, gamma, steps, M = 10, 5, 10.0, 10, 200_000
        sp = TinySpace(n, k, gamma=gamma, device=DEV)
        log_kap0, log_kap1 = sp.bridge_kernels(steps)
        rng = np.random.default_rng(3)
        g = torch.Generator(device=DEV).manual_seed(11)
        pow2 = (1 << np.arange(n)).astype(np.int64)

        for trial in range(3):
            p = rng.permutation(n)
            x1 = np.zeros(n, dtype=np.int64)
            x1[p[:k]] = 1
            X1 = torch.as_tensor(x1, device=DEV).expand(M, n).contiguous()
            for s_idx in (1, 3, 5, 8):
                t = s_idx / steps
                S, p_exact = _exact_bridge_law(n, k, gamma,
                                               sp.x0.cpu().numpy(), x1, t)
                lut = np.full(1 << n, -1, dtype=np.int64)
                lut[(S * pow2).sum(1)] = np.arange(S.shape[0])

                ti = torch.full((M,), s_idx, dtype=torch.long, device=DEV)
                xs = cuau.sample_bridge_direct(sp, X1, ti, log_kap0, log_kap1,
                                               generator=g)
                self.assertTrue(bool((xs.sum(1) == k).all()))
                idx = lut[(xs.cpu().numpy() * pow2).sum(1)]
                self.assertTrue(idx.min() >= 0)
                emp = np.bincount(idx, minlength=S.shape[0]) / M

                tv = 0.5 * np.abs(emp - p_exact).sum()
                # expected multinomial TV noise floor
                floor = float(np.sqrt(2.0 / np.pi)
                              * np.sqrt(p_exact * (1 - p_exact) / M).sum() * 0.5)
                self.assertLess(tv, max(6.0 * floor, 5e-3),
                                f"trial {trial} t={t} TV={tv:.5f} floor={floor:.5f}")

    def test_bridge_concentrates_at_endpoints(self):
        n, k, gamma, steps, M = 16, 8, 10.0, 1000, 4096
        sp = TinySpace(n, k, gamma=gamma, device=DEV)
        log_kap0, log_kap1 = sp.bridge_kernels(steps)
        x1 = cuau.random_fixed_source(n, k, seed=99).to(DEV)
        X1 = x1.expand(M, n).contiguous()
        g = torch.Generator(device=DEV).manual_seed(5)

        near0 = cuau.sample_bridge_direct(
            sp, X1, torch.full((M,), 1, dtype=torch.long, device=DEV),
            log_kap0, log_kap1, generator=g)
        near1 = cuau.sample_bridge_direct(
            sp, X1, torch.full((M,), steps - 1, dtype=torch.long, device=DEV),
            log_kap0, log_kap1, generator=g)

        d0_near0 = sp.distance(sp.x0.unsqueeze(0), near0).double().mean()
        d1_near0 = sp.distance(X1, near0).double().mean()
        d0_near1 = sp.distance(sp.x0.unsqueeze(0), near1).double().mean()
        d1_near1 = sp.distance(X1, near1).double().mean()
        self.assertLess(float(d0_near0), float(d1_near0))
        self.assertLess(float(d1_near1), float(d0_near1))
        self.assertLess(float(d0_near0), 1.0)
        self.assertLess(float(d1_near1), 1.0)


# ============================================================================
# D. terminal label
# ============================================================================


class TestTerminalLabel(unittest.TestCase):
    def test_edge_label_matches_common_binary_as_label(self):
        sp = _ce_space()
        n, k = sp.n, sp.k
        rng = np.random.default_rng(4)
        g = torch.Generator(device=DEV).manual_seed(2)
        B = 256
        xs = []
        for _ in range(B):
            p = rng.permutation(n)
            x = np.zeros(n, dtype=np.int64)
            x[p[:k]] = 1
            xs.append(x)
        X1 = torch.as_tensor(np.stack(xs), device=DEV)
        E1 = sp.energy.energy_torch(X1).to(torch.float64)
        i, j = cuau.sample_uniform_legal_edge(X1, generator=g)
        got = cuau.terminal_log_label_edge(sp, X1, E1, i, j, check=True)

        kap = np.exp(sp.log_kappa_full.cpu().numpy())
        x0 = sp.x0.cpu().numpy()
        efn = lambda z: float(sp.energy.energy_np(z[None, :])[0])
        for b in range(0, B, 17):
            x1 = X1[b].cpu().numpy()
            xg = x1.copy()
            a, c = int(i[b]), int(j[b])
            xg[a], xg[c] = x1[c], x1[a]
            want = C.binary_as_label(x1, xg, x0, efn, sp.tau, kap)
            self.assertAlmostEqual(float(torch.exp(got[b])) / want, 1.0,
                                   places=8)

    def test_matches_fixed_ising_all_pairs(self):
        """Same formula as the enumerated Ising branch, on the Ising energy."""
        from iasbs.fixed_ising import FixedIsingSpace, terminal_labels

        L, tau, gamma = 3, 2.0, 10.0        # n = 9 -> k = 4
        isp = FixedIsingSpace(L=L, tau=tau, gamma=gamma, device=DEV)
        n, k = isp.n, isp.k

        class _Ising:
            N = n

            @staticmethod
            def energy_torch(x):
                return isp.energies_of(x)

        sp = TinySpace(n, k, gamma=gamma, tau=tau, x0=isp.x0.cpu(),
                       energy=_Ising, device=DEV)
        idx = torch.arange(0, isp.M, max(1, isp.M // 64), device=DEV)
        want = terminal_labels(isp, idx)                      # (B, Np)
        X1 = isp.S[idx]
        E1 = isp.E[idx]
        got = cuau.terminal_log_label_all_pairs(sp, X1, E1, isp.pairs)
        self.assertLess(float((got - want).abs().max()), 1e-9)

    def test_equal_symbol_transposition_gives_unit_label(self):
        sp = _ce_space()
        n, k, B = sp.n, sp.k, 128
        X1 = torch.stack([cuau.random_fixed_source(n, k, seed=s)
                          for s in range(B)]).to(DEV)
        E1 = sp.energy.energy_torch(X1).to(torch.float64)
        au = [int(torch.nonzero(X1[r]).squeeze(1)[0]) for r in range(B)]
        au2 = [int(torch.nonzero(X1[r]).squeeze(1)[1]) for r in range(B)]
        i = torch.as_tensor(au, device=DEV)
        j = torch.as_tensor(au2, device=DEV)
        out = cuau.terminal_log_label_edge(sp, X1, E1, i, j, check=True)
        self.assertLess(float(out.abs().max()), 1e-10)


# ============================================================================
# E. edge minibatching
# ============================================================================


class TestEdgeMinibatch(unittest.TestCase):
    def test_sampled_edge_loss_matches_all_edge_loss(self):
        torch.manual_seed(0)
        n, k, B, M = 16, 8, 64, 20000
        x = torch.stack([cuau.random_fixed_source(n, k, seed=s)
                         for s in range(B)]).to(DEV)
        mask = cuau.legal_mask(x)
        a = torch.randn(B, n, n, dtype=torch.float64, device=DEV) * 0.5
        lam = torch.exp(torch.randn(B, n, n, dtype=torch.float64, device=DEV) * 0.3)
        full = float(cuau.all_edge_poisson_loss(a, lam, mask))

        g = torch.Generator(device=DEV).manual_seed(3)
        vals = []
        bb = torch.arange(B, device=DEV)
        for _ in range(M // B):
            i, j = cuau.sample_uniform_legal_edge(x, generator=g)
            av = a[bb, i, j]
            lv = lam[bb, i, j]
            vals.append(torch.exp(av) - av * lv)
        v = torch.cat(vals)
        mc = float(v.mean())
        se = float(v.std(unbiased=True) / math.sqrt(v.numel()))
        z = (mc - full) / se
        self.assertLess(abs(z), 4.0, f"full={full:.6f} mc={mc:.6f} z={z:.3f}")

    def test_edge_sampler_is_uniform(self):
        n, k = 8, 4
        x = torch.zeros(1, n, dtype=torch.long, device=DEV)
        x[0, :k] = 1
        M = 200_000
        xb = x.expand(M, n).contiguous()
        g = torch.Generator(device=DEV).manual_seed(9)
        i, j = cuau.sample_uniform_legal_edge(xb, generator=g)
        e = (i * n + j).cpu().numpy()
        cnt = np.bincount(e, minlength=n * n)
        legal = cnt[cnt > 0]
        self.assertEqual(legal.size, k * (n - k))
        exp = M / (k * (n - k))
        chi2 = float(((legal - exp) ** 2 / exp).sum())
        # dof = 15; 1e-6 upper tail is ~ 60
        self.assertLess(chi2, 60.0, f"chi2={chi2:.2f}")


# ============================================================================
# F. controlled simulation
# ============================================================================


class TestSimulation(unittest.TestCase):
    def test_zero_control_reproduces_kappa_orbit_masses(self):
        n, k, gamma = 16, 8, 10.0
        sp = _ce_space(gamma=gamma)
        self.assertEqual(sp.n, n)
        M, steps = 65536, 512
        g = torch.Generator(device=DEV).manual_seed(17)
        x = cuau.simulate_direct(sp, None, M, steps, generator=g)
        self.assertTrue(bool((x.sum(1) == k).all()))
        d = sp.distance(sp.x0.unsqueeze(0), x).cpu().numpy()
        emp = np.bincount(d, minlength=k + 1) / M

        kap = C.binary_orbit_kernel(n, k, gamma)
        want = np.array([math.comb(k, jj) * math.comb(n - k, jj) * kap[jj]
                         for jj in range(k + 1)])
        tv = 0.5 * np.abs(emp - want).sum()
        self.assertLess(tv, 0.02, f"orbit-mass TV={tv:.4f}\nemp={emp}\nref={want}")

    def test_zero_control_small_system_matches_exact_propagation(self):
        """``n = 10`` against the exact enumerated single-swap semigroup."""
        n, k, gamma, steps, M = 10, 5, 10.0, 512, 65536
        sp = TinySpace(n, k, gamma=gamma, device=DEV)
        sp.energy = None
        x = cuau.simulate_direct(sp, None, M, steps,
                                 generator=torch.Generator(device=DEV).manual_seed(21))
        S = C.enumerate_fixed_count(n, k).astype(np.int64)
        pow2 = (1 << np.arange(n)).astype(np.int64)
        lut = np.full(1 << n, -1, dtype=np.int64)
        lut[(S * pow2).sum(1)] = np.arange(S.shape[0])
        idx = lut[(x.cpu().numpy() * pow2).sum(1)]
        emp = np.bincount(idx, minlength=S.shape[0]) / M

        kap = C.binary_orbit_kernel(n, k, gamma)
        d = k - S @ sp.x0.cpu().numpy()
        want = kap[d]
        self.assertAlmostEqual(float(want.sum()), 1.0, places=10)
        tv = 0.5 * np.abs(emp - want).sum()
        floor = float(np.sqrt(2.0 / np.pi)
                      * np.sqrt(want * (1 - want) / M).sum() * 0.5)
        self.assertLess(tv, max(6.0 * floor, 0.02),
                        f"TV={tv:.5f} floor={floor:.5f}")

    def test_controlled_simulation_preserves_count(self):
        from iasbs.fixed_ising import SwapController
        sp = _ce_space()
        net = SwapController(sp.n, hidden=64).to(DEV)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        x = cuau.simulate_direct(sp, net, 512, 64,
                                 generator=torch.Generator(device=DEV).manual_seed(1))
        self.assertTrue(bool((x.sum(1) == sp.k).all()))
        self.assertEqual(set(torch.unique(x).cpu().numpy().tolist()) - {0, 1}, set())


# ============================================================================
# J. cost counter
# ============================================================================


class TestCostCounter(unittest.TestCase):
    def test_batch_increments_by_batch_size(self):
        tables = cuau.build_ce_tables(CE_SIZE, verbose=False)
        en = cuau.CountedEnergy(cuau.TorchCuAuEnergy(tables, device=DEV))
        x = torch.stack([cuau.random_fixed_source(16, 8, seed=s)
                         for s in range(37)]).to(DEV)
        en.energy_torch(x)
        self.assertEqual(en.n_configs, 37)
        self.assertEqual(en.n_calls, 1)
        en.tag = "eval"
        en.energy_torch(x[:5])
        self.assertEqual(en.n_configs, 42)
        rep = en.report()
        self.assertEqual(rep["by_tag"]["train"]["configs"], 37)
        self.assertEqual(rep["by_tag"]["eval"]["configs"], 5)

    def test_cached_terminal_energy_costs_one_call_per_edge(self):
        sp = _ce_space()
        B = 64
        X1 = torch.stack([cuau.random_fixed_source(sp.n, sp.k, seed=s)
                          for s in range(B)]).to(DEV)
        E1 = sp.energy.energy_torch(X1).to(torch.float64)
        before = sp.energy.n_configs
        g = torch.Generator(device=DEV).manual_seed(0)
        i, j = cuau.sample_uniform_legal_edge(X1, generator=g)
        cuau.terminal_log_label_edge(sp, X1, E1, i, j)
        self.assertEqual(sp.energy.n_configs - before, B)


if __name__ == "__main__":
    unittest.main(verbosity=2)
