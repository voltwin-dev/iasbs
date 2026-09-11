"""Canonical parallel-tempering reference for the CuAu cluster expansion.

Phase 4 of the CuAu playbook.  This module produces the *reference* numbers the
larger sectors are scored against, where exact enumeration is impossible
(N = 64 has C(64,32) = 1.8e18 states).  It is deliberately independent of the
IASBS machinery: fixed-composition swap Metropolis with parallel tempering, and
nothing else.

What it reports, and why each piece is needed before the number is usable:

* ``<E>/N``, ``<Qmax>``, ``Cv/N`` at the target temperature -- the observables.
* swap acceptance per adjacent pair -- a ladder with a dead rung transports
  nothing, and the run must be rejected rather than averaged.
* round trips per walker -- the actual mixing certificate for PT; a ladder can
  have healthy pairwise acceptance and still not carry walkers end to end.
* integrated autocorrelation time and ESS at the target -- turns a raw sample
  count into an honest error bar.
* split-half and between-chain spread over independent starts -- the empirical
  noise floor.  Any claimed difference smaller than this is not a difference.

The N = 16 sector is exactly enumerable, so ``--size 2 2 4`` doubles as a hard
correctness gate on this sampler before it is trusted at N = 64.

CLI
---
    python -m iasbs.cuau_reference --size 4 4 4 --temp 500 \
        --tmax 2500 --n-temps 24 --chains 4 --steps 400000 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from iasbs import cuau as CU


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------
def temperature_ladder(t_target, t_max, n_temps):
    """Geometric ladder whose *first* rung is exactly the target temperature.

    Geometric spacing keeps ``1/tau`` gaps roughly uniform, which is what sets
    the pairwise swap acceptance; putting the target at index 0 means the
    reported observables come from an unmodified rung rather than an
    interpolation.
    """
    if t_max <= t_target or n_temps < 2:
        raise ValueError("need t_max > t_target and n_temps >= 2")
    return np.geomspace(float(t_target), float(t_max), int(n_temps))


# ---------------------------------------------------------------------------
# moves
# ---------------------------------------------------------------------------
def propose_swap(x, generator=None):
    """Uniform legal transposition per row of ``x`` (B, N), composition fixed.

    ``x + U[0,1)`` sorts every occupied site above every empty one while
    randomising the order *within* each group, so taking the extreme column of
    each sort draws a uniform occupied site and a uniform empty site in two
    kernels and with no per-row masking.
    """
    r = torch.rand(x.shape, device=x.device, dtype=torch.float64,
                   generator=generator)
    key = x.to(torch.float64) + r
    order = key.argsort(dim=-1)
    j = order[:, 0]                      # uniform empty site  (x = 0)
    i = order[:, -1]                     # uniform occupied site (x = 1)
    return i, j


def apply_swap(x, i, j, accept):
    """Transpose sites ``i``/``j`` on the rows where ``accept`` is true."""
    b = torch.arange(x.shape[0], device=x.device)
    xi = x[b, i].clone()
    xj = x[b, j].clone()
    acc = accept.to(x.dtype)
    x[b, i] = xi * (1 - acc) + xj * acc
    x[b, j] = xj * (1 - acc) + xi * acc
    return x


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def integrated_act(series, c=6.0):
    """Integrated autocorrelation time with Sokal's automatic window.

    Returns ``tau_int`` in units of recorded samples.  ``ESS = n / (2 tau_int)``
    is the number of effectively independent draws behind the mean.
    """
    y = np.asarray(series, dtype=np.float64)
    n = y.size
    y = y - y.mean()
    v = float((y * y).mean())
    if n < 16 or v <= 0.0:
        return float("nan"), float("nan")
    f = np.fft.rfft(y, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * max(tau, 0.5):
            break
        if tau <= 0.0:
            tau = 0.5
            break
    tau = max(float(tau), 0.5)
    return tau, float(n / (2.0 * tau))


def round_trip_counts(hist_slot, n_temps):
    """Round trips per walker from its temperature-slot trajectory.

    A round trip is a full traversal of the ladder: bottom rung to top rung and
    back.  This is the mixing statement that matters -- pairwise acceptance can
    look healthy while no walker ever crosses the whole ladder.
    """
    hist = np.asarray(hist_slot)          # (T, n_walkers)
    trips = np.zeros(hist.shape[1], dtype=np.int64)
    state = np.zeros(hist.shape[1], dtype=np.int64)   # 0 none, 1 at top, -1 bot
    for row in hist:
        at_bot = row == 0
        at_top = row == n_temps - 1
        trips += (at_bot & (state == 1)).astype(np.int64)
        state = np.where(at_top, 1, np.where(at_bot, -1, state))
    return trips


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_pt(args):
    size = tuple(int(v) for v in args.size)
    dev = torch.device(args.device)
    tables = CU.build_ce_tables(size, verbose=False)
    energy = CU.TorchCuAuEnergy(tables, device=args.device)
    n = int(energy.N)
    k = n // 2

    temps = temperature_ladder(args.temp, args.tmax, args.n_temps)
    beta = torch.as_tensor(1.0 / (CU.KB_EV_PER_K * temps), dtype=torch.float64,
                           device=dev)                       # (R,)
    R = temps.size
    C = int(args.chains)

    # L10 phase tables for Qmax
    q = CU.l10_q_vectors(float(energy.a))
    ph = np.asarray(energy.positions, dtype=np.float64) @ q.T
    cosp = torch.as_tensor(np.cos(ph), dtype=torch.float64, device=dev)
    sinp = torch.as_tensor(np.sin(ph), dtype=torch.float64, device=dev)

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(args.seed))

    # Independent starts: every chain starts from its own random configuration
    # at every rung, so agreement between chains is evidence and not an artefact
    # of a shared initial condition.
    r0 = torch.rand((C * R, n), device=dev, dtype=torch.float64, generator=gen)
    x = torch.zeros((C * R, n), dtype=torch.int64, device=dev)
    x.scatter_(1, r0.argsort(dim=1)[:, :k], 1)
    E = energy.energy_torch(x)                                # (C*R,)

    bflat = beta.repeat(C)                                    # (C*R,)
    wid = torch.arange(R, device=dev).repeat(C)               # walker at slot
    slot_hist = []
    x_pool = []
    e_series, q_series = [], []
    acc_move = torch.zeros((), dtype=torch.float64, device=dev)
    n_move = 0
    acc_pair = torch.zeros(R - 1, dtype=torch.float64, device=dev)
    n_pair = torch.zeros(R - 1, dtype=torch.float64, device=dev)

    print(f"[PT cuau] size={size}  N={n}  k={k}  rungs={R} "
          f"[{temps[0]:.0f} .. {temps[-1]:.0f} K]  chains={C}  "
          f"steps={args.steps}  device={dev}", flush=True)
    t0 = time.time()

    for step in range(1, int(args.steps) + 1):
        i, j = propose_swap(x, generator=gen)
        xp = x.clone()
        b = torch.arange(x.shape[0], device=dev)
        xi, xj = xp[b, i].clone(), xp[b, j].clone()
        xp[b, i], xp[b, j] = xj, xi
        Ep = energy.energy_torch(xp)
        d = -bflat * (Ep - E)
        u = torch.rand(d.shape, device=dev, dtype=torch.float64, generator=gen)
        acc = torch.log(u) < d
        x = torch.where(acc.unsqueeze(1), xp, x)
        E = torch.where(acc, Ep, E)
        acc_move += acc.to(torch.float64).sum()
        n_move += acc.numel()

        if step % args.swap_every == 0:
            # Alternate even/odd adjacent pairs -- a fixed parity every sweep
            # would leave half the rungs permanently uncoupled.
            off = (step // args.swap_every) % 2
            idx = torch.arange(off, R - 1, 2, device=dev)
            if idx.numel():
                Em = E.view(C, R)
                lo = Em[:, idx]
                hi = Em[:, idx + 1]
                dd = (beta[idx] - beta[idx + 1]) * (lo - hi)
                uu = torch.rand(dd.shape, device=dev, dtype=torch.float64,
                                generator=gen)
                sw = torch.log(uu) < dd
                acc_pair.index_add_(0, idx, sw.to(torch.float64).sum(0))
                n_pair.index_add_(0, idx,
                                  torch.full((idx.numel(),), float(C),
                                             dtype=torch.float64, device=dev))
                xm = x.view(C, R, n)
                wm = wid.view(C, R)
                for a, s in zip(idx.tolist(), sw.t()):
                    m = s
                    if not bool(m.any()):
                        continue
                    xa = xm[:, a].clone()
                    xm[:, a] = torch.where(m.unsqueeze(1), xm[:, a + 1], xm[:, a])
                    xm[:, a + 1] = torch.where(m.unsqueeze(1), xa, xm[:, a + 1])
                    ea = Em[:, a].clone()
                    Em[:, a] = torch.where(m, Em[:, a + 1], Em[:, a])
                    Em[:, a + 1] = torch.where(m, ea, Em[:, a + 1])
                    wa = wm[:, a].clone()
                    wm[:, a] = torch.where(m, wm[:, a + 1], wm[:, a])
                    wm[:, a + 1] = torch.where(m, wa, wm[:, a + 1])
                x = xm.reshape(C * R, n)
                E = Em.reshape(C * R)
                wid = wm.reshape(C * R)

        if step > args.burn and step % args.record_every == 0:
            xt = x.view(C, R, n)[:, 0]                # target rung, all chains
            Qm = CU.l10_order_parameters_torch(xt, cosp, sinp).max(-1).values
            e_series.append(E.view(C, R)[:, 0].cpu().numpy().copy())
            q_series.append(Qm.cpu().numpy().copy())
            wm = wid.view(C, R)
            sl = np.zeros((C, R), dtype=np.int64)
            wcpu = wm.cpu().numpy()
            for c in range(C):
                sl[c, wcpu[c]] = np.arange(R)
            slot_hist.append(sl)
            if getattr(args, "save_samples", ""):
                x_pool.append(xt.to(torch.int8).cpu().numpy().copy())

        if step % args.log_every == 0:
            em = float(E.view(C, R)[:, 0].mean()) / n * 1000.0
            print(f"  step {step:8d}  <E>/N {em:9.4f} meV  "
                  f"move-acc {float(acc_move) / max(n_move, 1):.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    e_arr = np.asarray(e_series)                      # (T, C)
    q_arr = np.asarray(q_series)
    slot_arr = np.asarray(slot_hist)                  # (T, C, R)
    T = e_arr.shape[0]
    if T < 16:
        raise RuntimeError("too few recorded samples; raise --steps")

    tau = CU.KB_EV_PER_K * float(temps[0])
    per_chain = []
    for c in range(C):
        ec, qc = e_arr[:, c], q_arr[:, c]
        t_int, ess = integrated_act(ec)
        cv = float(ec.var(ddof=1)) / (tau * float(temps[0])) / n
        per_chain.append({
            "chain": c,
            "E_per_N_meV": float(ec.mean()) / n * 1000.0,
            "Qmax": float(qc.mean()),
            "Cv_per_N": cv,
            "tau_int": t_int,
            "ESS": ess,
            "half1_E_per_N_meV": float(ec[: T // 2].mean()) / n * 1000.0,
            "half2_E_per_N_meV": float(ec[T // 2:].mean()) / n * 1000.0,
            "half1_Qmax": float(qc[: T // 2].mean()),
            "half2_Qmax": float(qc[T // 2:].mean()),
            "round_trips": round_trip_counts(slot_arr[:, c], R).tolist(),
        })

    e_chain = np.array([r["E_per_N_meV"] for r in per_chain])
    q_chain = np.array([r["Qmax"] for r in per_chain])
    split = np.array([abs(r["half1_E_per_N_meV"] - r["half2_E_per_N_meV"])
                      for r in per_chain])
    trips = int(sum(sum(r["round_trips"]) for r in per_chain))
    pacc = (acc_pair / n_pair.clamp(min=1.0)).cpu().numpy()

    out = {
        "run": "cuau_pt_reference",
        "size": list(size), "N": n, "k": k,
        "temp_target_K": float(temps[0]),
        "ladder_K": temps.tolist(),
        "chains": C, "steps": int(args.steps), "burn": int(args.burn),
        "record_every": int(args.record_every),
        "swap_every": int(args.swap_every),
        "samples_per_chain": int(T),
        "seed": int(args.seed),
        "E_per_N_meV": float(e_chain.mean()),
        "E_per_N_meV_sd": float(e_chain.std(ddof=1)) if C > 1 else 0.0,
        "Qmax": float(q_chain.mean()),
        "Qmax_sd": float(q_chain.std(ddof=1)) if C > 1 else 0.0,
        "Cv_per_N": float(np.mean([r["Cv_per_N"] for r in per_chain])),
        "tau_int_mean": float(np.nanmean([r["tau_int"] for r in per_chain])),
        "ESS_mean": float(np.nanmean([r["ESS"] for r in per_chain])),
        "split_half_dE_max_meV": float(split.max()),
        "round_trips_total": trips,
        "round_trips_min_per_walker": int(min(
            min(r["round_trips"]) for r in per_chain)),
        "swap_acc_min": float(pacc.min()),
        "swap_acc_mean": float(pacc.mean()),
        "swap_acc_per_pair": pacc.tolist(),
        "move_acc": float(acc_move) / max(n_move, 1),
        "energy_calls": int(energy.n_calls),
        "wall_s": time.time() - t0,
        "per_chain": per_chain,
        "provenance": CU.provenance(),
    }

    # Gates.  These are the conditions under which the numbers above are
    # allowed to be used as a reference at all.
    gates = {
        "R1 min pairwise swap acceptance >= 0.20":
            bool(pacc.min() >= 0.20),
        "R2 every walker >= 1 round trip":
            bool(out["round_trips_min_per_walker"] >= 1),
        "R3 between-chain sd <= 0.5 meV/atom":
            bool(out["E_per_N_meV_sd"] <= 0.5),
        "R4 split-half drift <= 0.5 meV/atom":
            bool(split.max() <= 0.5),
    }
    out["gates"] = gates

    print("\n  " + "-" * 62)
    print(f"  <E>/N   {out['E_per_N_meV']:.4f} +/- {out['E_per_N_meV_sd']:.4f}"
          f" meV/atom   ({C} independent chains)")
    print(f"  <Qmax>  {out['Qmax']:.4f} +/- {out['Qmax_sd']:.4f}")
    print(f"  Cv/N    {out['Cv_per_N']:.6e} eV/K")
    print(f"  tau_int {out['tau_int_mean']:.2f} samples   "
          f"ESS {out['ESS_mean']:.0f} / {T}")
    print(f"  swap acc  min {pacc.min():.3f}  mean {pacc.mean():.3f}   "
          f"move acc {out['move_acc']:.3f}")
    print(f"  round trips {trips} total, min per walker "
          f"{out['round_trips_min_per_walker']}")
    print(f"  split-half |dE|/N  max {split.max():.4f} meV")
    print(f"  energy calls {out['energy_calls']:,}   "
          f"wall {out['wall_s']:.0f}s")
    print("\n=== GATES ===")
    for kk, vv in gates.items():
        print(f"{kk} : {'PASS' if vv else 'FAIL'}")

    if getattr(args, "save_samples", ""):
        os.makedirs(os.path.dirname(args.save_samples) or ".", exist_ok=True)
        X = np.concatenate(x_pool, axis=0)             # (T*C, N) target rung
        np.savez_compressed(args.save_samples, X=X, E=e_arr.reshape(-1),
                            temp_K=float(temps[0]), size=np.array(size),
                            record_every=int(args.record_every))
        print(f"\n  wrote {args.save_samples}  pool {X.shape}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, nargs=3, default=[4, 4, 4])
    p.add_argument("--temp", type=float, default=500.0,
                   help="target temperature; becomes rung 0")
    p.add_argument("--tmax", type=float, default=2500.0)
    p.add_argument("--n-temps", type=int, default=24)
    p.add_argument("--chains", type=int, default=4,
                   help="independent PT ensembles (independent starts)")
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--burn", type=int, default=50000)
    p.add_argument("--swap-every", type=int, default=10)
    p.add_argument("--record-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="")
    p.add_argument("--save-samples", default="",
                   help="npz pool of target-rung configurations (playbook 31)")
    a = p.parse_args(argv)
    if not a.out:
        sz = "x".join(str(v) for v in a.size)
        a.out = f"json/results_cuau_pt_{sz}_{int(a.temp)}K.json"
    run_pt(a)


if __name__ == "__main__":
    main()
