#!/usr/bin/env python
"""Falsification tests for the bimodal collapse reported by `rasbs_sphere_port.py`.

The port reproduces `asbs_sphere_sampler.m` line for line and yet puts ~100% of
its terminal mass on a single pole, where their paper's Section 4.1 reports
43.8% in the northern hemisphere.  Two explanations remain: the collapse is a
real property of their algorithm, or the port has a defect that breaks the
x_3 -> -x_3 symmetry.  A line-by-line reading cannot separate those, so this
script drives the *same* code path with targets whose answer is known in closed
form.  Nothing here is a new algorithm -- every function is imported from the
port, so a defect in the port is a defect in these tests too.

  uniform   E(x) = 0.  The target is Haar.  There is no energy gradient at all,
            so the only thing driving the terminal law is the corrector h and
            the geodesic random walk.  Haar is stationary under both.  If this
            run collapses, the instability is in the machinery and the bimodal
            result says nothing about their energy.  If it stays uniform, the
            machinery preserves a symmetric answer when the target is symmetric.

  vmf       E(x) = -6 x_3.  A single mode at the north pole, no symmetry to
            break, and the z-marginal is exactly p(z) ∝ exp(6 z) on [-1, 1]:

                north mass = (e^6 - 1) / (e^6 - e^-6)  = 0.9975274...
                <x_3>      = coth(6) - 1/6             = 0.8333505...

            This is the same 6 and the same energy scale as their bimodal
            target -- only the square is dropped.  If the port recovers these
            two numbers it integrates, regresses and transports correctly at
            this noise level, and the bimodal failure is specific to having two
            modes rather than generic to the port.

Neither test can prove the port correct.  Together they can make "the port has
a symmetry-breaking bug" a much more expensive hypothesis to hold.
"""
import argparse
import json
import math
import os as _os
import sys as _sys

import torch

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import common as C
import rasbs_sphere_port as P


class Zero:
    """E(x) = 0.  Target is Haar."""

    name = "bimodal"                      # selects their bimodal code path

    def grad_riem(self, x, kappa=None):
        return torch.zeros_like(x)

    def energy(self, x):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class VMF:
    """E(x) = -k x_3, i.e. pi ∝ exp(k x_3): one mode, no symmetry to break."""

    name = "bimodal"

    def __init__(self, k=6.0):
        self.k = k

    def grad_riem(self, x, kappa=None):
        g = torch.zeros_like(x)
        g[:, 2] = -self.k
        return P.proj(x, g)

    def energy(self, x):
        return -self.k * x[:, 2]


def exact_vmf(k):
    """north mass and <x_3> for p(z) ∝ exp(k z) on [-1, 1]."""
    north = (math.exp(k) - 1.0) / (math.exp(k) - math.exp(-k))
    mean = 1.0 / math.tanh(k) - 1.0 / k
    return north, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["uniform", "vmf"], default="uniform")
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--n-samples", dest="n_samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", dest="log_every", type=int, default=100)
    ap.add_argument("--max-drift", dest="max_drift", type=float, default=0.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--n-freq", dest="n_freq", type=int, default=256)
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    C.use_repo_root()
    a.tag = a.tag or f"rasbs_sphere_audit_{a.test}"
    a.out = a.out or f"json/results_rasbs_sphere_audit_{a.test}.json"

    tgt = Zero() if a.test == "uniform" else VMF(a.k)
    print(f"R-ASBS port audit: {a.test}", flush=True)
    unet, rff_u, sched, hist, n_par = P.train(a, tgt, P.DEV)

    print(f"\n  sampling {a.n_samples:,} terminal points ...\n", flush=True)
    gen = torch.Generator(device=P.DEV).manual_seed(a.seed + 12345)
    _, x = P.simulate(unet, rff_u, sched, a.steps, a.n_samples, P.DEV, gen,
                      a.max_drift)
    z = x[:, 2]
    north = float((z > 0).double().mean())
    mz = float(z.mean())

    if a.test == "uniform":
        # Haar on S^2 has z ~ Uniform[-1, 1]; KS against that is exact.
        zs = torch.sort(z).values
        n = zs.numel()
        cdf = (zs + 1.0) / 2.0
        i = torch.arange(1, n + 1, device=zs.device, dtype=zs.dtype)
        ks = float(torch.maximum(i / n - cdf, cdf - (i - 1) / n).max())
        e_north, e_mean, e_ks = 0.5, 0.0, 0.0
        extra = {"ks_uniform": ks}
        print(f"  north mass   {north:.4f}   (exact 0.5)")
        print(f"  <x_3>        {mz:+.5f}  (exact 0.00000)")
        print(f"  KS(x_3) vs U[-1,1]  {ks:.5f}   "
              f"(iid floor {1.36 / math.sqrt(a.n_samples):.5f})")
    else:
        e_north, e_mean = exact_vmf(a.k)
        extra = {}
        print(f"  north mass   {north:.4f}   (exact {e_north:.4f})")
        print(f"  <x_3>        {mz:+.5f}  (exact {e_mean:+.5f})")

    res = {"test": a.test, "k": a.k, "epochs": a.epochs, "steps": a.steps,
           "seed": a.seed, "n_samples": a.n_samples, "north": north,
           "mean_z": mz, "exact_north": e_north, "exact_mean_z": e_mean,
           "n_params": n_par, "history": hist, **extra}
    _os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
