"""Reviewer weakness 5, part 2: genuinely independent Stiefel references.

The shipped frame reference ``ckpt/stiefel_frame_s5_b1_mcmc.pt`` and
``ckpt/stiefel_frame600_b1_mcmc.pt`` are byte-identical -- both runs called
``mcmc_stiefel`` after ``torch.manual_seed(0)`` with the same chain count,
sweep count and proposal size, so the "cross-reference" MMD reported in the
frame-law audit was really a second split-half of one sample.  This script
produces two further references that are independent in every respect that
matters:

  refB  same sampler configuration, a different seed -> independent Haar
        initialisation and independent proposal/acceptance randomness.
        refA vs refB isolates Monte-Carlo variability at fixed sampler.

  refC  a different proposal scale (eps 0.20 instead of 0.35) and twice the
        sweeps on half the chains, from yet another seed.  A random-walk
        Metropolis chain that has equilibrated must be eps-independent, so
        refA/refB vs refC is the diagnostic for residual MCMC bias rather
        than sampling noise.

Usage:  python iasbs/_weakness5_ref.py <refB|refC>
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as C                                              # noqa: E402
import stiefel as ST                                            # noqa: E402

CFG = {
    # name:  (seed,     nchain,  nsweep, eps)
    "refB": (20260911, 200000, 3000, 0.35),
    "refC": (777, 100000, 6000, 0.20),
}


def main():
    name = sys.argv[1]
    seed, nchain, nsweep, eps = CFG[name]
    dev = "cuda:0"
    prob = ST.StiefelProblem(sigma=2.0 ** 0.5, beta=1.0, steps=199, nq=64,
                             frame=True, device=dev)
    torch.manual_seed(seed)
    t0 = time.time()
    X = ST.mcmc_stiefel(prob, nchain=nchain, nsweep=nsweep, eps=eps,
                        verbose=True)
    m = ST.energy_metrics(prob, X)
    m["wall_s"] = time.time() - t0
    print(name, m, flush=True)
    C.save_ckpt("ckpt", f"stiefel_frame_b1_mcmc_{name}",
                samples=X[:100000].cpu(),
                extra={"metrics": m, "beta": 1.0,
                       "seed": seed, "nchain": nchain,
                       "nsweep": nsweep, "eps": eps})
    print("saved", flush=True)


if __name__ == "__main__":
    main()
