"""KS(phi) and |Delta E[x_3^2]| for the repaired (--init matlab) R-ASBS seeds.

The R-ASBS audit script recorded north mass, KS(x_3) and W1(x_3) but not the
azimuthal-uniformity statistic nor the second-moment error.  Both are functions
of the stored samples alone, so they can be recovered from the checkpoints
without retraining.  Reuses remeasure.sphere_metrics so the definitions are
byte-identical to the IASBS numbers they are compared against.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remeasure as R

grid, cdf, _ = R.z_grid()
m2e, m4e, mEe, _ = R.z_moments()
print(f"exact E[x_3^2] = {m2e:.6f}")

rows = []
for s in range(5):
    p = os.path.join("ckpt", f"rasbs_sphere_matlabinit_s{s}.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    x = d["samples"]
    m = R.sphere_metrics(x, grid, cdf, m2e, m4e, mEe)
    rows.append(m)
    print(f"seed {s}  n={x.shape[0]}  KS_phi={m['KS_phi']:.5f}  "
          f"|d_m2|={abs(m['d_m2']):.5f}  W1_z={m['W1_z']:.5f}  "
          f"KS_z={m['KS_z']:.5f}  north_err={m['north_err']:.5f}")

n_used = rows and torch.load(
    os.path.join("ckpt", "rasbs_sphere_matlabinit_s0.pt"),
    map_location="cpu", weights_only=False)["samples"].shape[0]

gen = torch.Generator().manual_seed(0)
floors = [R.sphere_metrics(R.sample_exact(n_used, gen), grid, cdf,
                           m2e, m4e, mEe) for _ in range(5)]


def agg(rs, k, absval=False):
    v = np.array([abs(r[k]) if absval else r[k] for r in rs])
    return v.mean(), v.std()


print()
for k, a in (("KS_phi", False), ("d_m2", True), ("W1_z", False),
             ("KS_z", False), ("north_err", False)):
    mu, sd = agg(rows, k, a)
    fmu, fsd = agg(floors, k, a)
    print(f"{k:10s} mean {mu:.5f} +- {sd:.5f}   iid floor {fmu:.5f} +- {fsd:.5f}")
