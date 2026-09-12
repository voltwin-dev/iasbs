"""Reviewer weakness 5, part 1: fixed-endpoint bridge validation.

The shipped audit (`bridge_audit.py`) checks the bridge through the *mixture*
identity: draw Y from the unconditioned terminal law, run the bridge to Y, and
compare the intermediate marginal to the unconditioned marginal.  That test is
averaged over the terminal endpoint, so a bridge error that is odd in the
endpoint cancels.  This script removes the averaging.

For a fixed pair (X0, X1) and a fixed time t the exact conditional law is

    p(X_t = z | X0, X1)  =  p_{r_{0,t}}(X0, z) p_{r_{t,1}}(z, X1) / p_{r_{0,1}}(X0, X1)

with r_{s,u} = (1/2) sigma^2 (u - s).  Both factors are available in closed
form, so we can draw *exact* i.i.d. samples from this law by rejection with the
forward (or backward) heat kernel as proposal:

    propose   z ~ p_{r_small}(centre_small, .)          -- exact, inverse-CDF
    accept    u < p_{r_large}(centre_large, z) / M,     M = 1.2 * pilot max

The proposal is always taken to be the *shorter* of the two clocks, so the
weight kernel is the flatter one and the acceptance rate stays usable.  The
constant M is fixed from an independent pilot pool before the accept/reject
pass, and violations are counted (they must be zero), so the accepted sample is
an exact i.i.d. draw -- no SIR bias, no MCMC burn-in question.

Sampling from the heat kernel itself:

  S^2   theta ~ p_r(cos theta) sin theta            (inverse CDF on the
        z = cos theta * c + sin theta * v            HeatTable theta grid)

  St(4,2)  lift to the double cover: a, b ~ S^3 heat kernel at spin time r/2
        from the identity quaternion, theta ~ p_s(cos theta) sin^2 theta,
        z = R(a, b) E0.  The SO(4) heat kernel is the product of the two
        SU(2) factors at half the ambient clock, so the pushforward to
        St = SO(4)/SO(2) is exactly p^{St}_r(E0, .).

Arbitrary left endpoints.  `sphere.bridge_path` takes x0 directly.  The Stiefel
bridge hard-codes X0 = E0 (it starts both spin bridges at the identity
quaternion), so a general pair is handled by left-equivariance,

    p^{St}_r(g X, g Y) = p^{St}_r(X, Y)   for g in SO(4),

i.e. run the bridge to R_{X0}^{-1} X1 and push the whole path forward by
R_{X0}.  Pair Q4 below exercises exactly that route against the exact density
of the *general* pair, so the equivariance is validated rather than assumed.
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch


import _paths                                    # noqa: F401

import common as C                                              # noqa: E402
import sphere as SPH                                            # noqa: E402
import stiefel as ST                                            # noqa: E402
import bridge_audit as A                                       # noqa: E402

DEV = "cuda:0"
DT = torch.float64
SIGMA = math.sqrt(2.0)
R01 = C.heat_clock(0.0, 1.0, SIGMA)          # = 1.0
TIMES = [0.25, 0.5, 0.75]
N_REF = 20000          # exact rejection samples per (pair, time)
N_BR = 20000           # bridge samples per (pair, time)
MMD_N = 4000
MMD_REPS = 8
PILOT = 200000
SAFETY = 1.2


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def _cdf_from_grid(logp, extra_log, dth):
    """Normalised CDF of exp(logp + extra_log) on a uniform theta grid."""
    lw = logp + extra_log
    lw = lw - lw.max()
    w = np.exp(lw)
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return cdf


def _inv_cdf(cdf, th, n, gen, device):
    u = torch.rand(n, generator=gen, device=device, dtype=DT).cpu().numpy()
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 1, len(cdf) - 1)
    c0, c1 = cdf[idx - 1], cdf[idx]
    w = np.where(c1 > c0, (u - c0) / np.maximum(c1 - c0, 1e-300), 0.0)
    theta = th[idx - 1] + w * (th[idx] - th[idx - 1])
    return torch.tensor(theta, device=device, dtype=DT)


def _rand_orthonormal_to(v, n, gen):
    """n uniform unit vectors orthogonal to the unit vector v in R^3."""
    g = torch.randn(n, 3, generator=gen, device=v.device, dtype=DT)
    g = g - (g @ v)[:, None] * v[None, :]
    return g / g.norm(dim=-1, keepdim=True)


def _rand_s2(n, gen, device):
    g = torch.randn(n, 3, generator=gen, device=device, dtype=DT)
    return g / g.norm(dim=-1, keepdim=True)


def obs_stats(v):
    v = v.to(DT)
    return float(v.mean()), float(v.std(unbiased=True) / math.sqrt(len(v)))


# ---------------------------------------------------------------------------
# S^2
# ---------------------------------------------------------------------------
class S2Exact:
    """Exact sampler for p(z | x0, x1) at time t on S^2."""

    def __init__(self, t, device=DEV):
        self.r0t = float(C.heat_clock(0.0, t, SIGMA))
        self.rt1 = float(C.heat_clock(t, 1.0, SIGMA))
        self.tab = SPH.HeatTable([self.r0t, self.rt1], ntheta=16385,
                                 device=device)
        self.th = np.linspace(0.0, math.pi, 16385)
        self.dth = self.th[1] - self.th[0]
        self.device = device
        # propose from the shorter clock -> flatter weight kernel
        self.k_prop = 0 if self.r0t <= self.rt1 else 1
        self.k_wgt = 1 - self.k_prop
        lp = self.tab.logp.cpu().numpy()[self.k_prop]
        with np.errstate(divide="ignore"):
            self.cdf = _cdf_from_grid(lp, np.log(np.maximum(np.sin(self.th),
                                                            1e-300)), self.dth)

    def logp(self, k, c):
        si = torch.full(c.shape, k, device=c.device, dtype=torch.long)
        return self.tab.logp_at(si, c)

    def _propose(self, n, centre, gen):
        theta = _inv_cdf(self.cdf, self.th, n, gen, self.device)
        v = _rand_orthonormal_to(centre, n, gen)
        z = torch.cos(theta)[:, None] * centre[None, :] + \
            torch.sin(theta)[:, None] * v
        return z / z.norm(dim=-1, keepdim=True)

    def sample(self, x0, x1, n, gen):
        c_prop = x0 if self.k_prop == 0 else x1
        c_wgt = x1 if self.k_prop == 0 else x0
        zp = self._propose(PILOT, c_prop, gen)
        lw = self.logp(self.k_wgt, (zp @ c_wgt).clamp(-1., 1.))
        logM = float(lw.max()) + math.log(SAFETY)
        out, tried, acc, viol = [], 0, 0, 0
        while acc < n:
            z = self._propose(200000, c_prop, gen)
            lw = self.logp(self.k_wgt, (z @ c_wgt).clamp(-1., 1.))
            viol += int((lw > logM).sum())
            u = torch.rand(z.shape[0], generator=gen, device=z.device, dtype=DT)
            m = torch.log(u.clamp(min=1e-300)) < (lw - logM)
            out.append(z[m])
            acc += int(m.sum())
            tried += z.shape[0]
        Z = torch.cat(out)[:n]
        return Z, {"accept_rate": acc / tried, "n_proposed": tried,
                   "M_violations": viol, "r0t": self.r0t, "rt1": self.rt1}


def run_sphere(res, gen):
    e3 = torch.tensor([0., 0., 1.], device=DEV, dtype=DT)

    def at_angle(ang, phi=0.0):
        return torch.tensor([math.sin(ang) * math.cos(phi),
                             math.sin(ang) * math.sin(phi),
                             math.cos(ang)], device=DEV, dtype=DT)

    g0 = torch.tensor([0.4, -0.7, 0.5], device=DEV, dtype=DT)
    g0 = g0 / g0.norm()
    g1 = torch.tensor([-0.8, 0.1, -0.6], device=DEV, dtype=DT)
    g1 = g1 / g1.norm()

    pairs = {
        "P1_close_0.35": (e3, at_angle(0.35)),
        "P2_orthogonal_pi/2": (e3, at_angle(math.pi / 2)),
        "P3_near_antipodal_2.90": (e3, at_angle(2.90)),
        "P4_generic_x0": (g0, g1),
    }
    probs = {s: SPH.SphereProblem(sigma=SIGMA, steps=s, device=DEV)
             for s in (128, 1024)}

    for pname, (x0, x1) in pairs.items():
        sep = float(torch.arccos((x0 @ x1).clamp(-1., 1.)))
        for t in TIMES:
            key = f"{pname}@t={t}"
            t0 = time.time()
            ex = S2Exact(t, DEV)
            Z, info = ex.sample(x0, x1, N_REF, gen)
            s2 = A.median_bandwidth(Z, cap=4000)
            floor_m, floor_sd = A.split_half_floor(Z, MMD_N, s2, MMD_REPS, gen)

            entry = {"separation_angle": sep, "t": t,
                     "exact": info, "bandwidth_sigma2": s2,
                     "floor_mean": floor_m, "floor_sd": floor_sd,
                     "exact_obs": {
                         "E_z_dot_x0": obs_stats(Z @ x0),
                         "E_z_dot_x1": obs_stats(Z @ x1)},
                     "bridge": {}}

            for steps, prob in probs.items():
                si = int(round(t * steps))
                x0b = x0[None, :].expand(N_BR, 3).contiguous()
                y1b = x1[None, :].expand(N_BR, 3).contiguous()
                with torch.no_grad():
                    path = SPH.bridge_path(prob, y1b, steps, generator=gen,
                                           x0=x0b)
                Xb = path[si].contiguous()
                vals = [A.mmd2_unbiased(
                    Xb[torch.randperm(N_BR, generator=gen, device=DEV)[:MMD_N]],
                    Z[torch.randperm(N_REF, generator=gen, device=DEV)[:MMD_N]],
                    s2) for _ in range(MMD_REPS)]
                entry["bridge"][steps] = {
                    "t_grid": si / steps,
                    "MMD2": float(np.mean(vals)),
                    "MMD2_se": float(np.std(vals) / math.sqrt(MMD_REPS)),
                    "E_z_dot_x0": obs_stats(Xb @ x0),
                    "E_z_dot_x1": obs_stats(Xb @ x1)}
                del path, Xb
            entry["wall_s"] = time.time() - t0
            res["sphere"][key] = entry
            b = entry["bridge"]
            print(f"  S2 {key:28s} sep={sep:.3f} acc={info['accept_rate']:.4f} "
                  f"floor={floor_m:+.2e}+-{floor_sd:.1e} "
                  f"MMD2_128={b[128]['MMD2']:+.2e} "
                  f"MMD2_1024={b[1024]['MMD2']:+.2e}", flush=True)


# ---------------------------------------------------------------------------
# St(4,2)
# ---------------------------------------------------------------------------
def R_of(X):
    """SO(4) matrix R with R E0 = X, and its inverse."""
    a, b = ST.section_lift(X[None])
    R = ST.quat_to_matrix(a, b)[0]
    Rinv = ST.quat_to_matrix(ST.qconj(a), ST.qconj(b))[0]
    return R, Rinv


class StExact:
    """Exact sampler for p(z | X0, X1) at time t on St(4,2)."""

    def __init__(self, t, nq=256, device=DEV):
        self.r0t = float(C.heat_clock(0.0, t, SIGMA))
        self.rt1 = float(C.heat_clock(t, 1.0, SIGMA))
        self.ss = np.array([C.stiefel_spin_factor_time(self.r0t),
                            C.stiefel_spin_factor_time(self.rt1)])
        self.tab = ST.S3Table(self.ss, ntheta=16385, device=device)
        self.th = np.linspace(0.0, math.pi, 16385)
        self.device = device
        psi, wq = C.periodic_legendre_nodes(nq, device=device)
        self.psi, self.logwq = psi, torch.log(wq)
        self.k_prop = 0 if self.r0t <= self.rt1 else 1
        self.k_wgt = 1 - self.k_prop
        lp = self.tab.logp.cpu().numpy()[self.k_prop]
        sn = np.maximum(np.sin(self.th), 1e-300)
        with np.errstate(divide="ignore"):
            self.cdf = _cdf_from_grid(lp, 2.0 * np.log(sn),
                                      self.th[1] - self.th[0])

    # -- proposal: exact draw from p^{St}_{r_prop}(centre, .) ---------------
    def _quat(self, n, gen):
        theta = _inv_cdf(self.cdf, self.th, n, gen, self.device)
        u = _rand_s2(n, gen, self.device)
        return torch.cat([torch.cos(theta)[:, None],
                          torch.sin(theta)[:, None] * u], dim=-1)

    def _propose(self, n, R_centre, gen):
        a = self._quat(n, gen)
        b = self._quat(n, gen)
        z = ST.stiefel_from_quat(a, b)
        return R_centre[None] @ z

    # -- weight: log p^{St}_{r_wgt}(centre_wgt, z) --------------------------
    def logk(self, k, Rinv_centre, Z, chunk=20000):
        out = []
        s_idx = torch.tensor(k, device=Z.device, dtype=torch.long)
        for i in range(0, Z.shape[0], chunk):
            Y = Rinv_centre[None] @ Z[i:i + chunk]
            a, b = ST.section_lift(Y)
            cs, sn = torch.cos(self.psi)[None, :], torch.sin(self.psi)[None, :]
            ua = a[:, 0:1] * cs - a[:, 1:2] * sn
            ub = b[:, 0:1] * cs - b[:, 1:2] * sn
            si = s_idx.expand(ua.shape)
            lk = self.tab.logp_at(si, ua.clamp(-1., 1.)) \
                + self.tab.logp_at(si, ub.clamp(-1., 1.)) + self.logwq[None, :]
            out.append(torch.logsumexp(lk, dim=1))
        return torch.cat(out)

    def sample(self, X0, X1, n, gen):
        R0, R0i = R_of(X0)
        R1, R1i = R_of(X1)
        Rp, Rpi = (R0, R0i) if self.k_prop == 0 else (R1, R1i)
        Rwi = R1i if self.k_prop == 0 else R0i
        Zp = self._propose(PILOT, Rp, gen)
        logM = float(self.logk(self.k_wgt, Rwi, Zp).max()) + math.log(SAFETY)
        out, tried, acc, viol = [], 0, 0, 0
        while acc < n:
            Z = self._propose(200000, Rp, gen)
            lw = self.logk(self.k_wgt, Rwi, Z)
            viol += int((lw > logM).sum())
            u = torch.rand(Z.shape[0], generator=gen, device=Z.device, dtype=DT)
            m = torch.log(u.clamp(min=1e-300)) < (lw - logM)
            out.append(Z[m])
            acc += int(m.sum())
            tried += Z.shape[0]
        ZZ = torch.cat(out)[:n]
        return ZZ, {"accept_rate": acc / tried, "n_proposed": tried,
                    "M_violations": viol, "r0t": self.r0t, "rt1": self.rt1}


def run_stiefel(res, gen):
    E0 = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
                      device=DEV, dtype=DT)

    def rot(ang, i, j):
        M = torch.zeros(4, 4, device=DEV, dtype=DT)
        M[i, j], M[j, i] = ang, -ang
        return torch.matrix_exp(M)

    X_close = rot(0.25, 0, 2) @ E0
    X_mid = rot(math.pi / 4, 0, 2) @ rot(math.pi / 5, 1, 3) @ E0
    X_far = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [0., 1.]],
                         device=DEV, dtype=DT)
    G0 = rot(0.9, 0, 3) @ rot(-0.6, 1, 2) @ E0
    G1 = rot(2.1, 0, 2) @ rot(1.3, 1, 3) @ E0

    pairs = {
        "Q1_close": (E0, X_close),
        "Q2_mid": (E0, X_mid),
        "Q3_orthogonal_complement": (E0, X_far),
        "Q4_generic_X0_equivariance": (G0, G1),
    }
    probs = {s: ST.StiefelProblem(sigma=SIGMA, beta=1.0, steps=s, nq=64,
                                  frame=True, device=DEV) for s in (199, 1024)}

    for pname, (X0, X1) in pairs.items():
        sv = torch.linalg.svdvals(X0.T @ X1).clamp(-1., 1.)
        sep = float((X1 - X0).norm())
        pang = [float(a) for a in torch.arccos(sv)]
        R0, R0i = R_of(X0)
        Y1 = R0i @ X1                       # bridge target in the E0 frame
        for t in TIMES:
            key = f"{pname}@t={t}"
            t0 = time.time()
            ex = StExact(t, device=DEV)
            Z, info = ex.sample(X0, X1, N_REF, gen)
            V = Z.reshape(N_REF, 8)
            s2 = A.median_bandwidth(V, cap=4000)
            floor_m, floor_sd = A.split_half_floor(V, MMD_N, s2, MMD_REPS, gen)

            entry = {"sep_frob": sep, "principal_angles": pang, "t": t,
                     "exact": info, "bandwidth_sigma2": s2,
                     "floor_mean": floor_m, "floor_sd": floor_sd,
                     "exact_obs": {
                         "E_trX0tZ": obs_stats((Z * X0[None]).sum((-1, -2))),
                         "E_trX1tZ": obs_stats((Z * X1[None]).sum((-1, -2)))},
                     "bridge": {}}

            for steps, prob in probs.items():
                si = int(round(t * steps))
                Y1b = Y1[None].expand(N_BR, 4, 2).contiguous()
                with torch.no_grad():
                    path = ST.bridge_path(prob, Y1b, steps, generator=gen)
                Xb = (R0[None] @ path[si]).contiguous()      # push forward
                Vb = Xb.reshape(N_BR, 8)
                vals = [A.mmd2_unbiased(
                    Vb[torch.randperm(N_BR, generator=gen, device=DEV)[:MMD_N]],
                    V[torch.randperm(N_REF, generator=gen, device=DEV)[:MMD_N]],
                    s2) for _ in range(MMD_REPS)]
                entry["bridge"][steps] = {
                    "t_grid": si / steps,
                    "MMD2": float(np.mean(vals)),
                    "MMD2_se": float(np.std(vals) / math.sqrt(MMD_REPS)),
                    "constraint": float(C.stiefel_constraint_error(Xb).max()),
                    "E_trX0tZ": obs_stats((Xb * X0[None]).sum((-1, -2))),
                    "E_trX1tZ": obs_stats((Xb * X1[None]).sum((-1, -2)))}
                del path, Xb, Vb
            entry["wall_s"] = time.time() - t0
            res["stiefel"][key] = entry
            b = entry["bridge"]
            print(f"  St {key:36s} sep={sep:.3f} acc={info['accept_rate']:.4f} "
                  f"floor={floor_m:+.2e}+-{floor_sd:.1e} "
                  f"MMD2_199={b[199]['MMD2']:+.2e} "
                  f"MMD2_1024={b[1024]['MMD2']:+.2e}", flush=True)


def main():
    gen = torch.Generator(device=DEV)
    gen.manual_seed(20260911)
    res = {"config": {"N_REF": N_REF, "N_BR": N_BR, "MMD_N": MMD_N,
                      "MMD_REPS": MMD_REPS, "times": TIMES, "sigma": SIGMA,
                      "r01": R01, "pilot": PILOT, "safety": SAFETY},
           "sphere": {}, "stiefel": {}}
    t0 = time.time()
    print("S^2 fixed-endpoint bridge validation", flush=True)
    run_sphere(res, gen)
    print("St(4,2) fixed-endpoint bridge validation", flush=True)
    run_stiefel(res, gen)
    res["wall_s"] = time.time() - t0
    out = os.path.join(_paths.ROOT, "json", "results_weakness5_bridge.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out, f"({res['wall_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
