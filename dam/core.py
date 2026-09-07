"""
Generic Discrete Adjoint Matching (DAM) core.

Source: So, Karrer, Fan, Chen, Liu, "Discrete Adjoint Matching",
arXiv:2602.07132v2 / ICLR 2026.  Equations (6), (11)-(14), (40).

This file knows nothing about Ising or occupation.  It implements four things:

  1. an exact Gillespie simulator of the *piecewise-constant* controlled CTMC,
     together with the full CTMC Radon-Nikodym path weight;
  2. the paper's importance-weighted discrete-adjoint estimator;
  3. the generalized-KL matching loss;
  4. the tiny adapter contract that benchmark wrappers implement.

Why the full path weight
------------------------
The paper's masked-diffusion specialization lets the model and the base process
share a total escape rate, so the integrated escape-rate term cancels and the
path weight collapses to a product of jump-probability ratios.  Our controllers
(swap multipliers on the Ising ring, per-mode alpha/beta on the occupation
process) genuinely change the total escape rate, so

    log W = int_t^1 [ R_ubar(s, X_s) - R_base(s, X_s) ] ds
            + sum_j log [ r(X_tau_j , X_tau_j^-) / ubar(X_tau_j, X_tau_j^-) ]

must be evaluated in full.  Dropping the integral makes the estimator wrong,
not merely noisy.

Because the model is parameterized multiplicatively,

    ubar_t(y, x) = r_t(y, x) exp(a_theta(t, x, y)),

every jump contributes exactly  -a_theta  to log W.

Adapter contract
----------------
An adapter is any object with these methods (all batched, all float64 where a
number is returned):

    rate_state(net, t, x) -> Rates
        t : (B,) float64 bin-left-edge times.  Returns a Rates namespace with
        fields R_model (B,), R_base (B,) and whatever payload sample_edge and
        friends need.  MUST be evaluated with the network under no_grad.

    sample_edge(rates, generator) -> (edge, log_a, log_q)
        edge is adapter-private.  log_a is a_theta on the sampled edge, log_q
        is log q(y | x) = log [ ubar_t(y,x) / R_ubar(t,x) ].

    apply_edge(x, edge, mask) -> x'
        apply the jump only where mask is True.

    a_on_edge(net, t, x, edge) -> (B,) float64, WITH gradient
    base_rate_on_edge(x, edge) -> (B,) float64  r_t(y, x)
    log_f1(x) -> (B,) float64

The adapter must guarantee common support, r > 0 iff u_theta > 0, or the
Radon-Nikodym derivative is not finite.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

TINY = 1e-300


def rates_ns(R_model, R_base, **payload):
    """Small helper so adapters do not each invent a container."""
    return SimpleNamespace(R_model=R_model, R_base=R_base, **payload)


# ----------------------------------------------------------------------------
# 1.  exact piecewise-constant CTMC rollout + path Radon-Nikodym weight
# ----------------------------------------------------------------------------
@torch.no_grad()
def rollout_ctmc(adapter, net, x0, t0, steps, generator=None, trace=None,
                 max_iters=None):
    """Simulate the controlled CTMC from (t0, x0) to time 1 and return

        x1     terminal states
        logw   log dP_base / dP_model along the sampled path   (B,) float64
        jumps  number of genuine off-diagonal jumps            (B,) long

    The control is piecewise constant on the grid t_s = s / steps, so within a
    bin the chain is a time-homogeneous CTMC and Gillespie sampling is EXACT --
    no discretisation error enters the weight.  This is deliberately NOT the
    repository's one-jump-per-bin sampler: that sampler is a fine terminal-law
    evaluator but its path law is not the CTMC path law in the DAM derivation.

    t0 is a per-sample vector because every training label starts its rollouts
    at its own uniformly drawn time.
    """
    x = x0.clone()
    tau = t0.to(torch.float64).clone().reshape(-1)
    B = tau.shape[0]
    dev = tau.device
    logw = torch.zeros(B, dtype=torch.float64, device=dev)
    jumps = torch.zeros(B, dtype=torch.long, device=dev)
    zero = torch.zeros(B, dtype=torch.float64, device=dev)

    cap = int(max_iters) if max_iters is not None else 10_000_000
    it = 0
    while True:
        active = tau < 1.0 - 1e-14
        if not bool(active.any()):
            break
        it += 1
        if it > cap:
            raise RuntimeError(f"rollout_ctmc exceeded {cap} iterations")

        s = torch.clamp(torch.floor(tau * steps), min=0.0, max=steps - 1.0)
        t_s = s / steps
        bin_end = torch.clamp((s + 1.0) / steps, max=1.0)

        rt = adapter.rate_state(net, t_s, x)
        Ru = rt.R_model.to(torch.float64)
        Rb = rt.R_base.to(torch.float64)
        if not (torch.isfinite(Ru).all() and torch.isfinite(Rb).all()):
            raise RuntimeError("non-finite escape rate in rollout_ctmc")

        u = torch.rand(B, device=dev, dtype=torch.float64,
                       generator=generator).clamp_min(1e-300)
        dt_prop = -torch.log(u) / Ru.clamp_min(TINY)
        dt_bin = bin_end - tau

        jump = active & (dt_prop < dt_bin)
        step = torch.where(jump, dt_prop, dt_bin)
        step = torch.where(active, step, zero)

        # integrated escape-rate term -- the piece the masked shortcut drops
        logw = logw + (Ru - Rb) * step

        edge, log_a, log_q = adapter.sample_edge(rt, generator)
        logw = logw + torch.where(jump, -log_a.to(torch.float64), zero)

        if trace is not None:
            trace.append({"tau": tau.clone(), "x": x.clone(),
                          "h": step.clone(), "jump": jump.clone(),
                          "edge": edge, "R_model": Ru.clone(),
                          "R_base": Rb.clone(), "log_a": log_a.clone()})

        x = adapter.apply_edge(x, edge, jump)
        tau = torch.where(jump, tau + step,
                          torch.where(active, bin_end, tau))
        jumps = jumps + jump.long()

    n_bad = int((~torch.isfinite(logw)).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad} non-finite path weights in rollout_ctmc")
    return x, logw, jumps


# ----------------------------------------------------------------------------
# 2.  importance-weighted discrete-adjoint estimator
# ----------------------------------------------------------------------------
@torch.no_grad()
def estimate_log_adjoint(adapter, net, t, x, y, K, steps, generator=None,
                         K_num=1):
    """log m_hat_t(y, x), the paper's importance-weighted adjoint multiplier.

        m_hat_t(y, x) = W(zeta^y) f1(Z) / [ (1/K) sum_k W(zeta^{x,k}) f1(X1^k) ]

    Numerator: K_num rollouts started at (t, y).  Denominator: K rollouts
    started at (t, x).  Both use the CURRENT stop-gradient model, and both are
    reweighted onto the base path measure, which is what makes

        E_base[ f1(X1) | X_t = y ] / E_base[ f1(X1) | X_t = x ] = phi_t(y)/phi_t(x)

    the estimand.  The default K_num = 1 is the paper's practical estimator; a
    larger value is an ablation only.

    Returns (log_m, stats) with stats holding ESS, jump counts and the number
    of terminal f1 evaluations, which is exactly K + K_num per label.
    """
    t = t.to(torch.float64).reshape(-1)
    B = t.shape[0]

    yr = torch.repeat_interleave(y, K_num, dim=0)
    tn = torch.repeat_interleave(t, K_num, dim=0)
    z, logw_y, j_y = rollout_ctmc(adapter, net, yr, tn, steps,
                                  generator=generator)
    ell_y = (logw_y + adapter.log_f1(z)).view(B, K_num)
    num = torch.logsumexp(ell_y, dim=1) - math.log(K_num)

    xr = torch.repeat_interleave(x, K, dim=0)
    td = torch.repeat_interleave(t, K, dim=0)
    x1, logw_k, j_k = rollout_ctmc(adapter, net, xr, td, steps,
                                   generator=generator)
    ell_k = (logw_k + adapter.log_f1(x1)).view(B, K)
    den = torch.logsumexp(ell_k, dim=1) - math.log(K)

    w = torch.exp(ell_k - ell_k.max(dim=1, keepdim=True).values)
    ess = (w.sum(dim=1) ** 2) / (w ** 2).sum(dim=1).clamp_min(TINY)

    stats = {
        "ess": ess,
        "jumps": int(j_y.sum()) + int(j_k.sum()),
        "f1_evals": B * (K + K_num),
        "log_num": num,
        "log_den": den,
    }
    return num - den, stats


# ----------------------------------------------------------------------------
# 3.  generalized-KL matching loss
# ----------------------------------------------------------------------------
LOG_M_CLIP = 30.0


def gkl_loss(a, log_m, r, log_q, clip=LOG_M_CLIP, keep=None, coef_cap=None):
    """Edge-sampled generalized-KL matching objective.

    Per edge the paper matches u_theta to w = r * m_hat under

        D_gKL(u, w) = u - w + w log(w / u).

    With Y drawn from q(. | x) = ubar / R_ubar, the unbiased objective is
    D_gKL / q.  Dropping the theta-independent terms leaves the compact,
    gradient-equivalent form used here:

        l = (r / q) [ exp(a) - stopgrad(m_hat) * a ].

    The 1/q factor is part of the published method, not a heuristic: without it
    the edge-sampled estimator targets a q-weighted objective instead of the
    full-edge sum.

    Truncated importance weighting
    ------------------------------
    m_hat is a RATIO of importance-weighted averages, so at small K its right
    tail is unbounded: a single lucky denominator rollout sends log m_hat past
    709 and exp() returns +inf, the gradient becomes NaN, and every subsequent
    multinomial trips a device-side assert.  This is not hypothetical -- it is
    exactly how the K = 1 leg dies.  clip = 30 truncates the weight to
    [1e-13, 1e13], which is far outside any physically meaningful range yet
    keeps the loss finite.  Truncation biases the estimator, so the fraction of
    clipped labels is reported alongside the result rather than hidden.

    Stabilisers (all disabled by default, so the published numbers are
    reproduced bit-for-bit unless a flag is passed)
    ---------------------------------------------
    `keep` masks out labels whose adjoint estimate is untrustworthy (ESS
    filtering); the mean is then taken over the surviving labels only, so the
    loss scale does not depend on how many were dropped.  `coef_cap` winsorises
    the `r/q` prefactor at `coef_cap x` its own batch median: `q` is a sampled
    edge probability, so `1/q` has an unbounded right tail and one rare edge can
    otherwise dominate the whole minibatch gradient.
    """
    lm = log_m.to(torch.float64)
    if clip is not None:
        lm = lm.clamp(-clip, clip)
    m = torch.exp(lm).detach()
    coef = (r.to(torch.float64) * torch.exp(-log_q.to(torch.float64))).detach()
    if coef_cap is not None and coef_cap > 0:
        med = coef.median().clamp_min(TINY)
        coef = coef.clamp(max=float(coef_cap) * med)
    per = coef * (torch.exp(a) - m * a)
    if keep is not None:
        kf = keep.to(per.dtype)
        return (per * kf).sum() / kf.sum().clamp_min(1.0)
    return per.mean()


def gkl_literal(a, log_m, r, log_q):
    """Literal (1/q) D_gKL(u, w) -- used only by the gradient identity test."""
    m = torch.exp(log_m.to(torch.float64))
    u = r.to(torch.float64) * torch.exp(a)
    w = r.to(torch.float64) * m
    d = u - w + w * (torch.log(w) - torch.log(u))
    return (d * torch.exp(-log_q.to(torch.float64))).mean()
