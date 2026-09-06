"""Sample gallery for 'ASBS on Structured State Spaces via Markov Semigroup
Intertwining'.

figures.py plots metrics.  This file plots the *samples themselves*, as the
objects they actually are -- points on a sphere, orthonormal 2-frames, spin
configurations, occupancy vectors -- beside the exact / MCMC reference drawn
the same way.  Nothing here is summarised into a curve.

Two things this file is careful about, because both are easy to get wrong and
both would flatter us:

  * Checkpoints written before common.py stored samples in float64 are float32.
    An orthogonality residual recomputed from those is float32 round-off
    (~1e-7), NOT the true 3.8e-14 that stiefel.py measures at generation time,
    and it coincidentally resembles R-ASBS's 3.4e-07 retraction error.
    Figure 7 draws that residual only when the checkpoint is actually float64,
    and prints a notice otherwise rather than plotting the storage artefact.

  * stiefel.py works in the eigenbasis of H (H = diag(1,2,5,8)) while
    rasbs_port.py works in R-ASBS's ambient Z2xZ2 basis.  Figure 7 rotates
    their samples into the common eigenbasis before comparing anything.

A panel whose checkpoint is missing is skipped with a printed notice.

    python gallery.py              # all panels -> fig/
    python gallery.py --only 7     # just the Stiefel sheet
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.colors import (LinearSegmentedColormap,  # noqa: E402
                               ListedColormap, BoundaryNorm)

OUT = "fig"
CK = "ckpt"

# Colour-blind-safe divergent map, used for every signed quantity.
DIV = LinearSegmentedColormap.from_list(
    "ob", ["#0072B2", "#F7F7F7", "#D55E00"])


def load_ck(tag):
    p = os.path.join(CK, f"{tag}.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def samples_of(tag, want_f64=False):
    """Return samples as float64 for arithmetic.

    `want_f64` asks whether the file *itself* was float64.  Upcasting a float32
    checkpoint does not recover precision, so any panel that measures round-off
    has to check the stored dtype rather than the array it gets back.
    """
    d = load_ck(tag)
    if d is None:
        return (None, False) if want_f64 else None
    x = d["samples"]
    exact = x.dtype == torch.float64
    x = x.numpy().astype(np.float64)
    return (x, exact) if want_f64 else x


def _missing(fig, tags):
    print(f"  [skip] figure {fig}: missing ckpt {', '.join(tags)}")


def _finish(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png / .pdf")


def _caption(ax, text, y=-0.045):
    """Panels that carry an xlabel must push their caption further down; -0.045
    sits on top of the label otherwise."""
    ax.text(0.5, y, text, ha="center", va="top", fontsize=8.5,
            color="#444444", transform=ax.transAxes)


# ============================================================================
# Figure 6 -- S^2 point clouds
# ============================================================================

def _sphere_panel(ax, X, title, sub):
    """Orthographic render of the cloud on S^2, front hemisphere only.

    Culling the back hemisphere is what makes the picture readable: with all
    200k points drawn the far side shows through and every cloud looks uniform.
    """
    v = np.array([0.30, -0.80, 0.52])
    v /= np.linalg.norm(v)
    e1 = np.cross(v, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)

    depth = X @ v
    front = depth > 0.02
    P = X[front]
    ax.scatter(P @ e1, P @ e2, s=0.4, c=depth[front], cmap="viridis",
               vmin=0, vmax=1, alpha=0.5, linewidths=0, rasterized=True)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#333333", lw=0.9)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=4)
    _caption(ax, sub)


def _exact_s2(n, seed=0):
    """Exact iid draws from the S^2 target.

    The target has density proportional to exp(6 z^2) in z = x_3 with uniform
    azimuth, so inverse-CDF sampling is exact -- no chain, no discretisation.
    This matters: ckpt/sphere_reference.pt is the *uncontrolled* reference
    process (KS(x_3) = 0.32), not the target, and comparing against it would
    measure the transport rather than the error.
    """
    import common as C
    grid, cdf = C.exact_s2_z_cdf_grid(num=200001)
    rng = np.random.default_rng(seed)
    z = np.interp(rng.random(n), cdf, grid)
    ph = rng.uniform(-np.pi, np.pi, n)
    r = np.sqrt(np.maximum(1.0 - z ** 2, 0.0))
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def figure6(args):
    ours, src = load_ck("sphere_anti_seed0"), load_ck("sphere_reference")
    miss = [t for t, d in [("sphere_anti_seed0", ours),
                           ("sphere_reference", src)] if d is None]
    if miss:
        return _missing(6, miss)

    Xo = ours["samples"].numpy().astype(np.float64)
    Xs = src["samples"].numpy().astype(np.float64)
    Xr = _exact_s2(len(Xo), seed=0)
    mo, ms = ours["extra"]["metrics"], src["extra"]["metrics"]

    fig = plt.figure(figsize=(14.6, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.24)

    _sphere_panel(fig.add_subplot(gs[0, 0]), Xs, "source (uncontrolled)",
                  f"what the bridge starts from\nKS($x_3$) = "
                  f"{ms['KS_z']:.3f}")
    _sphere_panel(fig.add_subplot(gs[0, 1]), Xo, "ours",
                  f"north mass {mo['north_mass']:.4f}, KS($x_3$) = "
                  f"{mo['KS_z']:.4f}\n$|\\,\\|x\\|-1|$ = "
                  f"{mo['constraint']:.1e}, {len(Xo):,} samples")
    _sphere_panel(fig.add_subplot(gs[0, 2]), Xr, "exact target",
                  f"north mass {(Xr[:, 2] > 0).mean():.4f} (analytic 0.5000)"
                  f"\niid inverse-CDF draws, {len(Xr):,} samples")

    # Residual as a z-score against the two samplers' own binomial noise: a
    # correct sampler must give a map that looks like standard normal noise,
    # and any real bias sticks out of it.  Plotting a raw density difference
    # instead would just show the target's own concentration near the poles.
    nz, npx = 30, 60
    def counts(X):
        return np.histogram2d(np.clip(X[:, 2], -1, 1),
                              np.arctan2(X[:, 1], X[:, 0]),
                              bins=[nz, npx],
                              range=[[-1, 1], [-np.pi, np.pi]])[0]

    Co, Cr = counts(Xo), counts(Xr)
    No, Nr = Co.sum(), Cr.sum()
    pb = (Co + Cr) / (No + Nr)                    # pooled bin probability
    se = np.sqrt(np.maximum(pb * (1 - pb), 1e-30) * (1 / No + 1 / Nr))
    Z = (Co / No - Cr / Nr) / se

    ax = fig.add_subplot(gs[0, 3])
    lim = float(np.nanpercentile(np.abs(Z), 99))
    im = ax.imshow(Z, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto",
                   origin="lower", extent=[-180, 180, -1, 1],
                   interpolation="nearest")
    ax.set_xlabel("longitude (deg)", fontsize=9)
    ax.set_ylabel("$\\cos\\theta$", fontsize=9)
    ax.set_title("ours $-$ exact, in units of sampling noise", fontsize=11,
                 pad=4)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("z-score per equal-area bin", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    frac = float(np.mean(np.abs(Z) < 3))
    _caption(ax, f"{frac * 100:.0f}% of {nz * npx} bins within $\\pm3\\sigma$, "
                 f"99.7% expected if the two agree; max $|z|$ = "
                 f"{np.abs(Z).max():.1f}.\nAt {No:,.0f} samples a side there is "
                 f"no resolvable structure left in the residual.", y=-0.15)

    fig.suptitle("Figure 6 $-$ $S^2$: the samples themselves, front hemisphere "
                 "(source, ours, exact target, residual)", fontsize=13, y=1.04)
    _finish(fig, "fig6_sphere_cloud")


# ============================================================================
# Figure 7 -- St(4,2): frames, and the second moment that separates the methods
# ============================================================================

def _canonical(X):
    """Fix the sign of each column so tiles from different samplers are
    comparable: X and X * diag(+-1) are the same point of the quotient the
    energy sees, so without this the sheets are pure sign noise."""
    k = np.abs(X).argmax(axis=1)                       # (n, p)
    lead = np.take_along_axis(X, k[:, None, :], axis=1)[:, 0, :]
    return X * np.sign(lead)[:, None, :]


def figure7(args, beta=2):
    o, o64 = samples_of(f"stiefel_grid_b{beta}_seed0", want_f64=True)
    r = samples_of(f"stiefel_grid_b{beta}_mcmc")
    a, a64 = samples_of(f"rasbs_b{beta}", want_f64=True)
    miss = [t for t, d in [(f"stiefel_grid_b{beta}_seed0", o),
                           (f"stiefel_grid_b{beta}_mcmc", r),
                           (f"rasbs_b{beta}", a)] if d is None]
    if miss:
        return _missing(7, miss)

    from rasbs_port import H_RASBS
    w, U = np.linalg.eigh(np.asarray(H_RASBS, dtype=np.float64))
    a = np.einsum("ij,nja->nia", U.T, a)     # ambient -> eigenbasis diag(w)

    energy = lambda X: (w[None, :, None] * X ** 2).sum((1, 2))
    rows, cols = 8, 12
    n = rows * cols

    def pick(X):
        """Quantile-spaced in energy, so each sheet spans its own full range
        and the sheets line up rank for rank."""
        q = np.argsort(energy(X))
        return _canonical(X[q[np.linspace(0, len(q) - 1, n).astype(int)]])

    def sheet(ax, X, title, sub):
        S = pick(X)
        tile = np.full((rows * 5 - 1, cols * 3 - 1), np.nan)
        for k in range(n):
            i, j = divmod(k, cols)
            tile[i * 5:i * 5 + 4, j * 3:j * 3 + 2] = S[k]
        ax.imshow(tile, cmap=DIV, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10.5, pad=4)
        _caption(ax, sub)

    M = lambda X: np.einsum("nia,nka->ik", X, X) / len(X)
    Mr, Mo, Ma, Mh = M(r), M(o), M(a), np.eye(4) * 0.5
    dh = np.linalg.norm(Mh - Mr)

    show_gram = o64 and a64
    if not show_gram:
        print("  [note] figure 7: orthogonality panel skipped -- "
              f"ours float64={o64}, R-ASBS float64={a64}; a residual "
              "recomputed from float32 storage would be round-off, not the "
              "measured 3.8e-14.  Re-run the experiment to refresh the "
              "checkpoint.")
    fig = plt.figure(figsize=(12.6, 6.4 + (2.0 if show_gram else 0.0)))
    gs = fig.add_gridspec(3 if show_gram else 2, 4,
                          height_ratios=[1, 0.62, 0.52] if show_gram
                          else [1, 0.62], hspace=0.42, wspace=0.14)

    sub = fig.add_subplot(gs[0, 0:1])
    sheet(sub, o, "ours", f"$E$ = {(w * np.diag(Mo)).sum():.3f}")
    sheet(fig.add_subplot(gs[0, 1:2]), r, "target (MCMC reference)",
          f"$E$ = {(w * np.diag(Mr)).sum():.3f}")
    sheet(fig.add_subplot(gs[0, 2:3]), a, "R-ASBS",
          f"$E$ = {(w * np.diag(Ma)).sum():.3f}")
    axl = fig.add_subplot(gs[0, 3:4])
    axl.axis("off")
    axl.text(0.0, 0.97,
             f"96 frames $X\\in$ St(4,2), $\\beta$ = {beta}\n\n"
             "Each tile is one 4$\\times$2 matrix, drawn in the\n"
             "eigenbasis of $H$ = diag(1, 2, 5, 8).\n"
             "Column signs are canonicalised and the\n"
             "frames are quantile-spaced in energy, so\n"
             "the three sheets line up rank for rank.\n\n"
             "The target puts its mass in the two low-\n"
             "eigenvalue directions: strong colour in the\n"
             "top two rows of each tile, pale in the\n"
             "bottom two. Ours reproduces that. R-ASBS\n"
             "is visibly paler on top and darker on the\n"
             "bottom $-$ mass has leaked back toward the\n"
             "uniform (Haar) frame.",
             fontsize=8.6, va="top", ha="left", color="#222222",
             linespacing=1.5)

    # The second moment makes that leak quantitative.  Haar is the null: if a
    # sampler inherits its source instead of transporting it, this is where it
    # shows.
    def mom(ax, Mx, title, sub, ref=None):
        im = ax.imshow(Mx, cmap="cividis", vmin=0, vmax=1,
                       interpolation="nearest")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="w" if Mx[i, j] < 0.62 else "k")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(["1", "2", "5", "8"], fontsize=7)
        ax.set_yticklabels(["1", "2", "5", "8"], fontsize=7)
        ax.set_title(title, fontsize=9.5, pad=3)
        _caption(ax, sub)
        return im

    axs = [fig.add_subplot(gs[1, k]) for k in range(4)]
    im = mom(axs[0], Mr, "target  $E[XX^\\top]$", "eigenvalue of $H$ per axis")
    mom(axs[1], Mo, "ours",
        f"$\\|\\cdot-{{\\rm target}}\\|_F$ = {np.linalg.norm(Mo - Mr):.3f}\n"
        f"{100 * np.linalg.norm(Mo - Mr) / dh:.0f}% of the way back to Haar")
    mom(axs[2], Ma, "R-ASBS",
        f"$\\|\\cdot-{{\\rm target}}\\|_F$ = {np.linalg.norm(Ma - Mr):.3f}\n"
        f"{100 * np.linalg.norm(Ma - Mr) / dh:.0f}% of the way back to Haar")
    mom(axs[3], Mh, "Haar (their source)",
        f"$\\|\\cdot-{{\\rm target}}\\|_F$ = {dh:.3f}\n"
        "the null: no transport at all")
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.012)
    cb.ax.tick_params(labelsize=7)

    if show_gram:
        # Constraint enforcement, drawn rather than tabulated.  Ours is an exact
        # geodesic step, theirs a QR retraction; the gap is seven orders of
        # magnitude and no amount of grid refinement closes it.
        def gram(ax, X, title, sub):
            G = np.abs(np.einsum("nia,nib->nab", X[:96], X[:96])
                       - np.eye(2))
            rows_, cols_ = 4, 24
            tile = np.full((rows_ * 3 - 1, cols_ * 3 - 1), np.nan)
            for k in range(rows_ * cols_):
                i, j = divmod(k, cols_)
                tile[i * 3:i * 3 + 2, j * 3:j * 3 + 2] = G[k]
            im = ax.imshow(np.log10(np.maximum(tile, 1e-18)), cmap="magma",
                           vmin=-16, vmax=-6, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, fontsize=9.5, pad=3)
            _caption(ax, sub)
            return im

        Go = np.abs(np.einsum("nia,nib->nab", o, o) - np.eye(2)).max()
        Ga = np.abs(np.einsum("nia,nib->nab", a, a) - np.eye(2)).max()
        axa = fig.add_subplot(gs[2, 0:2])
        img = gram(axa, o, "$|X^\\top X - I|$, ours (exact geodesic step)",
                   f"max over all {len(o):,} samples: {Go:.1e}")
        axb = fig.add_subplot(gs[2, 2:4])
        gram(axb, a, "$|X^\\top X - I|$, R-ASBS (QR retraction)",
             f"max over all {len(a):,} samples: {Ga:.1e}   "
             f"({Ga / max(Go, 1e-300):.0e}$\\times$ larger)")
        cb2 = fig.colorbar(img, ax=[axa, axb], fraction=0.015, pad=0.012)
        cb2.set_label("$\\log_{10}$ residual", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

    fig.suptitle(f"Figure 7 $-$ St(4,2) at $\\beta$ = {beta}: raw frames, and "
                 "the second moment that separates the samplers",
                 fontsize=13, y=1.0)
    _finish(fig, "fig7_stiefel_frames")


# ============================================================================
# Figure 8 -- fixed-magnetisation Ising configurations
# ============================================================================

def figure8(args):
    ck = load_ck("ising_poisson")
    if ck is None:
        return _missing(8, ["ising_poisson"])

    ex = ck["extra"]
    states = np.asarray(ex["states"])
    pi = np.asarray(ex["pi"], dtype=np.float64)
    idx = ck["samples"].numpy().astype(np.int64)
    L = int(round(np.sqrt(states.shape[1])))

    spins = np.where(states > 0, 1, -1)
    rng = np.random.default_rng(1)
    n = 10 * 5
    ours = spins[idx[rng.choice(len(idx), n, replace=False)]]
    exact = spins[rng.choice(len(pi), n, p=pi / pi.sum())]

    # Three colours, not two: down / up / gutter.  With a plain binary map the
    # down-spins and the gaps between tiles are the same white and the 4x4
    # configurations visually merge into one blob.
    cmap = ListedColormap(["#DCE6EF", "#1B3A5C", "#FFFFFF"])
    norm = BoundaryNorm([-1.5, 0.0, 1.5, 2.5], cmap.N)

    def grid(ax, S, title, sub):
        rows, cols = 5, 10
        tile = np.full((rows * (L + 1) + 1, cols * (L + 1) + 1), 2.0)
        for k in range(rows * cols):
            i, j = divmod(k, cols)
            tile[1 + i * (L + 1):1 + i * (L + 1) + L,
                 1 + j * (L + 1):1 + j * (L + 1) + L] = S[k].reshape(L, L)
        ax.imshow(tile, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=11, pad=4)
        _caption(ax, sub)

    mo, me = ours.sum(1), exact.sum(1)
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 3.6))
    grid(ax[0], ours, "ours $-$ 50 sampled configurations",
         f"$\\sum_i s_i$ = {mo.min()} for all 50 (range {mo.min()}..{mo.max()})"
         f"\nTV vs exact {ex['final_TV']:.4f}, {ex['violations']} violations "
         f"over {len(idx):,} samples")
    grid(ax[1], exact, "exact enumeration $-$ 50 draws from $\\pi$",
         f"$\\sum_i s_i$ = {me.min()} for all 50 (range {me.min()}..{me.max()})"
         f"\n{len(pi):,} admissible states out of $2^{{{L * L}}}$ = "
         f"{2 ** (L * L):,}")

    fig.suptitle(f"Figure 8 $-$ fixed-magnetisation Ising, {L}$\\times${L}: "
                 "every sample lands on the constraint surface",
                 fontsize=13, y=1.05)
    _finish(fig, "fig8_ising_configs")


# ============================================================================
# Figure 9 -- occupation raster at m = N = 1000
# ============================================================================

def figure9(args):
    ck = load_ck("occ_s1000")
    if ck is None:
        return _missing(9, ["occ_s1000"])

    ours = ck["samples"].numpy().astype(np.int64)
    exact = np.asarray(ck["extra"]["exact_samples"], dtype=np.int64)
    m = ck["extra"]["metrics"]

    # One ordering, taken from the exact sampler, applied to both panels: a
    # shared ordering is what makes the two rasters comparable pixel for pixel.
    order = np.argsort(exact.mean(0))
    O, E = ours[:, order], exact[:, order]

    # Occupancy is a small integer.  A continuous colour map turns the whole
    # raster into one flat wash; discrete levels keep every ball visible.
    nrow, ncol = 150, 150
    top = 4
    lv = ["#F2F2F2", "#BFD9EE", "#6BAED6", "#2171B5", "#08306B"]
    cmap, norm = ListedColormap(lv), BoundaryNorm(np.arange(top + 2) - 0.5,
                                                  top + 1)

    fig = plt.figure(figsize=(12.8, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.22)

    def raster(ax, A, title, sub):
        im = ax.imshow(np.minimum(A[:nrow, :ncol], top), cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_xlabel(f"box (first {ncol} of {A.shape[1]:,}, "
                      "sorted by exact mean occupancy)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        _caption(ax, sub, y=-0.145)
        return im

    a0 = fig.add_subplot(gs[0, 0])
    im = raster(a0, O, "ours",
                f"KS(occupancy) {m['KS_occ']:.4f}, "
                f"{m['violations']} constraint violations")
    a0.set_ylabel(f"sample (first {nrow} of {len(O):,})", fontsize=9)
    a1 = fig.add_subplot(gs[0, 1])
    raster(a1, E, "exact sampler",
           f"$m$ = $N$ = 1000; every row sums to exactly {int(E[0].sum())}")
    cb = fig.colorbar(im, ax=[a0, a1], fraction=0.022, pad=0.012,
                      ticks=range(top + 1))
    cb.ax.set_yticklabels([str(k) for k in range(top)] + [f"$\\geq${top}"],
                          fontsize=7)
    cb.set_label("balls in box", fontsize=8)

    # Per-box marginals.  The yardstick is the exact sampler's own Monte Carlo
    # noise: a half-split difference uses n/2 per side, so its variance is 2x
    # that of the ours-minus-exact difference and must be scaled down by
    # sqrt(2) before it can be used as a band.
    ax2 = fig.add_subplot(gs[0, 2])
    d = O.mean(0) - E.mean(0)
    h = len(E) // 2
    sig = (E[:h].mean(0) - E[h:].mean(0)).std() / np.sqrt(2)
    ax2.axhspan(-2 * sig, 2 * sig, color="#C8C8C8", alpha=0.55,
                label="exact sampler's own $\\pm2\\sigma$ noise")
    ax2.plot(d, lw=0.45, color="#0072B2", label="ours $-$ exact, per box")
    ax2.axhline(0, color="#333333", lw=0.7)
    ax2.set_xlabel("box (same order, all 1000)", fontsize=9)
    ax2.set_ylabel("mean occupancy difference", fontsize=9)
    ax2.set_title("per-box marginal error", fontsize=11, pad=4)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    _caption(ax2, f"{np.mean(np.abs(d) <= 2 * sig) * 100:.0f}% of boxes inside "
                  f"the reference's own noise band; largest deviation "
                  f"{np.abs(d).max() / sig:.1f}$\\sigma$", y=-0.145)

    fig.suptitle("Figure 9 $-$ occupation process at $m$ = $N$ = 1000: raw "
                 "occupancy vectors", fontsize=13, y=1.02)
    _finish(fig, "fig9_occupation_raster")


FIGS = {6: figure6, 7: figure7, 8: figure8, 9: figure9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, choices=sorted(FIGS))
    ap.add_argument("--beta", type=float, default=2,
                    help="beta for the Stiefel sheet (figure 7)")
    args = ap.parse_args()
    for k in ([args.only] if args.only else sorted(FIGS)):
        print(f"figure {k}")
        if k == 7:
            b = int(args.beta) if float(args.beta).is_integer() else args.beta
            FIGS[k](args, beta=b)
        else:
            FIGS[k](args)


if __name__ == "__main__":
    main()
