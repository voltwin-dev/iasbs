# Appendix D.1.2 (draft) — The two occupation-process simulators

This draft closes a documented defect: Appendix D.1.1 states that discrete
IASBS admits at most one jump per grid interval, and Table 7 reports a step
count for every occupation row. Both statements are correct only for the
enumerated simulator. The large-`m` rows (`m = N = 32, 128, 1000`) use a
second, undocumented simulator, and there `steps` counts leap bins rather than
jumps. Nothing about the reported numbers changes; what changes is that the
step column stops meaning two different things without saying so.

## D.1.2.1 Simulator 1 — enumerated, one jump per bin

Used whenever the state space is enumerated, i.e. `m = N = 4`, `|X| = 35`
(§2.1, §2.2). Implemented as `simulate` in `iasbs/occupation.py`.

Per bin of width `dt = 1/steps`, at state `eta`:

1. form the controlled edge rates `u_t(eta -> eta - e_i + e_j)` on the
   `n_edges` legal transfers,
2. set `p = u dt`, rescale so `sum p <= 0.98`,
3. draw one categorical outcome from `[stay, p_1, ..., p_{n_edges}]`.

Exactly one particle moves, or none. This is the object Appendix D.1.1
describes, and the `steps` column of Table 7 is a genuine jump budget for the
`m = N = 4` rows.

## D.1.2.2 Simulator 2 — tau-leap, many jumps per bin

Used for `m = N = 32, 128, 1000` (§2.3, §2.4), where `|X|` is between `9.2e17`
and `1e600` and no enumeration exists. Implemented as `scale_step`.

The controller is parameterised in rank-1 log form,

    a_theta(t, eta, i, j) = alpha_i(t, eta) + beta_j(t, eta),

which is not a convenience approximation: `phi_t` is a semigroup applied to a
product-form `f_1`, so `log phi(eta - e_i + e_j) - log phi(eta)` is exactly of
this shape in the product-form limit, and `alpha = beta = 0` recovers the
reference process exactly.

Allowing the destination `j` to equal the source `i` as a no-op makes the
remaining `j != i` rates exactly `gamma/(m-1) eta_i exp(alpha_i + beta_j)`, so
the self-loop device is exact rather than approximate. Two consequences:

* the total departure rate from mode `i` factorises, so the number of particles
  leaving mode `i` in a bin is `Binomial(eta_i, p_i)` with
  `p_i = gamma/(m-1) exp(alpha_i) sum_j exp(beta_j) dt`;
* every departing particle draws its destination from the *same* law
  `softmax(beta)`, so all arrivals in a bin are a single
  `Multinomial(K, softmax(beta))` with `K = sum_i` departures.

Hence one bin can move many particles, and the update is `O(m)` rather than
`O(m^2)` or `O(|X|)`. Particle number is conserved exactly by construction
(`eta - dep + arr`), which is why the violation counts are exactly zero and are
not a learned property.

**Approximation content.** This is a tau-leap, not an exact CTMC simulation:
rates are frozen across a bin. Given the departure counts, the destination draw
is exact. `scale_step` additionally clamps `p_i` to `[0, 0.9]`, which is a bias
term that must be reported; it is inactive at every setting used in this paper
(below).

## D.1.2.3 Jump-count audit

`iasbs/occupation.py scale-jumps` replays a trained checkpoint and reports the
quantities that distinguish the two simulators. 2000 paths per row:

| m = N | steps | transfers/path mean (min–max) | max particles in one bin | max_i eta_i at t=1 mean (min–max) | peak p_i | clamp hits |
|---:|---:|---|---:|---|---:|---:|
| 32 | 128 | 127.5 (94–168) | 7 | 6.62 (3–17) | — | 0 |
| 128 | 128 | 513.1 (442–584) | 15 | 9.30 (4–24) | — | 0 |
| 1000 | 256 | 4103.8 (3868–4341) | 36 | 13.03 (8–26) | 1.6993e-01 | 0 / 2.50e8 |

Read against the one-jump-per-bin budget, the `m = 128` row already moves
4.0x more particles than it has bins, and the `m = 1000` row 16.0x more.

**The constraint this resolves.** Under simulator 1, a trajectory started from
the concentrated state `N e_c` and run for `steps` bins must satisfy
`max_i eta_i >= N - steps`, i.e. `>= 744` at `m = N = 1000` with 256 steps,
whereas the symmetric Dirichlet-multinomial target with `d = 0.5` has
`E_pi[max_i eta_i] = 14.47` (exact iid draw, 4000 samples). Simulator 1 at
these settings would therefore force `W1(max)/N >= 0.68`. Under simulator 2 the
bound does not apply: 2000 / 2000 audited paths finish below 744, with
`max_i eta_i` mean 13.03, and the reported `W1(max)/N = 0.00145` is attainable.

## D.1.2.4 Verification of the reported rows

Shipped checkpoint samples (`ckpt/occ_s1000.pt`, 4000 samples) re-scored
against a fresh exact iid Dirichlet-multinomial draw:

| metric | recomputed | Table 2 / 12 |
|---|---:|---:|
| W1(max)/N | 0.0014510 | 0.0014 |
| KS(occ) | 0.010289 | 0.0102 |
| KS(max) | 0.21700 | 0.2215 |
| E ours | 412.3196 | 412.32 |
| E exact | 405.5433 | 405.57 |
| violations | 0 / 4000 | 0 |

`E[max_i eta_i]`: ours 13.02, exact iid 14.47. The controlled sampler is
slightly under-dispersed in the maximum, which is what `KS(max) = 0.217`
records; the `W1` figure is small because the whole max-occupancy distribution
is concentrated near 13-15, not because the discrepancy is being hidden.

## D.1.2.5 Required edits elsewhere

1. Appendix D.1.1: scope the one-jump-per-bin statement to the enumerated
   simulator and forward-reference D.1.2.
2. Table 7: split the `steps` column into *jump bins* (`m = N = 4`) and *leap
   bins* (`m = N >= 32`), or add a simulator column.
3. Appendix D.3: state that the concentrated start `N e_c` is used with
   simulator 2 for the scale rows, so no `max_i eta_i >= N - steps` floor
   applies.
4. Report the departure-probability clamp and its audited inactivity wherever
   the leap is described.

## D.1.2.6 Reproduction

```bash
python iasbs/occupation.py scale-jumps --m 1000 --N 1000 --steps 256 \
    --batch 2000 --hidden 256 --ckpt-dir ckpt --tag occ_s1000 \
    --out json/results_occ_jumps_s1000.json
python iasbs/occupation.py scale-jumps --m 128 --N 128 --steps 128 \
    --batch 2000 --hidden 256 --ckpt-dir ckpt --tag occ_s128 \
    --out json/results_occ_jumps_s128.json
python iasbs/occupation.py scale-jumps --m 32 --N 32 --steps 128 \
    --batch 2000 --hidden 256 --ckpt-dir ckpt --tag occ_s32 \
    --out json/results_occ_jumps_s32.json
```
