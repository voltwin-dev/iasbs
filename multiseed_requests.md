# Multi-seed requests

Every headline number in the paper that is currently reported from a single
training run, together with the extra seeds needed to turn it into a mean and a
spread.  Target is **3 seeds everywhere**.  The "exact run/config to repeat"
column is the configuration the existing seed was trained under -- the added
seeds must reuse it verbatim, changing only the seed.

| Experiment | Method | Current seeds | Add | Target total | Exact run/config to repeat | Section 6 headline metrics | Appendix C metrics/details |
|---|---|---|---|---|---|---|---|
| Ising $4\times4$ | IASBS Dirac | 1 | +2 | 3 | 256 steps, 3000 outer iters, 40 inner updates, batch 2048, minibatch 1024 | exact TV | TV, KL, Hellinger, energy-hist TV, $\mathbb{E}[E]$, ESS fraction, violations, wall time |
| Ising $4\times4$ | IASBS non-Dirac | 1 | +2 | 3 | 256 steps, 3000 iters, same controller, 40 corrector updates | exact TV | exact TV, corrector RMSE/MAE, violations, wall time |
| Ising $4\times4$ | DAM **(LAST, 6.0 h)** | 1 | +2 | 3 | $K=32$, 128 steps, 5000 iters | exact TV | TV, energy-hist TV, ESS/K, ESS mean/p10, clipping counts, endpoint-$f_1$ evals, CTMC jumps, wall time |
| Ising $5\times5$ | IASBS Dirac | 1 | +2 | 3 | 512 steps, 6000 outer iters, 40 inner updates, batch 2048, minibatch 1024 | exact TV | TV, KL, Hellinger, energy-hist TV, $\mathbb{E}[E]$, ESS fraction, violations, wall time |
| Ising $5\times5$ | IASBS non-Dirac | 1 | +2 | 3 | 256 steps, 3000 iters, 40 corrector updates | exact TV | exact TV, corrector RMSE/MAE, violations, wall time |
| Ising $5\times5$ | DAM **(LAST, 6.4 h)** | 1 | +2 | 3 | $K=32$, 128 steps, 5000 iters | exact TV | TV, energy-hist TV, ESS/K, ESS mean/p10, clipping counts, endpoint-$f_1$ evals, CTMC jumps, wall time |
| Occupation $m=N=4$ | IASBS Dirac | 1 | +2 | 3 | 128 steps, 1500 iters, batch 512, minibatch 1024 | exact TV | TV, occupancy-hist TV, max-occupancy TV, mean energy, violations |
| Occupation $m=N=4$ | IASBS non-Dirac uniform | 1 | +2 | 3 | 128 steps, 1500 iters, source $\rho=1$, 4 corrector updates | exact TV | TV, occupancy/max-occupancy errors, corrector max/RMSE error, mean energy, violations |
| Occupation $m=N=4$ | DAM **(LAST, 6.1 h)** | 1 | +2 | 3 | $K=16$, 128 steps, 1200 iters | exact TV | TV, occupancy-hist TV, ESS/K, ESS mean/p10, endpoint-$f_1$ evals, CTMC jumps, wall time |
| Occupation $m=N=32$ | IASBS Dirac | 1 | +2 | 3 | 128 steps, 3000 iters, batch 512, minibatch 1024 | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + mean energy, violations, wall time |
| Occupation $m=N=32$ | IASBS non-Dirac | 2 | +1 | 3 | 128 steps, 3000 iters, batch 512, minibatch 1024 | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + mean energy, violations, corrector diagnostics |
| Occupation $m=N=32$ | DAM **(LAST, 8.3 h)** | 1 | +2 | 3 | $K=64$, 128 steps, 1000 iters | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + ESS mean/p10, clipping/nonfinite counts, endpoint-$f_1$ evals, jumps, wall time |
| Occupation $m=N=128$ | IASBS Dirac | 1 | +2 | 3 | 128 steps, 3000 iters, batch 512, minibatch 1024 | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + mean energy, violations, wall time |
| Occupation $m=N=128$ | IASBS non-Dirac | 1 | +2 | 3 | 128 steps, 3000 iters, corrector LR $10^{-2}$ | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + mean energy, violations, corrector diagnostics |
| Occupation $m=N=128$ | DAM **(LAST, 36.5 h)** | 1 | +2 | 3 | $K=64$, 128 steps, 500 iters | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + ESS collapse stats, clipped labels, nonfinite weights, endpoint-$f_1$ evals, jumps, wall time |
| Occupation $m=N=1000$ | IASBS Dirac | 1 | +2 | 3 | 256 steps, 1500 iters, batch 128, minibatch 512 | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + mean energy, violations, wall time |
| Occupation $m=N=1000$ | IASBS non-Dirac | 1 | +2 | 3 | 256 steps, 1500 iters, batch 512, minibatch 512 | $\mathrm{KS}_{\rm occ}$, $W_1(\max)/N$ | same + mean energy, violations, corrector diagnostics |
| Fixed-support toy $(14,4,4)$ | IASBS Dirac | 1 | +2 | 3 | 256 steps, 3000 iters, batch 2048, minibatch 1024 | exact TV | TV, KL, Hellinger, mean energy, wall time, violations |
| Fixed-support toy $(14,4,4)$ | IASBS non-Dirac | 1 | +2 | 3 | same + 40 corrector updates | exact TV | same + corrector RMSE/MAE/max error |
| Fixed-support toy $(14,4,4)$ | DAM | 1 | +2 | 3 | $K=16$, 256 steps, 1200 iters | exact TV | TV, KL, Hellinger, mean energy, ESS, endpoint-$f_1$ evals, jumps, wall time, violations |
| GB1 $k=3$ | IASBS Dirac | 1 | +2 | 3 | 256 steps, 1000+1000+1000 iters at $\tau=2,1.4,1$ | exact TV, $\mathbb{E}[F]$ | TV, KL, Hellinger, mean fitness, wall time, violations |
| GB1 $k=3$ | IASBS non-Dirac | 1 | +2 | 3 | same annealed schedule + corrector | exact TV, $\mathbb{E}[F]$ | same + corrector RMSE/MAE |
| GB1 $k=3$ | DAM | 1 | +2 | 3 | $K=16$, 256 steps, 400+400+400 at $\tau=2,1.4,1$ | exact TV, $\mathbb{E}[F]$ | TV, KL, Hellinger, mean fitness, ESS, endpoint-$f_1$ evals, jumps, wall time, violations |
| Stiefel trace $\beta=1.3$ | IASBS default | 1 | +2 | 3 | current default config | $\lvert\Delta E\rvert$, KS(E), relative spread | same + oracle calls, wall time |
| Stiefel trace $\beta=2$ | IASBS default | 1 | +2 | 3 | current default config | $\lvert\Delta E\rvert$, KS(E), relative spread | same + oracle calls, wall time |
| Stiefel trace $\beta=5$ | IASBS default | 1 | +2 | 3 | current default config | $\lvert\Delta E\rvert$, KS(E), relative spread | same + oracle calls, wall time |
| Stiefel trace $\beta=1.3$ | IASBS600 | 1 | +2 | 3 | 600k terminal-oracle budget | $\lvert\Delta E\rvert$, KS(E) | same + exact oracle budget, wall time |
| Stiefel trace $\beta=2$ | IASBS600 | 1 | +2 | 3 | 600k terminal-oracle budget | $\lvert\Delta E\rvert$, KS(E) | same + exact oracle budget, wall time |
| Stiefel trace $\beta=5$ | IASBS600 | 1 | +2 | 3 | 600k terminal-oracle budget | $\lvert\Delta E\rvert$, KS(E) | same + exact oracle budget, wall time |
| Stiefel trace $\beta=1.3$ | R-ASBS | 1 | +2 | 3 | current matched/native baseline | $\lvert\Delta E\rvert$, KS(E), relative spread | same + oracle budget, wall time |
| Stiefel trace $\beta=2$ | R-ASBS | 1 | +2 | 3 | current matched/native baseline | $\lvert\Delta E\rvert$, KS(E), relative spread | same + oracle budget, wall time |
| Stiefel trace $\beta=5$ | R-ASBS | 1 | +2 | 3 | current matched/native baseline | $\lvert\Delta E\rvert$, KS(E), relative spread | same + oracle budget, wall time |

## Run order

Five DAM rows are marked **LAST** in the table above.  Together they are 63 h
of the 100 h serial estimate -- occupation $m=128$ alone is 36.5 h, because
DAM needs 18.8 h per seed there and the seed-0 result is already a documented
estimator collapse (ESS 1.69 / 64).  Nothing else depends on them, so they
should be queued after everything else has landed.

| set | serial | on 2 GPUs |
|---|---|---|
| whole table | 100 h | 50 h |
| without the five LAST rows | 37 h | 18 h |
| the five LAST rows alone | 63 h | 32 h |

The four remaining DAM rows (GB1 $k=3$ 3.9 h, fixed-support toy 3.8 h) are
short enough to run with the main batch.

## Running the added seeds

`bash iasbs/scripts/multiseed.sh <row|group:X|all> [seeds] [drop|keep]`
re-issues each row's recorded flags with `--seed S --tag <base>_sS --out
json/results_<base>_sS.json`, so nothing collides with the seed-0 artifacts.
`bash iasbs/scripts/multiseed.sh list` prints the row names.

The third argument controls mid-run evaluation and defaults to `drop`, which
appends `--eval-every 1000000000`.  Every evaluation site in the four discrete
mains is guarded by `it % args.eval_every == 0 or it == args.iters`, so the
final evaluation still runs and none of the headline numbers above change --
each is taken from the last evaluation, never from a minimum over the training
history.  Only the TV-vs-iteration curve is lost, and seed 0 already has it.
On Ising $5\times5$ non-Dirac that removes 120 exact propagations over $2^{25}$
states, about 9 h per seed; over the whole table it takes the estimated
two-GPU wall from roughly 64 h to roughly 50 h.  Pass `keep` to reproduce a
row's original evaluation cadence as well.

Collapse the per-seed files with
`python iasbs/scripts/aggregate_seeds.py <base> [...]`
(add `--stage C` for the GB1 chain).

## Already in flight

The last nine rows -- the Stiefel trace legs at $\beta \in \{1.3, 2, 5\}$ for
IASBS600 and R-ASBS -- are covered by the fair-comparison run currently on the
box:

- `bash rasbs/run_fair.sh trace` -> `json/results_rasbs_m600.json`, 3 seeds per $\beta$
- `bash iasbs/scripts/run_fair_stiefel.sh trace600` -> `json/results_stiefel_matched_s3.json`, 3 seeds per $\beta$

The frame-sensitive St(4,2) legs (`stiefel_frame_s5`, `stiefel_frame600`,
`rasbs_frame`) already run 5 seeds and are not listed above.
