#!/bin/bash
# Re-run every headline experiment with --ckpt-dir so the learned control and
# the raw terminal samples are on disk for figure generation.
set -x
PY=/root/miniconda3/envs/SML_env/bin/python
CK=ckpt
cd "$(dirname "$0")"

# --- Phase 2a: fixed Ising -------------------------------------------------
$PY -u fixed_ising.py train --iters 1500 --eval-every 100 --n-samples 20000 \
    --loss poisson --ckpt-dir $CK --tag ising_poisson --out results_ising_poisson.json
$PY -u fixed_ising.py train --iters 1500 --eval-every 100 --n-samples 20000 \
    --loss mse --ckpt-dir $CK --tag ising_mse --out results_ising_mse.json
$PY -u fixed_ising.py train --steps 256 --iters 3000 --eval-every 200 --n-samples 20000 \
    --loss poisson --ckpt-dir $CK --tag ising_poisson256 --out results_ising_poisson256.json

# --- Phase 2b: occupation, enumerated m=N=4 --------------------------------
for est in full uniform occupancy; do
  $PY -u occupation.py train --m 4 --iters 1500 --eval-every 250 --n-samples 20000 \
      --estimator $est --ckpt-dir $CK --tag occ4_$est --out results_occ_$est.json
done
$PY -u occupation.py train --m 4 --iters 1500 --eval-every 250 --n-samples 20000 \
    --estimator full --loss mse --ckpt-dir $CK --tag occ4_mse --out results_occ_mse.json

# --- Phase 2b: occupation at scale -----------------------------------------
$PY -u occupation.py scale --m 32   --iters 3000 --eval-every 500 --n-samples 20000 \
    --ckpt-dir $CK --tag occ_s32   --out results_occ_s32.json
$PY -u occupation.py scale --m 128  --iters 3000 --eval-every 500 --n-samples 10000 \
    --ckpt-dir $CK --tag occ_s128  --out results_occ_s128.json
$PY -u occupation.py scale --m 1000 --steps 256 --batch 128 --mb 512 \
    --iters 1500 --eval-every 250 --n-samples 4000 \
    --ckpt-dir $CK --tag occ_s1000 --out results_occ_s1000.json
