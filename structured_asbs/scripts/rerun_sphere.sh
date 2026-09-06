#!/bin/bash
cd "$(dirname "$0")/.."
set -x
PY=/root/miniconda3/envs/SML_env/bin/python
CK=ckpt
cd "$(dirname "$0")"
$PY -u sphere.py exact --n-samples 200000 --ckpt-dir $CK --tag sphere \
    --out json/results_sphere_exact.json
for v in plain anti sym; do
  case $v in
    plain) FLAG="" ;;
    anti)  FLAG="--antithetic" ;;
    sym)   FLAG="--symmetrize" ;;
  esac
  $PY -u sphere.py train $FLAG --seeds 5 --iters 4000 --inner 16 \
      --batch 8192 --mb 16384 --ema 0.9995 --eval-every 1000 \
      --n-samples 200000 --ckpt-dir $CK --tag sphere_$v \
      --out json/results_sphere_train_$v.json
done
