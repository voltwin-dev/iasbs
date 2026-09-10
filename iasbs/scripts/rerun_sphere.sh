#!/bin/bash
cd "$(dirname "$0")/../.."   # repo root: json/, ckpt/ and fig/ live here
set -x
PY=${PY:-python}
CK=ckpt
$PY -u iasbs/sphere.py exact --n-samples 200000 --ckpt-dir $CK --tag sphere \
    --out json/results_sphere_exact.json
for v in plain anti sym; do
  case $v in
    plain) FLAG="" ;;
    anti)  FLAG="--antithetic" ;;
    sym)   FLAG="--symmetrize" ;;
  esac
  $PY -u iasbs/sphere.py train $FLAG --seeds 5 --iters 4000 --inner 16 \
      --batch 8192 --mb 16384 --ema 0.9995 --eval-every 1000 \
      --n-samples 200000 --ckpt-dir $CK --tag sphere_$v \
      --out json/results_sphere_train_$v.json
done
