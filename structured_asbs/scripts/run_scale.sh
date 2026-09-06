cd "$(dirname "$0")/../.."   # repo root: json/, ckpt/ and fig/ live here
set -x
P="conda run -n SML_env python -u structured_asbs/occupation.py"
echo "### scale m=N=32"
$P scale --m 32  --iters 3000 --eval-every 500 --n-samples 20000 --out json/results_occ_s32.json
echo "### scale m=N=128"
$P scale --m 128 --iters 3000 --eval-every 500 --n-samples 10000 --out json/results_occ_s128.json
echo "### scale m=N=1000"
$P scale --m 1000 --steps 256 --iters 1500 --eval-every 250 --batch 128 --mb 512 \
    --n-samples 4000 --out json/results_occ_s1000.json
echo "### scalevar m=128"
$P scalevar --m 128 --mb 2048 --reps 400 --out json/results_occ_var128.json
echo "### scalevar m=1000"
$P scalevar --m 1000 --mb 1024 --reps 200 --out json/results_occ_var1000.json
