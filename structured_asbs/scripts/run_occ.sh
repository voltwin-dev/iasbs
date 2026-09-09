cd "$(dirname "$0")/../.."   # repo root: json/, ckpt/ and fig/ live here
set -x
P="conda run -n SML_env python -u structured_asbs/occupation.py train --iters 1500 --eval-every 250 --n-samples 20000"
echo "### full-sum label, bregman"
$P --estimator full     --loss bregman --out json/results_occ_full.json
echo "### full-sum label, mse (ablation)"
$P --estimator full     --loss mse     --out json/results_occ_mse.json
