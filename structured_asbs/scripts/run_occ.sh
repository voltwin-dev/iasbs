cd "$(dirname "$0")/.."
set -x
P="conda run -n SML_env python -u occupation.py train --iters 1500 --eval-every 250 --n-samples 20000"
echo "### full-sum label, bregman"
$P --estimator full     --loss bregman --out json/results_occ_full.json
echo "### occupancy 1-sample, bregman"
$P --estimator occupancy --n-a 1 --loss bregman --out json/results_occ_occ1.json
echo "### uniform 1-sample, bregman"
$P --estimator uniform  --n-a 1 --loss bregman --out json/results_occ_unif1.json
echo "### full-sum label, mse (ablation)"
$P --estimator full     --loss mse     --out json/results_occ_mse.json
