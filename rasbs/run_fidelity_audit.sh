#!/bin/bash
# Phase 1 of the remaining-experiments plan: is the one-pole collapse in this
# port a property of R-ASBS, or an artefact of nn.Linear's nonzero default
# bias breaking the x_3 -> -x_3 symmetry at initialisation?
#
# Every run below is the SAME code as the collapsing baseline except for
# --init matlab (Glorot weights, zero bias, as MATLAB fullyConnectedLayer).
# No bridge, loss, optimizer, width, sigma, step count or geometry is touched.
#
#   R1  uniform audit  E = 0, target Haar.  Must NOT choose a pole.
#   R2  vmf audit      E = -6 x_3, one mode.  north -> 0.9975274, <x3> -> 0.8333505.
#   R3  bimodal, 5 seeds, published settings.  Baseline north was
#       0.00004 / 0.00004 / 0.99762 / 0.00173 / 0.00191.
cd "$(dirname "$0")/.."          # repo root: json/, ckpt/ and fig/ live here
PY=${PY:-python}
mkdir -p rasbs/logs

run () {   # run <gpu> <logname> <args...>
    local gpu=$1 name=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u "$@" \
        > "rasbs/logs/${name}.log" 2>&1 &
    echo "  launched ${name} on cuda:${gpu} (pid $!)"
}

echo "R1/R2 audits + R3 five bimodal seeds, --init matlab"

run 0 audit_uniform rasbs/rasbs_sphere_audit.py --test uniform --init matlab \
    --tag rasbs_audit_uniform_mi \
    --out json/results_rasbs_audit_uniform_mi.json
run 0 audit_vmf     rasbs/rasbs_sphere_audit.py --test vmf --init matlab \
    --tag rasbs_audit_vmf_mi \
    --out json/results_rasbs_audit_vmf_mi.json

for s in 0 1; do
    run 0 bimodal_mi_s$s rasbs/rasbs_sphere_port.py --problem bimodal \
        --seed $s --init matlab --tag rasbs_sphere_matlabinit_s$s \
        --out json/results_rasbs_sphere_matlabinit_s$s.json
done
for s in 2 3 4; do
    run 1 bimodal_mi_s$s rasbs/rasbs_sphere_port.py --problem bimodal \
        --seed $s --init matlab --tag rasbs_sphere_matlabinit_s$s \
        --out json/results_rasbs_sphere_matlabinit_s$s.json
done

wait
echo "all fidelity runs finished"
