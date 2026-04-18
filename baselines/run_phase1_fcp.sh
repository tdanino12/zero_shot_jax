#!/bin/bash
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

layout=$1

for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    echo "Running with --seed $seed --layout $layout"
    python fcp_stage1.py --seed $seed --layout $layout
done
