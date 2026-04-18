#!/bin/bash
#SBATCH --gpus=1
source ~/miniconda3/etc/profile.d/conda.sh

conda activate esr #esr_baselines #esr
export PYTHONPATH="/home/tom.danino/zero_shot_jax:$PYTHONPATH"

python MEP_all_phases.py --seed $1 --layout $2 
#bash run_phase1_fcp.sh $1 
