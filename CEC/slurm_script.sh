#!/bin/bash
#SBATCH --gpus=1
source ~/miniconda3/etc/profile.d/conda.sh

conda activate esr_baselines
#export PYTHONPATH="/home/tom.danino/crossEnvCooperation:$PYTHONPATH"
export PYTHONPATH="/home/tom.danino/CEC/crossEnvCooperation:$PYTHONPATH"

python baselines/CEC/cross_algo.py
#python baselines/CEC/fcp_general.py SEED=30
#python baselines/CEC/e3t.py SEED=30
#python baselines/CEC/ippo_general.py SEED=27



