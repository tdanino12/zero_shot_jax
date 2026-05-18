#!/bin/bash
#SBATCH --job-name=pytorch_train
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt


source ~/miniconda3/etc/profile.d/conda.sh

conda activate esr_baselines_web
#export PYTHONPATH="/home/tom.danino/crossEnvCooperation:$PYTHONPATH"
export PYTHONPATH="/home/tom.danino/CEC/crossEnvCooperation:$PYTHONPATH"

python web_app.py 'counter_circuit' 
#python baselines/CEC/cross_algo.py
#python baselines/CEC/fcp_general.py SEED=30
#python baselines/CEC/e3t.py SEED=30
#python baselines/CEC/ippo_general.py SEED=30



