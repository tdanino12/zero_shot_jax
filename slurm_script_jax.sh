#!/bin/bash
#SBATCH --gpus=1
source ~/miniconda3/etc/profile.d/conda.sh

conda activate esr #esr_baselines #esr
export PYTHONPATH="/home/tom.danino/zero_shot_jax:$PYTHONPATH"


python phase2_training.py --seed 30 --layout $1
#python marl_self_video.py --layout $1 --seed $2
#bash run_phase1.sh $1 
