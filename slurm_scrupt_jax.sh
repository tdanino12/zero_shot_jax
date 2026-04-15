#!/bin/bash
#SBATCH --gpus=1
source ~/miniconda3/etc/profile.d/conda.sh

conda activate esr #esr_baselines #esr
export PYTHONPATH="/home/tom.danino/zero_shot_jax:$PYTHONPATH"

#python phase_1_training.py --seed 3 --emp_reward_scale 0.4 --type socalizer --layout counter_circuit --self_play_seed 50
#python phase_1_training_general.py --seed 8 --layout "coord_ring"
#python phase_1_training_general.py --seed 8 --partner_lr 0.125 --layout "coord_ring" 
#python phase_1_training_general.py --seed 8 --partner_lr 5e-6 --layout "coord_ring"

python phase2_training_test4.py --seed 30 --layout $1
#python marl_self_video.py --layout $1 --seed $2
#bash run_phase1.sh $1 
