#!/bin/bash

seeds1=(0 1) #3)
#seed1_b=(4 5 6 9)
seeds2=(8 7 2) #11)
seeds3=(18 12 13 ) # 14 15 16 17)
seed4=(19 20) #21)
emps=(0.2 0.5)


# 0 achiever is a regular agent
achiever=(2e-5 5e-6 4e-6)
explorer=(0.1 0.35 0.75)
#discount=(0.99 0.5 0.2)

# Combine characteristic — each entry: (entropy, soc), (entropy,soc), (entropy,achiever)
combos=("0.25 0.15" "0.1 0.15" "0.25 2e-6")

layout=$1

echo "############################## achiever ################################"
# Achiever
for i in "${!seeds2[@]}"; do
    seed=${seeds2[$i]}
    achieve=${achiever[$i]}
    type="achiever"
    echo "Running with --seed $seed --partner_lr $achieve --type $type --layout $layout"
    python phase_1_training_general.py --seed $seed --partner_lr $achieve --layout $layout --type $type
done


echo "############################## Socalizer ################################"
# Socalizer
for i in "${!seeds1[@]}"; do
    seed=${seeds1[$i]}
    #seed2=${seed1_b[$i]}
    emp=${emps[$i]}
    type="socalizer"
    echo "Running with --seed $seed --emp_reward_scale $emp --type $type --layout $layout"
    python phase_1_training.py --seed $seed --emp_reward_scale $emp --type $type --layout $layout --self_play_seed 40
    #python phase_1_training.py --seed $seed2 --emp_reward_scale $emp --type $type --layout $layout --self_play_seed 50
done

echo "############################## Explorer ################################"
# Explorer
for i in "${!seeds3[@]}"; do
    seed=${seeds3[$i]}
    explore=${explorer[$i]}
    type="explorer"
    echo "Running with --seed $seed --type $type --layout $layout --entropy_0 $explore"
    python phase_1_training_general.py --seed $seed --layout $layout --type $type --entropy_0 $explore --entropy_1 0.1
done


echo "############################## Combination ################################"
python phase_1_training_general.py --seed 19 --layout $layout --type "explorer" --partner_lr 2e-6 --entropy_0 0.1 --entropy_1 0.01
python phase_1_training.py --seed 20 --emp_reward_scale 0.15 --type "socalizer" --layout $layout --self_play_seed 40 --ent_coef 0.1
#python phase_1_training.py --seed 21 --emp_reward_scale 0.15 --type "socalizer" --layout $layout --self_play_seed 40 --ent_coef 0.25
