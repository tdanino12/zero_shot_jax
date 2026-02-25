#!/bin/bash

seeds1=(1 2 3)
seeds2=(4 5 6)
seeds3=(7 8 9)
seeds4=(10 11 12)

emps=(10 20 30)
risks=(0.1 0.5 0.9)
discount=(0.99,0.95,0.93)

# Combine characteristic — each entry: "seed emp risk"
combos=("13 10 0.1 0.2" "14 20 0.5 0.3" "15 30 0.9 0.4")

# Socalizer
for i in "${!seeds1[@]}"; do
    seed=${seeds1[$i]}
    emp=${emps[$i]}
    echo "Running with --seed $seed --emp_reward_scale $emp"
    python phase_1_training.py --seed $seed --emp_reward_scale $emp --risk_reward_scale 0 --gamma 0.99
done

# Myopic
for i in "${!seeds2[@]}"; do
    seed=${seeds2[$i]}
    myopic=${discount[$i]}

    echo "Running with --seed $seed --gamma $myopic"
    python phase_1_training.py --seed $seed --emp_reward_scale 0 --risk_reward_scale 0 --gamma $myopic
done


# Risk
for i in "${!seeds3[@]}"; do
    seed=${seeds[$i]}
    risk=${risks[$i]}

    echo "Running with --seed $seed --risk_reward_scale $risk"
    python phase_1_training.py --seed $seed --emp_reward_scale 0 --risk_reward_scale $risk --gamma 0.99
done

# Combination
for entry in "${combos[@]}"; do
    read -r seed emp risk myopic<<< "$entry"
    echo "Running with --seed $seed --emp_reward_scale $emp --risk_reward_scale $risk --gamma $myopic"
    python phase_1_training.py --seed $seed --emp_reward_scale $emp --risk_reward_scale $risk --gamma $myopic
done
