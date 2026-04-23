layout=$1
seed=$2

# LR achieveres
python phase_2_training_opposite_and_self_play.py --seed $seed --layout $layout --MOE 3 --load_dir "phase1_lr"

# reward achiever MOE 1
python phase_2_training_opposite_and_self_play.py --seed $seed --layout $layout --MOE 1 --load_dir "phase1_r_achiever"

# reward achiever MOE 3
python phase_2_training_opposite_and_self_play.py --seed $seed --layout $layout --MOE 3 --load_dir "phase1_r_achiever"

# reward achiever MOE 5
python phase_2_training_opposite_and_self_play.py --seed $seed --layout $layout --MOE 5 --load_dir "phase1_r_achiever"

# reward achiever MOE 7
python phase_2_training_opposite_and_self_play.py --seed $seed --layout $layout --MOE 7 --load_dir "phase1_r_achiever"
