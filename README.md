## Training Phases

### Phase 0 — Base Partner Training

Train a **base partner** via self-play. This agent serves as the partner in the Phase 1 population of socializers.

```bash
python marl_self_video.py --seed=0 --layout=cramped_room
```

### Phase 1 — population Training

Train a **population** that will be used to train the ego agents.

1. For symmetric layouts (e.g., cramped_room, counter_circuit, coord_ring):

```bash
bash run_phase1.sh
```

2. For non-symmetric layouts (e.g., forced_coord, asymm_advantages):
```bash
bash run_phase1_opposite_agent0.sh  # train first half of the agents, as agent 0
bash run_phase1_opposite_agent1.sh  # train the second half of the agents, as agent 1
```

### Phase 2 — Training an Ego Agent

Train **the ego agent** .

```bash
python phase2_training.py --seed=0 --layout=cramped_room
```
-------------------------------------------

## Running Baseline Methods 

### MEP

In MEP all training is condecuted by running a single file.

1. For symmetric layouts (e.g., cramped_room, counter_circuit, coord_ring):

```bash
python baselines/MEP_all_phases.py --seed=0 --layout=cramped_room
```

2. For non-symmetric layouts (e.g., forced_coord, asymm_advantages):
```bash
python baselines/phase_2_training_opposite_and_self_play.py --seed=0 --layout=cramped_room
```

### FCP

### Phase 1 — population Training

Either run manually for all seeds (in FCP, you run each population agent with a different seed):

```bash
python baselines/fcp_stage1.py --seed=0 --layout=cramped_room
```

or use the bash file that iterates over all seeds:

```bash
python baselines/run_phase1_fcp.sh --layout=cramped_room
```

### Phase 2 — Training an Ego Agent

```bash
python baselines/phase_2_training_fcp.py --seed=0 --layout=cramped_room
```

-------------------------------------------


### ✨ Overcooked maps:

1. counter_circuit

2. cramped_room

3. coord_ring

4. forced_coord

5. asymm_advantages 
