## Training Phases

### Phase 0 — Base Partner Training

Train a **base partner** with initial skills. This agent serves as the seed for each partner in the Phase 1 population.

```bash
python marl_self_video.py --seed=0 --layout=cramped_room
```

### Phase 1 — population Training

Train a **population** that will be used as training parntner to the ego agents.

```bash
bash run_phase1.sh
```

### Phase 2 — population Training

Train **the ego agent** .

```bash
python phase2_training.py --seed=0 --layout=cramped_room
```


---

## ✨ Overcooked maps:

1. counter_circuit

2. cramped_room

3. coord_ring

4. forced_coord

5. asymm_advantages 
