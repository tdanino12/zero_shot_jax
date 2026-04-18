## Training Phases

### Phase 0 — Base Partner Training

Train a **base partner** via self-play. This agent serves as the partner in the Phase 1 population of socializers.

```bash
python marl_self_video.py --seed=0 --layout=cramped_room
```

### Phase 1 — population Training

Train a **population** that will be used to train the ego agents.

1. For symmetric layouts (e.g., cramped_room, counter_circuit, coord_ring)

```bash
bash run_phase1.sh
```

2. For non-symmetric layouts (e.g., forced_coord, asymm_advantages):

### Phase 2 — population Training

Train **the ego agent** .

```bash
python phase2_training.py --seed=0 --layout=cramped_room
```
-------------------------------------------

## Running Baseline Methods 

---

## ✨ Overcooked maps:

1. counter_circuit

2. cramped_room

3. coord_ring

4. forced_coord

5. asymm_advantages 
