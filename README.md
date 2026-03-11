## Training Phases

### Phase 0 — Base Partner Training

Train a **base partner** with initial skills. This agent serves as the seed for each partner in the Phase 1 population.

```bash
python marl_self_phase_0 --seed=0 --layout=cramped_room
```

### Phase 1 — population Training

Train a **population** that will be used as training parntner to the ego agents.

```bash
bash marl_self_phase_0 --seed=0 --layout=cramped_room
```
