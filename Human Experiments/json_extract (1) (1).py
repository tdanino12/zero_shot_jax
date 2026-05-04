"""
plot_crossplay.py
-----------------
Reads all counter_circuit debug files (MessagePack, length-prefixed) found in
the same directory, aggregates per-method rewards per layout across all
participants, and produces a "Cross Play Performance" bar chart matching the
style in the paper figure.

Usage:
    python plot_crossplay.py                      # reads *.json in current dir
    python plot_crossplay.py --dir /path/to/data  # reads *.json in given dir
    python plot_crossplay.py --out figure.png     # custom output path

Requirements:
    pip install msgpack numpy matplotlib
"""

import struct
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import msgpack
except ImportError:
    print("Install msgpack first:  pip install msgpack")
    sys.exit(1)


# ── Configuration ────────────────────────────────────────────────────────────────

# Map raw name-prefix → display label.  Edit here to rename methods.
METHOD_MAP = {
    "my":          "Ours",
    "sk_e3t":      "E3T",
    "fcp":         "FCP",
    "ik_finetune": "IK",
    "mep":         "MEP",
    "hsp":         "HSP",
}

# Bar colours in display order (same order as METHOD_MAP values)
METHOD_COLORS = {
    "Ours": "#4472C4",   # blue
    "E3T":  "#ED7D31",   # orange
    "FCP":  "#70AD47",   # green
    "IK":   "#FF0000",   # red
    "MEP":  "#9B59B6",   # purple (if present)
    "HSP":  "#17A2B8",   # teal
}

# Map raw selected_layout prefix → display name (trailing _N stripped first)
LAYOUT_MAP = {
    "counter_circuit":  "Counter Circuit",
    "cramped_room":     "Cramped Room",
    "coord_ring":       "Coord Ring",
    "forced_coord":     "Forced Coord",
    "asymm_advantages": "Asymm Advantages",
}

# Fixed display order for layouts (x-axis)
LAYOUT_ORDER = list(LAYOUT_MAP.values())

# ── File loading ─────────────────────────────────────────────────────────────────

def load_records(path: Path):
    """Load all MessagePack records from a length-prefixed binary file."""
    records = []
    raw = path.read_bytes()
    pos = 0
    while pos + 4 <= len(raw):
        length = struct.unpack(">I", raw[pos: pos + 4])[0]
        pos += 4
        if pos + length > len(raw):
            break
        chunk = raw[pos: pos + length]
        pos += length
        try:
            records.append(msgpack.unpackb(chunk, raw=False, strict_map_key=False))
        except Exception:
            pass
    return records


# ── JAX array decoding ────────────────────────────────────────────────────────────

def _decode_ext(ext) -> float:
    """Decode a JAX ExtType(code=1) scalar stored as [shape, dtype, bytes]."""
    inner = msgpack.unpackb(ext.data, raw=False, strict_map_key=False)
    shape, dtype_str, data = inner
    arr = np.frombuffer(data, dtype=np.dtype(dtype_str))
    return float(arr[0]) if arr.size == 1 else arr.tolist()


def decode_reward(value) -> float:
    """Return reward as a Python float regardless of storage format."""
    if hasattr(value, "data"):        # ExtType
        return _decode_ext(value)
    return float(value) if value is not None else 0.0


def decode_timestep_reward(ts_blob: bytes) -> float:
    """Unpack a timestep blob and return its reward."""
    ts = msgpack.unpackb(ts_blob, raw=False, strict_map_key=False)
    return decode_reward(ts.get("reward", 0.0))


# ── Name parsing ──────────────────────────────────────────────────────────────────

def parse_method(block_name: str) -> str | None:
    """
    Extract method key from a block name like 'sk_e3t_counter_circuit2'.
    Matches longest prefix in METHOD_MAP.
    """
    best = None
    best_len = 0
    for prefix in METHOD_MAP:
        if block_name.startswith(prefix) and len(prefix) > best_len:
            best = prefix
            best_len = len(prefix)
    return best


def parse_layout(selected_layout: str) -> str | None:
    """
    Convert 'counter_circuit_9' → 'Counter Circuit' via LAYOUT_MAP.
    Strips trailing _<digits> first.
    """
    # strip trailing _<digits>
    parts = selected_layout.rsplit("_", 1)
    key = parts[0] if len(parts) == 2 and parts[1].isdigit() else selected_layout
    return LAYOUT_MAP.get(key)


# ── Per-file extraction ───────────────────────────────────────────────────────────

def extract_session(records: list, source_path: Path | None = None) -> dict:
    """
    Return {method_label: total_reward} for one participant file.
    Also returns the layout display name.
    """
    # Layout from user_storage in the last record that has it
    layout_raw = None
    for r in reversed(records):
        us = r.get("user_storage", {})
        if isinstance(us, dict) and "selected_layout" in us:
            layout_raw = us["selected_layout"]
            break

    layout = parse_layout(layout_raw) if layout_raw else None

    # Fallback: infer layout from filename or block names
    if layout is None:
        candidates = []
        if source_path is not None:
            candidates.append(source_path.stem)
        for r in records:
            n = r.get("name", "")
            if isinstance(n, str):
                candidates.append(n)
        for cand in candidates:
            for key, label in LAYOUT_MAP.items():
                if key in cand:
                    layout = label
                    break
            if layout:
                break

    # Sum rewards per block name
    rewards_by_name: dict[str, float] = defaultdict(float)
    for r in records:
        meta = r.get("metadata", {})
        if not isinstance(meta, dict) or meta.get("type") != "EnvStage":
            continue
        name = r.get("name", "")
        if "tutorial" in name:
            continue
        ts_blob = r.get("data", {}).get("timestep")
        if not isinstance(ts_blob, bytes):
            continue
        try:
            rewards_by_name[name] += decode_timestep_reward(ts_blob)
        except Exception:
            pass

    # Map raw names → display labels
    method_totals: dict[str, float] = {}
    for name, total in rewards_by_name.items():
        prefix = parse_method(name)
        if prefix:
            label = METHOD_MAP[prefix]
            method_totals[label] = method_totals.get(label, 0.0) + total

    return layout, method_totals


# ── Aggregation across files ──────────────────────────────────────────────────────

def aggregate_all(data_dir: Path) -> dict:
    """
    Walk every .json file in data_dir, parse it, and collect
    {layout: {method: [reward_per_participant, ...]}} .
    """
    layout_method_rewards: dict[str, dict[str, list]] = {
        layout: {method: [] for method in METHOD_MAP.values()}
        for layout in LAYOUT_MAP.values()
    }

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"[WARNING] No .json files found in {data_dir}")
        return layout_method_rewards

    for path in json_files:
        print(f"  Reading {path.name} …", end=" ")
        try:
            records = load_records(path)
            layout, method_totals = extract_session(records, source_path=path)
            if layout is None:
                print("(no layout found, skipping)")
                continue
            if not any(v != 0 for v in method_totals.values()):
                print(f"(all-zero session, skipping)")
                continue
            for method, total in method_totals.items():
                layout_method_rewards[layout][method].append(total)
            print(f"layout={layout!r}  methods={list(method_totals)}")
        except Exception as e:
            print(f"(error: {e})")

    return layout_method_rewards


# ── Stats ─────────────────────────────────────────────────────────────────────────

def mean_sem(values: list) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    a = np.array(values, dtype=float)
    if len(a) > 1:
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
    return float(a.mean()), 0.0


# ── Plotting ──────────────────────────────────────────────────────────────────────

def plot_crossplay(layout_method_rewards: dict, out_path: Path):
    """
    Reproduce the "Cross Play Performance" grouped bar chart.
    Layouts are on the x-axis; one bar group per layout, one bar per method.
    """
    # Determine which methods and layouts are actually present
    all_methods = list(dict.fromkeys(
        m for lm in layout_method_rewards.values() for m in lm
        if m in METHOD_COLORS
    ))
    # Preserve preferred order: keys of METHOD_MAP values
    preferred_order = list(METHOD_MAP.values())
    all_methods = [m for m in preferred_order if m in all_methods] + \
                  [m for m in all_methods if m not in preferred_order]

    present_layouts = [l for l in LAYOUT_ORDER if l in layout_method_rewards]
    # If no data at all, fall back to all known layouts for example
    if not present_layouts:
        present_layouts = LAYOUT_ORDER

    n_layouts = len(present_layouts)
    n_methods = len(all_methods)

    bar_width = 0.18
    group_gap  = 0.05
    group_width = n_methods * bar_width + group_gap
    x_centers  = np.arange(n_layouts) * group_width

    fig, ax = plt.subplots(figsize=(max(8, n_layouts * 2.0), 5))

    for m_idx, method in enumerate(all_methods):
        color = METHOD_COLORS.get(method, "#888888")
        means, sems = [], []
        for layout in present_layouts:
            vals = layout_method_rewards.get(layout, {}).get(method, [])
            m, s = mean_sem(vals)
            means.append(m)
            sems.append(s)

        offsets = x_centers + (m_idx - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets, means,
            width=bar_width,
            color=color,
            label=method,
            zorder=3,
        )
        ax.errorbar(
            offsets, means, yerr=sems,
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1.2,
            zorder=4,
        )

    ax.set_title("Human Experiments", fontsize=13, fontweight="bold")
    ax.set_xlabel("Layouts", fontsize=11)
    ax.set_ylabel("Average Reward", fontsize=11)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(present_layouts, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(title=None, fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    print(f"\n  Chart saved → {out_path}")


# ── Example data injection ────────────────────────────────────────────────────────

def inject_example_data(layout_method_rewards: dict) -> dict:
    """
    When real data is sparse (e.g. only one layout, few participants, zero
    rewards), overlay plausible synthetic values so the chart is illustrative.
    Real data is kept wherever it is non-zero.

    Approximate values read from the paper figure:
        Ours ~153, E3T ~163, FCP ~84, IK ~80  (all layouts similar)
    """
    synthetic_base = {
        "Ours": (153, 4),
        "E3T":  (163, 3),
        "FCP":  (84,  3),
        "IK":   (80,  3),
    }
    rng = np.random.default_rng(42)
    N_FAKE = 5   # fake participants per cell

    for layout in LAYOUT_ORDER:
        for method, (mu, sigma) in synthetic_base.items():
            real = layout_method_rewards[layout][method]
            # Only inject if we have no real non-zero observations
            if not any(v != 0 for v in real):
                fake = rng.normal(mu, sigma, N_FAKE).tolist()
                layout_method_rewards[layout][method] = fake

    return layout_method_rewards


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross Play Performance chart")
    parser.add_argument("--dir", type=str, default=".",
                        help="Directory containing debug .json files (default: current dir)")
    parser.add_argument("--out", type=str, default="crossplay_performance.png",
                        help="Output image path (default: crossplay_performance.png)")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    out_path = Path(args.out)

    print(f"\nScanning {data_dir.resolve()} …\n")
    layout_method_rewards = aggregate_all(data_dir)



    # Summary table
    print(f"\n  {'LAYOUT':<22} {'METHOD':<12} {'N':>4}  {'MEAN':>8}  {'SEM':>7}")
    print("  " + "-" * 58)
    for layout in LAYOUT_ORDER:
        for method in METHOD_MAP.values():
            vals = layout_method_rewards.get(layout, {}).get(method, [])
            m, s = mean_sem(vals)
            print(f"  {layout:<22} {method:<12} {len(vals):>4}  {m:>8.1f}  {s:>7.2f}")

    plot_crossplay(layout_method_rewards, out_path)


if __name__ == "__main__":
    main()
