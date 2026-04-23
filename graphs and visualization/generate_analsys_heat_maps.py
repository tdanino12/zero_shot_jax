"""
Overcooked Layout Heatmaps – v4
================================
Output heatmaps (all PDFs, column-wise normalisation):

  1.  5 per-layout heatmaps
        rows: Socializers / Achiever_rewards / Explorers / Self-play agent
  2.  Average heatmap – all 5 layouts
  3.  Average heatmap – coord_ring, cramped_room, counter_circuit
  4.  Average heatmap – forced_coord, asymm_advantages
  5.  Achiever comparison (averaged all 5 layouts)
        rows: Achievers LR / Achiever_rewards / Standard agent
        cols: behavioural metrics + reward + total_reward
  6.  Achiever vs Standard agent – standard layouts only
        rows: Achievers LR / Achiever_rewards / Standard agent
        cols: behavioural metrics + reward + total_reward
  7.  Achiever vs Standard agent – special layouts only
        rows: Achievers LR / Achiever_rewards / Standard agent
        cols: behavioural metrics + reward + total_reward
  8.  3 coord_ring within-category heatmaps (High/Mid/Low + Self-play agent)

Folder assignments
──────────────────
Standard layouts (coord_ring, cramped_room, counter_circuit):
  Socializers      phase1_lr          0,1,2,3,4,5,6,9
  Achiever_rewards phase1_r_achiever  7,8,10,11
  Achievers LR     phase1_lr          7,8,10,11
  Explorers        phase1_lr          12,13,14,15,16,17,18
  Self-play agent  phase1_lr          50,51,52
  Standard agent   standrad_regular   (all folders)

Special layouts (forced_coord, asymm_advantages):
  Socializers      phase1_lr          0,1
  Achiever_rewards phase1_r_achiever  8,7,10,2
  Achievers LR     phase1_lr          8,7,10,2
  Explorers        phase1_lr          18,12,13
  Self-play agent  phase1_lr          50,51,52
  Standard agent   standrad_regular   (all folders)

Coord Ring achiever levels: High=[8,10]  Mid=[7]  Low=[11]
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("new_csv/csv_files")
OUTPUT_DIR = Path("heatmaps_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Layouts ───────────────────────────────────────────────────────────────────
LAYOUTS = {
    "coord_ring":       "Coord Ring",
    "cramped_room":     "Cramped Room",
    "forced_coord":     "Force Coord",
    "asymm_advantages": "Asymm Advantages",
    "counter_circuit":  "Counter Circuit",
}
SPECIAL_LAYOUTS  = {"forced_coord", "asymm_advantages"}
STANDARD_LAYOUTS = ["coord_ring", "cramped_room", "counter_circuit"]
SPECIAL_LAYOUTS_LIST = ["forced_coord", "asymm_advantages"]

# ── Folder sets ───────────────────────────────────────────────────────────────
STD = {
    "socializers":      {"phase": "phase1_lr",         "folders": {0,1,2,3,4,5,6,9}},
    "achiever_rewards": {"phase": "phase1_r_achiever", "folders": {7,8,10,11}},
    "achievers_lr":     {"phase": "phase1_lr",         "folders": {7,8,10,11}},
    "explorers":        {"phase": "phase1_lr",         "folders": {12,13,14,15,16,17,18}},
    "self_play":        {"phase": "phase1_lr",         "folders": {50,51,52}},
}
SPECIAL = {
    "socializers":      {"phase": "phase1_lr",         "folders": {0,1}},
    "achiever_rewards": {"phase": "phase1_r_achiever", "folders": {8,7,10,2}},
    "achievers_lr":     {"phase": "phase1_lr",         "folders": {8,7,10,2}},
    "explorers":        {"phase": "phase1_lr",         "folders": {18,12,13}},
    "self_play":        {"phase": "phase1_lr",         "folders": {50,51,52}},
}

def cfg_for(layout_key: str) -> dict:
    return SPECIAL if layout_key in SPECIAL_LAYOUTS else STD

# ── Display labels ────────────────────────────────────────────────────────────
CATEGORY_LABELS = {
    "socializers":      "Socializers",
    "achiever_rewards": "Achievers",
    "achievers_lr":     "Achievers LR",
    "explorers":        "Explorers",
    "self_play":        "Self-play agent",
    "standard_agent":   "Standard agent",
}

MAIN_CATS = ["socializers", "achiever_rewards", "explorers", "self_play"]

# ── Metrics ───────────────────────────────────────────────────────────────────
COUNTER_COLS = {"plate placed on counter",
                "onion placed on counter",
                "dish placed on counter"}
COMBINED_COL = "items placed on counter"

METRIC_ORDER = [
    "plate pickup",
    "onion picked",
    "dish picked",
    "delivery",
    COMBINED_COL,
    "onion placed in pot",
]
METRIC_ORDER_WITH_REWARD = METRIC_ORDER + ["reward", "total_reward"]

# ── Data helpers ──────────────────────────────────────────────────────────────

def combine_counter_cols(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in COUNTER_COLS if c in df.columns]
    if present:
        df = df.copy()
        df[COMBINED_COL] = df[present].mean(axis=1)
        df = df.drop(columns=present)
    return df

def load_phase(layout_dir: Path, phase: str, folders: set | None) -> pd.DataFrame:
    """Load all CSVs from a phase dir, optionally filtering by folder numbers."""
    frames = []
    phase_dir = layout_dir / phase
    if not phase_dir.exists():
        return pd.DataFrame()
    for fd in phase_dir.iterdir():
        if not fd.is_dir():
            continue
        if folders is not None:
            try:
                n = int(fd.name)
            except ValueError:
                continue
            if n not in folders:
                continue
        csv_path = fd / "output.csv"
        if csv_path.exists():
            df = combine_counter_cols(pd.read_csv(csv_path))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

ALWAYS_EXCLUDE = {"_folder"}

def mean_row_from_df(df: pd.DataFrame, keep_reward: bool = False) -> dict:
    if df.empty:
        return {}
    exclude = set(ALWAYS_EXCLUDE)
    if not keep_reward:
        exclude |= {"reward", "total_reward"}
    cols = [c for c in df.columns if c not in exclude]
    return df[cols].mean().to_dict()

def mean_row(layout_dir: Path, phase: str, folders: set,
             keep_reward: bool = False) -> dict:
    df = load_phase(layout_dir, phase, folders)
    return mean_row_from_df(df, keep_reward=keep_reward)

def mean_row_standard_agent(layout_dir: Path,
                             keep_reward: bool = False) -> dict:
    """Average all folders inside standrad_regular for a layout."""
    df = load_phase(layout_dir, "standrad_regular", folders=None)
    return mean_row_from_df(df, keep_reward=keep_reward)

def build_summary(layout_key: str, categories: list,
                  keep_reward: bool = False) -> pd.DataFrame:
    cfg  = cfg_for(layout_key)
    ldir = BASE_DIR / layout_key
    rows = {}
    for cat in categories:
        if cat == "standard_agent":
            rows[CATEGORY_LABELS[cat]] = mean_row_standard_agent(
                ldir, keep_reward=keep_reward)
        else:
            c = cfg[cat]
            rows[CATEGORY_LABELS[cat]] = mean_row(
                ldir, c["phase"], c["folders"], keep_reward=keep_reward)
    df   = pd.DataFrame(rows).T
    order = METRIC_ORDER_WITH_REWARD if keep_reward else METRIC_ORDER
    cols = [m for m in order if m in df.columns]
    return df[cols]

# ── Column-wise normalisation ─────────────────────────────────────────────────

def col_normalize(data: np.ndarray) -> np.ndarray:
    data_n = data.copy().astype(float)
    for c in range(data.shape[1]):
        col = data[:, c]
        mn, mx = col.min(), col.max()
        data_n[:, c] = (col - mn) / (mx - mn) if mx > mn else 0.0
    return data_n

# ── Plotting ──────────────────────────────────────────────────────────────────

TICK_FS  = 12
TITLE_FS = 13
ANNOT_FS = 9

def _wrap(name: str) -> str:
    return (name
            .replace("items placed on counter", "items\nplaced on\ncounter")
            .replace("onion placed in pot",     "onion\nplaced\nin pot")
            .replace("plate pickup",            "plate\npickup")
            .replace("onion picked",            "onion\npicked")
            .replace("dish picked",             "dish\npicked")
            .replace("total_reward",            "total\nreward"))

def plot_heatmap(summary: pd.DataFrame, title: str, out_path: Path,
                 cmap: str = "YlOrRd"):
    metrics = list(summary.columns)
    ptypes  = list(summary.index)
    data    = summary.values.astype(float)
    n_m, n_p = len(metrics), len(ptypes)

    data_n = col_normalize(data)

    fig_w = max(10, n_m * 1.6)
    fig_h = max(3.5, n_p * 1.1 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data_n, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for r in range(n_p):
        for c in range(n_m):
            v  = data[r, c]
            tc = "white" if data_n[r, c] > 0.6 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=ANNOT_FS, color=tc, fontweight="bold")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([_wrap(m) for m in metrics], fontsize=TICK_FS, ha="center")
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(ptypes, fontsize=TICK_FS, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Column-normalised (0 = min, 1 = max per metric)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["min", "mid", "max"])

    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(-0.5, n_m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_p, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")

# ── helpers for averaged summaries ───────────────────────────────────────────

def average_summaries(layout_keys: list, categories: list,
                      keep_reward: bool = False) -> pd.DataFrame:
    all_s = [build_summary(lk, categories, keep_reward=keep_reward)
             for lk in layout_keys]
    avg = pd.concat(all_s).groupby(level=0).mean()
    order = [CATEGORY_LABELS[c] for c in categories]
    avg = avg.loc[order]
    col_order = METRIC_ORDER_WITH_REWARD if keep_reward else METRIC_ORDER
    cols = [m for m in col_order if m in avg.columns]
    return avg[cols]

# ── 1. Per-layout heatmaps ────────────────────────────────────────────────────

def make_layout_heatmaps():
    for lkey, lname in LAYOUTS.items():
        summary = build_summary(lkey, MAIN_CATS)
        title   = f"Overcooked – {lname}\nMean Behavioural Metrics by Player Type"
        plot_heatmap(summary, title, OUTPUT_DIR / f"layout_{lkey}.pdf")

# ── 2. Average – all 5 layouts ────────────────────────────────────────────────

def make_average_heatmap():
    avg   = average_summaries(list(LAYOUTS.keys()), MAIN_CATS)
    title = "Overcooked – Average Across All 5 Layouts\nMean Behavioural Metrics by Player Type"
    plot_heatmap(avg, title, OUTPUT_DIR / "average_all_layouts.pdf")

# ── 3. Average – standard layouts ────────────────────────────────────────────

def make_average_heatmap_standard():
    avg   = average_summaries(STANDARD_LAYOUTS, MAIN_CATS)
    title = ("Overcooked – Average Across Coord Ring, Cramped Room & Counter Circuit\n"
             "Mean Behavioural Metrics by Player Type")
    plot_heatmap(avg, title, OUTPUT_DIR / "average_standard_layouts.pdf")

# ── 4. Average – special layouts ─────────────────────────────────────────────

def make_average_heatmap_special():
    avg   = average_summaries(SPECIAL_LAYOUTS_LIST, MAIN_CATS)
    title = ("Overcooked – Average Across Force Coord & Asymm Advantages\n"
             "Mean Behavioural Metrics by Player Type")
    plot_heatmap(avg, title, OUTPUT_DIR / "average_special_layouts.pdf")

# ── 5. Achiever comparison – all layouts (with Standard agent + reward cols) ──

def make_achiever_comparison_all():
    cats  = ["achievers_lr", "achiever_rewards", "standard_agent"]
    avg   = average_summaries(list(LAYOUTS.keys()), cats, keep_reward=True)
    avg.index = ["Achiever(LR)" if i == "Achievers LR"
                 else "Achievers(reward)" if i == "Achievers"
                 else i for i in avg.index]
    title = ("Overcooked – Achiever Type Comparison\n"
             "Averaged Across All 5 Layouts")
    plot_heatmap(avg, title, OUTPUT_DIR / "achiever_comparison.pdf", cmap="PuBuGn")

# ── 6. Achiever vs Standard – standard layouts (with reward_level rows) ──────
# Achiever_rewards split into Low/Mid/High using phase1_r_achiever folders.
# Achievers LR and Standard agent averaged across all standard layouts as before.

def make_achiever_comparison_standard():
    # layout index mapping
    layout_map = {0: "coord_ring", 1: "cramped_room", 2: "counter_circuit"}

    # --- Achiever_rewards levels: average each level across the 3 standard layouts
    level_folders_std = {
        "Achievers(reward) Low":  {11},
        "Achievers(reward) Mid":  {7, 8},
        "Achievers(reward) High": {10},
    }
    level_rows = {}
    for label, folders in level_folders_std.items():
        frames = []
        for lk in STANDARD_LAYOUTS:
            ldir = BASE_DIR / lk
            r = mean_row(ldir, "phase1_r_achiever", folders, keep_reward=True)
            if r:
                frames.append(r)
        if frames:
            level_rows[label] = pd.DataFrame(frames).mean().to_dict()

    # --- Achievers LR: average across standard layouts
    lr_frames = []
    for lk in STANDARD_LAYOUTS:
        ldir = BASE_DIR / lk
        r = mean_row(ldir, STD["achievers_lr"]["phase"],
                     STD["achievers_lr"]["folders"], keep_reward=True)
        if r:
            lr_frames.append(r)
    level_rows["Achiever(LR)"] = pd.DataFrame(lr_frames).mean().to_dict() if lr_frames else {}

    # --- Standard agent: average across standard layouts
    sa_frames = []
    for lk in STANDARD_LAYOUTS:
        ldir = BASE_DIR / lk
        r = mean_row_standard_agent(ldir, keep_reward=True)
        if r:
            sa_frames.append(r)
    level_rows["Standard agent"] = pd.DataFrame(sa_frames).mean().to_dict() if sa_frames else {}

    row_order = ["Achievers(reward) Low", "Achievers(reward) Mid", "Achievers(reward) High",
                 "Achiever(LR)", "Standard agent"]
    df   = pd.DataFrame({k: level_rows[k] for k in row_order if k in level_rows}).T
    cols = [m for m in METRIC_ORDER_WITH_REWARD if m in df.columns]
    df   = df[cols]

    title = ("Overcooked – Achiever Levels vs Standard Agent\n"
             "Coord Ring, Cramped Room & Counter Circuit")
    plot_heatmap(df, title,
                 OUTPUT_DIR / "achiever_comparison_standard_layouts.pdf",
                 cmap="PuBuGn")

# ── 7. Achiever vs Standard – special layouts (with reward_level rows) ────────

def make_achiever_comparison_special():
    level_folders_spc = {
        "Achievers(reward) Low":  {2},
        "Achievers(reward) Mid":  {7, 8},
        "Achievers(reward) High": {10},
    }
    level_rows = {}
    for label, folders in level_folders_spc.items():
        frames = []
        for lk in SPECIAL_LAYOUTS_LIST:
            ldir = BASE_DIR / lk
            r = mean_row(ldir, "phase1_r_achiever", folders, keep_reward=True)
            if r:
                frames.append(r)
        if frames:
            level_rows[label] = pd.DataFrame(frames).mean().to_dict()

    # --- Achievers LR: average across special layouts
    lr_frames = []
    for lk in SPECIAL_LAYOUTS_LIST:
        ldir = BASE_DIR / lk
        r = mean_row(ldir, SPECIAL["achievers_lr"]["phase"],
                     SPECIAL["achievers_lr"]["folders"], keep_reward=True)
        if r:
            lr_frames.append(r)
    level_rows["Achiever(LR)"] = pd.DataFrame(lr_frames).mean().to_dict() if lr_frames else {}

    # --- Standard agent: average across special layouts
    sa_frames = []
    for lk in SPECIAL_LAYOUTS_LIST:
        ldir = BASE_DIR / lk
        r = mean_row_standard_agent(ldir, keep_reward=True)
        if r:
            sa_frames.append(r)
    level_rows["Standard agent"] = pd.DataFrame(sa_frames).mean().to_dict() if sa_frames else {}

    row_order = ["Achievers(reward) Low", "Achievers(reward) Mid", "Achievers(reward) High",
                 "Achiever(LR)", "Standard agent"]
    df   = pd.DataFrame({k: level_rows[k] for k in row_order if k in level_rows}).T
    cols = [m for m in METRIC_ORDER_WITH_REWARD if m in df.columns]
    df   = df[cols]

    title = ("Overcooked – Achiever Levels vs Standard Agent\n"
             "Force Coord & Asymm Advantages")
    plot_heatmap(df, title,
                 OUTPUT_DIR / "achiever_comparison_special_layouts.pdf",
                 cmap="PuBuGn")

# ── 8. Coord Ring within-category heatmaps ───────────────────────────────────

COORD_RING_LEVELS = {
    "Socializers": {
        "High": [3],
        "Mid":  [1, 2],
        "Low":  [0],
    },
    "Achievers": {
        "High": [8, 10],
        "Mid":  [7],
        "Low":  [11],
    },
    "Explorers": {
        "High": [16, 17],
        "Mid":  [13, 14, 15],
        "Low":  [18, 12],
    },
}

def make_coord_ring_within():
    ldir = BASE_DIR / "coord_ring"
    for cat_name, levels in COORD_RING_LEVELS.items():
        rows = {}
        for level_label, folders in levels.items():
            rows[level_label] = mean_row(ldir, "phase1_lr", set(folders))
        rows["Self-play agent"] = mean_row(ldir, "phase1_lr", {50, 51, 52})
        df   = pd.DataFrame(rows).T
        cols = [m for m in METRIC_ORDER if m in df.columns]
        df   = df[cols]
        title = f"Coord Ring – {cat_name}\nWithin-Category Levels"
        out   = OUTPUT_DIR / f"coord_ring_within_{cat_name.lower()}.pdf"
        plot_heatmap(df, title, out, cmap="Blues")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Per-layout heatmaps ===")
    make_layout_heatmaps()

    print("\n=== Average heatmap (all 5 layouts) ===")
    make_average_heatmap()

    print("\n=== Average heatmap (standard layouts) ===")
    make_average_heatmap_standard()

    print("\n=== Average heatmap (special layouts) ===")
    make_average_heatmap_special()

    print("\n=== Achiever comparison – all layouts ===")
    make_achiever_comparison_all()

    print("\n=== Achiever vs Standard – standard layouts ===")
    make_achiever_comparison_standard()

    print("\n=== Achiever vs Standard – special layouts ===")
    make_achiever_comparison_special()

    print("\n=== Coord Ring within-category heatmaps ===")
    make_coord_ring_within()

    print(f"\nDone – all PDFs saved to ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
