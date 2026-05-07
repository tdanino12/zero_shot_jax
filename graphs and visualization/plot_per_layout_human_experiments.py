"""
Per-layout cross-play (XP) reward bar chart.

Standalone script — produces a single grouped bar chart showing each method's
XP reward on every Overcooked layout (5 layout groups × 6 method bars each).

Output:
  overcooked_xp_per_layout.pdf

Dependencies:
  pip install numpy matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ===========================================================================
# Output path — edit if you want it written elsewhere
# ===========================================================================
OUTPUT_PATH = "overcooked_xp_per_layout.pdf"

# ===========================================================================
# Data: per-layout (mean, std across 10 seeds)
# ===========================================================================
data = {
    "forced_coord": {
        "my":          (174.05786,  33.348522),
        "e3t":         (65.221786, 10.168269),
        "fcp":         (131.48107,  34.772738),
        "ik":          (82.0871434,  25.784751),
        "mep":         (155.52786,  30.212982),
        "ik_finetune": (96.924646,  18.547634),
        "hsp":         (156.618217, 29.847237),
    },
    "cramped_room": {
        "my":          (222.95691,  37.50269),
        "e3t":         (198.81189,  28.962366),
        "fcp":         (196.54393,  41.186893),
        "ik":          (153.1656,   14.494744),
        "mep":         (198.1218,   36.607937),
        "ik_finetune": (208.80215,  29.782776),
        "hsp":         (205.73752,  28.766764),
    },
    "coord_ring": {
        "my":          (265.16145,  40.60414),
        "e3t":         (140.97321,   35.318813),
        "fcp":         (190.83572,  65.48993),
        "ik":          (180.24965,  42.42527),
        "mep":         (143.44286,  34.729904),
        "ik_finetune": (185.49358,  51.220802),
        "hsp":         (261.46216,  37.514336),
    },
    "assym": {
        "my":          (287.2443,    62.02341),
        "e3t":         (222.06858,   37.042847),
        "fcp":         (250.8418,   85.75341),
        "ik":          (73.32643,    50.010845),
        "mep":         (219.65286,  113.60678),
        "ik_finetune": (225.17679,   52.16775),
        "hsp":         (220.7043,    49.227245),
    },
    "counter_circuit": {
        "my":          (195.89001,  28.839577),
        "e3t":         (95.797504,  33.243916),
        "fcp":         (110.91393,   47.8655),
        "ik":          (21.076786,  24.141754),
        "mep":         (102.35643,   62.64699),
        "ik_finetune": (99.616787,  34.221844),
        "hsp":         (190.42857,  41.236946),
    },
}

# Display names for methods (skip "ik")
RENAME = {
    "my":          "TEAM (ours)",
    "e3t":         "E3T",
    "fcp":         "FCP",
    "mep":         "MEP",
    "hsp":         "HSP",
    "ik_finetune": "CEC",
}
DISPLAY_ORDER = ["TEAM (ours)", "E3T", "FCP", "MEP", "HSP", "CEC"]

# Display names + plotting order for layouts
LAYOUT_LABELS = {
    "cramped_room":    "Cramped Room",
    "coord_ring":      "Coord. Ring",
    "assym":           "Asymm. Adv.",
    "counter_circuit": "Counter Circuit",
    "forced_coord":    "Forced Coord.",
}
LAYOUT_ORDER = ["cramped_room", "coord_ring", "assym",
                "counter_circuit", "forced_coord"]

# Colour palette
colors = {
    "TEAM (ours)": "#2E86AB",
    "E3T":         "#5D2E8C",
    "FCP":         "#E59500",
    "MEP":         "#7FB069",
    "HSP":         "#D62828",
    "CEC":         "#1B998B",
}

# ===========================================================================
# Build the means/SEM grid
# ===========================================================================
n_seeds   = 10
n_layouts = len(LAYOUT_ORDER)
n_methods = len(DISPLAY_ORDER)

means_grid = np.zeros((n_layouts, n_methods))
sems_grid  = np.zeros((n_layouts, n_methods))

inv_rename = {v: k for k, v in RENAME.items()}
for i, layout_key in enumerate(LAYOUT_ORDER):
    for j, display_name in enumerate(DISPLAY_ORDER):
        raw_name = inv_rename[display_name]
        m, s = data[layout_key][raw_name]
        means_grid[i, j] = m
        # SEM = std / sqrt(n_seeds)
        sems_grid[i, j]  = s / np.sqrt(n_seeds)

# ===========================================================================
# Plot
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 5.2))

bar_width   = 0.13
group_pitch = n_methods * bar_width + 1.5 * bar_width
group_centers = np.arange(n_layouts) * group_pitch

for j, method in enumerate(DISPLAY_ORDER):
    offsets = group_centers + (j - n_methods / 2 + 0.5) * bar_width
    ax.bar(
        offsets, means_grid[:, j],
        width=bar_width,
        yerr=sems_grid[:, j],
        color=colors[method],
        edgecolor="black", linewidth=0.6,
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "ecolor": "black"},
        label=method,
        zorder=3,
    )

ax.set_xticks(group_centers)
ax.set_xticklabels([LAYOUT_LABELS[k] for k in LAYOUT_ORDER], fontsize=13)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel("Layout", fontsize=15)
ax.set_ylabel("XP Reward", fontsize=15)
ax.set_title("Cross-Play Reward per Overcooked Layout",
             fontsize=15, pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Headroom for the legend at the top
ymax = (means_grid + sems_grid).max()
ax.set_ylim(0, ymax * 1.18)

# Legend in a single row
ax.legend(
    ncol=n_methods, fontsize=11, frameon=False,
    loc="upper center", bbox_to_anchor=(0.5, 1.0),
    handlelength=1.4, columnspacing=1.2,
)

plt.tight_layout()

# ===========================================================================
# Save and report
# ===========================================================================
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with PdfPages(OUTPUT_PATH) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)

print("Per-layout results (mean ± SEM, SEM = std/sqrt(10)):")
header = f"{'Layout':<18}" + "".join(f"{m:>14}" for m in DISPLAY_ORDER)
print(header)
print("-" * len(header))
for i, layout_key in enumerate(LAYOUT_ORDER):
    row = f"{LAYOUT_LABELS[layout_key]:<18}"
    for j in range(n_methods):
        row += f"{means_grid[i,j]:>7.1f}±{sems_grid[i,j]:>4.1f} "
    print(row)
print(f"\nFigure saved to: {OUTPUT_PATH}")
