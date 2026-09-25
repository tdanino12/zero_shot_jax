"""
Per-layout cross-play (XP) reward bar chart — 20-seed results.

Standalone script — produces a single grouped bar chart showing each method's
XP reward on every Overcooked layout (5 layout groups x 6 method bars each).

Output:
  overcooked_xp_per_layout_20seeds.pdf

Dependencies:
  pip install numpy matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ===========================================================================
# Output path
# ===========================================================================
OUTPUT_PATH = "overcooked_xp_per_layout_20seeds.pdf"

# ===========================================================================
# Data: per-layout (mean, SEM directly, across 20 seeds)
# taken from the LaTeX results table
# ===========================================================================
data = {
    "cramped_room": {
        "my":  (195.1, 7.1),
        "e3t": (159.9, 7.2),
        "fcp": (181.1, 12.3),
        "mep": (155.2, 11.7),
        "hsp": (190.7, 5.5),
        "cec": (149.8, 6.2),
    },
    "coord_ring": {
        "my":  (197.9, 9.9),
        "e3t": (97.0, 10.7),
        "fcp": (141.8, 22.1),
        "mep": (142.3, 9.0),
        "hsp": (191.1, 6.1),
        "cec": (135.4, 15.5),
    },
    "assym": {
        "my":  (283.2, 24.1),
        "e3t": (123.2, 9.7),
        "fcp": (123.9, 30.0),
        "mep": (215.7, 30.8),
        "hsp": (219.1, 15.7),
        "cec": (137.3, 20.0),
    },
    "counter_circuit": {
        "my":  (126.1, 7.8),
        "e3t": (51.8, 9.1),
        "fcp": (69.9, 18.54),
        "mep": (95.4, 15.8),
        "hsp": (118.0, 9.3),
        "cec": (26.6, 5.3),
    },
    "forced_coord": {
        "my":  (55.1, 6.9),
        "e3t": (27.2, 3.1),
        "fcp": (36.6, 7.0),
        "mep": (47.0, 9.1),
        "hsp": (42.7, 6.2),
        "cec": (27.8, 2.9),
    },
}

# Display names for methods
RENAME = {
    "my":  "TEAM (ours)",
    "e3t": "E3T",
    "fcp": "FCP",
    "mep": "MEP",
    "hsp": "HSP",
    "cec": "CEC",
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
# Build the means/SEM grid (SEM values are already given directly)
# ===========================================================================
n_layouts = len(LAYOUT_ORDER)
n_methods = len(DISPLAY_ORDER)

means_grid = np.zeros((n_layouts, n_methods))
sems_grid  = np.zeros((n_layouts, n_methods))

inv_rename = {v: k for k, v in RENAME.items()}
for i, layout_key in enumerate(LAYOUT_ORDER):
    for j, display_name in enumerate(DISPLAY_ORDER):
        raw_name = inv_rename[display_name]
        m, sem = data[layout_key][raw_name]
        means_grid[i, j] = m
        sems_grid[i, j]  = sem

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
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel("Layout", fontsize=20)
ax.set_ylabel("Reward", fontsize=20)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Headroom for the legend at the top
ymax = (means_grid + sems_grid).max()
ax.set_ylim(0, ymax * 1.25)

# Legend in a single row
ax.legend(
    ncol=n_methods, fontsize=16, frameon=False,
    loc="upper center", bbox_to_anchor=(0.5, 1.0),
    handlelength=1.6, handleheight=1.2,
    columnspacing=1.3, handletextpad=0.5,
)

plt.tight_layout()

# ===========================================================================
# Save and report
# ===========================================================================
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with PdfPages(OUTPUT_PATH) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)

print("Per-layout results (mean ± SEM, 20 seeds):")
header = f"{'Layout':<18}" + "".join(f"{m:>14}" for m in DISPLAY_ORDER)
print(header)
print("-" * len(header))
for i, layout_key in enumerate(LAYOUT_ORDER):
    row = f"{LAYOUT_LABELS[layout_key]:<18}"
    for j in range(n_methods):
        row += f"{means_grid[i,j]:>7.1f}±{sems_grid[i,j]:>4.1f} "
    print(row)

avg = means_grid.mean(axis=0)
print("\nAverage across layouts:")
for j, method in enumerate(DISPLAY_ORDER):
    print(f"  {method:<14}{avg[j]:.2f}")

print(f"\nFigure saved to: {OUTPUT_PATH}")
