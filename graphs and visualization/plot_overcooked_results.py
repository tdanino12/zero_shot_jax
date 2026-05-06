"""
Reproduces the style of Figure 6 (left) from the Cross-environment Cooperation paper:
Average XP Reward across all 5 Overcooked layouts, with standard error bars.

- Skips the "ik" method (per user request).
- Renames methods: my -> TEAM, e3t -> E3T, fcp -> FCP, mep -> MEP,
                   ik_finetune -> CEC, hsp -> HSP.
- Averages the per-layout means and computes SEM across the 5 layouts
  (SEM = std(layout_means) / sqrt(n_layouts)).
- Saves output as a PDF.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Raw per-layout results (mean, std across seeds), as provided by the user.
# ---------------------------------------------------------------------------
# Each layout dict: method_name -> (mean, std)
data = {
    "forced_coord": {
        "my":          (54.05786,  26.348522),
        "e3t":         (26.221786, 10.168269),
        "fcp":         (37.48107,  24.772738),
        "ik":          (7.0871434,  2.784751),
        "mep":         (47.52786,  40.212982),
        "ik_finetune": (26.924646,  8.547634),
        "hsp":         (41.618217, 22.847237),
    },
    "cramped_room": {
        "my":          (192.95691,  34.50269),
        "e3t":         (155.81189,  28.962366),
        "fcp":         (180.54393,  48.186893),
        "ik":          (153.1656,   14.494744),
        "mep":         (153.1218,   46.607937),
        "ik_finetune": (144.80215,  22.782776),
        "hsp":         (191.73752,  23.766764),
    },
    "coord_ring": {
        "my":          (197.16145,  40.60414),
        "e3t":         (94.97321,   37.318813),
        "fcp":         (140.83572,  85.48993),
        "ik":          (115.24965,  42.42527),
        "mep":         (143.44286,  34.729904),
        "ik_finetune": (134.49358,  61.220802),
        "hsp":         (192.46216,  27.514336),
    },
    "assym": {
        "my":          (277.2443,    92.02341),
        "e3t":         (122.06858,   37.042847),
        "fcp":         (113.8418,   107.75341),
        "ik":          (73.32643,    50.010845),
        "mep":         (213.65286,  113.60678),
        "ik_finetune": (131.17679,   72.16775),
        "hsp":         (217.7043,    59.227245),
    },
    "counter_circuit": {
        "my":          (125.89001,  32.839577),
        "e3t":         (49.797504,  33.243916),
        "fcp":         (66.91393,   67.8655),
        "ik":          (21.076786,  24.141754),
        "mep":         (91.35643,   62.64699),
        "ik_finetune": (22.616787,  21.221844),
        "hsp":         (118.42857,  43.236946),
    },
}

# ---------------------------------------------------------------------------
# Method renaming and ordering (skip "ik" entirely).
# ---------------------------------------------------------------------------
RENAME = {
    "my":          "TEAM",
    "e3t":         "E3T",
    "fcp":         "FCP",
    "mep":         "MEP",
    "hsp":         "HSP",
    "ik_finetune": "CEC",
    # "ik" is intentionally excluded
}

# Display order in the bar chart
DISPLAY_ORDER = ["TEAM", "E3T", "FCP", "MEP", "HSP", "CEC"]

# ---------------------------------------------------------------------------
# Aggregate across the 5 layouts.
# ---------------------------------------------------------------------------
layouts = list(data.keys())
n_layouts = len(layouts)

agg_mean = {}
agg_sem  = {}

for raw_name, display_name in RENAME.items():
    per_layout_means = np.array([data[layout][raw_name][0] for layout in layouts])
    per_layout_stds  = np.array([data[layout][raw_name][1] for layout in layouts])
    mean_of_means = per_layout_means.mean()
    # Convert each layout's std to standard error (std / sqrt(10),
    # since each method has 10 seeds), then average across the 5 layouts.
    per_layout_sems = per_layout_stds / np.sqrt(10.0)
    sem = per_layout_sems.mean()
    agg_mean[display_name] = mean_of_means
    agg_sem[display_name]  = sem

means = [agg_mean[name] for name in DISPLAY_ORDER]
sems  = [agg_sem[name]  for name in DISPLAY_ORDER]

# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------
# Color palette inspired by Figure 6 in the paper
colors = {
    "TEAM": "#2E86AB",   # blue
    "E3T":  "#5D2E8C",   # purple
    "FCP":  "#E59500",   # orange/gold
    "MEP":  "#7FB069",   # green
    "HSP":  "#D62828",   # red
    "CEC":  "#1B998B",   # teal
}
bar_colors = [colors[name] for name in DISPLAY_ORDER]

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(DISPLAY_ORDER))
bars = ax.bar(
    x, means,
    yerr=sems,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.8,
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": "black"},
)

ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_ORDER, fontsize=15)
ax.tick_params(axis="y", labelsize=13)
ax.set_xlabel("Algorithm", fontsize=16)
ax.set_ylabel("XP Reward", fontsize=16)
ax.set_title("Average Cross-Play Reward across 5 Overcooked Layouts",
             fontsize=16, pad=14)

# Light grid behind bars (matching the paper's clean look)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate each bar with the value, placed above the top of its error bar
y_offset = max(means) * 0.02  # small gap above the error bar
for xi, m, s in zip(x, means, sems):
    ax.text(xi, m + s + y_offset, f"{m:.1f}",
            ha="center", va="bottom", fontsize=13)

ax.set_ylim(0, max(m + s for m, s in zip(means, sems)) + max(means) * 0.10)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save to PDF.
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "overcooked_xp_reward.pdf")
with PdfPages(output_path) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)

# Print the aggregated numbers for reference
print("Aggregated results across 5 layouts:")
print(f"{'Method':<8} {'Mean':>10} {'SEM':>10}")
for name in DISPLAY_ORDER:
    print(f"{name:<8} {agg_mean[name]:>10.3f} {agg_sem[name]:>10.3f}")
print(f"\nSaved figure to: {output_path}")
