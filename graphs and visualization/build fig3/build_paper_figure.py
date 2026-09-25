"""
Build the paper figure (two panels, like Fig. 3 in the paper):
  Step 1: Render the bar chart of average cross-play reward across the
          5 Overcooked layouts.
  Step 2: Combine the bar chart with the heatmap into a styled two-panel
          figure (panel labels, frames, captions) — panel (a) bar chart,
          panel (b) heatmap. No layouts strip / panel (c).

Inputs (PDF):
  HEATMAP_PDF  : pre-made heatmap of behavioural metrics by player type

Output:
  - bar chart PDF (intermediate, kept alongside the final figure)
  - combined two-panel PDF (the publication figure)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter, Transformation, PageObject

# ===========================================================================
# Paths
# ===========================================================================
HEATMAP_PDF  = "/mnt/user-data/uploads/average_all_layouts__2_.pdf"

OUTPUT_DIR   = "/mnt/user-data/outputs"
BAR_PDF      = os.path.join(OUTPUT_DIR, "overcooked_xp_reward.pdf")
OUTPUT       = os.path.join(OUTPUT_DIR, "combined_two_panel.pdf")

DECOR_PDF    = "/tmp/_decor.pdf"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# STEP 1 — Build the bar chart
# ===========================================================================

# Raw per-layout (mean, SEM across 20 training seeds) — from the updated
# results table (Table: "Cross-play (XP) reward per Overcooked layout")
data = {
    "cramped_room": {
        "my":  (195.1,  7.1),
        "e3t": (159.9,  7.2),
        "fcp": (181.1, 12.3),
        "mep": (155.2, 11.7),
        "hsp": (190.7,  5.5),
        "cec": (149.8,  6.2),
    },
    "coord_ring": {
        "my":  (197.9,  9.9),
        "e3t": (97.0,  10.7),
        "fcp": (141.8, 22.1),
        "mep": (142.3,  9.0),
        "hsp": (191.1,  6.1),
        "cec": (135.4, 15.5),
    },
    "assym": {
        "my":  (283.2, 24.1),
        "e3t": (123.2,  9.7),
        "fcp": (123.9, 30.0),
        "mep": (215.7, 30.8),
        "hsp": (219.1, 15.7),
        "cec": (137.3, 20.0),
    },
    "counter_circuit": {
        "my":  (126.1,  7.8),
        "e3t": (51.8,   9.1),
        "fcp": (69.9,  18.54),
        "mep": (95.4,  15.8),
        "hsp": (118.0,  9.3),
        "cec": (26.6,   5.3),
    },
    "forced_coord": {
        "my":  (55.1,   6.9),
        "e3t": (27.2,   3.1),
        "fcp": (36.6,   7.0),
        "mep": (47.0,   9.1),
        "hsp": (42.7,   6.2),
        "cec": (27.8,   2.9),
    },
}

# Display names
RENAME = {
    "my":  "TEAM (ours)",
    "e3t": "E3T",
    "fcp": "FCP",
    "mep": "MEP",
    "hsp": "HSP",
    "cec": "CEC",
}
DISPLAY_ORDER = ["TEAM (ours)", "E3T", "FCP", "MEP", "HSP", "CEC"]

# Aggregate (average) across the 5 layouts.
# Table row values are already mean +/- SEM across the 20 training seeds,
# so we average the per-layout means and average the per-layout SEMs
# (matching the table's reported "Average" row, e.g. TEAM = 171.48).
layouts = list(data.keys())
agg_mean, agg_sem = {}, {}
for raw_name, display_name in RENAME.items():
    per_layout_means = np.array([data[L][raw_name][0] for L in layouts])
    per_layout_sems  = np.array([data[L][raw_name][1] for L in layouts])
    agg_mean[display_name] = per_layout_means.mean()
    agg_sem[display_name]  = per_layout_sems.mean()

means = [agg_mean[n] for n in DISPLAY_ORDER]
sems  = [agg_sem[n]  for n in DISPLAY_ORDER]

# Color palette
colors = {
    "TEAM (ours)": "#2E86AB",
    "E3T":         "#5D2E8C",
    "FCP":         "#E59500",
    "MEP":         "#7FB069",
    "HSP":         "#D62828",
    "CEC":         "#1B998B",
}
bar_colors = [colors[n] for n in DISPLAY_ORDER]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(DISPLAY_ORDER))
ax.bar(
    x, means, yerr=sems,
    color=bar_colors, edgecolor="black", linewidth=0.8,
    capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"},
)
ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_ORDER, fontsize=15)
ax.tick_params(axis="y", labelsize=13)
ax.set_xlabel("Algorithm", fontsize=16)
ax.set_ylabel("XP Reward", fontsize=16)
ax.set_title("Average Cross-Play Reward across 5 Overcooked Layouts",
             fontsize=16, pad=14)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

y_offset = max(means) * 0.02
for xi, m, s in zip(x, means, sems):
    ax.text(xi, m + s + y_offset, f"{m:.1f}",
            ha="center", va="bottom", fontsize=13)
ax.set_ylim(0, max(m + s for m, s in zip(means, sems)) + max(means) * 0.10)

plt.tight_layout()
with PdfPages(BAR_PDF) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)

print("Aggregated results across 5 layouts:")
print(f"{'Method':<12} {'Mean':>10} {'SEM':>10}")
for n in DISPLAY_ORDER:
    print(f"{n:<12} {agg_mean[n]:>10.3f} {agg_sem[n]:>10.3f}")
print(f"\nBar chart saved to: {BAR_PDF}\n")


# ===========================================================================
# STEP 2 — Build the styled two-panel figure (a: bar chart, b: heatmap)
# ===========================================================================

HEADERS = {
    "a": "Cross-play (XP) and cross-seed results",
    "b": "Behavioural metrics by player type",
}

# Layout parameters (PDF points; 72 pt = 1 inch)
PANEL_WIDTH      = 500   # shared width for panel A and panel B (50/50 split)
PADDING_BETWEEN  = 24    # horizontal gap between panel A and panel B
MARGIN           = 26    # outer page margin

PANEL_PAD_X      = 16    # horizontal padding inside a panel
PANEL_PAD_TOP    = 12
PANEL_PAD_BOTTOM = 14
HEADER_TO_RULE   = 6
RULE_TO_CONTENT  = 12

LABEL_FONTSIZE     = 19
SUBTITLE_FONTSIZE  = 15

BORDER_LW_PT     = 0.9
BORDER_COLOR     = "#9aa0a6"
HEADER_COLOR     = "#1a1a1a"
RULE_COLOR       = "#cfd4da"
CORNER_RADIUS    = 6.0

# --- read source pages
bar_page     = PdfReader(BAR_PDF).pages[0]
heatmap_page = PdfReader(HEATMAP_PDF).pages[0]

def wh(p):
    return float(p.mediabox.width), float(p.mediabox.height)

bar_w_src,  bar_h_src  = wh(bar_page)
heat_w_src, heat_h_src = wh(heatmap_page)

bar_aspect  = bar_w_src / bar_h_src
heat_aspect = heat_w_src / heat_h_src

# --- geometry
inner_w = PANEL_WIDTH - 2 * PANEL_PAD_X
header_h = LABEL_FONTSIZE + HEADER_TO_RULE + 1 + RULE_TO_CONTENT

# Both panels share the same width (50/50 split); each wraps its content
# to that width, and the row height is set by whichever panel is taller.
bar_width  = inner_w
bar_height = bar_width / bar_aspect

heat_width  = inner_w
heat_height = heat_width / heat_aspect

content_h  = max(bar_height, heat_height)
row_height = PANEL_PAD_TOP + header_h + content_h + PANEL_PAD_BOTTOM

panel_a_w, panel_a_h = PANEL_WIDTH, row_height
panel_b_w, panel_b_h = PANEL_WIDTH, row_height

bar_scale  = bar_width   / bar_w_src
heat_scale = heat_height / heat_h_src

page_w = MARGIN + panel_a_w + PADDING_BETWEEN + panel_b_w + MARGIN
page_h = MARGIN + row_height + MARGIN

left_x  = MARGIN
right_x = MARGIN + panel_a_w + PADDING_BETWEEN
panel_a = (left_x,  MARGIN, panel_a_w, panel_a_h)
panel_b = (right_x, MARGIN, panel_b_w, panel_b_h)

# --- decoration (frames + headers + rules)
fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, page_w); ax.set_ylim(0, page_h)
ax.set_aspect("equal"); ax.axis("off")

def draw_panel(rect, letter, subtitle):
    x, y, w, h = rect
    frame = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}",
        linewidth=BORDER_LW_PT, edgecolor=BORDER_COLOR, facecolor="white")
    ax.add_patch(frame)

    header_y = y + h - PANEL_PAD_TOP
    label_x  = x + PANEL_PAD_X
    letter_text = f"({letter})"
    ax.text(label_x, header_y, letter_text,
            ha="left", va="top",
            fontsize=LABEL_FONTSIZE, fontweight="bold",
            family="DejaVu Sans", color=HEADER_COLOR)
    letter_w = LABEL_FONTSIZE * 0.55 * len(letter_text)
    ax.text(label_x + letter_w + 6,
            header_y - (LABEL_FONTSIZE - SUBTITLE_FONTSIZE) * 0.15,
            subtitle, ha="left", va="top",
            fontsize=SUBTITLE_FONTSIZE, fontweight="bold",
            family="DejaVu Sans", color=HEADER_COLOR)

    rule_y = header_y - LABEL_FONTSIZE - HEADER_TO_RULE
    ax.plot([x + PANEL_PAD_X, x + w - PANEL_PAD_X],
            [rule_y, rule_y],
            color=RULE_COLOR, linewidth=0.7, solid_capstyle="butt")

draw_panel(panel_a, "a", HEADERS["a"])
draw_panel(panel_b, "b", HEADERS["b"])

fig.savefig(DECOR_PDF, format="pdf", bbox_inches=None, pad_inches=0)
plt.close(fig)

# --- compose
decor_page = PdfReader(DECOR_PDF).pages[0]
dec_w_src, dec_h_src = wh(decor_page)
new_page = PageObject.create_blank_page(width=page_w, height=page_h)
new_page.merge_transformed_page(
    decor_page,
    Transformation().scale(page_w / dec_w_src, page_h / dec_h_src).translate(0, 0))

def content_origin(rect):
    x, y, w, h = rect
    rule_y = (y + h - PANEL_PAD_TOP) - LABEL_FONTSIZE - HEADER_TO_RULE
    return x + PANEL_PAD_X, rule_y - RULE_TO_CONTENT

# Panel A: bar chart (vertically centered in the content area)
ax_x, ax_top = content_origin(panel_a)
a_slack = content_h - bar_height
new_page.merge_transformed_page(
    bar_page,
    Transformation().scale(bar_scale).translate(ax_x, ax_top - a_slack / 2 - bar_height))

# Panel B: heatmap (vertically centered in the content area)
bx_x, bx_top = content_origin(panel_b)
b_slack = content_h - heat_height
new_page.merge_transformed_page(
    heatmap_page,
    Transformation().scale(heat_scale).translate(bx_x, bx_top - b_slack / 2 - heat_height))

# --- write
writer = PdfWriter()
writer.add_page(new_page)
with open(OUTPUT, "wb") as f:
    writer.write(f)

print(f"Combined PDF written to: {OUTPUT}")
print(f"Page size: {page_w:.1f} x {page_h:.1f} pt "
      f"({page_w/72:.2f} x {page_h/72:.2f} in)")
