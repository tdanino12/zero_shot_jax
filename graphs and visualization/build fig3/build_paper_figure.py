"""
Build the full paper figure in one go:
  Step 1: Render the bar chart of average cross-play reward across the
          5 Overcooked layouts.
  Step 2: Combine the bar chart with the heatmap and the layouts strip into
          a styled three-panel figure (panel labels, frames, captions).

Inputs (all PDFs):
  HEATMAP_PDF  : pre-made heatmap of behavioural metrics by player type
  LAYOUTS_PDF  : pre-made strip showing the 5 Overcooked layouts

Output:
  - bar chart PDF (intermediate, kept alongside the final figure)
  - combined three-panel PDF (the publication figure)
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
LAYOUTS_PDF  = "/mnt/user-data/uploads/Reduction__19_.pdf"

OUTPUT_DIR   = "/mnt/user-data/outputs"
BAR_PDF      = os.path.join(OUTPUT_DIR, "overcooked_xp_reward.pdf")
OUTPUT       = os.path.join(OUTPUT_DIR, "combined_three_panel.pdf")

CAPTION_PDF  = "/tmp/_caption.pdf"
DECOR_PDF    = "/tmp/_decor.pdf"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# STEP 1 — Build the average bar chart (across 5 layouts)
# ===========================================================================

# Raw per-layout (mean, std across 10 seeds)
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

# Display names (skip "ik")
RENAME = {
    "my":          "TEAM (ours)",
    "e3t":         "E3T",
    "fcp":         "FCP",
    "mep":         "MEP",
    "hsp":         "HSP",
    "ik_finetune": "CEC",
}
DISPLAY_ORDER = ["TEAM (ours)", "E3T", "FCP", "MEP", "HSP", "CEC"]

# Aggregate across the 5 layouts
layouts = list(data.keys())
agg_mean, agg_sem = {}, {}
for raw_name, display_name in RENAME.items():
    per_layout_means = np.array([data[L][raw_name][0] for L in layouts])
    per_layout_stds  = np.array([data[L][raw_name][1] for L in layouts])
    agg_mean[display_name] = per_layout_means.mean()
    # Each layout's std is across 10 seeds → SEM = std / sqrt(10);
    # average those SEMs across the 5 layouts.
    agg_sem[display_name]  = (per_layout_stds / np.sqrt(10.0)).mean()

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
# STEP 2 — Build the styled three-panel figure
# ===========================================================================

CAPTION_TEXT = "Five Overcooked layouts used for evaluation"

HEADERS = {
    "a": "Cross-play (XP) and cross-seed results",
    "b": "Behavioural metrics by player type",
    "c": "Layouts",
}

# Layout parameters (PDF points; 72 pt = 1 inch)
COL_WIDTH        = 500   # left column inner width (panel A and C share it)
PADDING_BETWEEN  = 24    # horizontal gap between left column and right panel
ROW_GAP          = 18    # vertical gap between panel A and panel C
CAPTION_GAP      = 6     # gap between layouts strip and caption
MARGIN           = 26    # outer page margin

PANEL_PAD_X      = 16    # horizontal padding inside a panel
STRIP_PAD_X      = 12    # padding around the layouts strip (left/right gap)
STRIP_PAD_TOP    = 8     # gap from rule to the strip (panel C only)
STRIP_PAD_BOTTOM = 10    # gap from strip to caption (panel C only)
PANEL_PAD_TOP    = 12
PANEL_PAD_BOTTOM = 14
HEADER_TO_RULE   = 6
RULE_TO_CONTENT  = 12

STRIP_BG_COLOR   = "#f1f3f4"   # light gray background behind the layouts strip

LABEL_FONTSIZE     = 19
SUBTITLE_FONTSIZE  = 15
CAPTION_FONTSIZE   = 14

BORDER_LW_PT     = 0.9
BORDER_COLOR     = "#9aa0a6"
HEADER_COLOR     = "#1a1a1a"
RULE_COLOR       = "#cfd4da"
CORNER_RADIUS    = 6.0

# --- read source pages
bar_page     = PdfReader(BAR_PDF).pages[0]
heatmap_page = PdfReader(HEATMAP_PDF).pages[0]
layouts_page = PdfReader(LAYOUTS_PDF).pages[0]

def wh(p):
    return float(p.mediabox.width), float(p.mediabox.height)

bar_w_src,     bar_h_src     = wh(bar_page)
heat_w_src,    heat_h_src    = wh(heatmap_page)
layouts_w_src, layouts_h_src = wh(layouts_page)

bar_aspect     = bar_w_src / bar_h_src
layouts_aspect = layouts_w_src / layouts_h_src
heat_aspect    = heat_w_src / heat_h_src

# --- render caption
fig, ax = plt.subplots(figsize=(6, 0.4))
ax.axis("off")
ax.text(0.5, 0.5, CAPTION_TEXT, ha="center", va="center",
        fontsize=CAPTION_FONTSIZE, family="DejaVu Sans",
        color="#3c4043", style="italic")
with PdfPages(CAPTION_PDF) as pdf:
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
caption_page = PdfReader(CAPTION_PDF).pages[0]
cap_w_src, cap_h_src = wh(caption_page)
caption_natural_aspect = cap_w_src / cap_h_src

# --- geometry
inner_w = COL_WIDTH - 2 * PANEL_PAD_X
header_h = LABEL_FONTSIZE + HEADER_TO_RULE + 1 + RULE_TO_CONTENT

# Panel A: bar chart fills the inner width at its natural aspect
bar_width  = inner_w
bar_height = bar_width / bar_aspect
panel_a_h  = PANEL_PAD_TOP + header_h + bar_height + PANEL_PAD_BOTTOM
panel_a_w  = COL_WIDTH

# Panel C: layouts strip + caption.
# The strip at its natural aspect (~6.83:1) is too short relative to the
# panel. We allow a controlled vertical stretch (Y_STRETCH) to make the
# kitchens visibly taller without changing the column width or distorting
# them too much.
Y_STRETCH = 1.85  # 1.0 = natural aspect; >1 = taller kitchens

strip_inner_w  = COL_WIDTH - 2 * STRIP_PAD_X
layouts_w      = strip_inner_w
layouts_natural_h = layouts_w / layouts_aspect
layouts_strip_h   = layouts_natural_h * Y_STRETCH

caption_w = inner_w
caption_h = caption_w / caption_natural_aspect
panel_c_content_h = layouts_strip_h + CAPTION_GAP + caption_h
panel_c_h = (PANEL_PAD_TOP + LABEL_FONTSIZE + HEADER_TO_RULE + 1
             + STRIP_PAD_TOP + panel_c_content_h + STRIP_PAD_BOTTOM)
panel_c_w = COL_WIDTH

left_col_height = panel_a_h + ROW_GAP + panel_c_h

# Panel B: heatmap fills full left column height
heat_height = left_col_height - PANEL_PAD_TOP - header_h - PANEL_PAD_BOTTOM
heat_width  = heat_height * heat_aspect
panel_b_w   = heat_width + 2 * PANEL_PAD_X
panel_b_h   = left_col_height

bar_scale       = bar_width  / bar_w_src
layouts_scale_x = layouts_w  / layouts_w_src
layouts_scale_y = layouts_strip_h / layouts_h_src
caption_scale   = inner_w    / cap_w_src
heat_scale      = heat_height / heat_h_src

page_w = MARGIN + panel_a_w + PADDING_BETWEEN + panel_b_w + MARGIN
page_h = MARGIN + left_col_height + MARGIN

left_x  = MARGIN
right_x = MARGIN + panel_a_w + PADDING_BETWEEN
panel_c = (left_x, MARGIN, panel_c_w, panel_c_h)
panel_a = (left_x, MARGIN + panel_c_h + ROW_GAP, panel_a_w, panel_a_h)
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
draw_panel(panel_c, "c", HEADERS["c"])

# Light gray background framing the layouts strip — extends beyond the
# strip in all directions so it's visible as a frame.
panel_c_x_dec, panel_c_y_dec, _, panel_c_h_dec = panel_c
rule_y_c_dec = (panel_c_y_dec + panel_c_h_dec - PANEL_PAD_TOP) - LABEL_FONTSIZE - HEADER_TO_RULE
strip_top_y_dec = rule_y_c_dec - STRIP_PAD_TOP

# Inset the strip slightly so the gray bg shows around it (8pt frame)
STRIP_BG_FRAME = 8
bg_x = panel_c_x_dec + STRIP_PAD_X - STRIP_BG_FRAME
bg_y = strip_top_y_dec - layouts_strip_h - STRIP_BG_FRAME
bg_w = panel_c_w - 2 * STRIP_PAD_X + 2 * STRIP_BG_FRAME
bg_h = layouts_strip_h + 2 * STRIP_BG_FRAME
strip_bg = patches.FancyBboxPatch(
    (bg_x, bg_y), bg_w, bg_h,
    boxstyle=f"round,pad=0,rounding_size=4.0",
    linewidth=0,
    facecolor=STRIP_BG_COLOR,
)
ax.add_patch(strip_bg)

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

# Panel A: bar chart (fills inner width)
ax_x, ax_top = content_origin(panel_a)
new_page.merge_transformed_page(
    bar_page,
    Transformation().scale(bar_scale).translate(ax_x, ax_top - bar_height))

# Panel C: layouts strip (stretched vertically) + caption
panel_c_x, panel_c_y, _, panel_c_h_used = panel_c
rule_y_c = (panel_c_y + panel_c_h_used - PANEL_PAD_TOP) - LABEL_FONTSIZE - HEADER_TO_RULE
strip_top_y = rule_y_c - STRIP_PAD_TOP
strip_x = panel_c_x + STRIP_PAD_X
layouts_y = strip_top_y - layouts_strip_h
new_page.merge_transformed_page(
    layouts_page,
    Transformation().scale(layouts_scale_x, layouts_scale_y).translate(strip_x, layouts_y))
caption_x = panel_c_x + PANEL_PAD_X
caption_y = layouts_y - CAPTION_GAP - caption_h
new_page.merge_transformed_page(
    caption_page,
    Transformation().scale(caption_scale).translate(caption_x, caption_y))

# Panel B: heatmap
bx_x, bx_top = content_origin(panel_b)
new_page.merge_transformed_page(
    heatmap_page,
    Transformation().scale(heat_scale).translate(bx_x, bx_top - heat_height))

# --- write
writer = PdfWriter()
writer.add_page(new_page)
with open(OUTPUT, "wb") as f:
    writer.write(f)

print(f"Combined PDF written to: {OUTPUT}")
print(f"Page size: {page_w:.1f} x {page_h:.1f} pt "
      f"({page_w/72:.2f} x {page_h/72:.2f} in)")
