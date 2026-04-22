#!/usr/bin/env python3
"""Figure 1 — MIP-Bench pipeline overview.

v3 fixes:
  * All text strictly inside its box (verified bounds).
  * Wider figure (18 x 6.6) so panel titles + boxes breathe.
  * Panel (a) concept-graph stats wrapped onto two lines.
  * Panel (b) VLM/response labels shortened; arrows redrawn.
  * Panel (c) bar chart moved right, given explicit margins; tick
    labels fully visible; findings bullets do not collide.
  * Raw editable formats: PNG (300 dpi), PDF (vector), SVG (vector).

Panels:
  (a) Input & annotation        → concept graph  ⟨C, R, B⟩
  (b) Evaluation & IGS scoring  → CR/RV/SF weights + equation
  (c) Key findings              → bar chart + headline bullets
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

# ── palette ──────────────────────────────────────────────────────────
C_SLIDE  = "#DCE9F7"
C_TEXT   = "#F5E6CC"
C_GRAPH  = "#D7E8CD"
C_VLM    = "#E9DCF4"
C_JUDGE  = "#FCE3D4"
C_CR     = "#4A90E2"
C_RV     = "#E07A5F"
C_SF     = "#81B29A"
C_IGS    = "#3D405B"
C_HUMAN  = "#7D8597"
C_EDGE   = "#5C5F66"
C_ARROW  = "#2B2D42"


def arrow(ax, start, end, color=C_ARROW, lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=18,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=3,
    ))


def box(ax, x, y, w, h, text, color, edge=C_EDGE, fs=11, weight="normal",
        rounding=0.03, text_color="#1b1b1b"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2, edgecolor=edge, facecolor=color, zorder=2,
    ))
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fs, color=text_color, weight=weight, zorder=4)


# ── figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18.0, 6.6), dpi=140)
gs  = gridspec.GridSpec(
    1, 3, figure=fig,
    width_ratios=[3.0, 3.8, 3.4],
    wspace=0.16,
    left=0.02, right=0.98, top=0.96, bottom=0.04,
)

# ================================================================
# (a) Input & annotation
# ================================================================
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off")
ax_a.text(0.03, 0.955, "(a)  Input & annotation",
          fontsize=13.5, weight="bold", color=C_IGS)

# two input sources
box(ax_a, 0.04, 0.76, 0.36, 0.13,
    "1,117 slides\n23 lectures", C_SLIDE, fs=10.5, weight="bold")
box(ax_a, 0.04, 0.57, 0.36, 0.13,
    "Instructor\nnarration", C_TEXT, fs=10.5, weight="bold")

# extractor
box(ax_a, 0.55, 0.64, 0.41, 0.22,
    "GPT-4o extractor\n(non-evaluated family)",
    C_GRAPH, fs=10.5, weight="bold")
arrow(ax_a, (0.40, 0.83), (0.55, 0.78))
arrow(ax_a, (0.40, 0.63), (0.55, 0.71))

# concept graph (wider, stats on own line)
box(ax_a, 0.04, 0.22, 0.92, 0.30,
    (r"Concept graph   $\mathcal{I}(\mathcal{S}) "
     r"= \langle\, \mathcal{C},\ \mathcal{R},\ \mathcal{B}\, \rangle$"
     "\n\n"
     "3,712 nodes   ·   3,856 directed edges\n"
     "100 % scope-boundary coverage"),
    C_GRAPH, fs=11, weight="bold")
arrow(ax_a, (0.75, 0.64), (0.50, 0.52), rad=0.25)

ax_a.text(0.50, 0.11,
          "Audited extraction   concept P = .886  R = .709",
          ha="center", fontsize=8.8, color=C_EDGE, style="italic")
ax_a.text(0.50, 0.06,
          "relation P = .867  R = .843",
          ha="center", fontsize=8.8, color=C_EDGE, style="italic")


# ================================================================
# (b) Evaluation & IGS
# ================================================================
ax_b = fig.add_subplot(gs[1])
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")
ax_b.text(0.03, 0.955, "(b)  Evaluation  +  IGS",
          fontsize=13.5, weight="bold", color=C_IGS)

# slide → VLM → response  (compact labels, no overflow)
box(ax_b, 0.02, 0.80, 0.19, 0.10, "Slide\nimage", C_SLIDE, fs=9.5,
    weight="bold")
box(ax_b, 0.27, 0.80, 0.25, 0.10, "Open-weight\nVLM", C_VLM, fs=9.5,
    weight="bold")
box(ax_b, 0.58, 0.80, 0.40, 0.10,
    r"Free-form response  $\hat{y}$", C_VLM, fs=9.5, weight="bold")
arrow(ax_b, (0.21, 0.85), (0.27, 0.85))
arrow(ax_b, (0.52, 0.85), (0.58, 0.85))

# judge (shortened)
box(ax_b, 0.20, 0.64, 0.60, 0.09,
    "GPT-4o judge  ·  5-point rubric  ·  GT-conditioned",
    C_JUDGE, fs=9.5, weight="bold")
arrow(ax_b, (0.78, 0.80), (0.60, 0.73), rad=-0.25)

# dimension badges
dim_x = [0.06, 0.40, 0.74]
dim_w = 0.20
dim_h = 0.18
dim_y = 0.34

for i, (label, w_val, color) in enumerate(zip(
    ["CR", "RV", "SF"],
    [0.40, 0.35, 0.25],
    [C_CR, C_RV, C_SF],
)):
    box(ax_b, dim_x[i], dim_y, dim_w, dim_h, "",
        color, edge=color, rounding=0.04)
    ax_b.text(dim_x[i] + dim_w / 2, dim_y + dim_h * 0.66,
              label, ha="center", va="center",
              fontsize=21, weight="bold", color="white")
    ax_b.text(dim_x[i] + dim_w / 2, dim_y + dim_h * 0.26,
              f"w = {w_val:.2f}",
              ha="center", va="center",
              fontsize=13.5, weight="bold", color="white")
    arrow(ax_b, (0.50, 0.64), (dim_x[i] + dim_w / 2, dim_y + dim_h))

# dim labels under each  (clearly below the badge, no collision with eq box)
ax_b.text(dim_x[0] + dim_w / 2, dim_y - 0.045,
          "Concept Recall", ha="center", fontsize=8.8, color=C_CR,
          weight="bold")
ax_b.text(dim_x[1] + dim_w / 2, dim_y - 0.045,
          "Relational Validity", ha="center", fontsize=8.8, color=C_RV,
          weight="bold")
ax_b.text(dim_x[2] + dim_w / 2, dim_y - 0.045,
          "Scope Fidelity", ha="center", fontsize=8.8, color=C_SF,
          weight="bold")

# Final Score — boxed equation
eq_y = 0.02
box(ax_b, 0.02, eq_y, 0.96, 0.22, "", "#FFF8E7",
    edge=C_IGS, rounding=0.05)
ax_b.text(0.50, eq_y + 0.175,
          "Final Score",
          ha="center", va="center",
          fontsize=11, color=C_IGS, style="italic")
ax_b.text(
    0.50, eq_y + 0.075,
    r"$\mathbf{IGS \;=\; 0.40\cdot CR \;+\; 0.35\cdot RV \;+\; 0.25\cdot SF}$",
    ha="center", va="center",
    fontsize=17, color=C_IGS,
)

for i in range(3):
    arrow(ax_b, (dim_x[i] + dim_w / 2, dim_y - 0.075),
          (0.22 + 0.28 * i, eq_y + 0.22),
          color=C_EDGE, lw=1.3)


# ================================================================
# (c) Key findings
# ================================================================
ax_c = fig.add_subplot(gs[2])
ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis("off")
ax_c.text(0.03, 0.955, "(c)  Key findings",
          fontsize=13.5, weight="bold", color=C_IGS)

# bar chart (explicit figure-coord axes, clear margins)
bar_ax = fig.add_axes([0.725, 0.60, 0.235, 0.29])
models = ["LLaVA-1.6-34B", "InternVL2-8B", "InternVL2-26B",
          "InternVL2-40B", "Qwen2-VL-7B", "Human pooled"]
igs_v  = [0.501, 0.595, 0.596, 0.619, 0.642, 0.690]
clrs   = [C_IGS, C_IGS, C_IGS, C_IGS, "#A855F7", C_HUMAN]
bars   = bar_ax.barh(models, igs_v, color=clrs, edgecolor="white",
                     height=0.72)
for bar, v in zip(bars, igs_v):
    bar_ax.text(v + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8.2, color="#222")
bar_ax.set_xlim(0.45, 0.77)
bar_ax.set_xticks([0.5, 0.6, 0.7])
bar_ax.tick_params(axis="both", labelsize=8)
bar_ax.set_xlabel("IGS  (clean test, n=183)", fontsize=9)
for spine in ("top", "right"):
    bar_ax.spines[spine].set_visible(False)
bar_ax.axvline(0.690, color=C_HUMAN, ls=":", lw=1.2)

# findings block — tight spacing, no collision with bar chart
y = 0.38
dy = 0.058
bullet_fs = 9.5
ax_c.text(0.04, y,          "● No VLM exceeds IGS 0.642  <  human 0.690",
          fontsize=bullet_fs)
ax_c.text(0.04, y - dy,     "● Universal scope–concept gap   SF − CR = +0.10…+0.14",
          fontsize=bullet_fs)
ax_c.text(0.04, y - 2*dy,   "● Scale paradox — 7B matches 40B", fontsize=bullet_fs)
ax_c.text(0.04, y - 3*dy,   "● Judge–human  r = 0.729  (88 % within 0.2)",
          fontsize=bullet_fs)
ax_c.text(0.04, y - 4*dy,   "● Ranking stable to 30 % recall drop  (ρ̄ = 0.94)",
          fontsize=bullet_fs)
ax_c.text(0.04, y - 5*dy,   "● Two-judge Kendall τ = 1.0  (Claude vs GPT-4o)",
          fontsize=bullet_fs)
ax_c.text(0.04, y - 6*dy,   "● Supervised LoRA does not close gap  (ΔIGS ≤ 0)",
          fontsize=bullet_fs)


# ── save ───────────────────────────────────────────────────────────
out_png = "/sessions/peaceful-hopeful-hypatia/mnt/ALIVE-slide/medlecturebench_fig_1.png"
out_pdf = "/sessions/peaceful-hopeful-hypatia/mnt/ALIVE-slide/medlecturebench_fig_1.pdf"
out_svg = "/sessions/peaceful-hopeful-hypatia/mnt/ALIVE-slide/medlecturebench_fig_1.svg"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf,            bbox_inches="tight", facecolor="white")
plt.savefig(out_svg,            bbox_inches="tight", facecolor="white")
print("Saved:", out_png)
print("Saved:", out_pdf)
print("Saved:", out_svg)
