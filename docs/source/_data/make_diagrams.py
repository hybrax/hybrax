"""Generate the two architecture diagrams as theme-aware SVG pairs.

Not part of the Sphinx build (source/_data/** is excluded). Run directly to
regenerate source/_static/diagram_*_{light,dark}.svg:

    python source/_data/make_diagrams.py

Each diagram is rendered twice — once with dark ink for light page
backgrounds, once with light ink for dark page backgrounds — both on a fully
transparent canvas. custom.css toggles which one is visible, the same
technique already used for the sidebar logo.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).parent.parent / "_static"

BRAND_LIGHT = "#2563eb"   # matches conf.py html_theme_options light accent
BRAND_DARK = "#60a5fa"    # matches conf.py html_theme_options dark accent

THEMES = {
    "light": dict(ink="#1e293b", muted="#475569", accent=BRAND_LIGHT,
                  box_fill="#eff6ff", box_edge="#93b4f0"),
    "dark":  dict(ink="#e5e9f0", muted="#a8b3c5", accent=BRAND_DARK,
                  box_fill="#1e2a3f", box_edge="#3b5578"),
}

MONO = FontProperties(family="monospace")


def _save(fig, name, theme):
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
    path = OUT / f"{name}_{theme}.svg"
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def box(ax, xy, w, h, text, c, *, fontsize=10.5, weight="normal", mono=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=c["box_edge"], facecolor=c["box_fill"],
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=c["ink"], weight=weight,
            fontproperties=MONO if mono else None,
            linespacing=1.5)


def arrow(ax, p0, p1, c, *, style="-|>", lw=1.6, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=c["accent"], connectionstyle=connectionstyle,
        shrinkA=0, shrinkB=0,
    ))


# ---------------------------------------------------------------------------
# Diagram 1: format/index.md — "what bp-format derives from your description"
# ---------------------------------------------------------------------------
def make_format_diagram(theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    ax.text(0.15, 3.35, "your description", fontsize=11.5, weight="bold", color=c["ink"])
    ax.text(4.55, 3.35, "what bp-format derives from it", fontsize=11.5, weight="bold",
            color=c["ink"])

    left_items = ["reactor medium", "volume + feeds", "process variables", "biological ODE"]
    ys = [2.55, 1.85, 1.15, 0.45]
    for label, y in zip(left_items, ys):
        box(ax, (0.15, y), 2.55, 0.55, label, c, fontsize=10)

    # brace-like merge: four short lines from each left box converge to one point,
    # then one arrow continues to the right column.
    merge_x = 3.15
    merge_y = 1.7
    for y in ys:
        ax.plot([2.70, merge_x], [y + 0.275, merge_y], color=c["accent"], lw=1.3,
                solid_capstyle="round")
    arrow(ax, (merge_x, merge_y), (4.15, merge_y), c, lw=2.0)

    right = [
        ("ProcessOrdering", "canonical name / index layout"),
        ("ControlSplines", "controlled inputs, evaluable at any t"),
        ("RhsOde", "dc/dt = biology + transport"),
        ("pseudobatch", "dilution-corrected concentrations c*"),
    ]
    ry0 = 2.85
    for i, (name, desc) in enumerate(right):
        y = ry0 - i * 0.72
        ax.text(4.25, y, name, fontsize=10.5, weight="bold", color=c["accent"],
                fontproperties=MONO, va="center")
        ax.text(4.25, y - 0.30, desc, fontsize=9, color=c["muted"], va="center")

    fig.tight_layout()
    _save(fig, "diagram_format_pipeline", theme)


# ---------------------------------------------------------------------------
# Diagram 2: train/index.md — the prepare -> train -> forward/loo pipeline
# ---------------------------------------------------------------------------
def make_train_diagram(theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    stage_w = 3.0
    stage_x = 0.25

    box(ax, (stage_x, 6.65), stage_w, 0.55,
        "data.json", c, fontsize=11, weight="bold", mono=True)
    ax.text(stage_x + stage_w + 0.35, 6.925,
            "a bp-format BioProcessCollection",
            fontsize=9, color=c["muted"], va="center")

    arrow(ax, (stage_x + stage_w / 2, 6.65), (stage_x + stage_w / 2, 5.95), c)
    ax.text(stage_x + stage_w + 0.35, 6.30, "bp-train prepare", fontsize=10,
            weight="bold", color=c["accent"], va="center", fontproperties=MONO)
    ax.text(stage_x + stage_w + 0.35, 6.00,
            "+ custom.py: transform_process_collection\n"
            "             augment_state_values",
            fontsize=8.7, color=c["muted"], va="center", fontproperties=MONO)

    box(ax, (stage_x, 5.10), stage_w, 0.55, "prepared/", c, fontsize=11,
        weight="bold", mono=True)
    ax.text(stage_x + stage_w + 0.35, 5.375,
            "the training problem: layouts,\ncontrol splines, targets",
            fontsize=9, color=c["muted"], va="center")

    arrow(ax, (stage_x + stage_w / 2, 5.10), (stage_x + stage_w / 2, 4.30), c)
    ax.text(stage_x + stage_w + 0.35, 4.75, "bp-train train", fontsize=10,
            weight="bold", color=c["accent"], va="center", fontproperties=MONO)
    ax.text(stage_x + stage_w + 0.35, 4.30,
            "+ custom.py: estimate_all_scales\n"
            "             build_reaction_module\n"
            "             build_loss_module\n"
            "             build_learning_rate\n"
            "             build_optimizer",
            fontsize=8.7, color=c["muted"], va="center", fontproperties=MONO)

    box(ax, (stage_x, 3.05), stage_w, 1.25, "", c)
    ax.text(stage_x + 0.20, 4.02, "run/", fontsize=11, weight="bold", color=c["ink"],
            fontproperties=MONO)
    ax.text(stage_x + 0.35, 3.58,
            "model/params.eqx\nmetrics.csv\npredictions.csv\n<process>.png",
            fontsize=8.5, color=c["muted"], va="center", fontproperties=MONO,
            linespacing=1.6)

    # two branches down to forward / loo
    branch_y_top = 3.05
    branch_y_mid = 2.55
    cx = stage_x + stage_w / 2
    ax.plot([cx, cx], [branch_y_top, branch_y_mid], color=c["accent"], lw=1.6)
    left_x, right_x = cx - 1.05, cx + 1.05
    ax.plot([left_x, right_x], [branch_y_mid, branch_y_mid], color=c["accent"], lw=1.6)
    arrow(ax, (left_x, branch_y_mid), (left_x, 1.95), c)
    arrow(ax, (right_x, branch_y_mid), (right_x, 1.95), c)

    box_w = 2.0
    box(ax, (left_x - box_w / 2, 1.35), box_w, 0.55, "forward", c, fontsize=10.5,
        weight="bold", mono=True)
    ax.text(left_x - box_w / 2, 1.05, "re-simulate, export dense\ntrajectories, ensemble",
            fontsize=8.5, color=c["muted"], va="top")

    box(ax, (right_x - box_w / 2, 1.35), box_w, 0.55, "loo", c, fontsize=10.5,
        weight="bold", mono=True)
    ax.text(right_x - box_w / 2, 1.05, "cross-validate (wraps\ntrain + forward per fold)",
            fontsize=8.5, color=c["muted"], va="top")

    fig.tight_layout()
    _save(fig, "diagram_train_pipeline", theme)


# ---------------------------------------------------------------------------
# Diagram 3: start/concepts.md — "the shape of the whole thing"
# ---------------------------------------------------------------------------
def make_shape_diagram(theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(1.25, 6.6)
    ax.axis("off")

    stage_x, stage_w = 0.2, 2.7
    cx = stage_x + stage_w / 2
    label_x = stage_x + stage_w + 0.3

    ax.text(cx, 6.35, "your CSVs / exports", ha="center", va="center",
            fontsize=10.5, style="italic", color=c["muted"])

    arrow(ax, (cx, 6.18), (cx, 5.65), c)
    ax.text(label_x, 5.915, "you write this part once\n(Tutorial 1)",
            fontsize=9, color=c["muted"], va="center")

    box(ax, (stage_x, 4.95), stage_w, 0.65, "", c)
    ax.text(cx, 5.395, "bp-format", ha="center", va="center", fontsize=10.5,
            color=c["ink"], fontproperties=MONO)
    ax.text(cx, 5.105, "data model", ha="center", va="center", fontsize=10.5,
            color=c["ink"])
    ax.text(label_x, 5.375, "BioProcess", fontsize=9.2, color=c["muted"],
            va="center", fontproperties=MONO)
    ax.text(label_x, 5.15, "medium, volume, feeds,\nsamples, measurements",
            fontsize=9, color=c["muted"], va="center")

    arrow(ax, (cx, 4.95), (cx, 4.40), c)
    ax.text(label_x, 4.775, "build_rhs_ode()", fontsize=9.2, color=c["muted"],
            va="center", fontproperties=MONO)
    ax.text(label_x, 4.575, "bp-format assembles the physics",
            fontsize=9, color=c["muted"], va="center")

    rhs_y = 3.75
    rhs_cy = rhs_y + 0.325
    box(ax, (stage_x, rhs_y), stage_w, 0.65, "RhsOde", c, fontsize=10.5, weight="bold",
        mono=True)
    ax.text(label_x, rhs_cy, "dc/dt = biology(rates)\n+ transport(feeds, dilution, samples)",
            fontsize=8.8, color=c["muted"], va="center", fontproperties=MONO)

    arrow(ax, (cx, rhs_y), (cx, 3.20), c)

    train_y = 2.35
    train_h = 0.85
    train_cy = train_y + train_h / 2
    box(ax, (stage_x, train_y), stage_w, train_h,
        "bp-train\nprepare\N{RIGHTWARDS ARROW}train\n\N{RIGHTWARDS ARROW}forward/loo",
        c, fontsize=10.5, mono=True)

    arrow(ax, (cx, train_y), (cx, 1.80), c)
    ax.text(cx, 1.55, "predictions, rates, metrics", ha="center", va="center",
            fontsize=10, color=c["ink"])

    # Right column: the two things you supply. The reaction module aligns
    # with RhsOde (it feeds rates straight into the ODE); the loss module
    # aligns with bp-train and reaches it via custom.py, mirroring how
    # every hook is actually wired in (see train/index.md).
    right_x, right_w = 6.1, 2.2

    ax.text(right_x, rhs_y + 0.65 + 0.28, "you supply this part", fontsize=9.5,
            weight="bold", style="italic", color=c["accent"], va="center")

    rm_h = 0.6
    rm_y = rhs_cy - rm_h / 2
    box(ax, (right_x, rm_y), right_w, rm_h, "reaction\nmodule", c, fontsize=10)
    ax.text(right_x + right_w / 2, rm_y - 0.24, "predicts the rates",
            fontsize=8.7, color=c["muted"], ha="center", va="center")
    arrow(ax, (right_x, rhs_cy), (stage_x + stage_w, rhs_cy), c, lw=1.4)

    lm_h = 0.6
    lm_y = train_cy - lm_h / 2
    box(ax, (right_x, lm_y), right_w, lm_h, "loss\nmodule", c, fontsize=10)
    ax.text(right_x + right_w / 2, lm_y - 0.24, "scores the trajectory",
            fontsize=8.7, color=c["muted"], ha="center", va="center")

    # loss module -> custom.py -> bp-train, all at bp-train's row height.
    cpy_w, cpy_h = 1.45, 0.42
    cpy_x = stage_x + stage_w + 0.75
    cpy_y = train_cy - cpy_h / 2
    arrow(ax, (right_x, train_cy), (cpy_x + cpy_w, train_cy), c, lw=1.4)
    box(ax, (cpy_x, cpy_y), cpy_w, cpy_h, "custom.py", c, fontsize=9.5, mono=True)
    arrow(ax, (cpy_x, train_cy), (stage_x + stage_w, train_cy), c, lw=1.6)

    fig.tight_layout()
    _save(fig, "diagram_concepts_shape", theme)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        make_format_diagram(theme)
        make_train_diagram(theme)
        make_shape_diagram(theme)
