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
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    # Outer frame: everything in both columns is hybrax-format's; bp-train never
    # re-derives any of it (see the prose right below this figure).
    ax.add_patch(FancyBboxPatch(
        (0.1, 0.1), 9.4, 3.65,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor=c["muted"], facecolor="none", linestyle="--",
    ))
    ax.text(0.3, 3.58, "hybrax-format", fontsize=9.5, weight="bold", color=c["muted"],
            fontproperties=MONO)

    ax.text(0.3, 3.2, "your input", fontsize=11.5, weight="bold", color=c["ink"])
    ax.text(4.75, 3.2, "derived objects", fontsize=11.5, weight="bold", color=c["ink"])

    # Two plain text-stack columns, same style, same row spacing, no boxes and no
    # connecting lines: the four inputs together produce the four derived objects,
    # not a one-to-one pairing an arrow would misleadingly imply.
    left = [
        ("ReactorMediumComponent", "experimental concentrations of each species"),
        ("Volume, Inflow, Outflow", "feeds, boluses, sample draws"),
        ("ProcessVariable", "pH, DO, temperature, ..."),
        ("BiologicalOde", "your rate expressions"),
    ]
    right = [
        ("ProcessOrdering", "canonical name / index layout"),
        ("ControlSplines", "controlled inputs, evaluable at any t"),
        ("RhsOde", "dc/dt = biology + transport"),
        ("PseudobatchTransform", "dilution-corrected concentrations"),
    ]
    ry0 = 2.85
    for i, ((lname, ldesc), (rname, rdesc)) in enumerate(zip(left, right)):
        y = ry0 - i * 0.72
        ax.text(0.3, y, lname, fontsize=10.5, weight="bold", color=c["accent"],
                fontproperties=MONO, va="center")
        ax.text(0.3, y - 0.30, ldesc, fontsize=9, color=c["muted"], va="center")
        ax.text(4.75, y, rname, fontsize=10.5, weight="bold", color=c["accent"],
                fontproperties=MONO, va="center")
        ax.text(4.75, y - 0.30, rdesc, fontsize=9, color=c["muted"], va="center")

    fig.tight_layout()
    _save(fig, "diagram_format_pipeline", theme)


# ---------------------------------------------------------------------------
# Diagram 2: start/concepts.md and train/index.md — "the shape of the whole
# thing". Shared by both pages: the same pipeline, viewed from either side.
# ---------------------------------------------------------------------------
def make_shape_diagram(theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.05, 6.1))
    ax.set_xlim(-0.45, 8.4)
    ax.set_ylim(3.3, 8.95)
    ax.axis("off")

    stage_w, stage_h = 1.95, 0.6
    gap = 0.45
    row_pitch = stage_h + gap
    ARROW_LEN = 0.55   # every simple connector arrow in this diagram is this long

    main_x = 2.75
    main_cx = main_x + stage_w / 2
    ys = [8.15 - i * row_pitch for i in range(4)]   # box bottoms, top to bottom

    main_labels = ["hybrax-format\ndata", "hybrax\nprepare", "hybrax\ntrain",
                   "hybrax\nforward"]
    for y, label in zip(ys, main_labels):
        box(ax, (main_x, y), stage_w, stage_h, label, c, fontsize=9.5,
            weight="bold", mono=True)
    for y0, y1 in zip(ys[:3], ys[1:]):
        arrow(ax, (main_cx, y0), (main_cx, y1 + stage_h), c, lw=1.8)
    ax.text(main_cx + 0.15, ys[0] - (ys[0] - ys[1] - stage_h) / 2, "data.json",
            fontsize=7.8, color=c["muted"], fontproperties=MONO, va="center")
    arrow(ax, (main_cx, ys[3]), (main_cx, ys[3] - ARROW_LEN), c, lw=1.8)
    ax.text(main_cx, ys[3] - ARROW_LEN - 0.3, "predictions,\nrates, metrics",
            ha="center", va="top", fontsize=10, color=c["ink"])

    # Alternative route: prepare -> loo -> forward (ensemble), beside train and
    # forward respectively (each a stand-in for the stage level with it). Its
    # own output stays separate from the main route's, never merging into it.
    branch_x = main_x + stage_w + 1.55
    branch_cx = branch_x + stage_w / 2
    loo_y, fwd_ens_y = ys[2], ys[3]

    prep_cy = ys[1] + stage_h / 2
    ax.plot([main_x + stage_w, branch_cx], [prep_cy, prep_cy], color=c["accent"],
            lw=1.6, solid_capstyle="round")
    arrow(ax, (branch_cx, prep_cy), (branch_cx, loo_y + stage_h), c, lw=1.6)

    box(ax, (branch_x, loo_y), stage_w, stage_h, "hybrax\nloo", c, fontsize=9.5,
        weight="bold", mono=True)
    arrow(ax, (branch_cx, loo_y), (branch_cx, fwd_ens_y + stage_h), c, lw=1.8)
    box(ax, (branch_x, fwd_ens_y), stage_w, stage_h, "hybrax forward\n(ensemble)", c,
        fontsize=9, weight="bold", mono=True)
    arrow(ax, (branch_cx, fwd_ens_y), (branch_cx, fwd_ens_y - ARROW_LEN), c, lw=1.8)
    ax.text(branch_cx, fwd_ens_y - ARROW_LEN - 0.3, "ensemble predictions,\nrates, metrics",
            ha="center", va="top", fontsize=10, color=c["ink"])

    # What you supply, in plain language, feeding in from the left of each
    # main-pipeline stage. Every arrow here is exactly ARROW_LEN long.
    inputs = [
        (ys[0], ["Measured Data"]),
        (ys[1], ["Transformed Processes", "Augmented Processes"]),
        (ys[2], ["Reaction Module, Loss Module", "Scales, Optimizer, Learning Rate"]),
        (ys[3], ["New Controls"]),
    ]
    arrow_right = main_x - 0.05
    arrow_left = arrow_right - ARROW_LEN
    text_x = arrow_left - 0.15
    line_gap = 0.34

    def group_span(y, lines):
        cy = y + stage_h / 2
        half = (len(lines) - 1) / 2 * line_gap
        return cy + half, cy - half   # top, bottom

    for y, lines in inputs:
        cy = y + stage_h / 2
        n = len(lines)
        for i, line in enumerate(lines):
            ty = cy + (n - 1) / 2 * line_gap - i * line_gap
            ax.text(text_x, ty, line, ha="right", va="center", fontsize=8.5,
                    color=c["muted"])
        arrow(ax, (arrow_left, cy), (arrow_right, cy), c, lw=1.4)

    # A second layer: the real file each input group actually lives in.
    # prepare's and train's hooks are both just custom.py, so one dashed box
    # spans both groups rather than repeating the label.
    def file_box(y_top, y_bottom, width, label):
        pad_v, pad_h = 0.2, 0.18
        x0 = text_x - width - pad_h
        x1 = arrow_left + 0.05
        box_top = y_top + pad_v
        ax.add_patch(FancyBboxPatch(
            (x0, y_bottom - pad_v), x1 - x0, box_top - (y_bottom - pad_v),
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1, edgecolor=c["muted"], facecolor="none", linestyle="--",
        ))
        ax.text(x0 + 0.1, box_top + 0.06, label, fontsize=7.8,
                color=c["muted"], fontproperties=MONO, va="bottom")

    md_top, md_bottom = group_span(ys[0], inputs[0][1])
    file_box(md_top, md_bottom, 1.6, "data.csv, data.xlsx")

    prep_top, _ = group_span(ys[1], inputs[1][1])
    _, train_bottom = group_span(ys[2], inputs[2][1])
    file_box(prep_top, train_bottom, 2.2, "custom.py")

    nc_top, nc_bottom = group_span(ys[3], inputs[3][1])
    file_box(nc_top, nc_bottom, 1.05, "new_data.json")

    fig.tight_layout()
    _save(fig, "diagram_concepts_shape", theme)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        make_format_diagram(theme)
        make_shape_diagram(theme)
