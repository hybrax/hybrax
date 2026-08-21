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


def box(ax, xy, w, h, text, c, *, fontsize=10.5, weight="normal", mono=False,
        dashed=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1 if dashed else 1.3,
        edgecolor=c["muted"] if dashed else c["box_edge"],
        facecolor="none" if dashed else c["box_fill"],
        linestyle="--" if dashed else "-",
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=c["muted"] if dashed else c["ink"],
            weight=weight, fontproperties=MONO if mono else None,
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
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    # Two plain text-stack columns, same style, same row spacing, no arrows
    # between them: the four inputs together produce the four derived
    # objects, not a one-to-one pairing an arrow would misleadingly imply.
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
    left_x, right_x = 0.3, 3.92
    ry0 = 2.85
    for i, ((lname, ldesc), (rname, rdesc)) in enumerate(zip(left, right)):
        y = ry0 - i * 0.72
        ax.text(left_x, y, lname, fontsize=10.5, weight="bold", color=c["accent"],
                fontproperties=MONO, va="center")
        ax.text(left_x, y - 0.30, ldesc, fontsize=9, color=c["muted"], va="center")
        ax.text(right_x, y, rname, fontsize=10.5, weight="bold", color=c["accent"],
                fontproperties=MONO, va="center")
        ax.text(right_x, y - 0.30, rdesc, fontsize=9, color=c["muted"], va="center")

    # One dashed cell per column, both the same width (the wider column's
    # own content width), with the column's title sitting on top of its own
    # outline rather than floating above both.
    box_top, box_bottom = 3.15, 0.14
    cell_w = 2.87
    for x, title in ((left_x, "your input"), (right_x, "derived objects")):
        x0, x1 = x - 0.2, x + cell_w + 0.2
        ax.add_patch(FancyBboxPatch(
            (x0, box_bottom), x1 - x0, box_top - box_bottom,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.2, edgecolor=c["muted"], facecolor="none", linestyle="--",
        ))
        ax.text(x, box_top + 0.1, title, fontsize=11.5, weight="bold", color=c["ink"],
                va="bottom")

    fig.tight_layout()
    _save(fig, "diagram_format_pipeline", theme)


# ---------------------------------------------------------------------------
# Diagram 2: start/concepts.md and train/index.md — "the shape of the whole
# thing". Shared shape, two detail levels: train/index.md additionally shows
# the real file each arrow and input actually is; concepts.md stays plain.
# ---------------------------------------------------------------------------
def make_shape_diagram(theme, annotate_files):
    c = THEMES[theme]
    if annotate_files:
        fig, ax = plt.subplots(figsize=(8.05, 6.1))
        ax.set_xlim(-0.45, 8.4)
    else:
        fig, ax = plt.subplots(figsize=(7.6, 6.1))
        ax.set_xlim(0, 8.4)
    ax.set_ylim(3.3, 8.95)
    ax.axis("off")

    # One width for every cell in the diagram, solid or dashed: the main
    # pipeline boxes, the left-side input-file boxes, and (annotate_files
    # only) the right-side config-file boxes.
    stage_w, stage_h = 2.2, 0.6
    gap = 0.45
    row_pitch = stage_h + gap
    ARROW_LEN = 0.55   # every simple connector arrow in this diagram is this long

    main_x = 2.75
    main_cx = main_x + stage_w / 2
    ys = [8.15 - i * row_pitch for i in range(4)]   # box bottoms, top to bottom

    # loo folds into train/loo: both just mean "produce a trained model", the
    # only difference is whether it's held out and scored per fold. The
    # ensemble-vs-single distinction lives in prose elsewhere, not here.
    main_labels = ["hybrax-format", "hybrax\nprepare", "hybrax\ntrain / loo",
                   "hybrax\nforward"]
    for y, label in zip(ys, main_labels):
        box(ax, (main_x, y), stage_w, stage_h, label, c, fontsize=9.5,
            weight="bold", mono=True)
    for y0, y1 in zip(ys[:3], ys[1:]):
        arrow(ax, (main_cx, y0), (main_cx, y1 + stage_h), c, lw=1.8)
    if annotate_files:
        # What actually flows on each main-pipeline arrow, real names only:
        # prepared.json is the one artifact prepare writes that training reads
        # (prepare.md#what-it-writes); run/ is what forward actually consumes
        # ({"models": ["run"]}, forward.md) since params.eqx alone "is not a
        # model" (save_load_predict.md#gotchas) — the rest is rebuilt from the
        # rest of run/.
        for i, label in zip(range(3), ["data.json", "prepared.json", "run/"]):
            mid_y = ys[i] - (ys[i] - ys[i + 1] - stage_h) / 2
            ax.text(main_cx + 0.15, mid_y, label, fontsize=7.8, color=c["muted"],
                    fontproperties=MONO, va="center")
    # Same vertical rhythm as every inter-stage arrow above (gap, not
    # ARROW_LEN): the last box's own bottom edge already sets that spacing.
    arrow(ax, (main_cx, ys[3]), (main_cx, ys[3] - gap), c, lw=1.8)
    ax.text(main_cx, ys[3] - gap - 0.05, "predictions,\nrates, metrics",
            ha="center", va="top", fontsize=10, color=c["ink"])

    # Right side, train/index.md only: each stage's own config file. A config
    # file is an input (--config), same as everything on the left, so the
    # arrow points into the stage box, not out of it.
    if annotate_files:
        config_x = main_x + stage_w + ARROW_LEN
        configs = [
            (ys[0], "import.py"),
            (ys[1], "prepare-\nconfig.json"),
            (ys[2], "train-config.json,\nloo-config.json"),
            (ys[3], "forward-\nconfig.json"),
        ]
        for y, label in configs:
            cy = y + stage_h / 2
            arrow(ax, (config_x, cy), (main_x + stage_w, cy), c, lw=1.4)
            box(ax, (config_x, y), stage_w, stage_h, label, c, fontsize=7.8,
                mono=True, dashed=True)

    # What you supply, in plain language, feeding in from the left of each
    # main-pipeline stage. Every arrow here is exactly ARROW_LEN long.
    inputs = [
        (ys[0], ["Measured Data"]),
        (ys[1], ["Process Transformation", "Process Augmentation"]),
        (ys[2], ["Reaction Module, Loss Module", "Scales, Optimizer, Learning Rate"]),
        (ys[3], ["Old/New Controls"]),
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

    # A second layer, train/index.md only: the real file each input group
    # actually lives in. prepare's and train's hooks are both just custom.py,
    # so one dashed box spans both groups rather than repeating the label.
    if annotate_files:
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
        file_box(md_top, md_bottom, stage_w, "data.csv, data.xlsx")

        prep_top, _ = group_span(ys[1], inputs[1][1])
        _, train_bottom = group_span(ys[2], inputs[2][1])
        file_box(prep_top, train_bottom, stage_w, "custom.py")

        nc_top, nc_bottom = group_span(ys[3], inputs[3][1])
        file_box(nc_top, nc_bottom, stage_w, "data.json, new_data.json")

    fig.tight_layout()
    _save(fig, "diagram_train_pipeline" if annotate_files else "diagram_concepts_shape",
          theme)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        make_format_diagram(theme)
        make_shape_diagram(theme, annotate_files=False)
        make_shape_diagram(theme, annotate_files=True)
