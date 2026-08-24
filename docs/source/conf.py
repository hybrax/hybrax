import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

_HERE = Path(__file__).parent

project = "Bioprocess Modeling docs"
author = "Bioprocess Modeling Collective"
copyright = "2026, Bioprocess Modeling Collective"


def _v(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "unknown"


# Read from the installed package (importlib.metadata — never imports it).
release = version = f"hybrax {_v('hybrax')}"

extensions = [
    "autoapi.extension",  # static API gen over hybrax — no import
    "myst_nb",  # executable MyST markdown (supersedes myst_parser)
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",  # [source] links
    "sphinx.ext.intersphinx",  # external refs only (python/numpy/jax)
    "sphinx.ext.mathjax",  # renders the dollarmath below
    "sphinx_copybutton",
    "sphinx_design",  # dropdown directive (full-file listings in the gallery)
]

# myst-nb owns .md and .ipynb. Files without {code-cell} blocks are parsed as
# plain MyST markdown, so ordinary pages are unaffected.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_data/**",  # dataset generator + generated data — not documents
    "**/.ipynb_checkpoints",
]
# Project-local template overrides (e.g. sidebar/brand.html). Sphinx checks this
# before the theme's own templates, so it's the supported way to customize furo
# without touching the installed package.
templates_path = ["_templates"]

# --- Furo ---
html_theme = "furo"
# Without both of these, Sphinx generates only one static pygments.css (light),
# so Furo has nothing to swap in for dark mode and every code block stays light
# regardless of theme.
pygments_style = "sphinx"
pygments_dark_style = "monokai"
# Drives the browser <title>. The homepage renders this verbatim; every other
# page renders "{page title} - {this}". The sidebar brand text is a SEPARATE
# string, overridden in _templates/sidebar/brand.html — furo reads both from
# the same `docstitle` value by default, which is why a template override was
# needed to make them read differently on purpose.
html_title = "Bioprocess Modeling with Hybrax"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_logo = "_static/hybrax_logo.png"
html_favicon = "_static/favicon.png"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
}

# --- MyST-NB execution -------------------------------------------------------
# Pages with {code-cell} blocks run for real at build time against the installed
# package. That is deliberate: an API break must break the docs build instead of
# silently rotting a page. "cache" means only pages whose source changed re-run.
nb_execution_mode = "cache"
nb_execution_cache_path = str(_HERE.parent / "_scratch" / "jupyter_cache")
nb_execution_timeout = 900  # a tutorial that trains a model needs headroom
nb_execution_raise_on_error = True  # a failing cell fails the build
nb_merge_streams = True

# The kernel inherits this environment. Keep the docs build single-device and
# headless so it never competes for cores or tries to open a window.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYBRAX_TRAIN_DEVICES", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_HERE.parent / "_scratch" / "mpl"))

# --- AutoAPI over the hybrax source tree (paths relative to this conf.py) ---
# Each dir points directly at a subpackage, so autoapi generates
# autoapi/hybrax/format/... and autoapi/hybrax/train/.... diffrax_callbacks nests
# under train/, so it is picked up automatically as
# autoapi/hybrax/train/diffrax_callbacks/...
autoapi_type = "python"
autoapi_dirs = ["../../src/hybrax/format", "../../src/hybrax/train"]
# Runnable demo scripts, not an importable API: their top-level `for` loops
# rebind module-level names (e.g. `params`), which autoapi documents as
# duplicate attribute definitions, and they're never linked into a toctree.
autoapi_ignore = ["*/diffrax_callbacks/examples/*"]
autoapi_root = "autoapi"
autoapi_keep_files = False
autoapi_add_toctree_entry = False  # we place autoapi/index in index.md ourselves
autoapi_member_order = "groupwise"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autodoc_typehints = "description"

# --- Napoleon (Google-style + type hints) ---
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = napoleon_use_rtype = napoleon_use_ivar = True
napoleon_preprocess_types = True

# --- MyST ---
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_disabled_reftypes = ["*"]

# --- Warning policy ---
# `myst.xref_missing` is deliberately NOT suppressed any more: bp-docs owns all of
# its own prose now, so a dead cross-reference is a real bug and -W should catch it.
# What remains is third-party noise we cannot fix from here.
#
# `docutils` and `duplicate_object` are suppressed because every instance comes from
# autoapi's *generated* RST, i.e. from docstring formatting inside hybrax. That
# package is a read-only input to this build. Our own pages are MyST markdown
# and report under `myst.*`, which is not suppressed.
nitpicky = False
suppress_warnings = [
    "myst.header",
    "autoapi.python_import_resolution",
    "ref.python",
    "misc.highlighting_failure",
    "docutils",
    "duplicate_object",
]
