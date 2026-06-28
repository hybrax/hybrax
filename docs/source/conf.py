project   = "Bioprocess Modeling docs"
author    = "Bioprocess Modeling Collective"
copyright = "2026, Bioprocess Modeling Collective"
release   = version = "0.1.0"

extensions = [
    "autoapi.extension",       # static API gen over BOTH packages — no import
    "myst_parser",             # narrative .md
    "sphinx.ext.napoleon",     # Google-style docstrings
    "sphinx.ext.viewcode",     # [source] links
    "sphinx.ext.intersphinx",  # external refs only (python/numpy/jax)
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- Furo ---
html_theme = "furo"
html_title = "Bioprocess Modeling docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#2563eb", "color-brand-content": "#2563eb"},
    "dark_css_variables":  {"color-brand-primary": "#60a5fa", "color-brand-content": "#60a5fa"},
}

# --- AutoAPI over BOTH package source trees (paths relative to this conf.py) ---
autoapi_type = "python"
autoapi_dirs = ["../../bp-train/bp_train", "../../bp-format/bp_format"]
autoapi_root = "autoapi"
autoapi_keep_files = False
autoapi_add_toctree_entry = False     # we place autoapi/index in index.md ourselves
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"
autoapi_options = ["members", "undoc-members", "show-inheritance",
                   "show-module-summary"]
autodoc_typehints = "description"

# --- Napoleon (Google-style + type hints) ---
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = napoleon_use_rtype = napoleon_use_ivar = True
napoleon_preprocess_types = True

# --- MyST ---
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist",
                          "linkify", "substitution", "tasklist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
    "jax":    ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_disabled_reftypes = ["*"]

# --- Keep the build clean (dead ../bp_*/foo.py md links + 3rd-party type refs) ---
nitpicky = False
suppress_warnings = ["myst.xref_missing", "myst.header",
                     "autoapi.python_import_resolution", "ref.python",
                     "misc.highlighting_failure"]
