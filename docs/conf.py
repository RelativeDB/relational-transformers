import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Relational Transformers"
author = "RelativeDB"
copyright = "2026, RelativeDB"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinx_toolbox.collapse",
]
templates_path = ["_templates"]
exclude_patterns = [
    ".pytest_cache",
    "README.md",
    "docs/_build",
    ".venv-docs",
    "release",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "https://relationaltransformers.com",
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_title = "Relational Transformers"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sourcelink = False
html_context = {
    "display_github": True,
    "github_user": "RelativeDB",
    "github_repo": "relational-transformers",
    "github_version": "main/",
}

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
myst_heading_anchors = 3

intersphinx_mapping = {
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
