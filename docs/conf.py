import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Relational Transformers"
author = "RelativeDB"
copyright = "2026, RelativeDB"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser", "sphinx_copybutton"]
templates_path = ["_templates"]
exclude_patterns = [".pytest_cache", "README.md", "docs/_build"]
html_theme = "furo"
html_title = "Relational Transformers"
autodoc_typehints = "description"
myst_heading_anchors = 3
