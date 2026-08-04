from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_docs_headers.py"
SPEC = spec_from_file_location("compare_docs_headers", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_documentation_surface_matches_sentence_transformers_roles():
    reference = Path.home() / "sentence-transformers"
    if not reference.exists():
        return

    result = MODULE.audit(reference, ROOT)

    assert result["coverage"] == 1.0, "\n".join(result["missing"])


def test_header_parser_ignores_code_fence_comments(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("# Real heading\n\n```python\n# not a heading\n```\n")

    assert MODULE.headings(page) == ["Real heading"]
