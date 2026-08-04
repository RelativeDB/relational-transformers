.PHONY: test coverage lint docs docs-parity

test:
	python -m pytest

coverage:
	python -m pytest --cov --cov-report=term-missing

lint:
	python -m ruff check relational_transformers tests

docs:
	python -m sphinx -W -c docs -b html . docs/_build/html

docs-parity:
	python scripts/compare_docs_headers.py
