.PHONY: test lint docs

test:
	python -m pytest

lint:
	python -m ruff check relational_transformers tests

docs:
	python -m sphinx -W -c docs -b html . docs/_build/html
