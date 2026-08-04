#!/usr/bin/env python3
"""Audit Relational Transformers documentation against Sentence Transformers.

This compares structural product roles, not irrelevant feature names. For
example, Sentence Transformers' embedding-model quickstart maps to relational
prediction, while reranker and sparse-encoder sections map to ablation and
task-head/backends documentation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageComparison:
    reference: str
    target: str
    headings: tuple[tuple[str, str], ...]


COMPARISONS = (
    PageComparison(
        "README.md",
        "README.md",
        (
            ("Installation", "Installation"),
            ("Getting Started", "Getting Started"),
            ("Embedding Models", "Relational Prediction Models"),
            ("Reranker Models", "Ablation"),
            ("Sparse Encoder Models", "Backends"),
            ("Pre-Trained Models", "Pre-Trained Models"),
            ("Training", "Training"),
            ("Companion Blog Posts", "Companion Resources"),
            ("Application Examples", "Application Examples"),
            ("Development setup", "Development setup"),
            ("Citing & Authors", "Citing & Authors"),
            ("Maintainers", "Maintainers"),
        ),
    ),
    PageComparison(
        "docs/installation.md",
        "docs/installation.md",
        (
            ("Installation", "Installation"),
            ("Install with pip", "Install with pip"),
            ("Install from Source", "Install from Source"),
            ("Editable Install", "Editable Install"),
            ("Install PyTorch with CUDA support", "Install PyTorch with CUDA support"),
        ),
    ),
    PageComparison(
        "docs/sentence_transformer/pretrained_models.md",
        "docs/relational_transformer/pretrained_models.md",
        (
            ("Pretrained Models", "Pretrained Models"),
            ("Original Models", "Published Models"),
            ("Semantic Search Models", "Classification and Ranking"),
            ("Multimodal Models", "Regression and Forecasting"),
        ),
    ),
    PageComparison(
        "docs/sentence_transformer/dataset_overview.md",
        "docs/relational_transformer/dataset_overview.md",
        (
            ("Dataset Overview", "Dataset Overview"),
            ("Accepted column types", "Accepted Input Types"),
            ("Datasets on the Hugging Face Hub", "Dataset Construction"),
            ("Pre-existing Datasets", "Pre-existing Datasets"),
        ),
    ),
    PageComparison(
        "docs/sentence_transformer/loss_overview.md",
        "docs/relational_transformer/loss_overview.md",
        (
            ("Loss Overview", "Loss Overview"),
            ("Loss Table", "Loss Table"),
            ("Commonly used Loss Functions", "Classification"),
            ("Custom Loss Functions", "Custom Loss Functions"),
        ),
    ),
    PageComparison(
        "docs/sentence_transformer/training_overview.md",
        "docs/relational_transformer/training_overview.md",
        (
            ("Training Overview", "Training Overview"),
            ("Why Finetune?", "Why Fine-tune?"),
            ("Training Components", "Training Components"),
            ("Model", "Model"),
            ("Dataset", "Dataset"),
            ("Dataset Format", "Dataset Format"),
            ("Loss Function", "Loss Function"),
            ("Training Arguments", "Training Arguments"),
            ("Evaluator", "Evaluator"),
            ("Trainer", "Trainer"),
            ("End-to-End Example", "End-to-End Example"),
        ),
    ),
    PageComparison(
        "docs/package_reference/sentence_transformer/model.md",
        "docs/package_reference/model.md",
        (("SentenceTransformer", "RelationalTransformer"),),
    ),
    PageComparison(
        "docs/package_reference/sentence_transformer/trainer.md",
        "docs/package_reference/training.md",
        (("Trainer", "RelationalTrainer"),),
    ),
    PageComparison(
        "docs/package_reference/sentence_transformer/training_args.md",
        "docs/package_reference/training.md",
        (("Training Arguments", "RelationalTrainingArguments"),),
    ),
    PageComparison(
        "docs/package_reference/sentence_transformer/evaluation.md",
        "docs/package_reference/evaluation.md",
        (
            ("Evaluation", "Evaluation"),
            ("BinaryClassificationEvaluator", "BinaryClassificationEvaluator"),
            ("MSEEvaluator", "RegressionEvaluator"),
            ("RerankingEvaluator", "AblationEvaluator"),
        ),
    ),
    PageComparison(
        "docs/package_reference/sentence_transformer/datasets.md",
        "docs/package_reference/datasets.md",
        (("Datasets", "Datasets"),),
    ),
    PageComparison(
        "docs/package_reference/util/quantization.md",
        "docs/relational_transformer/quantization.md",
        (("quantization", "Quantization"),),
    ),
)


def headings(path: Path) -> list[str]:
    """Extract Markdown ATX and reStructuredText underline headings."""

    lines = path.read_text().splitlines()
    result = []
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if match:
            result.append(match.group(1).strip())
            continue
        if index + 1 < len(lines) and re.fullmatch(r"[=\-~^\"]{3,}", lines[index + 1]):
            if line.strip():
                result.append(line.strip())
    return result


def audit(reference_root: Path, target_root: Path) -> dict:
    pages = []
    missing = []
    for comparison in COMPARISONS:
        reference_path = reference_root / comparison.reference
        target_path = target_root / comparison.target
        if not reference_path.exists() or not target_path.exists():
            absent = reference_path if not reference_path.exists() else target_path
            missing.append(f"missing page: {absent}")
            continue
        reference_headings = headings(reference_path)
        target_headings = headings(target_path)
        checks = []
        for source, destination in comparison.headings:
            source_exists = source in reference_headings
            target_exists = destination in target_headings
            checks.append(
                {
                    "reference": source,
                    "target": destination,
                    "reference_exists": source_exists,
                    "target_exists": target_exists,
                }
            )
            if not source_exists:
                missing.append(f"{comparison.reference}: reference heading {source!r} not found")
            if not target_exists:
                missing.append(f"{comparison.target}: target heading {destination!r} not found")
        pages.append(
            {
                "reference": comparison.reference,
                "target": comparison.target,
                "checks": checks,
            }
        )
    total = sum(len(page["checks"]) for page in pages)
    passed = sum(
        check["reference_exists"] and check["target_exists"]
        for page in pages
        for check in page["checks"]
    )
    return {
        "passed": passed,
        "total": total,
        "coverage": passed / total if total else 0.0,
        "missing": missing,
        "pages": pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path.home() / "sentence-transformers",
        help="Sentence Transformers checkout",
    )
    parser.add_argument("--target", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.reference.expanduser().resolve(), args.target.resolve())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"documentation surface: {result['passed']}/{result['total']} "
            f"({result['coverage']:.0%}) mapped headings present"
        )
        for item in result["missing"]:
            print(f"- {item}")
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
