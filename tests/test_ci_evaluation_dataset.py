from __future__ import annotations

from pathlib import Path

from app.rag.evaluation import load_dataset
from scripts.build_demo_vault import DOCUMENTS


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluations" / "ci_questions.json"


def test_ci_dataset_covers_core_regression_paths():
    cases = load_dataset(DATASET)

    assert len(cases) >= 10
    assert any(case.expect_abstention for case in cases)
    assert any(case.status for case in cases)
    assert any(case.vigencia for case in cases)
    assert any(case.tags for case in cases)
    assert any(len(case.expected_paths) > 1 for case in cases)


def test_ci_dataset_only_references_demo_documents():
    cases = load_dataset(DATASET)
    referenced_paths = {
        path
        for case in cases
        for path in (*case.expected_paths, *case.forbidden_paths)
    }

    assert referenced_paths <= set(DOCUMENTS)
