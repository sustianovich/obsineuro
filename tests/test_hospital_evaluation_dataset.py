from __future__ import annotations

from pathlib import Path

from app.rag.evaluation import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TYPO_DATASET = ROOT / "evaluations" / "hospital_typos_questions.json"
VAULT = ROOT / "vault-hospital-ejemplo"


def test_hospital_dataset_only_references_existing_notes():
    cases = load_dataset(TYPO_DATASET)
    referenced_paths = {
        path
        for case in cases
        for path in (*case.expected_paths, *case.forbidden_paths)
    }
    vault_paths = {
        path.relative_to(VAULT).as_posix().casefold()
        for path in VAULT.rglob("*.md")
    }

    missing = sorted(path for path in referenced_paths if path not in vault_paths)
    assert missing == []


def test_hospital_typo_dataset_exercises_semantic_retrieval():
    cases = load_dataset(TYPO_DATASET)

    assert len(cases) == 8
    assert all(case.query_type == "factual" for case in cases)
    assert all(case.requires_semantic for case in cases)
    assert all(case.expected_terms for case in cases)
