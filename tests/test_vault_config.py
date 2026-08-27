from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    default_database_path_for_vault,
    migrate_legacy_database,
    settings,
    slugify_vault_path,
)
from app.db import create_project, list_projects
from app.vault_config import configure_vault_path


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # default_database_path_for_vault() anchors new databases under
    # BASE_DIR/data/vaults/<slug>/; redirect it into tmp_path so these
    # tests don't write into the real project's data/ directory.
    monkeypatch.setattr("app.config.BASE_DIR", tmp_path)
    original_vault_path = settings.vault_path
    original_database_path = settings.database_path
    original_database_path_explicit = settings.database_path_explicit
    settings.database_path_explicit = False
    try:
        yield tmp_path
    finally:
        settings.vault_path = original_vault_path
        settings.database_path = original_database_path
        settings.database_path_explicit = original_database_path_explicit


def test_slugify_vault_path_is_stable_and_distinct(tmp_path: Path):
    vault_a = tmp_path / "Mi Vault"
    vault_b = tmp_path / "otro" / "Mi Vault"
    vault_a.mkdir()
    vault_b.mkdir(parents=True)

    slug_a = slugify_vault_path(vault_a)
    slug_b = slugify_vault_path(vault_b)

    assert slug_a == slugify_vault_path(vault_a)
    assert slug_a != slug_b
    assert slug_a.startswith("mi-vault-")


def test_default_database_path_for_vault_is_per_vault(tmp_path: Path):
    vault_a = tmp_path / "vault_a"
    vault_b = tmp_path / "vault_b"
    vault_a.mkdir()
    vault_b.mkdir()

    assert default_database_path_for_vault(
        vault_a
    ) != default_database_path_for_vault(vault_b)


def test_configure_vault_path_isolates_projects(isolated_settings: Path):
    tmp_path = isolated_settings
    vault_a = tmp_path / "vault_a"
    vault_b = tmp_path / "vault_b"
    vault_a.mkdir()
    vault_b.mkdir()
    env_path = tmp_path / ".env"

    configure_vault_path(vault_a, env_path=env_path)
    project = create_project("Proyecto del vault A")
    assert any(p["id"] == project["id"] for p in list_projects())

    configure_vault_path(vault_b, env_path=env_path)
    assert not any(p["id"] == project["id"] for p in list_projects())

    configure_vault_path(vault_a, env_path=env_path)
    assert any(p["id"] == project["id"] for p in list_projects())


def test_configure_vault_path_respects_explicit_database_override(
    isolated_settings: Path,
):
    tmp_path = isolated_settings
    settings.database_path_explicit = True
    fixed_database_path = tmp_path / "shared.sqlite3"
    settings.database_path = fixed_database_path

    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    env_path = tmp_path / ".env"

    configure_vault_path(vault_a, env_path=env_path)

    assert settings.database_path == fixed_database_path


def test_migrate_legacy_database_moves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    legacy_path = tmp_path / "legacy" / "rag_index.sqlite3"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy-data")
    monkeypatch.setattr("app.config.LEGACY_DATABASE_PATH", legacy_path)

    target = tmp_path / "vaults" / "some-vault" / "rag_index.sqlite3"
    migrate_legacy_database(target)

    assert target.read_bytes() == b"legacy-data"
    assert not legacy_path.exists()


def test_migrate_legacy_database_skips_when_target_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    legacy_path = tmp_path / "legacy" / "rag_index.sqlite3"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy-data")
    monkeypatch.setattr("app.config.LEGACY_DATABASE_PATH", legacy_path)

    target = tmp_path / "vaults" / "some-vault" / "rag_index.sqlite3"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already-there")

    migrate_legacy_database(target)

    assert target.read_bytes() == b"already-there"
    assert legacy_path.exists()


def test_migrate_legacy_database_noop_when_legacy_missing(tmp_path: Path):
    target = tmp_path / "vaults" / "some-vault" / "rag_index.sqlite3"
    migrate_legacy_database(target)
    assert not target.exists()
