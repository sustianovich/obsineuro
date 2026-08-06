from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from app.config import BASE_DIR, resolve_config_path, settings
from app.env_config import update_env_value


VAULT_ENV_KEY = "OBSIDIAN_VAULT_PATH"
_config_lock = Lock()


def select_vault_directory(initial_path: Path) -> str:
    if os.name != "nt":
        raise RuntimeError(
            "El selector gráfico de vault está disponible en Windows."
        )

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "Esta instalación de Python no incluye tkinter. "
            "Configura OBSIDIAN_VAULT_PATH manualmente en .env."
        ) from exc

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial_directory = (
            initial_path
            if initial_path.exists() and initial_path.is_dir()
            else BASE_DIR
        )
        return str(
            filedialog.askdirectory(
                parent=root,
                title="Selecciona el vault de Obsidian",
                initialdir=str(initial_directory),
                mustexist=True,
            )
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo abrir el selector de carpetas de Windows. "
            "Configura OBSIDIAN_VAULT_PATH manualmente en .env."
        ) from exc
    finally:
        if root is not None:
            root.destroy()


def configure_vault_path(
    value: str | Path,
    *,
    env_path: Path | None = None,
) -> Path:
    resolved = resolve_config_path(str(value), BASE_DIR / "vault_demo")
    if not resolved.exists():
        raise FileNotFoundError(f"No existe la carpeta seleccionada: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(
            f"La ruta seleccionada no es una carpeta: {resolved}"
        )

    target_env = env_path or BASE_DIR / ".env"
    with _config_lock:
        update_env_value(
            VAULT_ENV_KEY,
            resolved.as_posix(),
            env_path=target_env,
        )
        settings.vault_path = resolved
    return resolved


def choose_and_configure_vault() -> Path | None:
    selected = select_vault_directory(settings.vault_path)
    if not selected:
        return None
    return configure_vault_path(selected)
