from __future__ import annotations

import json
import os
from pathlib import Path
import re
from threading import Lock

from app.config import BASE_DIR


ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_env_lock = Lock()


def update_env_value(
    key: str,
    value: str,
    *,
    env_path: Path | None = None,
    quote_value: bool = True,
) -> None:
    """Actualiza una variable de .env de forma atómica y sin duplicados."""
    if not ENV_KEY_RE.fullmatch(key):
        raise ValueError(f"Nombre de variable .env no válido: {key}")

    target = env_path or BASE_DIR / ".env"
    assignment_re = re.compile(
        rf"^\s*{re.escape(key)}\s*=",
        flags=re.IGNORECASE,
    )
    encoded_value = (
        json.dumps(value, ensure_ascii=False) if quote_value else value
    )
    assignment = f"{key}={encoded_value}"

    with _env_lock:
        lines = (
            target.read_text(encoding="utf-8-sig").splitlines()
            if target.exists()
            else []
        )
        output: list[str] = []
        replaced = False
        for line in lines:
            if assignment_re.match(line):
                if not replaced:
                    output.append(assignment)
                    replaced = True
                continue
            output.append(line)
        if not replaced:
            if output and output[-1]:
                output.append("")
            output.append(assignment)

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f"{target.name}.tmp"
        temporary.write_text(
            "\n".join(output) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
