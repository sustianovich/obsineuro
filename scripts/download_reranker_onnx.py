"""Descarga manualmente el artefacto ONNX usado por el reordenador local."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPOSITORY = "onnx-community/bge-reranker-v2-m3-ONNX"
MODEL_SHA256 = "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d"
REQUIRED_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/model_quantized.onnx",
)
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "bge-reranker-v2-m3-onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instala localmente el reranker BGE ONNX INT8."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Carpeta local que usara RAG_RERANK_ONNX_MODEL_DIR.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    destination = args.destination.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Falta huggingface_hub. Ejecuta: pip install -r requirements.txt")
        return 2

    snapshot_download(
        repo_id=REPOSITORY,
        local_dir=destination,
        allow_patterns=list(REQUIRED_FILES),
    )
    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (destination / relative_path).is_file()
    ]
    if missing:
        print("La descarga no contiene todos los archivos requeridos:")
        for relative_path in missing:
            print(f"- {relative_path}")
        return 1

    model_file = destination / "onnx" / "model_quantized.onnx"
    if sha256_file(model_file) != MODEL_SHA256:
        print("La comprobación SHA-256 del modelo ONNX ha fallado.")
        return 1

    print(f"Reranker ONNX instalado en: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
