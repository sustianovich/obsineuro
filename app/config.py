from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DEFAULT_RERANK_ONNX_DIR = BASE_DIR / "models" / "bge-reranker-v2-m3-onnx"
RERANK_ENABLED_MODELS = frozenset({"qwen3.5:2b", "qwen3.5:4b"})
LEGACY_DATABASE_PATH = BASE_DIR / "data" / "rag_index.sqlite3"


def resolve_config_path(value: str, default: Path) -> Path:
    raw = value.strip() if value else ""
    candidate = Path(raw).expanduser() if raw else default
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


def slugify_vault_path(vault_path: Path) -> str:
    """Identificador estable y legible para la base de datos del vault."""
    resolved = vault_path.resolve()
    name = re.sub(r"[^a-z0-9]+", "-", resolved.name.lower()).strip("-")
    digest = hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()
    return f"{name or 'vault'}-{digest[:10]}"


def default_database_path_for_vault(vault_path: Path) -> Path:
    slug = slugify_vault_path(vault_path)
    return BASE_DIR / "data" / "vaults" / slug / "rag_index.sqlite3"


def migrate_legacy_database(target: Path) -> None:
    """Traslada la base de datos global antigua al nuevo layout por vault.

    Se ejecuta una sola vez: sólo cuando el destino derivado para el vault
    activo aún no existe y queda una base global previa a la introducción
    del aislamiento por vault.
    """
    if target.exists() or not LEGACY_DATABASE_PATH.exists():
        return
    if LEGACY_DATABASE_PATH.resolve() == target.resolve():
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            legacy_file = LEGACY_DATABASE_PATH.with_name(
                LEGACY_DATABASE_PATH.name + suffix
            )
            if legacy_file.exists():
                legacy_file.replace(target.with_name(target.name + suffix))
    except OSError:
        # La BD legacy puede estar bloqueada (proceso en marcha, backup,
        # antivirus). No debe impedir el arranque; se reintentará en el
        # próximo inicio mientras el destino no exista.
        logger.warning(
            "No se pudo migrar la base de datos legacy %s a %s; "
            "se reintentará en el próximo arranque.",
            LEGACY_DATABASE_PATH,
            target,
            exc_info=True,
        )


def display_configured_path(path: Path) -> str:
    """Devuelve una ruta legible relativa al directorio padre del proyecto."""
    try:
        return str(path.resolve().relative_to(BASE_DIR.parent.resolve()))
    except ValueError:
        return str(path)


def normalize_local_ollama_url(value: str) -> str:
    raw = (value or "http://127.0.0.1:11434").strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "OLLAMA_BASE_URL debe ser una URL HTTP(S) local sin ruta, "
            "por ejemplo http://127.0.0.1:11434."
        )

    hostname = parsed.hostname.casefold()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError(
            "OLLAMA_BASE_URL sólo puede apuntar a localhost o a una "
            "dirección IP de loopback para impedir el envío de datos fuera "
            "del equipo."
        )

    netloc = parsed.netloc
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def validate_local_model_name(value: str, variable: str) -> str:
    model = value.strip()
    if not model:
        raise ValueError(f"{variable} no puede estar vacío.")
    if "cloud" in model.casefold():
        raise ValueError(
            f"{variable} no puede seleccionar un modelo cloud. "
            "Esta aplicación sólo admite modelos instalados localmente."
        )
    return model


def default_rerank_enabled(chat_model: str) -> bool:
    return chat_model.casefold() in RERANK_ENABLED_MODELS


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on", "sí", "si"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} debe ser true/false, yes/no, on/off o 1/0."
    )


@dataclass
class Settings:
    base_dir: Path
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    ollama_timeout_seconds: float
    vault_path: Path
    database_path: Path
    database_path_explicit: bool
    top_k: int
    min_similarity: float
    chunk_size: int
    chunk_overlap: int
    parent_child_chunking_enabled: bool
    parent_chunk_size: int
    child_chunk_size: int
    child_chunk_overlap: int
    max_output_tokens: int
    memory_summary_interval: int
    memory_recent_turns: int
    memory_max_context_chars: int
    memory_summary_max_input_chars: int
    memory_summary_max_tokens: int
    hybrid_search_enabled: bool
    hybrid_rrf_k: int
    hybrid_semantic_weight: float
    hybrid_lexical_weight: float
    hybrid_candidate_multiplier: int
    graph_search_enabled: bool
    hybrid_graph_weight: float
    graph_max_hops: int
    graph_decay: float
    graph_backlink_weight: float
    graph_seed_documents: int
    graph_max_candidates: int
    default_verifier_context_tokens: int
    default_writer_context_tokens: int
    verifier_max_output_tokens: int
    project_memory_max_context_chars: int
    project_memory_summary_interval: int
    project_memory_summary_max_input_chars: int
    project_memory_summary_max_tokens: int
    embedding_prefix_scheme: str
    mmr_enabled: bool
    mmr_lambda: float
    max_chunks_per_document: int
    ollama_keep_alive: str
    embedding_workers: int
    stream_enabled: bool
    min_relative_score: float
    rerank_enabled: bool
    rerank_backend: str
    rerank_model: str
    rerank_onnx_model_dir: Path
    rerank_onnx_max_tokens: int
    rerank_candidates: int
    rerank_weight: float
    rerank_max_passage_chars: int
    rerank_batch_size: int
    query_routing_enabled: bool
    query_router_relational_patterns: tuple[str, ...]
    query_router_structural_nouns: tuple[str, ...]
    query_router_min_entity_mentions: int
    query_router_relational_graph_weight: float
    query_router_hybrid_graph_weight: float
    query_router_weak_evidence_margin: float
    posterior_abstention_enabled: bool
    posterior_abstention_threshold: float
    verifier_abstain_on_insufficient: bool


def env_str_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Lee una lista separada por comas; conserva el orden y sin vacíos."""
    value = os.getenv(name)
    if value is None:
        return default
    items = [piece.strip() for piece in value.split(",")]
    return tuple(dict.fromkeys(piece for piece in items if piece))


DEFAULT_RELATIONAL_PATTERNS = (
    "se relaciona",
    "relacion entre",
    "relacion con",
    "como afecta",
    "como impacta",
    "impacto en",
    "impacto de",
    "depende de",
    "dependen de",
    "consecuencia de",
    "vinculad",
    "conecta con",
    "conexion entre",
    "flujo entre",
    "cadena de",
    "quien depende",
    "que relacion hay",
)

DEFAULT_STRUCTURAL_NOUNS = (
    "nodo",
    "proceso",
    "riesgo",
    "rol",
    "indicador",
    "kpi",
    "hallazgo",
    "documento",
    "nota",
    "circuito",
    "fase",
    "etapa",
)


configured_chat_model = validate_local_model_name(
    os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:0.8b"),
    "OLLAMA_CHAT_MODEL",
)

configured_parent_chunk_size = max(
    1000,
    min(20000, int(os.getenv("RAG_PARENT_CHUNK_SIZE", "6000"))),
)
configured_child_chunk_size = min(
    configured_parent_chunk_size,
    max(
        200,
        min(4000, int(os.getenv("RAG_CHILD_CHUNK_SIZE", "700"))),
    ),
)
configured_child_chunk_overlap = min(
    configured_child_chunk_size // 2,
    max(0, int(os.getenv("RAG_CHILD_CHUNK_OVERLAP", "100"))),
)


_configured_vault_path = resolve_config_path(
    os.getenv("OBSIDIAN_VAULT_PATH", ""),
    BASE_DIR / "vault_demo",
)
_database_path_explicit = bool(os.getenv("RAG_DATABASE_PATH", "").strip())
if _database_path_explicit:
    _configured_database_path = resolve_config_path(
        os.getenv("RAG_DATABASE_PATH", ""),
        BASE_DIR / "data" / "rag_index.sqlite3",
    )
else:
    _configured_database_path = default_database_path_for_vault(
        _configured_vault_path
    )
    migrate_legacy_database(_configured_database_path)

settings = Settings(
    base_dir=BASE_DIR,
    ollama_base_url=normalize_local_ollama_url(
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ),
    chat_model=configured_chat_model,
    embedding_model=validate_local_model_name(
        os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        "OLLAMA_EMBEDDING_MODEL",
    ),
    ollama_timeout_seconds=max(
        10.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    ),
    vault_path=_configured_vault_path,
    database_path=_configured_database_path,
    database_path_explicit=_database_path_explicit,
    top_k=max(1, int(os.getenv("RAG_TOP_K", "6"))),
    min_similarity=min(
        1.0,
        max(0.0, float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))),
    ),
    chunk_size=max(500, int(os.getenv("RAG_CHUNK_SIZE", "1800"))),
    chunk_overlap=max(0, int(os.getenv("RAG_CHUNK_OVERLAP", "250"))),
    parent_child_chunking_enabled=env_bool(
        "RAG_PARENT_CHILD_CHUNKING", False
    ),
    parent_chunk_size=configured_parent_chunk_size,
    child_chunk_size=configured_child_chunk_size,
    child_chunk_overlap=configured_child_chunk_overlap,
    max_output_tokens=max(
        300, int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1800"))
    ),
    memory_summary_interval=min(
        50,
        max(2, int(os.getenv("RAG_MEMORY_SUMMARY_INTERVAL", "10"))),
    ),
    memory_recent_turns=min(
        10,
        max(1, int(os.getenv("RAG_MEMORY_RECENT_TURNS", "4"))),
    ),
    memory_max_context_chars=min(
        50000,
        max(2000, int(os.getenv("RAG_MEMORY_MAX_CONTEXT_CHARS", "12000"))),
    ),
    memory_summary_max_input_chars=min(
        100000,
        max(
            10000,
            int(
                os.getenv(
                    "RAG_MEMORY_SUMMARY_MAX_INPUT_CHARS",
                    "40000",
                )
            ),
        ),
    ),
    memory_summary_max_tokens=min(
        1800,
        max(
            200,
            int(os.getenv("RAG_MEMORY_SUMMARY_MAX_TOKENS", "700")),
        ),
    ),
    hybrid_search_enabled=env_bool("RAG_HYBRID_SEARCH", True),
    hybrid_rrf_k=min(
        200,
        max(1, int(os.getenv("RAG_HYBRID_RRF_K", "60"))),
    ),
    hybrid_semantic_weight=max(
        0.0,
        float(os.getenv("RAG_HYBRID_SEMANTIC_WEIGHT", "1.0")),
    ),
    hybrid_lexical_weight=max(
        0.0,
        float(os.getenv("RAG_HYBRID_LEXICAL_WEIGHT", "0.8")),
    ),
    hybrid_candidate_multiplier=min(
        20,
        max(
            2,
            int(os.getenv("RAG_HYBRID_CANDIDATE_MULTIPLIER", "4")),
        ),
    ),
    graph_search_enabled=env_bool("RAG_GRAPH_SEARCH", False),
    hybrid_graph_weight=max(
        0.0,
        float(os.getenv("RAG_GRAPH_WEIGHT", "0.5")),
    ),
    graph_max_hops=min(
        5,
        max(1, int(os.getenv("RAG_GRAPH_MAX_HOPS", "2"))),
    ),
    graph_decay=min(
        1.0,
        max(0.0, float(os.getenv("RAG_GRAPH_DECAY", "0.5"))),
    ),
    graph_backlink_weight=min(
        1.0,
        max(0.0, float(os.getenv("RAG_GRAPH_BACKLINK_WEIGHT", "0.7"))),
    ),
    graph_seed_documents=min(
        50,
        max(1, int(os.getenv("RAG_GRAPH_SEED_DOCUMENTS", "4"))),
    ),
    graph_max_candidates=min(
        100,
        max(1, int(os.getenv("RAG_GRAPH_MAX_CANDIDATES", "20"))),
    ),
    default_verifier_context_tokens=min(
        262144,
        max(
            4096,
            int(os.getenv("RAG_VERIFIER_CONTEXT_TOKENS", "8192")),
        ),
    ),
    default_writer_context_tokens=min(
        262144,
        max(
            4096,
            int(os.getenv("RAG_WRITER_CONTEXT_TOKENS", "16384")),
        ),
    ),
    verifier_max_output_tokens=min(
        2000,
        max(
            200,
            int(os.getenv("RAG_VERIFIER_MAX_OUTPUT_TOKENS", "600")),
        ),
    ),
    project_memory_max_context_chars=min(
        30000,
        max(
            1000,
            int(
                os.getenv(
                    "RAG_PROJECT_MEMORY_MAX_CONTEXT_CHARS",
                    "6000",
                )
            ),
        ),
    ),
    project_memory_summary_interval=min(
        50,
        max(
            2,
            int(
                os.getenv(
                    "RAG_PROJECT_MEMORY_SUMMARY_INTERVAL",
                    "10",
                )
            ),
        ),
    ),
    project_memory_summary_max_input_chars=min(
        100000,
        max(
            10000,
            int(
                os.getenv(
                    "RAG_PROJECT_MEMORY_SUMMARY_MAX_INPUT_CHARS",
                    "40000",
                )
            ),
        ),
    ),
    project_memory_summary_max_tokens=min(
        1800,
        max(
            200,
            int(
                os.getenv(
                    "RAG_PROJECT_MEMORY_SUMMARY_MAX_TOKENS",
                    "700",
                )
            ),
        ),
    ),
    embedding_prefix_scheme=os.getenv(
        "RAG_EMBEDDING_PREFIX_SCHEME", "auto"
    ).strip()
    or "auto",
    mmr_enabled=env_bool("RAG_MMR", True),
    mmr_lambda=min(
        1.0,
        max(0.0, float(os.getenv("RAG_MMR_LAMBDA", "0.72"))),
    ),
    max_chunks_per_document=min(
        10,
        max(1, int(os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "3"))),
    ),
    ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
    or "30m",
    embedding_workers=min(
        8,
        max(1, int(os.getenv("RAG_EMBEDDING_WORKERS", "3"))),
    ),
    stream_enabled=env_bool("RAG_STREAMING", True),
    # Corte relativo al mejor candidato. Es mucho más robusto entre
    # modelos que un umbral absoluto: con nomic-embed-text las
    # similitudes absolutas rara vez bajan de 0,3 ni siquiera para
    # documentos irrelevantes, así que RAG_MIN_SIMILARITY por sí solo
    # casi nunca llega a abstenerse.
    min_relative_score=min(
        0.99,
        max(0.0, float(os.getenv("RAG_MIN_RELATIVE_SCORE", "0.62"))),
    ),
    # Equilibrado y Calidad habilitan la pasada ONNX local por defecto.
    rerank_enabled=env_bool(
        "RAG_RERANK",
        default_rerank_enabled(configured_chat_model),
    ),
    rerank_backend=os.getenv("RAG_RERANK_BACKEND", "onnx").strip().casefold()
    or "onnx",
    rerank_model=os.getenv(
        "RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
    ).strip(),
    rerank_onnx_model_dir=resolve_config_path(
        os.getenv("RAG_RERANK_ONNX_MODEL_DIR", ""),
        DEFAULT_RERANK_ONNX_DIR,
    ),
    rerank_onnx_max_tokens=min(
        2048,
        max(64, int(os.getenv("RAG_RERANK_ONNX_MAX_TOKENS", "512"))),
    ),
    rerank_candidates=min(
        40, max(2, int(os.getenv("RAG_RERANK_CANDIDATES", "12")))
    ),
    rerank_weight=min(
        1.0, max(0.0, float(os.getenv("RAG_RERANK_WEIGHT", "0.7")))
    ),
    rerank_max_passage_chars=min(
        4000, max(200, int(os.getenv("RAG_RERANK_MAX_PASSAGE_CHARS", "900")))
    ),
    rerank_batch_size=min(
        16, max(1, int(os.getenv("RAG_RERANK_BATCH_SIZE", "6")))
    ),
    # Enrutador determinista (sin LLM): decide por consulta si conviene
    # activar el grafo y con qué peso. Desactivado por defecto hasta que
    # un conjunto dorado propio confirme que no perjudica lo factual.
    query_routing_enabled=env_bool("RAG_QUERY_ROUTING", False),
    query_router_relational_patterns=env_str_list(
        "RAG_QUERY_ROUTER_RELATIONAL_PATTERNS",
        DEFAULT_RELATIONAL_PATTERNS,
    ),
    query_router_structural_nouns=env_str_list(
        "RAG_QUERY_ROUTER_STRUCTURAL_NOUNS",
        DEFAULT_STRUCTURAL_NOUNS,
    ),
    query_router_min_entity_mentions=max(
        1,
        int(os.getenv("RAG_QUERY_ROUTER_MIN_ENTITY_MENTIONS", "2")),
    ),
    query_router_relational_graph_weight=max(
        0.0,
        float(
            os.getenv("RAG_QUERY_ROUTER_RELATIONAL_GRAPH_WEIGHT", "0.5")
        ),
    ),
    query_router_hybrid_graph_weight=max(
        0.0,
        float(os.getenv("RAG_QUERY_ROUTER_HYBRID_GRAPH_WEIGHT", "0.25")),
    ),
    query_router_weak_evidence_margin=max(
        0.0,
        float(
            os.getenv("RAG_QUERY_ROUTER_WEAK_EVIDENCE_MARGIN", "0.05")
        ),
    ),
    # Abstención posterior a la fusión: infraestructura lista pero apagada
    # por defecto porque los pesos no están calibrados con evaluación real
    # todavía (ver scripts/calibrate_threshold.py).
    posterior_abstention_enabled=env_bool("RAG_POSTERIOR_ABSTENTION", False),
    posterior_abstention_threshold=min(
        1.0,
        max(
            0.0,
            float(os.getenv("RAG_POSTERIOR_ABSTENTION_THRESHOLD", "0.35")),
        ),
    ),
    verifier_abstain_on_insufficient=env_bool(
        "RAG_VERIFIER_ABSTAIN_ON_INSUFFICIENT", False
    ),
)
