from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatModelProfileRequest(BaseModel):
    profile: str = Field(min_length=1, max_length=30)


class ProjectNameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ProjectAgentSettingsRequest(BaseModel):
    verification_enabled: bool = True
    project_memory_enabled: bool = True
    verifier_context_tokens: int = Field(
        default=8192,
        ge=4096,
        le=262144,
    )
    writer_context_tokens: int = Field(
        default=16384,
        ge=4096,
        le=262144,
    )


class ConversationMemoryRequest(BaseModel):
    enabled: bool


VIGENCIA_VALUES = {
    "",
    "vigente",
    "futura",
    "caducada",
    "desconocida",
    "no_caducada",
}


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=15)
    status: str | None = Field(default=None, max_length=50)
    expand_links: bool = True
    vigencia: str | None = Field(default=None, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=10)
    iteration: int = Field(default=0, ge=0, le=10)
    iteration_feedback: str = Field(default="", max_length=500)

    @field_validator("vigencia")
    @classmethod
    def _check_vigencia(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().casefold()
        if normalized not in VIGENCIA_VALUES:
            admitidos = ", ".join(sorted(v for v in VIGENCIA_VALUES if v))
            raise ValueError(
                f"vigencia debe ser uno de: {admitidos}."
            )
        return normalized or None
    conversation_id: str | None = Field(default=None, max_length=64)
    project_id: str | None = Field(default=None, max_length=64)
    use_memory: bool = True


class SourceItem(BaseModel):
    reference: str
    title: str
    path: str
    heading: str
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    reason: str
    metadata: dict
    content: str
    matched_content: str | None = None
    context_expanded: bool = False
    matched_chunk_ids: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    vigencia: dict = Field(default_factory=dict)


class MemoryState(BaseModel):
    enabled: bool = True
    used_context: bool = False
    has_summary: bool = False
    summary_updated: bool = False
    summarized_turns: int = 0
    pending_turns: int = 0
    total_turns: int = 0
    summary_model: str = ""
    last_error: str | None = None
    warning: str | None = None
    updated_at: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    conversation_id: str
    project_id: str
    turn_id: int
    chat_model: str
    memory: MemoryState
    project_memory: dict = Field(default_factory=dict)
    agents: dict = Field(default_factory=dict)
    citations: dict = Field(default_factory=dict)


class IndexResponse(BaseModel):
    scanned: int
    indexed: int
    unchanged: int
    deleted: int
    chunks_created: int
    errors: list[str]
    rebuilt: bool
    rebuild_reasons: list[str]
    stats: dict[str, int]
