from __future__ import annotations

import pytest

from app.config import settings
from app.model_profiles import (
    CHAT_MODEL_PROFILES,
    DEFAULT_CONTEXT_WINDOW,
    active_profile,
    active_think_value,
    configure_chat_model_profile,
    context_window_from_model_info,
    effective_context_window,
    list_chat_model_profiles,
    resolve_model_context,
)
from app.rag import ollama_client


@pytest.fixture()
def modelo_restaurable():
    previo = (
        settings.chat_model,
        settings.rerank_enabled,
        settings.rerank_backend,
    )
    yield
    (
        settings.chat_model,
        settings.rerank_enabled,
        settings.rerank_backend,
    ) = previo


@pytest.fixture(autouse=True)
def sin_metadatos_ollama(monkeypatch):
    """Por defecto, como si Ollama no aportara metadatos de contexto.

    Evita depender de si hay un Ollama real escuchando en la máquina donde
    corren las pruebas. Los tests que quieran simular metadatos reales
    sobrescriben `get_model_show` explícitamente.
    """
    monkeypatch.setattr(ollama_client, "get_model_show", lambda model: None)


# ----------------------------------------------------------------------
# Catálogo
# ----------------------------------------------------------------------
def test_estan_los_perfiles_nuevos():
    assert "llama3" in CHAT_MODEL_PROFILES
    assert "gpt_oss" in CHAT_MODEL_PROFILES
    assert CHAT_MODEL_PROFILES["llama3"].model == "llama3:latest"
    assert CHAT_MODEL_PROFILES["gpt_oss"].model == "gpt-oss:latest"


def test_qwen3_14b_es_perfil_de_razonamiento_booleano():
    profile = CHAT_MODEL_PROFILES["qwen3_14b"]
    assert profile.model == "qwen3:14b"
    assert profile.reasoning is True
    # A diferencia de gpt-oss (nivel de esfuerzo), Qwen3 activa el
    # razonamiento con un booleano en la API de Ollama.
    assert profile.think is True
    assert profile.context_window >= 4096


def test_todos_los_perfiles_declaran_su_ventana():
    for profile in CHAT_MODEL_PROFILES.values():
        assert profile.context_window >= 4096
        assert isinstance(profile.reasoning, bool)


def test_llama3_declara_su_contexto_corto():
    # Llama 3 se entrenó con 8K. Anotar más aquí haría que el redactor
    # pidiera un contexto que el modelo no puede sostener.
    assert CHAT_MODEL_PROFILES["llama3"].context_window == 8192


def test_gpt_oss_es_modelo_de_razonamiento():
    profile = CHAT_MODEL_PROFILES["gpt_oss"]
    assert profile.reasoning is True
    assert profile.context_window == 131072
    # gpt-oss razona por diseño: se pide el esfuerzo mínimo, no `False`.
    assert profile.think == "low"


def test_el_listado_expone_los_campos_nuevos():
    campos = set(list_chat_model_profiles()[0])
    assert {
        "context_window",
        "reasoning",
        "think",
        "rerank_enabled",
        "rerank_backend",
    }.issubset(campos)


def test_equilibrado_y_calidad_activan_el_reranker_onnx():
    for profile_id in ("balanced", "quality"):
        profile = CHAT_MODEL_PROFILES[profile_id]
        assert profile.rerank_enabled is True
        assert profile.rerank_backend == "onnx"


def test_ligero_no_activa_el_reranker():
    profile = CHAT_MODEL_PROFILES["light"]
    assert profile.rerank_enabled is False


def test_mayusculas_del_modelo_no_impiden_reconocerlo(modelo_restaurable):
    settings.chat_model = "Llama3:Latest"
    profile = active_profile()
    assert profile is not None
    assert profile.context_window == 8192


# ----------------------------------------------------------------------
# Parámetro think
# ----------------------------------------------------------------------
def test_think_por_perfil(modelo_restaurable):
    settings.chat_model = "gpt-oss:latest"
    assert active_think_value() == "low"
    settings.chat_model = "llama3:latest"
    assert active_think_value() is False
    settings.chat_model = "modelo-raro"
    assert active_think_value() is False


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
def test_configurar_perfil_nuevo_escribe_el_modelo(
    tmp_path, modelo_restaurable
):
    env = tmp_path / ".env"
    env.write_text("OLLAMA_CHAT_MODEL=qwen3.5:0.8b\n", encoding="utf-8")
    resultado = configure_chat_model_profile("gpt_oss", env_path=env)
    assert resultado["model"] == "gpt-oss:latest"
    assert settings.chat_model == "gpt-oss:latest"
    content = env.read_text(encoding="utf-8")
    assert "gpt-oss:latest" in content
    assert "RAG_RERANK=false" in content
    assert "RAG_RERANK_BACKEND=onnx" in content


def test_configurar_equilibrado_activa_onnx(tmp_path, modelo_restaurable):
    env = tmp_path / ".env"
    configure_chat_model_profile("balanced", env_path=env)

    assert settings.rerank_enabled is True
    assert settings.rerank_backend == "onnx"
    content = env.read_text(encoding="utf-8")
    assert "RAG_RERANK=true" in content
    assert "RAG_RERANK_BACKEND=onnx" in content


def test_perfil_inexistente_enumera_los_disponibles(modelo_restaurable):
    with pytest.raises(ValueError, match="llama3"):
        configure_chat_model_profile("inventado")


# ----------------------------------------------------------------------
# context_window_from_model_info(): lectura segura de metadatos de Ollama
# ----------------------------------------------------------------------
def test_context_window_from_model_info_busca_por_sufijo():
    assert (
        context_window_from_model_info(
            {"general.architecture": "qwen2", "qwen2.context_length": 32768}
        )
        == 32768
    )


def test_context_window_from_model_info_ignora_valores_invalidos():
    assert context_window_from_model_info(None) is None
    assert context_window_from_model_info({}) is None
    assert (
        context_window_from_model_info({"llama.context_length": "no-numero"})
        is None
    )
    # Demasiado bajo para ser una ventana de contexto real: se descarta.
    assert context_window_from_model_info({"llama.context_length": 10}) is None
    # bool es subclase de int en Python; no debe colarse como recuento.
    assert context_window_from_model_info({"llama.context_length": True}) is None


def test_context_window_from_model_info_toma_el_minimo_entre_varias_claves():
    assert (
        context_window_from_model_info(
            {"a.context_length": 8192, "b.context_length": 4096}
        )
        == 4096
    )


# ----------------------------------------------------------------------
# effective_context_window(): min(configurado, capacidad del modelo)
# ----------------------------------------------------------------------
def test_effective_context_window_es_el_minimo():
    assert effective_context_window(16384, 8192) == 8192
    assert effective_context_window(4096, 32768) == 4096
    assert effective_context_window(32768, 32768) == 32768


# ----------------------------------------------------------------------
# resolve_model_context(): orden de resolución perfil / Ollama / fallback
# ----------------------------------------------------------------------
def test_resolve_model_context_para_todos_los_perfiles_conocidos():
    for profile in CHAT_MODEL_PROFILES.values():
        info = resolve_model_context(profile.model)
        assert info.model == profile.model
        assert info.profile_id == profile.id
        assert info.model_context_window == profile.context_window
        assert info.source == "profile"
        assert info.verified is True
        assert info.warning is None


def test_resolve_model_context_modelo_desconocido_usa_fallback_prudente():
    info = resolve_model_context("un-modelo-sin-perfil-ni-metadatos")
    assert info.profile_id is None
    assert info.model_context_window == DEFAULT_CONTEXT_WINDOW
    assert info.source == "fallback"
    assert info.verified is False
    assert info.warning is not None


def test_resolve_model_context_lee_metadatos_simulados_de_ollama(monkeypatch):
    monkeypatch.setattr(
        ollama_client,
        "get_model_show",
        lambda model: {"model_info": {"llama.context_length": 65536}},
    )
    info = resolve_model_context("modelo-personalizado-sin-perfil")
    assert info.model_context_window == 65536
    assert info.source == "ollama"
    assert info.verified is True
    assert info.profile_id is None


def test_resolve_model_context_conflicto_usa_el_valor_conservador(monkeypatch):
    # El perfil de llama3 declara 8192; si los metadatos locales de Ollama
    # informan menos, ese es el límite real y prevalece.
    monkeypatch.setattr(
        ollama_client,
        "get_model_show",
        lambda model: {"model_info": {"llama.context_length": 4096}},
    )
    info = resolve_model_context("llama3:latest")
    assert info.model_context_window == 4096
    assert info.source == "profile+ollama"
    assert info.verified is True
    assert "conservador" in (info.warning or "")


def test_resolve_model_context_coincidencia_no_genera_advertencia(monkeypatch):
    monkeypatch.setattr(
        ollama_client,
        "get_model_show",
        lambda model: {"model_info": {"llama.context_length": 8192}},
    )
    info = resolve_model_context("llama3:latest")
    assert info.model_context_window == 8192
    assert info.source == "profile+ollama"
    assert info.warning is None


def test_resolve_model_context_ollama_inaccesible_no_lanza(monkeypatch):
    def fallo(model):
        raise ollama_client.OllamaConnectionError("Ollama apagado")

    monkeypatch.setattr(ollama_client, "get_model_show", fallo)
    info = resolve_model_context("llama3:latest")
    assert info.model_context_window == 8192
    assert info.source == "profile"

