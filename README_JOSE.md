# RAG local para Obsidian

## ¿Qué hace?

Esta aplicación permite hacer preguntas sobre las notas Markdown de un vault de Obsidian.

Indexa las notas, busca los fragmentos relacionados con cada pregunta y genera una respuesta en español con referencias a los archivos utilizados.

Todo funciona de forma local:

- Ollama genera los embeddings y las respuestas.
- SQLite guarda el índice y las conversaciones.
- No usa OpenAI, claves API ni servicios externos de inteligencia artificial.

## ¿Qué necesitas?

- Windows 10 u 11.
- Python 3.11 o 3.12.
- Ollama en ejecución.
- Los modelos `qwen3.5:0.8b` y `nomic-embed-text` instalados en Ollama.
- Un vault de Obsidian con archivos `.md`.
- Las dependencias de Python ya instaladas en `.venv`.

El arranque no ejecuta `pip install` ni intenta conectarse a PyPI.

## ¿Cómo se inicia?

1. Abre Ollama.
2. Comprueba que `OBSIDIAN_VAULT_PATH` en `.env` apunta a tu vault.
3. Haz doble clic en `iniciar_windows.bat`.
4. La aplicación se abrirá en <http://127.0.0.1:8000>.

## ¿Cómo se usa?

1. Pulsa **Indexar documentos**.
2. Espera a que finalice la indexación.
3. Escribe una pregunta sobre tus notas.
4. Pulsa **Preguntar** y consulta la respuesta y sus fuentes.

Los documentos, el índice y las conversaciones permanecen en el equipo.
