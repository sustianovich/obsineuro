"""Ollama simulado para pruebas de integración sin modelos reales.

Genera embeddings deterministas por bolsa de palabras con hashing, de
modo que la similitud coseno mantiene sentido semántico aproximado y las
pruebas del recuperador son reproducibles.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
import zlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

DIMENSION = 512

# Palabras vacías: sin filtrarlas, la similitud la dominan "de", "la" y
# "que", y cualquier pregunta se parece a cualquier documento. El doble
# de pruebas quedaría inservible para medir la abstención.
STOPWORDS = {
    "algo", "ante", "cada", "como", "con", "cual", "cuales", "cuando",
    "cuanto", "desde", "donde", "durante", "entre", "esta", "estan",
    "este", "esto", "estos", "hace", "hasta", "para", "pero", "por",
    "porque", "que", "quien", "segun", "ser", "sin", "sobre", "son",
    "sus", "tiene", "todo", "todos", "una", "unos", "hay", "del",
    "los", "las", "sus", "mas", "muy", "ese", "esa", "esos", "esas",
    "debe", "deben", "puede", "pueden", "ocurre", "documento", "titulo",
    "seccion", "search", "query",
}

TASK_PREFIXES = ("search_document:", "search_query:", "passage:", "query:")


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def tokenize(text: str) -> list[str]:
    """Palabras de contenido, sin acentos ni prefijos de tarea."""
    normalized = strip_accents(text)
    for prefix in TASK_PREFIXES:
        normalized = normalized.replace(prefix, " ")
    return [
        token
        for token in re.findall(r"[a-z0-9_-]+", normalized)
        if len(token) >= 4 and token not in STOPWORDS
    ]


def embed(text: str) -> list[float]:
    """Bolsa de palabras con hashing estable y TF sublineal.

    No pretende ser semántico: pretende ser *discriminante*, para que la
    evaluación mida el comportamiento de la tubería y no el ruido del
    doble de pruebas.
    """
    counts: dict[int, float] = {}
    for token in tokenize(text):
        seed = zlib.crc32(token.encode("utf-8"))
        counts[seed % DIMENSION] = counts.get(seed % DIMENSION, 0.0) + 1.0
        # Segundo cubo por prefijo: acerca "lectura" y "lecturas".
        stem = token[:5]
        seed_stem = zlib.crc32(stem.encode("utf-8"))
        key = seed_stem % DIMENSION
        counts[key] = counts.get(key, 0.0) + 0.6

    vector = [0.0] * DIMENSION
    for index, count in counts.items():
        vector[index] = 1.0 + math.log(count)

    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silencio en las pruebas
        return

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.0.0-fake"})
        elif self.path == "/api/tags":
            self._send(
                {
                    "models": [
                        {"name": "qwen3.5:0.8b"},
                        {"name": "nomic-embed-text:latest"},
                    ]
                }
            )
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/api/embed":
            texts = body.get("input") or []
            if isinstance(texts, str):
                texts = [texts]
            self._send({"embeddings": [embed(text) for text in texts]})
            return

        if self.path == "/api/chat":
            messages = body.get("messages") or []
            system = ""
            user = ""
            for message in messages:
                if message.get("role") == "system":
                    system = str(message.get("content", ""))
                elif message.get("role") == "user":
                    user = str(message.get("content", ""))

            # Simula el reordenador: puntúa por solapamiento real de
            # palabras de contenido entre pregunta y fragmento.
            if "evaluador de relevancia" in system:
                self._send(
                    {
                        "message": {
                            "role": "assistant",
                            "content": _rerank_scores(user),
                        },
                        "done": True,
                        "prompt_eval_count": 400,
                        "eval_count": 30,
                    }
                )
                return

            reply = (
                "<think>razonamiento interno que no debe salir</think>"
                "Según la documentación recuperada [1], el procedimiento "
                "consta de tres fases. La segunda fase requiere validación "
                "documental [2]."
            )
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                step = 12
                for start in range(0, len(reply), step):
                    piece = reply[start : start + step]
                    line = (
                        json.dumps(
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": piece,
                                },
                                "done": False,
                            }
                        )
                        + "\n"
                    )
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                final = (
                    json.dumps(
                        {
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "prompt_eval_count": 900,
                            "eval_count": 60,
                        }
                    )
                    + "\n"
                )
                self.wfile.write(final.encode("utf-8"))
                self.wfile.flush()
                return
            self._send(
                {
                    "message": {"role": "assistant", "content": reply},
                    "done": True,
                    "prompt_eval_count": 900,
                    "eval_count": 60,
                }
            )
            return

        self._send({"error": "not found"}, 404)


def _rerank_scores(prompt: str) -> str:
    """Puntuaciones 0-10 por solapamiento de palabras de contenido."""
    pregunta = ""
    if "PREGUNTA" in prompt:
        resto = prompt.split("PREGUNTA", 1)[1]
        pregunta = resto.split("FRAGMENTOS", 1)[0].strip()
    terminos = set(tokenize(pregunta))

    bloques = re.split(r"\n\[(\d+)\]\n", "\n" + prompt.split("FRAGMENTOS", 1)[-1])
    puntuaciones: dict[str, int] = {}
    posicion = 1
    for indice in range(1, len(bloques), 2):
        texto = bloques[indice + 1] if indice + 1 < len(bloques) else ""
        palabras = set(tokenize(texto))
        if terminos:
            cobertura = len(terminos & palabras) / len(terminos)
        else:
            cobertura = 0.0
        puntuaciones[str(posicion)] = round(cobertura * 10)
        posicion += 1
    return json.dumps(puntuaciones)


def start(port: int = 11434) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
