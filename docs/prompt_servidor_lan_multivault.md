# Tarea: servidor LAN multi-vault con autenticación por usuario

## Contexto del proyecto

Trabajas sobre un RAG local para vaults de Obsidian: FastAPI + SQLite + Ollama,
sin servicios externos. Hoy el proyecto asume **un único usuario en su propia
máquina**: `app/http_security.py` sólo acepta cabeceras `Host` de loopback
(`localhost`, `127.0.0.1`, `::1`), `app/main.py` escucha exclusivamente en
`127.0.0.1:8000`, hay un único vault configurado en `.env`
(`OBSIDIAN_VAULT_PATH`) y una única base SQLite
(`RAG_DATABASE_PATH`). No existe ningún concepto de usuario ni de sesión.

**Objetivo de negocio**: convertir esto en un servidor que corre en una
máquina de una red local (LAN) y al que se conectan varios PCs por navegador.
Cada persona inicia sesión, ve sólo los vaults a los que tiene acceso, y todo
el cómputo (embeddings, recuperación, generación) sigue ocurriendo en el
servidor. Nada se almacena ni se calcula en los PCs cliente: sólo renderizan
HTML/JS servido por el backend, igual que hoy.

Antes de escribir código, lee: `app/http_security.py`, `app/main.py`,
`app/config.py`, `app/env_config.py`, `app/vault_config.py`, `app/db.py`,
`app/rag/vector_store.py`, `app/rag/retrieval.py`, `app/rag/agents.py`,
`app/rag/ollama_client.py`, `app/static/app.js`, `app/templates/index.html`,
`scripts/download_reranker_onnx.py` (como referencia de estilo para scripts
de utilidad).

## Restricciones invariables

- **Ollama sigue en loopback, siempre.** El servidor FastAPI es el único
  proceso que le habla, por `127.0.0.1`, exactamente igual que ahora. Ollama
  **nunca** se expone a la LAN. La validación existente que rechaza
  `OLLAMA_BASE_URL` no-loopback y modelos con `cloud` en el nombre no se toca.
- **Cero servicios externos nuevos.** Nada de Redis, Postgres, proveedores de
  identidad en la nube, ni dependencias pesadas para hashing (usa
  `hashlib.scrypt` de la biblioteca estándar, o como mucho `argon2-cffi` si
  prefieres el estándar de la industria, pero justifícalo).
- **Todo en SQLite, todo en el servidor.** Nada de almacenamiento en el
  cliente más allá de la cookie de sesión.
- **Compatibilidad con instalaciones existentes.** Una instalación actual
  (un solo vault, sin usuarios, `127.0.0.1`) debe poder migrar sin perder
  conversaciones, proyectos ni el índice ya construido.
- Mantén el estilo del repositorio: docstrings y comentarios en español,
  explicando el porqué de las decisiones y no el qué.
- No añadas frameworks de frontend con build. Sigue sirviendo HTML/CSS/JS
  vanilla sin `innerHTML` inseguro.

## Fase 1 — Modelo de datos: multi-vault y usuarios

Cada vault mantiene **su propia base SQLite independiente** (mismo esquema
que hoy tiene `RAG_DATABASE_PATH`, incluida la reconstrucción atómica vía
`staged_*`). Un registro central nuevo mapea vaults:

```sql
CREATE TABLE IF NOT EXISTS vaults (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    path        TEXT NOT NULL,
    db_path     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
```

En una base de datos separada de administración (o en la primera base
existente, decide y documenta el porqué), añade:

```sql
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    is_admin       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vault_access (
    user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vault_id  INTEGER NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
    role      TEXT NOT NULL DEFAULT 'viewer',
    PRIMARY KEY (user_id, vault_id)
);
```

Añade `user_id` (nullable, para no romper turnos históricos) a la tabla de
conversaciones existente, para poder trazar quién preguntó qué.

**Migración**: al arrancar sobre una instalación existente, crea
automáticamente una entrada en `vaults` a partir de `OBSIDIAN_VAULT_PATH` y
`RAG_DATABASE_PATH` actuales, sin mover ni reconstruir el índice ya generado.
No crees usuarios automáticamente: la ausencia de usuarios debe forzar el
flujo de creación de administrador de la Fase 2, no un acceso implícito.

`app/vault_config.py` deja de asumir un único vault: expón
`list_vaults()`, `get_vault(vault_id)`, `resolve_vault_db_path(vault_id)`.
Cada request de API que toque documentos/búsqueda/proyectos debe llevar
`vault_id`, resuelto del usuario autenticado y del vault seleccionado en la
UI, nunca de un valor global.

## Fase 2 — Autenticación y sesiones

Crea `app/auth.py`:

- `hash_password(password: str) -> str` y `verify_password(password, hash) -> bool`
  con `hashlib.scrypt` (sal aleatoria por usuario, parámetros de coste
  documentados).
- `create_session(user_id) -> token` (token aleatorio criptográficamente
  seguro, `secrets.token_urlsafe`), con expiración configurable
  (`RAG_SESSION_TTL_HOURS`, por defecto razonable, p. ej. 12).
- `require_user(...)` como dependencia de FastAPI que lee la cookie de
  sesión, valida contra `sessions`, la refresca si procede, y expone el
  usuario autenticado a la ruta. Rutas sin sesión válida devuelven 401 con
  mensaje genérico, igual que el resto de errores HTTP de la aplicación.
- `require_vault_access(user, vault_id, role="viewer")` para autorizar por
  vault, no sólo por usuario.

Rutas nuevas:

```text
POST /api/auth/login    # username + password -> cookie de sesión httponly
POST /api/auth/logout   # invalida la sesión actual en `sessions`
GET  /api/auth/me       # usuario actual y vaults a los que tiene acceso
```

La cookie de sesión es `httponly`, `samesite=strict` y `secure` cuando TLS
está activo (ver Fase 3). No uses JWT: la sesión server-side permite revocar
acceso con un `DELETE` inmediato en `sessions`, coherente con que todo el
estado vive en SQLite.

Script `scripts/create_admin.py` (mismo estilo que
`scripts/download_reranker_onnx.py`): crea el primer usuario administrador
de forma interactiva o por variables de entorno, nunca con una contraseña
hardcodeada en el repositorio. Sin al menos un usuario admin, el servidor
debe arrancar pero rechazar todo acceso salvo la documentación de este
script en el panel de estado.

## Fase 3 — Exposición de red y TLS

- Nueva variable `RAG_BIND_HOST` (por defecto `127.0.0.1`, preservando el
  comportamiento actual). Para uso en LAN se configura a la IP de la
  interfaz correspondiente, nunca `0.0.0.0` sin más advertencia.
- `app/http_security.py`: la validación de `Host` deja de exigir loopback
  exclusivamente y acepta el host/IP configurado explícitamente en
  `RAG_BIND_HOST` (o una lista `RAG_ALLOWED_HOSTS`). Las validaciones
  existentes de `Origin` y `Sec-Fetch-Site` para rutas que mutan estado se
  mantienen sin cambios como defensa en profundidad contra CSRF, además de
  (no en lugar de) la sesión de usuario.
- Soporte de TLS vía `ssl_keyfile`/`ssl_certfile` de uvicorn, configurables
  por `RAG_TLS_CERT_PATH` y `RAG_TLS_KEY_PATH`. Si `RAG_BIND_HOST` no es
  loopback y no hay TLS configurado, el arranque debe **negarse con un
  mensaje explícito** explicando el riesgo (contraseñas en claro en la LAN),
  no arrancar en silencio de forma insegura.
- Documenta en el README cómo generar un certificado autofirmado local y
  cómo instalarlo como confiable en los PCs cliente de la LAN.

## Fase 4 — Concurrencia

- Cola explícita de peticiones a Ollama (semáforo con tamaño configurable,
  `RAG_OLLAMA_MAX_CONCURRENCY`, por defecto 1 para hardware modesto en CPU).
  Las peticiones en espera deben reflejarse en la respuesta/estado (p. ej.
  posición en cola) en vez de bloquear sin explicación.
- La matriz NumPy en memoria de `vector_store.py` pasa de un único vault
  global a **carga perezosa por vault** con un límite de vaults simultáneos
  en memoria (`RAG_MAX_LOADED_VAULTS`, política LRU). Documenta el coste de
  memoria aproximado por vault cargado.

## Fase 5 — Frontend

- Pantalla de login antes de cualquier funcionalidad, sin frameworks nuevos.
- Selector de vault visible tras autenticarse, mostrando sólo los vaults de
  `vault_access` del usuario actual.
- El indicador de cola de Ollama (Fase 4) visible en la UI ya existente de
  estado/cabecera.
- Panel de administración mínimo (sólo para `is_admin`): alta de vaults,
  alta de usuarios, asignación de accesos. Reutiliza los patrones visuales
  ya existentes del panel **Proyectos**, no crees un sistema de diseño
  paralelo.

## Fase 6 — Pruebas

Mantén el principio actual: las pruebas simulan la API HTTP de Ollama y no
requieren Ollama instalado. Añade cobertura de:

- hashing y verificación de contraseña;
- creación, validación, expiración y revocación de sesión;
- que una ruta protegida sin sesión responde 401;
- que un usuario sin acceso a un vault no puede listarlo, consultarlo ni
  indexarlo, aunque conozca su `vault_id`;
- que el `Host` fuera de `RAG_ALLOWED_HOSTS`/`RAG_BIND_HOST` sigue siendo
  rechazado, igual que hoy lo es todo lo no-loopback;
- que el arranque con `RAG_BIND_HOST` no-loopback y sin TLS falla con
  mensaje explícito, no en silencio;
- migración automática de una instalación de un solo vault a la tabla
  `vaults`, preservando conversaciones y proyectos existentes;
- que la cola de Ollama limita la concurrencia real al valor configurado;
- que cargar un vault adicional respeta el límite `RAG_MAX_LOADED_VAULTS`
  y descarga el menos usado recientemente.

## Criterios de aceptación

1. Una instalación existente de un solo vault sigue arrancando en
   `127.0.0.1` sin usuarios y sin romper nada, hasta que se decida migrar.
2. Con `RAG_BIND_HOST` en una IP de LAN y TLS configurado, varios PCs pueden
   autenticarse, ver sólo sus vaults permitidos y conversar con streaming
   SSE igual que hoy.
3. Ollama nunca es alcanzable fuera de `127.0.0.1` del servidor.
4. Revocar el acceso de un usuario a un vault (o cerrar su sesión) corta el
   acceso de inmediato, sin esperar a que expire nada del lado del cliente.
5. `python -m pytest tests -q` pasa completo, incluidas las pruebas nuevas.

## Entregable adicional

Antes de dar la tarea por cerrada, informa de:

- archivos modificados y nuevos;
- esquema final de `vaults`, `users`, `sessions`, `vault_access`;
- cómo queda el flujo de arranque para una instalación nueva multi-usuario
  frente a una instalación existente de un solo vault;
- limitaciones conocidas que queden pendientes (por ejemplo: sin
  recuperación de contraseña, sin límite de intentos de login, sin registro
  de auditoría más allá de `user_id` en conversaciones) y una recomendación
  priorizada de qué de eso abordar primero.
