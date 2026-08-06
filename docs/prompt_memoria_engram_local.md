# Prompt futuro: memoria local tipo Engram para Obsidian RAG

## Propósito

Este documento guarda una propuesta reutilizable para implementar en este
proyecto una memoria local inspirada en Engram/Gentle-AI, sin incorporar
Gentle-AI completo como dependencia obligatoria.

La idea principal es convertir la memoria en una entidad de primera clase: no
solo historial de conversaciones, sino observaciones estructuradas, buscables,
auditables y exportables. El diseño debe estar preparado desde el principio
para que cada usuario tenga una experiencia personalizada y una memoria
aislada de forma inequívoca, aunque la primera instalación siga siendo local y
de un solo usuario.

## Contexto del proyecto

Proyecto actual: aplicación local RAG para Obsidian con FastAPI, Ollama,
SQLite/FTS5, embeddings y UI web.

Puntos relevantes del código:

- Backend FastAPI en `app/`.
- UI en `app/templates/index.html`, `app/static/app.js` y `app/static/styles.css`.
- Memoria y lógica RAG en `app/rag/`.
- Pruebas en `tests/`.
- El sistema ya usa búsqueda híbrida: embeddings + FTS5 + grafo opcional.
- Ya existen memoria conversacional resumida y memoria compartida resumida por
  proyecto. La memoria estructurada debe complementarlas, no sustituirlas ni
  duplicarlas.
- Cada turno ya conserva pregunta, respuesta, fuentes, modelo y métricas de
  agentes. La auditoría nueva debe ampliar ese recibo existente.
- La recuperación documental admite fragmentos padre-hijo: el hijo produce la
  coincidencia y el padre aporta contexto. Una memoria creada desde una fuente
  debe guardar por defecto la coincidencia precisa, no todo el padre.
- Todavía no debe asumirse que existe una identidad multiusuario segura. La
  identidad autenticada es un requisito previo para prometer aislamiento real.
- El usuario valora una UI limpia, compacta y útil, no una pantalla cargada de
  texto técnico.

## Decisión técnica recomendada

No integrar Gentle-AI entero.

Adaptar estas ideas:

- Memoria local estructurada y propiedad inequívoca mediante un UUID estable de
  usuario.
- Memoria privada por usuario y, opcionalmente, acotada a proyecto o vault.
- Observaciones persistentes con tipo, scope, topic key, fuente y estado.
- Búsqueda progresiva: primero resultados compactos, después detalle sólo si
  hace falta.
- Dedupe y actualización dentro del espacio del usuario, nunca de forma global.
- Export/import local estilo `.engram/`, pero con nombre propio del proyecto.
- UI dedicada para buscar, revisar, borrar, exportar e importar memoria.
- Botón para guardar una respuesta o fragmento como memoria del proyecto.
- Recibos de respuesta para auditoría: modelo, perfil, hijos/padres recuperados,
  memorias usadas y fuentes.
- Referencias de procedencia duraderas basadas en ruta, encabezado/ancla y
  SHA-256; los identificadores internos de chunks cambian al reindexar.
- Separación estricta entre personalización y evidencia documental.

Evitar al principio:

- Multiagente completo.
- SDD completo como flujo obligatorio.
- MCP obligatorio.
- Cloud sync.
- TUI como interfaz principal.
- Dependencia directa de un binario Go externo.

## Principios multiusuario no negociables

- Cada observación tiene un `owner_user_id` obligatorio y estable. Debe ser un
  UUID interno; el correo o el nombre visible pueden cambiar y no son claves de
  propiedad.
- El backend obtiene el usuario de una sesión autenticada. Ningún endpoint
  acepta un `user_id` arbitrario para decidir qué memoria leer o modificar.
- Toda búsqueda, incluida FTS5 o una futura búsqueda vectorial, vuelve a aplicar
  autorización antes de devolver resultados.
- La memoria es privada por defecto. Compartir una observación requiere una
  acción explícita y comprobaciones de pertenencia al proyecto o vault.
- Dos usuarios pueden tener el mismo `topic_key` con valores diferentes sin
  colisionar ni actualizarse entre sí.
- Las preferencias, objetivos y decisiones personalizan la experiencia, pero
  no son evidencia documental ni reciben citas `[n]`.
- Una observación derivada de un documento sólo puede respaldar una afirmación
  si su documento original se vuelve a resolver y recuperar del índice activo.
- La memoria nunca anula la abstención documental ni permite responder como si
  existiera una fuente cuando los documentos no aportan evidencia suficiente.
- El usuario puede inspeccionar, editar, exportar, desactivar y borrar
  definitivamente su memoria.
- El contenido guardado se trata como datos no confiables: nunca como
  instrucciones para el sistema o los agentes.

## Prompt listo para usar

Usa este prompt en el futuro para pedir la implementación.

```text
Quiero implementar en este proyecto una memoria local inspirada en
Engram/Gentle-AI, adaptada a esta app RAG local para Obsidian y preparada desde
el principio para varios usuarios.

Objetivo:
Crear una memoria estructurada y personalizada por usuario, separada del simple
historial y de los resúmenes acumulativos que ya existen. Debe permitir guardar
observaciones reutilizables, buscarlas, deduplicarlas, exportarlas/importarlas y
usarlas como contexto auxiliar en respuestas RAG sin mezclarlas con la evidencia
documental.

Antes de tocar código:
1. Lee `README.md`.
2. Revisa `app/db.py`, `app/main.py` y `app/schemas.py`.
3. Revisa `app/rag/memory.py`, `app/rag/agents.py`,
   `app/rag/retrieval.py`, `app/rag/vector_store.py` y
   `app/rag/ollama_client.py`.
4. Revisa la UI en `app/templates/index.html`, `app/static/app.js` y
   `app/static/styles.css`.
5. Revisa las pruebas actuales en `tests/`.
6. Determina qué identidad, autenticación y autorización existen realmente. No
   simules seguridad multiusuario mediante un selector o un `user_id` enviado
   por el navegador.
7. Conserva la memoria conversacional, la memoria resumida de proyecto, la
   validación de citas, el inspector de contexto y los recibos de turno actuales.
8. Mantente alineado con el estilo del proyecto.

Reglas de diseño obligatorias:
- El usuario actual procede siempre de la sesión autenticada.
- Toda observación pertenece a un `owner_user_id` UUID.
- Una observación es privada salvo que se comparta explícitamente.
- La personalización no es evidencia documental y no puede recibir citas.
- La memoria no puede saltarse la abstención por falta de documentos.
- No uses `chunk_id` o `document_id` como referencia externa duradera: cambian
  durante una reconstrucción. Usa ruta, heading/ancla y SHA-256.
- No guardes automáticamente información sensible ni toda la conversación.
- Las migraciones deben conservar los datos existentes y ser reversibles en la
  medida razonable.

Implementación deseada por fases:

Fase 0 - Identidad y aislamiento:
- Si todavía no existe soporte de usuarios, crear una base mínima y segura:
  - tabla `users` con UUID interno estable;
  - sesión autenticada y una función central para obtener `current_user`;
  - cookies de sesión `HttpOnly` y `SameSite` adecuadas;
  - propietario en proyectos y comprobación de acceso a conversaciones;
  - proyecto `General` independiente para cada usuario.
- En una instalación todavía monousuario, migrar los datos existentes a un
  usuario local estable, pero no presentar eso como autenticación multiusuario.
- Nunca confiar en un `user_id` incluido en query params, body o cabeceras
  controladas por el cliente.
- Centralizar las comprobaciones de propiedad para que ningún endpoint pueda
  olvidar el filtro por usuario.
- Si existen proyectos compartidos, representar la pertenencia y el rol en una
  tabla de membresías; ser propietario de una observación y poder leer un
  proyecto son permisos distintos.
- No ampliar la exposición de red de la aplicación como efecto colateral de
  esta funcionalidad.

Fase 1 - Modelo de memoria estructurada:
- Crear o ampliar almacenamiento local en SQLite para `observations`.
- Campos mínimos:
  - id
  - owner_user_id
  - project_id opcional
  - vault_id opcional
  - scope: personal | project | vault
  - visibility: private | shared
  - kind: decision | fact | preference | protocol | bugfix | risk | task | note
  - topic_key
  - title
  - content
  - source_type: chat | document | manual | answer
  - source_turn_id opcional
  - source_document_path opcional
  - source_heading_or_anchor opcional
  - source_document_sha256 opcional
  - content_hash
  - status: active | stale | review | deleted
  - duplicate_count
  - use_count
  - created_at
  - updated_at
  - last_used_at
- Añadir restricciones de coherencia: `personal` no requiere proyecto/vault,
  `project` exige un proyecto autorizado y `vault` exige un vault autorizado.
- Crear índices útiles para propietario, proyecto, vault, scope, kind, status y
  topic_key.
- La unicidad de `topic_key` debe incluir propietario, scope, proyecto/vault y
  kind. En SQLite, usar índices únicos parciales cuando haya campos nulos.
- Añadir FTS5 para buscar por title/content. Los rowids encontrados deben unirse
  de nuevo con `observations` y filtrarse por autorización antes de hidratarse.
- Mantener la primera versión sólo con FTS5. Si más adelante se añaden
  embeddings, usar una tabla/matriz separada de los chunks documentales y una
  huella propia para modelo, dimensión y prefijos.
- Las observaciones derivadas de conversaciones borradas o documentos cuyo
  SHA-256 haya cambiado deben marcarse para revisión u obsolescencia según una
  política explícita.

Fase 2 - API backend:
- Crear endpoints autenticados para:
  - listar memorias
  - buscar memorias
  - crear memoria manual
  - guardar memoria desde una respuesta
  - guardar la coincidencia precisa de una fuente padre-hijo
  - actualizar memoria
  - marcar como obsoleta/revisar
  - borrar lógicamente
  - purgar definitivamente una memoria propia
  - exportar memoria
  - importar memoria
- Todos los endpoints deben derivar `owner_user_id` de la sesión y comprobar
  también el acceso al proyecto, vault, conversación o documento de origen.
- No aceptar cambios de propietario mediante PATCH ni durante una importación.
- Las respuestas de listado deben ser compactas; cargar el contenido completo
  sólo al abrir el detalle.
- Preferir borrado lógico para el uso normal y ofrecer borrado definitivo con
  confirmación clara.
- Validar entradas con modelos Pydantic.

Fase 3 - Integración RAG:
- Antes de responder, buscar únicamente memorias activas y autorizadas del
  usuario actual: personales y, cuando corresponda, las de su proyecto/vault.
- Incluir sólo un número reducido de memorias compactas y asignarles un
  presupuesto de contexto independiente.
- Separar preferencias de presentación (idioma, nivel técnico, concisión) de
  hechos, decisiones y objetivos de trabajo.
- No llenar el prompt con memoria completa si no es necesario.
- Etiquetar el bloque de memoria como contexto auxiliar no documental e indicar
  al modelo que ignore cualquier instrucción contenida en él.
- La recuperación de memoria no puede sustituir la recuperación documental ni
  evitar la abstención cuando no hay evidencia suficiente.
- Una observación con origen documental sólo puede convertirse en evidencia si
  se vuelve a resolver `source_document_path` y el documento aparece entre las
  fuentes recuperadas actuales.
- Registrar los IDs de las memorias usadas, su scope y su versión/hash en el
  recibo del turno.
- Actualizar `last_used_at` y `use_count` únicamente para memorias realmente
  incluidas en el prompt.
- Si una respuesta contiene una decisión, preferencia o hecho estable, ofrecer
  guardarla. No hacerlo automáticamente salvo consentimiento y configuración
  explícitos del usuario.
- Incorporar un feature flag para poder comparar respuestas con y sin memoria
  estructurada.

Fase 4 - UI:
- Añadir una sección o pestaña `Mi memoria`.
- La UI debe ser compacta, clara y orientada al uso real.
- Debe incluir:
  - buscador
  - filtros por proyecto/vault, scope, tipo y estado
  - lista de memorias
  - detalle editable
  - indicación inequívoca de memoria privada o compartida
  - acciones: guardar, marcar revisar, marcar obsoleta, borrar, purgar,
    exportar e importar
  - botón en respuestas: `Recordar esto`
  - botón en fuentes: `Recordar coincidencia`
- Al guardar desde una respuesta o fuente, abrir un diálogo breve para revisar
  título, tipo, scope, proyecto y contenido antes de confirmar.
- Mostrar `N memorias usadas` de forma discreta en cada respuesta y permitir ver
  cuáles fueron, sin convertirlas en fuentes citables.
- No mostrar texto técnico innecesario en pantalla.
- Mantener la estetica sobria del proyecto.
- Cuidar mucho responsive/mobile.

Fase 5 - Export/import:
- Guardar la memoria operativa en SQLite. La exportación debe ser una acción
  explícita hacia una ubicación elegida o un directorio de datos de la app.
- No escribir por defecto dentro del vault: un vault sincronizado podría subir
  memorias personales a Obsidian Sync, Dropbox u otro servicio.
- Exportar:
  - `manifest.json`
  - archivos `.jsonl` con observaciones
- Incluir versión de esquema, fecha y checksums, pero nunca credenciales,
  hashes de contraseña, cookies o secretos de sesión.
- Exportar sólo la memoria autorizada del usuario actual.
- Al importar, ignorar cualquier `owner_user_id` externo y asignar las
  observaciones al usuario autenticado después de su confirmación.
- Importar de forma idempotente usando scope, proyecto/vault, kind, topic_key y
  contenido normalizado.
- Advertir que el archivo exportado puede contener información sensible.
- No activar sincronización automática por defecto.

Fase 6 - Auditoría:
- Ampliar el recibo que ya se guarda en `conversation_turns`; no crear un
  segundo historial completo sin necesidad.
- Registrar:
  - pregunta
  - modelo
  - perfil
  - documentos e hijos/padres usados
  - IDs y versiones de memorias usadas
  - política de recuperación, reranking y abstención
  - huella del índice
  - resultado de validación de citas
  - parámetros relevantes
  - fecha
- No duplicar en el recibo el contenido sensible completo si basta con IDs,
  hashes y referencias.
- Mostrar esta información de forma discreta en UI, como panel desplegable o
  detalle técnico opcional.

Fase 7 - Memoria compartida opcional:
- Implementarla sólo después de verificar el aislamiento privado.
- Una observación compartida conserva `owner_user_id` y registra quién la
  compartió.
- Comprobar pertenencia y rol en el proyecto/vault al leer, editar o retirar.
- No convertir automáticamente una memoria privada en compartida.
- La memoria personal de un usuario nunca debe aparecer a otros miembros del
  mismo proyecto.

Criterios de aceptación:
- Las pruebas existentes siguen pasando.
- Hay pruebas nuevas para almacenamiento, migraciones y endpoints principales.
- Dos usuarios con el mismo `topic_key` conservan valores independientes.
- Un usuario no puede listar, buscar, abrir, modificar, exportar ni borrar la
  memoria de otro, aunque manipule IDs o parámetros HTTP.
- FTS5 y cualquier búsqueda vectorial respetan el mismo aislamiento.
- Cada usuario tiene su propio proyecto `General` y sus conversaciones no se
  mezclan.
- La memoria privada no se filtra a miembros de un proyecto compartido.
- La UI no se vuelve más ruidosa.
- La memoria puede buscarse desde la UI.
- Una respuesta puede guardarse como memoria.
- Una fuente padre-hijo guarda por defecto el hijo coincidente y una referencia
  estable al documento.
- La memoria puede exportarse e importarse.
- Importar dos veces el mismo archivo no duplica observaciones.
- Borrar definitivamente elimina la observación de tabla, FTS y futuros
  embeddings.
- Las observaciones `stale`, `review` o `deleted` no entran en el prompt salvo
  una acción explícita prevista para revisión.
- El contenido de una memoria no puede inyectar instrucciones al sistema.
- La memoria nunca aparece como cita documental ni evita la abstención por
  falta de evidencia.
- El presupuesto de memoria limita de forma determinista el contexto añadido.
- El sistema sigue funcionando aunque no haya memorias.
- No se introduce Gentle-AI como dependencia obligatoria.

Prioridad:
Empieza por una primera versión pequeña y funcional:
1. identidad estable y capa central de autorización;
2. tabla `observations` con `owner_user_id` y migración del usuario local;
3. endpoints básicos siempre acotados al usuario autenticado;
4. panel UI `Mi memoria`;
5. botones `Recordar esto` y `Recordar coincidencia` con confirmación;
6. búsqueda FTS5 aislada por usuario;
7. integración RAG detrás de un feature flag y con presupuesto reducido.

Después de eso, implementar export/import, dedupe avanzado, recibos ampliados,
memoria compartida explícita, embeddings propios de observaciones y posible
integración opcional con Engram externo.
```

## Esquema conceptual recomendado

```text
Usuario autenticado pregunta
  -> backend obtiene current_user desde la sesión
  -> comprueba acceso al proyecto y al vault
  -> normaliza la pregunta
  -> busca observaciones privadas/autorizadas de ese usuario
  -> busca documentos autorizados y relevantes
  -> separa memoria auxiliar de evidencia documental
  -> compone ambos contextos con presupuestos independientes
  -> llama a Ollama
  -> muestra respuesta con fuentes
  -> valida que las citas sólo apunten a documentos
  -> registra memorias y fuentes usadas en el recibo del turno
  -> permite recordar una parte de la respuesta o coincidencia, previa revisión
```

Ejemplo de personalización esperada:

```text
Usuario A
  topic_key: estilo_respuesta
  preferencia: breve, ejecutiva y poco técnica

Usuario B
  topic_key: estilo_respuesta
  preferencia: detallada, técnica y con ejemplos
```

Ambos pueden consultar los mismos documentos y recibir los mismos hechos y
fuentes, pero la forma de redactar se adapta a cada usuario. Sus observaciones
no colisionan porque la unicidad incluye `owner_user_id`.

## UI recomendada

La memoria no debe sentirse como una herramienta de administración pesada.
Debe parecer una extensión natural del chat.

Ideas concretas:

- Pestaña o panel: `Mi memoria`.
- Buscador principal con placeholder corto: `Buscar memoria`.
- Filtros compactos con selects o chips:
  - proyecto o vault
  - tipo
  - scope
  - estado
- Lista densa, con título, tipo, fecha y estado.
- Panel lateral o modal para editar detalle.
- Acción contextual en cada respuesta: `Recordar esto`.
- Acción en fuentes: `Recordar coincidencia`.
- Confirmación editable antes de guardar cualquier contenido sugerido.
- Marca discreta `Privada` o `Compartida`.
- Indicador discreto en una respuesta: `3 memorias usadas`.
- Detalle técnico oculto por defecto.

Evitar:

- Grandes bloques explicativos.
- Mensajes permanentes de estado técnico.
- Tarjetas enormes.
- Duplicar información que ya aparece en el chat.
- Obligar al usuario a entender Engram, MCP o Gentle-AI.
- Mostrar o permitir editar IDs de usuario.
- Sugerir que una memoria es una fuente documental.
- Guardar automáticamente datos personales sensibles.

## Exportación local sugerida

La memoria operativa permanece en SQLite. Para una exportación manual, usar una
ubicación elegida por el usuario o un directorio de datos de la aplicación:

```text
data/
  exports/
    memory-<fecha>/
      manifest.json
      observations-0001.jsonl
```

No incluir el UUID en nombres visibles salvo necesidad técnica. No escribir por
defecto en `.obsidian-rag/` dentro del vault, porque el vault puede estar
sincronizado con servicios externos. Si el usuario elige expresamente el vault,
mostrar una advertencia de privacidad.

## Observaciones finales

La idea más valiosa de Engram para este proyecto no es el tooling de agentes,
sino el concepto de memoria persistente local, buscable y mantenible.

La implementación debe ser local-first, opcional, auditable y fácil de
borrar/exportar. También debe ser privada por defecto, inequívocamente propiedad
de un usuario y segura frente a accesos cruzados.

La memoria estructurada no reemplaza los resúmenes actuales. Su función es
aportar unidades pequeñas, editables y recuperables para personalizar la
experiencia: preferencias de estilo, nivel técnico, objetivos, decisiones y
criterios de trabajo. Los hechos documentales continúan procediendo del índice
RAG y de sus fuentes citables.
