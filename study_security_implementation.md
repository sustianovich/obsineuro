# Exponer el RAG a una red de PCs: implementación pendiente

Contexto: hoy la app está pensada para un único PC (`127.0.0.1`). Si en el
futuro se quiere que varios PCs de una LAN de confianza consulten el mismo
servidor, hace falta este trabajo antes de cambiar el bind a `0.0.0.0`.

## Diagnóstico (por qué no se puede exponer tal cual)

- El servidor arranca en `host="127.0.0.1"` ([app/main.py](app/main.py), función
  `if __name__ == "__main__"`).
- `reject_unsafe_request` en [app/http_security.py](app/http_security.py) exige
  que la cabecera `Host` sea loopback (`is_loopback_host`). Si se cambia el
  bind a `0.0.0.0` sin tocar esto, el middleware rechazará el 100% del
  tráfico desde otros PCs.
- No existe ninguna autenticación: cualquier dispositivo que alcance el
  puerto tiene acceso total (leer el vault indexado, borrar conversaciones/
  proyectos, cambiar de vault). Eso es aceptable en loopback puro, no en LAN.

## Decisión tomada

Usar **dos capas combinadas**, no una sola:

1. **Whitelist de subred** como primer filtro barato (sustituye/complementa
   la comprobación actual de `is_loopback_host`).
2. **Token compartido** como autenticación real por encima. Se prefirió
   frente a "solo IP whitelist" porque una IP no es identidad: cualquier
   malware, invitado de wifi o dispositivo IoT dentro del rango permitido la
   hereda gratis (y en una LAN, ARP spoofing hace trivial falsear el origen).

Fuera de alcance por ahora: HTTPS/TLS. Solo se justificaría si la red
incluyera wifi con invitados u otros equipos no controlados; para una LAN
doméstica/oficina de confianza no compensa la complejidad añadida.

## Piezas a implementar

- Generación del token: aleatorio criptográfico (`secrets.token_urlsafe(32)`
  en Python, o `RNGCryptoServiceProvider` en PowerShell), guardado en `.env`
  como `RAG_ACCESS_TOKEN` vía `update_env_value` de
  [app/env_config.py](app/env_config.py). Nunca en el código ni en git.
- Comparación del token con `hmac.compare_digest()`, no `==` (evita timing
  attack).
- Pantalla mínima "introduce el código de acceso": al validarlo, pone una
  cookie `HttpOnly` + `SameSite=Strict` (sin exponer el token a JS). Evita
  tener que tocar cada `fetch()` de [app/static/app.js](app/static/app.js)
  para añadir cabeceras a mano.
- Middleware (`local_request_security` en [app/main.py](app/main.py)):
  añadir la comprobación de subred + cookie de sesión, conservando las
  protecciones que ya existen (Host header, Origin/Sec-Fetch-Site, cabeceras
  de seguridad).
- Ajustar `is_loopback_host`/`reject_unsafe_request` para aceptar la subred
  configurada además de loopback.
- Documentar en `.env.example` las nuevas variables (token, subred
  permitida) siguiendo el estilo de comentarios ya usado ahí.

## Prompt para retomar esto

```
Quiero implementar el acceso desde una red de PCs (LAN de confianza) al RAG
local de Obsidian, siguiendo lo acordado en security_implementation.md:

1. Bind del servidor a la interfaz de LAN (no solo 127.0.0.1) en app/main.py.
2. Whitelist de subred configurable (variable de entorno) que sustituye/
   complementa la comprobación is_loopback_host de app/http_security.py.
3. Token de acceso compartido:
   - Generado con secrets.token_urlsafe(32), guardado en .env como
     RAG_ACCESS_TOKEN mediante update_env_value (app/env_config.py).
   - Pantalla simple de "código de acceso" que al validarse deja una cookie
     HttpOnly + SameSite=Strict (nada de exponer el token en JS ni en
     localStorage).
   - Comparación con hmac.compare_digest(), nunca con ==.
4. El middleware local_request_security en app/main.py debe seguir
   aplicando las protecciones existentes (Host header, Origin/
   Sec-Fetch-Site, cabeceras de seguridad) y sumar la comprobación de
   cookie + subred.
5. Actualizar .env.example con las variables nuevas, documentadas con el
   mismo estilo que las demás.
6. Cubrir con tests el rechazo de peticiones sin cookie válida, con IP
   fuera de la subred, y con token incorrecto (siguiendo el estilo de
   tests/test_http_security.py si existe, o creándolo).

No añadas HTTPS/TLS ni gestión de usuarios múltiples: es un único secreto
compartido para una LAN de confianza, no un sistema multiusuario.
```
