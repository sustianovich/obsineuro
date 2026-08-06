"""Protecciones HTTP para la interfaz local sin autenticacion."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse


UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
PUBLIC_REQUEST_ERROR = "Solicitud no permitida."


def _normalise_host(value: str) -> str:
    hostname = value.casefold()
    if hostname == "localhost":
        return hostname
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        return hostname


def _parse_host_header(value: str | None) -> tuple[str, int | None] | None:
    """Parsea Host sin aceptar credenciales, rutas ni puertos invalidos."""
    if not value:
        return None
    raw = value.strip()
    if not raw or any(char in raw for char in "@/?#\\"):
        return None

    hostname = raw
    port_text: str | None = None
    if raw.startswith("["):
        closing = raw.find("]")
        if closing == -1:
            return None
        hostname = raw[1:closing]
        suffix = raw[closing + 1 :]
        if suffix:
            if not suffix.startswith(":"):
                return None
            port_text = suffix[1:]
    elif raw.count(":") == 1:
        hostname, port_text = raw.rsplit(":", 1)
    elif raw.count(":") > 1:
        # Una IPv6 sin corchetes no es una cabecera Host valida.
        return None

    if not hostname:
        return None
    port: int | None = None
    if port_text is not None:
        if not port_text.isdecimal():
            return None
        port = int(port_text)
        if not 1 <= port <= 65535:
            return None
    return _normalise_host(hostname), port


def is_loopback_host(value: str | None) -> bool:
    parsed = _parse_host_header(value)
    if parsed is None:
        return False
    hostname, _ = parsed
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _origin_matches_request(origin: str, request: Request) -> bool:
    try:
        parsed = urlsplit(origin)
        origin_hostname = parsed.hostname
        origin_port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not origin_hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False

    request_host = _parse_host_header(request.headers.get("host"))
    if request_host is None:
        return False
    origin_host = _normalise_host(origin_hostname)
    request_port = request_host[1]
    origin_effective_port = origin_port or (
        443 if parsed.scheme == "https" else 80
    )
    request_effective_port = request_port or (
        443 if request.url.scheme == "https" else 80
    )
    return (
        parsed.scheme == request.url.scheme
        and origin_host == request_host[0]
        and origin_effective_port == request_effective_port
    )


def reject_unsafe_request(request: Request) -> JSONResponse | None:
    """Devuelve una respuesta de rechazo cuando la peticion no es local."""
    if not is_loopback_host(request.headers.get("host")):
        return JSONResponse(
            status_code=400,
            content={"detail": "Cabecera Host no permitida."},
        )

    if request.method not in UNSAFE_METHODS:
        return None

    fetch_site = request.headers.get("sec-fetch-site", "").casefold()
    if fetch_site and fetch_site not in {"same-origin", "none"}:
        return JSONResponse(
            status_code=403,
            content={"detail": PUBLIC_REQUEST_ERROR},
        )

    origin = request.headers.get("origin")
    if origin and not _origin_matches_request(origin, request):
        return JSONResponse(
            status_code=403,
            content={"detail": PUBLIC_REQUEST_ERROR},
        )
    return None
