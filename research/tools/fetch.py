"""SSRF-guarded URL fetch tool.

The model chooses the URLs, so the tool itself must be the backstop:
- only http(s) schemes (reject file://, gopher://, ftp://, ...),
- DNS resolution is checked and private / loopback / link-local / reserved
  IPs are refused (blocks 127.0.0.1, 10/8, 172.16/12, 192.168/16, 169.254/8,
  ::1, fc00::/7, fe80::/10, ...),
- response body is capped at FETCH_MAX_BYTES (default 256 KiB),
- every call is bounded by PER_CALL_TIMEOUT_SECONDS with one
  retry-with-backoff on timeout/5xx and no retry on 4xx.
"""

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlparse

FETCH_MAX_BYTES = 256 * 1024


class FetchError(ValueError):
    """Raised for blocked URLs (SSRF guard) or failed fetches."""


def _timeout() -> float:
    try:
        return float(os.environ.get("PER_CALL_TIMEOUT_SECONDS", "8"))
    except ValueError:
        return 8.0


def _max_bytes() -> int:
    try:
        return int(os.environ.get("FETCH_MAX_BYTES", str(FETCH_MAX_BYTES)))
    except ValueError:
        return FETCH_MAX_BYTES


def assert_url_allowed(url: str) -> str:
    """Validate scheme/host synchronously; returns the hostname. Raises FetchError."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError(f"Blocked non-http(s) URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise FetchError("Blocked URL with no hostname")
    return parsed.hostname


async def _resolve_guarded(hostname: str) -> None:
    """Resolve hostname and refuse private/loopback/link-local/reserved IPs."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
    except socket.gaierror as exc:
        raise FetchError(f"DNS resolution failed for {hostname!r}: {exc}") from exc
    if not infos:
        raise FetchError(f"No addresses resolved for {hostname!r}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise FetchError(f"Blocked private/link-local/reserved IP {ip} for {hostname!r}")


async def fetch_url(url: str) -> str:
    """Fetch a URL's text with SSRF guards, a size cap, and bounded retries."""
    import httpx

    hostname = assert_url_allowed(url)
    await _resolve_guarded(hostname)

    timeout = _timeout()
    cap = _max_bytes()
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", url) as resp:
                    if 400 <= resp.status_code < 500:
                        resp.raise_for_status()
                    if resp.status_code >= 500:
                        resp.raise_for_status()
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if total > cap:
                            break  # cap: truncate, don't fail
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    return raw.decode("utf-8", errors="replace")
        except FetchError:
            raise
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500:
                raise FetchError(f"Fetch failed with status {exc.response.status_code}") from exc
            last_exc = exc
            if attempt == 1:
                raise FetchError(f"Fetch failed: {exc}") from exc
            await asyncio.sleep(0.5 * (attempt + 1))
        except (httpx.HTTPError, asyncio.TimeoutError, TimeoutError) as exc:
            last_exc = exc
            if attempt == 1:
                raise FetchError(f"Fetch failed: {exc}") from exc
            await asyncio.sleep(0.5 * (attempt + 1))
    raise FetchError(f"Fetch failed: {last_exc}")
