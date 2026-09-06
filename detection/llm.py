"""
What ? its already 4 AM!
Thin LLM transport supporting Anthropic and OpenAI wire protocols.

Retries 5xx/network errors once with a 1s backoff; 4xx fail fast. Each call is
bounded by PER_CALL_TIMEOUT_SECONDS.
"""

import asyncio
import os
from typing import Optional

import httpx

_PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
}


async def llm_call(prompt: str, system: Optional[str] = None) -> str:
    """Runs a prompt against the configured provider, retrying transient 5xx/network failures once."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-20241022")
    timeout = int(os.environ.get("PER_CALL_TIMEOUT_SECONDS", "8"))

    for attempt in range(2):
        try:
            if provider == "anthropic":
                return await _call_anthropic(api_key, model, prompt, system, timeout)
            else:
                return await _call_openai(api_key, model, prompt, system, timeout)
        except httpx.HTTPError as exc:
            if isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500:
                raise
            if attempt == 1:
                raise
            await asyncio.sleep(1)

    return ""


async def _call_anthropic(
    api_key: str, model: str, prompt: str, system: Optional[str], timeout: int
) -> str:
    """POST prompt to the Anthropic Messages API."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body: dict = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(_PROVIDER_URLS["anthropic"], headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


async def _call_openai(
    api_key: str, model: str, prompt: str, system: Optional[str], timeout: int
) -> str:
    """POST prompt to the OpenAI Chat Completions API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            _PROVIDER_URLS["openai"],
            headers=headers,
            json={"model": model, "messages": messages, "max_tokens": 1024},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
