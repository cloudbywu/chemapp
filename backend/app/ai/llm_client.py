from __future__ import annotations

import json
import ipaddress
import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any, Iterator
from urllib.parse import urlsplit

logger = logging.getLogger("chemapp.llm")


def _build_client(api_key: str, base_url: str):
    try:
        from openai import DefaultHttpxClient, OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: uv add openai")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=600.0,
        http_client=DefaultHttpxClient(follow_redirects=False),
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_DNS_RESOLUTION_TIMEOUT_SECONDS = 5.0
_DNS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chemapp-llm-dns")


def _resolve_host_addresses(host: str, port: int) -> set[str]:
    """Resolve LLM hostnames with a short timeout.

    socket.getaddrinfo blocks with no timeout; run it in a worker thread so
    an unresponsive resolver fails closed (ValueError) instead of hanging the
    request thread.

    Residual TOCTOU risk: the addresses validated here may differ from the
    ones the HTTP client connects to moments later (DNS TTL expiry / rebinding).
    This check is a fail-closed deployment guard, not connection-level IP
    pinning.
    """
    future = _DNS_EXECUTOR.submit(
        socket.getaddrinfo,
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    try:
        items = future.result(timeout=_DNS_RESOLUTION_TIMEOUT_SECONDS)
    except _FutureTimeoutError as exc:
        future.cancel()
        raise ValueError(f"LLM host resolution timed out: {host}") from exc
    except socket.gaierror as exc:
        raise ValueError(f"LLM host could not be resolved: {host}") from exc
    addresses = {item[4][0] for item in items}
    if not addresses:
        raise ValueError(f"LLM host could not be resolved: {host}")
    return addresses


def _validate_client_base_url(base_url: str) -> str:
    if len(base_url) > 2048:
        raise ValueError("LLM base URL is too long")
    parsed = urlsplit(base_url)
    allowed_schemes = {"https"}
    if _truthy_env("CHEMAPP_ALLOW_INSECURE_LLM_BASE_URL"):
        allowed_schemes.add("http")
    if parsed.scheme.lower() not in allowed_schemes:
        raise ValueError("LLM base URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("LLM base URL must have a host and no embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM base URL must not contain a query or fragment")

    host = parsed.hostname.rstrip(".").lower()
    configured_hosts = {
        item.strip().rstrip(".").lower()
        for item in os.environ.get("CHEMAPP_LLM_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    }
    configured_provider = urlsplit(
        os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    ).hostname
    if configured_provider:
        configured_hosts.add(configured_provider.rstrip(".").lower())
    configured_hosts.update({"api.openai.com", "api.deepseek.com"})
    allow_arbitrary = _truthy_env("CHEMAPP_ALLOW_ARBITRARY_PUBLIC_LLM_HOSTS")
    if not allow_arbitrary and not any(
        host == item or host.endswith("." + item) for item in configured_hosts
    ):
        raise ValueError("LLM host is not in CHEMAPP_LLM_ALLOWED_HOSTS")

    if not _truthy_env("CHEMAPP_ALLOW_PRIVATE_LLM_BASE_URL"):
        addresses = _resolve_host_addresses(host, parsed.port or 443)
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
            if not ip.is_global:
                raise ValueError("Private, loopback, link-local, and reserved LLM hosts are not allowed")
    return base_url.rstrip("/")


def _resolve(model: str | None, api_key: str | None, base_url: str | None):
    if base_url is not None:
        if not api_key:
            raise ValueError("A client-supplied LLM base URL requires a client-supplied API key")
        url = _validate_client_base_url(base_url)
        key = api_key
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        # Server configuration is trusted and supports private on-prem providers.
        url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    m = model or os.environ.get("OPENAI_MODEL", "deepseek-v4-pro")
    return _build_client(key, url), m


def _build_messages(
    system_prompt: str,
    user_message: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system_prompt}]
    if history:
        for h in history[-50:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content[:20000]})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _bounded_max_tokens(value: int) -> int:
    configured = int(os.environ.get("CHEMAPP_LLM_MAX_TOKENS", "32768"))
    return max(1, min(int(value), max(1, configured)))


def truncate_text(text: str, limit: int = 512 * 1024) -> tuple[str, bool]:
    """Truncate prompt context to ``limit`` characters, returning a flag."""

    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[context truncated]...", True


def chat(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 128000,
    response_format: dict | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    client, m = _resolve(model, api_key, base_url)

    msgs = _build_messages(system_prompt, user_message, history)

    kwargs: dict[str, Any] = {
        "model": m,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": _bounded_max_tokens(max_tokens),
    }

    if response_format:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""

    if not content:
        # Never fall back to reasoning_content: the model's chain-of-thought
        # is not an answer and must not leak into user-visible output.
        logger.warning(
            "LLM returned empty content (model=%s); refusing reasoning_content fallback",
            m,
        )

    return content


def chat_json(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 128000,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    raw = chat(
        system_prompt,
        user_message,
        temperature=temperature,
        max_tokens=_bounded_max_tokens(max_tokens),
        response_format={"type": "json_object"},
        model=model,
        api_key=api_key,
        base_url=base_url,
        history=history,
    )
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": raw, "_parse_error": True}


def stream_chat(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 128000,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> Iterator[str]:
    """Stream chat completion, yielding JSON lines with type/content."""
    client, m = _resolve(model, api_key, base_url)

    msgs = _build_messages(system_prompt, user_message, history)

    resp = client.chat.completions.create(
        model=m,
        messages=msgs,
        temperature=temperature,
        max_tokens=_bounded_max_tokens(max_tokens),
        stream=True,
    )

    last_reasoning = False
    for chunk in resp:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        # DeepSeek V4: reasoning_content for thinking, content for answer
        reasoning = getattr(delta, "reasoning_content", None) or ""
        content = delta.content or ""

        if reasoning:
            if not last_reasoning:
                yield json.dumps({"type": "reasoning_start"})
                last_reasoning = True
            yield json.dumps({"type": "reasoning", "content": reasoning})

        if content:
            if last_reasoning:
                yield json.dumps({"type": "reasoning_end"})
                last_reasoning = False
            yield json.dumps({"type": "answer", "content": content})

    if last_reasoning:
        yield json.dumps({"type": "reasoning_end"})

    yield json.dumps({"type": "done"})
