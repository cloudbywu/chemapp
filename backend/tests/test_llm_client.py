"""Tests for chat_json parsing fallbacks."""

from __future__ import annotations

import socket as _socket
from types import SimpleNamespace

import pytest

from app.ai import llm_client


def test_chat_json_strips_code_fences(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "chat",
        lambda *args, **kwargs: '```json\n{"ok": true}\n```',
    )
    assert llm_client.chat_json("role", "msg") == {"ok": True}


def test_chat_json_extracts_embedded_object(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "chat",
        lambda *args, **kwargs: 'prefix {"ok": 1} suffix',
    )
    assert llm_client.chat_json("role", "msg") == {"ok": 1}


def test_chat_json_marks_parse_failure(monkeypatch) -> None:
    raw = "not json at all"
    monkeypatch.setattr(llm_client, "chat", lambda *args, **kwargs: raw)
    result = llm_client.chat_json("role", "msg")
    assert result["_raw"] == raw
    assert result["_parse_error"] is True


def test_truncate_text_under_limit_is_untouched() -> None:
    text = "a" * 100
    assert llm_client.truncate_text(text, limit=512) == (text, False)


def test_truncate_text_over_limit_marks_truncation() -> None:
    text = "a" * 600
    result, truncated = llm_client.truncate_text(text, limit=512)
    assert truncated is True
    assert len(result) <= 512 + len("\n...[context truncated]...")
    assert "context truncated" in result


def _fake_client(content: str, reasoning: str):
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    completions = SimpleNamespace(create=lambda **kwargs: completion)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_chat_never_falls_back_to_reasoning_content(monkeypatch) -> None:
    # A response that carries only reasoning_content must not leak the
    # chain-of-thought into the user-visible answer; empty content stays empty.
    client = _fake_client("", "SECRET_CHAIN_OF_THOUGHT_DO_NOT_LEAK")
    monkeypatch.setattr(llm_client, "_resolve", lambda *a, **k: (client, "m"))
    assert llm_client.chat("system", "user") == ""


def test_chat_returns_content_when_present(monkeypatch) -> None:
    client = _fake_client("visible answer", "hidden reasoning")
    monkeypatch.setattr(llm_client, "_resolve", lambda *a, **k: (client, "m"))
    assert llm_client.chat("system", "user") == "visible answer"


def test_validate_base_url_rejects_slow_dns(monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_LLM_ALLOWED_HOSTS", "slow.example")
    monkeypatch.delenv("CHEMAPP_ALLOW_PRIVATE_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(llm_client, "_DNS_RESOLUTION_TIMEOUT_SECONDS", 0.1)

    def hanging_getaddrinfo(*args, **kwargs):
        import time

        time.sleep(2.0)
        return []

    monkeypatch.setattr(llm_client.socket, "getaddrinfo", hanging_getaddrinfo)
    with pytest.raises(ValueError, match="timed out"):
        llm_client._validate_client_base_url("https://slow.example")


def test_validate_base_url_wraps_gaierror(monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_LLM_ALLOWED_HOSTS", "missing.example")
    monkeypatch.delenv("CHEMAPP_ALLOW_PRIVATE_LLM_BASE_URL", raising=False)

    def failing_getaddrinfo(*args, **kwargs):
        raise _socket.gaierror("name not found")

    monkeypatch.setattr(llm_client.socket, "getaddrinfo", failing_getaddrinfo)
    with pytest.raises(ValueError, match="could not be resolved"):
        llm_client._validate_client_base_url("https://missing.example")


def test_validate_base_url_rejects_private_addresses(monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_LLM_ALLOWED_HOSTS", "private.example")
    monkeypatch.delenv("CHEMAPP_ALLOW_PRIVATE_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(
        llm_client.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("192.168.1.10", 443))],
    )
    with pytest.raises(ValueError, match="not allowed"):
        llm_client._validate_client_base_url("https://private.example")
