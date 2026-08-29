"""Shared request, sampler, timing, and output helpers for public benchmarks."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def thinking_medium_sampler() -> dict[str, object]:
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        },
    }


def endpoint(base_url: str, suffix: str) -> str:
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("--base-url must be an http(s) URL")
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def root_endpoint(base_url: str, suffix: str) -> str:
    """Address server-root endpoints when --base-url conventionally ends in /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return endpoint(root, suffix)


def request_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from benchmark endpoint") from exc


def read_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from benchmark endpoint") from exc


def stream_completion(base_url: str, payload: dict[str, object], timeout: float) -> dict[str, Any]:
    """Measure reasoning-aware TTFT, decode, visible-content, and wall rates."""
    request = urllib.request.Request(
        endpoint(base_url, "chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from benchmark endpoint") from exc

    first_delta: float | None = None
    last_delta: float | None = None
    first_visible_delta: float | None = None
    last_visible_delta: float | None = None
    usage: dict[str, Any] | None = None
    content_parts: list[str] = []
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event_data = line[6:]
            if event_data == "[DONE]":
                break
            event = json.loads(event_data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content")
                text = delta.get("content")
                has_reasoning = isinstance(reasoning, str) and bool(reasoning)
                has_content = isinstance(text, str) and bool(text)
                if has_reasoning or has_content:
                    now = time.perf_counter()
                    first_delta = now if first_delta is None else first_delta
                    last_delta = now
                if has_content:
                    content_parts.append(text)
                    first_visible_delta = (
                        now if first_visible_delta is None else first_visible_delta
                    )
                    last_visible_delta = now
    finished = time.perf_counter()
    if usage is None or first_delta is None:
        raise RuntimeError("stream ended without usage or final output")
    decode_seconds = max((last_delta or finished) - first_delta, 1e-9)
    completion_tokens = int(usage["completion_tokens"])
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    visible_tokens = max(completion_tokens - reasoning_tokens, 0)
    visible_seconds = (
        max((last_visible_delta or finished) - first_visible_delta, 1e-9)
        if first_visible_delta is not None
        else None
    )
    return {
        "content": "".join(content_parts),
        "usage": usage,
        "ttft_seconds": first_delta - started,
        "time_to_first_visible_content_seconds": (
            first_visible_delta - started if first_visible_delta is not None else None
        ),
        "total_seconds": finished - started,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": int(usage["prompt_tokens"]) / max(first_delta - started, 1e-9),
        "decode_tokens_per_second": max(completion_tokens - 1, 0) / decode_seconds,
        "visible_content_tokens_per_second": (
            max(visible_tokens - 1, 0) / visible_seconds
            if visible_seconds is not None
            else None
        ),
        "end_to_end_tokens_per_second": completion_tokens / max(finished - started, 1e-9),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bounded(value: int, *, name: str, minimum: int, maximum: int) -> int:
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value
