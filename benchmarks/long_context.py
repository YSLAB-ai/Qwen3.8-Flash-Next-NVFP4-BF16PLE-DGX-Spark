#!/usr/bin/env python3
"""Opt-in exact retrieval at a bounded long-context target."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    bounded,
    request_json,
    root_endpoint,
    stream_completion,
    thinking_medium_sampler,
    write_report,
)


NEEDLE = "MOUNTAIN-CINDER-240079"


def build_messages(repetitions: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    before = "archive " * (repetitions * 3 // 4)
    after = "appendix " * (repetitions - repetitions * 3 // 4)
    content = f"Ignore revoked marker ASH-OLD-000001.\n{before}\nCURRENT marker: {NEEDLE}\n{after}\nReturn only the CURRENT marker."
    offset = content.index(NEEDLE)
    return [{"role": "user", "content": content}], {"filler_repetitions": repetitions, "needle_character_offset": offset, "characters_after_needle": len(content) - offset - len(NEEDLE)}


def calibrate(base_url: str, model: str, target: int, timeout: float) -> tuple[list[dict[str, str]], dict[str, object]]:
    repetitions = target
    history = []
    for _ in range(5):
        messages, metadata = build_messages(repetitions)
        tokenized = request_json(root_endpoint(base_url, "tokenize"), {"model": model, "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": thinking_medium_sampler()["chat_template_kwargs"]}, timeout)
        count = int(tokenized["count"])
        history.append({"filler_repetitions": repetitions, "prompt_tokens": count})
        if target <= count <= target + 80:
            return messages, {**metadata, "tokenize_count": count, "tokenization_history": history}
        repetitions += target - count
    raise RuntimeError("failed to calibrate long-context request within five attempts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-long-context", action="store_true")
    parser.add_argument("--target-tokens", type=int, default=240_000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=21_600.0)
    args = parser.parse_args(argv)
    if not args.enable_long_context:
        parser.error("--enable-long-context is required because this benchmark is expensive")
    try:
        bounded(args.target_tokens, name="--target-tokens", minimum=240_000, maximum=240_080)
        bounded(args.max_tokens, name="--max-tokens", minimum=1, maximum=512)
    except ValueError as exc:
        parser.error(str(exc))
    messages, context = calibrate(args.base_url, args.model, args.target_tokens, args.timeout)
    result = stream_completion(args.base_url, {"model": args.model, "messages": messages, "max_tokens": args.max_tokens, "stream": True, "stream_options": {"include_usage": True}, "seed": 1234, **thinking_medium_sampler()}, args.timeout)
    answer = str(result.pop("content")).strip().rstrip(".")
    write_report(args.output, {"sampler": thinking_medium_sampler(), "context": context, "validated": answer == NEEDLE, **result})
    return 0 if answer == NEEDLE else 1


if __name__ == "__main__":
    raise SystemExit(main())
