#!/usr/bin/env python3
"""Opt-in isolated-marker throughput waves for 1, 2, 4, or 8 streams."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import bounded, stream_completion, thinking_medium_sampler, write_report


MARKERS = tuple(f"EMBER-{index:02d}-4100{index:02d}" for index in range(1, 9))


def build_messages(marker: str, repetitions: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    distractor = marker.replace("EMBER", "ASH")
    before = "ledger " * (repetitions * 3 // 4)
    after = "appendix " * (repetitions - repetitions * 3 // 4)
    content = f"Do not return revoked marker {distractor}.\n{before}\nCURRENT marker: {marker}\n{after}\nReturn only the CURRENT marker."
    offset = content.index(marker)
    return [{"role": "user", "content": content}], {"filler_repetitions": repetitions, "marker_character_offset": offset, "characters_after_marker": len(content) - offset - len(marker)}


def run_one(base_url: str, model: str, marker: str, target_tokens: int, max_tokens: int, timeout: float) -> dict[str, object]:
    messages, context = build_messages(marker, target_tokens)
    result = stream_completion(base_url, {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}, "seed": 1234, **thinking_medium_sampler()}, timeout)
    result["correct"] = str(result.pop("content")).strip().rstrip(".") == marker
    result["context"] = context
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-concurrency", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if not args.enable_concurrency:
        parser.error("--enable-concurrency is required because this benchmark sends parallel work")
    try:
        bounded(args.target_tokens, name="--target-tokens", minimum=1024, maximum=8192)
        bounded(args.max_tokens, name="--max-tokens", minimum=1, maximum=512)
    except ValueError as exc:
        parser.error(str(exc))
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, args.base_url, args.model, marker, args.target_tokens, args.max_tokens, args.timeout) for marker in MARKERS[:args.concurrency]]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - started
    completion_tokens = sum(int(result["usage"]["completion_tokens"]) for result in results)
    write_report(args.output, {"sampler": thinking_medium_sampler(), "concurrency": args.concurrency, "target_tokens": args.target_tokens, "max_tokens": args.max_tokens, "wall_seconds": wall_seconds, "aggregate_completion_tokens_per_second": completion_tokens / wall_seconds, "results": results})
    return 0 if all(result["correct"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
