#!/usr/bin/env python3
"""Opt-in, bounded stability probes for an explicitly selected deployment."""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from common import endpoint, read_json, request_json, root_endpoint, thinking_medium_sampler, write_report


def memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.split()[0]) * 1024
    return {"mem_available_bytes": values["MemAvailable"], "swap_used_bytes": values["SwapTotal"] - values["SwapFree"]}


def run_probe(base_url: str, model: str, timeout: float) -> dict[str, object]:
    started = time.perf_counter()
    with urllib.request.urlopen(root_endpoint(base_url, "health"), timeout=timeout) as health:
        health_status = health.status
    models = read_json(endpoint(base_url, "models"), timeout)
    available = models.get("data") or []
    if len(available) != 1 or available[0].get("id") != model:
        raise AssertionError("model catalog did not match explicit --model")
    response = request_json(endpoint(base_url, "chat/completions"), {"model": model, "messages": [{"role": "user", "content": "Calculate 17 multiplied by 19. Return only the digits."}], "max_tokens": 256, "seed": 1234, **thinking_medium_sampler()}, timeout)
    answer = str(response["choices"][0]["message"].get("content") or "").strip().rstrip(".")
    return {"health_status": health_status, "passed": answer == "323", "usage": response.get("usage", {}), "latency_seconds": time.perf_counter() - started, **memory_snapshot()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-stability", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=7200.0)
    parser.add_argument("--interval-seconds", type=float, default=900.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if not args.enable_stability:
        parser.error("--enable-stability is required because this benchmark may run for two hours")
    if not 0 < args.duration_seconds <= 7200 or not 0 < args.interval_seconds <= 900 or not 0 < args.timeout <= 600:
        parser.error("duration must be <= 7200, interval <= 900, and timeout <= 600 seconds")
    started = time.monotonic()
    deadline = started + args.duration_seconds
    probes = []
    while True:
        probes.append(run_probe(args.base_url, args.model, args.timeout))
        if time.monotonic() + args.interval_seconds > deadline:
            break
        time.sleep(args.interval_seconds)
    write_report(args.output, {"sampler": thinking_medium_sampler(), "requested_duration_seconds": args.duration_seconds, "interval_seconds": args.interval_seconds, "elapsed_seconds": time.monotonic() - started, "probes": probes})
    return 0 if all(probe["passed"] for probe in probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
