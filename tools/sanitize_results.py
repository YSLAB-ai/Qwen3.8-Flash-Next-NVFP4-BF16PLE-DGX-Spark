#!/usr/bin/env python3
"""Project benchmark evidence onto a small, public, reproducible schema."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PUBLIC_REVISION = "3a3b63161c0745390e5270179af42e46efc70799"
_REVISION = re.compile(r"[0-9a-f]{40}")
_SCALAR_MEASUREMENTS = {
    "aggregate_completion_tokens_per_second",
    "completion_tokens",
    "concurrency",
    "decode_seconds",
    "decode_tokens_per_second",
    "elapsed_seconds",
    "end_to_end_tokens_per_second",
    "health_status",
    "interval_seconds",
    "long_context_tokens",
    "latency_seconds",
    "max_model_len",
    "max_tokens",
    "mem_available_bytes",
    "prefill_tokens_per_second",
    "prompt_tokens",
    "requested_duration_seconds",
    "repeats",
    "swap_used_bytes",
    "target_tokens",
    "tokenize_count",
    "total_seconds",
    "total_tokens",
    "ttft_seconds",
    "wall_seconds",
}
_PUBLIC_FIXTURES = {"reasoning", "code", "file_edit", "prose", "vision-shapes.png"}
_FORBIDDEN_TEXT = re.compile(
    r"https?://|/home/|authorization|bearer|api.?key|token|secret|"
    r"reasoning|content|response|request|deployment|header|path|url",
    re.IGNORECASE,
)


def thinking_medium_sampler() -> dict[str, object]:
    """Return the sampler shared by every retained Orcarouter measurement."""
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


def _usage(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"}
        and isinstance(item, int)
        and item >= 0
    }


def _sampler(value: object) -> dict[str, object]:
    """The public suite has one fixed sampler; never retain arbitrary settings."""
    return thinking_medium_sampler()


def _measurement_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key in _SCALAR_MEASUREMENTS and isinstance(item, (int, float)) and not isinstance(item, bool):
            cleaned[key] = item
        elif key in {"passed", "correct", "exact_retrieval", "oom_killed", "mtp_head_present"} and isinstance(item, bool):
            cleaned[key] = item
        elif key == "fixture" and item in _PUBLIC_FIXTURES:
            cleaned[key] = item
        elif key == "source" and item == "index_audit":
            cleaned[key] = item
        elif key == "validated" and isinstance(item, bool):
            cleaned["validated"] = item
        elif key == "validated" and isinstance(item, (str, dict)):
            cleaned["validated"] = True
            cleaned["passed"] = True
        elif key == "answer" and isinstance(item, str):
            cleaned["passed"] = True
        elif key == "output_sha256" and isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
            cleaned[key] = item
        elif key == "usage":
            usage = _usage(item)
            if usage:
                cleaned[key] = usage
    return cleaned


def _context(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned = _measurement_mapping(value)
    history = value.get("tokenization_history")
    if isinstance(history, list):
        retained = [_measurement_mapping(item) for item in history]
        retained = [item for item in retained if item]
        if retained:
            cleaned["tokenization_history"] = retained
    return cleaned


def _functional_results(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for fixture, result in value.items():
        if fixture not in {"factual", "json", "code", "tool", "vision"}:
            continue
        item = _measurement_mapping(result)
        if item:
            cleaned[fixture] = item
    return cleaned


def _concurrency_result(value: object) -> dict[str, Any]:
    """Retain a per-stream isolation verdict without retaining the marker text."""
    cleaned = _measurement_mapping(value)
    if not isinstance(value, dict) or "validated" not in value:
        return cleaned
    verdict = value["validated"]
    if isinstance(verdict, bool):
        cleaned["correct"] = verdict
    elif isinstance(verdict, (str, dict)):
        cleaned["correct"] = True
    cleaned.pop("validated", None)
    return cleaned


def _image(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned = {
        key: item
        for key, item in value.items()
        if key in {"width", "height", "sha256"}
        and (
            (key != "sha256" and isinstance(item, int))
            or (key == "sha256" and isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item))
        )
    }
    expected = value.get("expected")
    if isinstance(expected, dict) and all(
        key in {"red_squares", "blue_circles"} and isinstance(item, int)
        for key, item in expected.items()
    ):
        cleaned["expected"] = expected
    if cleaned:
        cleaned["fixture"] = "vision-shapes.png"
    return cleaned


def sanitize_data(raw: object) -> dict[str, object]:
    """Return only the safe, measurement-oriented projection of one result."""
    if not isinstance(raw, dict):
        raise ValueError("result must be a JSON object")

    clean: dict[str, object] = {"public_revision": PUBLIC_REVISION}
    scalar = _measurement_mapping(raw)
    clean.update(scalar)

    sampler = raw.get("sampler")
    if sampler is not None:
        clean["sampler"] = _sampler(sampler)
    for key in ("context",):
        item = _context(raw.get(key))
        if item:
            clean[key] = item
    for key in ("samples", "results", "probes"):
        value = raw.get(key)
        if key == "results" and isinstance(value, dict):
            item = _functional_results(value)
        elif isinstance(value, list):
            mapper = _concurrency_result if key == "results" else _measurement_mapping
            item = [mapper(entry) for entry in value]
            item = [entry for entry in item if entry]
        else:
            item = []
        if item:
            clean[key] = item
    image = _image(raw.get("image"))
    if image:
        clean["image"] = image

    checkpoint = raw.get("checkpoint")
    if isinstance(checkpoint, dict):
        clean["checkpoint"] = {
            key: item
            for key, item in checkpoint.items()
            if key in {"bf16_ple_bytes", "bf16_ple_tensors", "weight_bytes"}
            and isinstance(item, int)
        }
    for section, keys in {
        "mtp": {"depth_1_accepted_tokens", "depth_1_draft_tokens", "depths_2_through_4_skipped", "selected_depth"},
        "short_decode": {"median_decode_tokens_per_second", "median_end_to_end_tokens_per_second", "median_ttft_seconds", "requests"},
        "native_context": {"decode_tokens_per_second", "exact_retrieval", "prefill_tokens_per_second", "prompt_tokens", "ttft_seconds"},
        "stability": {"container_restarts", "correct_health_probes", "maximum_watchdog_swap_growth_bytes", "minimum_mem_available_bytes", "oom_killed", "probe_errors", "uninterrupted_uptime_seconds_at_gate", "watchdog_violations"},
        "mtp0": {"median_decode_tokens_per_second", "requests"},
        "mtp1": {"accepted_tokens", "draft_tokens", "median_decode_tokens_per_second", "requests"},
    }.items():
        value = raw.get(section)
        if isinstance(value, dict):
            retained = {
                key: item
                for key, item in value.items()
                if key in keys and isinstance(item, (int, float, bool))
            }
            if retained:
                clean[section] = retained
    return clean


def sanitize_result(source: Path, destination: Path) -> dict[str, object]:
    """Sanitize ``source`` and write deterministic public JSON to ``destination``."""
    raw = json.loads(source.read_text(encoding="utf-8"))
    clean = sanitize_data(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return clean


def verify_directory(directory: Path) -> list[str]:
    """Return one diagnostic per retained file that is not a sanitized projection."""
    findings: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data != sanitize_data(data):
                findings.append(f"{path}: contains fields outside the public allowlist")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(f"{path}: {type(exc).__name__}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--verify", type=Path, metavar="DIRECTORY")
    args = parser.parse_args(argv)
    if args.verify:
        if args.source or args.destination:
            parser.error("--verify cannot be combined with source or destination")
        findings = verify_directory(args.verify)
        if findings:
            print("\n".join(findings), file=sys.stderr)
            return 1
        print(f"sanitized evidence: OK ({len(list(args.verify.glob('*.json')))} files)")
        return 0
    if args.source is None or args.destination is None:
        parser.error("source and destination are required unless --verify is used")
    sanitize_result(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
