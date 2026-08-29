#!/usr/bin/env python3
"""Project benchmark evidence onto a small, public, reproducible schema."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PUBLIC_MODEL = "qwen3.8-flash-next-uncensored-nvfp4-bf16-ple-vllm"
PUBLIC_REPOSITORY = "orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4"
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
_CONCURRENCY_MARKERS = {
    "EMBER-ALPHA-410001",
    "EMBER-BRAVO-410002",
    "EMBER-CHARLIE-410003",
    "EMBER-DELTA-410004",
    "EMBER-ECHO-410005",
    "EMBER-FOXTROT-410006",
    "EMBER-GOLF-410007",
    "EMBER-HOTEL-410008",
}
_LONG_CONTEXT_EXPECTED = "ORANGE-COBALT-731946"
_FORBIDDEN_TEXT = re.compile(
    r"https?://|/home/|authorization|bearer|api.?key|token|secret|"
    r"reasoning|content|response|request|deployment|header|path|url",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Provenance:
    model: str
    repository: str
    revision: str
    sampler: dict[str, object]


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


def _expected_provenance() -> Provenance:
    return Provenance(
        model=PUBLIC_MODEL,
        repository=PUBLIC_REPOSITORY,
        revision=PUBLIC_REVISION,
        sampler=thinking_medium_sampler(),
    )


def _load_provenance(summary: Path) -> Provenance:
    try:
        raw = json.loads(summary.read_text(encoding="utf-8"))
        checkpoint = raw["checkpoint"]
        profile = raw["selected_profile"]
        model = raw["model"]
        repository = checkpoint["repository"]
        revision = checkpoint["revision"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("trusted experiment summary is missing provenance") from exc
    if model != PUBLIC_MODEL:
        raise ValueError("trusted experiment model does not match the publication target")
    if repository != PUBLIC_REPOSITORY:
        raise ValueError("trusted experiment repository does not match the publication target")
    if revision != PUBLIC_REVISION or not _REVISION.fullmatch(str(revision)):
        raise ValueError("trusted experiment revision does not match the publication target")
    if not isinstance(profile, dict):
        raise ValueError("trusted experiment runtime profile is invalid")
    sampler = {
        "temperature": profile.get("temperature"),
        "top_p": profile.get("top_p"),
        "top_k": profile.get("top_k"),
        "min_p": profile.get("min_p"),
        "presence_penalty": profile.get("presence_penalty"),
        "repetition_penalty": profile.get("repetition_penalty"),
        "chat_template_kwargs": {
            "enable_thinking": profile.get("thinking"),
            "reasoning_effort": profile.get("reasoning_effort"),
        },
    }
    if not _same_json_value(sampler, thinking_medium_sampler()):
        raise ValueError("trusted experiment sampler profile is not the publication sampler")
    if (
        profile.get("mtp_depth") != 0
        or profile.get("ple_mmap") is not True
        or profile.get("ple_prewarm") is not False
    ):
        raise ValueError("trusted experiment runtime profile is not the publication profile")
    return Provenance(model, repository, revision, sampler)


def _validate_source_provenance(raw: object, provenance: Provenance) -> None:
    if not isinstance(raw, dict):
        raise ValueError("result must be a JSON object")
    if raw.get("model") != provenance.model:
        raise ValueError("raw artifact model does not match trusted experiment provenance")
    _sampler(raw.get("sampler"), provenance)


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


def _sampler(value: object, provenance: Provenance) -> dict[str, object]:
    """Retain the trusted sampler only when the raw artifact agrees exactly."""
    if not _same_json_value(value, provenance.sampler):
        raise ValueError("raw artifact sampler does not match the trusted experiment profile")
    return provenance.sampler


def _same_json_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


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
        item.pop("validated", None)
        if isinstance(result, dict) and _has_response_content(result):
            verdict = _functional_verdict(fixture, result)
            source_verdict = result.get("passed")
            if isinstance(source_verdict, bool) and source_verdict != verdict:
                raise ValueError(f"functional {fixture} boolean verdict disagrees with response")
            item["passed"] = verdict
        elif isinstance(result, dict) and isinstance(result.get("passed"), bool):
            item["passed"] = result["passed"]
        elif isinstance(result, dict) and isinstance(
            result.get("validated"), (str, dict)
        ):
            raise ValueError(f"functional {fixture} verdict is unverifiable")
        if item:
            cleaned[fixture] = item
    return cleaned


def _concurrency_result(value: object) -> dict[str, Any]:
    """Retain a per-stream isolation verdict without retaining the marker text."""
    cleaned = _measurement_mapping(value)
    cleaned.pop("passed", None)
    cleaned.pop("validated", None)
    if not isinstance(value, dict) or "validated" not in value:
        return cleaned
    verdict = value["validated"]
    if isinstance(verdict, bool):
        cleaned["correct"] = verdict
    elif isinstance(verdict, (str, dict)):
        marker = value.get("marker")
        content = value.get("content_text")
        if marker not in _CONCURRENCY_MARKERS or not isinstance(content, str):
            raise ValueError("concurrency verdict is unverifiable")
        if verdict != marker:
            raise ValueError("concurrency validated marker does not match its fixture")
        cleaned["correct"] = content.strip().rstrip(".") == marker
    return cleaned


def _stability_probe(value: object) -> dict[str, Any]:
    cleaned = _measurement_mapping(value)
    if isinstance(value, dict) and isinstance(value.get("answer"), str):
        cleaned["passed"] = value["answer"].strip().rstrip(".") == "323"
    return cleaned


def _has_response_content(value: dict[str, object]) -> bool:
    try:
        message = value["response"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False
    return isinstance(message, dict)


def _functional_verdict(fixture: str, value: dict[str, object]) -> bool:
    try:
        message = value["response"]["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(f"functional {fixture} response is unverifiable") from exc
    if fixture == "factual":
        return content.rstrip(".") == "323"
    if fixture == "json":
        try:
            return json.loads(content) == {"product": 323}
        except (json.JSONDecodeError, TypeError):
            return False
    if fixture == "code":
        return _is_exact_square_function(content)
    if fixture == "tool":
        try:
            function = message["tool_calls"][0]["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return function.get("name") == "get_weather" and arguments == {"city": "Boston"}
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return False
    if fixture == "vision":
        try:
            return json.loads(content) == {"red_squares": 3, "blue_circles": 2}
        except (json.JSONDecodeError, TypeError):
            return False
    raise ValueError(f"unknown functional fixture: {fixture}")


def _is_exact_square_function(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    function = tree.body[0]
    arguments = function.args
    if (
        function.name != "square"
        or function.decorator_list
        or function.returns is not None
        or getattr(function, "type_params", ())
        or arguments.posonlyargs
        or len(arguments.args) != 1
        or arguments.args[0].arg != "n"
        or arguments.args[0].annotation is not None
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kw_defaults
        or arguments.kwarg is not None
        or arguments.defaults
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
    ):
        return False
    expression = function.body[0].value
    return (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Mult)
        and isinstance(expression.left, ast.Name)
        and expression.left.id == "n"
        and isinstance(expression.right, ast.Name)
        and expression.right.id == "n"
    )


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


def sanitize_data(
    raw: object, provenance: Provenance | None = None
) -> dict[str, object]:
    """Return only the safe, measurement-oriented projection of one result."""
    if not isinstance(raw, dict):
        raise ValueError("result must be a JSON object")
    provenance = provenance or _expected_provenance()

    clean: dict[str, object] = {"public_revision": provenance.revision}
    scalar = _measurement_mapping(raw)
    clean.update(scalar)

    raw_verdict = raw.get("validated")
    if isinstance(raw_verdict, (str, dict)):
        if not isinstance(raw.get("context"), dict) or not _has_response_content(raw):
            raise ValueError("string or dict verdict is unverifiable")
        try:
            content = raw["response"]["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError("long-context verdict is unverifiable") from exc
        verdict = str(content).strip().rstrip(".") == _LONG_CONTEXT_EXPECTED
        clean["validated"] = verdict
        clean["passed"] = verdict

    sampler = raw.get("sampler")
    if sampler is not None:
        clean["sampler"] = _sampler(sampler, provenance)
    for key in ("context",):
        item = _context(raw.get(key))
        if item:
            clean[key] = item
    for key in ("samples", "results", "probes"):
        value = raw.get(key)
        if key == "results" and isinstance(value, dict):
            item = _functional_results(value)
        elif isinstance(value, list):
            if key == "results":
                mapper = _concurrency_result
            elif key == "probes":
                mapper = _stability_probe
            else:
                mapper = _measurement_mapping
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


def sanitize_result(
    source: Path, destination: Path, *, summary: Path | None = None
) -> dict[str, object]:
    """Sanitize ``source`` and write deterministic public JSON to ``destination``."""
    source = Path(source)
    summary = Path(summary) if summary is not None else source.parent / "summary.json"
    provenance = _load_provenance(summary)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if source.resolve(strict=False) != summary.resolve(strict=False):
        _validate_source_provenance(raw, provenance)
    clean = sanitize_data(raw, provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return clean


def verify_directory(directory: Path) -> list[str]:
    """Return one diagnostic per retained file that is not a sanitized projection."""
    findings: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data != sanitize_data(data, _expected_provenance()):
                findings.append(f"{path}: contains fields outside the public allowlist")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(f"{path}: {type(exc).__name__}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--summary", type=Path, help="trusted raw experiment summary")
    parser.add_argument("--verify", type=Path, metavar="DIRECTORY")
    args = parser.parse_args(argv)
    if args.verify:
        if args.source or args.destination or args.summary:
            parser.error("--verify cannot be combined with source, destination, or --summary")
        findings = verify_directory(args.verify)
        if findings:
            print("\n".join(findings), file=sys.stderr)
            return 1
        print(f"sanitized evidence: OK ({len(list(args.verify.glob('*.json')))} files)")
        return 0
    if args.source is None or args.destination is None:
        parser.error("source and destination are required unless --verify is used")
    sanitize_result(args.source, args.destination, summary=args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
