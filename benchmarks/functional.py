#!/usr/bin/env python3
"""Run bounded factual, JSON, code, tool, and optional vision checks."""

from __future__ import annotations

import argparse
import ast
import base64
import json
import time
from pathlib import Path

from common import endpoint, request_json, thinking_medium_sampler, write_report


def _message(response: dict) -> dict:
    return response["choices"][0]["message"]


def validate(kind: str, response: dict) -> bool:
    message = _message(response)
    content = str(message.get("content") or "").strip()
    if kind == "factual":
        return content.rstrip(".") == "323"
    if kind == "json":
        return json.loads(content) == {"product": 323}
    if kind == "code":
        return _is_exact_square_function(content)
    if kind == "tool":
        calls = message.get("tool_calls") or []
        return bool(calls and calls[0].get("function", {}).get("name") == "get_weather")
    if kind == "vision":
        return json.loads(content) == {"red_squares": 3, "blue_circles": 2}
    raise ValueError(f"unknown fixture: {kind}")


def _is_exact_square_function(source: str) -> bool:
    """Validate the fixture's one safe function without executing model output."""
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
    value = function.body[0].value
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Mult)
        and isinstance(value.left, ast.Name)
        and value.left.id == "n"
        and isinstance(value.left.ctx, ast.Load)
        and isinstance(value.right, ast.Name)
        and value.right.id == "n"
        and isinstance(value.right.ctx, ast.Load)
    )


def payload(kind: str, model: str, image: Path | None) -> dict[str, object]:
    prompts = {
        "factual": "Calculate 17 multiplied by 19. Return only the digits.",
        "json": "Calculate 17 multiplied by 19. Return JSON exactly matching {\"product\": 323}.",
        "code": "Return only Python source defining square(n) that returns n * n.",
        "tool": "Use get_weather with city Boston. Do not answer from memory.",
        "vision": "Count red squares and blue circles. Return JSON with those two keys.",
    }
    message: dict[str, object] = {"role": "user", "content": prompts[kind]}
    if kind == "vision":
        if image is None:
            raise ValueError("--image is required with --include-vision")
        message["content"] = [
            {"type": "text", "text": prompts[kind]},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
        ]
    result: dict[str, object] = {"model": model, "messages": [message], "max_tokens": 512, **thinking_medium_sampler()}
    if kind == "tool":
        result["tools"] = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    if kind == "json":
        result["response_format"] = {"type": "json_object"}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--include-vision", action="store_true")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 1800:
        parser.error("--timeout must be between 0 and 1800 seconds")
    fixtures = ["factual", "json", "code", "tool"] + (["vision"] if args.include_vision else [])
    results = {}
    for kind in fixtures:
        started = time.perf_counter()
        response = request_json(endpoint(args.base_url, "chat/completions"), payload(kind, args.model, args.image), args.timeout)
        results[kind] = {"passed": validate(kind, response), "elapsed_seconds": time.perf_counter() - started, "usage": response.get("usage", {})}
    write_report(args.output, {"sampler": thinking_medium_sampler(), "results": results})
    return 0 if all(item["passed"] for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
