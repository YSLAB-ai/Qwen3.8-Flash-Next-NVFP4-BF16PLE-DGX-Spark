"""Thin command-line interface for pinned checkpoint preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

from .audit import audit_checkpoint
from .download import download_target, local_target_path, login
from .manifest import ManifestError, Target, load_manifest
from .runtime import (
    RuntimeOptions,
    build_docker_command,
    served_model_alias,
    start_container,
    validate_environment,
)


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _ROOT / "hf-cache"
_DEFAULT_IMAGE = "qwen38-flash-dgx"
_VALIDATION_STATES = {
    "orca-uncensored": "runtime-validated",
    "inferact": "structure-audited",
    "radixark": "hybrid PLE preparation",
    "orca-uncensored-bf16-mtp": "experimental BF16 MTP overlay",
}


def main(
    argv: Sequence[str] | None = None,
    *,
    manifest_path: Path | None = None,
    runner: object = subprocess.run,
) -> int:
    """Run the recipe CLI without exposing credentials in rendered commands."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = load_manifest(manifest_path or _ROOT / "compatibility.json")

    if args.command == "list":
        _print_targets(manifest.targets.values())
        return 0
    if args.command == "login":
        login(Path(args.cache), args.image)
        return 0

    try:
        target = manifest.target(args.target)
    except ManifestError as exc:
        parser.error(str(exc))
    cache = Path(args.cache)
    if args.command == "path":
        print(local_target_path(target, cache))
        return 0
    if args.command == "download":
        print(download_target(target, cache, args.image))
        return 0
    if args.command == "prepare":
        model_path = download_target(target, cache, args.image)
        approved_roots = (
            (cache, cache.parent / "recipe-views")
            if target.mode in {"hybrid_bf16", "mtp_overlay"}
            else (cache,)
        )
        audit_checkpoint(model_path, target, approved_roots)
        print(model_path)
        return 0
    if args.command == "audit":
        model_path = local_target_path(target, cache)
        audit_checkpoint(model_path, target, (cache,))
        print(model_path)
        return 0
    if args.command in {"serve", "dry-run"}:
        options = RuntimeOptions(
            context=args.context,
            sequences=args.sequences,
            gpu_memory=args.gpu_memory,
            port=args.port,
            bind=args.bind,
            mtp=args.mtp,
        )
        model_path = local_target_path(target, cache)
        command = build_docker_command(
            target,
            model_path,
            cache,
            options,
            image=args.image,
            unsafe_override=args.unsafe_override,
        )
        if args.command == "dry-run":
            print(" ".join(command))
            return 0
        validate_environment(
            options,
            args.unsafe_override,
            minimum_free_bytes=target.minimum_free_bytes,
            disk_path=cache,
            allow_mtp=target.mode == "mtp_overlay",
        )
        approved_roots = (cache, cache.parent / "recipe-views")
        audit_checkpoint(model_path, target, approved_roots)
        start_container(command, target, replace=args.replace, runner=runner)
        print(f"started {served_model_alias(target)} on {options.bind}:{options.port}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned Qwen3.8 Flash-Next checkpoints")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list approved pinned targets")
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--cache", default=str(_DEFAULT_CACHE), help="local Hugging Face cache")
    options.add_argument("--image", default=_DEFAULT_IMAGE, help="image containing the hf CLI")
    commands.add_parser("login", parents=[options], help="authenticate interactively with Hugging Face")
    for command, help_text in (
        ("download", "download a pinned checkpoint"),
        ("prepare", "download and audit a pinned checkpoint"),
        ("audit", "audit an already-downloaded checkpoint"),
        ("path", "print the pinned local checkpoint path"),
    ):
        child = commands.add_parser(command, parents=[options], help=help_text)
        child.add_argument("target", help="approved manifest target alias")
    runtime_options = argparse.ArgumentParser(add_help=False)
    runtime_options.add_argument("--cache", default=str(_DEFAULT_CACHE), help="local Hugging Face cache")
    runtime_options.add_argument("--image", default=_DEFAULT_IMAGE, help="vLLM image to run")
    runtime_options.add_argument("--context", type=int, default=262_144)
    runtime_options.add_argument("--sequences", type=int, default=8)
    runtime_options.add_argument("--gpu-memory", type=float, default=0.80)
    runtime_options.add_argument("--port", type=int, default=18_300)
    runtime_options.add_argument("--bind", default="127.0.0.1")
    runtime_options.add_argument("--mtp", type=int, default=0)
    runtime_options.add_argument("--unsafe-override", action="store_true")
    for command, help_text in (
        ("serve", "audit and serve a pinned local checkpoint"),
        ("dry-run", "render a guarded Docker command without executing it"),
    ):
        child = commands.add_parser(command, parents=[runtime_options], help=help_text)
        child.add_argument("target", help="approved manifest target alias")
        if command == "serve":
            child.add_argument("--replace", action="store_true")
    return parser


def _print_targets(targets: Iterable[Target]) -> None:
    for target in sorted(targets, key=lambda item: item.name):
        state = _VALIDATION_STATES.get(target.name, "manifest-pinned")
        print(f"{target.name}\t{state}\t{target.repo_id}@{target.revision}")


if __name__ == "__main__":
    raise SystemExit(main())
