"""Thin command-line interface for pinned checkpoint preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from .audit import audit_checkpoint
from .download import download_target, local_target_path, login
from .manifest import ManifestError, Target, load_manifest


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _ROOT / "hf-cache"
_DEFAULT_IMAGE = "qwen38-flash-dgx"
_VALIDATION_STATES = {
    "orca-uncensored": "runtime-validated",
    "inferact": "structure-audited",
    "radixark": "hybrid PLE preparation",
}


def main(argv: Sequence[str] | None = None, *, manifest_path: Path | None = None) -> int:
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
        audit_checkpoint(model_path, target, (cache,))
        print(model_path)
        return 0
    if args.command == "audit":
        model_path = local_target_path(target, cache)
        audit_checkpoint(model_path, target, (cache,))
        print(model_path)
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
    return parser


def _print_targets(targets: Iterable[Target]) -> None:
    for target in sorted(targets, key=lambda item: item.name):
        state = _VALIDATION_STATES.get(target.name, "manifest-pinned")
        print(f"{target.name}\t{state}\t{target.repo_id}@{target.revision}")


if __name__ == "__main__":
    raise SystemExit(main())
