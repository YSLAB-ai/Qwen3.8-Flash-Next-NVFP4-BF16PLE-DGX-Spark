#!/usr/bin/env python3
"""Reject tracked artifacts that are unsafe to publish with this recipe."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MAX_FILE_BYTES = 10 * 1024 * 1024
TEST_FIXTURE_MARKER = "# publication-audit: allow-test-fixture"
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
PRIVATE_TERMS = (
    "lab" + "tools",
    "cloud" + "flare",
    "/home/" + "yiwen",
    "open" + "code",
    "pal" + "world",
    ".config/" + "systemd/user/",
)
ASSIGNMENT_PATTERN = re.compile(
    r"(?m)^\s*(?:export\s+)?[A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|ACCESS[_-]?KEY)[A-Z0-9_]*\s*[:=]"
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: Path
    detail: str


def git_tracked_files(root: Path) -> list[Path]:
    """Return only the files Git will publish from *root*."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [root / Path(name) for name in result.stdout.decode("utf-8").split("\0") if name]


def all_files(root: Path) -> list[Path]:
    """Return every non-``.git`` entry without following directory symlinks."""
    files: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            files.append(directory)
            return
        for entry in entries:
            if entry.name == ".git":
                continue
            path = Path(entry.path)
            if entry.is_symlink():
                files.append(path)
            elif entry.is_dir(follow_symlinks=False):
                visit(path)
            else:
                files.append(path)

    visit(root)
    return files


def audit_tree(root: Path, tracked_files: list[Path]) -> list[Finding]:
    """Inspect the supplied tracked files without reading ignored local state."""
    root = root.resolve()
    findings: list[Finding] = []
    for tracked_path in tracked_files:
        path = tracked_path if tracked_path.is_absolute() else root / tracked_path
        display_path = _display_path(root, path)
        resolved = path.resolve(strict=False)

        if path.is_symlink() and not _inside(root, resolved):
            findings.append(Finding("external-symlink", display_path, "symlink resolves outside audit root"))
            continue
        if not _inside(root, resolved):
            findings.append(Finding("external-file", display_path, "path resolves outside audit root"))
            continue
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            findings.append(Finding("model-weight", display_path, "model-weight file extension"))
        try:
            size = path.stat().st_size
        except OSError:
            findings.append(Finding("unreadable-file", display_path, "tracked path cannot be inspected"))
            continue
        if size > MAX_FILE_BYTES:
            findings.append(Finding("oversized-file", display_path, "tracked file exceeds 10 MiB"))
            continue
        if not path.is_file() or _is_allowed_test_fixture(root, path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            findings.append(Finding("unreadable-file", display_path, "tracked path cannot be read"))
            continue
        if ASSIGNMENT_PATTERN.search(text):
            findings.append(Finding("secret-assignment", display_path, "secret-like assignment"))
        if any(term in text.lower() for term in PRIVATE_TERMS):
            findings.append(Finding("private-term", display_path, "private endpoint term"))
    return findings


def _display_path(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_allowed_test_fixture(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if relative.parent != Path("tests") or not relative.name.startswith("test_"):
        return False
    try:
        return TEST_FIXTURE_MARKER in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository tree to audit",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="scan every non-.git file instead of only Git-tracked files",
    )
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
    except OSError:
        parser.error("--root must be an existing directory")
    if not root.is_dir():
        parser.error("--root must be an existing directory")
    if args.all_files:
        files = all_files(root)
    else:
        files = [
            path
            for path in git_tracked_files(root)
            if not path.is_relative_to(root / "docs/superpowers")
        ]
    findings = audit_tree(root, files)
    for finding in findings:
        print(f"{finding.code}: {finding.path}: {finding.detail}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
