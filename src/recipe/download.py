"""Safely download pinned Hugging Face snapshots through Docker."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .hybrid import build_hybrid_view
from .manifest import ModelRef, Target
from .safetensors import DuplicateJsonKeyError, strict_json_loads


class DownloadError(ValueError):
    """Raised when a pinned checkpoint cannot be prepared safely."""


_REVISION = re.compile(r"[0-9a-f]{40}")
_SOURCE_INDEX = "model.safetensors.index.json"


def build_hf_download_command(
    ref: ModelRef, filenames: Sequence[str], cache: Path, image: str
) -> list[str]:
    """Return a Docker argv list for a revision-pinned ``hf download`` invocation."""
    _validate_ref(ref)
    _validate_image(image)
    _validate_filenames(filenames)
    command = _docker_prefix(Path(cache).resolve(), image)
    command.extend(["download", ref.repo_id])
    command.extend(filenames)
    command.extend(["--revision", ref.revision])
    return command


def build_hf_login_command(cache: Path, image: str) -> list[str]:
    """Return a Docker argv list for interactive Hugging Face authentication."""
    _validate_image(image)
    command = _docker_prefix(Path(cache).resolve(), image)
    command.insert(3, "-it")
    command.extend(["auth", "login"])
    return command


def login(
    cache: Path, image: str, runner: Callable[..., object] = subprocess.run
) -> None:
    """Run interactive authentication using a writable mounted cache, not a token argv."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    runner(build_hf_login_command(cache, image), check=True)


def download_target(
    target: Target,
    cache: Path,
    image: str,
    runner: Callable[..., object] = subprocess.run,
) -> Path:
    """Download a pinned target and return its local snapshot or audited hybrid view."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    _require_free_space(cache, target.minimum_free_bytes)
    target_ref = ModelRef(target.repo_id, target.revision)
    runner(build_hf_download_command(target_ref, (), cache, image), check=True)
    target_snapshot = snapshot_path(cache, target_ref)
    if target.mode == "direct_bf16":
        return target_snapshot
    if target.mode != "hybrid_bf16" or target.ple_source is None:
        raise DownloadError(f"unsupported target mode: {target.mode}")

    source = target.ple_source
    runner(build_hf_download_command(source, (_SOURCE_INDEX,), cache, image), check=True)
    source_snapshot = snapshot_path(cache, source)
    filenames = _source_ple_filenames(source_snapshot, target)
    runner(build_hf_download_command(source, filenames, cache, image), check=True)
    return build_hybrid_view(target_snapshot, source_snapshot, cache.parent / "recipe-views", target)


def local_target_path(target: Target, cache: Path) -> Path:
    """Return the deterministic local path a target will occupy after preparation."""
    cache = Path(cache)
    if target.mode == "direct_bf16":
        return snapshot_path(cache, ModelRef(target.repo_id, target.revision))
    if target.mode == "hybrid_bf16" and target.ple_source is not None:
        identity = (
            f"{target.repo_id}@{target.revision}:"
            f"{target.ple_source.repo_id}@{target.ple_source.revision}"
        )
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return cache.parent / "recipe-views" / target.name / fingerprint
    raise DownloadError(f"unsupported target mode: {target.mode}")


def snapshot_path(cache: Path, ref: ModelRef) -> Path:
    """Return the canonical Hugging Face cache snapshot path for a pinned model."""
    _validate_ref(ref)
    return (
        Path(cache)
        / "hub"
        / f"models--{ref.repo_id.replace('/', '--')}"
        / "snapshots"
        / ref.revision
    )


def _docker_prefix(cache: Path, image: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-e",
        "HF_HOME=/hf",
        "-v",
        f"{cache}:/hf:rw",
        "--entrypoint",
        "hf",
        image,
    ]


def _source_ple_filenames(source_snapshot: Path, target: Target) -> tuple[str, ...]:
    try:
        index = strict_json_loads(
            (source_snapshot / _SOURCE_INDEX).read_text(encoding="utf-8")
        )
        weight_map = index["weight_map"]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJsonKeyError,
        KeyError,
        TypeError,
    ) as exc:
        raise DownloadError("unable to read the official PLE safetensors index") from exc
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise DownloadError("official PLE safetensors index has an invalid weight map")

    prefix = (
        f"model.language_model.layers.{target.expected_ple.layer_id}."
        "ple.ple_embedding.ngram_embedding"
    )
    expected = {
        f"{prefix}.shard_{index}.weight"
        for index in range(target.expected_ple.tensor_count)
    }
    found = {name for name in weight_map if name.startswith(f"{prefix}.shard_")}
    if found != expected:
        raise DownloadError("official PLE index does not contain exactly the expected PLE names")
    filenames = tuple(sorted({weight_map[name] for name in expected}))
    _validate_filenames(filenames)
    if not all(filename.endswith(".safetensors") for filename in filenames):
        raise DownloadError("official PLE index references a non-safetensors shard")
    return filenames


def _validate_ref(ref: ModelRef) -> None:
    if not _REVISION.fullmatch(ref.revision):
        raise DownloadError("Hugging Face revision must be a full lowercase commit SHA")
    repository = Path(ref.repo_id)
    if (
        not ref.repo_id
        or ref.repo_id.count("/") != 1
        or repository.is_absolute()
        or any(part in {"", ".", ".."} for part in repository.parts)
    ):
        raise DownloadError("Hugging Face repository identifier is invalid")


def _validate_filenames(filenames: Sequence[str]) -> None:
    for filename in filenames:
        if not isinstance(filename, str) or not filename:
            raise DownloadError("Hugging Face filename is invalid")
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts:
            raise DownloadError("Hugging Face filename is invalid")


def _validate_image(image: str) -> None:
    if not isinstance(image, str) or not image:
        raise DownloadError("Docker image is required")


def _require_free_space(cache: Path, minimum_free_bytes: int) -> None:
    try:
        free_bytes = shutil.disk_usage(cache).free
    except OSError as exc:
        raise DownloadError("unable to inspect free disk space on the download cache") from exc
    if free_bytes < minimum_free_bytes:
        raise DownloadError(
            f"free disk {free_bytes} is below required {minimum_free_bytes} bytes"
        )
