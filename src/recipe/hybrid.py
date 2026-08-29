"""Build audited local checkpoint views with a BF16 PLE overlay."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .audit import audit_checkpoint
from .manifest import Target
from .safetensors import DuplicateJsonKeyError, strict_json_loads


class HybridError(ValueError):
    """Raised when a hybrid checkpoint view cannot be built safely."""


_INDEX_NAME = "model.safetensors.index.json"
_METADATA_NAME = "recipe-metadata.json"
_METADATA_FIELDS = {"target", "ple_source", "index_sha256"}
_MODEL_REF_FIELDS = {"repo_id", "revision"}
_SERVING_METADATA_FILES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}")


def build_hybrid_view(
    target_snapshot: Path, source_snapshot: Path, output_root: Path, target: Target
) -> Path:
    """Build or verify a local view replacing a target's PLE with BF16 shards.

    The generated directory owns only small metadata and relative symlinks.  It is
    audited while still temporary, then atomically renamed into its final path.
    """
    _validate_hybrid_target(target)
    target_snapshot = _resolve_snapshot(
        target_snapshot, "target snapshot", target.repo_id, target.revision
    )
    source_snapshot = _resolve_snapshot(
        source_snapshot,
        "PLE source snapshot",
        target.ple_source.repo_id,
        target.ple_source.revision,
    )
    output_root = _resolve_output_root(output_root, target_snapshot, source_snapshot)
    fingerprint = _fingerprint(target)
    final = output_root / _safe_target_name(target) / fingerprint

    if _path_exists(final):
        _verify_existing_metadata(final, target)
        return final

    try:
        final.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HybridError(f"unable to create hybrid output directory: {final.parent}") from exc

    temporary = final.with_name(f".{fingerprint}.building")
    if _path_exists(temporary):
        raise HybridError(f"refusing stale hybrid build directory: {temporary}")
    try:
        temporary.mkdir()
    except FileExistsError as exc:
        raise HybridError(f"refusing stale hybrid build directory: {temporary}") from exc
    except OSError as exc:
        raise HybridError(f"unable to create hybrid build directory: {temporary}") from exc

    _build_links_and_index(temporary, target_snapshot, source_snapshot, target)
    audit_checkpoint(
        temporary,
        target,
        (_approved_root(target_snapshot), _approved_root(source_snapshot)),
    )
    if _path_exists(final):
        raise HybridError(f"hybrid view appeared during build: {final}")
    try:
        temporary.replace(final)
    except OSError as exc:
        raise HybridError(f"unable to finalize hybrid view: {final}") from exc
    return final


def _build_links_and_index(
    temporary: Path, target_snapshot: Path, source_snapshot: Path, target: Target
) -> None:
    target_weights = _load_weight_map(target_snapshot)
    source_weights = _load_weight_map(source_snapshot)
    ple_names = _expected_ple_names(target)
    scale_name = f"{_ple_prefix(target)}.weight_scale"

    if not ple_names.issubset(target_weights):
        raise HybridError("target safetensors index is missing expected PLE tensors")
    source_ple_names = {
        name for name in source_weights if name.startswith(f"{_ple_prefix(target)}.shard_")
    }
    if source_ple_names != ple_names:
        raise HybridError("PLE source safetensors index does not exactly match expected PLE tensors")

    combined = {
        name: filename
        for name, filename in target_weights.items()
        if name not in ple_names and name != scale_name
    }
    for name in ple_names:
        combined[name] = source_weights[name]

    _copy_model_metadata(target_snapshot, temporary)
    target_filenames = {filename for name, filename in combined.items() if name not in ple_names}
    source_filenames = {source_weights[name] for name in ple_names}
    target_links = _make_weight_links(temporary, target_snapshot, target_filenames, "target")
    source_links = _make_weight_links(temporary, source_snapshot, source_filenames, "bf16-ple")
    generated_map = {
        name: (source_links if name in ple_names else target_links)[filename]
        for name, filename in combined.items()
    }
    index_bytes = _json_bytes({"weight_map": generated_map})
    (temporary / _INDEX_NAME).write_bytes(index_bytes)
    metadata = {
        "target": {"repo_id": target.repo_id, "revision": target.revision},
        "ple_source": {
            "repo_id": target.ple_source.repo_id,
            "revision": target.ple_source.revision,
        },
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
    }
    (temporary / _METADATA_NAME).write_bytes(_json_bytes(metadata))


def _copy_model_metadata(snapshot: Path, destination: Path) -> None:
    for filename in sorted(_SERVING_METADATA_FILES):
        entry = snapshot / filename
        if not _path_exists(entry):
            continue
        try:
            resolved = entry.resolve(strict=True)
        except OSError as exc:
            raise HybridError(f"unable to resolve model metadata: {entry}") from exc
        if not resolved.is_file():
            raise HybridError(f"model metadata is not a regular file: {entry}")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise HybridError(f"unable to stat model metadata: {entry}") from exc
        if size > _MAX_METADATA_BYTES:
            raise HybridError(f"model metadata exceeds 16 MiB: {entry}")
        try:
            shutil.copyfile(resolved, destination / filename)
        except OSError as exc:
            raise HybridError(f"unable to copy model metadata: {entry}") from exc


def _make_weight_links(
    destination: Path, snapshot: Path, filenames: set[str], prefix: str
) -> dict[str, str]:
    links: dict[str, str] = {}
    used_names: dict[str, Path] = {}
    for filename in sorted(filenames):
        source = _resolve_weight_file(snapshot, filename)
        link_name = _link_name(prefix, filename)
        previous = used_names.get(link_name)
        if previous is not None and previous != source:
            raise HybridError(f"safetensors link-name collision for {filename}")
        used_names[link_name] = source
        link = destination / link_name
        if _path_exists(link):
            raise HybridError(f"safetensors link already exists: {link}")
        relative_target = os.path.relpath(source, start=link.parent)
        try:
            link.symlink_to(relative_target)
        except OSError as exc:
            raise HybridError(f"unable to create safetensors link: {link}") from exc
        links[filename] = link_name
    return links


def _verify_existing_metadata(final: Path, target: Target) -> None:
    if final.is_symlink() or not final.is_dir():
        raise HybridError(f"existing hybrid view is not a directory: {final}")
    metadata = _load_json_object(final / _METADATA_NAME, "hybrid recipe metadata")
    if set(metadata) != _METADATA_FIELDS:
        raise HybridError("existing hybrid metadata has unexpected fields")
    if not _matches_model_ref(metadata.get("target"), target.repo_id, target.revision):
        raise HybridError("existing hybrid metadata target identity does not match")
    if target.ple_source is None or not _matches_model_ref(
        metadata.get("ple_source"), target.ple_source.repo_id, target.ple_source.revision
    ):
        raise HybridError("existing hybrid metadata PLE source identity does not match")
    recorded_digest = metadata.get("index_sha256")
    if not isinstance(recorded_digest, str) or not _is_sha256(recorded_digest):
        raise HybridError("existing hybrid metadata index SHA-256 is invalid")
    try:
        actual_digest = hashlib.sha256((final / _INDEX_NAME).read_bytes()).hexdigest()
    except OSError as exc:
        raise HybridError("unable to read existing hybrid safetensors index") from exc
    if actual_digest != recorded_digest:
        raise HybridError("existing hybrid safetensors index digest does not match metadata")


def _load_weight_map(snapshot: Path) -> dict[str, str]:
    index = _load_json_object(snapshot / _INDEX_NAME, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise HybridError("safetensors index weight_map must be a non-empty object")
    if not all(isinstance(name, str) and isinstance(filename, str) for name, filename in weight_map.items()):
        raise HybridError("safetensors index weight_map entries must be strings")
    return weight_map


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except DuplicateJsonKeyError as exc:
        raise HybridError(str(exc)) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HybridError(f"{label} must be an object: {path}")
    return value


def _resolve_weight_file(snapshot: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
        raise HybridError(f"invalid safetensors index path: {filename}")
    try:
        resolved = (snapshot / path).resolve(strict=True)
    except OSError as exc:
        raise HybridError(f"missing safetensors shard: {filename}") from exc
    if not resolved.is_file():
        raise HybridError(f"safetensors shard is not a file: {filename}")
    return resolved


def _resolve_snapshot(path: Path, label: str, repo_id: str, revision: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise HybridError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise HybridError(f"{label} is not a directory: {path}")
    expected_cache_dir = "models--" + repo_id.replace("/", "--")
    if (
        not _REVISION.fullmatch(revision)
        or resolved.name != revision
        or resolved.parent.name != "snapshots"
        or resolved.parent.parent.name != expected_cache_dir
    ):
        raise HybridError(f"{label} does not match its pinned Hugging Face snapshot identity")
    return resolved


def _resolve_output_root(
    output_root: Path, target_snapshot: Path, source_snapshot: Path
) -> Path:
    try:
        resolved = Path(output_root).resolve(strict=False)
    except OSError as exc:
        raise HybridError(f"unable to resolve hybrid output root: {output_root}") from exc
    for snapshot in (target_snapshot, source_snapshot):
        if _is_relative_to(resolved, snapshot):
            raise HybridError("hybrid output root must not be inside an upstream snapshot")
    return resolved


def _approved_root(snapshot: Path) -> Path:
    return snapshot.parent.parent


def _validate_hybrid_target(target: Target) -> None:
    if target.mode != "hybrid_bf16" or target.ple_source is None:
        raise HybridError("hybrid view requires a hybrid_bf16 target with a PLE source")
    if target.expected_ple.dtype != "BF16":
        raise HybridError("hybrid view requires BF16 PLE expectations")


def _safe_target_name(target: Target) -> str:
    name = target.name
    if not name or Path(name).name != name or name in {".", ".."}:
        raise HybridError("hybrid target name must be one safe path component")
    return name


def _fingerprint(target: Target) -> str:
    return hashlib.sha256(
        f"{target.repo_id}@{target.revision}:{target.ple_source.repo_id}@{target.ple_source.revision}".encode()
    ).hexdigest()[:16]


def _expected_ple_names(target: Target) -> set[str]:
    return {
        f"{_ple_prefix(target)}.shard_{index}.weight"
        for index in range(target.expected_ple.tensor_count)
    }


def _ple_prefix(target: Target) -> str:
    return (
        f"model.language_model.layers.{target.expected_ple.layer_id}."
        "ple.ple_embedding.ngram_embedding"
    )


def _link_name(prefix: str, filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return f"{prefix}--{digest}.safetensors"


def _matches_model_ref(value: Any, repo_id: str, revision: str) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _MODEL_REF_FIELDS
        and value.get("repo_id") == repo_id
        and value.get("revision") == revision
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
