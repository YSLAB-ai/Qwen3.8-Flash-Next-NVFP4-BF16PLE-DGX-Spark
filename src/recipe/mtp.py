"""Build an immutable Orcarouter checkpoint view with the native BF16 MTP head."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .manifest import Target
from .safetensors import (
    DuplicateJsonKeyError,
    SafetensorsError,
    read_header,
    strict_json_loads,
    write_subset,
)


class MtpOverlayError(ValueError):
    """Raised when the native MTP overlay cannot be built exactly and safely."""


MTP_TENSOR_NAMES = (
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


# Compressed-tensors treats ordinary ignore entries as exact module names.  In
# particular, strings such as ``mtp.*`` are not globs.  These are the modules
# backed by the native BF16 MTP tensors.  vLLM remaps the layer-zero paths to
# the standalone draft layer index (currently layer 48) before construction.
MTP_QUANTIZATION_IGNORE = (
    "mtp.fc_embedding",
    "mtp.fc_hidden",
    "mtp.layers.0.mlp.experts",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj",
    "mtp.layers.0.mlp.shared_expert.up_proj",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.self_attn.indexer.index_qk_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
)


_INDEX_NAME = "model.safetensors.index.json"
_METADATA_NAME = "recipe-metadata.json"
_COMPACT_NAME = "mtp-bf16.safetensors"
_REVISION = re.compile(r"[0-9a-f]{40}")
_SERVING_METADATA_FILES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
_MAX_METADATA_BYTES = 32 * 1024 * 1024
_LAYOUT_VERSION = "bf16-mtp-v2"


def mtp_overlay_fingerprint(target: Target) -> str:
    """Return the single canonical identity for the current overlay layout."""
    source = target.mtp_source
    if target.mode != "mtp_overlay" or source is None:
        raise MtpOverlayError("MTP overlay fingerprint requires an mtp_overlay target")
    identity = (
        f"{target.repo_id}@{target.revision}:"
        f"{source.repo_id}@{source.revision}:{_LAYOUT_VERSION}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build_mtp_overlay(
    target_snapshot: Path,
    source_snapshot: Path,
    output_root: Path,
    target: Target,
) -> Path:
    """Build or verify the deterministic local BF16 MTP overlay."""
    _validate_target(target)
    target_snapshot = _resolve_snapshot(
        target_snapshot, "target snapshot", target.repo_id, target.revision
    )
    source = target.mtp_source
    assert source is not None
    source_snapshot = _resolve_snapshot(
        source_snapshot, "MTP source snapshot", source.repo_id, source.revision
    )
    output_root = _resolve_output_root(output_root, target_snapshot, source_snapshot)

    target_index = _load_index(target_snapshot)
    source_index = _load_index(source_snapshot)
    _validate_source_shards(source_snapshot, target)
    _validate_weight_maps(target_index, source_index, target)

    fingerprint = mtp_overlay_fingerprint(target)
    final = output_root / _safe_name(target.name) / fingerprint
    if _path_exists(final):
        _verify_existing(final, target, target_snapshot, output_root)
        return final

    try:
        final.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MtpOverlayError(f"unable to create MTP overlay directory: {final.parent}") from exc
    temporary = final.with_name(f".{fingerprint}.building")
    if _path_exists(temporary):
        raise MtpOverlayError(f"refusing stale MTP overlay build directory: {temporary}")
    try:
        temporary.mkdir()
    except OSError as exc:
        raise MtpOverlayError(f"unable to create MTP overlay build directory: {temporary}") from exc

    _build(
        temporary,
        target_snapshot,
        source_snapshot,
        target_index,
        source_index,
        target,
    )
    _audit(temporary, target, target_snapshot, output_root)
    if _path_exists(final):
        raise MtpOverlayError(f"MTP overlay appeared during build: {final}")
    try:
        temporary.replace(final)
    except OSError as exc:
        raise MtpOverlayError(f"unable to finalize MTP overlay: {final}") from exc
    return final


def _build(
    destination: Path,
    target_snapshot: Path,
    source_snapshot: Path,
    target_index: dict[str, Any],
    source_index: dict[str, Any],
    target: Target,
) -> None:
    target_weights = target_index["weight_map"]
    source_weights = source_index["weight_map"]
    _copy_metadata(target_snapshot, destination)
    config_bytes = _overlay_config(target_snapshot / "config.json")
    (destination / "config.json").write_bytes(config_bytes)

    target_links = _link_target_shards(
        destination, target_snapshot, set(target_weights.values())
    )
    entries = tuple(
        (name, _resolve_weight_file(source_snapshot, source_weights[name]))
        for name in MTP_TENSOR_NAMES
    )
    try:
        subset = write_subset(
            entries,
            destination / _COMPACT_NAME,
            expected_dtype=target.mtp_source.dtype,
        )
    except SafetensorsError as exc:
        raise MtpOverlayError(str(exc)) from exc

    combined = {name: target_links[filename] for name, filename in target_weights.items()}
    combined.update({name: _COMPACT_NAME for name in MTP_TENSOR_NAMES})
    metadata = target_index.get("metadata")
    output_index: dict[str, Any] = {"weight_map": combined}
    if isinstance(metadata, dict):
        output_index["metadata"] = dict(metadata)
        original_total = output_index["metadata"].get("total_size")
        if isinstance(original_total, int) and not isinstance(original_total, bool):
            compact_header = read_header(destination / _COMPACT_NAME)
            output_index["metadata"]["total_size"] = original_total + sum(
                meta.data_end - meta.data_start for meta in compact_header.values()
            )
    index_bytes = _json_bytes(output_index)
    (destination / _INDEX_NAME).write_bytes(index_bytes)

    source = target.mtp_source
    assert source is not None
    compact_header = read_header(destination / _COMPACT_NAME)
    source_shards = {shard.filename: shard for shard in source.shards}
    mtp_tensors = {
        name: {
            "dtype": compact_header[name].dtype,
            "shape": list(compact_header[name].shape),
            "bytes": compact_header[name].data_end - compact_header[name].data_start,
            "source_file": source_weights[name],
        }
        for name in MTP_TENSOR_NAMES
    }
    recipe_metadata = {
        "schema_version": 1,
        "target": {"repo_id": target.repo_id, "revision": target.revision},
        "mtp_source": {"repo_id": source.repo_id, "revision": source.revision},
        "source_shards": [
            {
                "filename": filename,
                "size": source_shards[filename].size,
                "sha256": source_shards[filename].sha256,
            }
            for filename in sorted(source_shards)
        ],
        "mtp_tensor_names": list(MTP_TENSOR_NAMES),
        "mtp_tensors": mtp_tensors,
        "compact_shard": {
            "filename": _COMPACT_NAME,
            "size": subset.size,
            "sha256": subset.sha256,
        },
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    (destination / _METADATA_NAME).write_bytes(_json_bytes(recipe_metadata))


def _overlay_config(path: Path) -> bytes:
    config = _load_json(path, "target config")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise MtpOverlayError("target config has no quantization_config object")
    ignore = quantization.get("ignore")
    if not isinstance(ignore, list) or not all(isinstance(item, str) for item in ignore):
        raise MtpOverlayError("target quantization ignore list is invalid")
    obsolete = {"mtp.*", "model.mtp.*"}
    if any(item in ignore for item in (*MTP_QUANTIZATION_IGNORE, *obsolete)):
        raise MtpOverlayError("target config already contains an MTP quantization ignore")
    quantization["ignore"] = [*ignore, *MTP_QUANTIZATION_IGNORE]
    return _json_bytes(config)


def _copy_metadata(source: Path, destination: Path) -> None:
    for filename in sorted(_SERVING_METADATA_FILES):
        entry = source / filename
        if not _path_exists(entry):
            continue
        try:
            resolved = entry.resolve(strict=True)
            size = resolved.stat().st_size
        except OSError as exc:
            raise MtpOverlayError(f"unable to inspect serving metadata: {entry}") from exc
        if not resolved.is_file() or size > _MAX_METADATA_BYTES:
            raise MtpOverlayError(f"invalid or oversized serving metadata: {entry}")
        try:
            shutil.copyfile(resolved, destination / filename)
        except OSError as exc:
            raise MtpOverlayError(f"unable to copy serving metadata: {entry}") from exc


def _link_target_shards(
    destination: Path, snapshot: Path, filenames: set[str]
) -> dict[str, str]:
    links: dict[str, str] = {}
    for filename in sorted(filenames):
        source = _resolve_weight_file(snapshot, filename)
        link_name = f"target--{hashlib.sha256(filename.encode()).hexdigest()}.safetensors"
        link = destination / link_name
        if _path_exists(link):
            raise MtpOverlayError(f"target shard link already exists: {link_name}")
        try:
            link.symlink_to(os.path.relpath(source, start=destination))
        except OSError as exc:
            raise MtpOverlayError(f"unable to link target shard: {filename}") from exc
        links[filename] = link_name
    return links


def _validate_source_shards(source_snapshot: Path, target: Target) -> None:
    source = target.mtp_source
    assert source is not None
    for shard in source.shards:
        path = _resolve_weight_file(source_snapshot, shard.filename)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise MtpOverlayError(f"unable to stat MTP source shard: {shard.filename}") from exc
        if size != shard.size:
            raise MtpOverlayError(f"MTP source shard size mismatch: {shard.filename}")
        if _sha256(path) != shard.sha256:
            raise MtpOverlayError(f"MTP source shard SHA-256 mismatch: {shard.filename}")


def _validate_weight_maps(
    target_index: dict[str, Any], source_index: dict[str, Any], target: Target
) -> None:
    target_weights = target_index.get("weight_map")
    source_weights = source_index.get("weight_map")
    if not _valid_weight_map(target_weights) or not _valid_weight_map(source_weights):
        raise MtpOverlayError("safetensors index has an invalid weight_map")
    if any(name.startswith("mtp.") for name in target_weights):
        raise MtpOverlayError("target checkpoint unexpectedly contains MTP tensors")
    found = {name for name in source_weights if name.startswith("mtp.")}
    if found != set(MTP_TENSOR_NAMES):
        raise MtpOverlayError("MTP source index does not contain the canonical 31 tensor names")
    source = target.mtp_source
    assert source is not None
    if source.tensor_count != len(MTP_TENSOR_NAMES):
        raise MtpOverlayError("manifest MTP tensor count is not canonical")
    approved_files = {shard.filename for shard in source.shards}
    referenced_files = {source_weights[name] for name in MTP_TENSOR_NAMES}
    if referenced_files != approved_files:
        raise MtpOverlayError("MTP source index does not reference exactly the approved shards")


def _verify_existing(
    final: Path, target: Target, target_snapshot: Path, output_root: Path
) -> None:
    if final.is_symlink() or not final.is_dir():
        raise MtpOverlayError(f"existing MTP overlay is not a directory: {final}")
    metadata = _load_json(final / _METADATA_NAME, "MTP overlay metadata")
    try:
        source = target.mtp_source
        assert source is not None
        if metadata.get("schema_version") != 1:
            raise MtpOverlayError("existing MTP metadata schema is invalid")
        if metadata.get("target") != {"repo_id": target.repo_id, "revision": target.revision}:
            raise MtpOverlayError("existing MTP metadata target identity does not match")
        if metadata.get("mtp_source") != {"repo_id": source.repo_id, "revision": source.revision}:
            raise MtpOverlayError("existing MTP metadata source identity does not match")
        if metadata.get("mtp_tensor_names") != list(MTP_TENSOR_NAMES):
            raise MtpOverlayError("existing MTP metadata tensor names do not match")
        compact = metadata["compact_shard"]
        compact_path = final / compact["filename"]
        if compact.get("filename") != _COMPACT_NAME:
            raise MtpOverlayError("existing MTP metadata compact filename is invalid")
        if compact_path.stat().st_size != compact.get("size") or _sha256(compact_path) != compact.get("sha256"):
            raise MtpOverlayError("existing MTP metadata compact shard digest does not match")
        if _sha256(final / _INDEX_NAME) != metadata.get("index_sha256"):
            raise MtpOverlayError("existing MTP metadata index digest does not match")
        if _sha256(final / "config.json") != metadata.get("config_sha256"):
            raise MtpOverlayError("existing MTP metadata config digest does not match")
    except (AssertionError, KeyError, OSError, TypeError) as exc:
        if isinstance(exc, MtpOverlayError):
            raise
        raise MtpOverlayError("existing MTP metadata is incomplete") from exc
    _audit(final, target, target_snapshot, output_root)


def _audit(view: Path, target: Target, target_snapshot: Path, output_root: Path) -> None:
    from .audit import AuditError, audit_checkpoint

    try:
        audit_checkpoint(view, target, (target_snapshot.parent.parent, output_root))
    except AuditError as exc:
        raise MtpOverlayError(str(exc)) from exc


def _load_index(snapshot: Path) -> dict[str, Any]:
    index = _load_json(snapshot / _INDEX_NAME, "safetensors index")
    if not _valid_weight_map(index.get("weight_map")):
        raise MtpOverlayError("safetensors index has an invalid weight_map")
    return index


def _valid_weight_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(name, str) and isinstance(filename, str) for name, filename in value.items())
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except DuplicateJsonKeyError as exc:
        raise MtpOverlayError(str(exc)) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MtpOverlayError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MtpOverlayError(f"{label} must be an object")
    return value


def _resolve_snapshot(path: Path, label: str, repo_id: str, revision: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise MtpOverlayError(f"{label} does not exist: {path}") from exc
    expected_repo = "models--" + repo_id.replace("/", "--")
    if (
        not resolved.is_dir()
        or not _REVISION.fullmatch(revision)
        or resolved.name != revision
        or resolved.parent.name != "snapshots"
        or resolved.parent.parent.name != expected_repo
    ):
        raise MtpOverlayError(f"{label} does not match its pinned identity")
    return resolved


def _resolve_output_root(
    output_root: Path, target_snapshot: Path, source_snapshot: Path
) -> Path:
    try:
        resolved = Path(output_root).resolve(strict=False)
    except OSError as exc:
        raise MtpOverlayError(f"unable to resolve MTP output root: {output_root}") from exc
    if _is_relative_to(resolved, target_snapshot) or _is_relative_to(resolved, source_snapshot):
        raise MtpOverlayError("MTP output root must not be inside an upstream snapshot")
    return resolved


def _resolve_weight_file(snapshot: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise MtpOverlayError(f"invalid safetensors path: {filename}")
    try:
        resolved = (snapshot / relative).resolve(strict=True)
    except OSError as exc:
        raise MtpOverlayError(f"missing safetensors shard: {filename}") from exc
    if not resolved.is_file():
        raise MtpOverlayError(f"safetensors shard is not a file: {filename}")
    return resolved


def _validate_target(target: Target) -> None:
    if target.mode != "mtp_overlay" or target.mtp_source is None:
        raise MtpOverlayError("MTP overlay requires an mtp_overlay target")
    if target.mtp_source.dtype != "BF16":
        raise MtpOverlayError("MTP overlay requires BF16 source tensors")


def _safe_name(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise MtpOverlayError("MTP target name must be one safe path component")
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MtpOverlayError(f"unable to hash file: {path}") from exc
    return digest.hexdigest()


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
