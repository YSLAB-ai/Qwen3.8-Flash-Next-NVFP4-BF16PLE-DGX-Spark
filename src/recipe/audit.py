"""Validate pinned checkpoint layouts without reading tensor payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import Target
from .safetensors import SafetensorsError, TensorMeta, read_header


class AuditError(ValueError):
    """Raised when a checkpoint does not match its pinned target."""


@dataclass(frozen=True)
class AuditResult:
    revision: str
    ple_dtype: str
    ple_tensor_count: int
    ple_rows: int
    ple_width: int
    ple_files: tuple[Path, ...]
    resolved_bytes: int


_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FN": 1,
    "F8_E5M2FNUZ": 1,
}


def audit_checkpoint(
    model_dir: Path, target: Target, allowed_roots: tuple[Path, ...]
) -> AuditResult:
    """Fail closed unless a local checkpoint exactly matches ``target``.

    This inspects JSON metadata, safetensors headers, and filesystem metadata only.
    """
    model_dir = _resolve_model_dir(model_dir)
    _validate_identity(model_dir, target)
    _validate_config(model_dir, target)
    roots = _resolve_allowed_roots(allowed_roots)
    weight_map = _load_weight_map(model_dir)

    headers: dict[Path, dict[str, TensorMeta]] = {}
    mapped: dict[str, tuple[Path, TensorMeta]] = {}
    for name, filename in weight_map.items():
        resolved = _resolve_shard(model_dir, filename, roots)
        if resolved not in headers:
            try:
                headers[resolved] = read_header(resolved)
            except SafetensorsError as exc:
                raise AuditError(str(exc)) from exc
        try:
            mapped[name] = (resolved, headers[resolved][name])
        except KeyError as exc:
            raise AuditError(
                f"index/header disagreement: {name} is absent from {filename}"
            ) from exc
    for path, header in headers.items():
        indexed_names = {name for name, (mapped_path, _meta) in mapped.items() if mapped_path == path}
        if set(header) != indexed_names:
            raise AuditError(f"index/header disagreement for shard: {path.name}")

    expected_names = _expected_ple_names(target)
    prefix = _ple_prefix(target)
    shard_pattern = re.compile(rf"^{re.escape(prefix)}\.shard_(\d+)\.weight$")
    shard_prefix = f"{prefix}.shard_"
    scale_name = f"{prefix}.weight_scale"
    if any(name.startswith(shard_prefix) and not shard_pattern.fullmatch(name) for name in weight_map):
        raise AuditError("unexpected PLE shard name in safetensors index")
    index_ple_names = {name for name in weight_map if shard_pattern.fullmatch(name)}
    if index_ple_names != expected_names:
        raise AuditError(
            "PLE shard indices do not match expected layout: "
            f"found {sorted(index_ple_names)}, expected {sorted(expected_names)}"
        )

    header_ple_names = [
        name
        for header in headers.values()
        for name in header
        if shard_pattern.fullmatch(name)
    ]
    if any(
        name.startswith(shard_prefix) and not shard_pattern.fullmatch(name)
        for header in headers.values()
        for name in header
    ):
        raise AuditError("unexpected PLE shard name in safetensors header")
    if set(header_ple_names) != expected_names or len(header_ple_names) != len(expected_names):
        raise AuditError("index/header disagreement for PLE shard tensors")

    scale_in_headers = [
        name for header in headers.values() for name in header if name == scale_name
    ]
    if target.expected_ple.dtype == "BF16" and scale_in_headers:
        raise AuditError("BF16 PLE must not have weight_scale")
    if target.expected_ple.dtype.startswith("F8_"):
        if len(scale_in_headers) != 1 or scale_name not in mapped:
            raise AuditError("FP8 PLE must have weight_scale")
        _validate_weight_scale(mapped[scale_name][1])

    ple_entries = [mapped[name] for name in sorted(expected_names)]
    _validate_ple_entries(ple_entries, target)

    resolved_files = tuple(sorted(headers))
    ple_files = tuple(sorted({path for path, _meta in ple_entries}))
    return AuditResult(
        revision=target.revision,
        ple_dtype=target.expected_ple.dtype,
        ple_tensor_count=len(ple_entries),
        ple_rows=sum(meta.shape[0] for _path, meta in ple_entries),
        ple_width=target.expected_ple.width,
        ple_files=ple_files,
        resolved_bytes=sum(path.stat().st_size for path in resolved_files),
    )


def _resolve_model_dir(model_dir: Path) -> Path:
    try:
        resolved = model_dir.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"model directory does not exist: {model_dir}") from exc
    if not resolved.is_dir():
        raise AuditError(f"model directory is not a directory: {model_dir}")
    return resolved


def _validate_identity(model_dir: Path, target: Target) -> None:
    metadata_path = model_dir / "recipe-metadata.json"
    if target.mode == "hybrid_bf16":
        if not metadata_path.is_file():
            raise AuditError("hybrid checkpoint requires recipe metadata identity")
        _validate_hybrid_metadata(metadata_path, target)
        return

    expected_cache_dir = "models--" + target.repo_id.replace("/", "--")
    if (
        model_dir.name != target.revision
        or model_dir.parent.name != "snapshots"
        or model_dir.parent.parent.name != expected_cache_dir
    ):
        raise AuditError(
            "snapshot revision or repository identity does not match canonical HF layout"
        )


def _validate_hybrid_metadata(metadata_path: Path, target: Target) -> None:
    metadata = _load_json_object(metadata_path, "recipe metadata")
    target_metadata = metadata.get("target")
    if not _matches_model_ref(target_metadata, target.repo_id, target.revision):
        raise AuditError("recipe metadata target identity does not match manifest target")
    if target.mode == "hybrid_bf16":
        if target.ple_source is None or not _matches_model_ref(
            metadata.get("ple_source"), target.ple_source.repo_id, target.ple_source.revision
        ):
            raise AuditError("recipe metadata PLE source identity does not match manifest target")


def _matches_model_ref(value: Any, repo_id: str, revision: str) -> bool:
    return (
        isinstance(value, dict)
        and value.get("repo_id") == repo_id
        and value.get("revision") == revision
        and isinstance(value.get("repo_id"), str)
        and isinstance(value.get("revision"), str)
    )


def _validate_config(model_dir: Path, target: Target) -> None:
    config = _load_json_object(model_dir / "config.json", "config")
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise AuditError("config architectures must be [Qwen4ExpForConditionalGeneration]")
    if config.get("model_type") != "qwen4_exp":
        raise AuditError("config model_type must be qwen4_exp")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen4_exp_text":
        raise AuditError("config text_config.model_type must be qwen4_exp_text")

    expected_layer_ids = [target.expected_ple.layer_id + 1]
    if text_config.get("ple_layer_ids") != expected_layer_ids:
        raise AuditError(f"config ple_layer_ids must be {expected_layer_ids}")
    if not _is_strict_positive_int(text_config.get("split_ngram_parts")) or (
        text_config["split_ngram_parts"] != target.expected_ple.split_parts
    ):
        raise AuditError("config split_ngram_parts does not match manifest target")
    ngram_size = text_config.get("ngram_size")
    heads_per_ngram = text_config.get("heads_per_ngram")
    if not _is_strict_positive_int(ngram_size) or not _is_strict_positive_int(heads_per_ngram):
        raise AuditError("config ngram_size and heads_per_ngram must be positive integers")
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if ngram_heads <= 0:
        raise AuditError("config ngram_heads must be positive")
    if text_config.get("ple_embed_dim") != target.expected_ple.width * ngram_heads:
        raise AuditError("config ple_embed_dim does not match PLE width and ngram heads")


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index = _load_json_object(model_dir / "model.safetensors.index.json", "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AuditError("safetensors index weight_map must be a non-empty object")
    if not all(isinstance(name, str) and isinstance(filename, str) for name, filename in weight_map.items()):
        raise AuditError("safetensors index weight_map entries must be strings")
    return weight_map


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be an object: {path}")
    return value


def _resolve_allowed_roots(allowed_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    if not allowed_roots:
        raise AuditError("at least one approved cache root is required")
    roots: list[Path] = []
    for root in allowed_roots:
        try:
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise AuditError(f"approved cache root does not exist: {root}") from exc
        if not resolved.is_dir():
            raise AuditError(f"approved cache root is not a directory: {root}")
        roots.append(resolved)
    return tuple(roots)


def _resolve_shard(model_dir: Path, filename: str, roots: tuple[Path, ...]) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditError(f"invalid safetensors index path: {filename}")
    if relative.suffix != ".safetensors":
        raise AuditError(f"index path is not a safetensors file: {filename}")
    try:
        resolved = (model_dir / relative).resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"missing or broken safetensors shard: {filename}") from exc
    if not resolved.is_file():
        raise AuditError(f"safetensors shard is not a file: {filename}")
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise AuditError(f"safetensors shard is outside approved cache roots: {filename}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _expected_ple_names(target: Target) -> set[str]:
    prefix = _ple_prefix(target)
    return {
        f"{prefix}.shard_{index}.weight"
        for index in range(target.expected_ple.tensor_count)
    }


def _ple_prefix(target: Target) -> str:
    return (
        f"model.language_model.layers.{target.expected_ple.layer_id}."
        "ple.ple_embedding.ngram_embedding"
    )


def _validate_ple_entries(
    entries: list[tuple[Path, TensorMeta]], target: Target
) -> None:
    rows = 0
    for _path, meta in entries:
        if meta.dtype != target.expected_ple.dtype:
            raise AuditError(
                f"PLE dtype {meta.dtype} does not match expected {target.expected_ple.dtype}"
            )
        if len(meta.shape) != 2 or meta.shape[1] != target.expected_ple.width:
            raise AuditError(f"PLE tensor shape {meta.shape} does not match expected width")
        if meta.shape[0] <= 0:
            raise AuditError("PLE tensor row count must be positive")
        rows += meta.shape[0]
        _validate_tensor_byte_size(meta, "PLE tensor")
    if rows != target.expected_ple.total_rows:
        raise AuditError(
            f"PLE rows {rows} does not match expected {target.expected_ple.total_rows}"
        )


def _validate_tensor_byte_size(meta: TensorMeta, label: str) -> None:
    try:
        item_size = _DTYPE_BYTES[meta.dtype]
    except KeyError as exc:
        raise AuditError(f"unsupported {label} dtype: {meta.dtype}") from exc
    expected_bytes = item_size
    for dimension in meta.shape:
        expected_bytes *= dimension
    if meta.data_end - meta.data_start != expected_bytes:
        raise AuditError(f"{label} byte size does not match dtype and shape")


def _validate_weight_scale(meta: TensorMeta) -> None:
    if meta.dtype not in {"BF16", "F16", "F32"}:
        raise AuditError(f"unsupported weight_scale dtype: {meta.dtype}")
    element_count = 1
    for dimension in meta.shape:
        element_count *= dimension
    if element_count != 1:
        raise AuditError("weight_scale must contain exactly one value")
    _validate_tensor_byte_size(meta, "weight_scale")


def _is_strict_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
