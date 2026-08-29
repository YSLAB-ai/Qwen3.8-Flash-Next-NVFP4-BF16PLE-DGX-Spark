"""Load and validate the pinned checkpoint compatibility manifest."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = 1
_REVISION = re.compile(r"[0-9a-f]{40}")
_MODES = {"direct_bf16", "hybrid_bf16"}
_TARGET_FIELDS = {
    "name",
    "repo_id",
    "revision",
    "mode",
    "served_model_name",
    "requires_auth",
    "ple_source",
    "minimum_free_bytes",
    "expected_ple",
}
_PLE_FIELDS = {
    "tensor_count",
    "total_rows",
    "width",
    "dtype",
    "layer_id",
    "split_parts",
}
_MODEL_REF_FIELDS = {"repo_id", "revision"}


class ManifestError(ValueError):
    """Raised when a compatibility manifest is invalid."""


@dataclass(frozen=True)
class ModelRef:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class PleExpectation:
    tensor_count: int
    total_rows: int
    width: int
    dtype: str
    layer_id: int
    split_parts: int


@dataclass(frozen=True)
class Target:
    name: str
    repo_id: str
    revision: str
    mode: str
    served_model_name: str
    requires_auth: bool
    ple_source: ModelRef | None
    minimum_free_bytes: int
    expected_ple: PleExpectation


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    targets: dict[str, Target]

    def target(self, name: str) -> Target:
        try:
            return self.targets[name]
        except KeyError as exc:
            raise ManifestError(f"unknown target: {name}") from exc


def load_manifest(path: Path) -> Manifest:
    """Load a version-1 manifest and reject unsupported or malformed input."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"unable to read manifest: {path}") from exc

    _require_object(document, "manifest")
    _require_exact_fields(document, {"schema_version", "targets"}, "manifest")
    if document["schema_version"] != _SCHEMA_VERSION or isinstance(
        document["schema_version"], bool
    ):
        raise ManifestError(f"unsupported schema version: {document['schema_version']!r}")
    if not isinstance(document["targets"], list):
        raise ManifestError("manifest.targets must be a list")

    targets: dict[str, Target] = {}
    for entry in document["targets"]:
        target = _parse_target(entry)
        if target.name in targets:
            raise ManifestError(f"duplicate target alias: {target.name}")
        targets[target.name] = target
    return Manifest(schema_version=_SCHEMA_VERSION, targets=targets)


def _parse_target(value: Any) -> Target:
    _require_object(value, "target")
    _require_fields(value, _TARGET_FIELDS - {"ple_source"}, "target")
    _reject_unknown_fields(value, _TARGET_FIELDS, "target")

    mode = _require_string(value["mode"], "target.mode")
    if mode not in _MODES:
        raise ManifestError(f"unknown target mode: {mode}")
    ple_source_value = value.get("ple_source")
    if mode == "hybrid_bf16" and ple_source_value is None:
        raise ManifestError("hybrid_bf16 target requires ple_source")
    if mode == "direct_bf16" and ple_source_value is not None:
        raise ManifestError("direct_bf16 target must not define ple_source")

    return Target(
        name=_require_string(value["name"], "target.name"),
        repo_id=_require_string(value["repo_id"], "target.repo_id"),
        revision=_parse_revision(value["revision"], "target.revision"),
        mode=mode,
        served_model_name=_require_string(value["served_model_name"], "target.served_model_name"),
        requires_auth=_require_bool(value["requires_auth"], "target.requires_auth"),
        ple_source=_parse_model_ref(ple_source_value) if ple_source_value is not None else None,
        minimum_free_bytes=_require_positive_int(value["minimum_free_bytes"], "target.minimum_free_bytes"),
        expected_ple=_parse_ple_expectation(value["expected_ple"]),
    )


def _parse_model_ref(value: Any) -> ModelRef:
    _require_object(value, "ple_source")
    _require_exact_fields(value, _MODEL_REF_FIELDS, "ple_source")
    return ModelRef(
        repo_id=_require_string(value["repo_id"], "ple_source.repo_id"),
        revision=_parse_revision(value["revision"], "ple_source.revision"),
    )


def _parse_ple_expectation(value: Any) -> PleExpectation:
    _require_object(value, "expected_ple")
    _require_exact_fields(value, _PLE_FIELDS, "expected_ple")
    return PleExpectation(
        tensor_count=_require_positive_int(value["tensor_count"], "expected_ple.tensor_count"),
        total_rows=_require_positive_int(value["total_rows"], "expected_ple.total_rows"),
        width=_require_positive_int(value["width"], "expected_ple.width"),
        dtype=_require_string(value["dtype"], "expected_ple.dtype"),
        layer_id=_require_positive_int(value["layer_id"], "expected_ple.layer_id"),
        split_parts=_require_positive_int(value["split_parts"], "expected_ple.split_parts"),
    )


def _parse_revision(value: Any, field: str) -> str:
    revision = _require_string(value, field)
    if not _REVISION.fullmatch(revision):
        raise ManifestError(f"{field} must be 40 lowercase hexadecimal characters")
    return revision


def _require_object(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")


def _require_exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    _require_fields(value, fields, label)
    _reject_unknown_fields(value, fields, label)


def _require_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = fields - value.keys()
    if missing:
        raise ManifestError(f"{label} missing fields: {', '.join(sorted(missing))}")


def _reject_unknown_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    unknown = value.keys() - fields
    if unknown:
        raise ManifestError(f"{label} unknown fields: {', '.join(sorted(unknown))}")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{field} must be a boolean")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{field} must be a positive integer")
    return value
