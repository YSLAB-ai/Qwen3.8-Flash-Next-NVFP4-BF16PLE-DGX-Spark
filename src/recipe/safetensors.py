"""Read safetensors headers without loading tensor payloads."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SafetensorsError(ValueError):
    """Raised when a safetensors header is malformed or unsafe to inspect."""


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


_MAX_HEADER_BYTES = 100_000_000


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


def strict_json_loads(document: str | bytes) -> Any:
    """Decode JSON while rejecting duplicate keys at every object level."""
    return json.loads(document, object_pairs_hook=_reject_duplicate_keys)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path: Path) -> dict[str, TensorMeta]:
    """Return tensor metadata and absolute data offsets without reading payloads."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise SafetensorsError(f"truncated header length: {path}")
            (header_len,) = struct.unpack("<Q", raw)
            if header_len > _MAX_HEADER_BYTES:
                raise SafetensorsError(f"header exceeds 100 MiB: {path}")
            if header_len > file_size - 8:
                raise SafetensorsError(f"truncated header: {path}")
            raw_header = handle.read(header_len)
    except OSError as exc:
        raise SafetensorsError(f"unable to read header: {path}") from exc

    if len(raw_header) != header_len:
        raise SafetensorsError(f"truncated header: {path}")
    try:
        header = strict_json_loads(raw_header)
    except DuplicateJsonKeyError as exc:
        raise SafetensorsError(str(exc)) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"invalid header JSON: {path}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"header must be an object: {path}")

    base = 8 + header_len
    tensors: dict[str, TensorMeta] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, raw_meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_meta, dict):
            raise SafetensorsError(f"invalid tensor metadata: {path}")
        dtype = raw_meta.get("dtype")
        shape = raw_meta.get("shape")
        offsets = raw_meta.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise SafetensorsError(f"invalid tensor dtype: {path}: {name}")
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in shape
        ):
            raise SafetensorsError(f"invalid tensor shape: {path}: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
                for offset in offsets
            )
            or offsets[0] > offsets[1]
        ):
            raise SafetensorsError(f"invalid tensor offsets: {path}: {name}")
        data_start, data_end = base + offsets[0], base + offsets[1]
        if data_end > file_size:
            raise SafetensorsError(f"tensor data exceeds file size: {path}: {name}")
        tensors[name] = TensorMeta(dtype, tuple(shape), data_start, data_end)
        ranges.append((data_start, data_end, name))
    expected_start = base
    for data_start, data_end, name in sorted(ranges):
        if data_start != expected_start:
            raise SafetensorsError(f"tensor data ranges must be contiguous: {path}: {name}")
        expected_start = data_end
    if expected_start != file_size:
        raise SafetensorsError(f"tensor data does not occupy the full data buffer: {path}")
    return tensors
