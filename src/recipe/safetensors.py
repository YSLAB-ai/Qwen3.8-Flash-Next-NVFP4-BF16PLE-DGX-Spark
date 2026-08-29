"""Read safetensors headers without loading tensor payloads."""

from __future__ import annotations

import json
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class SafetensorsError(ValueError):
    """Raised when a safetensors header is malformed or unsafe to inspect."""


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


_MAX_HEADER_BYTES = 100_000_000
_COPY_BUFFER_BYTES = 8 * 1024 * 1024
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


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


@dataclass(frozen=True)
class SubsetResult:
    tensor_names: tuple[str, ...]
    sha256: str
    size: int


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


def write_subset(
    entries: Iterable[tuple[str, Path]],
    destination: Path,
    *,
    expected_dtype: str,
) -> SubsetResult:
    """Stream selected tensors into one deterministic safetensors file.

    Payload bytes are copied directly from the source files. The function never
    materializes a tensor and refuses to overwrite any existing destination.
    """
    requested = list(entries)
    if not requested:
        raise SafetensorsError("tensor subset must not be empty")
    if not isinstance(expected_dtype, str) or expected_dtype not in _DTYPE_BYTES:
        raise SafetensorsError("unsupported expected dtype")

    names = [name for name, _source in requested]
    if any(not isinstance(name, str) or not name for name in names):
        raise SafetensorsError("tensor subset contains an invalid name")
    if len(names) != len(set(names)):
        raise SafetensorsError("duplicate tensor selection")

    destination = Path(destination)
    destination_resolved = destination.resolve(strict=False)
    sources: dict[Path, dict[str, TensorMeta]] = {}
    selected: list[tuple[str, Path, TensorMeta]] = []
    for name, raw_source in requested:
        try:
            source = Path(raw_source).resolve(strict=True)
        except OSError as exc:
            raise SafetensorsError(f"unable to resolve tensor source: {raw_source}") from exc
        if source == destination_resolved:
            raise SafetensorsError("destination aliases a tensor source")
        if not source.is_file():
            raise SafetensorsError(f"tensor source is not a file: {raw_source}")
        header = sources.get(source)
        if header is None:
            header = read_header(source)
            sources[source] = header
        try:
            meta = header[name]
        except KeyError as exc:
            raise SafetensorsError(f"missing selected tensor: {name}") from exc
        if meta.dtype != expected_dtype:
            raise SafetensorsError(
                f"tensor dtype {meta.dtype} does not match expected {expected_dtype}: {name}"
            )
        expected_bytes = _tensor_byte_size(meta)
        if meta.data_end - meta.data_start != expected_bytes:
            raise SafetensorsError(f"tensor byte size does not match dtype and shape: {name}")
        selected.append((name, source, meta))

    selected.sort(key=lambda item: item[0])
    raw_header: dict[str, dict[str, object]] = {}
    offset = 0
    for name, _source, meta in selected:
        size = meta.data_end - meta.data_start
        raw_header[name] = {
            "dtype": meta.dtype,
            "shape": list(meta.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(raw_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded

    digest = hashlib.sha256()
    try:
        with destination.open("xb") as output:
            output.write(prefix)
            digest.update(prefix)
            for name, source, meta in selected:
                remaining = meta.data_end - meta.data_start
                with source.open("rb") as input_file:
                    input_file.seek(meta.data_start)
                    while remaining:
                        chunk = input_file.read(min(remaining, _COPY_BUFFER_BYTES))
                        if not chunk:
                            raise SafetensorsError(f"truncated tensor payload while copying: {name}")
                        output.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
    except FileExistsError as exc:
        raise SafetensorsError(f"destination already exists: {destination}") from exc
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    try:
        size = destination.stat().st_size
    except OSError as exc:
        raise SafetensorsError(f"unable to stat tensor subset: {destination}") from exc
    return SubsetResult(tuple(name for name, _source, _meta in selected), digest.hexdigest(), size)


def _tensor_byte_size(meta: TensorMeta) -> int:
    size = _DTYPE_BYTES[meta.dtype]
    for dimension in meta.shape:
        size *= dimension
    return size
