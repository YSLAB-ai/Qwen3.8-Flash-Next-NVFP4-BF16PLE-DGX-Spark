"""Tests for dependency-free safetensors checkpoint auditing."""

from __future__ import annotations

import contextlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from recipe.audit import AuditError, audit_checkpoint
from recipe.manifest import PleExpectation, Target
from recipe.safetensors import SafetensorsError, read_header


REVISION = "a" * 40
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


@contextlib.contextmanager
def assert_raises(error_type, message):
    try:
        yield
    except error_type as error:
        if message not in str(error):
            raise AssertionError(f"{message!r} not found in {error!s}") from error
    else:
        raise AssertionError(f"{error_type.__name__} not raised")


class SafetensorsAuditTests(unittest.TestCase):
    def test_audit_accepts_exact_bf16_ple(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory), dtype="BF16", scale=False)

            result = audit_checkpoint(model_dir, target, (root,))

            self.assertEqual((result.ple_dtype, result.ple_tensor_count), ("BF16", 4))
            self.assertEqual((result.ple_rows, result.ple_width), (25, 7))
            self.assertEqual(result.revision, REVISION)
            self.assertEqual(len(result.ple_files), 4)
            self.assertEqual(
                result.resolved_bytes,
                sum(path.stat().st_size for path in model_dir.glob("*.safetensors")),
            )

    def test_audit_rejects_bf16_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory), dtype="BF16", scale=True)

            with assert_raises(AuditError, "BF16 PLE must not have weight_scale"):
                audit_checkpoint(model_dir, target, (root,))

    def test_audit_rejects_fp8_ple_without_weight_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory), dtype="F8_E4M3")

            with assert_raises(AuditError, "FP8 PLE must have weight_scale"):
                audit_checkpoint(model_dir, target, (root,))

    def test_audit_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            target, model_dir, root = make_checkpoint(tmp_path)
            replace_weight_with_escape(model_dir, tmp_path / "outside.safetensors")

            with assert_raises(AuditError, "outside approved cache roots"):
                audit_checkpoint(model_dir, target, (root,))

    def test_audit_rejects_config_that_is_not_qualified_qwen4exp(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory))
            config_path = model_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["architectures"] = ["OtherModel"]
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with assert_raises(AuditError, "architectures"):
                audit_checkpoint(model_dir, target, (root,))

    def test_audit_rejects_index_mapping_to_missing_header_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory))
            index_path = model_dir / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["model.language_model.layers.0.mlp.weight"] = (
                "model-00002-of-00004.safetensors"
            )
            index_path.write_text(json.dumps(index), encoding="utf-8")

            with assert_raises(AuditError, "index/header disagreement"):
                audit_checkpoint(model_dir, target, (root,))

    def test_audit_rejects_unexpected_ple_shard_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory))
            filename = "model-00005-of-00005.safetensors"
            name = f"{PLE_PREFIX}.shard_unexpected.weight"
            write_safetensors(
                model_dir / filename,
                {name: {"dtype": "BF16", "shape": [1, 7]}},
            )
            index_path = model_dir / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"][name] = filename
            index_path.write_text(json.dumps(index), encoding="utf-8")

            with assert_raises(AuditError, "unexpected PLE shard"):
                audit_checkpoint(model_dir, target, (root,))

    def test_audit_rejects_noncanonical_direct_snapshot_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory))
            moved = model_dir.with_name("b" * 40)
            model_dir.rename(moved)

            with assert_raises(AuditError, "snapshot revision"):
                audit_checkpoint(moved, target, (root,))

    def test_audit_does_not_accept_direct_metadata_as_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            target, model_dir, _root = make_checkpoint(tmp_path)
            arbitrary = tmp_path / "arbitrary" / "not-a-snapshot"
            arbitrary.parent.mkdir()
            model_dir.rename(arbitrary)
            (arbitrary / "recipe-metadata.json").write_text(
                json.dumps(
                    {"target": {"repo_id": target.repo_id, "revision": target.revision}}
                ),
                encoding="utf-8",
            )

            with assert_raises(AuditError, "snapshot revision"):
                audit_checkpoint(arbitrary, target, (tmp_path,))

    def test_audit_rejects_unindexed_header_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            target, model_dir, root = make_checkpoint(Path(directory))
            write_safetensors(
                model_dir / "model-00001-of-00004.safetensors",
                {
                    f"{PLE_PREFIX}.shard_0.weight": {"dtype": "BF16", "shape": [7, 7]},
                    "model.language_model.layers.0.mlp.weight": {"dtype": "F32", "shape": [1]},
                    "unindexed.weight": {"dtype": "F32", "shape": [1]},
                },
            )

            with assert_raises(AuditError, "index/header disagreement"):
                audit_checkpoint(model_dir, target, (root,))

    def test_read_header_rejects_truncated_header_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.safetensors"
            path.write_bytes(b"bad")

            with assert_raises(SafetensorsError, "truncated header length"):
                read_header(path)

    def test_read_header_rejects_oversized_header_before_reading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.safetensors"
            path.write_bytes(struct.pack("<Q", 100_000_001))

            with assert_raises(SafetensorsError, "header exceeds 100 MiB"):
                read_header(path)

    def test_read_header_rejects_overlapping_tensor_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.safetensors"
            header = {
                "first": {"dtype": "BF16", "shape": [7, 7], "data_offsets": [0, 98]},
                "second": {"dtype": "BF16", "shape": [7, 7], "data_offsets": [0, 98]},
            }
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(98))

            with assert_raises(SafetensorsError, "contiguous"):
                read_header(path)


def make_checkpoint(tmp_path: Path, *, dtype: str = "BF16", scale: bool = False):
    root = tmp_path / "cache"
    model_dir = root / "models--example--approved" / "snapshots" / REVISION
    model_dir.mkdir(parents=True)
    target = Target(
        name="approved",
        repo_id="example/approved",
        revision=REVISION,
        mode="direct_bf16",
        served_model_name="approved",
        requires_auth=False,
        ple_source=None,
        minimum_free_bytes=1,
        expected_ple=PleExpectation(
            tensor_count=4,
            total_rows=25,
            width=7,
            dtype=dtype,
            layer_id=1,
            split_parts=4,
        ),
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "ple_layer_ids": [2],
                    "split_ngram_parts": 4,
                    "ngram_size": 3,
                    "heads_per_ngram": 8,
                    "ple_embed_dim": 112,
                },
            }
        ),
        encoding="utf-8",
    )
    weight_map = {"model.language_model.layers.0.mlp.weight": "model-00001-of-00004.safetensors"}
    rows_per_shard = (7, 6, 6, 6)
    for shard_index, rows in enumerate(rows_per_shard):
        filename = f"model-{shard_index + 1:05d}-of-00004.safetensors"
        tensor_name = f"{PLE_PREFIX}.shard_{shard_index}.weight"
        tensors = {
            tensor_name: {"dtype": dtype, "shape": [rows, 7]},
        }
        if shard_index == 0:
            tensors["model.language_model.layers.0.mlp.weight"] = {
                "dtype": "F32",
                "shape": [1],
            }
        if scale and shard_index == 0:
            tensors[f"{PLE_PREFIX}.weight_scale"] = {"dtype": "F32", "shape": [1]}
        write_safetensors(model_dir / filename, tensors)
        weight_map[tensor_name] = filename
        if scale and shard_index == 0:
            weight_map[f"{PLE_PREFIX}.weight_scale"] = filename
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return target, model_dir, root


def replace_weight_with_escape(model_dir: Path, outside: Path) -> None:
    source = model_dir / "model-00001-of-00004.safetensors"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)


def write_safetensors(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    offsets = 0
    header = {}
    for name, tensor in tensors.items():
        size = _tensor_bytes(tensor["dtype"], tensor["shape"])
        header[name] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offsets, offsets + size],
        }
        offsets += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offsets))


def _tensor_bytes(dtype: str, shape: object) -> int:
    item_size = {"BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
    result = item_size
    for dimension in shape:
        result *= dimension
    return result


if __name__ == "__main__":
    unittest.main()
