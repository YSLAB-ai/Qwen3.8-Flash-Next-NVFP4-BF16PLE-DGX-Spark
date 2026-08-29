"""Tests for building an audited BF16 PLE hybrid checkpoint view."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from recipe.audit import AuditError, audit_checkpoint
from recipe.hybrid import HybridError, build_hybrid_view
from recipe.manifest import ModelRef, PleExpectation, Target


TRUNK_NAME = "model.language_model.layers.0.mlp.weight"
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
PLE_SHARD_0 = f"{PLE_PREFIX}.shard_0.weight"
PLE_SHARD_1 = f"{PLE_PREFIX}.shard_1.weight"
PLE_SCALE = f"{PLE_PREFIX}.weight_scale"


class HybridViewTests(unittest.TestCase):
    def test_hybrid_keeps_trunk_and_replaces_only_ple(self):
        """Replacing PLE entries must leave the target trunk and omit its FP8 scale."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)

            view = build_hybrid_view(target_dir, source_dir, temporary / "views", target)

            weight_map = load_weight_map(view)
            self.assertTrue(weight_map[TRUNK_NAME].startswith("target--"))
            self.assertTrue(weight_map[PLE_SHARD_0].startswith("bf16-ple--"))
            self.assertTrue(weight_map[PLE_SHARD_1].startswith("bf16-ple--"))
            self.assertNotIn(PLE_SCALE, weight_map)
            self.assertTrue((view / weight_map[PLE_SHARD_0]).is_symlink())
            self.assertTrue((view / weight_map[TRUNK_NAME]).is_symlink())
            self.assertFalse(Path((view / weight_map[PLE_SHARD_0]).readlink()).is_absolute())
            audit_checkpoint(view, target, (target_dir, source_dir))

            metadata = json.loads((view / "recipe-metadata.json").read_text(encoding="utf-8"))
            index = (view / "model.safetensors.index.json").read_bytes()
            self.assertEqual(metadata["index_sha256"], hashlib.sha256(index).hexdigest())

    def test_hybrid_is_idempotent_when_existing_metadata_matches(self):
        """A complete view with matching identity and index digest is reused."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)

            first = build_hybrid_view(target_dir, source_dir, temporary / "views", target)

            self.assertEqual(
                build_hybrid_view(target_dir, source_dir, temporary / "views", target),
                first,
            )

    def test_hybrid_refuses_existing_view_with_tampered_metadata(self):
        """A stale final directory is never reused merely because its path matches."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)
            view = build_hybrid_view(target_dir, source_dir, temporary / "views", target)
            (view / "recipe-metadata.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(HybridError):
                build_hybrid_view(target_dir, source_dir, temporary / "views", target)

    def test_hybrid_refuses_stale_build_directory(self):
        """A previous partial build is preserved for inspection rather than deleted."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)
            fingerprint = fingerprint_for(target)
            stale = temporary / "views" / target.name / f".{fingerprint}.building"
            stale.mkdir(parents=True)
            (stale / "evidence.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(HybridError):
                build_hybrid_view(target_dir, source_dir, temporary / "views", target)

            self.assertEqual((stale / "evidence.txt").read_text(encoding="utf-8"), "keep")

    def test_hybrid_uses_distinct_links_for_same_shard_basename(self):
        """Target/source paths sharing a basename must not overwrite one another."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(
                temporary,
                target_shard="target/shared.safetensors",
                source_shards=("source/shared.safetensors", "source/second.safetensors"),
            )

            view = build_hybrid_view(target_dir, source_dir, temporary / "views", target)
            weight_map = load_weight_map(view)

            self.assertNotEqual(weight_map[TRUNK_NAME], weight_map[PLE_SHARD_0])
            self.assertTrue((view / weight_map[TRUNK_NAME]).is_symlink())
            self.assertTrue((view / weight_map[PLE_SHARD_0]).is_symlink())

    def test_hybrid_does_not_copy_non_safetensors_weight_payloads(self):
        """Only serving metadata, never alternate weight formats, belongs in the view."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)
            (target_dir / "pytorch_model.bin").write_bytes(b"not metadata")

            view = build_hybrid_view(target_dir, source_dir, temporary / "views", target)

            self.assertFalse((view / "pytorch_model.bin").exists())

    def test_hybrid_keeps_failed_audit_out_of_final_path(self):
        """Header/index disagreement fails audit before the temporary view is finalized."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(
                temporary, source_extra_tensor=True
            )
            fingerprint = fingerprint_for(target)
            final = temporary / "views" / target.name / fingerprint
            building = final.with_name(f".{fingerprint}.building")

            with self.assertRaises(AuditError):
                build_hybrid_view(target_dir, source_dir, temporary / "views", target)

            self.assertFalse(final.exists())
            self.assertTrue((building / "recipe-metadata.json").is_file())


def make_target_and_source(
    tmp_path: Path,
    *,
    target_shard: str = "target/trunk.safetensors",
    source_shards: tuple[str, str] = ("source/ple-0.safetensors", "source/ple-1.safetensors"),
    source_extra_tensor: bool = False,
) -> tuple[Path, Path, Target]:
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    target_dir.mkdir()
    source_dir.mkdir()
    target = Target(
        name="radixark",
        repo_id="example/radixark",
        revision="a" * 40,
        mode="hybrid_bf16",
        served_model_name="radixark",
        requires_auth=False,
        ple_source=ModelRef("example/official", "b" * 40),
        minimum_free_bytes=1,
        expected_ple=PleExpectation(
            tensor_count=2,
            total_rows=5,
            width=3,
            dtype="BF16",
            layer_id=1,
            split_parts=2,
        ),
    )
    write_json(
        target_dir / "config.json",
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "ple_layer_ids": [2],
                "split_ngram_parts": 2,
                "ngram_size": 2,
                "heads_per_ngram": 1,
                "ple_embed_dim": 3,
            },
        },
    )
    (target_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    write_safetensors(target_dir / target_shard, {TRUNK_NAME: {"dtype": "F32", "shape": [1]}})
    write_safetensors(
        target_dir / "target/old-ple.safetensors",
        {
            PLE_SHARD_0: {"dtype": "F8_E4M3", "shape": [2, 3]},
            PLE_SHARD_1: {"dtype": "F8_E4M3", "shape": [3, 3]},
            PLE_SCALE: {"dtype": "F32", "shape": [1]},
        },
    )
    write_json(
        target_dir / "model.safetensors.index.json",
        {
            "weight_map": {
                TRUNK_NAME: target_shard,
                PLE_SHARD_0: "target/old-ple.safetensors",
                PLE_SHARD_1: "target/old-ple.safetensors",
                PLE_SCALE: "target/old-ple.safetensors",
            }
        },
    )
    first_source_tensors = {PLE_SHARD_0: {"dtype": "BF16", "shape": [2, 3]}}
    if source_extra_tensor:
        first_source_tensors["unexpected.source.weight"] = {"dtype": "F32", "shape": [1]}
    write_safetensors(source_dir / source_shards[0], first_source_tensors)
    write_safetensors(source_dir / source_shards[1], {PLE_SHARD_1: {"dtype": "BF16", "shape": [3, 3]}})
    write_json(
        source_dir / "model.safetensors.index.json",
        {"weight_map": {PLE_SHARD_0: source_shards[0], PLE_SHARD_1: source_shards[1]}},
    )
    return target_dir, source_dir, target


def load_weight_map(view: Path) -> dict[str, str]:
    return json.loads((view / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]


def fingerprint_for(target: Target) -> str:
    return hashlib.sha256(
        f"{target.repo_id}@{target.revision}:{target.ple_source.repo_id}@{target.ple_source.revision}".encode()
    ).hexdigest()[:16]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_safetensors(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    header = {}
    for name, tensor in tensors.items():
        size = tensor_bytes(tensor["dtype"], tensor["shape"])
        header[name] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def tensor_bytes(dtype: object, shape: object) -> int:
    size = {"BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
    for dimension in shape:
        size *= dimension
    return size


if __name__ == "__main__":
    unittest.main()
