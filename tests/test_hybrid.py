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

    def test_hybrid_refuses_existing_view_with_tampered_index(self):
        """A final view is not reusable when its index digest no longer matches metadata."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)
            view = build_hybrid_view(target_dir, source_dir, temporary / "views", target)
            (view / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

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

    def test_hybrid_ignores_unknown_metadata_payloads(self):
        """An unlisted extensionless payload is not copied into the serving view."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)
            (target_dir / "unknown-payload").write_bytes(b"not serving metadata")

            view = build_hybrid_view(target_dir, source_dir, temporary / "views", target)

            self.assertFalse((view / "unknown-payload").exists())

    def test_hybrid_rejects_oversized_serving_metadata(self):
        """An allowlisted serving file over 16 MiB fails instead of being copied."""
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_dir, source_dir, target = make_target_and_source(temporary)
            (target_dir / "tokenizer.json").write_bytes(b"x" * (16 * 1024 * 1024 + 1))

            with self.assertRaises(HybridError):
                build_hybrid_view(target_dir, source_dir, temporary / "views", target)

    def test_hybrid_rejects_noncanonical_input_snapshots(self):
        """Only the target and source's exact HF snapshot paths can establish identity."""
        cases = (
            ("target repo", "target", "repo"),
            ("target revision", "target", "revision"),
            ("source repo", "source", "repo"),
            ("source revision", "source", "revision"),
        )
        for label, snapshot_name, mismatch in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                target_dir, source_dir, target = make_target_and_source(temporary)
                original = target_dir if snapshot_name == "target" else source_dir
                if mismatch == "repo":
                    moved = (
                        original.parent.parent.parent
                        / "models--example--wrong"
                        / "snapshots"
                        / original.name
                    )
                else:
                    moved = original.with_name("c" * 40)
                moved.parent.mkdir(parents=True, exist_ok=True)
                original.rename(moved)
                if snapshot_name == "target":
                    target_dir = moved
                else:
                    source_dir = moved

                with self.assertRaises(HybridError):
                    build_hybrid_view(target_dir, source_dir, temporary / "views", target)

                self.assertFalse((temporary / "views").exists())

    def test_hybrid_rejects_output_root_inside_upstream_snapshot(self):
        """A view root nested inside either immutable upstream snapshot is unsafe."""
        for label, snapshot_name in (("target", "target"), ("source", "source")):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                target_dir, source_dir, target = make_target_and_source(temporary)
                snapshot = target_dir if snapshot_name == "target" else source_dir
                output_root = snapshot / "generated" / "views"

                with self.assertRaises(HybridError):
                    build_hybrid_view(target_dir, source_dir, output_root, target)

                self.assertFalse(output_root.exists())

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
    target_dir = snapshot_path(tmp_path, target.repo_id, target.revision)
    source_dir = snapshot_path(
        tmp_path, target.ple_source.repo_id, target.ple_source.revision
    )
    target_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
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


def snapshot_path(tmp_path: Path, repo_id: str, revision: str) -> Path:
    return (
        tmp_path
        / "cache"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


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
