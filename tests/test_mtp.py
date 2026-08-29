"""Tests for the immutable BF16 MTP checkpoint overlay."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recipe.audit import audit_checkpoint
from recipe.download import local_target_path
from recipe.manifest import MtpShard, MtpSource, PleExpectation, Target
from recipe.mtp import (
    MTP_QUANTIZATION_IGNORE,
    MTP_TENSOR_NAMES,
    MtpOverlayError,
    build_mtp_overlay,
)
from recipe.safetensors import read_header
from tests.test_safetensors_audit import write_safetensors


class MtpOverlayTests(unittest.TestCase):
    def test_builds_exact_bf16_mtp_overlay_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_snapshot, source_snapshot, target = make_inputs(root)

            view = build_mtp_overlay(
                target_snapshot, source_snapshot, root / "recipe-views", target
            )
            repeated = build_mtp_overlay(
                target_snapshot, source_snapshot, root / "recipe-views", target
            )

            self.assertEqual(repeated, view)
            self.assertEqual(view, local_target_path(target, root / "cache"))
            index = load_json(view / "model.safetensors.index.json")["weight_map"]
            self.assertEqual(
                {name for name in index if name.startswith("mtp.")},
                set(MTP_TENSOR_NAMES),
            )
            compact = view / "mtp-bf16.safetensors"
            self.assertEqual(set(read_header(compact)), set(MTP_TENSOR_NAMES))
            self.assertEqual({meta.dtype for meta in read_header(compact).values()}, {"BF16"})
            self.assertTrue((view / index[TRUNK]).is_symlink())
            self.assertFalse(Path((view / index[TRUNK]).readlink()).is_absolute())

            source_config = load_json(target_snapshot / "config.json")
            output_config = load_json(view / "config.json")
            self.assertEqual(
                output_config["quantization_config"]["ignore"],
                source_config["quantization_config"]["ignore"]
                + list(MTP_QUANTIZATION_IGNORE),
            )
            self.assertNotIn("mtp.*", output_config["quantization_config"]["ignore"])
            self.assertNotIn(
                "model.mtp.*", output_config["quantization_config"]["ignore"]
            )
            self.assertIn(
                "mtp.layers.0.mlp.experts",
                output_config["quantization_config"]["ignore"],
            )
            source_config["quantization_config"].pop("ignore")
            output_config["quantization_config"].pop("ignore")
            self.assertEqual(output_config, source_config)

            metadata = load_json(view / "recipe-metadata.json")
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["target"]["revision"], target.revision)
            self.assertEqual(metadata["mtp_source"]["revision"], target.mtp_source.revision)
            self.assertEqual(metadata["mtp_tensor_names"], list(MTP_TENSOR_NAMES))
            self.assertEqual(
                metadata["compact_shard"]["sha256"], sha256(compact)
            )
            audit_checkpoint(view, target, (root / "cache", root / "recipe-views"))

    def test_rejects_source_hash_change_before_finalizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_snapshot, source_snapshot, target = make_inputs(root)
            first_shard = source_snapshot / target.mtp_source.shards[0].filename
            first_shard.write_bytes(first_shard.read_bytes() + b"tamper")

            with self.assertRaisesRegex(MtpOverlayError, "size|SHA-256"):
                build_mtp_overlay(
                    target_snapshot, source_snapshot, root / "recipe-views", target
                )

            self.assertFalse(any((root / "recipe-views").glob("**/[!.]*")))

    def test_rejects_existing_overlay_with_tampered_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_snapshot, source_snapshot, target = make_inputs(root)
            view = build_mtp_overlay(
                target_snapshot, source_snapshot, root / "recipe-views", target
            )
            (view / "recipe-metadata.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(MtpOverlayError, "metadata"):
                build_mtp_overlay(
                    target_snapshot, source_snapshot, root / "recipe-views", target
                )

    def test_rejects_noncanonical_snapshots_and_nested_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_snapshot, source_snapshot, target = make_inputs(root)
            moved = root / "arbitrary-target"
            target_snapshot.rename(moved)
            with self.assertRaisesRegex(MtpOverlayError, "identity"):
                build_mtp_overlay(moved, source_snapshot, root / "views", target)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_snapshot, source_snapshot, target = make_inputs(root)
            with self.assertRaisesRegex(MtpOverlayError, "output root"):
                build_mtp_overlay(
                    target_snapshot,
                    source_snapshot,
                    target_snapshot / "generated",
                    target,
                )


TRUNK = "model.language_model.layers.0.mlp.weight"
PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"


def make_inputs(root: Path) -> tuple[Path, Path, Target]:
    target_repo = "example/orca"
    source_repo = "example/radix"
    target_revision = "a" * 40
    source_revision = "b" * 40
    target_snapshot = snapshot(root, target_repo, target_revision)
    source_snapshot = snapshot(root, source_repo, source_revision)
    target_snapshot.mkdir(parents=True)
    source_snapshot.mkdir(parents=True)

    target_shard = target_snapshot / "model-00001.safetensors"
    write_safetensors(
        target_shard,
        {
            TRUNK: {"dtype": "F32", "shape": [1]},
            PLE: {"dtype": "BF16", "shape": [2, 3]},
        },
    )
    write_json(
        target_snapshot / "model.safetensors.index.json",
        {"weight_map": {TRUNK: target_shard.name, PLE: target_shard.name}},
    )
    write_json(
        target_snapshot / "config.json",
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "ple_layer_ids": [2],
                "split_ngram_parts": 1,
                "ngram_size": 2,
                "heads_per_ngram": 1,
                "ple_embed_dim": 3,
                "mtp_num_hidden_layers": 1,
            },
            "quantization_config": {
                "format": "mixed-precision",
                "ignore": ["existing.module"],
            },
        },
    )
    (target_snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    filenames = (
        "model-bf16-00010.safetensors",
        "model-bf16-00011.safetensors",
        "model-bf16-00012.safetensors",
    )
    groups = (MTP_TENSOR_NAMES[:10], MTP_TENSOR_NAMES[10:20], MTP_TENSOR_NAMES[20:])
    source_map: dict[str, str] = {}
    shard_specs: list[MtpShard] = []
    for index, (filename, names) in enumerate(zip(filenames, groups, strict=True)):
        tensors = {name: {"dtype": "BF16", "shape": [index + 1, 2]} for name in names}
        if index == 0:
            tensors["source.ignored.weight"] = {"dtype": "F32", "shape": [1]}
        path = source_snapshot / filename
        write_safetensors(path, tensors)
        source_map.update({name: filename for name in tensors})
        shard_specs.append(MtpShard(filename, path.stat().st_size, sha256(path)))
    write_json(source_snapshot / "model.safetensors.index.json", {"weight_map": source_map})

    target = Target(
        name="orca-bf16-mtp",
        repo_id=target_repo,
        revision=target_revision,
        mode="mtp_overlay",
        served_model_name="qwen3.8-flash-next",
        requires_auth=True,
        ple_source=None,
        minimum_free_bytes=1,
        expected_ple=PleExpectation(1, 2, 3, "BF16", 1, 1),
        mtp_source=MtpSource(
            source_repo,
            source_revision,
            len(MTP_TENSOR_NAMES),
            "BF16",
            tuple(shard_specs),
        ),
    )
    return target_snapshot, source_snapshot, target


def snapshot(root: Path, repo_id: str, revision: str) -> Path:
    return (
        root
        / "cache"
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
