"""Tests for the pinned compatibility manifest."""

import json
import tempfile
import unittest
from pathlib import Path

from recipe.manifest import ManifestError, load_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_manifest_contains_only_approved_targets(self):
        manifest = load_manifest(ROOT / "compatibility.json")
        self.assertEqual(
            set(manifest.targets),
            {"orca-uncensored", "orca-uncensored-bf16-mtp", "inferact", "radixark"},
        )
        self.assertEqual(
            manifest.target("orca-uncensored").revision,
            "3a3b63161c0745390e5270179af42e46efc70799",
        )
        self.assertEqual(manifest.target("inferact").mode, "direct_bf16")
        self.assertEqual(manifest.target("radixark").mode, "hybrid_bf16")
        self.assertEqual(
            manifest.target("radixark").ple_source.repo_id,
            "Qwen/Qwen3.8-Flash-Next",
        )
        mtp = manifest.target("orca-uncensored-bf16-mtp")
        self.assertEqual(mtp.mode, "mtp_overlay")
        self.assertIsNotNone(mtp.mtp_source)
        self.assertEqual(mtp.mtp_source.repo_id, "RadixArk/Qwen3.8-Flash-Next-NVFP4")
        self.assertEqual(mtp.mtp_source.tensor_count, 31)
        self.assertEqual(mtp.mtp_source.dtype, "BF16")
        self.assertEqual(len(mtp.mtp_source.shards), 3)
        self.assertEqual(
            {shard.filename for shard in mtp.mtp_source.shards},
            {
                "model-bf16-00010.safetensors",
                "model-bf16-00011.safetensors",
                "model-bf16-00012.safetensors",
            },
        )

    def test_every_target_uses_the_production_ple_shape(self):
        for target in load_manifest(ROOT / "compatibility.json").targets.values():
            self.assertEqual(target.expected_ple.tensor_count, 128)
            self.assertEqual(target.expected_ple.total_rows, 320_001_536)
            self.assertEqual(target.expected_ple.width, 160)
            self.assertEqual(target.expected_ple.dtype, "BF16")

    def test_rejects_invalid_manifest_invariants(self):
        cases = {
            "unknown schema version": {"schema_version": 2},
            "non-integer schema version": {"schema_version": True},
            "floating-point schema version": {"schema_version": 1.0},
            "invalid revision": {
                "targets": [
                    _target(revision="A" * 40),
                ]
            },
            "duplicate alias": {"targets": [_target(), _target()]},
            "unknown mode": {"targets": [_target(mode="unsupported")]},
            "hybrid without source": {"targets": [_target(mode="hybrid_bf16")]},
            "direct with source": {
                "targets": [
                    _target(ple_source=_source()),
                ]
            },
            "mtp overlay without source": {"targets": [_target(mode="mtp_overlay")]},
            "direct with mtp source": {
                "targets": [_target(mtp_source=_mtp_source())]
            },
            "hybrid with mtp source": {
                "targets": [
                    _target(
                        mode="hybrid_bf16",
                        ple_source=_source(),
                        mtp_source=_mtp_source(),
                    )
                ]
            },
            "mtp overlay with ple source": {
                "targets": [
                    _target(
                        mode="mtp_overlay",
                        ple_source=_source(),
                        mtp_source=_mtp_source(),
                    )
                ]
            },
            "invalid mtp digest": {
                "targets": [
                    _target(mode="mtp_overlay", mtp_source=_mtp_source(sha256="bad"))
                ]
            },
            "duplicate mtp shard": {
                "targets": [
                    _target(
                        mode="mtp_overlay",
                        mtp_source=_mtp_source(duplicate=True),
                    )
                ]
            },
            "zero mtp size": {
                "targets": [
                    _target(mode="mtp_overlay", mtp_source=_mtp_source(size=0))
                ]
            },
            "unknown mtp source field": {
                "targets": [
                    _target(
                        mode="mtp_overlay",
                        mtp_source={**_mtp_source(), "unexpected": True},
                    )
                ]
            },
        }

        for label, changes in cases.items():
            with self.subTest(label=label):
                contents = _manifest()
                contents.update(changes)
                with self.assertRaises(ManifestError):
                    _load(contents)

    def test_target_rejects_unknown_alias(self):
        manifest = load_manifest(ROOT / "compatibility.json")

        with self.assertRaisesRegex(ManifestError, "unknown target: missing"):
            manifest.target("missing")


def _manifest():
    return {"schema_version": 1, "targets": [_target()]}


def _target(
    *,
    name="approved",
    revision="a" * 40,
    mode="direct_bf16",
    ple_source=None,
    mtp_source=None,
):
    return {
        "name": name,
        "repo_id": "example/approved",
        "revision": revision,
        "mode": mode,
        "served_model_name": "approved",
        "requires_auth": False,
        "ple_source": ple_source,
        "mtp_source": mtp_source,
        "minimum_free_bytes": 1,
        "expected_ple": {
            "tensor_count": 128,
            "total_rows": 320_001_536,
            "width": 160,
            "dtype": "BF16",
            "layer_id": 1,
            "split_parts": 128,
        },
    }


def _source():
    return {"repo_id": "example/source", "revision": "b" * 40}


def _mtp_source(*, sha256="c" * 64, size=123, duplicate=False):
    shards = [
        {
            "filename": "model-bf16-00010.safetensors",
            "size": size,
            "sha256": sha256,
        },
        {
            "filename": "model-bf16-00011.safetensors",
            "size": 456,
            "sha256": "d" * 64,
        },
    ]
    if duplicate:
        shards.append(dict(shards[0]))
    return {
        "repo_id": "example/mtp-source",
        "revision": "b" * 40,
        "tensor_count": 31,
        "dtype": "BF16",
        "shards": shards,
    }


def _load(contents):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(contents), encoding="utf-8")
        return load_manifest(path)


if __name__ == "__main__":
    unittest.main()
