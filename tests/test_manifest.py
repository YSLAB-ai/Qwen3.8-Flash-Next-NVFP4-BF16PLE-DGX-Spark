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
        self.assertEqual(set(manifest.targets), {"orca-uncensored", "inferact", "radixark"})
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
):
    return {
        "name": name,
        "repo_id": "example/approved",
        "revision": revision,
        "mode": mode,
        "served_model_name": "approved",
        "requires_auth": False,
        "ple_source": ple_source,
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


def _load(contents):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(contents), encoding="utf-8")
        return load_manifest(path)


if __name__ == "__main__":
    unittest.main()
