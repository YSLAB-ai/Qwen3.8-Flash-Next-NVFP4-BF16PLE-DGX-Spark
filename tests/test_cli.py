"""Tests for the recipe command-line interface."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from recipe.cli import main
from recipe.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_list_prints_all_targets_and_validation_states(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["list"], manifest_path=ROOT / "compatibility.json")

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("orca-uncensored", text)
        self.assertIn("runtime-validated", text)
        self.assertIn("inferact", text)
        self.assertIn("structure-audited", text)
        self.assertIn("radixark", text)
        self.assertIn("hybrid", text)

    def test_path_prints_the_pinned_direct_snapshot_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["path", "inferact", "--cache", directory],
                    manifest_path=ROOT / "compatibility.json",
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("models--Inferact--Qwen3.8-Flash-Next-NVFP4", output.getvalue())
        self.assertIn("103a7608316173ca6edd49929544244de7ffda70", output.getvalue())

    def test_prepare_downloads_then_audits_the_resolved_path(self):
        target = load_manifest(ROOT / "compatibility.json").target("inferact")
        resolved = Path("/safe/cache/snapshot")
        with patch("recipe.cli.download_target", return_value=resolved) as download, patch(
            "recipe.cli.audit_checkpoint"
        ) as audit, redirect_stdout(io.StringIO()):
            exit_code = main(["prepare", target.name, "--cache", "/safe/cache"], manifest_path=ROOT / "compatibility.json")

        self.assertEqual(exit_code, 0)
        download.assert_called_once()
        audit.assert_called_once_with(resolved, target, (Path("/safe/cache"),))


if __name__ == "__main__":
    unittest.main()
