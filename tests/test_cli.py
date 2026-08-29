"""Tests for the recipe command-line interface."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import shlex
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
        self.assertIn("orca-uncensored-bf16-mtp", text)
        self.assertIn("experimental", text)

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

    def test_login_and_download_share_the_requested_cache_contract(self):
        """CLI authentication must populate the same mounted cache used by download."""
        cache = Path("/safe/shared-hf-cache")
        with patch("recipe.cli.login") as authenticate:
            self.assertEqual(
                main(
                    ["login", "--cache", str(cache), "--image", "hf-client"],
                    manifest_path=ROOT / "compatibility.json",
                ),
                0,
            )
        authenticate.assert_called_once_with(cache, "hf-client")

        with patch("recipe.cli.download_target", return_value=Path("/safe/model")) as download, redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(
                main(
                    ["download", "inferact", "--cache", str(cache), "--image", "hf-client"],
                    manifest_path=ROOT / "compatibility.json",
                ),
                0,
            )
        self.assertEqual(download.call_args.args[1:], (cache, "hf-client"))

    def test_dry_run_renders_a_command_without_calling_docker(self):
        output = io.StringIO()
        with patch("recipe.cli.validate_environment") as validate, redirect_stdout(output):
            exit_code = main(
                ["dry-run", "orca-uncensored", "--unsafe-override"],
                manifest_path=ROOT / "compatibility.json",
            )

        self.assertEqual(exit_code, 0)
        validate.assert_not_called()
        command = output.getvalue()
        self.assertIn("qwen3.8-flash-next-orca-uncensored", command)
        self.assertIn("127.0.0.1:18300:8000", command)
        self.assertNotIn("HF_TOKEN", command)

    def test_dry_run_uses_requested_image(self):
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                exit_code = main(
                    [
                        "dry-run",
                        "orca-uncensored",
                        "--unsafe-override",
                        "--image",
                        "yslab-qwen38-flash-next-bf16ple:0.1.0-rc1",
                    ],
                    manifest_path=ROOT / "compatibility.json",
                )
            except SystemExit as exc:
                self.fail(f"dry-run rejected the requested image: {exc}")

        self.assertEqual(exit_code, 0)
        requested = "yslab-qwen38-flash-next-bf16ple:0.1.0-rc1"
        tokens = shlex.split(output.getvalue())
        self.assertEqual(tokens.count(requested), 1)
        self.assertNotIn("qwen38-flash-dgx", tokens)

    def test_serve_refuses_existing_container_without_replace(self):
        runner = _FakeRunner(existing=True)
        with patch("recipe.cli.validate_environment"), patch("recipe.cli.audit_checkpoint"):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                main(
                    ["serve", "orca-uncensored", "--unsafe-override"],
                    manifest_path=ROOT / "compatibility.json",
                    runner=runner,
                )

        self.assertEqual(runner.calls, [["docker", "container", "inspect", "qwen38-flash-orca-uncensored"]])

    def test_runtime_cli_rejects_mtp_on_original_target(self):
        with self.assertRaisesRegex(ValueError, "MTP"), redirect_stdout(io.StringIO()):
            main(
                ["dry-run", "orca-uncensored", "--unsafe-override", "--mtp", "1"],
                manifest_path=ROOT / "compatibility.json",
            )

    def test_runtime_cli_accepts_mtp_on_overlay_target(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "dry-run",
                    "orca-uncensored-bf16-mtp",
                    "--unsafe-override",
                    "--mtp",
                    "3",
                ],
                manifest_path=ROOT / "compatibility.json",
            )

        self.assertEqual(exit_code, 0)
        self.assertIn('num_speculative_tokens":3', output.getvalue())

    def test_runtime_cli_still_rejects_unknown_prewarm_flag(self):
        with self.assertRaises(SystemExit) as exited, patch(
            "sys.stderr", new=io.StringIO()
        ), redirect_stdout(io.StringIO()):
            main(
                ["dry-run", "orca-uncensored", "--unsafe-override", "--prewarm"],
                manifest_path=ROOT / "compatibility.json",
            )
        self.assertEqual(exited.exception.code, 2)


class _Result:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


class _FakeRunner:
    def __init__(self, *, existing: bool):
        self.existing = existing
        self.calls: list[list[str]] = []

    def __call__(self, command, **_kwargs):
        self.calls.append(list(command))
        if command[:3] == ["docker", "container", "inspect"]:
            return _Result(0 if self.existing else 1)
        raise AssertionError(f"unexpected runtime command: {command}")


if __name__ == "__main__":
    unittest.main()
