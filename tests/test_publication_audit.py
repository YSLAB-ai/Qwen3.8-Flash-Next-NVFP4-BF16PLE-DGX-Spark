"""Tests for the tracked-file publication audit."""

from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from tools.audit_publication import audit_tree, git_tracked_files, main


ROOT = Path(__file__).resolve().parents[1]


class PublicationAuditTests(unittest.TestCase):
    def test_audit_rejects_model_weights(self):
        # publication-audit: allow-test-fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "model.safetensors"
            weight.write_bytes(b"fixture")

            findings = audit_tree(root, [weight])

            self.assertIn("model-weight", {item.code for item in findings})

    def test_audit_rejects_private_endpoint_and_secret_without_echoing_value(self):
        # publication-audit: allow-test-fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "config.txt"
            bad.write_text("API_KEY=secret-value\nhttps://llm.labtools.studio\n", encoding="utf-8")

            findings = audit_tree(root, [bad])

            self.assertGreaterEqual(
                {item.code for item in findings}, {"secret-assignment", "private-term"}
            )
            self.assertNotIn("secret-value", "\n".join(item.detail for item in findings))

    def test_audit_covers_every_binding_private_term_without_echoing_it(self):
        # publication-audit: allow-test-fixture
        private_values = (
            "https://llm.labtools.studio/v1",
            "cloudflared tunnel run private-model",
            "/home/yiwen/.config/systemd/user/private-model.service",
            "OpenCode workstation configuration",
            "Palworld service configuration",
        )
        for private_value in private_values:
            with self.subTest(private_value=private_value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bad = root / "config.txt"
                bad.write_text(private_value, encoding="utf-8")

                findings = audit_tree(root, [bad])

                self.assertIn("private-term", {item.code for item in findings})
                self.assertNotIn(
                    private_value,
                    "\n".join(item.detail for item in findings),
                )

    def test_audit_rejects_external_symlink_and_oversized_tracked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "publication-audit-outside.txt"
            outside.write_text("outside", encoding="utf-8")
            link = root / "linked.txt"
            link.symlink_to(outside)
            oversized = root / "oversized.txt"
            oversized.write_bytes(b"x" * (10 * 1024 * 1024 + 1))

            findings = audit_tree(root, [link, oversized])

            self.assertGreaterEqual(
                {item.code for item in findings}, {"external-symlink", "oversized-file"}
            )

    def test_marker_only_allows_root_test_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_module = root / "tests" / "test_fixture.py"
            test_module.parent.mkdir()
            test_module.write_text(
                "# publication-audit: allow-test-fixture\nAPI_KEY=fixture_value\n",
                encoding="utf-8",
            )
            test_helper = root / "tests" / "fixture.py"
            test_helper.write_text(
                "# publication-audit: allow-test-fixture\nAPI_KEY=fixture_value\n",
                encoding="utf-8",
            )
            nested_helper = root / "src" / "tests" / "test_helper.py"
            nested_helper.parent.mkdir(parents=True)
            nested_helper.write_text(
                "# publication-audit: allow-test-fixture\nAPI_KEY=fixture_value\n",
                encoding="utf-8",
            )

            allowed = audit_tree(root, [test_module])
            rejected = audit_tree(root, [test_helper, nested_helper])

            self.assertEqual(allowed, [])
            self.assertEqual(
                {item.path for item in rejected if item.code == "secret-assignment"},
                {Path("tests/fixture.py"), Path("src/tests/test_helper.py")},
            )

    def test_cli_root_defaults_to_tracked_files_and_all_files_scans_untracked(self):
        # publication-audit: allow-test-fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("public", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            untracked = root / "ignored-local.txt"
            untracked.write_text("Cloudflare private configuration", encoding="utf-8")

            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--root", str(root)]), 0)
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(main(["--root", str(root), "--all-files"]), 1)

        self.assertIn("private-term: ignored-local.txt", errors.getvalue())
        self.assertNotIn("Cloudflare private configuration", errors.getvalue())

    def test_cli_rejects_unknown_arguments(self):
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as exited:
            main(["--unknown-option"])

        self.assertEqual(exited.exception.code, 2)

    def test_all_files_reports_external_symlink_without_following_it(self):
        # publication-audit: allow-test-fixture
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(directory)
            outside = Path(outside_dir) / "private.txt"
            outside.write_text("Cloudflare private configuration", encoding="utf-8")
            (root / "external").symlink_to(Path(outside_dir), target_is_directory=True)
            errors = io.StringIO()

            with redirect_stderr(errors):
                self.assertEqual(main(["--root", str(root), "--all-files"]), 1)

        self.assertIn("external-symlink: external", errors.getvalue())
        self.assertNotIn("private.txt", errors.getvalue())
        self.assertNotIn("Cloudflare private configuration", errors.getvalue())

    def test_current_public_files_are_publishable(self):
        files = [
            path
            for path in git_tracked_files(ROOT)
            if not path.is_relative_to(ROOT / "docs/superpowers")
        ]

        self.assertEqual(audit_tree(ROOT, files), [])


if __name__ == "__main__":
    unittest.main()
