"""Tests for the tracked-file publication audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.audit_publication import audit_tree, git_tracked_files


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

    def test_marker_only_allows_deliberate_fixture_in_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_fixture = root / "tests" / "fixture.py"
            test_fixture.parent.mkdir()
            test_fixture.write_text(
                "# publication-audit: allow-test-fixture\nAPI_KEY=fixture_value\n",
                encoding="utf-8",
            )
            non_test_fixture = root / "config.py"
            non_test_fixture.write_text(
                "# publication-audit: allow-test-fixture\nAPI_KEY=fixture_value\n",
                encoding="utf-8",
            )

            allowed = audit_tree(root, [test_fixture])
            rejected = audit_tree(root, [non_test_fixture])

            self.assertEqual(allowed, [])
            self.assertIn("secret-assignment", {item.code for item in rejected})

    def test_current_public_files_are_publishable(self):
        files = [
            path
            for path in git_tracked_files(ROOT)
            if not path.is_relative_to(ROOT / "docs/superpowers")
        ]

        self.assertEqual(audit_tree(ROOT, files), [])


if __name__ == "__main__":
    unittest.main()
