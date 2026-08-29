"""Contract tests for the public documentation set."""

# publication-audit: allow-test-fixture

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HF_CARD = ROOT / "huggingface" / "README.md"
PUBLIC_DOCS = (
    README,
    ROOT / "docs" / "HOW-IT-WORKS.md",
    ROOT / "docs" / "COMPATIBILITY.md",
    ROOT / "docs" / "BENCHMARKS.md",
    ROOT / "docs" / "SWE-BENCH-PILOT.md",
    ROOT / "docs" / "TROUBLESHOOTING.md",
    HF_CARD,
)


class PublicationDocumentationTests(unittest.TestCase):
    def test_readmes_open_with_approved_sentence(self) -> None:
        expected = (
            "A recipe to run NVFP4 Qwen3.8-Flash-Next with its full-precision BF16 "
            "PLE memory-mapped from NVMe on a single NVIDIA DGX Spark (GB10)."
        )
        self.assertIn(expected, "\n".join(README.read_text().splitlines()[:8]))
        self.assertIn(expected, "\n".join(HF_CARD.read_text().splitlines()[:16]))

    def test_only_orcarouter_is_runtime_validated(self) -> None:
        docs = "\n".join(path.read_text() for path in PUBLIC_DOCS)
        self.assertIn("Measured results - Orcarouter checkpoint only", docs)
        readme_rows = README.read_text(encoding="utf-8").splitlines()
        compatibility_rows = (ROOT / "docs" / "COMPATIBILITY.md").read_text(
            encoding="utf-8"
        ).splitlines()
        for target in ("Inferact", "RadixArk"):
            with self.subTest(target=target, document="README"):
                row = next(line for line in readme_rows if line.startswith(f"| {target} |"))
                self.assertIn("runtime unvalidated", row)
            with self.subTest(target=target, document="compatibility"):
                row = next(
                    line for line in compatibility_rows if line.startswith(f"| {target} |")
                )
                self.assertIn("runtime unvalidated", row)

    def test_quick_start_uses_publication_clone_and_recipe_login(self) -> None:
        """Quick-start auth must populate the recipe-local cache used by downloads."""
        docs = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
        expected_url = (
            "https://github.com/YSLAB-ai/"
            "Qwen3.8-Flash-Next-NVFP4-BF16PLE-DGX-Spark.git"
        )
        self.assertIn(f"git clone {expected_url}", README.read_text(encoding="utf-8"))
        self.assertIn("scripts/qwen38-dgx-spark login", docs)
        self.assertNotIn("github.com/blazux/qwen3.8-Flash-DGX", docs)
        self.assertNotIn("huggingface-cli login", docs)

    def test_gated_orcarouter_access_is_an_explicit_browser_prerequisite(self) -> None:
        docs = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
        gate_url = (
            "https://huggingface.co/orcarouter/"
            "Qwen3.8-Flash-Next-Uncensored-NVFP4"
        )
        self.assertIn(gate_url, README.read_text(encoding="utf-8"))
        self.assertIn("accept", docs.lower())
        self.assertIn("share your contact information", docs.lower())
        self.assertIn("scripts/qwen38-dgx-spark login", docs)

    def test_hugging_face_card_contains_the_complete_readme(self) -> None:
        card = HF_CARD.read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\n"))
        _opening, separator, body = card[4:].partition("\n---\n")
        self.assertTrue(separator)
        self.assertEqual(body.lstrip("\n"), README.read_text(encoding="utf-8"))

    def test_troubleshooting_shows_target_specific_replace_command(self) -> None:
        """Recovery guidance must include the target required by the serve parser."""
        troubleshooting = (ROOT / "docs" / "TROUBLESHOOTING.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "scripts/qwen38-dgx-spark serve orca-uncensored-bf16-mtp --mtp 2 --replace",
            troubleshooting,
        )

    def test_bf16_mtp_overlay_is_the_documented_qualified_default(self) -> None:
        docs = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
        self.assertIn("orca-uncensored-bf16-mtp", docs)
        self.assertIn("31/31", docs)
        self.assertIn("MTP=2", docs)
        self.assertIn("57.13 tok/s", docs)
        self.assertIn("240,051 prompt tokens", docs)
        for stale_claim in (
            "requires `MTP=0`",
            "MTP depths 2-4 | Not run",
            "future graft experiment is not validated",
        ):
            self.assertNotIn(stale_claim, docs)

    def test_mtp_sweep_has_a_machine_readable_record(self) -> None:
        import json

        record = json.loads(
            (ROOT / "results" / "orcarouter" / "mtp-bf16-sweep.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(record["selected_depth"], 2)
        self.assertEqual([row["depth"] for row in record["results"]], [0, 1, 2, 3, 4, 5, 6, 8, 10])
        selected = next(row for row in record["results"] if row["depth"] == 2)
        self.assertEqual(selected["median_decode_tokens_per_second"], 57.1303)
        self.assertEqual(record["mtp_overlay"]["loaded_tensors"], 31)

    def test_private_deployment_terms_are_absent(self) -> None:
        forbidden = ("llm.labtools", "Cloudflare", "/home/yiwen", "OpenCode", "Palworld")
        docs = "\n".join(path.read_text() for path in PUBLIC_DOCS).lower()
        self.assertFalse([term for term in forbidden if term.lower() in docs])


if __name__ == "__main__":
    unittest.main()
