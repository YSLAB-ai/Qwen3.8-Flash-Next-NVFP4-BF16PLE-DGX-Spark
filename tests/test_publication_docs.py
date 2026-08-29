"""Contract tests for the public documentation set."""

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
        self.assertIn("Inferact", docs)
        self.assertIn("runtime unvalidated", docs)
        self.assertIn("RadixArk", docs)
        self.assertIn("runtime unvalidated", docs)

    def test_private_deployment_terms_are_absent(self) -> None:
        forbidden = ("llm.labtools", "Cloudflare", "/home/yiwen", "OpenCode", "Palworld")
        docs = "\n".join(path.read_text() for path in PUBLIC_DOCS).lower()
        self.assertFalse([term for term in forbidden if term.lower() in docs])


if __name__ == "__main__":
    unittest.main()
