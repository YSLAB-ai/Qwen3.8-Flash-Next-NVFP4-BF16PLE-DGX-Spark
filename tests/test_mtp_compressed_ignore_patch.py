"""Tests for remapping compressed-tensors ignore paths in the MTP draft."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from src.patch_qwen4_exp_mtp_compressed_ignore import MARKER, patch_source


FIXTURE = '''
def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    draft_quant_config = get_draft_quant_config(vllm_config)
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen3_8FlashNextMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
'''.lstrip()


class MtpCompressedIgnorePatchTests(unittest.TestCase):
    def test_patch_is_exact_idempotent_and_syntax_valid(self):
        patched = patch_source(FIXTURE)

        self.assertIn(MARKER, patched)
        self.assertIn('getattr(draft_quant_config, "ignore", None)', patched)
        self.assertIn('"ignore",', patched)
        self.assertEqual(patch_source(patched), patched)
        ast.parse(patched)

    def test_patch_rejects_unexpected_upstream_layout(self):
        for source in ("def other():\n    pass\n", FIXTURE + FIXTURE):
            with self.subTest(length=len(source)), self.assertRaisesRegex(
                RuntimeError, "exactly one"
            ):
                patch_source(source)

    def test_image_build_and_container_smoke_require_the_patch(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        smoke = (root / "scripts" / "test-container.sh").read_text(encoding="utf-8")

        self.assertIn("patch_qwen4_exp_mtp_compressed_ignore.py", dockerfile)
        self.assertIn(MARKER, smoke)


if __name__ == "__main__":
    unittest.main()
