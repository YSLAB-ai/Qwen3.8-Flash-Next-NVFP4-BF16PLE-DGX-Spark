"""Tests for the fail-closed native MTP checkpoint-load patch."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from recipe.mtp import MTP_TENSOR_NAMES
from src.patch_qwen4_exp_mtp_load_guard import (
    MARKER,
    PATCH_MTP_TENSOR_NAMES,
    patch_source,
    validate_observed,
)


FIXTURE = '''
@support_torch_compile(
    dynamic_arg_dims={}
)
class Qwen3_8FlashNextMTP(nn.Module, SupportsPP, Qwen3_8FlashNextMixtureOfExperts):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(remap_weight_names())
'''.lstrip()


class MtpLoadGuardPatchTests(unittest.TestCase):
    def test_patch_is_exact_idempotent_and_syntax_valid(self):
        patched = patch_source(FIXTURE)

        self.assertIn(MARKER, patched)
        self.assertIn("_qwen38_native_mtp_checkpoint", patched)
        self.assertIn("_qwen38_validate_native_mtp_checkpoint", patched)
        self.assertIn("loaded = loader.load_weights(remap_weight_names())", patched)
        self.assertLess(patched.index(MARKER), patched.index("@support_torch_compile"))
        self.assertEqual(patch_source(patched), patched)
        ast.parse(patched)

    def test_patch_tensor_names_cannot_drift_from_overlay_builder(self):
        self.assertEqual(PATCH_MTP_TENSOR_NAMES, MTP_TENSOR_NAMES)
        self.assertEqual(len(PATCH_MTP_TENSOR_NAMES), 31)

    def test_validation_requires_exact_names_and_bf16(self):
        valid = {name: "torch.bfloat16" for name in PATCH_MTP_TENSOR_NAMES}
        validate_observed(valid)

        cases = {
            "missing": {name: dtype for name, dtype in valid.items() if name != MTP_TENSOR_NAMES[0]},
            "unexpected": {**valid, "mtp.unexpected.weight": "torch.bfloat16"},
            "dtype": {**valid, MTP_TENSOR_NAMES[0]: "torch.float16"},
        }
        for message, observed in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                validate_observed(observed)

    def test_patch_rejects_unexpected_upstream_layout(self):
        for source in ("class Other:\n    pass\n", FIXTURE + FIXTURE):
            with self.subTest(length=len(source)), self.assertRaisesRegex(
                RuntimeError, "exactly one"
            ):
                patch_source(source)

    def test_image_build_and_container_smoke_test_require_the_guard(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        smoke = (root / "scripts" / "test-container.sh").read_text(encoding="utf-8")

        self.assertIn("patch_qwen4_exp_mtp_load_guard.py", dockerfile)
        self.assertIn("${QWEN_MTP}", dockerfile)
        self.assertIn(MARKER, smoke)


if __name__ == "__main__":
    unittest.main()
