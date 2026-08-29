"""Tests for safe, pinned Hugging Face downloads."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from recipe.download import (
    DownloadError,
    build_hf_download_command,
    build_hf_login_command,
    download_target,
    snapshot_path as downloaded_snapshot_path,
)
from recipe.manifest import ModelRef, PleExpectation, Target


def radixark_target() -> Target:
    return Target(
        name="radixark",
        repo_id="RadixArk/Qwen3.8-Flash-Next-NVFP4",
        revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        mode="hybrid_bf16",
        served_model_name="qwen3.8-flash-next",
        requires_auth=False,
        ple_source=ModelRef(
            "Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540"
        ),
        minimum_free_bytes=1,
        expected_ple=PleExpectation(128, 128, 160, "BF16", 1, 128),
    )


def direct_target() -> Target:
    return Target(
        name="direct",
        repo_id="example/direct",
        revision="a" * 40,
        mode="direct_bf16",
        served_model_name="direct",
        requires_auth=False,
        ple_source=None,
        minimum_free_bytes=10**30,
        expected_ple=PleExpectation(1, 1, 1, "BF16", 1, 1),
    )


class FakeRunner:
    def __init__(self, cache: Path, target: Target):
        self.cache = cache
        self.target = target
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, check: bool) -> None:
        self.assert_true(check)
        self.calls.append(command)
        if "model.safetensors.index.json" in command:
            source = self.target.ple_source
            assert source is not None
            snapshot = snapshot_path(self.cache, source)
            snapshot.mkdir(parents=True)
            prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
            weights = {
                f"{prefix}.shard_{index}.weight": (
                    "model-00005-of-00131.safetensors"
                    if index < 64
                    else "model-00037-of-00131.safetensors"
                )
                for index in range(128)
            }
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weights}), encoding="utf-8"
            )

    @staticmethod
    def assert_true(value: bool) -> None:
        if not value:
            raise AssertionError("runner must require successful subprocess completion")


class DownloadTests(unittest.TestCase):
    def test_snapshot_path_uses_hf_home_hub_layout(self):
        ref = ModelRef("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540")
        cache = Path("/safe/hf-cache")

        self.assertEqual(
            downloaded_snapshot_path(cache, ref),
            cache
            / "hub"
            / "models--Qwen--Qwen3.8-Flash-Next"
            / "snapshots"
            / "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        )

    def test_download_command_pins_revision_without_token(self):
        with tempfile.TemporaryDirectory() as directory:
            command = build_hf_download_command(
                ModelRef("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540"),
                ("config.json",),
                Path(directory),
                "qwen38-flash-dgx",
            )

        self.assertIsInstance(command, list)
        self.assertIn("--revision", command)
        self.assertIn("de4b8e4d43b917e7706784d8bb445c9af86a3540", command)
        joined = " ".join(command)
        self.assertNotIn("HF_TOKEN", joined)
        self.assertNotIn("Authorization", joined)

    def test_login_command_is_interactive_and_uses_writable_hf_home(self):
        with tempfile.TemporaryDirectory() as directory:
            command = build_hf_login_command(Path(directory), "qwen38-flash-dgx")

        self.assertIsInstance(command, list)
        self.assertIn("-it", command)
        self.assertIn("HF_HOME=/hf", command)
        self.assertTrue(any(mount.endswith(":/hf:rw") for mount in command))
        self.assertIn("auth", command)
        self.assertIn("login", command)
        self.assertNotIn("HF_TOKEN", " ".join(command))

    def test_radix_download_fetches_index_before_source_ple(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "hf-cache"
            target = radixark_target()
            fake_runner = FakeRunner(cache, target)
            expected_view = Path(directory) / "recipe-views" / "radixark"

            with patch("recipe.download.build_hybrid_view", return_value=expected_view) as build:
                result = download_target(target, cache, "qwen38-flash-dgx", runner=fake_runner)

        self.assertEqual(result, expected_view)
        calls = [" ".join(call) for call in fake_runner.calls]
        self.assertEqual(len(calls), 3)
        self.assertIn(target.repo_id, calls[0])
        self.assertIn("model.safetensors.index.json", calls[1])
        self.assertIn("model-00005-of-00131.safetensors", calls[2])
        self.assertIn("model-00037-of-00131.safetensors", calls[2])
        build.assert_called_once_with(
            snapshot_path(cache, target),
            snapshot_path(cache, target.ple_source),
            cache.parent / "recipe-views",
            target,
        )

    def test_gated_target_checks_small_file_access_before_full_download(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "hf-cache"
            target = replace(direct_target(), requires_auth=True, minimum_free_bytes=1)
            calls: list[list[str]] = []

            result = download_target(
                target,
                cache,
                "qwen38-flash-dgx",
                runner=lambda command, **_kwargs: calls.append(command),
            )

        self.assertEqual(result, downloaded_snapshot_path(cache, target))
        self.assertEqual(len(calls), 2)
        self.assertIn("config.json", calls[0])
        self.assertNotIn("config.json", calls[1])
        self.assertIn(target.repo_id, calls[0])
        self.assertIn(target.revision, calls[0])

    def test_gated_access_failure_explains_browser_acceptance_and_login(self):
        with tempfile.TemporaryDirectory() as directory:
            target = replace(direct_target(), requires_auth=True, minimum_free_bytes=1)

            def denied(command, **_kwargs):
                raise subprocess.CalledProcessError(1, command)

            with self.assertRaisesRegex(
                DownloadError,
                r"accept.*https://huggingface.co/example/direct.*recipe login",
            ):
                download_target(
                    target,
                    Path(directory) / "hf-cache",
                    "qwen38-flash-dgx",
                    runner=denied,
                )

    def test_hybrid_download_refuses_source_index_missing_an_expected_ple_name(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "hf-cache"
            target = radixark_target()

            def incomplete_index(command: list[str], *, check: bool) -> None:
                if "model.safetensors.index.json" not in command:
                    return
                source = target.ple_source
                assert source is not None
                snapshot = snapshot_path(cache, source)
                snapshot.mkdir(parents=True)
                (snapshot / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {}}), encoding="utf-8"
                )

            with self.assertRaises(DownloadError):
                download_target(target, cache, "qwen38-flash-dgx", runner=incomplete_index)

    def test_direct_download_rejects_low_cache_space_before_downloader_starts(self):
        """The target disk gate must run before the first direct download command."""
        with tempfile.TemporaryDirectory() as directory:
            calls: list[list[str]] = []
            target = direct_target()
            with self.assertRaisesRegex(DownloadError, "free disk"):
                download_target(
                    target,
                    Path(directory) / "hf-cache",
                    "qwen38-flash-dgx",
                    runner=lambda command, **_kwargs: calls.append(command),
                )

        self.assertEqual(calls, [])

    def test_hybrid_download_rejects_low_cache_space_before_downloader_starts(self):
        """Hybrid preparation must enforce its target gate before any source download."""
        with tempfile.TemporaryDirectory() as directory:
            calls: list[list[str]] = []
            target = replace(radixark_target(), minimum_free_bytes=10**30)
            with self.assertRaisesRegex(DownloadError, "free disk"):
                download_target(
                    target,
                    Path(directory) / "hf-cache",
                    "qwen38-flash-dgx",
                    runner=lambda command, **_kwargs: calls.append(command),
                )

        self.assertEqual(calls, [])


def snapshot_path(cache: Path, ref: ModelRef | Target) -> Path:
    return (
        cache
        / "hub"
        / f"models--{ref.repo_id.replace('/', '--')}"
        / "snapshots"
        / ref.revision
    )


if __name__ == "__main__":
    unittest.main()
