"""Behavioral tests for the guarded serving command."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recipe.download import snapshot_path
from recipe.manifest import ModelRef, load_manifest
from recipe.runtime import (
    RuntimeConfigurationError,
    RuntimeOptions,
    build_docker_command,
    validate_environment,
)


ROOT = Path(__file__).resolve().parents[1]


def orca_target():
    return load_manifest(ROOT / "compatibility.json").target("orca-uncensored")


def model_path(cache: Path) -> Path:
    target = orca_target()
    path = snapshot_path(cache, ModelRef(target.repo_id, target.revision))
    path.mkdir(parents=True)
    return path


class RuntimeTests(unittest.TestCase):
    def test_defaults_match_validated_profile(self):
        options = RuntimeOptions()

        self.assertEqual((options.context, options.sequences, options.gpu_memory), (262_144, 8, 0.80))
        self.assertEqual(options.mtp, 0)
        self.assertFalse(options.prewarm)
        self.assertEqual(options.bind, "127.0.0.1")

    def test_command_is_loopback_read_only_and_non_restarting(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            command = build_docker_command(orca_target(), model_path(cache), cache, RuntimeOptions())

        joined = " ".join(command)
        self.assertIn("127.0.0.1:18300:8000", joined)
        self.assertIn(":/hf:ro", joined)
        self.assertIn("--restart no", joined)
        self.assertIn("--gpu-memory-utilization 0.8", joined)
        self.assertIn("--max-num-seqs 8", joined)
        self.assertNotIn("speculative-config", joined)
        self.assertIn("--served-model-name qwen3.8-flash-next-orca-uncensored", joined)
        self.assertIn("VLLM_PLE_MMAP_WORKERS=32", joined)
        self.assertIn("VLLM_PLE_MMAP_PREWARM=0", joined)
        self.assertIn("--kv-cache-dtype bfloat16", joined)
        self.assertIn("--no-enable-prefix-caching", joined)
        self.assertIn("--enable-chunked-prefill", joined)
        self.assertIn("-cc.cudagraph_mode=PIECEWISE", joined)
        self.assertIn("vllm::ple_mmap_lookup", joined)

    def test_rejects_unvalidated_options_without_override(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with self.assertRaisesRegex(RuntimeConfigurationError, "validated"):
                build_docker_command(
                    orca_target(), model_path(cache), cache, RuntimeOptions(mtp=1)
                )

    def test_rejects_non_loopback_bind_without_override(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with self.assertRaisesRegex(RuntimeConfigurationError, "loopback"):
                build_docker_command(
                    orca_target(), model_path(cache), cache, RuntimeOptions(bind="0.0.0.0")
                )

    def test_rejects_remote_cache_path(self):
        target = orca_target()
        remote_cache = Path("s3://models")
        remote_model = snapshot_path(remote_cache, ModelRef(target.repo_id, target.revision))

        with self.assertRaisesRegex(RuntimeConfigurationError, "remote"):
            build_docker_command(target, remote_model, remote_cache, RuntimeOptions())

    def test_environment_rejects_low_memory_and_target_disk_without_override(self):
        options = RuntimeOptions()
        with patch("recipe.runtime._mem_available_bytes", return_value=99 * 1024**3), patch(
            "recipe.runtime._disk_free_bytes", return_value=100 * 1024**3
        ):
            with self.assertRaisesRegex(RuntimeConfigurationError, "MemAvailable"):
                validate_environment(options, False, minimum_free_bytes=101 * 1024**3)

        with patch("recipe.runtime._mem_available_bytes", return_value=100 * 1024**3), patch(
            "recipe.runtime._disk_free_bytes", return_value=100 * 1024**3
        ):
            with self.assertRaisesRegex(RuntimeConfigurationError, "free disk"):
                validate_environment(options, False, minimum_free_bytes=101 * 1024**3)

    def test_environment_reports_warnings_when_unsafe_override_is_explicit(self):
        with patch("recipe.runtime._mem_available_bytes", return_value=1), patch(
            "recipe.runtime._disk_free_bytes", return_value=1
        ):
            report = validate_environment(RuntimeOptions(mtp=1), True, minimum_free_bytes=2)

        self.assertEqual((report.mem_available_bytes, report.disk_free_bytes), (1, 1))
        self.assertGreaterEqual(len(report.warnings), 3)


if __name__ == "__main__":
    unittest.main()
