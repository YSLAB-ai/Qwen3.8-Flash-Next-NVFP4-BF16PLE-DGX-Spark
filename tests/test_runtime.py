"""Behavioral tests for the guarded serving command."""

from __future__ import annotations

import os
import posixpath
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from recipe.download import snapshot_path
from recipe.hybrid import build_hybrid_view
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
    path.mkdir(parents=True, exist_ok=True)
    return path


def hybrid_model_path(cache: Path) -> Path:
    target = load_manifest(ROOT / "compatibility.json").target("radixark")
    # The hybrid view has the deterministic location returned by local_target_path.
    from recipe.download import local_target_path

    path = local_target_path(target, cache)
    path.mkdir(parents=True, exist_ok=True)
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
        self.assertIn(f"{cache}:/recipe/{cache.name}:ro", joined)
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
            with self.assertRaisesRegex(RuntimeConfigurationError, "MTP"):
                build_docker_command(
                    orca_target(), model_path(cache), cache, RuntimeOptions(mtp=1)
                )

    def test_mtp_and_prewarm_are_rejected_even_with_unsafe_override(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for options, message in (
                (RuntimeOptions(mtp=1), "MTP"),
                (RuntimeOptions(prewarm=True), "prewarm"),
            ):
                with self.subTest(options=options), self.assertRaisesRegex(
                    RuntimeConfigurationError, message
                ):
                    validate_environment(options, True)
                with self.subTest(options=options), self.assertRaisesRegex(
                    RuntimeConfigurationError, message
                ):
                    build_docker_command(
                        orca_target(), model_path(cache), cache, options, unsafe_override=True
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

    def test_hybrid_mounts_only_the_recipe_views_subtree(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "hf-cache"
            target = load_manifest(ROOT / "compatibility.json").target("radixark")
            command = build_docker_command(target, hybrid_model_path(cache), cache, RuntimeOptions())

        mounts = docker_mounts(command)
        self.assertEqual(set(mounts), {cache, cache.parent / "recipe-views"})
        self.assertEqual(mounts[cache].parent, mounts[cache.parent / "recipe-views"].parent)
        self.assertNotIn(cache.parent, mounts)

    def test_every_hybrid_index_shard_resolves_under_rendered_mounts(self):
        """Host-relative view symlinks must retain their topology in the container."""
        from tests.test_hybrid import load_weight_map, make_target_and_source

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            target_snapshot, source_snapshot, target = make_target_and_source(temporary)
            cache = temporary / "cache"
            view = build_hybrid_view(
                target_snapshot,
                source_snapshot,
                temporary / "recipe-views",
                target,
            )
            command = build_docker_command(target, view, cache, RuntimeOptions())
            mounts = docker_mounts(command)
            image_index = command.index("qwen38-flash-dgx")
            model_in_container = PurePosixPath(command[image_index + 1])

            for filename in set(load_weight_map(view).values()):
                with self.subTest(filename=filename):
                    host_link = view / filename
                    link_in_container = model_in_container / filename
                    resolved_in_container = PurePosixPath(
                        posixpath.normpath(
                            str(link_in_container.parent / os.readlink(host_link))
                        )
                    )
                    resolved_on_host = host_link.resolve(strict=True)
                    expected = mounts[cache] / resolved_on_host.relative_to(cache)
                    self.assertEqual(resolved_in_container, expected)

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
            report = validate_environment(RuntimeOptions(context=131_072), True, minimum_free_bytes=2)

        self.assertEqual((report.mem_available_bytes, report.disk_free_bytes), (1, 1))
        self.assertGreaterEqual(len(report.warnings), 3)


def docker_mounts(command: list[str]) -> dict[Path, PurePosixPath]:
    mounts: dict[Path, PurePosixPath] = {}
    for index, token in enumerate(command):
        if token != "-v":
            continue
        host, container, mode = command[index + 1].rsplit(":", 2)
        if mode == "ro":
            mounts[Path(host)] = PurePosixPath(container)
    return mounts


if __name__ == "__main__":
    unittest.main()
