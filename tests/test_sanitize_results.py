"""Tests for fail-closed benchmark evidence sanitization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.sanitize_results import sanitize_result


MODEL = "qwen3.8-flash-next-uncensored-nvfp4-bf16-ple-vllm"
REVISION = "3a3b63161c0745390e5270179af42e46efc70799"
SAMPLER = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    },
}


class SanitizeResultsTests(unittest.TestCase):
    def test_sanitizer_keeps_measurements_and_removes_private_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "base_url": "https://private.example/v1",
                    "content": "private output",
                    "decode_tokens_per_second": 28.57,
                    "usage": {"prompt_tokens": 59, "completion_tokens": 68},
                    "local_path": "/home/person/model",
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            serialized = json.dumps(clean)
            self.assertEqual(clean["decode_tokens_per_second"], 28.57)
            self.assertEqual(clean["usage"]["completion_tokens"], 68)
            self.assertNotIn("private.example", serialized)
            self.assertNotIn("/home/", serialized)
            self.assertNotIn("private output", serialized)

    def test_sanitizer_requires_trusted_checkpoint_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_summary(root, revision="f" * 40)
            source = write_raw(root, {"elapsed_seconds": 1.0}, write_trust=False)

            with self.assertRaisesRegex(ValueError, "revision"):
                sanitize_result(source, root / "clean.json")

    def test_sanitizer_requires_exact_source_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(root, {"model": "different-model"})

            with self.assertRaisesRegex(ValueError, "model"):
                sanitize_result(source, root / "clean.json")

    def test_sanitizer_rejects_sampler_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sampler = json.loads(json.dumps(SAMPLER))
            sampler["temperature"] = 0.1
            source = write_raw(root, {"sampler": sampler})

            with self.assertRaisesRegex(ValueError, "sampler"):
                sanitize_result(source, root / "clean.json")

    def test_sanitizer_preserves_source_boolean_verdicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "samples": [
                        {
                            "fixture": "reasoning",
                            "correct": False,
                            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                            "response": {"choices": [{"message": {"content": "private"}}]},
                        }
                    ]
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertIs(clean["samples"][0]["correct"], False)
            self.assertEqual(clean["samples"][0]["usage"]["prompt_tokens"], 12)
            self.assertNotIn("private", json.dumps(clean))

    def test_sanitizer_recomputes_known_functional_verdicts_from_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "results": {
                        "factual": {
                            "validated": "323",
                            "response": {"choices": [{"message": {"content": "323"}}]},
                        },
                        "json": {
                            "validated": {"product": 323},
                            "response": {
                                "choices": [{"message": {"content": '{"product": 323}'}}]
                            },
                        },
                    }
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertIs(clean["results"]["factual"]["passed"], True)
            self.assertIs(clean["results"]["json"]["passed"], True)
            self.assertNotIn("validated", json.dumps(clean))

    def test_sanitizer_does_not_turn_false_answers_into_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "answer": "not the expected answer",
                    "results": {
                        "factual": {
                            "validated": "323",
                            "response": {"choices": [{"message": {"content": "999"}}]},
                        }
                    },
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertNotIn("passed", clean)
            self.assertIs(clean["results"]["factual"]["passed"], False)

    def test_sanitizer_rejects_unverifiable_string_and_dict_verdicts(self):
        for verdict in ("looks-valid", {"looks": "valid"}):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = write_raw(root, {"validated": verdict})

                with self.assertRaisesRegex(ValueError, "unverifiable"):
                    sanitize_result(source, root / "clean.json")

    def test_sanitizer_preserves_list_shaped_concurrency_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "results": [
                        {
                            "marker": "EMBER-ALPHA-410001",
                            "validated": "EMBER-ALPHA-410001",
                            "usage": {"prompt_tokens": 8192, "completion_tokens": 64},
                            "ttft_seconds": 0.2,
                            "decode_seconds": 2.0,
                            "decode_tokens_per_second": 31.5,
                            "content_text": "EMBER-ALPHA-410001",
                        }
                    ]
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertTrue(clean["results"][0]["correct"])
            self.assertEqual(clean["results"][0]["usage"]["prompt_tokens"], 8192)
            self.assertEqual(clean["results"][0]["decode_tokens_per_second"], 31.5)
            self.assertNotIn("EMBER-ALPHA-410001", json.dumps(clean))

    def test_sanitizer_marks_false_concurrency_content_incorrect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "results": [
                        {
                            "marker": "EMBER-ALPHA-410001",
                            "validated": "EMBER-ALPHA-410001",
                            "content_text": "EMBER-BRAVO-410002",
                        }
                    ]
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertIs(clean["results"][0]["correct"], False)

    def test_sanitizer_preserves_boolean_long_context_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(root, {"validated": False})

            clean = sanitize_result(source, root / "clean.json")

            self.assertIs(clean["validated"], False)

    def test_sanitizer_only_retains_known_public_fixture_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "samples": [
                        {"fixture": "file_edit", "correct": True},
                        {"fixture": "unreviewed_fixture", "correct": True},
                    ]
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertEqual(clean["samples"][0]["fixture"], "file_edit")
            self.assertNotIn("fixture", clean["samples"][1])

    def test_sanitizer_preserves_stability_probe_latency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(root, {"probes": [{"latency_seconds": 1.25}]})

            clean = sanitize_result(source, root / "clean.json")

            self.assertEqual(clean["probes"][0]["latency_seconds"], 1.25)

    def test_sanitizer_recomputes_stability_factual_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw(
                root,
                {
                    "probes": [
                        {"answer": "323", "latency_seconds": 1.0},
                        {"answer": "999", "latency_seconds": 1.0},
                    ]
                },
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertIs(clean["probes"][0].get("passed"), True)
            self.assertIs(clean["probes"][1].get("passed"), False)


def write_raw(
    root: Path, value: dict[str, object], *, write_trust: bool = True
) -> Path:
    if write_trust:
        write_summary(root)
    payload = {"model": MODEL, "sampler": SAMPLER, **value}
    source = root / "raw.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source


def write_summary(root: Path, *, revision: str = REVISION) -> Path:
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "model": MODEL,
                "checkpoint": {
                    "repository": "orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4",
                    "revision": revision,
                },
                "selected_profile": {
                    "context_tokens": 262144,
                    "max_num_seqs": 2,
                    "gpu_memory_utilization": 0.8,
                    "mtp_depth": 0,
                    "ple_mmap": True,
                    "ple_prewarm": False,
                    "thinking": True,
                    "reasoning_effort": "medium",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 0.0,
                    "repetition_penalty": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    unittest.main()
