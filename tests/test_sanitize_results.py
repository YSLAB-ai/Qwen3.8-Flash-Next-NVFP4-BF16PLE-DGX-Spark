import json
import tempfile
import unittest
from pathlib import Path

from tools.sanitize_results import sanitize_result


class SanitizeResultsTests(unittest.TestCase):
    def test_sanitizer_keeps_measurements_and_removes_private_fields(self):
        """Dropping a timing, usage count, or private response is a publication bug."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(
                json.dumps(
                    {
                        "base_url": "https://private.example/v1",
                        "content": "private output",
                        "decode_tokens_per_second": 28.57,
                        "usage": {"prompt_tokens": 59, "completion_tokens": 68},
                        "local_path": "/home/person/model",
                    }
                ),
                encoding="utf-8",
            )

            clean = sanitize_result(source, root / "clean.json")

            serialized = json.dumps(clean)
            self.assertEqual(clean["decode_tokens_per_second"], 28.57)
            self.assertEqual(clean["usage"]["completion_tokens"], 68)
            self.assertNotIn("private.example", serialized)
            self.assertNotIn("/home/", serialized)
            self.assertNotIn("private output", serialized)

    def test_sanitizer_allows_only_safe_nested_evidence(self):
        """A new response or credential field must not leak through nested records."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(
                json.dumps(
                    {
                        "sampler": {
                            "temperature": 1.0,
                            "chat_template_kwargs": {
                                "enable_thinking": True,
                                "reasoning_effort": "medium",
                            },
                        },
                        "samples": [
                            {
                                "fixture": "reasoning",
                                "correct": True,
                                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                                "response": {"choices": [{"message": {"content": "secret"}}]},
                                "headers": {"Authorization": "Bearer secret"},
                                "request_id": "deployment-internal-123",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertEqual(clean["samples"][0]["fixture"], "reasoning")
            self.assertTrue(clean["samples"][0]["correct"])
            self.assertEqual(clean["samples"][0]["usage"]["prompt_tokens"], 12)
            serialized = json.dumps(clean)
            for forbidden in ("secret", "Authorization", "deployment-internal-123"):
                self.assertNotIn(forbidden, serialized)

    def test_sanitizer_converts_validated_output_to_a_pass_flag(self):
        """An exact-retrieval response may prove success without publishing the response."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(
                json.dumps(
                    {
                        "validated": "MOUNTAIN-CINDER-240079",
                        "response": {"choices": [{"message": {"content": "MOUNTAIN-CINDER-240079"}}]},
                    }
                ),
                encoding="utf-8",
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertTrue(clean["passed"])
            self.assertNotIn("MOUNTAIN-CINDER-240079", json.dumps(clean))

    def test_sanitizer_preserves_list_shaped_concurrency_results(self):
        """Concurrency waves need their per-stream timing, usage, and isolation status."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "validated": "EMBER-01-410001",
                                "usage": {"prompt_tokens": 8192, "completion_tokens": 64},
                                "ttft_seconds": 0.2,
                                "decode_seconds": 2.0,
                                "decode_tokens_per_second": 31.5,
                                "content_text": "private marker",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertTrue(clean["results"][0]["correct"])
            self.assertEqual(clean["results"][0]["usage"]["prompt_tokens"], 8192)
            self.assertEqual(clean["results"][0]["decode_tokens_per_second"], 31.5)
            self.assertNotIn("private marker", json.dumps(clean))

    def test_sanitizer_preserves_boolean_long_context_validation(self):
        """A boolean validation outcome is evidence, unlike the response text behind it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps({"validated": True}), encoding="utf-8")

            clean = sanitize_result(source, root / "clean.json")

            self.assertIs(clean["validated"], True)

    def test_sanitizer_only_retains_known_public_fixture_ids(self):
        """Fixture names are an allowlist, not an arbitrary safe-looking string."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"fixture": "file_edit", "correct": True},
                            {"fixture": "unreviewed_fixture", "correct": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            clean = sanitize_result(source, root / "clean.json")

            self.assertEqual(clean["samples"][0]["fixture"], "file_edit")
            self.assertNotIn("fixture", clean["samples"][1])

    def test_sanitizer_preserves_stability_probe_latency(self):
        """A stability probe's latency is a public timing measurement."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps({"probes": [{"latency_seconds": 1.25}]}), encoding="utf-8")

            clean = sanitize_result(source, root / "clean.json")

            self.assertEqual(clean["probes"][0]["latency_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
