"""Focused behavior tests for public benchmark validators and scheduling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from functional import validate  # noqa: E402
import common  # noqa: E402
import long_context  # noqa: E402
import stability  # noqa: E402


class _StreamResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self._lines)


class StreamCompletionMetricTests(unittest.TestCase):
    def test_reasoning_and_visible_output_use_separate_token_clocks(self):
        """Hidden tokens must never be divided by visible-content-only time."""
        events = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"A"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"B"}}]}\n',
            (
                b'data: {"choices":[],"usage":{"prompt_tokens":10,'
                b'"completion_tokens":5,"completion_tokens_details":'
                b'{"reasoning_tokens":2}}}\n'
            ),
            b'data: [DONE]\n',
        ]
        clock = [0.0, 1.0, 3.0, 4.0, 5.0]

        with patch.object(
            common.urllib.request,
            "urlopen",
            return_value=_StreamResponse(events),
        ), patch.object(common.time, "perf_counter", side_effect=clock):
            result = common.stream_completion(
                "http://127.0.0.1:1/v1",
                {"stream": True},
                1.0,
            )

        self.assertEqual(result["content"], "AB")
        self.assertEqual(result["ttft_seconds"], 1.0)
        self.assertEqual(result["time_to_first_visible_content_seconds"], 3.0)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 4 / 3)
        self.assertEqual(result["visible_content_tokens_per_second"], 2.0)
        self.assertEqual(result["end_to_end_tokens_per_second"], 1.0)


class FunctionalBenchmarkTests(unittest.TestCase):
    def test_code_validator_rejects_malicious_output_without_executing_it(self):
        """An extra payload after square must never be executed by validation."""
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            content = (
                "def square(n):\n"
                "    return n * n\n"
                f"open({str(marker)!r}, 'w').write('executed')\n"
            )
            response = {"choices": [{"message": {"content": content}}]}

            self.assertFalse(validate("code", response))
            self.assertFalse(marker.exists())

    def test_code_validator_accepts_only_the_exact_square_function(self):
        """Calls, attributes, decorators, and extra definitions violate the fixture."""
        accepted = {"choices": [{"message": {"content": "def square(n):\n    return n * n\n"}}]}
        self.assertTrue(validate("code", accepted))

        rejected = (
            "import os\ndef square(n):\n    return n * n\n",
            "def square(n):\n    return pow(n, 2)\n",
            "def square(n):\n    return n.__mul__(n)\n",
            "@decorator\ndef square(n):\n    return n * n\n",
            "def square(n):\n    return n * n\ndef other():\n    return 1\n",
        )
        for content in rejected:
            with self.subTest(content=content):
                response = {"choices": [{"message": {"content": content}}]}
                try:
                    verdict = validate("code", response)
                except Exception as exc:
                    self.fail(f"invalid model code raised instead of returning false: {exc}")
                self.assertFalse(verdict)


class StabilityBenchmarkTests(unittest.TestCase):
    def test_stability_waits_through_the_bounded_remaining_duration(self):
        """A partial final interval must not make the stability run exit early."""
        now = [0.0]
        sleeps: list[float] = []
        probes: list[float] = []

        def clock() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        def probe() -> dict[str, object]:
            probes.append(now[0])
            return {"passed": True}

        report: dict[str, object] = {}
        with patch.object(stability.time, "monotonic", side_effect=clock), patch.object(
            stability.time, "sleep", side_effect=sleep
        ), patch.object(stability, "run_probe", side_effect=lambda *_args: probe()), patch.object(
            stability,
            "write_report",
            side_effect=lambda _path, value: report.update(value),
        ):
            exit_code = stability.main(
                [
                    "--base-url",
                    "http://127.0.0.1:1",
                    "--model",
                    "fixture-model",
                    "--output",
                    "unused.json",
                    "--enable-stability",
                    "--duration-seconds",
                    "10",
                    "--interval-seconds",
                    "6",
                ]
            )

        self.assertEqual(sleeps, [6.0, 4.0])
        self.assertEqual(probes, [0.0, 6.0, 10.0])
        self.assertEqual(report["probes"], [{"passed": True}] * 3)
        self.assertGreaterEqual(report["elapsed_seconds"], 10.0)
        self.assertEqual(exit_code, 0)


class LongContextBenchmarkTests(unittest.TestCase):
    def test_tokenization_uses_server_root_when_api_base_ends_in_v1(self):
        urls: list[str] = []

        def tokenize(url, _payload, _timeout):
            urls.append(url)
            return {"count": 240_000}

        with patch.object(long_context, "request_json", side_effect=tokenize):
            _messages, context = long_context.calibrate(
                "http://127.0.0.1:18300/v1", "fixture-model", 240_000, 1.0
            )

        self.assertEqual(urls, ["http://127.0.0.1:18300/tokenize"])
        self.assertEqual(context["tokenize_count"], 240_000)


if __name__ == "__main__":
    unittest.main()
