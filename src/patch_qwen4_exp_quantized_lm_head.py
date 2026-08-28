"""Give Qwen3.8 Flash-Next output heads their configured quantization method."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "qwen38-flash-dgx: support a quantized lm_head checkpoint"
CONSTRUCTOR_HOOK = re.compile(
    r'^(?P<indent>[ \t]+)config\.hidden_size,\n'
    r'(?P=indent)prefix=maybe_prefix\(prefix, "lm_head"\),$',
    re.MULTILINE,
)


def patch_source(source: str) -> str:
    if MARKER in source:
        return source

    def replacement(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            f"{indent}config.hidden_size,\n"
            f"{indent}# {MARKER}\n"
            f"{indent}quant_config=self.quant_config,\n"
            f'{indent}prefix=maybe_prefix(prefix, "lm_head"),'
        )

    patched, occurrences = CONSTRUCTOR_HOOK.subn(replacement, source)
    if occurrences != 1:
        raise RuntimeError(
            "expected exactly one Qwen output-head constructor hook, "
            f"found {occurrences}"
        )
    return patched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=Path, nargs="+")
    args = parser.parse_args(argv)
    for path in args.model_paths:
        source = path.read_text(encoding="utf-8")
        patched = patch_source(source)
        if patched != source:
            path.write_text(patched, encoding="utf-8")
            print(f"patched quantized Qwen output head: {path}")
        else:
            print(f"quantized Qwen output-head support already present: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
