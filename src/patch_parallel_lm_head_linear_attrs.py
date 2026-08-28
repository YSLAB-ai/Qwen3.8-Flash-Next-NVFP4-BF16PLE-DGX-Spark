"""Expose the Linear-like metadata required by quantized LM-head kernels."""

from __future__ import annotations

import argparse
from pathlib import Path


NEEDLE = '''\
        self.quant_config = quant_config
        if bias:
'''
MARKER = "qwen38-flash-dgx: quantized LM-head linear metadata"
REPLACEMENT = f'''\
        self.quant_config = quant_config
        # {MARKER}
        self.output_partition_sizes = [self.num_embeddings_per_partition]
        self.has_bias = bias
        if bias:
'''


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    occurrences = source.count(NEEDLE)
    if occurrences != 1:
        raise RuntimeError(
            "expected exactly one ParallelLMHead metadata hook, "
            f"found {occurrences}"
        )
    return source.replace(NEEDLE, REPLACEMENT, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=Path)
    args = parser.parse_args(argv)
    source = args.source_path.read_text(encoding="utf-8")
    patched = patch_source(source)
    if patched != source:
        args.source_path.write_text(patched, encoding="utf-8")
        print(f"patched quantized ParallelLMHead metadata: {args.source_path}")
    else:
        print(f"quantized ParallelLMHead metadata already present: {args.source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
