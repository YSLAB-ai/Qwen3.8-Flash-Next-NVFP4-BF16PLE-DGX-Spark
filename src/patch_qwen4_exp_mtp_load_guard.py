"""Make Qwen3.8 Flash-Next native MTP checkpoint loading fail closed."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "qwen38-flash-dgx: require exact native BF16 MTP checkpoint tensors"
PATCH_MTP_TENSOR_NAMES = (
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)

_CLASS_ANCHOR = "class Qwen3_8FlashNextMTP(nn.Module, SupportsPP, Qwen3_8FlashNextMixtureOfExperts):"
_METHOD_ANCHOR = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
'''
_RETURN_ANCHOR = "        return loader.load_weights(remap_weight_names())"


def validate_observed(observed: dict[str, str]) -> None:
    """Exercise the exact runtime guard without importing torch in build tooling."""
    expected = set(PATCH_MTP_TENSOR_NAMES)
    found = set(observed)
    missing = expected - found
    unexpected = found - expected
    wrong_dtype = {name for name, dtype in observed.items() if dtype != "torch.bfloat16"}
    if missing:
        raise RuntimeError(f"missing native MTP checkpoint tensors: missing={len(missing)}")
    if unexpected:
        raise RuntimeError(
            f"unexpected native MTP checkpoint tensors: unexpected={len(unexpected)}"
        )
    if wrong_dtype:
        raise RuntimeError(f"native MTP checkpoint dtype mismatch: dtype={len(wrong_dtype)}")


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    for label, anchor in (
        ("Qwen3.8 MTP class", _CLASS_ANCHOR),
        ("Qwen3.8 MTP load method", _METHOD_ANCHOR),
        ("Qwen3.8 MTP loader return", _RETURN_ANCHOR),
    ):
        occurrences = source.count(anchor)
        if occurrences != 1:
            raise RuntimeError(f"expected exactly one {label} hook, found {occurrences}")

    names_literal = f"frozenset({PATCH_MTP_TENSOR_NAMES!r})"
    guard = f'''# {MARKER}
_QWEN38_NATIVE_MTP_CHECKPOINT_NAMES = {names_literal}


def _qwen38_validate_native_mtp_checkpoint(observed: dict[str, torch.dtype]) -> None:
    found = set(observed)
    missing = _QWEN38_NATIVE_MTP_CHECKPOINT_NAMES - found
    unexpected = found - _QWEN38_NATIVE_MTP_CHECKPOINT_NAMES
    wrong_dtype = {{name for name, dtype in observed.items() if dtype != torch.bfloat16}}
    if missing:
        raise RuntimeError(f"missing native MTP checkpoint tensors: missing={{len(missing)}}")
    if unexpected:
        raise RuntimeError(
            f"unexpected native MTP checkpoint tensors: unexpected={{len(unexpected)}}"
        )
    if wrong_dtype:
        raise RuntimeError(
            f"native MTP checkpoint dtype mismatch: dtype={{len(wrong_dtype)}}"
        )
    print(
        "qwen38-flash-dgx: validated native BF16 MTP checkpoint tensors: 31/31",
        flush=True,
    )


'''
    class_index = source.index(_CLASS_ANCHOR)
    decorator_index = source.rfind("@support_torch_compile(", 0, class_index)
    if decorator_index < 0 or (
        decorator_index > 0 and source[decorator_index - 1] != "\n"
    ):
        raise RuntimeError("expected exactly one decorated Qwen3.8 MTP class hook, found 0")
    insertion = decorator_index
    patched = source[:insertion] + guard + source[insertion:]
    patched = patched.replace(
        _METHOD_ANCHOR,
        '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        _qwen38_native_mtp_checkpoint: dict[str, torch.dtype] = {}

        def remap_weight_names():
            for name, weight in weights:
                if name.startswith("mtp."):
                    _qwen38_native_mtp_checkpoint[name] = weight.dtype
''',
        1,
    )
    patched = patched.replace(
        _RETURN_ANCHOR,
        '''        loaded = loader.load_weights(remap_weight_names())
        _qwen38_validate_native_mtp_checkpoint(_qwen38_native_mtp_checkpoint)
        return loaded''',
        1,
    )
    return patched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args(argv)
    source = args.model_path.read_text(encoding="utf-8")
    patched = patch_source(source)
    if patched != source:
        args.model_path.write_text(patched, encoding="utf-8")
        print(f"patched native MTP checkpoint load guard: {args.model_path}")
    else:
        print(f"native MTP checkpoint load guard already present: {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
