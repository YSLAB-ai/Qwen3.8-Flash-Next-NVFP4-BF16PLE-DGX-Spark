"""Guarded, deterministic Docker invocation for the qualified serving profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Sequence

from .download import local_target_path
from .manifest import Target


DEFAULT_IMAGE = "qwen38-flash-dgx"
RECIPE_LABEL = "qwen38-flash-next-bf16-ple"
MIN_MEM_AVAILABLE_BYTES = 100 * 1024**3
_SPLITTING_OPS = (
    '["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output",'
    '"vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv",'
    '"vllm::qwen3_8_flash_next_ple_short_conv",'
    '"vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention",'
    '"vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed",'
    '"vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
)


class RuntimeConfigurationError(ValueError):
    """Raised when a request falls outside the qualified runtime profile."""


@dataclass(frozen=True)
class RuntimeOptions:
    """The only runtime profile qualified for this recipe by default."""

    context: int = 262_144
    sequences: int = 8
    gpu_memory: float = 0.80
    mtp: int = 0
    prewarm: bool = False
    port: int = 18_300
    bind: str = "127.0.0.1"


@dataclass(frozen=True)
class EnvironmentReport:
    """Capacity observed before starting the serving container."""

    mem_available_bytes: int
    disk_free_bytes: int
    warnings: tuple[str, ...]


def validate_environment(
    options: RuntimeOptions,
    unsafe_override: bool,
    *,
    minimum_free_bytes: int = 0,
    disk_path: Path | None = None,
) -> EnvironmentReport:
    """Fail closed when host capacity or options are outside the safe profile."""
    mem_available = _mem_available_bytes()
    disk_free = _disk_free_bytes(disk_path or Path.cwd())
    warnings = _profile_warnings(options)
    if mem_available < MIN_MEM_AVAILABLE_BYTES:
        warnings.append(
            f"MemAvailable {mem_available} is below required {MIN_MEM_AVAILABLE_BYTES} bytes"
        )
    if disk_free < minimum_free_bytes:
        warnings.append(
            f"free disk {disk_free} is below required {minimum_free_bytes} bytes"
        )
    if warnings and not unsafe_override:
        raise RuntimeConfigurationError("; ".join(warnings))
    return EnvironmentReport(mem_available, disk_free, tuple(warnings))


def build_docker_command(
    target: Target,
    model_path: Path,
    cache_root: Path,
    options: RuntimeOptions,
    *,
    unsafe_override: bool = False,
) -> list[str]:
    """Return a Docker argv list for a pinned local model, without executing it."""
    warnings = _profile_warnings(options)
    if warnings and not unsafe_override:
        raise RuntimeConfigurationError("; ".join(warnings))
    cache = _require_local_path(cache_root, "cache root")
    model = _require_local_path(model_path, "model path")
    expected = local_target_path(target, cache).resolve(strict=False)
    if model != expected:
        raise RuntimeConfigurationError("model path is not the exact pinned local snapshot")

    model_in_container, extra_mounts = _model_mounts(model, cache)
    container_name = container_name_for(target)
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--restart",
        "no",
        "--label",
        f"ai.yslab.recipe={RECIPE_LABEL}",
        "--label",
        f"ai.yslab.target={target.name}",
        "--gpus",
        "all",
        "--ipc=host",
        "--shm-size",
        "16g",
        "-p",
        f"{options.bind}:{options.port}:8000",
        "-v",
        f"{cache}:/hf:ro",
        *extra_mounts,
        "-e",
        "HF_HOME=/hf",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "VLLM_PLE_MMAP=1",
        "-e",
        "VLLM_PLE_MMAP_WORKERS=32",
        "-e",
        "VLLM_PLE_MMAP_PREWARM=0",
        "-e",
        "VLLM_USE_FLASHINFER_SAMPLER=1",
        DEFAULT_IMAGE,
        str(model_in_container),
        "--served-model-name",
        served_model_alias(target),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--load-format",
        "safetensors",
        "--max-model-len",
        str(options.context),
        "--max-num-seqs",
        str(options.sequences),
        "--gpu-memory-utilization",
        str(options.gpu_memory),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        "8192",
        "-cc.cudagraph_mode=PIECEWISE",
        f"-cc.splitting_ops={_SPLITTING_OPS}",
        "--no-enable-flashinfer-autotune",
        "--kv-cache-dtype",
        "bfloat16",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
    ]


def container_name_for(target: Target) -> str:
    """Return the target-specific container name used for safe replacement."""
    return f"qwen38-flash-{target.name}"


def served_model_alias(target: Target) -> str:
    """Return a model alias that identifies the manifest target unambiguously."""
    return f"{target.served_model_name}-{target.name}"


def start_container(
    command: Sequence[str],
    target: Target,
    *,
    replace: bool,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Start the command, replacing only an existing recipe-labelled container."""
    name = container_name_for(target)
    inspected = runner(
        ["docker", "container", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(inspected, "returncode", 1) == 0:
        if not replace:
            raise RuntimeError(f"container already exists: {name}; pass --replace to replace it")
        label = runner(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                '{{ index .Config.Labels "ai.yslab.recipe" }}',
                name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if getattr(label, "stdout", "").strip() != RECIPE_LABEL:
            raise RuntimeError(f"refusing to replace container without recipe label: {name}")
        runner(["docker", "rm", "-f", name], check=True)
    runner(list(command), check=True)


def _profile_warnings(options: RuntimeOptions) -> list[str]:
    warnings: list[str] = []
    if options.context != 262_144 or options.sequences != 8 or options.gpu_memory != 0.80:
        warnings.append("runtime options are outside the validated profile")
    if options.mtp != 0:
        warnings.append("MTP must be 0 in the validated profile")
    if options.prewarm:
        warnings.append("PLE prewarm must be disabled in the validated profile")
    if options.bind != "127.0.0.1":
        warnings.append("bind must use loopback 127.0.0.1")
    if not isinstance(options.port, int) or isinstance(options.port, bool) or not 1 <= options.port <= 65_535:
        warnings.append("port must be an integer in the range 1..65535")
    return warnings


def _require_local_path(value: Path, label: str) -> Path:
    if not isinstance(value, Path):
        raise RuntimeConfigurationError(f"{label} must be a local pathlib path")
    raw = str(value)
    scheme, separator, _ = raw.partition(":")
    if separator and scheme and scheme[0].isalpha() and all(
        character.isalnum() or character in "+.-" for character in scheme
    ):
        raise RuntimeConfigurationError(f"{label} must be local, not remote")
    if not value.is_absolute():
        value = value.resolve(strict=False)
    text = str(value)
    if "://" in text:
        raise RuntimeConfigurationError(f"{label} must be local, not remote")
    return value.resolve(strict=False)


def _model_mounts(model: Path, cache: Path) -> tuple[Path, tuple[str, ...]]:
    try:
        return Path("/hf") / model.relative_to(cache), ()
    except ValueError:
        recipe_root = cache.parent.resolve(strict=False)
        try:
            relative = model.relative_to(recipe_root)
        except ValueError as exc:
            raise RuntimeConfigurationError("model path is outside the approved local cache") from exc
        return Path("/recipe") / relative, ("-v", f"{recipe_root}:/recipe:ro")


def _mem_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value, unit = line.split()
            if key == "MemAvailable:" and unit == "kB":
                return int(value) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _disk_free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0
