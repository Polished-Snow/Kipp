#!/usr/bin/env python3
"""Shared provenance for the bench harnesses.

Every result JSON must self-describe the engine (commit, dirty flag, binary,
compiler), the hardware it ran on, and the model artifact it measured, so a
number quoted anywhere traces back to one committed file. The dirty flag
ignores ``bench/results/`` so a tree whose only changes are freshly written
result files still records ``dirty: false``; any other modification marks
the result dirty as before.
"""
from __future__ import annotations

import json
import pathlib
import platform
import re
import shutil
import subprocess

BUILD_FLAGS = "-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror"


def checked_command_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"required tool {command[0]!r} was not found") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"{command[0]} exited {error.returncode}{suffix}"
        ) from error
    return completed.stdout.strip()


def command_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError:
        return f"unknown ({command[0]} unavailable)"
    if completed.returncode != 0:
        return f"unknown ({command[0]} exited {completed.returncode})"
    return completed.stdout.strip()


def git_metadata(root: pathlib.Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    # Freshly written result files must not mark their own run dirty.
    # Every other working-tree change still counts.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".",
         ":(exclude)bench/results"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        revision.stdout.strip() if revision.returncode == 0 else "uncommitted",
        bool(status.stdout.strip()) if status.returncode == 0 else True,
    )


def macos_hardware_metadata() -> dict[str, object]:
    memory_text = command_text(["sysctl", "-n", "hw.memsize"])
    gpu_cores: int | str = "unknown"
    profiler = command_text(["system_profiler", "SPDisplaysDataType", "-json"])
    try:
        displays = json.loads(profiler)["SPDisplaysDataType"]
        gpu_cores = int(displays[0]["sppci_cores"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {
        "platform": platform.platform(),
        "processor": command_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "gpu_cores": gpu_cores,
        "unified_memory_bytes": (
            int(memory_text) if memory_text.isdigit() else memory_text
        ),
    }


def linux_value(path: pathlib.Path, key: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and name.strip() == key:
            return value.strip()
    return None


def linux_hardware_metadata() -> dict[str, object]:
    processor = linux_value(pathlib.Path("/proc/cpuinfo"), "model name")
    memory_text = linux_value(pathlib.Path("/proc/meminfo"), "MemTotal")
    memory_match = (
        re.fullmatch(r"(\d+)\s+kB", memory_text) if memory_text else None
    )
    return {
        "platform": platform.platform(),
        "processor": processor or platform.processor() or "unknown",
        "memory_bytes": (
            int(memory_match.group(1)) * 1024
            if memory_match is not None
            else "unknown"
        ),
    }


def cuda_hardware_metadata(nvidia_smi: str) -> dict[str, object]:
    gpu_output = checked_command_text(
        [
            nvidia_smi,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpu_names: list[str] = []
    driver_versions: list[str] = []
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2:
            raise RuntimeError(f"could not parse nvidia-smi GPU metadata: {line!r}")
        gpu_names.append(fields[0])
        driver_versions.append(fields[1])
    if not gpu_names:
        raise RuntimeError("nvidia-smi reported no NVIDIA GPUs")

    nvcc_output = command_text(["nvcc", "--version"])
    nvcc_lines = nvcc_output.splitlines()
    return {
        "gpu_name": "; ".join(gpu_names),
        "gpu_driver": "; ".join(dict.fromkeys(driver_versions)),
        "nvcc_version": nvcc_lines[-1] if nvcc_lines else "unknown",
    }


def hardware_metadata(
    backend: str = "metal", nvidia_smi: str | None = None
) -> dict[str, object]:
    system = platform.system()
    if system == "Darwin":
        metadata = macos_hardware_metadata()
    elif system == "Linux":
        metadata = linux_hardware_metadata()
    else:
        raise RuntimeError(f"unsupported benchmark platform {system!r}")
    if backend == "cuda":
        if nvidia_smi is None:
            nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise RuntimeError("CUDA benchmarking requires nvidia-smi in PATH")
        metadata.update(cuda_hardware_metadata(nvidia_smi))
    return metadata


def engine_metadata(
    binary: str | pathlib.Path, backend: str, root: str | pathlib.Path
) -> dict[str, object]:
    root_path = pathlib.Path(root).resolve()
    binary_path = pathlib.Path(binary)
    if not binary_path.is_absolute():
        binary_path = root_path / binary_path
    binary_path = binary_path.resolve()
    try:
        binary_text = str(binary_path.relative_to(root_path))
    except ValueError:
        binary_text = str(binary_path)
    commit, dirty = git_metadata(root_path)
    if backend == "cuda":
        compiler_output = command_text(["nvcc", "--version"])
    else:
        compiler_output = command_text(["clang", "--version"])
        if compiler_output.startswith("unknown"):
            compiler_output = command_text(["cc", "--version"])
    compiler_lines = compiler_output.splitlines() or [
        "unknown (no compiler version output)"
    ]
    compiler = compiler_lines[-1] if backend == "cuda" else compiler_lines[0]
    return {
        "backend": backend,
        "binary": binary_text,
        "commit": commit,
        "dirty": dirty,
        "compiler": compiler,
        "build_flags": BUILD_FLAGS,
    }


def model_metadata(gguf_path: str | pathlib.Path) -> dict[str, object]:
    """Self-description from the artifact's sibling ``*.gguf.manifest.json``.

    The manifest records the pinned repository/revision, so scripts never
    hardcode a checkpoint; without a manifest only the path and size are
    reported, marked explicitly.
    """
    path = pathlib.Path(gguf_path)
    metadata: dict[str, object] = {
        "artifact_bytes": path.stat().st_size if path.is_file() else "missing",
    }
    manifest_path = path.with_name(path.name + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata["path"] = str(path)
        metadata["manifest"] = "missing"
        return metadata
    for key in ("checkpoint", "repository", "revision", "format", "sha256"):
        if key in manifest:
            metadata[key] = manifest[key]
    checkpoint = manifest.get("checkpoint")
    stem = path.name[: -len(".gguf")] if path.name.endswith(".gguf") else path.name
    prefix = f"kipp-{checkpoint}-" if checkpoint else None
    if prefix and stem.startswith(prefix):
        metadata["scheme"] = stem[len(prefix):]
    return metadata


def provenance(
    binary: str | pathlib.Path,
    backend: str,
    model_path: str | pathlib.Path,
    root: str | pathlib.Path,
    nvidia_smi: str | None = None,
) -> dict[str, object]:
    return {
        "engine": engine_metadata(binary, backend, root),
        "hardware": hardware_metadata(backend, nvidia_smi),
        "model": model_metadata(model_path),
    }
