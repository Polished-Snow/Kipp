#!/usr/bin/env python3
"""Run repeatable Kipp GPU benchmarks as isolated native subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import resource
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass

METRIC_PATTERN = re.compile(
    r"KIPP_METRIC backend=(\w+) prefill_tokens=(\d+) "
    r"prefill_seconds=([0-9.eE+-]+) decode_tokens=(\d+) "
    r"decode_seconds=([0-9.eE+-]+)"
)
DEFAULT_BINARIES = {
    "metal": pathlib.Path("build/kipp-metal"),
    "cuda": pathlib.Path("build/kipp-cuda-generic"),
}
POLL_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class Measurement:
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    prefill_seconds: float
    decode_seconds: float
    peak_resident_bytes: int
    peak_device_bytes: int | None = None

    def as_dict(self, backend: str) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "peak_resident_bytes": self.peak_resident_bytes,
        }
        if backend == "cuda":
            if self.peak_device_bytes is None:
                raise RuntimeError("CUDA measurement is missing peak device memory")
            result["peak_device_bytes"] = self.peak_device_bytes
        return result


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


def cuda_process_bytes(nvidia_smi: str, pid: int) -> int | None:
    output = checked_command_text(
        [
            nvidia_smi,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    used_bytes = 0
    found = False
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2 or fields[0] != str(pid):
            continue
        try:
            used_mib = int(fields[1])
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi returned invalid memory for benchmark PID {pid}: "
                f"{fields[1]!r}"
            ) from error
        used_bytes += used_mib * 1024 * 1024
        found = True
    return used_bytes if found else None


def peak_rss_bytes(usage: resource.struct_rusage) -> int:
    if platform.system() == "Darwin":
        return int(usage.ru_maxrss)
    if platform.system() == "Linux":
        return int(usage.ru_maxrss) * 1024
    raise RuntimeError("peak RSS collection supports only macOS and Linux")


def run_process(
    command: list[str],
    backend: str,
    nvidia_smi: str | None,
) -> tuple[str, str, int, int | None]:
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        try:
            process = subprocess.Popen(
                command, stdout=stdout_file, stderr=stderr_file
            )
        except OSError as error:
            raise RuntimeError(
                f"could not start benchmark binary {command[0]!r}: {error}"
            ) from error

        peak_device_bytes: int | None = None
        status = 0
        usage: resource.struct_rusage
        try:
            while True:
                waited_pid, status, usage = os.wait4(
                    process.pid, os.WNOHANG
                )
                if waited_pid == process.pid:
                    break

                if backend == "cuda":
                    if nvidia_smi is None:
                        raise RuntimeError(
                            "CUDA benchmarking requires nvidia-smi in PATH"
                        )
                    used_bytes = cuda_process_bytes(nvidia_smi, process.pid)
                    if used_bytes is not None:
                        peak_device_bytes = max(
                            peak_device_bytes or 0, used_bytes
                        )

                time.sleep(POLL_INTERVAL_SECONDS)
        except BaseException:
            process.kill()
            _, status, _ = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            raise

        process.returncode = os.waitstatus_to_exitcode(status)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")

    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark subprocess exited {process.returncode}\n{stdout}\n{stderr}"
        )
    if backend == "cuda" and peak_device_bytes is None:
        raise RuntimeError(
            f"nvidia-smi did not report GPU memory for benchmark PID {process.pid}"
        )
    return stdout, stderr, peak_rss_bytes(usage), peak_device_bytes


def run_once(
    binary: pathlib.Path,
    model: pathlib.Path,
    prompt: str,
    decode: int,
    backend: str = "metal",
    nvidia_smi: str | None = None,
) -> Measurement:
    requested_backend = backend
    command = [
        str(binary),
        "--backend",
        requested_backend,
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--decode",
        str(decode),
        "--top",
        "1",
    ]
    if requested_backend == "cuda" and nvidia_smi is None:
        nvidia_smi = shutil.which("nvidia-smi")
    _, stderr, peak_resident_bytes, peak_device_bytes = run_process(
        command, requested_backend, nvidia_smi
    )
    metric = METRIC_PATTERN.search(stderr)
    if metric is None:
        raise RuntimeError(
            "benchmark output is missing KIPP_METRIC on stderr\n"
            f"{stderr}"
        )
    (
        metric_backend,
        prefill_tokens,
        prefill_seconds,
        decode_tokens,
        decode_seconds,
    ) = metric.groups()
    if metric_backend != requested_backend:
        raise RuntimeError(
            f"benchmark used unexpected backend {metric_backend}; "
            f"expected {requested_backend}"
        )
    prefill_count = int(prefill_tokens)
    decode_count = int(decode_tokens)
    prefill_duration = float(prefill_seconds)
    decode_duration = float(decode_seconds)
    if prefill_duration <= 0 or decode_count != decode or decode_duration <= 0:
        raise RuntimeError(f"invalid benchmark durations\n{stderr}")
    return Measurement(
        prefill_tokens_per_second=prefill_count / prefill_duration,
        decode_tokens_per_second=decode_count / decode_duration,
        prefill_seconds=prefill_duration,
        decode_seconds=decode_duration,
        peak_resident_bytes=peak_resident_bytes,
        peak_device_bytes=peak_device_bytes,
    )


def summarize(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "median": median,
        "median_absolute_deviation": statistics.median(deviations),
        "minimum": min(values),
        "maximum": max(values),
    }


def git_metadata(root: pathlib.Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        revision.stdout.strip() if revision.returncode == 0 else "uncommitted",
        bool(status.stdout.strip()) if status.returncode == 0 else True,
    )


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
            raise RuntimeError("CUDA benchmarking requires nvidia-smi in PATH")
        metadata.update(cuda_hardware_metadata(nvidia_smi))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=sorted(DEFAULT_BINARIES), default="metal"
    )
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path(
            "models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf"
        ),
    )
    parser.add_argument("--prompt", default="Hi 世界")
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.decode <= 0 or args.warmup < 0 or args.runs <= 0:
        parser.error("decode and runs must be positive; warmup cannot be negative")

    root = pathlib.Path(__file__).resolve().parents[1]
    binary_argument = args.binary or DEFAULT_BINARIES[args.backend]
    binary = (
        (root / binary_argument).resolve()
        if not binary_argument.is_absolute()
        else binary_argument
    )
    model = (
        (root / args.model).resolve()
        if not args.model.is_absolute()
        else args.model
    )
    if not binary.is_file():
        parser.error(f"benchmark binary does not exist: {binary}")
    if not model.is_file():
        parser.error(f"converted GGUF model does not exist: {model}")
    nvidia_smi = shutil.which("nvidia-smi") if args.backend == "cuda" else None
    if args.backend == "cuda" and nvidia_smi is None:
        parser.error("CUDA benchmarking requires nvidia-smi in PATH")

    for index in range(args.warmup):
        print(f"warm-up {index + 1}/{args.warmup}", flush=True)
        run_once(
            binary,
            model,
            args.prompt,
            args.decode,
            args.backend,
            nvidia_smi,
        )
    measurements = []
    for index in range(args.runs):
        print(f"measured run {index + 1}/{args.runs}", flush=True)
        measurements.append(
            run_once(
                binary,
                model,
                args.prompt,
                args.decode,
                args.backend,
                nvidia_smi,
            )
        )

    revision, dirty = git_metadata(root)
    compiler_command = ["clang", "--version"]
    if args.backend == "cuda":
        compiler_command = ["nvcc", "--version"]
    compiler_lines = command_text(compiler_command).splitlines() or [
        "unknown (no compiler version output)"
    ]
    compiler = compiler_lines[-1] if args.backend == "cuda" else compiler_lines[0]
    report = {
        "engine": {
            "commit": revision,
            "dirty": dirty,
            "backend": args.backend,
            "binary": str(binary),
            "compiler": compiler,
            "build_flags": "-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror",
        },
        "model": {
            "repository": "Qwen/Qwen3-4B-Base",
            "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            "format": "BF16 Kipp GGUF-v3",
            "artifact_bytes": model.stat().st_size,
        },
        "hardware": hardware_metadata(args.backend, nvidia_smi),
        "configuration": {
            "prompt": args.prompt,
            "decode_tokens": args.decode,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "batch_size": 1,
            "concurrency": 1,
            "sampling": "greedy",
        },
        "prefill_tokens_per_second": summarize(
            [item.prefill_tokens_per_second for item in measurements]
        ),
        "decode_tokens_per_second": summarize(
            [item.decode_tokens_per_second for item in measurements]
        ),
        "peak_resident_bytes": summarize(
            [float(item.peak_resident_bytes) for item in measurements]
        ),
        "runs": [item.as_dict(args.backend) for item in measurements],
    }
    if args.backend == "cuda":
        report["peak_device_bytes"] = summarize(
            [
                float(item.peak_device_bytes)
                for item in measurements
                if item.peak_device_bytes is not None
            ]
        )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = (
            (root / args.output).resolve()
            if not args.output.is_absolute()
            else args.output
        )
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
