from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


RUNTIME_CONFIG_KEYS = {
    "backend",
    "gpu_workers",
    "max_gpu_workers",
    "candidate_levels",
    "cpu_workers",
    "calibration_epochs",
    "max_gpu_memory_fraction",
    "minimum_throughput_gain",
    "retries",
    "gpu_worker_cpu_threads",
    "baseline_worker_cpu_threads",
    "calibration",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class SplitSpec:
    protocol: str
    fold: str
    seed: int


@dataclass(frozen=True)
class TuningTask:
    model: str
    parameters: tuple[tuple[str, float | int], ...]
    seed: int
    calibration_slot: int | None = None

    @classmethod
    def create(
        cls,
        model: str,
        parameters: Mapping[str, float | int],
        seed: int,
        calibration_slot: int | None = None,
    ) -> "TuningTask":
        return cls(model, tuple(sorted(parameters.items())), seed, calibration_slot)

    @property
    def params(self) -> dict[str, float | int]:
        return dict(self.parameters)


@dataclass(frozen=True)
class NeuralTrainingTask:
    model: str
    parameters: tuple[tuple[str, float | int], ...]
    split: SplitSpec

    @classmethod
    def create(
        cls, model: str, parameters: Mapping[str, float | int], split: SplitSpec
    ) -> "NeuralTrainingTask":
        return cls(model, tuple(sorted(parameters.items())), split)

    @property
    def params(self) -> dict[str, float | int]:
        return dict(self.parameters)


@dataclass(frozen=True)
class BaselineTrainingTask:
    split: SplitSpec
    models: tuple[str, ...] = (
        "2010-carry-forward",
        "city-mean",
        "inverse-distance",
        "ridge",
        "non-spatial-mlp",
    )


def task_payload(task: TuningTask | NeuralTrainingTask | BaselineTrainingTask) -> dict[str, Any]:
    return json.loads(_canonical({"type": type(task).__name__, **asdict(task)}))


def task_id(task: TuningTask | NeuralTrainingTask | BaselineTrainingTask) -> str:
    kind = {
        TuningTask: "tune",
        NeuralTrainingTask: "neural",
        BaselineTrainingTask: "baseline",
    }[type(task)]
    return f"{kind}-{_digest(task_payload(task))[:20]}"


def semantic_fingerprint(values: Mapping[str, Any]) -> str:
    payload = dict(values)
    payload.pop("parallel", None)
    paths = dict(payload.get("paths", {}))
    paths.pop("artifacts", None)
    payload["paths"] = paths
    payload.pop("device", None)
    return _digest(payload)


def execution_fingerprint(values: Mapping[str, Any], device: str) -> str:
    parallel = dict(values.get("parallel", {}))
    return _digest({"parallel": parallel, "device": device})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task_manifest(
    task_root: Path,
    task: TuningTask | NeuralTrainingTask | BaselineTrainingTask,
    fingerprint: str,
) -> dict[str, Any] | None:
    directory = task_root / task_id(task)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("task_id") != task_id(task)
            or manifest.get("semantic_fingerprint") != fingerprint
            or manifest.get("task") != task_payload(task)
        ):
            return None
        for metadata in manifest.get("files", {}).values():
            path = directory / metadata["path"]
            if (
                not path.is_file()
                or path.stat().st_size != int(metadata["size"])
                or sha256_file(path) != metadata["sha256"]
            ):
                return None
        return manifest
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def commit_task_artifacts(
    task_root: Path,
    task: TuningTask | NeuralTrainingTask | BaselineTrainingTask,
    *,
    semantic: str,
    execution: str,
    started_at: float,
    metrics: Mapping[str, Any],
    files: Mapping[str, Path],
    row_count: int,
    peak_gpu_memory: int,
) -> dict[str, Any]:
    task_root.mkdir(parents=True, exist_ok=True)
    identifier = task_id(task)
    final = task_root / identifier
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}.tmp-", dir=task_root))
    try:
        file_metadata: dict[str, dict[str, Any]] = {}
        for name, source in files.items():
            destination = temporary / source.name
            os.replace(source, destination)
            file_metadata[name] = {
                "path": destination.name,
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        manifest = {
            "status": "complete",
            "task_id": identifier,
            "task": task_payload(task),
            "semantic_fingerprint": semantic,
            "execution_fingerprint": execution,
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": time.time(),
            "duration_seconds": time.time() - started_at,
            "peak_gpu_memory": int(peak_gpu_memory),
            "row_count": int(row_count),
            "metrics": dict(metrics),
            "files": file_metadata,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(_canonical(manifest) + "\n", encoding="utf-8")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if final.exists():
            invalid = task_root / f".{identifier}.invalid-{uuid.uuid4().hex}"
            os.replace(final, invalid)
        os.replace(temporary, final)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass(frozen=True)
class MPSService:
    pipe_directory: Path
    log_directory: Path
    previous_environment: dict[str, str | None]


@contextlib.contextmanager
def private_mps_service(run_root: Path, enabled: bool) -> Iterator[MPSService | None]:
    if not enabled:
        yield None
        return
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        raise RuntimeError("nvidia-cuda-mps-control is required for parallel.backend=nvidia-mps")
    # NVIDIA's MPS sockets are constrained by the Unix-domain socket path limit.
    # Keep the private service path short even when the artifact namespace is descriptive.
    service_root = Path(tempfile.mkdtemp(prefix="ecuador-mps-"))
    pipe_directory = service_root / "pipe"
    log_directory = service_root / "log"
    pipe_directory.mkdir()
    log_directory.mkdir()
    names = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_directory)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(log_directory)
    try:
        started = subprocess.run([executable, "-d"], check=False, capture_output=True, text=True)
        if started.returncode:
            logs = []
            for path in sorted(log_directory.glob("*.log")):
                logs.append(path.read_text(encoding="utf-8", errors="replace"))
            detail = started.stderr.strip() or started.stdout.strip() or "".join(logs).strip()
            raise RuntimeError(f"NVIDIA MPS controller failed to start: {detail}")
        yield MPSService(pipe_directory, log_directory, previous)
    finally:
        subprocess.run(
            [executable], input="quit\n", check=False, capture_output=True, text=True
        )
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(service_root, ignore_errors=True)


@contextlib.contextmanager
def active_thread_percentage(worker_count: int) -> Iterator[None]:
    name = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
    previous = os.environ.get(name)
    os.environ[name] = str(max(1, 100 // max(worker_count, 1)))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def configure_worker_threads(count: int) -> None:
    value = str(max(int(count), 1))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    try:
        import torch

        torch.set_num_threads(int(value))
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def select_calibrated_worker_count(
    reports: list[Mapping[str, Any]],
    *,
    maximum_memory_fraction: float,
    minimum_throughput_gain: float,
) -> int:
    selected = 1
    previous_throughput: float | None = None
    for report in sorted(reports, key=lambda item: int(item["workers"])):
        if report.get("error"):
            break
        throughput = float(report["throughput"])
        memory_fraction = float(report["gpu_memory_fraction"])
        gain = None if previous_throughput is None else throughput / previous_throughput - 1
        if memory_fraction > maximum_memory_fraction or (
            gain is not None and gain < minimum_throughput_gain
        ):
            break
        selected = int(report["workers"])
        previous_throughput = throughput
    return selected


def mps_client_pids() -> set[int]:
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        return set()
    servers = subprocess.run(
        [executable], input="get_server_list\n", check=False, capture_output=True, text=True
    ).stdout.split()
    clients: set[int] = set()
    for server in servers:
        if not server.isdigit():
            continue
        output = subprocess.run(
            [executable],
            input=f"get_client_list {server}\n",
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        clients.update(int(value) for value in output.split() if value.isdigit())
    return clients


def is_recoverable_worker_error(error: BaseException) -> bool:
    from concurrent.futures.process import BrokenProcessPool

    message = str(error).lower()
    return isinstance(error, BrokenProcessPool) or (
        "cuda" in message and ("out of memory" in message or "worker" in message)
    )
