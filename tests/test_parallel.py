from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import torch

from ecuador_evolution.config import Settings
from ecuador_evolution.experiment import _run_gpu_tasks
from ecuador_evolution.parallel import (
    NeuralTrainingTask,
    SplitSpec,
    TuningTask,
    active_thread_percentage,
    commit_task_artifacts,
    mps_client_pids,
    private_mps_service,
    select_calibrated_worker_count,
    semantic_fingerprint,
    task_id,
    validate_task_manifest,
)


def _timed_fake_worker(delay: float) -> tuple[int, float, float]:
    started = time.monotonic()
    time.sleep(delay)
    return os.getpid(), started, time.monotonic()


def _cuda_fit_worker(seconds: float) -> int:
    torch.cuda.init()
    left = torch.randn(1536, 1536, device="cuda")
    right = torch.randn(1536, 1536, device="cuda")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        left = torch.mm(left, right)
        left = left / left.norm().clamp_min(1)
    torch.cuda.synchronize()
    return os.getpid()


def test_fake_workers_overlap_in_distinct_processes() -> None:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        results = list(pool.map(_timed_fake_worker, [0.5, 0.5]))
    assert len({result[0] for result in results}) == 2
    assert max(result[1] for result in results) < min(result[2] for result in results)
    execution_span = max(result[2] for result in results) - min(result[1] for result in results)
    assert execution_span < 0.75 * sum(result[2] - result[1] for result in results)


def test_task_ids_and_semantic_fingerprint_are_deterministic() -> None:
    first = TuningTask.create("spatial", {"width": 8, "dropout": 0.1}, 7)
    reordered = TuningTask.create("spatial", {"dropout": 0.1, "width": 8}, 7)
    assert task_id(first) == task_id(reordered)
    split = SplitSpec("random-mask", "20%", 7)
    assert task_id(NeuralTrainingTask.create("spatial", first.params, split)) != task_id(first)
    base = {"model": {"width": 8}, "parallel": {"gpu_workers": 1}, "paths": {"artifacts": "a"}}
    changed_runtime = {
        "model": {"width": 8},
        "parallel": {"gpu_workers": 8, "backend": "nvidia-mps"},
        "paths": {"artifacts": "b"},
    }
    assert semantic_fingerprint(base) == semantic_fingerprint(changed_runtime)
    assert semantic_fingerprint(base) != semantic_fingerprint({**base, "model": {"width": 16}})


def test_atomic_commit_resume_and_corruption_detection(tmp_path: Path) -> None:
    task = TuningTask.create("spatial", {"width": 8}, 7)
    task_root = tmp_path / "tasks"
    task_root.mkdir()
    source = tmp_path / "predictions.parquet"
    source.write_bytes(b"valid shard")
    manifest = commit_task_artifacts(
        task_root,
        task,
        semantic="semantic",
        execution="execution",
        started_at=time.time(),
        metrics={"validation_mae": 1.0},
        files={"predictions": source},
        row_count=1,
        peak_gpu_memory=0,
    )
    assert validate_task_manifest(task_root, task, "semantic") == manifest
    interrupted = task_root / f".{task_id(task)}.tmp-interrupted"
    interrupted.mkdir()
    (interrupted / "predictions.parquet").write_bytes(b"partial")
    assert validate_task_manifest(task_root, task, "semantic") == manifest
    shard = task_root / task_id(task) / "predictions.parquet"
    shard.write_bytes(b"corrupted")
    assert validate_task_manifest(task_root, task, "semantic") is None


def test_adaptive_worker_selection_stops_on_gain_or_memory() -> None:
    reports = [
        {"workers": 1, "throughput": 1.0, "gpu_memory_fraction": 0.20},
        {"workers": 2, "throughput": 1.20, "gpu_memory_fraction": 0.40},
        {"workers": 4, "throughput": 1.22, "gpu_memory_fraction": 0.60},
        {"workers": 8, "throughput": 2.0, "gpu_memory_fraction": 0.90},
    ]
    assert select_calibrated_worker_count(
        reports, maximum_memory_fraction=0.85, minimum_throughput_gain=0.05
    ) == 2


def test_oom_backoff_restarts_pool_and_resubmits(monkeypatch, tmp_path: Path) -> None:
    tasks = [TuningTask.create("spatial", {"width": width}, 7) for width in (8, 16, 32, 64)]
    worker_counts: list[int] = []

    class FakePool:
        attempts = 0

        def __init__(self, *, max_workers, **_):
            worker_counts.append(max_workers)
            self.attempt = FakePool.attempts
            FakePool.attempts += 1

        def submit(self, _worker, task):
            from concurrent.futures import Future

            future = Future()
            if self.attempt == 0:
                future.set_exception(RuntimeError("CUDA out of memory"))
            else:
                future.set_result({"task_id": task_id(task), "status": "complete"})
            return future

        def shutdown(self, **_):
            return None

    monkeypatch.setattr("ecuador_evolution.experiment.ProcessPoolExecutor", FakePool)
    values = {"parallel": {"retries": 2, "gpu_worker_cpu_threads": 1}}
    settings = Settings(values, tmp_path / "config.toml", tmp_path, "cpu", 7)
    results = _run_gpu_tasks(
        tasks,
        lambda task: task,
        settings=settings,
        task_root=tmp_path / "tasks",
        semantic="semantic",
        execution="execution",
        worker_count=4,
    )
    assert worker_counts == [4, 2]
    assert {result["task_id"] for result in results} == {task_id(task) for task in tasks}


@pytest.mark.cuda
def test_two_cuda_fits_register_as_private_mps_clients(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    context = multiprocessing.get_context("spawn")
    observed: set[int] = set()
    with private_mps_service(tmp_path, True), active_thread_percentage(2):
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
            futures = [pool.submit(_cuda_fit_worker, 4.0) for _ in range(2)]
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and len(observed) < 2:
                observed |= mps_client_pids()
                time.sleep(0.1)
            worker_pids = {future.result() for future in futures}
    assert len(worker_pids) == 2
    assert worker_pids <= observed
