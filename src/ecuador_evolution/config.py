from __future__ import annotations

import os
import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    values: dict[str, Any]
    config_path: Path
    data_root: Path
    device: str
    seed: int

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values.get(name, {}))

    def data_path(self, key: str) -> Path:
        relative = self.values.get("paths", {}).get(key, key)
        return self.data_root / relative

    def project_path(self, key: str) -> Path:
        relative = self.values.get("paths", {}).get(key, key)
        path = Path(relative)
        if path.is_absolute():
            return path
        return (self.config_path.parent.parent / path).resolve()


def load_settings(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    device: str | None = None,
    seed: int | None = None,
    parallel_backend: str | None = None,
    gpu_workers: str | int | None = None,
    cpu_workers: int | None = None,
    calibration: bool | None = None,
) -> Settings:
    path = Path(config_path).resolve()
    with path.open("rb") as stream:
        values = copy.deepcopy(tomllib.load(stream))
    parallel = values.setdefault("parallel", {})
    if parallel_backend is not None:
        parallel["backend"] = parallel_backend
    if gpu_workers is not None:
        parallel["gpu_workers"] = gpu_workers
    if cpu_workers is not None:
        parallel["cpu_workers"] = cpu_workers
    if calibration is not None:
        parallel["calibration"] = calibration
    root_value = data_root or os.environ.get("ECUADOR_DATA_ROOT") or values.get("data_root", "data")
    root = Path(root_value)
    if not root.is_absolute():
        root = (path.parent.parent / root).resolve()
    return Settings(
        values=values,
        config_path=path,
        data_root=root,
        device=device or str(values.get("device", "auto")),
        seed=int(seed if seed is not None else values.get("seed", 1605)),
    )
