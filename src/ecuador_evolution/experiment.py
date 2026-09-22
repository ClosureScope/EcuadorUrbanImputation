from __future__ import annotations

import itertools
import copy
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, TensorDataset

from .config import Settings
from .device import seed_everything, select_device
from .indicators import read_indicator_dictionary
from .metrics import evaluate_predictions, morans_i
from .model import build_reconstruction_model
from .neighbors import build_neighbor_index
from .parallel import (
    BaselineTrainingTask,
    NeuralTrainingTask,
    SplitSpec,
    TuningTask,
    active_thread_percentage,
    commit_task_artifacts,
    configure_worker_threads,
    execution_fingerprint,
    is_recoverable_worker_error,
    private_mps_service,
    select_calibrated_worker_count,
    semantic_fingerprint,
    task_id,
    validate_task_manifest,
)
from .splits import spatial_blocks, stable_random_mask, transfer_folds
from .training import fit_model
from .transforms import Standardizer, UnitAwareChangeTransformer


@dataclass(frozen=True)
class ReconstructionData:
    frame: pd.DataFrame
    geometry: gpd.GeoDataFrame
    indicators: list[str]
    values_2010: np.ndarray
    values_2022: np.ndarray
    focal_raw: np.ndarray
    neighbor_index: np.ndarray
    neighbor_mask: np.ndarray
    neighbor_distances: np.ndarray
    units: tuple[str, ...] = ()
    geometry_features: np.ndarray | None = None
    neighbor_relative: np.ndarray | None = None


def _paths(settings: Settings) -> tuple[Path, Path]:
    processed = settings.data_path("processed")
    artifacts = settings.data_path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    return processed, artifacts


def _load(settings: Settings) -> ReconstructionData:
    processed, _ = _paths(settings)
    matrix = pd.read_parquet(processed / "model_matrix.parquet")
    geometry = gpd.read_parquet(processed / "target_geometry.parquet").to_crs(
        settings.values.get("target_crs", "EPSG:32717")
    )
    dictionary = read_indicator_dictionary(settings.project_path("indicator_dictionary"))
    indicators = [
        item.name
        for item in dictionary
        if item.comparability in {"high", "medium"}
        and item.name in matrix
        and all(matrix.loc[matrix["year"].eq(year), item.name].notna().any() for year in (2010, 2022))
    ]
    units_by_name = {item.name: item.unit for item in dictionary}
    key = ["city_id", "target_2022_sector_id"]
    year_2010 = matrix.loc[matrix["year"] == 2010, key + indicators].set_index(key).sort_index()
    year_2022 = matrix.loc[matrix["year"] == 2022, key + indicators].set_index(key).sort_index()
    common = year_2010.index.intersection(year_2022.index)
    year_2010, year_2022 = year_2010.loc[common], year_2022.loc[common]
    frame = year_2010.reset_index()[key].copy()
    geometry = geometry.set_index("target_2022_sector_id").loc[frame["target_2022_sector_id"]].reset_index()
    if "city_id" in geometry:
        geometry = geometry.drop(columns="city_id")
    geometry.insert(0, "city_id", frame["city_id"].to_numpy())
    centroids = geometry.geometry.centroid
    coordinates = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    frame["sector_id"] = frame["target_2022_sector_id"]
    frame["centroid_x"], frame["centroid_y"] = coordinates[:, 0], coordinates[:, 1]
    geo = np.column_stack(
        [np.log1p(geometry.geometry.area), np.log1p(geometry.geometry.length), centroids.x, centroids.y]
    )
    cities = pd.get_dummies(frame["city_id"], dtype=float).to_numpy()
    values_2010, values_2022 = year_2010.to_numpy(float), year_2022.to_numpy(float)
    focal_raw = np.column_stack([values_2010, np.isnan(values_2010).astype(float), geo, cities])
    max_neighbors = int(settings.section("model").get("max_neighbors", 12))
    neighbor_index = np.zeros((len(frame), max_neighbors), dtype=np.int64)
    neighbor_mask = np.zeros((len(frame), max_neighbors), dtype=bool)
    neighbor_distances = np.zeros((len(frame), max_neighbors), dtype=np.float32)
    for _, positions in frame.groupby("city_id", sort=True).indices.items():
        positions = np.asarray(positions)
        local_index, local_mask, local_distance = build_neighbor_index(
            geometry.iloc[positions], max_neighbors=max_neighbors
        )
        neighbor_index[positions] = positions[local_index]
        neighbor_mask[positions] = local_mask
        neighbor_distances[positions] = local_distance
    normalized_coordinates = np.zeros_like(coordinates, dtype=np.float32)
    for _, positions in frame.groupby("city_id", sort=True).indices.items():
        positions = np.asarray(positions)
        local = coordinates[positions]
        center = np.median(local, axis=0)
        scale = np.percentile(local, 75, axis=0) - np.percentile(local, 25, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        normalized_coordinates[positions] = (local - center) / scale
    geometry_features = np.column_stack(
        [np.log1p(geometry.geometry.area), np.log1p(geometry.geometry.length), normalized_coordinates]
    ).astype(np.float32)
    neighbor_relative = np.zeros((len(frame), max_neighbors, 3), dtype=np.float32)
    for row in range(len(frame)):
        valid = neighbor_mask[row]
        if not valid.any():
            continue
        delta = coordinates[neighbor_index[row, valid]] - coordinates[row]
        distance = np.maximum(neighbor_distances[row, valid], 1e-6)
        neighbor_relative[row, valid, :2] = delta / distance[:, None]
        neighbor_relative[row, valid, 2] = np.log1p(distance)
    if neighbor_mask.any():
        distance_scale = float(np.median(neighbor_relative[..., 2][neighbor_mask]))
        neighbor_relative[..., 2] /= distance_scale if distance_scale > 0 else 1.0
    return ReconstructionData(
        frame,
        geometry,
        indicators,
        values_2010,
        values_2022,
        focal_raw,
        neighbor_index,
        neighbor_mask,
        neighbor_distances,
        tuple(units_by_name[name] for name in indicators),
        geometry_features,
        neighbor_relative,
    )


def _fit_imputer(values: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, Standardizer]:
    clean = values.copy()
    medians = np.nanmedian(clean[fit_mask], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    rows, columns = np.where(~np.isfinite(clean))
    clean[rows, columns] = medians[columns]
    scaler = Standardizer().fit(clean, fit_mask)
    return scaler.transform(clean).astype(np.float32), scaler


def build_masked_inputs(
    data: ReconstructionData,
    fit_mask: np.ndarray,
    jointly_masked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Standardizer]:
    """Create inputs whose 2022 statistics and values exclude every masked row."""
    if np.any(fit_mask & jointly_masked):
        raise ValueError("fit_mask and jointly_masked must be disjoint")
    focal, _ = _fit_imputer(data.focal_raw, fit_mask)
    targets, target_scaler = _fit_imputer(data.values_2022, fit_mask)
    visible = (~jointly_masked)[:, None] & np.isfinite(data.values_2022)
    safe_2022 = np.where(visible, targets, 0.0).astype(np.float32)
    neighbor_rows = data.neighbor_index
    neighbor_values = np.concatenate(
        [focal[neighbor_rows], safe_2022[neighbor_rows], visible.astype(np.float32)[neighbor_rows]], axis=-1
    )
    return focal, neighbor_values, target_scaler


def build_residual_inputs(
    data: ReconstructionData,
    fit_mask: np.ndarray,
    jointly_masked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, UnitAwareChangeTransformer]:
    """Build fit-only unit-aware change inputs without exposing jointly masked 2022 values."""
    if np.any(fit_mask & jointly_masked):
        raise ValueError("fit_mask and jointly_masked must be disjoint")
    if not data.units or data.geometry_features is None or data.neighbor_relative is None:
        raise ValueError("GAT data requires units and relative geometry")
    transformer = UnitAwareChangeTransformer(data.units).fit(data.values_2010, data.values_2022, fit_mask)
    base = transformer.base_transformed(data.values_2010)
    missing = ~np.isfinite(data.values_2010)
    focal_raw = np.column_stack([base, missing.astype(float), data.geometry_features])
    focal, _ = _fit_imputer(focal_raw, fit_mask)
    targets = transformer.transform(data.values_2010, data.values_2022).astype(np.float32)
    visible = (~jointly_masked)[:, None] & np.isfinite(targets)
    safe_changes = np.where(visible, targets, 0.0).astype(np.float32)
    neighbor_rows = data.neighbor_index
    neighbors = np.concatenate(
        [
            focal[neighbor_rows],
            safe_changes[neighbor_rows],
            visible.astype(np.float32)[neighbor_rows],
            data.neighbor_relative,
        ],
        axis=-1,
    ).astype(np.float32)
    return focal, neighbors, targets, base.astype(np.float32), transformer


def _loader(
    data: ReconstructionData,
    focal: np.ndarray,
    neighbors: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    shuffle: bool,
    extras: tuple[np.ndarray, ...] = (),
) -> DataLoader:
    target_mask = np.isfinite(data.values_2022[rows])
    tensors = [
        torch.from_numpy(focal[rows]),
        torch.from_numpy(neighbors[rows]),
        torch.from_numpy(data.neighbor_mask[rows]),
        torch.from_numpy(np.nan_to_num(targets[rows]).astype(np.float32)),
        torch.from_numpy(target_mask),
        *(torch.from_numpy(extra[rows].astype(np.float32)) for extra in extras),
    ]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=min(batch_size, max(len(dataset), 1)), shuffle=shuffle)


def _architecture_metadata(architecture: str, config: dict[str, object]) -> dict[str, object]:
    if architecture == "spatial":
        return {}
    if architecture == "gat":
        return {
            "heads": int(config.get("heads", 4)),
            "batch_size": int(config.get("batch_size", 512)),
            "context_dropout": float(config.get("context_dropout", 0.25)),
            "indicator_dropout": float(config.get("indicator_dropout", 0.10)),
            "native_mae_weight": float(config.get("native_mae_weight", 0.10)),
        }
    return {
        "heads": int(config.get("heads", 8)),
        "encoder_layers": int(config.get("encoder_layers", 6)),
        "decoder_layers": int(config.get("decoder_layers", 4)),
        "ff_ratio": int(config.get("ff_ratio", 4)),
        "batch_size": int(config.get("batch_size", 128)),
        "accumulation_steps": int(config.get("accumulation_steps", 2)),
    }


def _fit_architecture(
    architecture: str,
    data: ReconstructionData,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    params: dict[str, float | int],
    settings: Settings,
    seed: int,
):
    seed_everything(seed)
    jointly_masked = validation_mask | test_mask
    section = "model" if architecture == "spatial" else architecture.replace("-", "_")
    if architecture == "mae":
        section = "mae"
    config = settings.section(section)
    if architecture == "gat":
        config["heads"] = 4 if int(params["width"]) <= 128 else 8
    if architecture == "gat":
        focal, neighbors, targets, base, target_scaler = build_residual_inputs(
            data, fit_mask, jointly_masked
        )
    else:
        focal, neighbors, target_scaler = build_masked_inputs(data, fit_mask, jointly_masked)
        targets = target_scaler.transform(data.values_2022).astype(np.float32)
        base = None
    model = build_reconstruction_model(
        architecture, focal.shape[1], neighbors.shape[2], len(data.indicators), params, config
    )
    batch_size = int(config.get("batch_size", 128 if architecture == "mae" else 256))
    train_extras: tuple[np.ndarray, ...] = () if base is None else (base,)
    loss_function = validation_metric = None
    if architecture == "gat":
        native_weight = float(config.get("native_mae_weight", 0.10))

        def native_macro_mae(
            prediction: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            base_transformed: torch.Tensor,
        ) -> torch.Tensor:
            predicted_native = target_scaler.inverse_torch(prediction, base_transformed)
            target_native = target_scaler.inverse_torch(target, base_transformed)
            indicator_losses = []
            for index in range(target.shape[1]):
                selected = mask[:, index]
                if selected.any():
                    indicator_losses.append(
                        torch.abs(predicted_native[selected, index] - target_native[selected, index]).mean()
                    )
            return torch.stack(indicator_losses).mean() if indicator_losses else prediction.sum() * 0

        def combined_loss(
            prediction: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            base_transformed: torch.Tensor,
        ) -> torch.Tensor:
            from .model import masked_huber_loss

            return masked_huber_loss(prediction, target, mask) + native_weight * native_macro_mae(
                prediction, target, mask, base_transformed
            )

        loss_function = combined_loss
        validation_metric = native_macro_mae
    result = fit_model(
        model,
        _loader(
            data, focal, neighbors, targets, np.flatnonzero(fit_mask), batch_size, True, train_extras
        ),
        _loader(
            data,
            focal,
            neighbors,
            targets,
            np.flatnonzero(validation_mask),
            batch_size,
            False,
            train_extras,
        ),
        device=select_device(settings.device),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0001)),
        max_epochs=int(config.get("max_epochs", 200)),
        patience=int(config.get("patience", 20)),
        accumulation_steps=int(config.get("accumulation_steps", 2 if architecture == "mae" else 1)),
        loss_function=loss_function,
        validation_metric=validation_metric,
        gradient_clip=float(config.get("gradient_clip", 1.0)) if architecture == "gat" else None,
        warmup_fraction=float(config.get("warmup_fraction", 0.05)) if architecture == "gat" else 0.0,
    )
    device = select_device(settings.device)
    result.model.eval()
    with torch.no_grad():
        transformed = result.model(
            torch.from_numpy(focal).to(device),
            torch.from_numpy(neighbors).to(device),
            torch.from_numpy(data.neighbor_mask).to(device),
        ).cpu().numpy()
    prediction = (
        target_scaler.inverse(transformed, data.values_2010)
        if architecture == "gat"
        else target_scaler.inverse(transformed)
    )
    return result, prediction, target_scaler


def _idw(data: ReconstructionData, visible: np.ndarray) -> np.ndarray:
    result = np.full_like(data.values_2022, np.nan)
    global_mean = np.nanmean(data.values_2022[visible], axis=0)
    for row in range(len(result)):
        neighbors = data.neighbor_index[row, data.neighbor_mask[row]]
        distances = data.neighbor_distances[row, data.neighbor_mask[row]]
        usable = visible[neighbors]
        if usable.any():
            weights = 1.0 / np.maximum(distances[usable], 1e-6)
            values = data.values_2022[neighbors[usable]]
            valid = np.isfinite(values)
            weighted = np.sum(valid * weights[:, None], axis=0)
            result[row] = np.divide(
                np.nansum(values * weights[:, None], axis=0),
                weighted,
                out=global_mean.copy(),
                where=weighted > 0,
            )
        else:
            result[row] = global_mean
    return result


def _architectures(settings: Settings) -> list[str]:
    requested = settings.section("model").get("architectures")
    architectures = ["spatial"] if requested is None else [str(item) for item in requested]
    unknown = set(architectures) - {"spatial", "mae", "gat"}
    if unknown:
        raise ValueError(f"Unsupported reconstruction architectures: {sorted(unknown)}")
    return architectures


def _candidates(architecture: str, settings: Settings) -> list[dict[str, float | int]]:
    if architecture == "gat":
        config = settings.section("gat")
        return [
            {"width": int(width), "dropout": float(dropout), "learning_rate": float(rate)}
            for width, dropout, rate in itertools.product(
                config.get("widths", [128, 256]),
                config.get("dropouts", [0.05, 0.15]),
                config.get("learning_rates", [0.0003, 0.001]),
            )
        ]
    config = settings.section("model" if architecture == "spatial" else "mae")
    widths_key = "hidden_widths" if architecture == "spatial" else "widths"
    width_key = "hidden_width" if architecture == "spatial" else "width"
    default_widths = [64, 128] if architecture == "spatial" else [128, 256]
    default_dropouts = [0.1, 0.25] if architecture == "spatial" else [0.1, 0.2]
    default_rates = [0.001, 0.0003] if architecture == "spatial" else [0.0003, 0.0001]
    return [
        {width_key: int(width), "dropout": float(dropout), "learning_rate": float(rate)}
        for width, dropout, rate in itertools.product(
            config.get(widths_key, default_widths),
            config.get("dropouts", default_dropouts),
            config.get("learning_rates", default_rates),
        )
    ]


def _macro_mae(truth: np.ndarray, prediction: np.ndarray, rows: np.ndarray) -> float:
    scores = []
    for index in range(truth.shape[1]):
        valid = rows & np.isfinite(truth[:, index]) & np.isfinite(prediction[:, index])
        if valid.any():
            scores.append(float(np.mean(np.abs(truth[valid, index] - prediction[valid, index]))))
    return float(np.mean(scores)) if scores else float("inf")


_WORKER_SETTINGS: Settings | None = None
_WORKER_DATA: ReconstructionData | None = None
_WORKER_TASK_ROOT: Path | None = None
_WORKER_SEMANTIC = ""
_WORKER_EXECUTION = ""


def _ensure_fingerprints(artifacts: Path, settings: Settings) -> tuple[str, str]:
    semantic = semantic_fingerprint(settings.values)
    execution = execution_fingerprint(settings.values, settings.device)
    semantic_path = artifacts / "semantic-fingerprint.txt"
    if semantic_path.exists() and semantic_path.read_text(encoding="utf-8").strip() != semantic:
        raise RuntimeError(
            f"Artifact directory {artifacts} belongs to a different semantic experiment; use a fresh artifacts path"
        )
    semantic_path.write_text(semantic + "\n", encoding="utf-8")
    (artifacts / "config-fingerprint.txt").write_text(semantic + "\n", encoding="utf-8")
    (artifacts / "execution-fingerprint.txt").write_text(execution + "\n", encoding="utf-8")
    return semantic, execution


def _tuning_masks(data: ReconstructionData, settings: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation_fraction = float(settings.section("evaluation").get("validation_fraction", 0.10))
    test = stable_random_mask(data.frame["sector_id"], 0.20, settings.seed)
    remaining_rows = np.flatnonzero(~test)
    local_validation = stable_random_mask(
        data.frame.loc[~test, "sector_id"], validation_fraction, settings.seed + 91
    )
    validation = np.zeros(len(data.frame), dtype=bool)
    validation[remaining_rows[local_validation]] = True
    return ~(test | validation), validation, test


def _split_specs(data: ReconstructionData, settings: Settings) -> list[SplitSpec]:
    evaluation = settings.section("evaluation")
    blocks = spatial_blocks(data.frame, blocks_per_city=int(evaluation.get("spatial_blocks", 5)))
    seeds = [int(seed) for seed in settings.section("model").get("seeds", [1605, 1606, 1607])]
    specs: list[SplitSpec] = []
    for seed in seeds:
        for fraction in evaluation.get("random_mask_fractions", [0.10, 0.20, 0.30]):
            specs.append(SplitSpec("random-mask", f"{int(float(fraction) * 100)}%", seed))
        for block in sorted(np.unique(blocks).tolist()):
            specs.append(SplitSpec("block-mask", f"20%-block-{block}", seed))
        for held_city in transfer_folds(data.frame["city_id"]):
            specs.append(SplitSpec("held-city", str(held_city), seed))
    return specs


def _split_masks(
    data: ReconstructionData, settings: Settings, split: SplitSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split.protocol == "random-mask":
        test = stable_random_mask(data.frame["sector_id"], float(split.fold.rstrip("%")) / 100, split.seed)
    elif split.protocol == "block-mask":
        blocks = spatial_blocks(
            data.frame, blocks_per_city=int(settings.section("evaluation").get("spatial_blocks", 5))
        )
        test = blocks == int(split.fold.rsplit("-", 1)[-1])
    elif split.protocol == "held-city":
        test = data.frame["city_id"].astype(str).eq(split.fold).to_numpy()
    else:
        raise ValueError(f"Unknown protocol {split.protocol!r}")
    remaining = ~test
    local_validation = stable_random_mask(
        data.frame.loc[remaining, "sector_id"],
        float(settings.section("evaluation").get("validation_fraction", 0.10)),
        split.seed + 701,
    )
    validation = np.zeros(len(test), dtype=bool)
    validation[np.flatnonzero(remaining)[local_validation]] = True
    return ~(test | validation), validation, test


def _prediction_frame(
    data: ReconstructionData,
    predictions: dict[str, np.ndarray],
    split: SplitSpec,
    test: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, prediction in sorted(predictions.items()):
        for row in np.flatnonzero(test):
            for index, indicator in enumerate(data.indicators):
                rows.append(
                    {
                        "model": model,
                        "protocol": split.protocol,
                        "fold": split.fold,
                        "seed": split.seed,
                        "city_id": data.frame.loc[row, "city_id"],
                        "sector_id": data.frame.loc[row, "sector_id"],
                        "indicator": indicator,
                        "truth": data.values_2022[row, index],
                        "prediction": prediction[row, index],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["model", "city_id", "sector_id", "indicator"], kind="stable"
    ).reset_index(drop=True)


def _gpu_worker_initializer(
    settings: Settings,
    task_root: Path,
    semantic: str,
    execution: str,
    cpu_threads: int,
    max_memory_fraction: float,
) -> None:
    global _WORKER_SETTINGS, _WORKER_DATA, _WORKER_TASK_ROOT, _WORKER_SEMANTIC, _WORKER_EXECUTION
    configure_worker_threads(cpu_threads)
    _WORKER_SETTINGS = settings
    _WORKER_TASK_ROOT = task_root
    _WORKER_SEMANTIC = semantic
    _WORKER_EXECUTION = execution
    _WORKER_DATA = _load(settings)
    if select_device(settings.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
        torch.empty(1, device="cuda")


def _baseline_worker_initializer(
    settings: Settings,
    task_root: Path,
    semantic: str,
    execution: str,
    cpu_threads: int,
) -> None:
    global _WORKER_SETTINGS, _WORKER_DATA, _WORKER_TASK_ROOT, _WORKER_SEMANTIC, _WORKER_EXECUTION
    configure_worker_threads(cpu_threads)
    _WORKER_SETTINGS = settings
    _WORKER_TASK_ROOT = task_root
    _WORKER_SEMANTIC = semantic
    _WORKER_EXECUTION = execution
    _WORKER_DATA = _load(settings)


def _worker_state() -> tuple[Settings, ReconstructionData, Path]:
    if _WORKER_SETTINGS is None or _WORKER_DATA is None or _WORKER_TASK_ROOT is None:
        raise RuntimeError("Worker was not initialized")
    return _WORKER_SETTINGS, _WORKER_DATA, _WORKER_TASK_ROOT


def _peak_gpu_memory() -> int:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return 0


def _run_tuning_task(task: TuningTask) -> dict[str, object]:
    settings, data, task_root = _worker_state()
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fit, validation, test = _tuning_masks(data, settings)
    result, prediction, _ = _fit_architecture(
        task.model, data, fit, validation, test, task.params, settings, task.seed
    )
    section = "model" if task.model == "spatial" else task.model.replace("-", "_")
    metrics = {
        "model": task.model,
        **task.params,
        **_architecture_metadata(task.model, settings.section(section)),
        "validation_mae": _macro_mae(data.values_2022, prediction, validation),
        "best_epoch": result.best_epoch,
    }
    split = SplitSpec("tuning", "validation", task.seed)
    with tempfile.TemporaryDirectory(dir=task_root) as scratch:
        shard = Path(scratch) / "predictions.parquet"
        _prediction_frame(data, {task.model: prediction}, split, validation).to_parquet(shard, index=False)
        return commit_task_artifacts(
            task_root,
            task,
            semantic=_WORKER_SEMANTIC,
            execution=_WORKER_EXECUTION,
            started_at=started,
            metrics=metrics,
            files={"predictions": shard},
            row_count=int(validation.sum() * len(data.indicators)),
            peak_gpu_memory=_peak_gpu_memory(),
        )


def _checkpoint_payload(
    task: NeuralTrainingTask, data: ReconstructionData, result, scaler, settings: Settings
) -> dict[str, object]:
    section = "model" if task.model == "spatial" else task.model.replace("-", "_")
    architecture_config = settings.section(section)
    if task.model == "gat":
        architecture_config["heads"] = 4 if int(task.params["width"]) <= 128 else 8
        focal_dim = len(data.indicators) * 2 + 4
        neighbor_dim = focal_dim + len(data.indicators) * 2 + 3
        scaler_metadata = {
            "target_center": scaler.center.tolist(),
            "target_scale": scaler.scale.tolist(),
            "base_median": scaler.base_median.tolist(),
            "units": list(scaler.units),
        }
    else:
        focal_dim = data.focal_raw.shape[1]
        neighbor_dim = data.focal_raw.shape[1] + len(data.indicators) * 2
        scaler_metadata = {"target_mean": scaler.mean.tolist(), "target_scale": scaler.scale.tolist()}
    return {
        "model": task.model,
        "state_dict": result.model.state_dict(),
        "params": task.params,
        "architecture": {
            "focal_dim": focal_dim,
            "neighbor_dim": neighbor_dim,
            "output_dim": len(data.indicators),
            **_architecture_metadata(task.model, architecture_config),
        },
        "indicators": data.indicators,
        "seed": task.split.seed,
        "protocol": task.split.protocol,
        "fold": task.split.fold,
        "visibility_rule": "validation and test sector 2022 values excluded from every input token",
        **scaler_metadata,
    }


def _run_neural_task(task: NeuralTrainingTask) -> dict[str, object]:
    settings, data, task_root = _worker_state()
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fit, validation, test = _split_masks(data, settings, task.split)
    result, prediction, scaler = _fit_architecture(
        task.model, data, fit, validation, test, task.params, settings, task.split.seed
    )
    with tempfile.TemporaryDirectory(dir=task_root) as scratch:
        shard = Path(scratch) / "predictions.parquet"
        checkpoint = Path(scratch) / "checkpoint.pt"
        frame = _prediction_frame(data, {task.model: prediction}, task.split, test)
        frame.to_parquet(shard, index=False)
        torch.save(_checkpoint_payload(task, data, result, scaler, settings), checkpoint)
        return commit_task_artifacts(
            task_root,
            task,
            semantic=_WORKER_SEMANTIC,
            execution=_WORKER_EXECUTION,
            started_at=started,
            metrics={
                "model": task.model,
                "protocol": task.split.protocol,
                "fold": task.split.fold,
                "seed": task.split.seed,
                "best_epoch": result.best_epoch,
                "validation_loss": result.best_validation_loss,
            },
            files={"predictions": shard, "checkpoint": checkpoint},
            row_count=len(frame),
            peak_gpu_memory=_peak_gpu_memory(),
        )


def _run_baseline_task(task: BaselineTrainingTask) -> dict[str, object]:
    settings, data, task_root = _worker_state()
    started = time.time()
    fit, _, test = _split_masks(data, settings, task.split)
    focal, _ = _fit_imputer(data.focal_raw, fit)
    target_mean = np.nanmean(data.values_2022[fit], axis=0)
    clean_target = np.where(np.isfinite(data.values_2022), data.values_2022, target_mean)
    city_mean = np.empty_like(data.values_2022)
    for city in data.frame["city_id"].unique():
        city_rows = data.frame["city_id"].eq(city).to_numpy()
        available = fit & city_rows
        city_mean[city_rows] = np.nanmean(data.values_2022[available], axis=0) if available.any() else target_mean
    config = settings.section("baselines")
    width = int(config.get("non_spatial_mlp_width", 128))
    predictions = {
        "2010-carry-forward": data.values_2010,
        "city-mean": city_mean,
        "inverse-distance": _idw(data, fit),
        "ridge": Ridge(alpha=float(config.get("ridge_alpha", 1.0))).fit(
            focal[fit], clean_target[fit]
        ).predict(focal),
        "non-spatial-mlp": MLPRegressor(
            hidden_layer_sizes=(width,),
            max_iter=int(config.get("non_spatial_mlp_max_iter", 500)),
            random_state=task.split.seed,
        ).fit(focal[fit], clean_target[fit]).predict(focal),
    }
    frame = _prediction_frame(data, predictions, task.split, test)
    with tempfile.TemporaryDirectory(dir=task_root) as scratch:
        shard = Path(scratch) / "predictions.parquet"
        frame.to_parquet(shard, index=False)
        return commit_task_artifacts(
            task_root,
            task,
            semantic=_WORKER_SEMANTIC,
            execution=_WORKER_EXECUTION,
            started_at=started,
            metrics={
                "models": list(task.models),
                "protocol": task.split.protocol,
                "fold": task.split.fold,
                "seed": task.split.seed,
            },
            files={"predictions": shard},
            row_count=len(frame),
            peak_gpu_memory=0,
        )


def _run_calibration_task(task: TuningTask) -> dict[str, float | int]:
    settings, data, _ = _worker_state()
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fit, validation, test = _tuning_masks(data, settings)
    _fit_architecture(task.model, data, fit, validation, test, task.params, settings, task.seed)
    return {"duration_seconds": time.perf_counter() - started, "peak_gpu_memory": _peak_gpu_memory()}


def _calibration_settings(settings: Settings, architecture: str, epochs: int) -> Settings:
    values = copy.deepcopy(settings.values)
    section = "model" if architecture == "spatial" else architecture.replace("-", "_")
    values.setdefault(section, {})["max_epochs"] = epochs
    values[section]["patience"] = epochs
    return Settings(values, settings.config_path, settings.data_root, settings.device, settings.seed)


def _calibrate_workers(
    settings: Settings,
    task_root: Path,
    semantic: str,
    execution: str,
    architecture: str,
    candidate: dict[str, float | int],
) -> tuple[int, list[dict[str, object]]]:
    config = settings.section("parallel")
    levels = sorted(
        {int(level) for level in config.get("candidate_levels", [1, 2, 4, 8]) if int(level) > 0}
    )
    maximum = int(config.get("max_gpu_workers", 8))
    levels = [level for level in levels if level <= maximum]
    calibrated = _calibration_settings(settings, architecture, int(config.get("calibration_epochs", 3)))
    reports: list[dict[str, object]] = []
    selected = 1
    for level in levels:
        context = multiprocessing.get_context("spawn")
        started = time.perf_counter()
        try:
            with active_thread_percentage(level), ProcessPoolExecutor(
                max_workers=level,
                mp_context=context,
                initializer=_gpu_worker_initializer,
                initargs=(
                    calibrated,
                    task_root,
                    semantic,
                    execution,
                    int(config.get("gpu_worker_cpu_threads", 1)),
                    float(config.get("max_gpu_memory_fraction", 0.85)),
                ),
            ) as pool:
                results = list(
                    pool.map(
                        _run_calibration_task,
                        [TuningTask.create(architecture, candidate, settings.seed, slot) for slot in range(level)],
                    )
                )
            elapsed = time.perf_counter() - started
            throughput = level / max(elapsed, 1e-9)
            peak = sum(int(item["peak_gpu_memory"]) for item in results)
            total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 1
            memory_fraction = peak / total_memory
            report = {
                "workers": level,
                "elapsed_seconds": elapsed,
                "throughput": throughput,
                "aggregate_peak_gpu_memory": peak,
                "gpu_memory_fraction": memory_fraction,
            }
            reports.append(report)
            chosen = select_calibrated_worker_count(
                reports,
                maximum_memory_fraction=float(config.get("max_gpu_memory_fraction", 0.85)),
                minimum_throughput_gain=float(config.get("minimum_throughput_gain", 0.05)),
            )
            report["throughput_gain"] = None if len(reports) == 1 else (
                throughput / float(reports[-2]["throughput"]) - 1
            )
            report["accepted"] = chosen == level
            if chosen != level:
                break
            selected = chosen
        except BaseException as error:
            if not is_recoverable_worker_error(error):
                raise
            reports.append({"workers": level, "accepted": False, "error": str(error)})
            break
    return selected, reports


def _run_gpu_tasks(
    tasks: list[TuningTask] | list[NeuralTrainingTask],
    worker,
    *,
    settings: Settings,
    task_root: Path,
    semantic: str,
    execution: str,
    worker_count: int,
) -> list[dict[str, object]]:
    config = settings.section("parallel")
    remaining = {task_id(task): task for task in tasks}
    completed: dict[str, dict[str, object]] = {}
    retries = int(config.get("retries", 2))
    current_workers = max(1, min(worker_count, len(tasks) or 1))
    while remaining:
        for identifier, task in list(remaining.items()):
            manifest = validate_task_manifest(task_root, task, semantic)
            if manifest is not None:
                completed[identifier] = manifest
                del remaining[identifier]
        if not remaining:
            break
        context = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=current_workers,
            mp_context=context,
            initializer=_gpu_worker_initializer,
            initargs=(
                settings,
                task_root,
                semantic,
                execution,
                int(config.get("gpu_worker_cpu_threads", 1)),
                float(config.get("max_gpu_memory_fraction", 0.85)),
            ),
        )
        recoverable: BaseException | None = None
        futures = {pool.submit(worker, task): identifier for identifier, task in remaining.items()}
        try:
            for future in as_completed(futures):
                identifier = futures[future]
                try:
                    completed[identifier] = future.result()
                    remaining.pop(identifier, None)
                except BaseException as error:
                    if not is_recoverable_worker_error(error):
                        raise
                    recoverable = error
                    break
        finally:
            pool.shutdown(wait=recoverable is None, cancel_futures=recoverable is not None)
        if recoverable is not None:
            if retries <= 0:
                raise RuntimeError("GPU worker retries exhausted") from recoverable
            retries -= 1
            current_workers = max(1, current_workers // 2)
            print(
                f"recoverable GPU failure; retrying {len(remaining)} task(s) with {current_workers} worker(s)",
                flush=True,
            )
    return [completed[task_id(task)] for task in tasks]


def _aggregate_predictions(
    artifacts: Path,
    task_root: Path,
    manifests: list[dict[str, object]],
) -> None:
    output = artifacts / "predictions.parquet"
    temporary = artifacts / ".predictions.parquet.tmp"
    writer = None
    try:
        for manifest in sorted(manifests, key=lambda item: str(item["task_id"])):
            metadata = manifest.get("files", {}).get("predictions")
            if metadata is None:
                continue
            shard = task_root / str(manifest["task_id"]) / str(metadata["path"])
            table = parquet.read_table(shard)
            if writer is None:
                writer = parquet.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
        if writer is None:
            raise RuntimeError("No prediction shards were produced")
        writer.close()
        writer = None
        os.replace(temporary, output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _materialize_checkpoint_contract(
    artifacts: Path,
    task_root: Path,
    tasks: list[NeuralTrainingTask],
    manifests: list[dict[str, object]],
) -> None:
    for task, manifest in zip(tasks, manifests, strict=True):
        metadata = manifest["files"]["checkpoint"]
        source = task_root / str(manifest["task_id"]) / str(metadata["path"])
        safe_fold = task.split.fold.replace("/", "-")
        destination = artifacts / (
            f"{task.model}-{task.split.protocol}-{safe_fold}-seed-{task.split.seed}.pt"
        )
        if destination.exists() and os.path.samefile(source, destination):
            continue
        temporary = artifacts / f".{destination.name}.tmp-{os.getpid()}"
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        os.replace(temporary, destination)


def train(settings: Settings) -> dict[str, object]:
    data = _load(settings)
    architectures = _architectures(settings)
    _, artifacts = _paths(settings)
    task_root = artifacts / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    semantic, execution = _ensure_fingerprints(artifacts, settings)
    parallel = settings.section("parallel")
    backend = str(parallel.get("backend", "serial"))
    cuda = select_device(settings.device).type == "cuda"
    mps_enabled = backend == "nvidia-mps" and cuda
    if backend == "nvidia-mps" and not cuda:
        raise RuntimeError("parallel.backend=nvidia-mps requires a CUDA device")

    tuning_tasks = [
        TuningTask.create(architecture, candidate, settings.seed)
        for architecture in architectures
        for candidate in _candidates(architecture, settings)
    ]
    requested = parallel.get("gpu_workers", 1)
    maximum = int(parallel.get("max_gpu_workers", 8))
    explicit_workers = None if str(requested) == "auto" else max(1, int(requested))
    worker_count = min(explicit_workers or maximum, maximum)
    calibration_report: list[dict[str, object]] = []
    calibration_path = artifacts / "calibration.json"

    with private_mps_service(artifacts, mps_enabled):
        if explicit_workers is None and bool(parallel.get("calibration", True)) and cuda:
            cached = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
            if cached.get("execution_fingerprint") == execution:
                worker_count = int(cached["selected_gpu_workers"])
                calibration_report = list(cached.get("levels", []))
            else:
                representative = tuning_tasks[0]
                worker_count, calibration_report = _calibrate_workers(
                    settings,
                    task_root,
                    semantic,
                    execution,
                    representative.model,
                    representative.params,
                )
        calibration_path.write_text(
            json.dumps(
                {
                    "execution_fingerprint": execution,
                    "selected_gpu_workers": worker_count,
                    "levels": calibration_report,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tuning_manifests = _run_gpu_tasks(
            tuning_tasks,
            _run_tuning_task,
            settings=settings,
            task_root=task_root,
            semantic=semantic,
            execution=execution,
            worker_count=worker_count,
        )
        tuning_rows = [dict(manifest["metrics"]) for manifest in tuning_manifests]
        selected: dict[str, dict[str, float | int]] = {}
        for architecture in architectures:
            candidates = [row for row in tuning_rows if row["model"] == architecture]
            winner = min(
                candidates,
                key=lambda row: (
                    float(row["validation_mae"]),
                    json.dumps(
                        {key: row[key] for key in sorted(_candidates(architecture, settings)[0])},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            parameter_keys = sorted(_candidates(architecture, settings)[0])
            selected[architecture] = {key: winner[key] for key in parameter_keys}

        splits = _split_specs(data, settings)
        neural_tasks = [
            NeuralTrainingTask.create(architecture, selected[architecture], split)
            for split in splits
            for architecture in architectures
        ]
        baseline_tasks = [BaselineTrainingTask(split) for split in splits]
        pending_baselines = [
            task for task in baseline_tasks if validate_task_manifest(task_root, task, semantic) is None
        ]
        baseline_pool = None
        baseline_futures = {}
        if pending_baselines:
            context = multiprocessing.get_context("spawn")
            baseline_pool = ProcessPoolExecutor(
                max_workers=min(int(parallel.get("cpu_workers", 4)), len(pending_baselines)),
                mp_context=context,
                initializer=_baseline_worker_initializer,
                initargs=(
                    settings,
                    task_root,
                    semantic,
                    execution,
                    int(parallel.get("baseline_worker_cpu_threads", 3)),
                ),
            )
            baseline_futures = {baseline_pool.submit(_run_baseline_task, task): task for task in pending_baselines}
        try:
            neural_manifests = _run_gpu_tasks(
                neural_tasks,
                _run_neural_task,
                settings=settings,
                task_root=task_root,
                semantic=semantic,
                execution=execution,
                worker_count=worker_count,
            )
            for future in as_completed(baseline_futures):
                future.result()
        finally:
            if baseline_pool is not None:
                baseline_pool.shutdown(wait=True, cancel_futures=False)

    baseline_manifests = [
        validate_task_manifest(task_root, task, semantic) for task in baseline_tasks
    ]
    if any(manifest is None for manifest in baseline_manifests):
        raise RuntimeError("One or more baseline tasks did not commit valid artifacts")
    valid_baselines = [manifest for manifest in baseline_manifests if manifest is not None]
    fit_summaries = [dict(manifest["metrics"]) for manifest in neural_manifests]
    fit_summaries.sort(key=lambda row: (str(row["protocol"]), str(row["fold"]), str(row["model"]), int(row["seed"])))
    tuning_rows.sort(
        key=lambda row: (str(row["model"]), json.dumps(row, sort_keys=True, default=str))
    )
    pd.DataFrame(tuning_rows).to_csv(artifacts / "tuning.csv", index=False)
    (artifacts / "tuning-progress.json").write_text(json.dumps(tuning_rows, indent=2) + "\n", encoding="utf-8")
    (artifacts / "fit-progress.json").write_text(json.dumps(fit_summaries, indent=2) + "\n", encoding="utf-8")
    _aggregate_predictions(artifacts, task_root, [*neural_manifests, *valid_baselines])
    _materialize_checkpoint_contract(artifacts, task_root, neural_tasks, neural_manifests)
    run = {
        "task": "masked-reconstruction",
        "config_fingerprint": semantic,
        "semantic_fingerprint": semantic,
        "execution_fingerprint": execution,
        "parallel_backend": backend,
        "gpu_workers": worker_count,
        "architectures": architectures,
        "selected": selected,
        "architecture_settings": {
            architecture: _architecture_metadata(
                architecture,
                settings.section("model" if architecture == "spatial" else architecture.replace("-", "_")),
            )
            for architecture in architectures
        },
        "tuning_seed": settings.seed,
        "fits": fit_summaries,
        "indicators": data.indicators,
        "mask_protocols": sorted({split.protocol for split in splits}),
        "task_manifest_count": len(tuning_manifests) + len(neural_manifests) + len(valid_baselines),
    }
    (artifacts / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return run


def evaluate(settings: Settings) -> dict[str, object]:
    data = _load(settings)
    _, artifacts = _paths(settings)
    predictions = pd.read_parquet(artifacts / "predictions.parquet")
    population_index = data.indicators.index("population") if "population" in data.indicators else 0
    population = dict(zip(data.frame["sector_id"], data.values_2022[:, population_index], strict=True))
    tables = []
    for (protocol, fold, model, seed), group in predictions.groupby(["protocol", "fold", "model", "seed"]):
        truth = group.pivot(index="sector_id", columns="indicator", values="truth").reindex(columns=data.indicators)
        prediction = group.pivot(index="sector_id", columns="indicator", values="prediction").reindex(columns=data.indicators)
        table = evaluate_predictions(
            truth.to_numpy(),
            prediction.to_numpy(),
            data.indicators,
            population_weights=np.array([population[item] for item in truth.index]),
        )
        table.insert(0, "seed", seed)
        table.insert(0, "model", model)
        table.insert(0, "fold", fold)
        table.insert(0, "protocol", protocol)
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(artifacts / "metrics.csv", index=False)
    moran_rows = []
    neural = predictions.loc[predictions["model"].isin(_architectures(settings))]
    for (model, protocol, fold, seed, indicator), group in neural.groupby(
        ["model", "protocol", "fold", "seed", "indicator"]
    ):
        ordered = data.frame[["sector_id"]].merge(
            group[["sector_id", "truth", "prediction"]], on="sector_id", how="left"
        )
        moran_rows.append(
            {
                "model": model,
                "protocol": protocol,
                "fold": fold,
                "seed": seed,
                "indicator": indicator,
                "morans_i": morans_i(
                    (ordered["truth"] - ordered["prediction"]).to_numpy(),
                    data.neighbor_index,
                    data.neighbor_mask,
                ),
            }
        )
    pd.DataFrame(moran_rows).to_csv(artifacts / "residual-morans-i.csv", index=False)
    macro = metrics.loc[metrics["indicator"] == "macro"].groupby(["protocol", "model"])[
        ["mae", "rmse", "r2", "weighted_mae"]
    ].agg(["mean", "std"])
    summary: dict[str, object] = {
        f"{protocol}/{model}": {f"{metric}_{stat}": value for (metric, stat), value in row.items()}
        for (protocol, model), row in macro.iterrows()
    }
    baseline_names = {
        "2010-carry-forward",
        "city-mean",
        "inverse-distance",
        "ridge",
        "non-spatial-mlp",
    }
    protocol_means = (
        metrics.loc[metrics["indicator"] == "macro"]
        .groupby(["protocol", "model"], as_index=False)["mae"]
        .mean()
    )
    primary_model = _architectures(settings)[0]
    protocol_gate: dict[str, dict[str, object]] = {}
    for protocol, group in protocol_means.groupby("protocol"):
        baselines = group.loc[group["model"].isin(baseline_names)].sort_values("mae")
        model_rows = group.loc[group["model"] == primary_model]
        if baselines.empty or model_rows.empty:
            protocol_gate[str(protocol)] = {"passed": False, "reason": "missing model or baseline metrics"}
            continue
        baseline = baselines.iloc[0]
        model_mae = float(model_rows.iloc[0]["mae"])
        baseline_mae = float(baseline["mae"])
        protocol_gate[str(protocol)] = {
            "passed": model_mae < baseline_mae,
            "model": primary_model,
            "model_mae": model_mae,
            "best_baseline": str(baseline["model"]),
            "baseline_mae": baseline_mae,
            "mae_difference": model_mae - baseline_mae,
        }
    summary["win_gate"] = {
        "primary_model": primary_model,
        "passed": len(protocol_gate) == 3 and all(bool(item.get("passed")) for item in protocol_gate.values()),
        "protocols": protocol_gate,
    }
    (artifacts / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def report(settings: Settings) -> str:
    _, artifacts = _paths(settings)
    metrics = pd.read_csv(artifacts / "metrics.csv")
    macro = metrics.loc[metrics["indicator"] == "macro"]
    figure = artifacts / "macro-mae.png"
    macro.groupby("model")["mae"].mean().sort_values().plot.bar(color="#5b8c5a")
    plt.ylabel("Macro MAE (native units)")
    plt.title("Masked 2022 reconstruction")
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()
    run = json.loads((artifacts / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads((artifacts / "evaluation.json").read_text(encoding="utf-8"))
    gate = evaluation.get("win_gate", {})
    architecture_lines = [f"- `{name}`: `{params}`" for name, params in run["selected"].items()]
    gate_lines = [
        f"- `{protocol}`: {'PASS' if result.get('passed') else 'FAIL'} — "
        f"{result.get('model')} {result.get('model_mae', float('nan')):.6g} vs "
        f"{result.get('best_baseline')} {result.get('baseline_mae', float('nan')):.6g}"
        for protocol, result in gate.get("protocols", {}).items()
    ]
    output = artifacts / "report.md"
    output.write_text(
        "\n".join(
            [
                "# Masked reconstruction report",
                "",
                "The gat model learns unit-aware 2010-to-2022 changes and a gated correction from "
                "visible 2022 neighbors. Focal, validation, and test 2022 values are excluded from fitted preprocessing "
                "statistics and input construction.",
                "",
                "## Selected neural architectures",
                "",
                *architecture_lines,
                "",
                f"Completed {len(run['fits'])} evaluated neural fits across all seed/protocol combinations.",
                "",
                "## Win gate",
                "",
                f"Overall: **{'PASS' if gate.get('passed') else 'FAIL'}**",
                "",
                *gate_lines,
                "",
                "## Macro metrics",
                "",
                macro.groupby(["protocol", "model"])[["mae", "rmse", "r2", "weighted_mae"]]
                .agg(["mean", "std"])
                .to_markdown(),
                "",
                f"![Macro MAE]({figure.name})",
                "",
                "Baselines are 2010 carry-forward, city mean, inverse-distance visible-neighbor average, ridge, and "
                "non-spatial MLP. Every baseline is retrained for the same fold and seed using fit rows only. No baseline "
                "prediction is supplied to the gat model.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(output)
