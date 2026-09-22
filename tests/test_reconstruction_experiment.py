from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecuador_evolution.cli import main
from ecuador_evolution.experiment import ReconstructionData, build_masked_inputs, build_residual_inputs


def test_jointly_masked_values_never_enter_inputs() -> None:
    n = 4
    data = ReconstructionData(
        frame=None, geometry=None, indicators=["x"], values_2010=np.arange(n, dtype=float)[:, None],
        values_2022=np.array([[10.0], [20.0], [30.0], [40.0]]), focal_raw=np.arange(n, dtype=float)[:, None],
        neighbor_index=np.array([[1, 2], [0, 2], [0, 1], [1, 2]]), neighbor_mask=np.ones((n, 2), dtype=bool),
        neighbor_distances=np.ones((n, 2), dtype=np.float32),
    )
    fit = np.array([True, True, False, False]); masked = ~fit
    _, before, scaler = build_masked_inputs(data, fit, masked)
    changed = ReconstructionData(**{**data.__dict__, "values_2022": np.array([[10.0], [20.0], [-9999.0], [9999.0]])})
    _, after, changed_scaler = build_masked_inputs(changed, fit, masked)
    assert np.array_equal(before, after)
    assert np.array_equal(scaler.mean, changed_scaler.mean)
    assert np.array_equal(scaler.scale, changed_scaler.scale)


def test_residual_inputs_exclude_jointly_masked_values() -> None:
    n = 4
    data = ReconstructionData(
        frame=None,
        geometry=None,
        indicators=["count", "rate"],
        values_2010=np.array([[10.0, 0.2], [20.0, 0.3], [30.0, 0.4], [40.0, 0.5]]),
        values_2022=np.array([[12.0, 0.25], [22.0, 0.35], [32.0, 0.45], [42.0, 0.55]]),
        focal_raw=np.arange(n, dtype=float)[:, None],
        neighbor_index=np.array([[1, 2], [0, 2], [0, 1], [1, 2]]),
        neighbor_mask=np.ones((n, 2), dtype=bool),
        neighbor_distances=np.ones((n, 2), dtype=np.float32),
        units=("count", "rate"),
        geometry_features=np.zeros((n, 4), dtype=np.float32),
        neighbor_relative=np.zeros((n, 2, 3), dtype=np.float32),
    )
    fit = np.array([True, True, False, False]); masked = ~fit
    focal_before, neighbors_before, targets_before, base_before, transformer_before = build_residual_inputs(
        data, fit, masked
    )
    changed_values = data.values_2022.copy(); changed_values[masked] = [[9_999, 0.999], [8_888, 0.001]]
    changed = ReconstructionData(**{**data.__dict__, "values_2022": changed_values})
    focal_after, neighbors_after, _, base_after, transformer_after = build_residual_inputs(changed, fit, masked)
    assert np.array_equal(focal_before, focal_after)
    assert np.array_equal(neighbors_before, neighbors_after)
    assert np.array_equal(base_before, base_after)
    assert np.array_equal(transformer_before.center, transformer_after.center)
    assert np.array_equal(transformer_before.scale, transformer_after.scale)
    assert np.isfinite(targets_before[fit]).all()


def test_reconstruction_cpu_end_to_end(synthetic_project) -> None:
    base_config, data_root = synthetic_project
    project = base_config.parents[1]
    config = project / "configs" / "reconstruction.toml"
    config.write_text(f'''task = "masked-reconstruction"\nseed = 1605\ndevice = "cpu"\ndata_root = "{data_root}"\ntarget_crs = "EPSG:32717"\n[model]\nmax_neighbors = 2\nhidden_widths = [8]\ndropouts = [0.0]\nlearning_rates = [0.001]\nweight_decay = 0.0\nbatch_size = 32\nmax_epochs = 2\npatience = 1\nseeds = [1605]\n[evaluation]\nrandom_mask_fractions = [0.20]\nspatial_blocks = 3\nvalidation_fraction = 0.20\n[paths]\nindicator_dictionary = "configs/indicators.toml"\ninterim = "interim"\nprocessed = "processed"\nartifacts = "artifacts/reconstruction-test"\n''')
    assert main(["prepare", "--config", str(config)]) == 0
    assert main(["train", "--config", str(config)]) == 0
    assert main(["evaluate", "--config", str(config)]) == 0
    assert main(["report", "--config", str(config)]) == 0
    assert (data_root / "artifacts" / "reconstruction-test" / "report.md").exists()


def test_reconstruction_mae_comparison_cpu_end_to_end(synthetic_project) -> None:
    base_config, data_root = synthetic_project
    project = base_config.parents[1]
    config = project / "configs" / "reconstruction-mae.toml"
    config.write_text(f'''task = "masked-reconstruction"
seed = 1605
device = "cpu"
data_root = "{data_root}"
target_crs = "EPSG:32717"
[model]
architectures = ["spatial", "mae"]
max_neighbors = 2
hidden_widths = [8]
dropouts = [0.0]
learning_rates = [0.001]
weight_decay = 0.0
batch_size = 32
max_epochs = 1
patience = 1
seeds = [1605]
[mae]
widths = [8]
dropouts = [0.0]
learning_rates = [0.001]
heads = 2
encoder_layers = 1
decoder_layers = 1
ff_ratio = 2
weight_decay = 0.0
batch_size = 32
accumulation_steps = 2
max_epochs = 1
patience = 1
[evaluation]
random_mask_fractions = [0.20]
spatial_blocks = 3
validation_fraction = 0.20
[paths]
indicator_dictionary = "configs/indicators.toml"
interim = "interim"
processed = "processed"
artifacts = "artifacts/reconstruction-mae-test"
''')
    assert main(["prepare", "--config", str(config)]) == 0
    assert main(["train", "--config", str(config)]) == 0
    first_run = json.loads((data_root / "artifacts" / "reconstruction-mae-test" / "run.json").read_text())
    assert main(["train", "--config", str(config)]) == 0
    assert main(["evaluate", "--config", str(config)]) == 0
    assert main(["report", "--config", str(config)]) == 0
    artifacts = data_root / "artifacts" / "reconstruction-mae-test"
    run = json.loads((artifacts / "run.json").read_text())
    tuning = pd.read_csv(artifacts / "tuning.csv")
    predictions = pd.read_parquet(artifacts / "predictions.parquet")
    moran = pd.read_csv(artifacts / "residual-morans-i.csv")
    assert run["architectures"] == ["spatial", "mae"]
    assert len(run["fits"]) == 14
    assert run["fits"] == first_run["fits"]
    assert len(json.loads((artifacts / "tuning-progress.json").read_text())) == 2
    assert len(json.loads((artifacts / "fit-progress.json").read_text())) == 14
    assert set(tuning["model"]) == {"spatial", "mae"}
    assert set(predictions["model"]) >= {"spatial", "mae", "2010-carry-forward"}
    assert set(moran["model"]) == {"spatial", "mae"}
    assert len(list(artifacts.glob("spatial-*.pt"))) == 7
    assert len(list(artifacts.glob("mae-*.pt"))) == 7
    assert (artifacts / "metrics.csv").exists()
    assert (artifacts / "evaluation.json").exists()
    assert (artifacts / "macro-mae.png").exists()
    assert (artifacts / "report.md").exists()


def test_gat_cpu_end_to_end_and_win_gate_schema(synthetic_project) -> None:
    base_config, data_root = synthetic_project
    project = base_config.parents[1]
    config = project / "configs" / "reconstruction-residual.toml"
    config.write_text(f'''task = "masked-reconstruction"
seed = 1605
device = "cpu"
data_root = "{data_root}"
target_crs = "EPSG:32717"
[model]
architectures = ["gat"]
max_neighbors = 2
seeds = [1605]
[gat]
widths = [8]
dropouts = [0.0]
learning_rates = [0.001]
context_dropout = 0.0
indicator_dropout = 0.0
native_mae_weight = 0.01
weight_decay = 0.0
batch_size = 32
max_epochs = 1
patience = 1
gradient_clip = 1.0
warmup_fraction = 0.0
[baselines]
non_spatial_mlp_width = 8
non_spatial_mlp_max_iter = 2
[evaluation]
random_mask_fractions = [0.20]
spatial_blocks = 3
validation_fraction = 0.20
[paths]
indicator_dictionary = "configs/indicators.toml"
interim = "interim"
processed = "processed"
artifacts = "artifacts/reconstruction-residual-test"
''')
    assert main(["prepare", "--config", str(config)]) == 0
    assert main(["train", "--config", str(config)]) == 0
    assert main(["evaluate", "--config", str(config)]) == 0
    assert main(["report", "--config", str(config)]) == 0
    artifacts = data_root / "artifacts" / "reconstruction-residual-test"
    predictions = pd.read_parquet(artifacts / "predictions.parquet")
    evaluation = json.loads((artifacts / "evaluation.json").read_text())
    assert set(predictions["seed"]) == {1605}
    assert set(predictions["model"]) >= {"gat", "non-spatial-mlp", "ridge"}
    assert evaluation["win_gate"]["primary_model"] == "gat"
    assert set(evaluation["win_gate"]["protocols"]) == {"random-mask", "block-mask", "held-city"}


def test_single_and_multi_worker_predictions_are_equivalent(synthetic_project) -> None:
    base_config, data_root = synthetic_project
    project = base_config.parents[1]

    def write_config(name: str, workers: int) -> Path:
        config = project / "configs" / f"{name}.toml"
        config.write_text(f'''task = "masked-reconstruction"
seed = 1605
device = "cpu"
data_root = "{data_root}"
target_crs = "EPSG:32717"
[model]
max_neighbors = 2
hidden_widths = [8]
dropouts = [0.0]
learning_rates = [0.001]
weight_decay = 0.0
batch_size = 32
max_epochs = 1
patience = 1
seeds = [1605]
[baselines]
non_spatial_mlp_width = 8
non_spatial_mlp_max_iter = 2
[evaluation]
random_mask_fractions = [0.20]
spatial_blocks = 3
validation_fraction = 0.20
[parallel]
backend = "process"
gpu_workers = {workers}
max_gpu_workers = 2
cpu_workers = 2
calibration = false
gpu_worker_cpu_threads = 1
baseline_worker_cpu_threads = 1
[paths]
indicator_dictionary = "configs/indicators.toml"
interim = "interim"
processed = "processed"
artifacts = "artifacts/{name}"
''')
        return config

    single = write_config("reconstruction-single-worker", 1)
    parallel = write_config("reconstruction-multi-worker", 2)
    assert main(["prepare", "--config", str(single)]) == 0
    assert main(["train", "--config", str(single)]) == 0
    assert main(["train", "--config", str(parallel)]) == 0
    keys = ["model", "protocol", "fold", "seed", "city_id", "sector_id", "indicator"]
    left = pd.read_parquet(data_root / "artifacts/reconstruction-single-worker/predictions.parquet").sort_values(keys)
    right = pd.read_parquet(data_root / "artifacts/reconstruction-multi-worker/predictions.parquet").sort_values(keys)
    assert left[keys].reset_index(drop=True).equals(right[keys].reset_index(drop=True))
    assert np.array_equal(left["truth"].to_numpy(), right["truth"].to_numpy())
    assert np.allclose(left["prediction"].to_numpy(), right["prediction"].to_numpy(), rtol=1e-6, atol=1e-7)
    assert set(left["model"]) == set(right["model"])
