from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ablation
from ecuador_evolution.config import load_settings
from ecuador_evolution import experiment

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_and_processing_paths_are_repo_local():
    from ecuador_evolution.cli import parser
    import process_census

    args = parser().parse_args(["prepare"])
    settings = load_settings(ROOT / args.config)
    assert settings.data_root == ROOT / "data"
    assert process_census.ROOT == ROOT
    assert process_census.CPV_SECTOR_DIR == settings.data_path("raw")
    assert settings.project_path("indicator_dictionary").is_file()
    from ecuador_evolution.download import read_manifest
    sources = read_manifest(settings.project_path("manifest"))
    assert len(sources) == 6
    assert all(source.sha256 and len(source.sha256) == 64 for source in sources)
    assert all((settings.data_path("raw") / source.filename).resolve().is_relative_to(ROOT / "data" / "raw") for source in sources)


def test_smoke_config_is_small_and_keeps_full_run_artifacts_separate():
    configs = ROOT / "configs"
    small = load_settings(configs / "smoke.toml")
    full = load_settings(configs / "reconstruction.toml")
    assert small.device == "cpu"
    assert small.section("parallel")["backend"] == "serial"
    assert small.section("model")["seeds"] == [1605]
    assert small.section("gat")["max_epochs"] == 2
    assert small.data_path("processed") == full.data_path("processed")
    assert small.data_path("artifacts") != full.data_path("artifacts")


def test_ablation_exports_raw_metrics_and_invalidates_stale_cache(tmp_path, monkeypatch):
    Data = dataclasses.make_dataclass("Data", ["frame", "indicators", "values_2022", "neighbor_mask"])
    truth = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = Data(list(range(4)), ["population"], truth, np.ones((4, 1), dtype=bool))
    settings = SimpleNamespace(values={"task": "masked-reconstruction"}, device="cpu")
    monkeypatch.setattr("ecuador_evolution.config.load_settings", lambda *a, **kw: settings)
    monkeypatch.setattr(experiment, "_load", lambda settings: data)
    monkeypatch.setattr(experiment, "_split_masks", lambda *a: (np.array([1, 1, 0, 0], dtype=bool), np.zeros(4, dtype=bool), np.array([0, 0, 1, 1], dtype=bool)))
    calls = []

    def fit(architecture, run_data, *args):
        calls.append(architecture)
        return None, truth + (0.5 if run_data.neighbor_mask.any() else 1.0), None

    monkeypatch.setattr(experiment, "_fit_architecture", fit)
    monkeypatch.setattr(ablation, "FOLDS", {"random-mask": ["20%"]})
    monkeypatch.setattr(ablation, "SEEDS", [1605])
    monkeypatch.setattr("sys.argv", ["ablation.py", "--output-dir", str(tmp_path)])
    runs = tmp_path / "ablation_runs"
    runs.mkdir()
    # An old cache from the broken exporter must not bypass recomputation.
    (runs / "run_random-mask_20pct_seed1605.json").write_text('{"ablated": {"MAE_z": 0.1}}')
    ablation.main()
    assert len(calls) == 2
    with (tmp_path / "ablation_metrics.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["raw_macro_MAE"]) == pytest.approx(1.0)
    assert float(row["raw_macro_RMSE"]) == pytest.approx(1.0)
    assert float(row["MAE_z"]) == pytest.approx(1 / np.std(truth), abs=1e-4)
    ablation.main()
    assert len(calls) == 2
    data.values_2022 = truth + 1
    ablation.main()
    assert len(calls) == 4


def test_city_merge_and_panel_export_from_other_working_directory(tmp_path, monkeypatch):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    import merge_cities
    import export_panel

    monkeypatch.chdir(tmp_path)
    reverse = {value: key for key, value in merge_cities.RENAME.items()}
    inputs = []
    for index, code in enumerate(("1701", "0101", "0901")):
        row = {reverse.get(name, name): 0.5 for name in merge_cities.INDICATORS}
        row.update(tract_id=code + "00000001", area_sq_km=1.0)
        path = tmp_path / f"{code}.gpkg"
        gpd.GeoDataFrame([row], geometry=[box(index, 0, index + 1, 1)], crs="EPSG:32717").to_file(path)
        inputs.append(str(path))
    output = tmp_path / "exports"
    monkeypatch.setattr("sys.argv", ["merge_cities.py", *inputs, "--output-dir", str(output)])
    merge_cities.main()
    merged = pd.read_csv(output / "city_tract_indicators.csv", dtype={"tract_id": str, "city_id": str})
    assert set(merged.city_id) == {"1701", "0101", "0901"}
    assert merged.tract_id.str.len().eq(12).all()
    assert gpd.read_file(output / "city_tract_indicators.gpkg").crs.to_epsg() == 32717

    row = {name: 0.5 for name in export_panel.RENAME}
    row.update(target_2022_sector_id="010100000001", city_id="0101", city_name="Cuenca", year=2022, household_count=10)
    source = tmp_path / "model_matrix.parquet"
    pd.DataFrame([row]).to_parquet(source)
    panel = output / "panel.csv"
    monkeypatch.setattr("sys.argv", ["export_panel.py", str(source), "--output", str(panel)])
    export_panel.main()
    exported = pd.read_csv(panel, dtype={"tract_id": str})
    assert exported.tract_id.tolist() == ["010100000001"]
    assert exported.year.tolist() == [2022]
