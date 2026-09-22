from __future__ import annotations

import json

import pandas as pd

from ecuador_evolution.cli import main


def test_prepare_validate_report_end_to_end(synthetic_project) -> None:
    config, data_root = synthetic_project
    assert main(["prepare", "--config", str(config)]) == 0
    assert main(["validate", "--config", str(config)]) == 0
    assert main(["report", "--config", str(config)]) == 0
    assert (data_root / "processed" / "model_matrix.parquet").exists()
    validation = json.loads((data_root / "artifacts" / "validation-summary.json").read_text())
    assert validation["passed"]
    assert (data_root / "artifacts" / "data-report.md").exists()


def test_prepare_from_nested_raw_archives(synthetic_raw_project) -> None:
    config, data_root = synthetic_raw_project
    assert main(["prepare", "--config", str(config)]) == 0
    processed = data_root / "processed"
    long = pd.read_parquet(processed / "sector_indicator_long.parquet")
    matrix = pd.read_parquet(processed / "model_matrix.parquet")
    summary = json.loads((processed / "preparation_summary.json").read_text())
    validation = json.loads((data_root / "artifacts" / "validation-summary.json").read_text())
    assert len(long) == 26
    assert long["indicator"].nunique() == 13
    assert set(long["year"]) == {2010, 2022}
    assert len(matrix) == 2
    assert summary["multipart_duplicate_sector_ids"] == 1
    assert summary["target_geometry_count"] == 1
    assert validation["passed"]
    assert (data_root / "interim" / "records_2010_population.parquet").exists()
    assert (data_root / "interim" / "sector_components_2022.parquet").exists()
