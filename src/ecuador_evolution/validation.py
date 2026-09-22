from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings


@dataclass
class ValidationReport:
    checks: list[dict[str, object]] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(bool(item["passed"]) for item in self.checks)

    def require_passed(self) -> None:
        if not self.passed:
            failures = [f"{item['name']}: {item['detail']}" for item in self.checks if not item["passed"]]
            raise ValueError("Validation failed:\n- " + "\n- ".join(failures))


def _crosswalk_conservation(crosswalk: pd.DataFrame) -> tuple[bool, float | str]:
    """Check source-weight conservation before target-quality exclusions."""
    sums = crosswalk.groupby("source_sector_id")["weight"].sum()
    maximum_deviation: float | str = float((sums - 1).abs().max()) if len(sums) else "n/a"
    return bool(len(sums) and np.allclose(sums, 1.0, atol=1e-6)), maximum_deviation


def validate(settings: Settings, *, strict: bool = True) -> ValidationReport:
    processed = settings.data_path("processed")
    report = ValidationReport()
    long_path = processed / "sector_indicator_long.parquet"
    geometry_path = processed / "target_geometry.parquet"
    crosswalk_path = processed / "crosswalk.parquet"
    conservation_path = processed / "transfer_conservation.parquet"
    report.check("prepared files", all(path.exists() for path in (long_path, geometry_path, crosswalk_path, conservation_path)), str(processed))
    if not long_path.exists():
        if strict:
            report.require_passed()
        return report
    long = pd.read_parquet(long_path)
    configured_cities = settings.values.get("city_codes")
    expected_cities = (
        {str(item).zfill(4) for item in configured_cities}
        if configured_cities
        else set(long["city_id"].astype(str))
    )
    report.check("all cities", set(long["city_id"].astype(str)) == expected_cities, f"found {sorted(long['city_id'].astype(str).unique())}")
    report.check("both years", set(long["year"].astype(int)) == {2010, 2022}, f"found {sorted(long['year'].unique())}")
    expected_rows = long["target_2022_sector_id"].nunique() * 2 * long["indicator"].nunique()
    unique_keys = not long[["target_2022_sector_id", "year", "indicator"]].duplicated().any()
    report.check(
        "complete target-year-indicator grid",
        unique_keys and len(long) == expected_rows,
        f"rows={len(long)}, expected={expected_rows}, unique_keys={unique_keys}",
    )
    acceptable = long["comparability"].isin(["high", "medium"])
    coverage = (
        long.loc[acceptable]
        .assign(available=lambda frame: frame["value"].notna())
        .groupby(["city_id", "year", "indicator"])["available"]
        .mean()
    )
    threshold = float(settings.section("prepare").get("minimum_indicator_coverage", 0.90))
    counts = coverage.ge(threshold).groupby(["city_id", "year"]).sum()
    minimum = int(settings.section("prepare").get("minimum_indicators", 8))
    report.check("indicator coverage", len(counts) == len(expected_cities) * 2 and bool((counts >= minimum).all()), f"passing indicators by city-year: {counts.to_dict()}")
    if geometry_path.exists():
        import geopandas as gpd

        geometry = gpd.read_parquet(geometry_path)
        id_column = "target_2022_sector_id"
        valid = geometry.geometry.notna() & ~geometry.geometry.is_empty & geometry.geometry.is_valid
        report.check("valid geometry", bool(valid.all()), f"{int((~valid).sum())} invalid/empty")
        report.check("unique target IDs", id_column in geometry and not geometry[id_column].duplicated().any(), id_column)
    if crosswalk_path.exists():
        crosswalk = pd.read_parquet(crosswalk_path)
        conserved, maximum_deviation = _crosswalk_conservation(crosswalk)
        report.check("crosswalk conservation", conserved, f"max deviation {maximum_deviation}")
        report.check("interpolation exclusions reported", bool((crosswalk["match_quality"] == "exclude").any() or len(crosswalk)), crosswalk["match_quality"].value_counts().to_dict().__str__())
    if conservation_path.exists():
        conservation = pd.read_parquet(conservation_path)
        tolerance = float(settings.section("prepare").get("conservation_tolerance", 0.02))
        required = {"city_id", "indicator", "component", "source_total", "transferred_total", "relative_error"}
        valid_schema = required.issubset(conservation.columns)
        errors = conservation["relative_error"] if valid_schema else pd.Series([np.inf])
        complete_cities = set(conservation["city_id"].astype(str)) == expected_cities if "city_id" in conservation else False
        report.check("transferred totals", valid_schema and complete_cities and bool((errors <= tolerance).all()), f"tolerance={tolerance}, max relative error={float(errors.max())}")
    summary_path = processed / "preparation_summary.json"
    report.check("coverage summary", summary_path.exists(), str(summary_path))
    output = settings.data_path("artifacts") / "validation-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps({"passed": report.passed, "checks": report.checks}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if strict:
        report.require_passed()
    return report
