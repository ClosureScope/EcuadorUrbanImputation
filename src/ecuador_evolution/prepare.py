from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .config import Settings
from .indicators import read_indicator_dictionary

REQUIRED_LONG_COLUMNS = [
    "city_id",
    "city_name",
    "year",
    "source_sector_id",
    "target_2022_sector_id",
    "indicator",
    "value",
    "numerator",
    "denominator",
    "unit",
    "provenance",
    "target_coverage",
    "contributor_count",
    "match_quality",
    "comparability",
]


def _atomic_parquet(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_prepared_outputs(
    long_table: pd.DataFrame,
    crosswalk: pd.DataFrame,
    target_geometry,
    settings: Settings,
    transfer_conservation: pd.DataFrame | None = None,
    preparation_details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write canonical outputs after year-specific recoding has produced rows."""
    missing = set(REQUIRED_LONG_COLUMNS) - set(long_table.columns)
    if missing:
        raise ValueError(f"Long table missing required fields: {sorted(missing)}")
    output = settings.data_path("processed")
    output.mkdir(parents=True, exist_ok=True)
    long_table = long_table.loc[:, REQUIRED_LONG_COLUMNS].sort_values(
        ["city_id", "target_2022_sector_id", "year", "indicator"]
    )
    _atomic_parquet(long_table, output / "sector_indicator_long.parquet")
    key = ["city_id", "city_name", "target_2022_sector_id", "year"]
    wide = long_table.pivot(index=key, columns="indicator", values="value").reset_index()
    wide.columns.name = None
    indicator_order = [item.name for item in read_indicator_dictionary(settings.project_path("indicator_dictionary"))]
    wide = wide.reindex(columns=[*key, *indicator_order])
    _atomic_parquet(wide, output / "model_matrix.parquet")
    _atomic_parquet(crosswalk, output / "crosswalk.parquet")
    _atomic_parquet(target_geometry, output / "target_geometry.parquet")
    if transfer_conservation is None:
        raise ValueError("A source-versus-transferred totals table is required for conservation validation")
    conservation_columns = {"city_id", "indicator", "component", "source_total", "transferred_total"}
    missing_conservation = conservation_columns - set(transfer_conservation.columns)
    if missing_conservation:
        raise ValueError(f"Transfer conservation table missing: {sorted(missing_conservation)}")
    conservation = transfer_conservation.copy()
    conservation["relative_error"] = (
        (conservation["transferred_total"] - conservation["source_total"]).abs()
        / conservation["source_total"].abs().clip(lower=1e-12)
    )
    _atomic_parquet(conservation, output / "transfer_conservation.parquet")
    dictionary = pd.DataFrame(item.__dict__ for item in read_indicator_dictionary(settings.project_path("indicator_dictionary")))
    for column in ("source_2010", "source_2022", "flags"):
        dictionary[column] = dictionary[column].map(json.dumps)
    _atomic_parquet(dictionary, output / "indicator_dictionary.parquet")
    coverage = (
        long_table.assign(available=long_table["value"].notna())
        .groupby(["city_id", "year", "indicator"])["available"]
        .mean()
        .reset_index(name="coverage")
    )
    exclusions = crosswalk["match_quality"].value_counts(dropna=False).to_dict() if "match_quality" in crosswalk else {}
    summary = {
        "rows": len(long_table),
        "sectors": int(long_table["target_2022_sector_id"].nunique()),
        "indicators": int(long_table["indicator"].nunique()),
        "cities": sorted(long_table["city_id"].unique().tolist()),
        "years": sorted(int(year) for year in long_table["year"].unique()),
        "crosswalk_quality": {str(key): int(value) for key, value in exclusions.items()},
        "coverage": coverage.to_dict(orient="records"),
    }
    if preparation_details:
        summary.update(preparation_details)
    _atomic_json(summary, output / "preparation_summary.json")
    return summary


def prepare(settings: Settings) -> dict[str, object]:
    """Prepare raw archives when available, retaining the staged fixture fallback."""
    from .raw_prepare import official_raw_inputs_available, prepare_raw_data

    if official_raw_inputs_available(settings):
        long, crosswalk, geometry, conservation, details = prepare_raw_data(settings)
        summary = finalize_prepared_outputs(
            long,
            crosswalk,
            geometry,
            settings,
            conservation,
            preparation_details=details,
        )
    else:
        summary = _prepare_staged(settings)

    from .reporting import write_data_report
    from .validation import validate

    validate(settings)
    write_data_report(settings)
    return summary


def _prepare_staged(settings: Settings) -> dict[str, object]:
    """Finalize canonical staged inputs used by lightweight synthetic tests."""
    interim = settings.data_path("interim")
    required = {
        "long": interim / "sector_indicator_long.parquet",
        "crosswalk": interim / "crosswalk.parquet",
        "geometry": interim / "target_geometry.parquet",
        "conservation": interim / "transfer_conservation.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Preparation requires staged year-specific recoding outputs. Missing: "
            + ", ".join(missing)
            + ". Use the documented archive helpers and official dictionaries to stage these files."
        )
    import geopandas as gpd

    return finalize_prepared_outputs(
        pd.read_parquet(required["long"]),
        pd.read_parquet(required["crosswalk"]),
        gpd.read_parquet(required["geometry"]),
        settings,
        pd.read_parquet(required["conservation"]),
    )
