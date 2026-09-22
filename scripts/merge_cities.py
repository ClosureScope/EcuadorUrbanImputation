"""Merge three processed city layers into the 2022 census-sector dataset.

Indicators use consistent ratio, count, area and density field names.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CITY_META = {  # canton code -> (city_id, city_name)
    "1701": ("1701", "Quito"),
    "0101": ("0101", "Cuenca"),
    "0901": ("0901", "Guayaquil"),
}

# Processing fields -> public dataset fields.
RENAME = {
    "area_sq_km": "area_km2",
    "pop_density": "pop_density_per_km2",
    "female_share": "female_ratio",
    "age_0_14_share": "age_0_14_ratio",
    "age_65plus_share": "age_65plus_ratio",
    "education_high_share": "education_high_ratio",
    "literacy_rate": "literacy_ratio",
    "employment_rate": "employment_ratio",
    "unemployment_rate": "unemployment_ratio",
    "labor_force_participation_rate": "labor_force_participation_ratio",
    "rent_share": "rent_household_ratio",
    "homeownership_rate": "homeownership_ratio",
    "cellphone_service_share": "cellphone_service_ratio",
    "fixed_internet_share": "fixed_internet_ratio",
    "computer_access_share": "computer_access_ratio",
    "basic_services_share": "basic_services_ratio",
    "housing_deficit_share": "housing_deficit_ratio",
    # pop_total / household_count / avg_household_size 已合规，保留
}

# final 19 indicators (renamed), in report order
INDICATORS = [
    "pop_total", "pop_density_per_km2", "female_ratio", "age_0_14_ratio",
    "age_65plus_ratio", "education_high_ratio", "literacy_ratio",
    "employment_ratio", "unemployment_ratio", "labor_force_participation_ratio",
    "household_count", "avg_household_size", "homeownership_ratio",
    "rent_household_ratio", "cellphone_service_ratio", "fixed_internet_ratio",
    "computer_access_ratio", "basic_services_ratio", "housing_deficit_ratio",
]
AREA = ["area_km2"]


def load_city(path: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    g["tract_id"] = g["tract_id"].astype(str).str.zfill(12)
    g = g.rename(columns=RENAME)
    canton = g["tract_id"].str.slice(0, 4)
    g["city_id"] = canton.map(lambda c: CITY_META[c][0])
    g["city_name"] = canton.map(lambda c: CITY_META[c][1])
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="Three city GeoPackages; defaults to processed Quito, Cuenca, Guayaquil")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "final")
    args = parser.parse_args()
    inputs = args.inputs or [ROOT / "data" / "interim" / "census" / f"ecuador_{city}" / "stage1" / "3_joined_data" / "city_tract_indicators.gpkg" for city in ("quito", "cuenca", "guayaquil")]
    if len(inputs) != 3:
        parser.error("provide exactly three city GeoPackages")
    parts = [load_city(p) for p in inputs]
    g = pd.concat(parts, ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=parts[0].crs)

    g["source_tract_id"] = g["tract_id"]
    g["country_iso3"] = "ECU"
    g["country_name"] = "Ecuador"
    g["tract_like_unit"] = "census sector (sector censal)"
    g["boundary_year"] = 2022

    base = [
        "tract_id", "source_tract_id", "country_iso3", "country_name",
        "city_id", "city_name", "tract_like_unit", "boundary_year",
    ]
    ordered = base + AREA + INDICATORS + ["geometry"]
    missing = set(ordered) - set(g.columns)
    assert not missing, f"missing cols: {missing}"
    g = g[ordered]

    assert g["tract_id"].is_unique, "duplicate tract_id"
    assert int(g.geometry.is_empty.sum()) == 0
    assert int((~g.geometry.is_valid).sum()) == 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    g.to_file(args.output_dir / "city_tract_indicators.gpkg", driver="GPKG")
    pd.DataFrame(g.drop(columns="geometry")).to_csv(args.output_dir / "city_tract_indicators.csv", index=False, encoding="utf-8")

    print("rows:", len(g), "| crs:", g.crs)
    print("per-city:", g.city_id.value_counts().to_dict())
    print("cols:", list(g.drop(columns="geometry").columns))


if __name__ == "__main__":
    main()
