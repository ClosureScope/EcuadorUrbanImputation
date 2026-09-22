"""Export the harmonized 2010/2022 modeling panel as a CSV with stable sector IDs."""
from __future__ import annotations

import pandas as pd

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RENAME = {
    "target_2022_sector_id": "tract_id",
    "female_share": "female_ratio",
    "age_0_14_share": "age_0_14_ratio",
    "age_65_plus_share": "age_65plus_ratio",
    "literacy_15_plus": "literacy_ratio",
    "homeownership_share": "homeownership_ratio",
    "renting_share": "rent_household_ratio",
    "public_water_share": "public_water_ratio",
    "public_electricity_share": "public_electricity_ratio",
    "public_sewer_share": "public_sewer_ratio",
    "overcrowding_share": "overcrowding_ratio",
    "population": "pop_total",
    "average_household_size": "avg_household_size",
    # household_count already compliant
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=ROOT / "data" / "processed" / "model_matrix.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "final" / "model_2010_2022_13indicators.csv")
    args = parser.parse_args()
    m = pd.read_parquet(args.input)
    m = m.rename(columns=RENAME)
    m["tract_id"] = m["tract_id"].astype(str).str.zfill(12)
    m["country_iso3"] = "ECU"
    m["country_name"] = "Ecuador"
    m["tract_like_unit"] = "census sector (sector censal)"

    ordered = [
        "tract_id", "country_iso3", "country_name", "city_id", "city_name",
        "tract_like_unit", "year",
        "pop_total", "female_ratio", "age_0_14_ratio", "age_65plus_ratio",
        "household_count", "avg_household_size", "literacy_ratio",
        "homeownership_ratio", "rent_household_ratio", "public_water_ratio",
        "public_electricity_ratio", "public_sewer_ratio", "overcrowding_ratio",
    ]
    missing = set(ordered) - set(m.columns)
    assert not missing, f"missing: {missing}"
    m = m[ordered].sort_values(["city_id", "tract_id", "year"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.output, index=False, encoding="utf-8")

    print("rows:", len(m))
    print("years:", sorted(m.year.unique()), "| cities:", sorted(m.city_id.unique()))
    print("13 indicators + year; per year rows:", m.year.value_counts().to_dict())


if __name__ == "__main__":
    main()
