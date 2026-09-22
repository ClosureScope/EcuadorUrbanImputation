from __future__ import annotations

import shutil
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


@pytest.fixture
def synthetic_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "configs").mkdir()
    source_configs = Path(__file__).parents[1] / "configs"
    shutil.copy(source_configs / "indicators.toml", project / "configs" / "indicators.toml")
    data_root = tmp_path / "data"
    interim = data_root / "interim"
    interim.mkdir(parents=True)
    cities = {"0101": "Cuenca", "0901": "Guayaquil", "1701": "Quito"}
    indicators = [
        "population", "female_share", "age_0_14_share", "age_65_plus_share",
        "household_count", "average_household_size", "literacy_15_plus",
        "homeownership_share", "renting_share", "public_water_share",
        "public_electricity_share", "public_sewer_share",
    ]
    units = {name: ("count" if name in {"population", "household_count"} else "rate") for name in indicators}
    rows = []
    polygons = []
    crosswalk = []
    for city_index, (city_id, city_name) in enumerate(cities.items()):
        for sector in range(5):
            target_id = f"{city_id}{sector:08d}"
            polygons.append({"city_id": city_id, "target_2022_sector_id": target_id, "geometry": box(city_index * 10 + sector, 0, city_index * 10 + sector + 1, 1)})
            crosswalk.append({"source_sector_id": f"s-{target_id}", "target_2022_sector_id": target_id, "weight": 1.0, "target_coverage": 1.0, "contributor_count": 1, "effective_contributors": 1, "match_quality": "high"})
            for year in (2010, 2022):
                for indicator_index, indicator in enumerate(indicators):
                    base = 100 + sector + indicator_index
                    value = base + (12 if year == 2022 else 0) if units[indicator] == "count" else 0.4 + indicator_index * 0.01 + (0.02 if year == 2022 else 0)
                    denominator = 1.0 if units[indicator] == "count" else 100.0
                    numerator = value if units[indicator] == "count" else value * denominator
                    rows.append({"city_id": city_id, "city_name": city_name, "year": year, "source_sector_id": f"s-{target_id}" if year == 2010 else target_id, "target_2022_sector_id": target_id, "indicator": indicator, "value": value, "numerator": numerator, "denominator": denominator, "unit": units[indicator], "provenance": "synthetic-fixture", "target_coverage": 1.0, "contributor_count": 1, "match_quality": "high", "comparability": "high"})
    pd.DataFrame(rows).to_parquet(interim / "sector_indicator_long.parquet", index=False)
    pd.DataFrame(crosswalk).to_parquet(interim / "crosswalk.parquet", index=False)
    conservation = []
    long_frame = pd.DataFrame(rows)
    for (city_id, indicator), group in long_frame.loc[long_frame["year"] == 2010].groupby(["city_id", "indicator"]):
        total = float(group["numerator"].sum())
        conservation.append({"city_id": city_id, "indicator": indicator, "component": "numerator", "source_total": total, "transferred_total": total})
    pd.DataFrame(conservation).to_parquet(interim / "transfer_conservation.parquet", index=False)
    gpd.GeoDataFrame(polygons, crs="EPSG:32717").to_parquet(interim / "target_geometry.parquet", index=False)
    config = project / "configs" / "test.toml"
    config.write_text(f'''task = "shared"\nseed = 1605\ndevice = "cpu"\ndata_root = "{data_root}"\ncity_codes = ["0101", "0901", "1701"]\n[prepare]\nminimum_indicator_coverage = 0.90\nminimum_indicators = 8\n[paths]\nindicator_dictionary = "configs/indicators.toml"\ninterim = "interim"\nprocessed = "processed"\nartifacts = "artifacts"\n''')
    return config, data_root


def _write_nested_table(outer: zipfile.ZipFile, table: str, header: list[str], rows: list[list[object]]) -> None:
    payload = io.StringIO()
    payload.write(",".join(header) + "\n")
    for row in rows:
        payload.write(",".join(str(value) for value in row) + "\n")
    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w") as nested:
        nested.writestr(f"City_CSV_{table}.csv", payload.getvalue())
    outer.writestr(f"City_CSV_{table}.zip", nested_bytes.getvalue())


@pytest.fixture
def synthetic_raw_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    source_configs = Path(__file__).parents[1] / "configs"
    shutil.copy(source_configs / "indicators.toml", project / "configs" / "indicators.toml")
    data_root = tmp_path / "data"
    raw = data_root / "raw"
    raw.mkdir(parents=True)

    geography = ["01", "01", "01", "001", "001"]
    archive_2010 = raw / "city-2010.zip"
    with zipfile.ZipFile(archive_2010, "w") as outer:
        _write_nested_table(
            outer,
            "Poblacion",
            ["I01", "I02", "I03", "I04", "I05", "I09", "I10", "P01", "P02", "P03", "P19"],
            [[*geography, "001", 1, 1, 1, 30, 1], [*geography, "001", 1, 2, 2, 10, 1]],
        )
        _write_nested_table(
            outer,
            "Hogar",
            ["I01", "I02", "I03", "I04", "I05", "I09", "I10", "H01", "H15"],
            [[*geography, "001", 1, 1, 1]],
        )
        _write_nested_table(
            outer,
            "Vivienda",
            ["I01", "I02", "I03", "I04", "I05", "I09", "I10", "VTV", "VCO", "V07", "V09", "V10"],
            [[*geography, "001", 1, 1, 1, 1, 1, 1]],
        )

    archive_2022 = raw / "city-2022.zip"
    with zipfile.ZipFile(archive_2022, "w") as output:
        output.writestr(
            "CPV_2022_Poblacion_Sector.csv",
            ";".join(["I01", "I02", "I03", "I04", "I05", "ID_VIV", "ID_HOG", "P02", "P03", "P19"])
            + "\n"
            + ";".join([*geography, "v1", "h1", "1", "30", "1"])
            + "\n"
            + ";".join([*geography, "v1", "h1", "2", "10", "1"])
            + "\n",
        )
        output.writestr(
            "CPV_2022_Hogar_Sector.csv",
            ";".join(["I01", "I02", "I03", "I04", "I05", "ID_VIV", "ID_HOG", "H09", "H1303", "HAC"])
            + "\n"
            + ";".join([*geography, "v1", "h1", "1", "2", "2"])
            + "\n",
        )
        output.writestr(
            "CPV_2022_Vivienda_Sector.csv",
            ";".join(["I01", "I02", "I03", "I04", "I05", "ID_VIV", "V01", "V0201", "V10", "V11", "V12"])
            + "\n"
            + ";".join([*geography, "v1", "1", "1", "1", "1", "1"])
            + "\n",
        )

    source_gpkg = tmp_path / "source.gpkg"
    source_id = "010101001001"
    gpd.GeoDataFrame({"DPA_SECTOR": [source_id]}, geometry=[box(0, 0, 0.5, 1)], crs="EPSG:32717").to_file(
        source_gpkg, layer="GEO_SEC2014", driver="GPKG"
    )
    gpd.GeoDataFrame({"DPA_SECDIS": [source_id]}, geometry=[box(0.5, 0, 1, 1)], crs="EPSG:32717").to_file(
        source_gpkg, layer="GEO_SECDIS2014", driver="GPKG"
    )
    with zipfile.ZipFile(raw / "source-geometry.zip", "w") as output:
        output.write(source_gpkg, "source.gpkg")

    target_gpkg = tmp_path / "target.gpkg"
    gpd.GeoDataFrame({"sec_anm": [source_id]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32717").to_file(
        target_gpkg, layer="targets", driver="GPKG"
    )
    with zipfile.ZipFile(raw / "target-geometry.zip", "w") as output:
        output.write(target_gpkg, "target.gpkg")

    config = project / "configs" / "raw-test.toml"
    config.write_text(
        f'''task = "shared"
seed = 1605
device = "cpu"
data_root = "{data_root}"
target_crs = "EPSG:32717"
city_codes = ["0101"]
city_names = {{ "0101" = "Cuenca" }}
[prepare]
chunksize = 1
raw_2022_csv = "city-2022.zip"
raw_2010_csv = ["city-2010.zip"]
raw_2022_geometry = "target-geometry.zip"
raw_2010_geometry = "source-geometry.zip"
target_geometry_dataset = "target.gpkg"
target_geometry_layer = "targets"
source_geometry_dataset = "source.gpkg"
source_geometry_layers = ["GEO_SEC2014", "GEO_SECDIS2014"]
minimum_indicator_coverage = 0.90
minimum_indicators = 8
conservation_tolerance = 0.000001
[paths]
indicator_dictionary = "configs/indicators.toml"
raw = "raw"
interim = "interim"
processed = "processed"
artifacts = "artifacts"
'''
    )
    return config, data_root
