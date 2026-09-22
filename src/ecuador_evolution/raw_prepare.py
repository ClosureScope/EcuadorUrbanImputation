from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .archive import filtered_zip_csv_to_parquet
from .config import Settings
from .crosswalk import build_crosswalk, transfer_components
from .identifiers import add_2010_record_ids, add_geographic_ids, normalize_sector_id
from .indicators import Indicator, read_indicator_dictionary, safe_ratio

GEOGRAPHY_COLUMNS = ["I01", "I02", "I03", "I04", "I05"]
TABLE_COLUMNS = {
    (2010, "population"): [*GEOGRAPHY_COLUMNS, "I09", "I10", "P01", "P02", "P03", "P19"],
    (2010, "household"): [*GEOGRAPHY_COLUMNS, "I09", "I10", "H01", "H15"],
    (2010, "dwelling"): [*GEOGRAPHY_COLUMNS, "I09", "I10", "VTV", "VCO", "V07", "V09", "V10"],
    (2022, "population"): [*GEOGRAPHY_COLUMNS, "ID_VIV", "ID_HOG", "P02", "P03", "P19"],
    (2022, "household"): [*GEOGRAPHY_COLUMNS, "ID_VIV", "ID_HOG", "H09", "H1303", "HAC"],
    (2022, "dwelling"): [*GEOGRAPHY_COLUMNS, "ID_VIV", "V01", "V0201", "V10", "V11", "V12"],
}


def _atomic_parquet(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _raw_options(settings: Settings) -> dict[str, object]:
    return settings.section("prepare")


def raw_input_paths(settings: Settings) -> dict[str, object]:
    raw = settings.data_path("raw")
    options = _raw_options(settings)
    return {
        "csv_2022": raw / str(options.get("raw_2022_csv", "BDD_CPV2022_SECT_CSV.zip")),
        "csv_2010": [
            raw / str(name)
            for name in options.get(
                "raw_2010_csv",
                ["CPV2010S_CSV_Azuay.zip", "CPV2010S_CSV_Guayas.zip", "CPV2010S_CSV_Pichincha.zip"],
            )
        ],
        "geometry_2022": raw / str(options.get("raw_2022_geometry", "CapaSectores.zip")),
        "geometry_2010": raw / str(options.get("raw_2010_geometry", "GEODATABASE2014-AJUSTADA-V3.zip")),
    }


def official_raw_inputs_available(settings: Settings) -> bool:
    paths = raw_input_paths(settings)
    required = [paths["csv_2022"], paths["geometry_2022"], paths["geometry_2010"], *paths["csv_2010"]]
    return all(Path(path).exists() for path in required)


def _normalize_records(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    if year == 2010:
        return add_2010_record_ids(frame)
    result = add_geographic_ids(frame)
    result["dwelling_id"] = result["ID_VIV"].astype("string")
    if "ID_HOG" in result:
        result["household_id"] = result["ID_HOG"].astype("string")
    return result


def _ingest_tables(settings: Settings) -> tuple[dict[tuple[int, str], pd.DataFrame], dict[str, object]]:
    paths = raw_input_paths(settings)
    interim = settings.data_path("interim")
    chunksize = int(_raw_options(settings).get("chunksize", 200_000))
    city_codes = settings.values.get("city_codes", [])
    summaries: dict[str, object] = {}
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    for year, source_paths, separator, encoding in (
        (2010, paths["csv_2010"], ",", "latin-1"),
        (2022, paths["csv_2022"], ";", "utf-8-sig"),
    ):
        for table in ("population", "household", "dwelling"):
            output = interim / f"records_{year}_{table}.parquet"
            summary = filtered_zip_csv_to_parquet(
                source_paths,
                output,
                city_codes=city_codes,
                chunksize=chunksize,
                member_contains={"population": "Poblacion", "household": "Hogar", "dwelling": "Vivienda"}[table],
                encoding=encoding,
                separator=separator,
                columns=TABLE_COLUMNS[(year, table)],
            )
            if not output.exists():
                raise ValueError(f"No {year} {table} records matched the configured cities")
            frame = _normalize_records(pd.read_parquet(output), year)
            _atomic_parquet(frame, output)
            frames[(year, table)] = frame
            summaries[f"{year}_{table}"] = summary
    return frames, summaries


def _sum_by_sector(sector: pd.Series, values: pd.Series, name: str) -> pd.DataFrame:
    return pd.DataFrame({"source_sector_id": sector, name: values.astype(float)}).groupby(
        "source_sector_id", as_index=False
    )[name].sum()


def _indicator_frame(
    sector: pd.Series,
    indicator: str,
    numerator: pd.Series,
    denominator: pd.Series | None = None,
) -> pd.DataFrame:
    numerator_totals = _sum_by_sector(sector, numerator, "numerator")
    if denominator is None:
        numerator_totals["denominator"] = np.nan
        numerator_totals["value"] = numerator_totals["numerator"]
    else:
        denominator_totals = _sum_by_sector(sector, denominator, "denominator")
        numerator_totals = numerator_totals.merge(denominator_totals, on="source_sector_id", how="outer")
        numerator_totals[["numerator", "denominator"]] = numerator_totals[["numerator", "denominator"]].fillna(0.0)
        numerator_totals["value"] = safe_ratio(numerator_totals["numerator"], numerator_totals["denominator"])
    numerator_totals["indicator"] = indicator
    return numerator_totals


def _person_components(frame: pd.DataFrame, year: int) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    sector = frame["sector_id"]
    sex_field = "P01" if year == 2010 else "P02"
    sex = pd.to_numeric(frame[sex_field], errors="coerce")
    age = pd.to_numeric(frame["P03"], errors="coerce")
    literacy = pd.to_numeric(frame["P19"], errors="coerce")
    age_known = age.between(0, 120)
    literacy_known = age.ge(15) & literacy.isin([1, 2])
    components = [
        _indicator_frame(sector, "population", pd.Series(1.0, index=frame.index)),
        _indicator_frame(sector, "female_share", sex.eq(2), sex.isin([1, 2])),
        _indicator_frame(sector, "age_0_14_share", age.between(0, 14), age_known),
        _indicator_frame(sector, "age_65_plus_share", age.between(65, 120), age_known),
        _indicator_frame(sector, "literacy_15_plus", literacy_known & literacy.eq(1), literacy_known),
    ]
    if year == 2010:
        relationship = pd.to_numeric(frame["P02"], errors="coerce")
        eligible_member = relationship.between(1, 9)
        members = (
            frame.loc[eligible_member, ["sector_id", "household_id"]]
            .groupby(["sector_id", "household_id"], as_index=False)
            .size()
            .rename(columns={"size": "members"})
        )
    else:
        members = pd.DataFrame(columns=["sector_id", "household_id", "members"])
    return components, members


def _household_components(
    frame: pd.DataFrame,
    year: int,
    members_2010: pd.DataFrame,
) -> list[pd.DataFrame]:
    households = frame.drop_duplicates("household_id").copy()
    sector = households["sector_id"]
    tenure_field = "H15" if year == 2010 else "H09"
    tenure = pd.to_numeric(households[tenure_field], errors="coerce")
    known_tenure = tenure.isin(range(1, 8) if year == 2010 else range(1, 7))
    renting = tenure.isin([6, 7] if year == 2010 else [4])
    components = [
        _indicator_frame(sector, "household_count", pd.Series(1.0, index=households.index)),
        _indicator_frame(sector, "homeownership_share", tenure.isin([1, 2, 3]), known_tenure),
        _indicator_frame(sector, "renting_share", renting, known_tenure),
    ]
    if year == 2010:
        member_counts = households[["sector_id", "household_id", "H01"]].merge(
            members_2010, on=["sector_id", "household_id"], how="left", validate="one_to_one"
        )
        member_counts["members"] = member_counts["members"].fillna(0.0)
        valid_members = member_counts["members"].gt(0)
        components.append(
            _indicator_frame(
                member_counts["sector_id"],
                "average_household_size",
                member_counts["members"].where(valid_members, 0.0),
                valid_members,
            )
        )
        rooms = pd.to_numeric(member_counts["H01"], errors="coerce")
        valid = rooms.gt(0) & member_counts["members"].gt(0)
        components.append(
            _indicator_frame(
                member_counts["sector_id"],
                "overcrowding_share",
                valid & member_counts["members"].div(rooms).gt(3),
                valid,
            )
        )
    else:
        size = pd.to_numeric(households["H1303"], errors="coerce")
        valid_size = size.between(1, 7196)
        components.append(
            _indicator_frame(sector, "average_household_size", size.where(valid_size, 0.0), valid_size)
        )
        crowding = pd.to_numeric(households["HAC"], errors="coerce")
        components.append(
            _indicator_frame(sector, "overcrowding_share", crowding.eq(1), crowding.isin([1, 2]))
        )
    return components


def _dwelling_components(
    households: pd.DataFrame,
    dwellings: pd.DataFrame,
    year: int,
) -> list[pd.DataFrame]:
    linked_ids = households[["dwelling_id"]].dropna().drop_duplicates()
    eligible = dwellings.drop_duplicates("dwelling_id").merge(
        linked_ids, on="dwelling_id", how="inner", validate="one_to_one"
    )
    if year == 2010:
        private = pd.to_numeric(eligible["VTV"], errors="coerce").isin(range(1, 9))
        occupied = pd.to_numeric(eligible["VCO"], errors="coerce").eq(1)
        fields = {
            "public_water_share": ("V07", [1], range(1, 6)),
            "public_electricity_share": ("V10", [1], range(1, 6)),
            "public_sewer_share": ("V09", [1], range(1, 7)),
        }
    else:
        private = pd.to_numeric(eligible["V01"], errors="coerce").isin(range(1, 9))
        occupied = pd.to_numeric(eligible["V0201"], errors="coerce").eq(1)
        fields = {
            "public_water_share": ("V10", [1, 2], range(1, 6)),
            "public_electricity_share": ("V12", [1], range(1, 3)),
            "public_sewer_share": ("V11", [1], range(1, 8)),
        }
    eligible = eligible.loc[private & occupied].copy()
    return [
        _indicator_frame(
            eligible["sector_id"],
            indicator,
            pd.to_numeric(eligible[field], errors="coerce").isin(numerator_categories),
            pd.to_numeric(eligible[field], errors="coerce").isin(denominator_categories),
        )
        for indicator, (field, numerator_categories, denominator_categories) in fields.items()
    ]


def aggregate_components(
    population: pd.DataFrame,
    households: pd.DataFrame,
    dwellings: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    person, members = _person_components(population, year)
    frames = person + _household_components(households, year, members) + _dwelling_components(households, dwellings, year)
    result = pd.concat(frames, ignore_index=True)
    result["year"] = year
    return result.sort_values(["source_sector_id", "indicator"]).reset_index(drop=True)


def _vsi_path(archive: Path, inner: str) -> str:
    return f"/vsizip/{archive.resolve()}/{inner}"


def _load_geometries(settings: Settings):
    import geopandas as gpd

    options = _raw_options(settings)
    paths = raw_input_paths(settings)
    target_inner = str(options.get("target_geometry_dataset", "sectores_anonimizados.gpkg"))
    target_layer = str(options.get("target_geometry_layer", "sectores_anonimizados_indicadores_censo_2022"))
    source_inner = str(
        options.get(
            "source_geometry_dataset",
            "Geodatabase 2014 Ajustada Versión3/GEODATABASE2014.gdb",
        )
    )
    source_layers = list(options.get("source_geometry_layers", ["GEO_SEC2014", "GEO_SECDIS2014"]))
    target = gpd.read_file(_vsi_path(Path(paths["geometry_2022"]), target_inner), layer=target_layer)
    if "sec_anm" not in target:
        raise KeyError("2022 geometry is missing sec_anm")
    target["target_2022_sector_id"] = target["sec_anm"].map(normalize_sector_id)
    target["city_id"] = target["target_2022_sector_id"].str[:4]
    selected = {str(item).zfill(4) for item in settings.values.get("city_codes", [])}
    target = target.loc[target["city_id"].isin(selected), ["city_id", "target_2022_sector_id", "geometry"]].copy()
    target = target.to_crs(str(settings.values.get("target_crs", "EPSG:32717")))
    target.geometry = target.geometry.make_valid()
    if target["target_2022_sector_id"].duplicated().any():
        target = target.dissolve(by="target_2022_sector_id", as_index=False, aggfunc={"city_id": "first"})

    source_frames = []
    for layer in source_layers:
        frame = gpd.read_file(_vsi_path(Path(paths["geometry_2010"]), source_inner), layer=str(layer))
        id_field = "DPA_SECTOR" if "DPA_SECTOR" in frame else "DPA_SECDIS"
        frame["source_sector_id"] = frame[id_field].map(normalize_sector_id)
        frame["city_id"] = frame["source_sector_id"].str[:4]
        source_frames.append(frame.loc[frame["city_id"].isin(selected), ["city_id", "source_sector_id", "geometry"]])
    source_raw = gpd.GeoDataFrame(pd.concat(source_frames, ignore_index=True), crs=source_frames[0].crs)
    duplicate_ids = int(source_raw.loc[source_raw["source_sector_id"].duplicated(keep=False), "source_sector_id"].nunique())
    source = source_raw.to_crs(str(settings.values.get("target_crs", "EPSG:32717")))
    source.geometry = source.geometry.make_valid()
    source = source.dissolve(by="source_sector_id", as_index=False, aggfunc={"city_id": "first"})
    return source, target, duplicate_ids


def _build_city_crosswalk(source, target, settings: Settings) -> pd.DataFrame:
    options = _raw_options(settings)
    pieces = []
    for city_id in settings.values.get("city_codes", []):
        city = str(city_id).zfill(4)
        piece = build_crosswalk(
            source.loc[source["city_id"] == city],
            target.loc[target["city_id"] == city],
            target_crs=str(settings.values.get("target_crs", "EPSG:32717")),
            source_match_minimum=float(options.get("source_match_minimum", 0.95)),
            high_coverage=float(options.get("target_high_coverage", 0.98)),
            medium_coverage=float(options.get("target_medium_coverage", 0.90)),
            high_max_contributors=int(options.get("high_max_contributors", 3)),
            medium_max_contributors=int(options.get("medium_max_contributors", 8)),
        )
        piece["city_id"] = city
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _transfer_2010(
    components: pd.DataFrame,
    crosswalk: pd.DataFrame,
    dictionary: dict[str, Indicator],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    conservation: list[dict[str, object]] = []
    accepted = crosswalk.loc[crosswalk["match_quality"].isin(["high", "medium"])]
    for (city_id, indicator), values in components.assign(city_id=lambda x: x["source_sector_id"].str[:4]).groupby(
        ["city_id", "indicator"], sort=False
    ):
        item = dictionary[indicator]
        component_names = ("numerator",) if item.unit == "count" else ("numerator", "denominator")
        city_crosswalk = crosswalk.loc[crosswalk["city_id"] == city_id]
        transferred = transfer_components(values, city_crosswalk, components=component_names)
        if item.unit == "count":
            transferred["value"] = transferred["transferred_numerator"]
        transferred["city_id"] = city_id
        transferred["indicator"] = indicator
        transferred["numerator"] = transferred["transferred_numerator"]
        transferred["denominator"] = (
            transferred["transferred_denominator"] if "transferred_denominator" in transferred else np.nan
        )
        rows.append(transferred)

        allocations = accepted.loc[accepted["city_id"] == city_id].merge(
            values[["source_sector_id", *component_names]], on="source_sector_id", how="inner", validate="many_to_one"
        )
        for component in component_names:
            expected = float((allocations[component] * allocations["weight"]).sum())
            observed = float(transferred[f"transferred_{component}"].sum())
            conservation.append(
                {
                    "city_id": city_id,
                    "indicator": indicator,
                    "component": component,
                    "source_total": expected,
                    "transferred_total": observed,
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(conservation)


def _complete_long_table(
    transferred_2010: pd.DataFrame,
    direct_2022: pd.DataFrame,
    crosswalk: pd.DataFrame,
    target,
    indicators: list[Indicator],
    settings: Settings,
) -> pd.DataFrame:
    city_names = {str(key): value for key, value in settings.values.get("city_names", {}).items()}
    base = target[["city_id", "target_2022_sector_id"]].drop_duplicates().copy()
    base["city_name"] = base["city_id"].map(city_names).fillna(base["city_id"])
    grid = base.merge(pd.DataFrame({"year": [2010, 2022]}), how="cross").merge(
        pd.DataFrame({"indicator": [item.name for item in indicators]}), how="cross"
    )
    quality_columns = ["target_2022_sector_id", "target_coverage", "contributor_count", "match_quality"]
    target_quality = crosswalk[quality_columns].drop_duplicates("target_2022_sector_id")

    data_2010 = transferred_2010[
        ["city_id", "target_2022_sector_id", "indicator", "value", "numerator", "denominator"]
    ].copy()
    data_2010["year"] = 2010
    data_2010 = data_2010.merge(target_quality, on="target_2022_sector_id", how="left")
    data_2010["source_sector_id"] = pd.NA
    data_2010["provenance"] = "2010 geometry-only area transfer"

    data_2022 = direct_2022.rename(columns={"source_sector_id": "target_2022_sector_id"}).copy()
    data_2022["city_id"] = data_2022["target_2022_sector_id"].str[:4]
    data_2022["year"] = 2022
    data_2022["source_sector_id"] = data_2022["target_2022_sector_id"]
    data_2022["target_coverage"] = 1.0
    data_2022["contributor_count"] = 1
    data_2022["match_quality"] = "direct"
    data_2022["provenance"] = "2022 direct sector aggregation"
    measured = pd.concat([data_2010, data_2022], ignore_index=True)
    long = grid.merge(
        measured,
        on=["city_id", "target_2022_sector_id", "year", "indicator"],
        how="left",
        validate="one_to_one",
    )
    long = long.merge(
        target_quality.rename(
            columns={
                "target_coverage": "crosswalk_target_coverage",
                "contributor_count": "crosswalk_contributor_count",
                "match_quality": "crosswalk_match_quality",
            }
        ),
        on="target_2022_sector_id",
        how="left",
        validate="many_to_one",
    )
    year_2010 = long["year"].eq(2010)
    long.loc[year_2010, "target_coverage"] = long.loc[year_2010, "target_coverage"].fillna(
        long.loc[year_2010, "crosswalk_target_coverage"]
    )
    long.loc[year_2010, "contributor_count"] = long.loc[year_2010, "contributor_count"].fillna(
        long.loc[year_2010, "crosswalk_contributor_count"]
    )
    long.loc[year_2010, "match_quality"] = long.loc[year_2010, "match_quality"].fillna(
        long.loc[year_2010, "crosswalk_match_quality"]
    )
    long = long.drop(
        columns=["crosswalk_target_coverage", "crosswalk_contributor_count", "crosswalk_match_quality"]
    )
    metadata = pd.DataFrame(
        [{"indicator": item.name, "unit": item.unit, "comparability": item.comparability} for item in indicators]
    )
    long = long.merge(metadata, on="indicator", how="left", validate="many_to_one")
    missing_2010 = long["year"].eq(2010) & long["provenance"].isna()
    missing_2022 = long["year"].eq(2022) & long["provenance"].isna()
    long.loc[missing_2010, "provenance"] = "2010 geometry transfer unavailable or excluded"
    long.loc[missing_2022, "provenance"] = "2022 source record unavailable"
    long.loc[missing_2010 & long["match_quality"].isna(), "match_quality"] = "unmatched"
    long.loc[missing_2010 & long["target_coverage"].isna(), "target_coverage"] = 0.0
    long.loc[missing_2010 & long["contributor_count"].isna(), "contributor_count"] = 0
    long.loc[missing_2022, "match_quality"] = "direct-missing"
    long.loc[missing_2022, "target_coverage"] = 1.0
    long.loc[missing_2022, "contributor_count"] = 1
    return long


def prepare_raw_data(settings: Settings):
    """Build the complete harmonized contract directly from official raw archives."""
    tables, row_counts = _ingest_tables(settings)
    dictionary_items = read_indicator_dictionary(settings.project_path("indicator_dictionary"))
    dictionary = {item.name: item for item in dictionary_items}
    components_2010 = aggregate_components(
        tables[(2010, "population")], tables[(2010, "household")], tables[(2010, "dwelling")], 2010
    )
    components_2022 = aggregate_components(
        tables[(2022, "population")], tables[(2022, "household")], tables[(2022, "dwelling")], 2022
    )
    _atomic_parquet(components_2010, settings.data_path("interim") / "sector_components_2010.parquet")
    _atomic_parquet(components_2022, settings.data_path("interim") / "sector_components_2022.parquet")

    source_geometry, target_geometry, duplicate_ids = _load_geometries(settings)
    crosswalk = _build_city_crosswalk(source_geometry, target_geometry, settings)
    transferred, conservation = _transfer_2010(components_2010, crosswalk, dictionary)
    long = _complete_long_table(transferred, components_2022, crosswalk, target_geometry, dictionary_items, settings)

    source_data_ids = set(components_2010["source_sector_id"])
    source_geometry_ids = set(source_geometry["source_sector_id"])
    target_data_ids = set(components_2022["source_sector_id"])
    target_geometry_ids = set(target_geometry["target_2022_sector_id"])
    target_quality = crosswalk[["target_2022_sector_id", "match_quality"]].drop_duplicates()
    matched_targets = set(target_quality["target_2022_sector_id"])
    excluded = int(target_quality["match_quality"].eq("exclude").sum()) + len(target_geometry_ids - matched_targets)
    details = {
        "source_row_counts": row_counts,
        "unmatched_data_only_sector_ids": {
            "2010": sorted(source_data_ids - source_geometry_ids),
            "2022": sorted(target_data_ids - target_geometry_ids),
        },
        "excluded_target_count": excluded,
        "source_geometry_count": int(len(source_geometry)),
        "target_geometry_count": int(len(target_geometry)),
        "multipart_duplicate_sector_ids": duplicate_ids,
    }
    return long, crosswalk, target_geometry, conservation, details
