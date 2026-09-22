from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import pyogrio


# Raw inputs and generated city datasets are local to this repository.
ROOT = Path(__file__).resolve().parents[1]
CPV_SECTOR_DIR = ROOT / "data" / "raw"
CPV_SECTOR_CSV_ZIP = CPV_SECTOR_DIR / "BDD_CPV2022_SECT_CSV.zip"
ANON_SECTOR_ZIP = CPV_SECTOR_DIR / "original_boundary.zip"
ZIP_PATH = CPV_SECTOR_DIR / "17_PICHINCHA.zip"  # Optional legacy boundary fallback.
PROCESSED_DIR = ROOT / "data" / "interim" / "census" / "ecuador_quito" / "stage1"
EXPORT_DIR = PROCESSED_DIR / "4_metadata" / "fallback_exports"

ZIP_GPKG_URI = f"zip://{ZIP_PATH.resolve()}!17_pichincha.gpkg"
QUITO_CANTON_PREFIX = "1701"
QUITO_PROVINCE_CODE = "17"
QUITO_CANTON_CODE = "01"

CPV_SECTOR_CSV_URL = "https://www.ecuadorencifras.gob.ec/documentos/web-inec/bd-censo/sector/BDD_CPV2022_SECT_CSV.zip"
ANON_SECTOR_URL = "https://www.ecuadorencifras.gob.ec/documentos/web-inec/capa/CapaSectores.zip"
CPV_DATA_PAGE = "https://www.censoecuador.gob.ec/data-censo-ecuador/"
ANDA_METADATA_URL = "https://anda.inec.gob.ec/anda5/index.php/catalog/1085"
REDATAM_URL = "https://redatam.inec.gob.ec/binecu/RpWebEngine.exe/Portal?BASE=CPV2022"
GEOPORTAL_URL = "https://www.ecuadorencifras.gob.ec/documentos/web-inec/Geografia_Estadistica/Micrositio_geoportal/descargas.html"
PICHINCHA_BOUNDARY_URL = "https://www.ecuadorencifras.gob.ec/documentos/web-inec/Geografia_Estadistica/Documentos/17_PICHINCHA.zip"
QUITO_AREA_CRS = "EPSG:32717"
INEC_PUBLIC_USE_LICENSE = "Free access public-use data; © 2022 INEC; cite INEC CPV 2022 v1.6."
INEC_REQUIRED_CITATION = (
    "Instituto Nacional de Estadística y Censos del Ecuador, VIII Censo de Población y VII "
    "de Vivienda - 2022, version 1.6 public-use dataset."
)
OFFICIAL_SOURCE_ACCESS_DATE = "2026-07-10"

STANDARD_INDICATORS = [
    ("pop_total", "Total population", "persons"),
    ("pop_density", "Population density", "persons per sq km"),
    ("age_0_14_share", "Population aged 0-14 share", "share"),
    ("age_65plus_share", "Population aged 65+ share", "share"),
    ("female_share", "Female population share", "share"),
    ("household_count", "Household count", "households"),
    ("avg_household_size", "Average household size", "persons per household"),
    ("unemployment_rate", "Unemployment rate", "share"),
    ("education_high_share", "Higher education population share", "share"),
    ("median_income", "Median income", "local currency"),
    ("poverty_rate", "Poverty rate", "share"),
    ("rent_share", "Renting household share", "share"),
    ("homeownership_rate", "Homeownership rate", "share"),
    ("median_rent", "Median rent", "local currency"),
    ("house_price", "House price", "local currency"),
]

SUPPLEMENTAL_INDICATORS = [
    ("labor_force_participation_rate", "Labor force participation rate", "share"),
    ("employment_rate", "Employment-to-working-age-population rate", "share"),
    ("literacy_rate", "Literacy rate among population aged 15+", "share"),
    ("fixed_internet_share", "Fixed internet service share", "share"),
    ("computer_access_share", "Computer access share", "share"),
    ("cellphone_service_share", "Cellphone service share", "share"),
    ("basic_services_share", "Housing units with all basic services share", "share"),
    ("housing_deficit_share", "Housing deficit share", "share"),
]
INVENTORY_INDICATORS = tuple(
    name for name, _, _ in [*STANDARD_INDICATORS, *SUPPLEMENTAL_INDICATORS]
)

XRAY_TARGET_INDICATORS = (
    "pop_total",
    "household_count",
    "age_0_14_share",
    "age_65plus_share",
    "female_share",
    "unemployment_rate",
    "education_high_share",
    "rent_share",
    "homeownership_rate",
    "labor_force_participation_rate",
    "employment_rate",
    "literacy_rate",
    "fixed_internet_share",
    "computer_access_share",
    "cellphone_service_share",
    "basic_services_share",
    "housing_deficit_share",
    "avg_household_size",
)
XRAY_MINIMUM_COVERAGE = 0.90
MISSING_ECONOMIC_PRICE_INDICATORS = {
    "median_income",
    "poverty_rate",
    "median_rent",
    "house_price",
}
BOUNDARY_SOURCE_COLUMNS = [
    "sec_anm",
    "anio",
    "fuente",
    "parroquia",
    "nom_par",
    "canton",
    "nom_can",
    "provincia",
    "nom_pro",
    "fcode",
]
BOUNDARY_ATTRIBUTE_COLUMNS = [
    "city",
    "country",
    "boundary_unit",
    "source_year",
    "source_agency",
    *BOUNDARY_SOURCE_COLUMNS,
    "area_sq_km",
]
INVENTORY_COLUMNS = [
    "country",
    "city",
    "tract_like_unit",
    "indicator_name",
    "indicator_category",
    "source_agency",
    "source_table",
    "source_url",
    "download_url",
    "source_year",
    "spatial_unit",
    "file_format",
    "join_id",
    "status",
    "notes",
]
INDICATOR_CATEGORIES = {
    "pop_total": "population",
    "pop_density": "population",
    "age_0_14_share": "population",
    "age_65plus_share": "population",
    "female_share": "population",
    "household_count": "household",
    "avg_household_size": "household",
    "unemployment_rate": "employment",
    "labor_force_participation_rate": "employment",
    "employment_rate": "employment",
    "education_high_share": "education",
    "literacy_rate": "education",
    "median_income": "income and poverty",
    "poverty_rate": "income and poverty",
    "rent_share": "housing",
    "homeownership_rate": "housing",
    "median_rent": "housing price",
    "house_price": "housing price",
    "fixed_internet_share": "digital access",
    "computer_access_share": "digital access",
    "cellphone_service_share": "digital access",
    "basic_services_share": "housing services",
    "housing_deficit_share": "housing quality",
}
PERSON_INDICATORS = {
    "pop_total",
    "age_0_14_share",
    "age_65plus_share",
    "female_share",
    "unemployment_rate",
    "education_high_share",
    "labor_force_participation_rate",
    "employment_rate",
    "literacy_rate",
}
HOUSEHOLD_INDICATORS = {
    "household_count",
    "avg_household_size",
    "rent_share",
    "homeownership_rate",
    "fixed_internet_share",
    "computer_access_share",
    "cellphone_service_share",
}
HOUSING_INDICATORS = {
    "basic_services_share",
    "housing_deficit_share",
}

DERIVED_PARROQUIA_INDICATORS = {
    "pop_total",
    "pop_density",
    "household_count",
    "avg_household_size",
    "education_high_share",
    "unemployment_rate",
    "labor_force_participation_rate",
    "employment_rate",
    "literacy_rate",
    "fixed_internet_share",
    "computer_access_share",
    "cellphone_service_share",
    "basic_services_share",
}

DERIVED_PARROQUIA_NOTE = (
    "Derived from CPV 2022 parroquia-level REDATAM exports and repeated across sectors in the same parroquia."
)
REQUIRED_CPV_MEMBER_KINDS = {"person", "household", "housing"}
BASIC_SERVICES_FORMULA = (
    "Occupied private dwellings (V0201R=1) with V10 in {1,2}, V11=1, V12=1, "
    "and V14 in {1,2}, divided by occupied private dwellings."
)

BASE_DICTIONARY_ROWS = [
    ("tract_id", "tract_id", "Stable 12-digit census sector identifier", "code", "sec_anm", "INEC statistical geography", "2022", "available"),
    ("sec_anm", "sec_anm", "Original anonymous census sector identifier", "code", "sec_anm", "INEC statistical geography", "2022", "available"),
    ("city", "city", "City name", "text", "constant", "student standardization", "2026", "available"),
    ("country", "country", "Country name", "text", "constant", "student standardization", "2026", "available"),
    ("boundary_unit", "boundary_unit", "Boundary unit used for analysis", "text", "constant", "student standardization", "2026", "available"),
    ("source_year", "source_year", "CPV product and indicator reference year", "year", "constant", "INEC CPV 2022 product", "2022", "available"),
    ("source_agency", "source_agency", "Publishing agency", "text", "constant", "INEC statistical geography", "2022", "available"),
    ("anio", "anio", "Original boundary cartographic year", "year", "anio", "INEC statistical geography", "2020/2022", "available"),
    ("fuente", "fuente", "Original boundary source label", "text", "fuente", "INEC statistical geography", "2020/2022", "available"),
    ("parroquia", "parroquia", "Parroquia code", "code", "parroquia", "INEC statistical geography", "2022", "available"),
    ("nom_par", "nom_par", "Parroquia name", "text", "nom_par", "INEC statistical geography", "2022", "available"),
    ("canton", "canton", "Canton code", "code", "canton", "INEC statistical geography", "2022", "available"),
    ("nom_can", "nom_can", "Canton name", "text", "nom_can", "INEC statistical geography", "2022", "available"),
    ("provincia", "provincia", "Province code", "code", "provincia", "INEC statistical geography", "2022", "available"),
    ("nom_pro", "nom_pro", "Province name", "text", "nom_pro", "INEC statistical geography", "2022", "available"),
    ("fcode", "fcode", "Official feature code", "code", "fcode", "INEC statistical geography", "2022", "available"),
    ("area_sq_km", "area_sq_km", "Sector polygon area", "sq km", "geometry area", "INEC statistical geography", "2020/2022", "available"),
]
BOUNDARY_YEAR_VARIABLES = {
    "tract_id",
    "sec_anm",
    "boundary_unit",
    "source_year",
    "source_agency",
}

SOURCE_COLUMNS = [
    "city",
    "country",
    "data_type",
    "indicator_name",
    "source_agency",
    "source_url",
    "download_url",
    "source_year",
    "access_date",
    "license",
    "notes",
]

DICTIONARY_COLUMNS = [
    "variable_name",
    "standard_name",
    "description",
    "unit",
    "original_name",
    "source",
    "year",
    "missing_rate",
    "quality_flag",
]

QUALITY_COLUMNS = [
    "variable_name",
    "boundary_unit",
    "indicator_unit",
    "join_method",
    "source_year",
    "quality_flag",
    "notes",
]


# Each city's 2022 processing artifacts are generated under data/interim/census.
PROCESSED_ASSIGNMENT_DIR = ROOT / "data" / "interim" / "census"

# Built-in registry of major Ecuadorian cantons (city selector -> canton code +
# display name). Any 4-digit canton code that is not listed here can still be
# selected directly; unknown *names* are rejected. Codes are the 4-digit
# province(2)+canton(2) INEC DPA prefix.
CITY_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("quito", "1701", "Quito"),
    ("guayaquil", "0901", "Guayaquil"),
    ("cuenca", "0101", "Cuenca"),
    ("ambato", "1801", "Ambato"),
    ("santo_domingo", "2301", "Santo Domingo"),
    ("riobamba", "0601", "Riobamba"),
    ("portoviejo", "1301", "Portoviejo"),
    ("loja", "1101", "Loja"),
    ("latacunga", "0501", "Latacunga"),
    ("duran", "0907", "Durán"),
    ("machala", "0701", "Machala"),
    ("manta", "1308", "Manta"),
    ("ibarra", "1001", "Ibarra"),
    ("esmeraldas", "0801", "Esmeraldas"),
    ("milagro", "0910", "Milagro"),
    ("quevedo", "1205", "Quevedo"),
    ("santa_elena", "2401", "Santa Elena"),
    ("babahoyo", "1201", "Babahoyo"),
)
NATIONAL_SELECTORS = frozenset({"all", "national", "ecuador", "todos"})


@dataclass(frozen=True)
class CityRegion:
    """A selectable Stage 1 processing region.

    ``canton_codes`` is the set of 4-digit INEC canton codes to keep; ``None``
    means national scope (no canton filter). ``slug`` keys the per-city output
    directory and ``display_name`` is the human label used in reports/logs.
    """

    slug: str
    display_name: str
    canton_codes: frozenset[str] | None

    @property
    def is_national(self) -> bool:
        return self.canton_codes is None

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_ASSIGNMENT_DIR / f"ecuador_{self.slug}" / "stage1"

    def keeps_canton(self, canton_code: str) -> bool:
        return self.canton_codes is None or canton_code in self.canton_codes

    def filter_description(self) -> str:
        if self.canton_codes is None:
            return "national (all cantons)"
        codes = ", ".join(f"`{code}`" for code in sorted(self.canton_codes))
        return f"canton {codes}"


def _normalize_city_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    return re.sub(r"[\s\-]+", "_", text)


def list_cities() -> list[tuple[str, str]]:
    """Return ``(canton_code, display_name)`` pairs for the built-in registry."""

    return [(code, display) for _, code, display in CITY_REGISTRY]


def resolve_city(selector: str | None) -> CityRegion:
    """Resolve a CLI selector (city name, 4-digit canton code, or "all") to a
    :class:`CityRegion`. Unknown 4-digit codes are accepted as ``canton_<code>``;
    unknown names raise ``ValueError``."""

    if selector is None:
        return QUITO_REGION
    raw = str(selector).strip()
    if not raw:
        return QUITO_REGION
    key = _normalize_city_key(raw)
    if key in NATIONAL_SELECTORS:
        return CityRegion(slug="national", display_name="Ecuador (national)", canton_codes=None)

    by_slug = {slug: (slug, code, display) for slug, code, display in CITY_REGISTRY}
    by_code = {code: (slug, code, display) for slug, code, display in CITY_REGISTRY}
    by_name = {_normalize_city_key(display): (slug, code, display) for slug, code, display in CITY_REGISTRY}

    digits = re.sub(r"\D", "", raw)
    if digits and len(digits) == 4:
        entry = by_code.get(digits)
        if entry is not None:
            slug, code, display = entry
            return CityRegion(slug=slug, display_name=display, canton_codes=frozenset({code}))
        return CityRegion(slug=f"canton_{digits}", display_name=f"Canton {digits}", canton_codes=frozenset({digits}))

    entry = by_slug.get(key) or by_name.get(key)
    if entry is not None:
        slug, code, display = entry
        return CityRegion(slug=slug, display_name=display, canton_codes=frozenset({code}))
    raise ValueError(
        f"Unknown city selector {selector!r}. Use a 4-digit canton code, 'all', "
        f"or one of: {', '.join(slug for slug, _, _ in CITY_REGISTRY)}."
    )


QUITO_REGION = CityRegion(slug="quito", display_name="Quito", canton_codes=frozenset({QUITO_CANTON_PREFIX}))


@dataclass(frozen=True)
class ExportResult:
    path: Path
    rows: int
    merged_columns: list[str]
    join_key: str = ""
    join_method: str = ""
    skipped_reason: str = ""
    source_kind: str = "manual_export"


@dataclass(frozen=True)
class XRayDataAudit:
    join_coverage: float
    target_table: pd.DataFrame
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise ValueError("Quito X-Ray data audit failed: " + "; ".join(self.errors))


def ensure_dirs() -> None:
    for path in [
        PROCESSED_DIR / "1_boundary",
        PROCESSED_DIR / "2_indicators" / "raw_indicator_tables",
        PROCESSED_DIR / "3_joined_data",
        PROCESSED_DIR / "4_metadata",
        PROCESSED_DIR / "5_maps",
        EXPORT_DIR,
        CPV_SECTOR_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "unnamed"


def unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        base = normalize_name(column)
        seen[base] = seen.get(base, 0) + 1
        result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return result


def clean_code(value: object, width: int) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\.0$", "", text)
    digits = re.sub(r"\D", "", text)
    return digits.zfill(width)[-width:] if digits else ""


def add_sector_tract_id(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["I01", "I02", "I03", "I04", "I05"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing CPV geography columns: {', '.join(missing)}")

    result = frame.copy()
    result["tract_id"] = (
        result["I01"].map(lambda value: clean_code(value, 2))
        + result["I02"].map(lambda value: clean_code(value, 2))
        + result["I03"].map(lambda value: clean_code(value, 2))
        + result["I04"].map(lambda value: clean_code(value, 3))
        + result["I05"].map(lambda value: clean_code(value, 3))
    )
    return result


def filter_cpv(frame: pd.DataFrame, region: CityRegion = QUITO_REGION) -> pd.DataFrame:
    frame = add_sector_tract_id(frame)
    if region.is_national:
        return frame.copy()
    canton4 = (
        frame["I01"].map(lambda value: clean_code(value, 2))
        + frame["I02"].map(lambda value: clean_code(value, 2))
    )
    return frame.loc[canton4.isin(region.canton_codes)].copy()


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def count_matches(frame: pd.DataFrame, column: str, values: set[int] | set[str]) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    series = frame[column].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    normalized_values = {str(value) for value in values}
    return series.isin(normalized_values).astype("int64")


def aggregate_count_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        numeric_columns = [column for column in frame.columns if column != "tract_id"]
        grouped = frame.groupby("tract_id", as_index=False)[numeric_columns].sum(min_count=1)
        if combined is None:
            combined = grouped
        else:
            combined = combined.merge(grouped, on="tract_id", how="outer", suffixes=("", "__new"))
            for column in numeric_columns:
                new_column = f"{column}__new"
                if new_column in combined.columns:
                    combined[column] = combined[column].fillna(0) + combined[new_column].fillna(0)
                    combined = combined.drop(columns=new_column)
    return combined if combined is not None else pd.DataFrame(columns=["tract_id"])


def person_sector_counts(frame: pd.DataFrame, region: CityRegion = QUITO_REGION) -> pd.DataFrame:
    quito = filter_cpv(frame, region)
    if quito.empty:
        return pd.DataFrame(columns=["tract_id"])

    age = numeric_series(quito, "P03")
    sex = numeric_series(quito, "P02")
    education = numeric_series(quito, "P17R")
    conda = numeric_series(quito, "CONDACT1")
    working_age = age.ge(15).fillna(False)
    education_known = working_age & education.between(1, 11, inclusive="both").fillna(False)
    literacy_known = working_age & count_matches(quito, "P19", {1, 2}).eq(1)
    counts = pd.DataFrame(
        {
            "tract_id": quito["tract_id"],
            "person_count": 1,
            "female_count": count_matches(quito, "P02", {2}),
            "sex_known_count": sex.isin([1, 2]).astype("int64"),
            "age_0_14_count": age.between(0, 14, inclusive="both").fillna(False).astype("int64"),
            "age_65plus_count": age.ge(65).fillna(False).astype("int64"),
            "age_known_count": age.notna().astype("int64"),
            "working_age_count": working_age.astype("int64"),
            "higher_education_count": (education_known & education.isin([8, 9, 10, 11])).astype("int64"),
            "education_known_count": education_known.astype("int64"),
            "literacy_yes_count": (working_age & count_matches(quito, "P19", {1}).eq(1)).astype("int64"),
            "literacy_known_count": literacy_known.astype("int64"),
            "employed_count": (working_age & conda.eq(2).fillna(False)).astype("int64"),
            "unemployed_count": (working_age & conda.eq(3).fillna(False)).astype("int64"),
        }
    )
    return aggregate_count_frames([counts])


def household_sector_counts(frame: pd.DataFrame, region: CityRegion = QUITO_REGION) -> pd.DataFrame:
    quito = filter_cpv(frame, region)
    if quito.empty:
        return pd.DataFrame(columns=["tract_id"])

    counts = pd.DataFrame(
        {
            "tract_id": quito["tract_id"],
            "household_count_raw": 1,
            "household_person_sum": numeric_series(quito, "H1303"),
            "owned_household_count": count_matches(quito, "H09", {1, 2, 3}),
            "rent_household_count": count_matches(quito, "H09", {4}),
            "tenure_known_count": count_matches(quito, "H09", {1, 2, 3, 4, 5, 6}),
            "cellphone_household_count": count_matches(quito, "H1002", {1}),
            "cellphone_known_count": count_matches(quito, "H1002", {1, 2}),
            "fixed_internet_household_count": count_matches(quito, "H1004", {1}),
            "fixed_internet_known_count": count_matches(quito, "H1004", {1, 2}),
            "computer_household_count": count_matches(quito, "H1005", {1}),
            "computer_known_count": count_matches(quito, "H1005", {1, 2}),
        }
    )
    return aggregate_count_frames([counts])


def housing_sector_counts(frame: pd.DataFrame, region: CityRegion = QUITO_REGION) -> pd.DataFrame:
    quito = filter_cpv(frame, region)
    if quito.empty:
        return pd.DataFrame(columns=["tract_id"])

    occupied = count_matches(quito, "V0201R", {1})
    basic_service = (
        occupied.eq(1)
        & count_matches(quito, "V10", {1, 2}).eq(1)
        & count_matches(quito, "V11", {1}).eq(1)
        & count_matches(quito, "V12", {1}).eq(1)
        & count_matches(quito, "V14", {1, 2}).eq(1)
    )
    deficit_known = count_matches(quito, "DEF_HAB", {1, 2, 3})
    deficit = count_matches(quito, "DEF_HAB", {2, 3})
    counts = pd.DataFrame(
        {
            "tract_id": quito["tract_id"],
            "occupied_housing_count": occupied,
            "basic_services_housing_count": basic_service.astype("int64"),
            "housing_deficit_count": deficit,
            "housing_deficit_known_count": deficit_known,
        }
    )
    return aggregate_count_frames([counts])


def finalize_sector_counts(
    person_counts: pd.DataFrame,
    household_counts: pd.DataFrame,
    housing_counts: pd.DataFrame,
) -> pd.DataFrame:
    result = person_counts.copy()
    for frame in [household_counts, housing_counts]:
        if not frame.empty:
            result = result.merge(frame, on="tract_id", how="outer")
    if result.empty:
        return pd.DataFrame(columns=["tract_id"])

    result = result.fillna(0)
    zero = pd.Series(0, index=result.index, dtype="float64")

    def counts(column: str) -> pd.Series:
        return result[column] if column in result.columns else zero

    result["pop_total"] = counts("person_count")
    result["female_share"] = divide_or_na(counts("female_count"), counts("sex_known_count"))
    result["age_0_14_share"] = divide_or_na(counts("age_0_14_count"), counts("age_known_count"))
    result["age_65plus_share"] = divide_or_na(counts("age_65plus_count"), counts("age_known_count"))
    result["education_high_share"] = divide_or_na(counts("higher_education_count"), counts("education_known_count"))
    result["literacy_rate"] = divide_or_na(counts("literacy_yes_count"), counts("literacy_known_count"))

    employed = counts("employed_count")
    unemployed = counts("unemployed_count")
    labor_force = employed + unemployed
    result["employment_rate"] = divide_or_na(employed, counts("working_age_count"))
    result["unemployment_rate"] = divide_or_na(unemployed, labor_force)
    result["labor_force_participation_rate"] = divide_or_na(labor_force, counts("working_age_count"))

    result["household_count"] = counts("household_count_raw")
    result["avg_household_size"] = divide_or_na(counts("household_person_sum"), result["household_count"])
    result["homeownership_rate"] = divide_or_na(counts("owned_household_count"), counts("tenure_known_count"))
    result["rent_share"] = divide_or_na(counts("rent_household_count"), counts("tenure_known_count"))
    result["cellphone_service_share"] = divide_or_na(counts("cellphone_household_count"), counts("cellphone_known_count"))
    result["fixed_internet_share"] = divide_or_na(counts("fixed_internet_household_count"), counts("fixed_internet_known_count"))
    result["computer_access_share"] = divide_or_na(counts("computer_household_count"), counts("computer_known_count"))
    result["basic_services_share"] = divide_or_na(counts("basic_services_housing_count"), counts("occupied_housing_count"))
    result["housing_deficit_share"] = divide_or_na(counts("housing_deficit_count"), counts("housing_deficit_known_count"))

    output_columns = [
        "tract_id",
        "pop_total",
        "female_share",
        "age_0_14_share",
        "age_65plus_share",
        "education_high_share",
        "literacy_rate",
        "employment_rate",
        "unemployment_rate",
        "labor_force_participation_rate",
        "household_count",
        "avg_household_size",
        "homeownership_rate",
        "rent_share",
        "cellphone_service_share",
        "fixed_internet_share",
        "computer_access_share",
        "basic_services_share",
        "housing_deficit_share",
    ]
    return result[output_columns].sort_values("tract_id").reset_index(drop=True)


def aggregate_cpv_sector_indicators(
    *,
    person: pd.DataFrame | None = None,
    household: pd.DataFrame | None = None,
    housing: pd.DataFrame | None = None,
    region: CityRegion = QUITO_REGION,
) -> pd.DataFrame:
    person_counts = person_sector_counts(person, region) if person is not None else pd.DataFrame(columns=["tract_id"])
    household_counts = household_sector_counts(household, region) if household is not None else pd.DataFrame(columns=["tract_id"])
    housing_counts = housing_sector_counts(housing, region) if housing is not None else pd.DataFrame(columns=["tract_id"])
    return finalize_sector_counts(person_counts, household_counts, housing_counts)


def zip_member_kind(name: str) -> str | None:
    normalized = normalize_name(Path(name).stem)
    if "poblacion" in normalized or "poblacin" in normalized or "poblaci" in normalized or "persona" in normalized:
        return "person"
    if "hogar" in normalized:
        return "household"
    if "vivienda" in normalized:
        return "housing"
    return None


def validate_official_source_pair(csv_path: Path, boundary_path: Path) -> bool:
    csv_exists = csv_path.exists()
    boundary_exists = boundary_path.exists()
    if csv_exists != boundary_exists:
        raise RuntimeError(
            "Official CPV 2022 sector CSV and anonymous boundary must be present together: "
            f"csv={csv_exists}, boundary={boundary_exists}"
        )
    return csv_exists


def read_cpv_csv_chunks(zip_path: Path, member: str) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            for chunk in pd.read_csv(
                handle,
                dtype=str,
                sep=";",
                chunksize=200_000,
                encoding="utf-8-sig",
                low_memory=False,
            ):
                yield chunk


def write_raw_indicator_tables(
    person_counts: pd.DataFrame,
    household_counts: pd.DataFrame,
    housing_counts: pd.DataFrame,
) -> None:
    """Persist the pre-cleaning sector-level count tables aggregated from CPV
    microdata. These are the raw indicator tables (per-sector counts before any
    share/rate computation or field standardization) that back cleaned_indicators.csv."""
    raw_dir = PROCESSED_DIR / "2_indicators" / "raw_indicator_tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "person_sector_counts.csv": person_counts,
        "household_sector_counts.csv": household_counts,
        "housing_sector_counts.csv": housing_counts,
    }
    for filename, table in tables.items():
        if table is not None and not table.empty:
            table.sort_values("tract_id").to_csv(raw_dir / filename, index=False)


def aggregate_cpv_sector_zip(
    zip_path: Path, region: CityRegion = QUITO_REGION
) -> tuple[pd.DataFrame, ExportResult] | None:
    if not zip_path.exists():
        return None

    person_frames: list[pd.DataFrame] = []
    household_frames: list[pd.DataFrame] = []
    housing_frames: list[pd.DataFrame] = []
    scanned_rows = 0
    scanned_members: list[str] = []

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith((".csv", ".txt"))]

    member_kinds = {kind for name in members if (kind := zip_member_kind(name)) is not None}
    missing_kinds = sorted(REQUIRED_CPV_MEMBER_KINDS - member_kinds)
    if missing_kinds:
        raise ValueError(f"CPV sector ZIP missing required table kinds: {', '.join(missing_kinds)}")

    for member in members:
        kind = zip_member_kind(member)
        if kind is None:
            continue
        scanned_members.append(member)
        for chunk in read_cpv_csv_chunks(zip_path, member):
            scanned_rows += len(chunk)
            if kind == "person":
                person_frames.append(person_sector_counts(chunk, region))
            elif kind == "household":
                household_frames.append(household_sector_counts(chunk, region))
            elif kind == "housing":
                housing_frames.append(housing_sector_counts(chunk, region))

    if not scanned_members:
        return (
            pd.DataFrame(columns=["tract_id"]),
            ExportResult(
                path=zip_path,
                rows=0,
                merged_columns=[],
                skipped_reason="no person, household, or housing CSV members found in sector zip",
                source_kind="cpv_sector",
            ),
        )

    person_counts = aggregate_count_frames(person_frames)
    household_counts = aggregate_count_frames(household_frames)
    housing_counts = aggregate_count_frames(housing_frames)
    write_raw_indicator_tables(person_counts, household_counts, housing_counts)
    sector_indicators = finalize_sector_counts(
        person_counts,
        household_counts,
        housing_counts,
    )
    merged_columns = [column for column in sector_indicators.columns if column != "tract_id"]
    return (
        sector_indicators,
        ExportResult(
            path=zip_path,
            rows=scanned_rows,
            merged_columns=merged_columns,
            join_key="tract_id",
            join_method="direct_join",
            source_kind="cpv_sector",
        ),
    )


def merge_cpv_sector_dataset(
    indicators: pd.DataFrame, region: CityRegion = QUITO_REGION
) -> tuple[pd.DataFrame, ExportResult | None]:
    aggregated = aggregate_cpv_sector_zip(CPV_SECTOR_CSV_ZIP, region)
    if aggregated is None:
        return indicators, None

    sector_indicators, result = aggregated
    if sector_indicators.empty:
        return indicators, result

    overlap = [column for column in sector_indicators.columns if column != "tract_id" and column in indicators.columns]
    indicators = indicators.drop(columns=overlap, errors="ignore")
    indicators = indicators.merge(sector_indicators, on="tract_id", how="left")
    return indicators, result


def local_utm_epsg(longitude: float, latitude: float) -> str:
    """Return the WGS84/UTM EPSG code for a lon/lat, so areas are measured in a
    projection local to each sector. Ecuador spans zones 15S-18N (mainland
    17S/18S, Galápagos 15S/16S)."""

    zone = int((float(longitude) + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    # Ecuador is published entirely in southern-hemisphere UTM (INEC uses 17S
    # nationwide). Area is invariant to the N/S false-northing, so southern
    # zones give INEC-consistent, undistorted areas across the equator.
    return f"EPSG:327{zone:02d}"


def area_sq_km(geometry: gpd.GeoSeries) -> pd.Series:
    """Area in km^2, computed per feature in its own local UTM zone.

    For a single-zone region (e.g. Quito, entirely in 17S/EPSG:32717) this is
    identical to reprojecting the whole series to that zone. Grouping by zone
    keeps national/cross-zone runs undistorted."""

    if not geometry.crs:
        return geometry.area / 1_000_000

    reps = geometry.representative_point().to_crs("EPSG:4326")
    zones = pd.Series(
        [local_utm_epsg(point.x, point.y) for point in reps],
        index=geometry.index,
    )
    areas = pd.Series(0.0, index=geometry.index, dtype="float64")
    for epsg, group_index in zones.groupby(zones).groups.items():
        subset = geometry.loc[group_index]
        areas.loc[group_index] = (subset.to_crs(epsg).area / 1_000_000).values
    return areas


def standardize_sector_boundary(
    gdf: gpd.GeoDataFrame, *, source_year: str, region: CityRegion = QUITO_REGION
) -> gpd.GeoDataFrame:
    result = gdf.copy()
    upper_columns = {column.upper(): column for column in result.columns}
    if all(column in upper_columns for column in ["I01", "I02", "I03", "I04", "I05"]):
        result["tract_id"] = (
            result[upper_columns["I01"]].map(lambda value: clean_code(value, 2))
            + result[upper_columns["I02"]].map(lambda value: clean_code(value, 2))
            + result[upper_columns["I03"]].map(lambda value: clean_code(value, 2))
            + result[upper_columns["I04"]].map(lambda value: clean_code(value, 3))
            + result[upper_columns["I05"]].map(lambda value: clean_code(value, 3))
        )
    elif "SEC_ANM" in upper_columns:
        result["tract_id"] = (
            result[upper_columns["SEC_ANM"]]
            .astype(str)
            .str.extract(r"(\d{12,15})", expand=False)
            .str.slice(0, 12)
        )
    elif "tract_id" not in result.columns:
        tract_column = None
        for column in result.columns:
            if column == result.geometry.name:
                continue
            values = result[column].dropna().astype(str).str.strip()
            if not values.empty and values.str.match(r"^\d{12,15}$").mean() >= 0.5:
                tract_column = column
                break
        if not tract_column:
            return gpd.GeoDataFrame(columns=["tract_id", "geometry"], geometry="geometry", crs=gdf.crs)
        result["tract_id"] = result[tract_column].astype(str).str.extract(r"(\d{12,15})", expand=False).str.slice(0, 12)

    result["tract_id"] = result["tract_id"].astype(str).str.strip().str.slice(0, 12)
    if region.is_national:
        result = result.loc[result["tract_id"].str.fullmatch(r"\d{12}", na=False)].copy()
    else:
        canton_prefix = result["tract_id"].str.slice(0, 4)
        result = result.loc[canton_prefix.isin(region.canton_codes)].copy()
    if result.empty:
        return result
    if result["tract_id"].duplicated().any():
        duplicates = result.loc[result["tract_id"].duplicated(keep=False), "tract_id"].head(5).tolist()
        raise ValueError(f"Duplicate sector IDs found: {duplicates}")

    def source_text(name: str) -> pd.Series:
        original = upper_columns.get(name.upper())
        if original is None:
            return pd.Series(pd.NA, index=result.index, dtype="string")
        return result[original].astype("string").str.strip().replace("", pd.NA)

    result["sec_anm"] = result["tract_id"]
    for name, width, fallback_width in [
        ("parroquia", 6, 6),
        ("canton", 4, 4),
        ("provincia", 2, 2),
    ]:
        cleaned = source_text(name).map(lambda value: clean_code(value, width))
        fallback = result["tract_id"].str.slice(0, fallback_width)
        result[name] = cleaned.mask(cleaned.eq(""), fallback)
    for name in ["nom_par", "nom_can", "nom_pro"]:
        result[name] = source_text(name)
    result["anio"] = source_text("anio").str.replace(r"\.0$", "", regex=True)
    result["fuente"] = source_text("fuente")
    fcode = source_text("fcode")
    result["fcode"] = fcode.fillna("sector_censal")

    if region.canton_codes is not None and len(region.canton_codes) == 1:
        result["city"] = region.display_name
    else:
        nom_can = result["nom_can"].astype("string")
        result["city"] = nom_can.str.title().fillna(region.display_name)
    result["country"] = "Ecuador"
    result["boundary_unit"] = "sector censal"
    result["source_year"] = source_year
    result["source_agency"] = "INEC Ecuador"
    # Measure area in each sector's local UTM zone before forcing the shared
    # storage CRS, so cross-zone (national) runs stay undistorted and Quito
    # (zone 17S == storage CRS) is unchanged.
    result["area_sq_km"] = area_sq_km(result.geometry)
    if result.crs is not None and result.crs.to_string() != QUITO_AREA_CRS:
        result = result.to_crs(QUITO_AREA_CRS)

    keep_columns = [
        "tract_id",
        *BOUNDARY_ATTRIBUTE_COLUMNS,
        "geometry",
    ]
    output = result[keep_columns].sort_values("tract_id").reset_index(drop=True)
    output.attrs["boundary_source"] = "cpv_2022_anonymous_sector"
    return output


def materialize_boundary_geopackage(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if info.filename.lower().endswith(".gpkg")]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one GeoPackage in {zip_path.name}, found {len(members)}"
            )
        member = members[0]
        # Extract the inner GeoPackage into the build-scratch dir, NOT next to the
        # source zip: the boundary zip ships read-only under the delivered data/raw/,
        # and materializing the ~688 MB inner .gpkg there would pollute the package.
        extract_dir = PROCESSED_DIR / "0_boundary_cache"
        extract_dir.mkdir(parents=True, exist_ok=True)
        target = extract_dir / Path(member.filename).name
        if target.exists():
            if target.stat().st_size != member.file_size:
                raise ValueError(
                    f"Materialized {target.name} has {target.stat().st_size} bytes; expected {member.file_size}"
                )
            return target

        part = target.with_suffix(f"{target.suffix}.part")
        with archive.open(member) as source, part.open("wb") as destination:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(chunk)
        if part.stat().st_size != member.file_size:
            actual_size = part.stat().st_size
            part.unlink()
            raise ValueError(
                f"Extracted {target.name} has {actual_size} bytes; expected {member.file_size}"
            )
        part.replace(target)
        return target


def anonymous_boundary_where(
    fields: Iterable[str], region: CityRegion = QUITO_REGION
) -> str | None:
    if region.is_national:
        return None
    lookup = {str(field).lower(): str(field) for field in fields}
    codes = sorted(region.canton_codes)
    if "canton" in lookup:
        field = lookup["canton"]
        if len(codes) == 1:
            return f"{field} = '{codes[0]}'"
        joined = ", ".join(f"'{code}'" for code in codes)
        return f"{field} IN ({joined})"
    if "sec_anm" in lookup:
        field = lookup["sec_anm"]
        clauses = " OR ".join(f"{field} LIKE '{code}%'" for code in codes)
        return clauses
    return None


def read_anonymous_quito_sectors(region: CityRegion = QUITO_REGION) -> gpd.GeoDataFrame | None:
    if not ANON_SECTOR_ZIP.exists():
        return None

    uris = [str(materialize_boundary_geopackage(ANON_SECTOR_ZIP))]
    for uri in uris:
        try:
            layers = gpd.list_layers(uri)
        except Exception:
            layers = pd.DataFrame()

        layer_names = layers["name"].tolist() if "name" in layers.columns else [None]
        for layer in layer_names or [None]:
            try:
                info = pyogrio.read_info(uri, layer=layer)
                where = anonymous_boundary_where(info.get("fields", []), region)
                gdf = (
                    gpd.read_file(uri, layer=layer, where=where, engine="pyogrio")
                    if layer
                    else gpd.read_file(uri, where=where, engine="pyogrio")
                )
            except Exception:
                continue
            standardized = standardize_sector_boundary(gdf, source_year="2022", region=region)
            if not standardized.empty:
                standardized.attrs["national_sector_count"] = int(info.get("features", len(gdf)))
                standardized.attrs["region_sector_count"] = len(standardized)
                standardized.attrs["region_parroquia_count"] = int(
                    standardized["parroquia"].nunique(dropna=True)
                )
                return standardized
    return None


def read_legacy_quito_sectors() -> gpd.GeoDataFrame:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing raw geography ZIP: {ZIP_PATH}")

    gdf = gpd.read_file(
        ZIP_GPKG_URI,
        layer="sec_a",
        engine="pyogrio",
        where=f"sec LIKE '{QUITO_CANTON_PREFIX}%'",
    )
    if gdf.empty:
        raise RuntimeError("No Quito sector features found with sec LIKE '1701%'.")
    if gdf["sec"].duplicated().any():
        duplicates = gdf.loc[gdf["sec"].duplicated(), "sec"].head(5).tolist()
        raise RuntimeError(f"Duplicate sector IDs found: {duplicates}")

    gdf = gdf.rename(columns={"sec": "tract_id"})
    gdf["tract_id"] = gdf["tract_id"].astype(str)
    gdf["sec_anm"] = gdf["tract_id"]
    gdf["city"] = "Quito"
    gdf["country"] = "Ecuador"
    gdf["boundary_unit"] = "sector censal"
    gdf["source_year"] = gdf["anio"].astype("string").str.replace(r"\.0$", "", regex=True)
    gdf["source_agency"] = "INEC Ecuador"
    for name, width in [("parroquia", 6), ("canton", 4), ("provincia", 2)]:
        if name not in gdf.columns:
            gdf[name] = gdf["tract_id"].str.slice(0, width)
    for name in ["nom_par", "nom_can", "nom_pro"]:
        if name not in gdf.columns:
            gdf[name] = pd.Series(pd.NA, index=gdf.index, dtype="string")
    if "fuente" not in gdf.columns:
        gdf["fuente"] = pd.Series(pd.NA, index=gdf.index, dtype="string")
    if "fcode" not in gdf.columns:
        gdf["fcode"] = "sector_censal"
    if gdf.crs is not None and gdf.crs.to_string() != QUITO_AREA_CRS:
        gdf = gdf.to_crs(QUITO_AREA_CRS)
    gdf["area_sq_km"] = area_sq_km(gdf.geometry)

    keep_columns = [
        "tract_id",
        *BOUNDARY_ATTRIBUTE_COLUMNS,
        "geometry",
    ]
    output = gdf[keep_columns].sort_values("tract_id").reset_index(drop=True)
    output.attrs["boundary_source"] = "pichincha_geopackage"
    return output


def read_quito_sectors(region: CityRegion = QUITO_REGION) -> gpd.GeoDataFrame:
    anonymous = read_anonymous_quito_sectors(region)
    if anonymous is not None:
        return anonymous
    if region.canton_codes != QUITO_REGION.canton_codes:
        raise FileNotFoundError(
            "The 2020 Pichincha legacy boundary only covers Quito; the official "
            "national CapaSectores.zip is required for "
            f"{region.display_name}. Missing: {ANON_SECTOR_ZIP}"
        )
    return read_legacy_quito_sectors()


def read_indicator_export(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return normalize_redatam_report(pd.read_excel(path, dtype=str, header=None))
    if suffix in {".csv", ".txt", ".tsv"}:
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                return normalize_redatam_report(pd.read_csv(path, dtype=str, sep=None, engine="python", encoding=encoding, header=None))
            except UnicodeDecodeError:
                continue
        return normalize_redatam_report(pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="latin1", header=None))
    raise ValueError(f"Unsupported indicator export type: {path.suffix}")


def normalize_redatam_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert REDATAM report-style tables into a normal dataframe.

    REDATAM Excel exports often put metadata in the first rows and the real
    header in a later row beginning with "Codigo". If no such row is found,
    return the table with its original first row header behavior preserved.
    """
    frame = frame.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if frame.empty:
        return frame

    header_index: int | None = None
    for index, row in frame.iterrows():
        normalized_values = {normalize_name(value) for value in row.dropna().tolist()}
        if "codigo" in normalized_values and any(value.startswith("nombre") for value in normalized_values):
            header_index = int(index)
            break

    if header_index is None:
        headers = frame.iloc[0].fillna("").astype(str).tolist()
        data = frame.iloc[1:].copy()
        data.columns = unique_columns(headers)
        return data.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    headers = frame.iloc[header_index].fillna("").astype(str).tolist()
    data = frame.iloc[header_index + 1 :].copy()
    data.columns = unique_columns(headers)
    data = data.dropna(how="all").dropna(axis=1, how="all")
    unnamed_columns = [column for column in data.columns if column.startswith("unnamed")]
    data = data.drop(columns=unnamed_columns, errors="ignore")
    return data.reset_index(drop=True)


def find_geocode_column(frame: pd.DataFrame) -> str | None:
    preferred_names = [
        "tract_id",
        "sec",
        "sector",
        "sector_censal",
        "codigo_sector",
        "cod_sector",
        "cod_sec",
        "codigo",
        "codigo_geografico",
        "codgeo",
        "geocodigo",
        "geocode",
    ]
    for name in preferred_names:
        if name in frame.columns:
            return name

    best_column = None
    best_rate = 0.0
    for column in frame.columns:
        values = frame[column].dropna().astype(str).str.strip()
        if values.empty:
            continue
        rate = values.str.match(r"^1701\d{8,11}$").mean()
        if rate > best_rate:
            best_rate = float(rate)
            best_column = column
    return best_column if best_rate >= 0.5 else None


def numeric_or_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.6)):
        return numeric
    return series


def collapse_export(frame: pd.DataFrame, geocode_column: str) -> tuple[str, pd.DataFrame]:
    frame = frame.copy()
    raw_codes = frame[geocode_column].astype(str).str.strip()
    tract_codes = raw_codes.str.extract(r"((?:1701)\d{8,11})", expand=False)
    parish_codes = raw_codes.str.extract(r"((?:1701)\d{2})", expand=False)
    if tract_codes.notna().any():
        key_column = "tract_id"
        frame[key_column] = tract_codes
        frame = frame.dropna(subset=[key_column])
        frame[key_column] = frame[key_column].str.slice(0, 12)
    elif parish_codes.notna().any():
        key_column = "parroquia"
        frame[key_column] = parish_codes
        frame = frame.dropna(subset=[key_column])
    else:
        return "tract_id", pd.DataFrame(columns=["tract_id"])

    value_columns = [column for column in frame.columns if column not in {geocode_column, key_column}]
    collapsed = pd.DataFrame({key_column: sorted(frame[key_column].unique())})
    for column in value_columns:
        converted = numeric_or_text(frame[column])
        temp = pd.DataFrame({key_column: frame[key_column], column: converted})
        if pd.api.types.is_numeric_dtype(temp[column]):
            grouped = temp.groupby(key_column, as_index=False)[column].sum(min_count=1)
        else:
            grouped = temp.groupby(key_column, as_index=False)[column].first()
        collapsed = collapsed.merge(grouped, on=key_column, how="left")
    return key_column, collapsed


def path_reference(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def merge_indicator_exports(indicators: pd.DataFrame) -> tuple[pd.DataFrame, list[ExportResult]]:
    results: list[ExportResult] = []
    export_paths = sorted(
        path
        for path in EXPORT_DIR.glob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".tsv", ".xlsx", ".xls"}
    )

    for path in export_paths:
        try:
            frame = read_indicator_export(path)
        except Exception as exc:
            results.append(ExportResult(path=path, rows=0, merged_columns=[], skipped_reason=str(exc)))
            continue

        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        if frame.empty:
            results.append(ExportResult(path=path, rows=0, merged_columns=[], skipped_reason="empty table"))
            continue

        geocode_column = find_geocode_column(frame)
        if not geocode_column:
            results.append(
                ExportResult(path=path, rows=len(frame), merged_columns=[], skipped_reason="no sector, manzana, or parroquia code column")
            )
            continue

        key_column, collapsed = collapse_export(frame, geocode_column)
        if collapsed.empty:
            results.append(
                ExportResult(path=path, rows=len(frame), merged_columns=[], skipped_reason="no Quito geocodes matched 1701 prefix")
            )
            continue

        file_prefix = normalize_name(path.stem)
        rename_map: dict[str, str] = {}
        for column in collapsed.columns:
            if column in {"tract_id", "parroquia"}:
                continue
            if column in STANDARD_INDICATOR_NAMES or column in SUPPLEMENTAL_INDICATOR_NAMES or column.startswith(f"{file_prefix}_"):
                rename_map[column] = column
            else:
                rename_map[column] = f"{file_prefix}_{column}"
        collapsed = collapsed.rename(columns=rename_map)

        import_renames: dict[str, str] = {}
        for column in collapsed.columns:
            if column != key_column and column in indicators.columns:
                import_renames[column] = f"{column}__import"
        collapsed = collapsed.rename(columns=import_renames)

        indicators = indicators.merge(collapsed, on=key_column, how="left")
        new_columns = [column for column in collapsed.columns if column != key_column and not column.endswith("__import")]
        for original, imported in import_renames.items():
            filled = indicators[original].isna() & indicators[imported].notna()
            if filled.any():
                indicators.loc[filled, original] = indicators.loc[filled, imported]
                new_columns.append(original)
            indicators = indicators.drop(columns=imported)

        join_method = "direct_join" if key_column == "tract_id" else "disaggregate"
        results.append(ExportResult(path=path, rows=len(frame), merged_columns=new_columns, join_key=key_column, join_method=join_method))

    return indicators, results


STANDARD_INDICATOR_NAMES = {name for name, _, _ in STANDARD_INDICATORS}
SUPPLEMENTAL_INDICATOR_NAMES = {name for name, _, _ in SUPPLEMENTAL_INDICATORS}


def to_numeric(column: pd.Series) -> pd.Series:
    return pd.to_numeric(column, errors="coerce")


def divide_or_na(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = to_numeric(numerator)
    denominator = to_numeric(denominator)
    result = numerator / denominator.where(denominator != 0)
    return result.where(result.map(math.isfinite))


def percent_to_share(column: pd.Series) -> pd.Series:
    numeric = to_numeric(column)
    if numeric.dropna().gt(1).any():
        numeric = numeric / 100
    return numeric


def fill_missing(indicators: pd.DataFrame, target: str, values: pd.Series) -> None:
    if target not in indicators.columns:
        indicators[target] = pd.Series(pd.NA, index=indicators.index, dtype="Float64")
    indicators[target] = indicators[target].where(indicators[target].notna(), values)


def derive_assignment_indicators(
    indicators: pd.DataFrame,
    *,
    sector_population_available: bool | None = None,
    allow_parroquia_fallback: bool = True,
) -> pd.DataFrame:
    result = indicators.copy()
    result = add_standard_columns(result)
    for column, _, _ in SUPPLEMENTAL_INDICATORS:
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="Float64")

    if sector_population_available is None:
        sector_population_available = "parroquia" not in result.columns and result.get("pop_total", pd.Series(dtype="Float64")).notna().any()
    if sector_population_available and "pop_total" in result.columns and "area_sq_km" in result.columns:
        fill_missing(result, "pop_density", divide_or_na(result["pop_total"], result["area_sq_km"]))

    if not allow_parroquia_fallback:
        return result

    population = "counts_quito_parroquia_poblacion"
    households = "counts_quito_parroquia_hogares"
    if population in result.columns:
        parish_population = to_numeric(result[population])
        fill_missing(result, "pop_total", parish_population)
        if "area_sq_km" in result.columns and "parroquia" in result.columns:
            parish_area = to_numeric(result["area_sq_km"]).groupby(result["parroquia"]).transform("sum")
            fill_missing(result, "pop_density", divide_or_na(parish_population, parish_area))
    if households in result.columns:
        household_count = to_numeric(result[households])
        fill_missing(result, "household_count", household_count)
        if population in result.columns:
            fill_missing(result, "avg_household_size", divide_or_na(to_numeric(result[population]), household_count))

    pop_15 = "education_quito_parroquia_poblacion_de_15_anos_y_mas"
    if pop_15 in result.columns:
        pop_15_series = to_numeric(result[pop_15])
        higher_education_columns = [
            "education_quito_parroquia_cine_p_6_educacion_terciaria_o_equivalente",
            "education_quito_parroquia_cine_p_7_maestria_especializacion_o_equivalente",
            "education_quito_parroquia_cine_p_8_doctorado_o_equivalente",
        ]
        if all(column in result.columns for column in higher_education_columns):
            higher_education = sum(to_numeric(result[column]) for column in higher_education_columns)
            fill_missing(result, "education_high_share", divide_or_na(higher_education, pop_15_series))
        literacy = "education_quito_parroquia_alfabeto"
        if literacy in result.columns:
            fill_missing(result, "literacy_rate", divide_or_na(to_numeric(result[literacy]), pop_15_series))

    employed = "employment_quito_parroquia_poblacion_ocupada"
    unemployed = "employment_quito_parroquia_poblacion_desocupada"
    employment_pop_15 = "employment_quito_parroquia_poblacion_de_15_anos_y_mas"
    if employed in result.columns and unemployed in result.columns:
        employed_series = to_numeric(result[employed])
        unemployed_series = to_numeric(result[unemployed])
        labor_force = employed_series + unemployed_series
        fill_missing(result, "unemployment_rate", divide_or_na(unemployed_series, labor_force))
        if employment_pop_15 in result.columns:
            employment_denominator = to_numeric(result[employment_pop_15])
            fill_missing(result, "labor_force_participation_rate", divide_or_na(labor_force, employment_denominator))
            fill_missing(result, "employment_rate", divide_or_na(employed_series, employment_denominator))

    percent_sources = {
        "cellphone_service_share": "household_ict_quito_parroquia_servicio_de_telefono_celular",
        "computer_access_share": "household_ict_quito_parroquia_disponibilidad_de_computadora",
        "fixed_internet_share": "household_ict_quito_parroquia_servicio_de_internet_fijo",
        "basic_services_share": "housing_services_quito_parroquia_viviendas_con_disponibilidad_de_todos_los_servicios_basicos",
    }
    for target, source in percent_sources.items():
        if source in result.columns:
            fill_missing(result, target, percent_to_share(result[source]))

    return result


def add_standard_columns(indicators: pd.DataFrame) -> pd.DataFrame:
    for column, _, _ in STANDARD_INDICATORS:
        if column not in indicators.columns:
            indicators[column] = pd.Series(pd.NA, index=indicators.index, dtype="Float64")
    return indicators


def indicator_source_table(variable: str) -> str:
    if variable in PERSON_INDICATORS:
        return "CPV_2022_Poblacion_Sector.csv"
    if variable in HOUSEHOLD_INDICATORS:
        return "CPV_2022_Hogar_Sector.csv"
    if variable in HOUSING_INDICATORS:
        return "CPV_2022_Vivienda_Sector.csv"
    return ""


def build_indicator_inventory(
    indicators: pd.DataFrame,
    export_results: list[ExportResult],
) -> pd.DataFrame:
    join_id = "tract_id = I01(2)+I02(2)+I03(2)+I04(3)+I05(3); boundary sec_anm"
    rows: list[dict[str, object]] = []
    for variable in INVENTORY_INDICATORS:
        has_values = variable in indicators.columns and indicators[variable].notna().any()
        direct_sector = (
            variable in XRAY_TARGET_INDICATORS
            and has_values
            and sector_source_available(export_results, variable)
        )
        if variable in MISSING_ECONOMIC_PRICE_INDICATORS:
            status = "missing"
            source_agency = ""
            source_table = ""
            source_url = ""
            download_url = ""
            file_format = "not available"
            current_join_id = ""
            notes = "No defensible official sector-level source was selected."
        elif variable == "pop_density":
            status = (
                "derived"
                if has_values and sector_source_available(export_results, "pop_total")
                else "missing"
            )
            source_agency = "INEC Ecuador"
            source_table = "CPV_2022_Poblacion_Sector.csv; CapaSectores.zip"
            source_url = CPV_DATA_PAGE
            download_url = f"{CPV_SECTOR_CSV_URL}; {ANON_SECTOR_URL}"
            file_format = "CSV in ZIP; GeoPackage in ZIP"
            current_join_id = join_id if status == "derived" else ""
            notes = "Derived as sector population divided by projected polygon area in square kilometres."
        else:
            status = "available" if direct_sector else "missing"
            source_agency = "INEC Ecuador"
            source_table = indicator_source_table(variable)
            source_url = CPV_DATA_PAGE
            download_url = CPV_SECTOR_CSV_URL
            file_format = "CSV in ZIP"
            current_join_id = join_id if direct_sector else ""
            notes = (
                "Aggregated from official CPV 2022 records to the 12-digit sector key and directly joined to sec_anm."
                if direct_sector
                else "Official direct sector values were not available in this processing run."
            )
            if variable == "basic_services_share" and direct_sector:
                notes = f"{notes} Formula: {BASIC_SERVICES_FORMULA}"
        rows.append(
            {
                "country": "Ecuador",
                "city": "Quito",
                "tract_like_unit": "sector censal",
                "indicator_name": variable,
                "indicator_category": INDICATOR_CATEGORIES[variable],
                "source_agency": source_agency,
                "source_table": source_table,
                "source_url": source_url,
                "download_url": download_url,
                "source_year": "2022" if status in {"available", "derived"} else "",
                "spatial_unit": "sector censal" if status in {"available", "derived"} else "",
                "file_format": file_format,
                "join_id": current_join_id,
                "status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def write_indicator_inventory(
    indicators: pd.DataFrame,
    export_results: list[ExportResult],
) -> None:
    path = PROCESSED_DIR / "4_metadata" / "indicator_inventory.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    build_indicator_inventory(indicators, export_results).to_excel(
        path,
        sheet_name="indicator_inventory",
        index=False,
        engine="openpyxl",
    )


def validate_indicator_inventory(path: Path) -> None:
    try:
        inventory = pd.read_excel(path, dtype=str)
    except Exception as error:
        raise ValueError(f"unable to read indicator inventory: {error}") from error
    if inventory.columns.tolist() != INVENTORY_COLUMNS:
        raise ValueError("indicator inventory columns do not match the required 15-column schema")
    if len(inventory) != len(INVENTORY_INDICATORS):
        raise ValueError(
            f"indicator inventory contains {len(inventory)} rows; expected {len(INVENTORY_INDICATORS)}"
        )
    names = inventory["indicator_name"].astype(str)
    if names.duplicated().any() or set(names) != set(INVENTORY_INDICATORS):
        raise ValueError("indicator inventory indicator_name values do not match the 23-indicator contract")
    expected_statuses = {"available": 18, "derived": 1, "missing": 4}
    actual_statuses = inventory["status"].astype(str).value_counts().to_dict()
    if actual_statuses != expected_statuses:
        raise ValueError(
            f"indicator inventory status distribution is {actual_statuses}; expected {expected_statuses}"
        )
    status_by_name = inventory.set_index("indicator_name")["status"].astype(str).to_dict()
    if {name for name, status in status_by_name.items() if status == "available"} != set(
        XRAY_TARGET_INDICATORS
    ):
        raise ValueError("indicator inventory available rows do not match the 18 direct targets")
    if status_by_name.get("pop_density") != "derived":
        raise ValueError("indicator inventory must mark pop_density as derived")
    if {
        name for name, status in status_by_name.items() if status == "missing"
    } != MISSING_ECONOMIC_PRICE_INDICATORS:
        raise ValueError("indicator inventory missing rows do not match the four economic/price fields")
    usable = inventory["status"].isin(["available", "derived"])
    if inventory.loc[usable, "join_id"].fillna("").str.strip().eq("").any():
        raise ValueError("indicator inventory usable rows require a join_id")
    if not inventory.loc[usable, "source_year"].fillna("").str.strip().eq("2022").all():
        raise ValueError("indicator inventory usable rows must have source_year 2022")
    missing = ~usable
    for column in ("source_year", "spatial_unit"):
        if not inventory.loc[missing, column].fillna("").str.strip().eq("").all():
            raise ValueError(f"indicator inventory missing rows must leave {column} blank")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_source_list(export_results: list[ExportResult]) -> None:
    cpv_result = next(
        (result for result in export_results if result.source_kind == "cpv_sector"),
        None,
    )
    manual_results = [result for result in export_results if result.source_kind == "manual_export"]
    cpv_active = bool(cpv_result and cpv_result.merged_columns and not cpv_result.skipped_reason)
    cpv_notes = (
        f"Active source. Aggregated {cpv_result.rows} CPV records to sector indicators: "
        f"{', '.join(cpv_result.merged_columns)}. Access policy and citation: {ANDA_METADATA_URL}. "
        f"Required citation: {INEC_REQUIRED_CITATION}"
        if cpv_active and cpv_result
        else f"Not used in this run. Download with scripts/download/download_ecuador_cpv_sector.py. Access policy: {ANDA_METADATA_URL}."
    )
    rows: list[dict[str, object]] = [
        {
            "city": "Quito",
            "country": "Ecuador",
            "data_type": "boundary",
            "indicator_name": "INEC Pichincha sector boundary",
            "source_agency": "INEC Ecuador",
            "source_url": GEOPORTAL_URL,
            "download_url": PICHINCHA_BOUNDARY_URL,
            "source_year": "2020",
            "access_date": OFFICIAL_SOURCE_ACCESS_DATE,
            "license": INEC_PUBLIC_USE_LICENSE,
            "notes": (
                "Not used in this run. Fallback sector geometry filtered from sec_a to Quito canton prefix 1701."
                if cpv_active
                else "Active fallback source. Sector geometry filtered from sec_a to Quito canton prefix 1701."
            ),
        },
        {
            "city": "Quito",
            "country": "Ecuador",
            "data_type": "boundary",
            "indicator_name": "CPV 2022 anonymous sector boundary",
            "source_agency": "INEC Ecuador",
            "source_url": CPV_DATA_PAGE,
            "download_url": ANON_SECTOR_URL,
            "source_year": "2022",
            "access_date": OFFICIAL_SOURCE_ACCESS_DATE,
            "license": INEC_PUBLIC_USE_LICENSE,
            "notes": (
                f"Active source: CPV 2022 product. Local file: {ANON_SECTOR_ZIP.relative_to(ROOT)}. "
                "Original Quito boundary fields retain mixed anio/fuente values: "
                "7,169 sectors are 2020/CPV2021 and 10 are 2022/CPV2022. "
                f"Access policy and citation: {ANDA_METADATA_URL}."
                if cpv_active
                else f"Not used in this run. Optional local file: {ANON_SECTOR_ZIP.relative_to(ROOT)}."
            ),
        },
        {
            "city": "Quito",
            "country": "Ecuador",
            "data_type": "indicator table",
            "indicator_name": "CPV 2022 sector CSV database",
            "source_agency": "INEC Ecuador",
            "source_url": CPV_DATA_PAGE,
            "download_url": CPV_SECTOR_CSV_URL,
            "source_year": "2022",
            "access_date": OFFICIAL_SOURCE_ACCESS_DATE,
            "license": INEC_PUBLIC_USE_LICENSE,
            "notes": cpv_notes,
        },
        {
            "city": "Quito",
            "country": "Ecuador",
            "data_type": "indicator portal",
            "indicator_name": "CPV 2022 REDATAM indicator exports",
            "source_agency": "INEC Ecuador",
            "source_url": REDATAM_URL,
            "download_url": "",
            "source_year": "2022",
            "access_date": OFFICIAL_SOURCE_ACCESS_DATE,
            "license": INEC_PUBLIC_USE_LICENSE,
            "notes": (
                "Active supplemental or fallback source. Interactive exports are stored under "
                f"{EXPORT_DIR.relative_to(ROOT)}/. Access policy and citation: {ANDA_METADATA_URL}."
                if manual_results
                else "Not used in this run. Interactive REDATAM exports are an assignment fallback."
            ),
        },
    ]
    for result in manual_results:
        notes = result.skipped_reason or f"Merged on {result.join_key} using {result.join_method}: {', '.join(result.merged_columns)}"
        rows.append(
            {
                "city": "Quito",
                "country": "Ecuador",
                "data_type": "indicator table",
                "indicator_name": result.path.stem,
                "source_agency": "INEC Ecuador",
                "source_url": REDATAM_URL,
                "download_url": path_reference(result.path),
                "source_year": "2022",
                "access_date": OFFICIAL_SOURCE_ACCESS_DATE,
                "license": INEC_PUBLIC_USE_LICENSE,
                "notes": notes,
            }
        )
    deduplicated: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in rows:
        identity = (row["source_agency"], row["indicator_name"], row["download_url"])
        deduplicated[identity] = row
    write_csv(
        PROCESSED_DIR / "4_metadata" / "source_list.csv",
        SOURCE_COLUMNS,
        list(deduplicated.values()),
    )


def missing_rate(series: pd.Series) -> str:
    if len(series) == 0:
        return ""
    return f"{series.isna().mean():.4f}"


def indicator_quality_override(variable: str) -> dict[str, str] | None:
    if variable not in DERIVED_PARROQUIA_INDICATORS:
        return None
    return {
        "indicator_unit": "parroquia",
        "join_method": "disaggregate",
        "source_year": "2022",
        "quality_flag": "partial",
        "notes": DERIVED_PARROQUIA_NOTE,
    }


def imported_metadata(export_results: list[ExportResult]) -> dict[str, ExportResult]:
    metadata: dict[str, ExportResult] = {}
    for result in export_results:
        for column in result.merged_columns:
            existing = metadata.get(column)
            if existing and existing.source_kind == "cpv_sector" and result.source_kind != "cpv_sector":
                continue
            metadata[column] = result
    return metadata


def audit_xray_model_data(
    boundary: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    export_results: list[ExportResult],
    *,
    minimum_coverage: float = XRAY_MINIMUM_COVERAGE,
) -> XRayDataAudit:
    errors: list[str] = []
    required_boundary_source_columns = {
        "sec_anm",
        "anio",
        "fuente",
        "parroquia",
        "nom_par",
        "canton",
        "nom_can",
        "provincia",
        "nom_pro",
    }
    missing_boundary_columns = sorted(required_boundary_source_columns - set(boundary.columns))
    if missing_boundary_columns:
        errors.append(
            "boundary is missing original source columns: " + ", ".join(missing_boundary_columns)
        )
    if "tract_id" not in boundary.columns:
        errors.append("boundary is missing tract_id")
        boundary_ids = pd.Index([], dtype="object")
    else:
        if boundary["tract_id"].duplicated().any():
            errors.append("duplicate boundary tract_id values")
        boundary_ids = pd.Index(boundary["tract_id"].dropna().astype(str).unique())
        if "sec_anm" in boundary.columns and not boundary["tract_id"].astype(str).equals(
            boundary["sec_anm"].astype(str)
        ):
            errors.append("boundary tract_id values do not match original sec_anm values")

    if "tract_id" not in indicators.columns:
        errors.append("indicator table is missing tract_id")
        indexed = pd.DataFrame(index=boundary_ids)
    else:
        if indicators["tract_id"].duplicated().any():
            errors.append("duplicate indicator tract_id values")
        normalized = indicators.copy()
        normalized["tract_id"] = normalized["tract_id"].astype(str)
        indexed = normalized.drop_duplicates("tract_id").set_index("tract_id")

    boundary_source = boundary.attrs.get("boundary_source")
    if boundary_source != "cpv_2022_anonymous_sector":
        errors.append(f"boundary source is not CPV 2022 anonymous sector geography: {boundary_source}")
    if "source_year" not in boundary.columns or not boundary["source_year"].astype(str).eq("2022").all():
        errors.append("boundary CPV product source_year is not uniformly 2022")
    for column in ["anio", "fuente"]:
        if column in boundary.columns and boundary[column].astype("string").fillna("").str.strip().eq("").any():
            errors.append(f"boundary original {column} contains missing values")

    available_target_columns = [feature for feature in XRAY_TARGET_INDICATORS if feature in indexed.columns]
    if len(boundary_ids) and available_target_columns:
        matched_count = int(
            indexed.reindex(boundary_ids)[available_target_columns].notna().any(axis=1).sum()
        )
    else:
        matched_count = 0
    join_coverage = matched_count / len(boundary_ids) if len(boundary_ids) else 0.0
    if join_coverage < minimum_coverage:
        errors.append(
            f"boundary-to-indicator join coverage {join_coverage:.1%} is below {minimum_coverage:.0%}"
        )

    metadata = imported_metadata(export_results)
    target_rows: list[dict[str, object]] = []
    indirect_targets: list[str] = []
    for feature in XRAY_TARGET_INDICATORS:
        if feature in indexed.columns:
            values = indexed.reindex(boundary_ids)[feature]
            coverage = float(values.notna().mean()) if len(values) else 0.0
            unique_values = int(values.nunique(dropna=True))
        else:
            coverage = 0.0
            unique_values = 0

        source = metadata.get(feature)
        direct_sector = bool(
            source
            and source.source_kind == "cpv_sector"
            and source.join_key == "tract_id"
            and source.join_method == "direct_join"
        )
        source_status = "direct_sector" if direct_sector else "not_direct_sector"
        target_rows.append(
            {
                "feature": feature,
                "coverage": coverage,
                "unique_values": unique_values,
                "source_status": source_status,
            }
        )
        if coverage < minimum_coverage:
            errors.append(f"{feature} coverage {coverage:.1%} is below {minimum_coverage:.0%}")
        if not direct_sector:
            indirect_targets.append(feature)

    if indirect_targets:
        errors.append("targets are not direct sector indicators: " + ", ".join(indirect_targets))

    return XRayDataAudit(
        join_coverage=join_coverage,
        target_table=pd.DataFrame(target_rows),
        errors=tuple(errors),
    )


def dictionary_source_for_variable(
    variable: str,
    export_results: list[ExportResult],
    *,
    has_values: bool,
) -> tuple[str, str, str]:
    metadata = imported_metadata(export_results)
    result = metadata.get(variable)
    if result and result.source_kind == "cpv_sector" and has_values:
        if variable == "basic_services_share":
            return "available", BASIC_SERVICES_FORMULA, "CPV 2022 sector CSV database"
        return "available", "derived from CPV 2022 sector CSV records", "CPV 2022 sector CSV database"
    if variable == "pop_density" and sector_source_available(export_results, "pop_total") and has_values:
        return "derived", "derived from sector-level pop_total and sector polygon area", "CPV 2022 sector CSV database; derived from INEC geometry"
    if not has_values:
        return "missing", variable, "not joined"
    if variable in DERIVED_PARROQUIA_INDICATORS:
        return "partial", "derived from parroquia-level CPV 2022 REDATAM columns", "CPV 2022 REDATAM export; parroquia-level derivation"
    if result and result.source_kind == "manual_export":
        source = "CPV 2022 REDATAM export"
        if result.join_key == "parroquia":
            return "partial", variable, f"{source}; parroquia-level derivation"
        return "available", variable, source
    return "available", variable, "CPV 2022 REDATAM export"


def sector_source_available(export_results: list[ExportResult], variable: str) -> bool:
    return any(result.source_kind == "cpv_sector" and variable in result.merged_columns for result in export_results)


def dataset_boundary_year(indicators: pd.DataFrame) -> str:
    if "source_year" not in indicators.columns:
        return ""
    years = indicators["source_year"].dropna().astype(str).unique().tolist()
    return years[0] if len(years) == 1 else ""


def write_data_dictionary(indicators: pd.DataFrame, export_results: list[ExportResult]) -> None:
    rows: list[dict[str, object]] = []
    boundary_year = dataset_boundary_year(indicators)
    for variable, standard, description, unit, original, source, year, quality in BASE_DICTIONARY_ROWS:
        rows.append(
            {
                "variable_name": variable,
                "standard_name": standard,
                "description": description,
                "unit": unit,
                "original_name": original,
                "source": source,
                "year": boundary_year if boundary_year and variable in BOUNDARY_YEAR_VARIABLES else year,
                "missing_rate": missing_rate(indicators[variable]) if variable in indicators.columns else "",
                "quality_flag": quality,
            }
        )

    for variable, description, unit in STANDARD_INDICATORS:
        has_values = variable in indicators.columns and indicators[variable].notna().any()
        quality, original, source = dictionary_source_for_variable(variable, export_results, has_values=has_values)
        rows.append(
            {
                "variable_name": variable,
                "standard_name": variable,
                "description": description,
                "unit": unit,
                "original_name": original,
                "source": source,
                "year": "2022",
                "missing_rate": missing_rate(indicators[variable]) if variable in indicators.columns else "1.0000",
                "quality_flag": quality,
            }
        )

    for variable, description, unit in SUPPLEMENTAL_INDICATORS:
        has_values = variable in indicators.columns and indicators[variable].notna().any()
        quality, original, source = dictionary_source_for_variable(variable, export_results, has_values=has_values)
        rows.append(
            {
                "variable_name": variable,
                "standard_name": variable,
                "description": description,
                "unit": unit,
                "original_name": original,
                "source": source,
                "year": "2022",
                "missing_rate": missing_rate(indicators[variable]) if variable in indicators.columns else "1.0000",
                "quality_flag": quality,
            }
        )

    generated_columns = {
        column
        for result in export_results
        for column in result.merged_columns
        if column in indicators.columns and column not in STANDARD_INDICATOR_NAMES and column not in SUPPLEMENTAL_INDICATOR_NAMES
    }
    for variable in sorted(generated_columns):
        result = imported_metadata(export_results).get(variable)
        source = "CPV 2022 REDATAM export"
        description = "Imported REDATAM export column; inspect original table before modeling."
        quality = "partial"
        if result and result.source_kind == "cpv_sector":
            source = "CPV 2022 sector CSV database"
            description = "Imported CPV 2022 sector CSV column."
            quality = "available"
        rows.append(
            {
                "variable_name": variable,
                "standard_name": variable,
                "description": description,
                "unit": "unknown",
                "original_name": variable,
                "source": source,
                "year": "2022",
                "missing_rate": missing_rate(indicators[variable]),
                "quality_flag": quality,
            }
        )

    write_csv(PROCESSED_DIR / "4_metadata" / "data_dictionary.csv", DICTIONARY_COLUMNS, rows)


def write_quality_table(indicators: pd.DataFrame, export_results: list[ExportResult]) -> None:
    metadata = imported_metadata(export_results)
    boundary_year = dataset_boundary_year(indicators)

    rows: list[dict[str, object]] = []
    for variable in indicators.columns:
        if variable == "tract_id" or (
            variable in BOUNDARY_ATTRIBUTE_COLUMNS and variable != "area_sq_km"
        ):
            continue
        if variable == "area_sq_km":
            rows.append(
                {
                    "variable_name": variable,
                    "boundary_unit": "sector censal",
                    "indicator_unit": "sector censal",
                    "join_method": "direct_join",
                    "source_year": "2020/2022",
                    "quality_flag": "available",
                    "notes": "Measured directly from the official INEC sector polygon geometry (same spatial unit).",
                }
            )
            continue
        else:
            result = metadata.get(variable)
            if result and result.source_kind == "cpv_sector" and result.join_key == "tract_id":
                notes = "Derived from official CPV 2022 sector-level microdata and joined by anonymized sector code."
                if variable == "basic_services_share":
                    notes = f"{notes} Formula: {BASIC_SERVICES_FORMULA}"
                rows.append(
                    {
                        "variable_name": variable,
                        "boundary_unit": "sector censal",
                        "indicator_unit": "sector censal",
                        "join_method": result.join_method or "direct_join",
                        "source_year": "2022",
                        "quality_flag": "available",
                        "notes": notes,
                    }
                )
                continue
            if variable == "pop_density" and sector_source_available(export_results, "pop_total") and indicators[variable].notna().any():
                rows.append(
                    {
                        "variable_name": variable,
                        "boundary_unit": "sector censal",
                        "indicator_unit": "sector censal",
                        "join_method": "direct_join",
                        "source_year": "2022",
                        "quality_flag": "derived",
                        "notes": "Derived from CPV 2022 sector-level population divided by sector polygon area (same spatial unit).",
                    }
                )
                continue
        if override := indicator_quality_override(variable):
            rows.append(
                {
                    "variable_name": variable,
                    "boundary_unit": "sector censal",
                    **override,
                }
            )
        elif variable in (STANDARD_INDICATOR_NAMES | SUPPLEMENTAL_INDICATOR_NAMES) and indicators[variable].isna().all():
            rows.append(
                {
                    "variable_name": variable,
                    "boundary_unit": "sector censal",
                    "indicator_unit": "not_joined",
                    "join_method": "not_joined",
                    "source_year": "",
                    "quality_flag": "missing",
                    "notes": "Awaiting an official sector-level or assignment-compatible source.",
                }
            )
        else:
            result = metadata.get(variable)
            indicator_unit = "parroquia" if result and result.join_key == "parroquia" else "sector/manzana export"
            join_method = result.join_method if result and result.join_method else "direct_join_or_aggregate"
            rows.append(
                {
                    "variable_name": variable,
                    "boundary_unit": "sector censal",
                    "indicator_unit": indicator_unit,
                    "join_method": join_method,
                    "source_year": "2022",
                    "quality_flag": "partial",
                    "notes": "Imported automatically; parroquia-level values are repeated across sectors when join_method is disaggregate.",
                }
            )
    write_csv(PROCESSED_DIR / "4_metadata" / "indicator_quality.csv", QUALITY_COLUMNS, rows)


def boundary_vintage_counts(boundary: gpd.GeoDataFrame) -> list[tuple[str, str, int]]:
    if not {"anio", "fuente"}.issubset(boundary.columns):
        return []
    values = boundary.loc[:, ["anio", "fuente"]].copy()
    values["anio"] = values["anio"].astype("string").fillna("missing").str.strip()
    values["fuente"] = values["fuente"].astype("string").fillna("missing").str.strip()
    grouped = (
        values.groupby(["anio", "fuente"], dropna=False)
        .size()
        .reset_index(name="sector_count")
        .sort_values(["anio", "fuente"])
    )
    return [
        (str(row.anio), str(row.fuente), int(row.sector_count))
        for row in grouped.itertuples(index=False)
    ]


def write_quality_report(
    boundary: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    export_results: list[ExportResult],
) -> None:
    merged = [result for result in export_results if result.merged_columns]
    skipped = [result for result in export_results if result.skipped_reason]
    missing_standard = [
        variable
        for variable, _, _ in STANDARD_INDICATORS
        if variable not in indicators.columns or indicators[variable].isna().all()
    ]
    populated_standard = len(STANDARD_INDICATORS) - len(missing_standard)
    populated_supplemental = [
        variable
        for variable, _, _ in SUPPLEMENTAL_INDICATORS
        if variable in indicators.columns and indicators[variable].notna().any()
    ]
    cpv_sector_results = [result for result in export_results if result.source_kind == "cpv_sector"]
    has_cpv_sector = any(result.merged_columns for result in cpv_sector_results)
    manual_results = [result for result in export_results if result.source_kind == "manual_export"]
    boundary_source = boundary.attrs.get("boundary_source", "pichincha_geopackage")
    sector_boundary_used = boundary_source == "cpv_2022_anonymous_sector"
    sector_standard = sorted(
        {
            column
            for result in cpv_sector_results
            for column in result.merged_columns
            if column in STANDARD_INDICATOR_NAMES
        }
    )
    parroquia_standard = sorted(
        variable
        for variable in DERIVED_PARROQUIA_INDICATORS & STANDARD_INDICATOR_NAMES
        if manual_results and variable in indicators.columns and indicators[variable].notna().any()
    )
    xray_audit = audit_xray_model_data(boundary, indicators, export_results)
    def numeric_values(column: str) -> pd.Series:
        if column not in indicators.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(indicators[column], errors="coerce").dropna()

    population = numeric_values("pop_total")
    sector_area = numeric_values("area_sq_km")

    def summary_value(values: pd.Series, statistic: str, decimals: int) -> str:
        if values.empty:
            return "not available"
        value = values.mean() if statistic == "mean" else values.median()
        return f"{value:.{decimals}f}"

    mean_population = summary_value(population, "mean", 1)
    median_population = summary_value(population, "median", 1)
    mean_area = summary_value(sector_area, "mean", 4)
    median_area = summary_value(sector_area, "median", 4)
    direct_target_count = int(xray_audit.target_table["source_status"].eq("direct_sector").sum())
    missing_standard_text = ", ".join(f"`{variable}`" for variable in missing_standard) or "none"
    national_sector_count = boundary.attrs.get("national_sector_count")
    national_sector_text = (
        f"{int(national_sector_count):,}" if national_sector_count is not None else "not recorded"
    )
    region_parroquia_count = boundary.attrs.get(
        "region_parroquia_count",
        boundary["parroquia"].nunique(dropna=True) if "parroquia" in boundary.columns else 0,
    )
    region_display = boundary.attrs.get("region_display", "Quito")
    region_filter = boundary.attrs.get("region_filter", f"canton `{QUITO_CANTON_PREFIX}`")
    null_geometry_count = int(boundary.geometry.isna().sum())
    empty_geometry_count = int(boundary.geometry.is_empty.sum())
    invalid_geometry_count = int(
        ((~boundary.geometry.is_valid) & boundary.geometry.notna()).sum()
    )
    geometry_wkb = boundary.geometry.loc[boundary.geometry.notna()].to_wkb()
    duplicate_geometry_count = int(geometry_wkb.duplicated().sum())
    multipart_count = int(
        boundary.geometry.map(
            lambda geometry: bool(
                geometry is not None
                and geometry.geom_type.startswith("Multi")
                and len(geometry.geoms) > 1
            )
        ).sum()
    )
    nonpositive_area_count = int((sector_area <= 0).sum()) if not sector_area.empty else 0
    vintage_counts = boundary_vintage_counts(boundary)

    lines = [
        f"# Ecuador / {region_display} Stage 1 Quality Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Boundary",
        "",
        f"- Boundary unit: sector censal",
        "- Boundary source: "
        + (
            "CPV 2022 anonymous sector product (`CapaSectores.zip`)."
            if sector_boundary_used
            else "2020 INEC Pichincha sector GeoPackage fallback (`17_PICHINCHA.zip`)."
        ),
        f"- Filter: {region_filter}.",
        f"- National sector count: {national_sector_text}",
        f"- {region_display} sector count: {len(boundary):,}",
        f"- {region_display} parroquia count: {int(region_parroquia_count):,}",
        "- CPV product/indicator year: `2022`",
        f"- CRS: {boundary.crs}",
        f"- Duplicate tract_id count: {int(boundary['tract_id'].duplicated().sum())}",
        f"- Null geometry count: {null_geometry_count}",
        f"- Empty geometry count: {empty_geometry_count}",
        f"- Invalid geometry count: {invalid_geometry_count}",
        f"- Duplicate exact geometry count: {duplicate_geometry_count}",
        f"- Multipart geometry count: {multipart_count}",
        f"- Non-positive area count: {nonpositive_area_count}",
        "- Original boundary vintage distribution:",
        *(
            [
                f"  - `anio={anio}`, `fuente={fuente}`: {count:,} sectors"
                for anio, fuente, count in vintage_counts
            ]
            or ["  - not recorded"]
        ),
        "",
        "## Tract-like Unit Profile",
        "",
        "- Official unit: Ecuadorian `sector censal`, an official census/statistical enumeration unit.",
        "- Similarity assessment: Medium tract-like similarity. Sectors provide stable small-area identifiers and citywide coverage, but their typical population is smaller than a United States Census Tract.",
        f"- Mean sector population: {mean_population} residents",
        f"- Median sector population: {median_population} residents",
        f"- Mean sector area: {mean_area} sq km",
        f"- Median sector area: {median_area} sq km",
        "- Selection rationale: the sector is the finest official CPV 2022 geography that can be joined reproducibly to the anonymous sector database across Quito.",
        "",
        "## Indicator Exports",
        "",
    ]
    if not export_results:
        lines.extend(
            [
                "- No CPV 2022 sector ZIP or REDATAM fallback exports were found.",
                f"- Run `uv run python scripts/download/download_ecuador_cpv_sector.py` to download official sector ZIPs into `{CPV_SECTOR_DIR.relative_to(ROOT)}/`, or save CSV/XLS/XLSX fallback exports under `{EXPORT_DIR.relative_to(ROOT)}/`.",
            ]
        )
    else:
        for result in export_results:
            if result.skipped_reason:
                lines.append(f"- Skipped `{result.path.name}`: {result.skipped_reason}.")
            else:
                lines.append(f"- Merged `{result.path.name}` on `{result.join_key}` using `{result.join_method}`: {len(result.merged_columns)} columns.")
    lines.extend(
        [
            "",
            "## Current Standard Indicator Status",
            "",
            f"- Standard indicators missing: {len(missing_standard)} / {len(STANDARD_INDICATORS)}",
            f"- Standard indicators populated: {populated_standard} / {len(STANDARD_INDICATORS)}",
            f"- Supplemental indicators populated: {len(populated_supplemental)} / {len(SUPPLEMENTAL_INDICATORS)}",
            "- Standard indicators from official sector CSV: "
            + (", ".join(f"`{variable}`" for variable in sector_standard) if sector_standard else "none"),
            "- Standard indicators filled from REDATAM fallback exports: "
            + (", ".join(f"`{variable}`" for variable in parroquia_standard) if parroquia_standard else "none"),
            "",
            "Missing standard indicators:",
        ]
    )
    if missing_standard:
        lines.extend([f"- `{variable}`" for variable in missing_standard])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Assignment Quality Checklist",
            "",
            "1. **Why is the unit tract-like?** It is an official, stable, fine-grained census statistical unit with complete city coverage and direct socioeconomic joins.",
            f"2. **Is it sufficiently fine?** Yes for neighbourhood-scale analysis: this run contains {len(boundary)} sectors with a median population of {median_population} residents and median area of {median_area} sq km.",
            "3. **Are indicators and boundaries at the same scale?** "
            + (
                "Yes. Active CPV 2022 indicators are aggregated by anonymous sector code and joined directly to the matching CPV 2022 sector boundary."
                if has_cpv_sector and sector_boundary_used
                else "Not completely. Fallback exports or geography require the partial/disaggregate flags documented below."
            ),
            f"4. **Which indicators are most reliable?** The {direct_target_count} direct-sector Quito X-Ray targets and population density derived from sector population and polygon area are the strongest fields.",
            "5. **Which indicators are partial or derived?** Only `pop_density` is derived (sector population divided by sector polygon area). `area_sq_km` is measured directly from the official sector geometry. No parroquia disaggregation is used in the strict run."
            if has_cpv_sector and not manual_results
            else "5. **Which indicators are partial or derived?** Geometry and density fields are derived; any parroquia fallback fields are marked `partial` and `disaggregate`.",
            f"6. **Which indicators are missing, and why?** {missing_standard_text}. No defensible official sector-level source was selected for the missing economic or price fields.",
            "7. **Are years mixed?** The modeled indicators and `source_year` use the CPV 2022 product year. Original boundary cartography is mixed and remains explicitly recorded in `anio` and `fuente`."
            if has_cpv_sector and sector_boundary_used
            else "7. **Are years mixed?** Fallback geometry may be 2020 while CPV indicators are 2022; metadata records this mismatch explicitly.",
            "8. **Do boundary and indicator years mismatch?** Product and indicator provenance is 2022, while the official package contains original 2020/CPV2021 and 2022/CPV2022 boundary records; both fields are preserved rather than relabeled."
            if has_cpv_sector and sector_boundary_used
            else "8. **Do boundary and indicator years mismatch?** Yes in fallback mode; consult `indicator_quality.csv` before analysis.",
            f"9. **Are any steps not automatically reproducible?** The official ZIP downloads require network access and source availability, but download, checksum validation, aggregation, joins, metadata, and maps are scripted. Raw files remain ignored under `{CPV_SECTOR_DIR.relative_to(ROOT)}/`.",
            "10. **How is the dataset reproduced?** Run `uv run python scripts/download/download_ecuador_cpv_sector.py`, then `uv run python scripts/process/process_census.py --strict-xray`, and finally `uv run pytest`.",
            "",
            "## Quito X-Ray Model Data Audit",
            "",
            f"- Status: {'ready' if xray_audit.is_valid else 'not ready'}",
            f"- Boundary-to-indicator join coverage: {xray_audit.join_coverage:.1%}",
            f"- Required per-target coverage: {XRAY_MINIMUM_COVERAGE:.0%}",
        ]
    )
    if xray_audit.errors:
        lines.append("- Audit failures:")
        lines.extend([f"  - {error}" for error in xray_audit.errors])
    else:
        lines.append("- Audit failures: none")
    lines.extend(
        f"- `{row.feature}`: coverage {row.coverage:.1%}, unique values {row.unique_values}, source `{row.source_status}`"
        for row in xray_audit.target_table.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Spatial Scale Warning",
            "",
            (
                "Official CPV 2022 sector-level indicators are used where the sector CSV ZIP is present and successfully aggregated."
                if has_cpv_sector
                else "CPV 2022 socioeconomic indicators currently come from REDATAM fallback exports."
            ),
            (
                "Fallback REDATAM indicators joined at parroquia level are repeated across sectors and remain marked partial/disaggregate."
                if manual_results
                else "No parroquia-level fallback indicators were merged in this run."
            ),
            "Income, poverty, median rent, and house price fields remain missing unless a separate official source is added.",
            "",
            "## Output Maps",
            "",
            "Formal thematic maps are exported from QGIS by `scripts/qgis/build_quito_data_stage_report.py` "
            "into `5_maps/` after this processing run produces the boundary and joined indicators.",
        ]
    )
    lines.extend(
        [
            "",
            "## Next Manual Step",
            "",
            f"Keep the large official ZIPs out of Git. Download them into the ignored folder `{CPV_SECTOR_DIR.relative_to(ROOT)}/` with `uv run python scripts/download/download_ecuador_cpv_sector.py` when regeneration with sector-level CPV data is needed.",
            "Keep these reproducible raw ZIPs outside version control.",
            "",
        ]
    )
    quality_report_path = PROCESSED_DIR / "4_metadata" / "quality_report.md"
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_path.write_text("\n".join(lines), encoding="utf-8")


def write_export_readme() -> None:
    readme = f"""# CPV 2022 REDATAM fallback exports for Quito

The preferred Stage 1 source is the official CPV 2022 sector CSV ZIP downloaded by:

```bash
uv run python scripts/download/download_ecuador_cpv_sector.py
```

That command writes large raw ZIP files to:

```text
{CPV_SECTOR_DIR.relative_to(ROOT)}/
```

This REDATAM directory is a fallback and supplement location for smaller manual CSV, XLS, or XLSX exports.

Required geography for fallback exports:
- Provincia: 17 Pichincha
- Canton: 1701 Quito
- Preferred output level: sector censal
- Keep the official sector code column. Manzana-level exports are acceptable; the processing script aggregates them to the first 12 digits.

Recommended files:
- population_by_sector.csv
- age_sex_by_sector.csv
- households_by_sector.csv
- education_by_sector.csv
- employment_by_sector.csv
- housing_tenure_by_sector.csv
- housing_services_by_sector.csv
- ict_by_sector.csv

After adding exports, run:

```bash
uv run python scripts/process/process_census.py
```
"""
    (EXPORT_DIR / "README.md").write_text(readme, encoding="utf-8")


def validate_stage1_outputs(processed_dir: Path = PROCESSED_DIR) -> None:
    required_paths = [
        processed_dir / "1_boundary" / "city_boundary.gpkg",
        processed_dir / "2_indicators" / "cleaned_indicators.csv",
        processed_dir / "3_joined_data" / "city_tract_indicators.gpkg",
        processed_dir / "3_joined_data" / "city_tract_indicators.csv",
        processed_dir / "4_metadata" / "source_list.csv",
        processed_dir / "4_metadata" / "data_dictionary.csv",
        processed_dir / "4_metadata" / "indicator_quality.csv",
        processed_dir / "4_metadata" / "indicator_inventory.xlsx",
        processed_dir / "4_metadata" / "quality_report.md",
        processed_dir / "README.md",
    ]
    missing_paths = [str(path.relative_to(processed_dir)) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise ValueError("Stage 1 output validation failed: missing files: " + ", ".join(missing_paths))

    boundary = gpd.read_file(processed_dir / "1_boundary" / "city_boundary.gpkg")
    indicators = pd.read_csv(
        processed_dir / "2_indicators" / "cleaned_indicators.csv",
        dtype={"tract_id": "string"},
    )
    joined_gdf = gpd.read_file(processed_dir / "3_joined_data" / "city_tract_indicators.gpkg")
    joined_csv = pd.read_csv(
        processed_dir / "3_joined_data" / "city_tract_indicators.csv",
        dtype={"tract_id": "string"},
    )
    sources = pd.read_csv(processed_dir / "4_metadata" / "source_list.csv")
    dictionary = pd.read_csv(processed_dir / "4_metadata" / "data_dictionary.csv")
    quality = pd.read_csv(processed_dir / "4_metadata" / "indicator_quality.csv")
    inventory_path = processed_dir / "4_metadata" / "indicator_inventory.xlsx"

    errors: list[str] = []
    try:
        validate_indicator_inventory(inventory_path)
    except ValueError as error:
        errors.append(str(error))
    frames = {
        "boundary": boundary,
        "cleaned indicators": indicators,
        "joined GeoPackage": joined_gdf,
        "joined CSV": joined_csv,
    }
    id_sets: dict[str, set[str]] = {}
    required_boundary_columns = set(BOUNDARY_ATTRIBUTE_COLUMNS)
    for label, frame in frames.items():
        missing_boundary_columns = sorted(required_boundary_columns - set(frame.columns))
        if missing_boundary_columns:
            errors.append(f"{label} is missing columns: {', '.join(missing_boundary_columns)}")
        if "tract_id" not in frame.columns:
            errors.append(f"{label} is missing tract_id")
            continue
        tract_ids = frame["tract_id"].astype("string")
        if tract_ids.isna().any():
            errors.append(f"{label} contains missing tract_id")
        if tract_ids.duplicated().any():
            errors.append(f"{label} contains duplicate tract_id")
        id_sets[label] = set(tract_ids.dropna().astype(str))
        if "sec_anm" in frame.columns and not tract_ids.astype(str).equals(
            frame["sec_anm"].astype("string").astype(str)
        ):
            errors.append(f"{label} tract_id values do not match sec_anm")

    if id_sets:
        expected_ids = next(iter(id_sets.values()))
        for label, tract_ids in id_sets.items():
            if tract_ids != expected_ids:
                errors.append(f"{label} tract_id set does not match the boundary")

    for label, frame in {"boundary": boundary, "joined GeoPackage": joined_gdf}.items():
        if frame.crs is None or frame.crs.to_string() != QUITO_AREA_CRS:
            errors.append(f"{label} CRS is not {QUITO_AREA_CRS}")
        if frame.geometry.isna().any():
            errors.append(f"{label} contains null geometry")
        if frame.geometry.is_empty.any():
            errors.append(f"{label} contains empty geometry")
        if (~frame.geometry.is_valid).any():
            errors.append(f"{label} contains invalid geometry")
        geometry_wkb = frame.geometry.loc[frame.geometry.notna()].to_wkb()
        if geometry_wkb.duplicated().any():
            errors.append(f"{label} contains duplicate exact geometry")
        if "area_sq_km" not in frame.columns:
            errors.append(f"{label} is missing area_sq_km")
        else:
            area = pd.to_numeric(frame["area_sq_km"], errors="coerce")
            if area.isna().any() or area.le(0).any():
                errors.append(f"{label} contains missing or non-positive area_sq_km")

    for label, frame in frames.items():
        if "source_year" not in frame.columns or not frame["source_year"].astype(str).eq("2022").all():
            errors.append(f"{label} CPV product source_year is not uniformly 2022")
        for column in ["anio", "fuente"]:
            if column in frame.columns and frame[column].astype("string").fillna("").str.strip().eq("").any():
                errors.append(f"{label} contains missing original {column}")

    if {"anio", "fuente"}.issubset(boundary.columns):
        observed_vintages = set(
            zip(
                boundary["anio"].astype(str),
                boundary["fuente"].astype(str),
                strict=True,
            )
        )
        expected_vintages = {("2020", "CPV2021"), ("2022", "CPV2022")}
        unexpected_vintages = sorted(observed_vintages - expected_vintages)
        if unexpected_vintages:
            errors.append(f"boundary contains unexpected anio/fuente pairs: {unexpected_vintages}")

    for label, frame, required_columns in [
        ("source list", sources, SOURCE_COLUMNS),
        ("data dictionary", dictionary, DICTIONARY_COLUMNS),
        ("indicator quality", quality, QUALITY_COLUMNS),
    ]:
        missing_columns = sorted(set(required_columns) - set(frame.columns))
        if missing_columns:
            errors.append(f"{label} is missing columns: {', '.join(missing_columns)}")

    if "variable_name" in dictionary.columns and dictionary["variable_name"].duplicated().any():
        errors.append("data dictionary contains duplicate variable_name")
    if "variable_name" in quality.columns and quality["variable_name"].duplicated().any():
        errors.append("indicator quality contains duplicate variable_name")
    source_identity = ["source_agency", "indicator_name", "download_url"]
    if set(source_identity).issubset(sources.columns) and sources.duplicated(source_identity).any():
        errors.append("source list contains duplicate source rows")
    if "indicator_name" in sources.columns:
        cpv_rows = sources["indicator_name"].eq("CPV 2022 sector CSV database").sum()
        if cpv_rows != 1:
            errors.append(f"source list must contain exactly one CPV 2022 sector CSV row, found {cpv_rows}")

    target_frames = {
        "cleaned indicators": indicators,
        "joined GeoPackage": joined_gdf,
        "joined CSV": joined_csv,
    }
    for label, frame in target_frames.items():
        missing_targets = [feature for feature in XRAY_TARGET_INDICATORS if feature not in frame.columns]
        if missing_targets:
            errors.append(f"{label} is missing direct targets: {', '.join(missing_targets)}")
            continue
        numeric_targets = frame.loc[:, list(XRAY_TARGET_INDICATORS)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        for feature in XRAY_TARGET_INDICATORS:
            coverage = float(numeric_targets[feature].notna().mean())
            if coverage < XRAY_MINIMUM_COVERAGE:
                errors.append(
                    f"{label} {feature} coverage {coverage:.1%} is below {XRAY_MINIMUM_COVERAGE:.0%}"
                )
        if numeric_targets["pop_total"].notna().mean() != 1.0:
            errors.append(f"{label} boundary-to-indicator join is not 100%")

    if all(set(XRAY_TARGET_INDICATORS).issubset(frame.columns) for frame in target_frames.values()):
        expected_targets = (
            indicators.set_index("tract_id")
            .loc[:, list(XRAY_TARGET_INDICATORS)]
            .apply(pd.to_numeric, errors="coerce")
            .sort_index()
            .reset_index(drop=True)
        )
        for label, frame in [("joined GeoPackage", joined_gdf), ("joined CSV", joined_csv)]:
            actual_targets = (
                frame.set_index("tract_id")
                .loc[:, list(XRAY_TARGET_INDICATORS)]
                .apply(pd.to_numeric, errors="coerce")
                .sort_index()
                .reset_index(drop=True)
            )
            try:
                pd.testing.assert_frame_equal(
                    expected_targets,
                    actual_targets,
                    check_dtype=False,
                    check_names=False,
                    atol=1e-12,
                    rtol=1e-12,
                )
            except AssertionError:
                errors.append(f"{label} target values do not match cleaned indicators")

    if set(QUALITY_COLUMNS).issubset(quality.columns):
        quality_by_name = quality.set_index("variable_name")
        for feature in XRAY_TARGET_INDICATORS:
            if feature not in quality_by_name.index:
                errors.append(f"indicator quality is missing direct target: {feature}")
                continue
            row = quality_by_name.loc[feature]
            direct = (
                str(row["indicator_unit"]).strip().lower() == "sector censal"
                and str(row["join_method"]).strip().lower() == "direct_join"
                and str(row["source_year"]).strip() == "2022"
                and str(row["quality_flag"]).strip().lower() == "available"
            )
            if not direct:
                errors.append(f"indicator quality does not mark {feature} as an available direct 2022 sector target")
        if "pop_density" not in quality_by_name.index:
            errors.append("indicator quality is missing pop_density")
        else:
            density = quality_by_name.loc["pop_density"]
            if not (
                str(density["join_method"]).strip().lower() == "direct_join"
                and str(density["quality_flag"]).strip().lower() == "derived"
            ):
                errors.append("indicator quality must mark pop_density as quality_flag=derived with join_method=direct_join")

    if errors:
        raise ValueError("Stage 1 output validation failed: " + "; ".join(errors))


def write_outputs(
    boundary: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    export_results: list[ExportResult],
    region: CityRegion = QUITO_REGION,
) -> None:
    boundary_path = PROCESSED_DIR / "1_boundary" / "city_boundary.gpkg"
    joined_path = PROCESSED_DIR / "3_joined_data" / "city_tract_indicators.gpkg"
    for path in [boundary_path, joined_path]:
        if path.exists():
            path.unlink()

    boundary.to_file(boundary_path, layer="city_boundary", driver="GPKG", engine="pyogrio")

    joined = boundary.merge(
        indicators.drop(columns=BOUNDARY_ATTRIBUTE_COLUMNS, errors="ignore"),
        on="tract_id",
        how="left",
    )
    joined.to_file(joined_path, layer="city_tract_indicators", driver="GPKG", engine="pyogrio")
    joined.drop(columns="geometry").to_csv(PROCESSED_DIR / "3_joined_data" / "city_tract_indicators.csv", index=False)
    indicators.to_csv(PROCESSED_DIR / "2_indicators" / "cleaned_indicators.csv", index=False)

    raw_readme = PROCESSED_DIR / "2_indicators" / "raw_indicator_tables" / "README.md"
    raw_readme.write_text(
        "# Raw indicator tables\n\n"
        "These are the per-sector count tables aggregated directly from the official "
        "CPV 2022 microdata records, before any share/rate computation or field "
        "standardization. They are the raw indicator tables that back "
        "`../cleaned_indicators.csv`.\n\n"
        "- `person_sector_counts.csv`: population counts per sector (sex, age bands, "
        "education, literacy, employment numerators and their valid-response denominators).\n"
        "- `household_sector_counts.csv`: household counts per sector (tenure, ICT access, "
        "size).\n"
        "- `housing_sector_counts.csv`: dwelling counts per sector (occupancy, basic "
        "services, housing deficit).\n\n"
        f"Each row is one {region.display_name} sector (`tract_id`). Dividing these counts yields the "
        "shares and rates in `cleaned_indicators.csv`.\n\n"
        "The full official source microdata are too large to vendor here and remain in "
        "the ignored raw path:\n"
        f"- Official CPV 2022 sector ZIPs: `{CPV_SECTOR_DIR.relative_to(ROOT)}/`.\n"
        f"- Manual REDATAM fallback exports: `{EXPORT_DIR.relative_to(ROOT)}/`.\n",
        encoding="utf-8",
    )

    write_source_list(export_results)
    write_data_dictionary(indicators, export_results)
    write_quality_table(indicators, export_results)
    write_indicator_inventory(indicators, export_results)
    write_quality_report(boundary, indicators, export_results)
    write_stage_readme(region)


def publish_stage1_outputs(
    boundary: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    export_results: list[ExportResult],
    *,
    strict_xray: bool,
    region: CityRegion = QUITO_REGION,
) -> None:
    audit = audit_xray_model_data(boundary, indicators, export_results)
    if strict_xray:
        audit.require_valid()
    write_outputs(boundary, indicators, export_results, region)
    if strict_xray:
        validate_stage1_outputs(PROCESSED_DIR)


def write_stage_readme(region: CityRegion = QUITO_REGION) -> None:
    display = region.display_name
    if region.canton_codes == QUITO_REGION.canton_codes:
        geography_line = f"- Geography: CPV 2022 anonymous sectors filtered to Quito canton prefix `{QUITO_CANTON_PREFIX}`."
        vintage_line = (
            "- Product-year semantics: `source_year=2022` identifies the CPV 2022 product and indicators. "
            "Original boundary provenance remains in `anio` and `fuente`: 7,169 sectors are `2020/CPV2021`, "
            "and 10 sectors are `2022/CPV2022`."
        )
    else:
        geography_line = f"- Geography: CPV 2022 anonymous sectors filtered to {region.filter_description()}."
        vintage_line = (
            "- Product-year semantics: `source_year=2022` identifies the CPV 2022 product and indicators. "
            "Original boundary provenance remains per sector in `anio` and `fuente`."
        )
    readme = f"""# Ecuador / {display} Stage 1 Data Product

This folder contains the census-sector data product for {display} built from the official CPV 2022 anonymous sector boundary and sector CSV database published by INEC Ecuador.

## Current product

- Official unit: `sector censal`, assessed as medium tract-like similarity because it is a stable, fine-grained census unit that is typically smaller than a United States Census Tract.
{geography_line}
{vintage_line}
- Indicators: 18 model-ready sector indicators from population, household, housing, education, employment, ICT, tenure, and services records.
- Inventory: 23 investigated indicators comprise 18 directly available targets, one derived population-density field, and four missing economic/price fields.
- Missing but not required: median income, poverty, median rent, and house price, because no defensible official sector-level source was selected.

## Artifact inventory

- `1_boundary/city_boundary.gpkg`: unique sector geometry, `sec_anm`, administrative codes/names, and original `anio`/`fuente` fields.
- `2_indicators/cleaned_indicators.csv`: standardized sector indicators.
- `3_joined_data/city_tract_indicators.gpkg` and `.csv`: model-ready spatial and tabular joins.
- `4_metadata/indicator_inventory.xlsx`: required 15-column, 23-row availability inventory.
- `4_metadata/`: source register, data dictionary, indicator quality table, inventory, and assignment quality report.
- `5_maps/`: formal QGIS thematic maps exported by `scripts/qgis/build_quito_data_stage_report.py`.

## Reproduce and validate

```bash
uv run python scripts/download/download_ecuador_cpv_sector.py
uv run python scripts/process/process_census.py --city {region.slug} --strict-xray
uv run pytest
```

The downloader validates file sizes and SHA-256 checksums and writes the large raw ZIP files to this ignored directory:

```text
{CPV_SECTOR_DIR.relative_to(ROOT)}/
```

Do not commit raw ZIPs, generated tables, GeoPackages, or maps. The processor verifies matching unique IDs, CPV 2022 product provenance, retained raw boundary vintages, geometry validity, all 18 direct targets, the 15-column inventory, and metadata uniqueness before a strict run succeeds. Formal thematic maps are then exported from QGIS by `scripts/qgis/build_quito_data_stage_report.py`.

Manual REDATAM exports remain available only as a documented assignment fallback under:

```text
{EXPORT_DIR.relative_to(ROOT)}/
```

Fallback fields joined at parroquia scale are marked `partial` and `disaggregate` and cannot qualify as Quito X-Ray model targets.

"""
    (PROCESSED_DIR / "README.md").write_text(readme.rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Ecuador CPV 2022 Stage 1 data product for a selected city.")
    parser.add_argument(
        "--city",
        default="quito",
        help=(
            "City to build: a 4-digit INEC canton code, a city name (e.g. quito, "
            "guayaquil, cuenca), or 'all' for national scope. Default: quito."
        ),
    )
    parser.add_argument(
        "--list-cities",
        action="store_true",
        help="Print the built-in city registry (canton code and name) and exit.",
    )
    parser.add_argument(
        "--strict-xray",
        action="store_true",
        help="Require paired CPV 2022 sources and pass the X-Ray model-target audit.",
    )
    return parser.parse_args(argv)


def main(*, strict_xray: bool = False, city: str | None = None) -> None:
    global PROCESSED_DIR
    region = resolve_city(city)
    PROCESSED_DIR = region.processed_dir

    ensure_dirs()
    write_export_readme()
    validate_official_source_pair(CPV_SECTOR_CSV_ZIP, ANON_SECTOR_ZIP)

    boundary = read_quito_sectors(region)
    boundary.attrs["region_display"] = region.display_name
    boundary.attrs["region_filter"] = region.filter_description()
    indicators = boundary.drop(columns="geometry")
    indicators = add_standard_columns(indicators)
    indicators, cpv_sector_result = merge_cpv_sector_dataset(indicators, region)
    export_results: list[ExportResult] = []
    if cpv_sector_result is not None:
        export_results.append(cpv_sector_result)
    indicators, manual_export_results = merge_indicator_exports(indicators)
    export_results.extend(manual_export_results)
    indicators = derive_assignment_indicators(
        indicators,
        sector_population_available=bool(cpv_sector_result and "pop_total" in cpv_sector_result.merged_columns),
        allow_parroquia_fallback=not strict_xray,
    )
    publish_stage1_outputs(
        boundary,
        indicators,
        export_results,
        strict_xray=strict_xray,
    )

    print(f"Wrote {region.display_name} stage 1 outputs under {PROCESSED_DIR.relative_to(ROOT)}")
    print(f"Boundary sectors: {len(boundary)}")
    if cpv_sector_result and cpv_sector_result.merged_columns:
        print(f"CPV sector ZIP aggregated: {len(cpv_sector_result.merged_columns)} indicators from {cpv_sector_result.rows} records")
    else:
        print(f"No CPV sector ZIP found under {CPV_SECTOR_CSV_ZIP.relative_to(ROOT)}")
    if export_results:
        print(f"Indicator export files scanned: {len(export_results)}")
    else:
        print(f"No indicator exports found under {EXPORT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    args = parse_args()
    if args.list_cities:
        print("Built-in city registry (canton code, name); any 4-digit code or 'all' also works:")
        for code, display in list_cities():
            print(f"  {code}  {display}")
    else:
        main(strict_xray=args.strict_xray, city=args.city)
