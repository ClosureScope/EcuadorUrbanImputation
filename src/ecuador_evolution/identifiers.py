from __future__ import annotations

import pandas as pd

SECTOR_WIDTHS = (2, 2, 2, 3, 3)


def code(value: object, width: int) -> str:
    if pd.isna(value):
        raise ValueError("Geographic code cannot be missing")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid geographic code {value!r}")
    if len(text) > width:
        raise ValueError(f"Code {text!r} exceeds width {width}")
    return text.zfill(width)


def canton_code(province: object, canton: object) -> str:
    return code(province, 2) + code(canton, 2)


def sector_id(*parts: object) -> str:
    if len(parts) != len(SECTOR_WIDTHS):
        raise ValueError(f"Expected {len(SECTOR_WIDTHS)} sector components")
    return "".join(code(value, width) for value, width in zip(parts, SECTOR_WIDTHS, strict=True))


def normalize_sector_id(value: object) -> str:
    """Normalize a complete census-sector code to the canonical 12 digits."""
    return code(value, sum(SECTOR_WIDTHS))


def dwelling_id_2010(sector: object, dwelling: object) -> str:
    """Reconstruct the identifier omitted from the 2010 sector CSV release."""
    return normalize_sector_id(sector) + code(dwelling, 3)


def household_id_2010(sector: object, dwelling: object, household: object) -> str:
    """Reconstruct the 2010 household identifier from its released components."""
    return dwelling_id_2010(sector, dwelling) + code(household, 1)


def add_geographic_ids(frame: pd.DataFrame, columns: tuple[str, ...] = ("I01", "I02", "I03", "I04", "I05")) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing geographic fields: {sorted(missing)}")
    result = frame.copy()
    result["city_id"] = [canton_code(a, b) for a, b in zip(result[columns[0]], result[columns[1]], strict=True)]
    result["sector_id"] = [sector_id(*row) for row in result.loc[:, columns].itertuples(index=False, name=None)]
    return result


def add_2010_record_ids(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = ("I01", "I02", "I03", "I04", "I05"),
    dwelling_column: str = "I09",
    household_column: str = "I10",
) -> pd.DataFrame:
    """Add canonical geography, dwelling, and household IDs to a 2010 table."""
    result = add_geographic_ids(frame, columns)
    if dwelling_column not in result:
        raise KeyError(f"Missing dwelling field: {dwelling_column}")
    result["dwelling_id"] = [
        dwelling_id_2010(sector, dwelling)
        for sector, dwelling in zip(result["sector_id"], result[dwelling_column], strict=True)
    ]
    if household_column in result:
        result["household_id"] = [
            household_id_2010(sector, dwelling, household)
            for sector, dwelling, household in zip(
                result["sector_id"], result[dwelling_column], result[household_column], strict=True
            )
        ]
    return result
