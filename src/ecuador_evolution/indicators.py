from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Indicator:
    name: str
    unit: str
    universe: str
    numerator: str
    denominator: str
    comparability: str
    recode: str
    fallback: bool = False
    source_2010: dict[str, object] = field(default_factory=dict)
    source_2022: dict[str, object] = field(default_factory=dict)
    flags: tuple[str, ...] = ()


def read_indicator_dictionary(path: str | Path) -> list[Indicator]:
    with Path(path).open("rb") as stream:
        entries = tomllib.load(stream).get("indicator", [])
    indicators = [Indicator(**entry) for entry in entries]
    if len(indicators) < 12:
        raise ValueError("Indicator dictionary must define at least 12 candidates")
    return indicators


def safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    return np.divide(num, den, out=result, where=np.isfinite(den) & (den > 0))


def category_counts(
    values: pd.Series,
    *,
    numerator_categories: Iterable[object],
    denominator_categories: Iterable[object],
) -> tuple[int, int, float]:
    numerator_set = set(numerator_categories)
    denominator_set = set(denominator_categories)
    denominator = int(values.isin(denominator_set).sum())
    numerator = int(values.isin(numerator_set).sum())
    value = float(numerator / denominator) if denominator else float("nan")
    return numerator, denominator, value


def occupied_private_dwellings(
    households: pd.DataFrame,
    dwellings: pd.DataFrame,
    *,
    dwelling_key: str,
    private_field: str,
    occupied_field: str,
    private_values: set[object],
    occupied_values: set[object],
) -> pd.DataFrame:
    """Return unique eligible dwellings reached from household records."""
    linked = dwellings.merge(
        households[[dwelling_key]].dropna().drop_duplicates(),
        on=dwelling_key,
        how="inner",
        validate="one_to_one",
    )
    return linked.loc[
        linked[private_field].isin(private_values) & linked[occupied_field].isin(occupied_values)
    ].drop_duplicates(dwelling_key)
