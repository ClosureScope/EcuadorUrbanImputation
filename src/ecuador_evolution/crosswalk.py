from __future__ import annotations

import numpy as np
import pandas as pd


def classify_quality(
    coverage: float,
    effective_contributors: int,
    *,
    high_coverage: float = 0.98,
    medium_coverage: float = 0.90,
    high_max_contributors: int = 3,
    medium_max_contributors: int = 8,
) -> str:
    if coverage >= high_coverage and effective_contributors <= high_max_contributors:
        return "high"
    if coverage >= medium_coverage and effective_contributors <= medium_max_contributors:
        return "medium"
    return "exclude"


def build_crosswalk(
    source,
    target,
    *,
    source_id: str = "source_sector_id",
    target_id: str = "target_2022_sector_id",
    target_crs: str = "EPSG:32717",
    source_match_minimum: float = 0.95,
    high_coverage: float = 0.98,
    medium_coverage: float = 0.90,
    high_max_contributors: int = 3,
    medium_max_contributors: int = 8,
    effective_weight_minimum: float = 0.01,
) -> pd.DataFrame:
    """Construct a geometry-only area crosswalk without target attributes."""
    import geopandas as gpd

    for frame, column, label in ((source, source_id, "source"), (target, target_id, "target")):
        if column not in frame:
            raise KeyError(f"Missing {label} ID {column}")
        if frame[column].duplicated().any():
            raise ValueError(f"Duplicate {label} IDs")
    sources = source[[source_id, "geometry"]].to_crs(target_crs).copy()
    targets = target[[target_id, "geometry"]].to_crs(target_crs).copy()
    if sources.geometry.is_empty.any() or targets.geometry.is_empty.any():
        raise ValueError("Empty geometry cannot be crosswalked")
    sources["source_area"] = sources.geometry.area
    targets["target_area"] = targets.geometry.area
    intersections = gpd.overlay(sources, targets, how="intersection", keep_geom_type=False)
    if intersections.empty:
        return pd.DataFrame(columns=[source_id, target_id, "weight", "match_quality"])
    intersections["intersection_area"] = intersections.geometry.area
    intersections["raw_weight"] = intersections["intersection_area"] / intersections["source_area"]
    source_coverage = intersections.groupby(source_id)["raw_weight"].sum().rename("source_coverage")
    intersections = intersections.join(source_coverage, on=source_id)
    intersections = intersections.loc[intersections["source_coverage"] >= source_match_minimum].copy()
    intersections["weight"] = intersections["raw_weight"] / intersections["source_coverage"]

    intersections["incoming_share"] = intersections["intersection_area"] / intersections.groupby(target_id)[
        "intersection_area"
    ].transform("sum")
    intersections["effective"] = intersections["incoming_share"] >= effective_weight_minimum
    summary = intersections.groupby(target_id).agg(
        matched_area=("intersection_area", "sum"),
        target_area=("target_area", "first"),
        effective_contributors=("effective", "sum"),
        contributor_count=(source_id, "nunique"),
    )
    summary["target_coverage"] = (summary["matched_area"] / summary["target_area"]).clip(upper=1.0)
    summary["match_quality"] = [
        classify_quality(
            float(coverage),
            int(contributors),
            high_coverage=high_coverage,
            medium_coverage=medium_coverage,
            high_max_contributors=high_max_contributors,
            medium_max_contributors=medium_max_contributors,
        )
        for coverage, contributors in zip(summary["target_coverage"], summary["effective_contributors"], strict=True)
    ]
    result = intersections.drop(columns=["geometry", "effective"]).merge(summary.reset_index(), on=target_id)
    return result.sort_values([target_id, source_id]).reset_index(drop=True)


def transfer_components(
    source_values: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    source_id: str = "source_sector_id",
    target_id: str = "target_2022_sector_id",
    components: tuple[str, ...] = ("numerator", "denominator"),
    accepted_quality: tuple[str, ...] = ("high", "medium"),
) -> pd.DataFrame:
    missing = set((source_id, *components)) - set(source_values.columns)
    if missing:
        raise KeyError(f"Missing source transfer columns: {sorted(missing)}")
    accepted = crosswalk.loc[crosswalk["match_quality"].isin(accepted_quality)]
    merged = accepted.merge(source_values, on=source_id, how="inner", validate="many_to_one")
    for component in components:
        merged[f"transferred_{component}"] = merged[component] * merged["weight"]
    aggregations = {f"transferred_{component}": "sum" for component in components}
    aggregations |= {
        "target_coverage": "first",
        "contributor_count": "first",
        "effective_contributors": "first",
        "match_quality": "first",
    }
    result = merged.groupby(target_id, as_index=False).agg(aggregations)
    if {"numerator", "denominator"}.issubset(components):
        numerator = result["transferred_numerator"].to_numpy(float)
        denominator = result["transferred_denominator"].to_numpy(float)
        result["value"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(result), np.nan),
            where=denominator > 0,
        )
    return result
