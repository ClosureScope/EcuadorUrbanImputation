from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from ecuador_evolution.crosswalk import build_crosswalk, classify_quality, transfer_components
from ecuador_evolution.validation import _crosswalk_conservation


def test_quality_thresholds() -> None:
    assert classify_quality(0.98, 3) == "high"
    assert classify_quality(0.90, 8) == "medium"
    assert classify_quality(0.89, 1) == "exclude"


def test_synthetic_crosswalk_conservation() -> None:
    source = gpd.GeoDataFrame({"source_sector_id": ["a", "b"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:32717")
    target = gpd.GeoDataFrame({"target_2022_sector_id": ["x"]}, geometry=[box(0, 0, 2, 1)], crs="EPSG:32717")
    crosswalk = build_crosswalk(source, target)
    assert np.allclose(crosswalk.groupby("source_sector_id")["weight"].sum(), 1)
    assert crosswalk["match_quality"].eq("high").all()
    values = pd.DataFrame({"source_sector_id": ["a", "b"], "numerator": [20.0, 40.0], "denominator": [40.0, 80.0]})
    transferred = transfer_components(values, crosswalk)
    assert transferred.loc[0, "transferred_numerator"] == 60
    assert transferred.loc[0, "value"] == 0.5


def test_conservation_includes_excluded_target_rows() -> None:
    crosswalk = pd.DataFrame(
        {
            "source_sector_id": ["a", "a", "b"],
            "weight": [0.6, 0.4, 1.0],
            "match_quality": ["high", "exclude", "medium"],
        }
    )
    conserved, maximum_deviation = _crosswalk_conservation(crosswalk)
    assert conserved
    assert maximum_deviation == 0.0
