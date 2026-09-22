from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def spatial_blocks(frame: pd.DataFrame, *, blocks_per_city: int = 5) -> np.ndarray:
    """Deterministic contiguous-ish centroid grid blocks, target-independent."""
    required = {"city_id", "centroid_x", "centroid_y", "sector_id"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing split fields: {sorted(missing)}")
    result = np.zeros(len(frame), dtype=np.int64)
    for _, positions in frame.groupby("city_id", sort=True).indices.items():
        positions = np.asarray(positions)
        city = frame.iloc[positions]
        order = np.lexsort((city["sector_id"].astype(str), city["centroid_y"], city["centroid_x"]))
        result[positions[order]] = np.arange(len(positions)) * blocks_per_city // max(len(positions), 1)
    return np.minimum(result, blocks_per_city - 1)


def transfer_folds(city_ids: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cities = sorted(city_ids.astype(str).unique())
    values = city_ids.astype(str).to_numpy()
    return {city: (values != city, values == city) for city in cities}


def stable_random_mask(ids: pd.Series, fraction: float, seed: int) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    scores = np.array(
        [int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big") for value in ids],
        dtype=np.uint64,
    )
    count = round(len(ids) * fraction)
    mask = np.zeros(len(ids), dtype=bool)
    if count:
        mask[np.argsort(scores)[:count]] = True
    return mask
