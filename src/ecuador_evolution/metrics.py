from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def global_indicator_sigma(
    truth_2022: np.ndarray,
    indicator_names: list[str],
) -> dict[str, float]:
    """Per-indicator global standard deviation over all 2022 truth rows.

    This is the fixed ruler used to normalize errors: one std per indicator,
    computed once over every sector (np.nanstd), identical for every method,
    protocol, and fold. Feed the result to ``evaluate_predictions`` as
    ``indicator_sigma`` to obtain the std-normalized MAE_z / RMSE_z caliber.
    """
    truth_2022 = np.asarray(truth_2022, dtype=float)
    return {name: float(np.nanstd(truth_2022[:, index])) for index, name in enumerate(indicator_names)}


def evaluate_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    indicator_names: list[str],
    *,
    population_weights: np.ndarray | None = None,
    indicator_sigma: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-indicator error table plus a ``macro`` equal-weight mean row.

    When ``indicator_sigma`` is supplied, two dimensionless columns are added:
    ``mae_z`` = mae / sigma and ``rmse_z`` = rmse / sigma per indicator, and the
    ``macro`` row's ``mae_z`` / ``rmse_z`` are the equal-weight means across
    indicators. This std-normalized equal-weight caliber matches the standardized
    space the model trains in and avoids the population/household-count dominance
    of a plain mean of native-unit errors; it is the caliber reported in the
    reference results. Without ``indicator_sigma`` the output is unchanged
    (raw native-unit metrics only).
    """
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    normalize = indicator_sigma is not None
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(indicator_names):
        valid = np.isfinite(truth[:, index]) & np.isfinite(prediction[:, index])
        if not valid.any():
            row: dict[str, float | str] = {"indicator": name, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "weighted_mae": np.nan}
            if normalize:
                row["mae_z"] = np.nan
                row["rmse_z"] = np.nan
            rows.append(row)
            continue
        weights = None if population_weights is None else np.asarray(population_weights)[valid]
        mae = mean_absolute_error(truth[valid, index], prediction[valid, index])
        rmse = mean_squared_error(truth[valid, index], prediction[valid, index]) ** 0.5
        row = {
            "indicator": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2_score(truth[valid, index], prediction[valid, index]) if valid.sum() >= 2 else np.nan,
            "weighted_mae": np.average(np.abs(truth[valid, index] - prediction[valid, index]), weights=weights),
        }
        if normalize:
            sigma = indicator_sigma.get(name, np.nan)
            row["mae_z"] = mae / sigma if sigma else np.nan
            row["rmse_z"] = rmse / sigma if sigma else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    columns = ["mae", "rmse", "r2", "weighted_mae"] + (["mae_z", "rmse_z"] if normalize else [])
    macro = {column: float(table[column].mean()) for column in columns}
    return pd.concat([table, pd.DataFrame([{"indicator": "macro", **macro}])], ignore_index=True)


def morans_i(residuals: np.ndarray, neighbor_index: np.ndarray, neighbor_mask: np.ndarray) -> float:
    values = np.asarray(residuals, dtype=float)
    valid = np.isfinite(values)
    centered = values - np.nanmean(values)
    numerator = 0.0
    weight_sum = 0.0
    for row in np.flatnonzero(valid):
        for neighbor in neighbor_index[row, neighbor_mask[row]]:
            if valid[neighbor]:
                numerator += centered[row] * centered[neighbor]
                weight_sum += 1
    denominator = float(np.nansum(centered**2))
    return float(valid.sum() / weight_sum * numerator / denominator) if weight_sum and denominator else float("nan")
