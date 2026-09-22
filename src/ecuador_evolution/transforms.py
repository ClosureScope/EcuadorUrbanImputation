from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ChangeTransformer:
    units: tuple[str, ...]
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, values_2010: np.ndarray, values_2022: np.ndarray, train_mask: np.ndarray) -> "ChangeTransformer":
        raw = self._raw(values_2010, values_2022)
        continuous = np.array([unit != "count" for unit in self.units])
        self.means = np.zeros(raw.shape[1], dtype=float)
        self.scales = np.ones(raw.shape[1], dtype=float)
        if continuous.any():
            train = raw[np.asarray(train_mask, dtype=bool)][:, continuous]
            self.means[continuous] = np.nanmean(train, axis=0)
            scales = np.nanstd(train, axis=0)
            self.scales[continuous] = np.where(scales > 0, scales, 1.0)
        return self

    def _raw(self, values_2010: np.ndarray, values_2022: np.ndarray) -> np.ndarray:
        a = np.asarray(values_2010, dtype=float)
        b = np.asarray(values_2022, dtype=float)
        result = b - a
        for index, unit in enumerate(self.units):
            if unit == "count":
                result[:, index] = np.log1p(np.clip(b[:, index], 0, None)) - np.log1p(np.clip(a[:, index], 0, None))
        return result

    def transform(self, values_2010: np.ndarray, values_2022: np.ndarray) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("ChangeTransformer has not been fitted")
        return (self._raw(values_2010, values_2022) - self.means) / self.scales

    def inverse(self, transformed: np.ndarray, values_2010: np.ndarray) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("ChangeTransformer has not been fitted")
        change = np.asarray(transformed) * self.scales + self.means
        base = np.asarray(values_2010, dtype=float)
        result = base + change
        for index, unit in enumerate(self.units):
            if unit == "count":
                result[:, index] = np.expm1(np.log1p(np.clip(base[:, index], 0, None)) + change[:, index])
                result[:, index] = np.clip(result[:, index], 0, None)
        return result


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "Standardizer":
        selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
        self.mean = np.nanmean(selected, axis=0)
        scale = np.nanstd(selected, axis=0)
        self.scale = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Standardizer has not been fitted")
        return (np.asarray(values) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Standardizer has not been fitted")
        return np.asarray(values) * self.scale + self.mean


@dataclass
class UnitAwareChangeTransformer:
    """Fit-only, invertible transforms for heterogeneous census indicators."""

    units: tuple[str, ...]
    epsilon: float = 1e-4
    base_median: np.ndarray | None = None
    center: np.ndarray | None = None
    scale: np.ndarray | None = None

    def _forward(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        result = source.copy()
        for index, unit in enumerate(self.units):
            column = source[:, index]
            if unit == "count":
                result[:, index] = np.log1p(np.clip(column, 0, None))
            elif unit == "rate":
                clipped = np.clip(column, self.epsilon, 1 - self.epsilon)
                result[:, index] = np.log(clipped) - np.log1p(-clipped)
            elif unit == "average":
                result[:, index] = np.log(np.clip(column, self.epsilon, None))
        return result

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        result = source.copy()
        for index, unit in enumerate(self.units):
            column = np.clip(source[:, index], -20, 20)
            if unit == "count":
                result[:, index] = np.expm1(column).clip(0)
            elif unit == "rate":
                result[:, index] = 1 / (1 + np.exp(-column))
            elif unit == "average":
                result[:, index] = np.exp(column).clip(self.epsilon)
        return result

    def fit(self, values_2010: np.ndarray, values_2022: np.ndarray, fit_mask: np.ndarray) -> "UnitAwareChangeTransformer":
        selected = np.asarray(fit_mask, dtype=bool)
        transformed_2010 = self._forward(values_2010)
        transformed_2022 = self._forward(values_2022)
        medians = np.nanmedian(transformed_2010[selected], axis=0)
        self.base_median = np.where(np.isfinite(medians), medians, 0.0)
        base = self.impute_base_transformed(transformed_2010)
        changes = transformed_2022 - base
        center = np.nanmedian(changes[selected], axis=0)
        self.center = np.where(np.isfinite(center), center, 0.0)
        q25 = np.nanpercentile(changes[selected], 25, axis=0)
        q75 = np.nanpercentile(changes[selected], 75, axis=0)
        robust = (q75 - q25) / 1.349
        fallback = np.nanstd(changes[selected], axis=0)
        scale = np.where(np.isfinite(robust) & (robust > 1e-6), robust, fallback)
        self.scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        return self

    def impute_base_transformed(self, transformed_2010: np.ndarray) -> np.ndarray:
        if self.base_median is None:
            raise RuntimeError("UnitAwareChangeTransformer has not been fitted")
        base = np.asarray(transformed_2010, dtype=float).copy()
        rows, columns = np.where(~np.isfinite(base))
        base[rows, columns] = self.base_median[columns]
        return base

    def base_transformed(self, values_2010: np.ndarray) -> np.ndarray:
        return self.impute_base_transformed(self._forward(values_2010))

    def transform(self, values_2010: np.ndarray, values_2022: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("UnitAwareChangeTransformer has not been fitted")
        changes = self._forward(values_2022) - self.base_transformed(values_2010)
        return (changes - self.center) / self.scale

    def inverse(self, transformed_change: np.ndarray, values_2010: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("UnitAwareChangeTransformer has not been fitted")
        changed = np.asarray(transformed_change) * self.scale + self.center
        return self._inverse(self.base_transformed(values_2010) + changed)

    def inverse_torch(self, transformed_change: torch.Tensor, base_transformed: torch.Tensor) -> torch.Tensor:
        if self.center is None or self.scale is None:
            raise RuntimeError("UnitAwareChangeTransformer has not been fitted")
        center = torch.as_tensor(self.center, dtype=transformed_change.dtype, device=transformed_change.device)
        scale = torch.as_tensor(self.scale, dtype=transformed_change.dtype, device=transformed_change.device)
        latent = base_transformed + transformed_change * scale + center
        columns: list[torch.Tensor] = []
        for index, unit in enumerate(self.units):
            column = latent[:, index].clamp(-20, 20)
            if unit == "count":
                column = torch.expm1(column).clamp_min(0)
            elif unit == "rate":
                column = torch.sigmoid(column)
            elif unit == "average":
                column = torch.exp(column).clamp_min(self.epsilon)
            columns.append(column)
        return torch.stack(columns, dim=-1)
