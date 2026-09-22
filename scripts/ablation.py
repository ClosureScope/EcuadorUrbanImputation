#!/usr/bin/env python3
"""Compare the GAT reconstructor with and without spatial neighbors.

Run from the repository root: uv run python scripts/ablation.py --device cpu
Uses the selected full-experiment parameters, 11 folds and three seeds.
Outputs are separate from the reference results in results/metrics.csv.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import ecuador_evolution

PARAMS = {"width": 128, "dropout": 0.15, "learning_rate": 0.001}
SEEDS = [1605, 1606, 1607]
# 与主结果表 metrics.csv 完全一致的 11 折构成
FOLDS = {
    "random-mask": ["10%", "20%", "30%"],
    "held-city": ["0101", "0901", "1701"],
    "block-mask": [f"20%-block-{i}" for i in range(5)],
}


def _slug(text: str) -> str:
    return text.replace("%", "pct").replace("/", "-").replace(" ", "_")


def summarize_predictions(truth, prediction, test_mask, indicators, indicator_std) -> dict:
    """Use the same per-indicator metrics as the training/evaluation pipeline."""
    from ecuador_evolution.metrics import evaluate_predictions

    table = evaluate_predictions(
        truth[test_mask], prediction[test_mask], indicators,
        indicator_sigma=dict(zip(indicators, indicator_std)),
    ).set_index("indicator")
    macro = table.loc["macro"]
    return {
        "MAE_z": float(macro["mae_z"]),
        "RMSE_z": float(macro["rmse_z"]),
        "R2": float(macro["r2"]),
        "per_indicator": table.drop(index="macro")[["mae", "rmse", "r2"]].to_dict(orient="index"),
    }


def experiment_fingerprint(settings, data) -> str:
    """Reject cached scores from other inputs, settings, or model implementations."""
    import hashlib
    import numpy as np

    digest = hashlib.sha256()
    metadata = {"schema": 2, "settings": settings.values, "params": PARAMS,
                "seeds": SEEDS, "folds": FOLDS, "indicators": data.indicators,
                "device": settings.device}
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode())
    for field in dataclasses.fields(data):
        value = getattr(data, field.name)
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(f"{field.name}:{array.shape}:{array.dtype}".encode())
            digest.update(array.tobytes())
    for source in sorted(Path(ecuador_evolution.__file__).parent.glob("*.py")):
        digest.update(source.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Spatial ablation for the GAT reconstructor")
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "reconstruction.toml"),
        help="与 train.py 相同的实验配置",
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "artifacts" / "ablation",
                    help="Directory for ablation runs and ablation_metrics.csv")
    args = ap.parse_args()

    import numpy as np

    from ecuador_evolution.config import load_settings
    from ecuador_evolution import experiment as E
    from ecuador_evolution.experiment import SplitSpec

    settings = load_settings(args.config, device=args.device)
    data = E._load(settings)
    print(f"loaded {len(data.frame)} sectors, {len(data.indicators)} indicators", flush=True)

    # 一把固定标尺：各指标在全部 15,949 个 sector 上的 2022 全局标准差
    # （与划分无关，与主表 metrics.csv 归一化口径一致）。
    truth = data.values_2022
    indicator_std = np.nanstd(truth, axis=0)
    indicator_std = np.where(indicator_std > 0, indicator_std, 1.0)

    def normalized(prediction: np.ndarray, test_mask: np.ndarray) -> dict:
        return summarize_predictions(truth, prediction, test_mask, data.indicators, indicator_std)

    def fit(run_data, fit_mask, val_mask, test_mask, seed):
        _, prediction, _ = E._fit_architecture(
            "gat", run_data, fit_mask, val_mask, test_mask, dict(PARAMS), settings, seed
        )
        return prediction

    run_dir = args.output_dir / "ablation_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = experiment_fingerprint(settings, data)
    all_runs = {}
    for protocol, folds in FOLDS.items():
        for fold in folds:
            for seed in SEEDS:
                cache = run_dir / f"run_{protocol}_{_slug(fold)}_seed{seed}.json"
                if cache.exists():
                    try:
                        cached = json.loads(cache.read_text())
                    except (OSError, ValueError):
                        cached = {}
                    if (cached.get("fingerprint") == fingerprint
                            and all("per_indicator" in cached.get(arm, {}) for arm in ("full", "ablated"))):
                        all_runs[(protocol, fold, seed)] = cached
                        continue
                split = SplitSpec(protocol, fold, seed)
                fit_mask, val_mask, test_mask = E._split_masks(data, settings, split)

                full = normalized(fit(data, fit_mask, val_mask, test_mask, seed), test_mask)

                mask = data.neighbor_mask.copy()
                mask[:] = False  # 切断全部邻居
                ablated_data = dataclasses.replace(data, neighbor_mask=mask)
                ablated = normalized(
                    fit(ablated_data, fit_mask, val_mask, test_mask, seed), test_mask
                )

                rec = {"protocol": protocol, "fold": fold, "seed": seed,
                       "full": full, "ablated": ablated, "fingerprint": fingerprint}
                cache.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
                all_runs[(protocol, fold, seed)] = rec
                print(f"[{protocol}/{fold} seed{seed}] "
                      f"FULL MAE_z={full['MAE_z']:.4f}  ABL MAE_z={ablated['MAE_z']:.4f}",
                      flush=True)

    # 聚合消融（ablated）臂为协议级均值，输出 metrics.csv 口径的 gat-ablation 行。
    # full 臂协议级结果即 metrics.csv 的 gat 行，不重复落盘；此处仅打印邻居净贡献供核对。
    def macro(arm, key, keys):
        return sum(all_runs[k][arm][key] for k in keys) / len(keys)

    def raw_macro(arm, field, keys):
        # 逐 run 先对 13 指标等权平均，再对 run 平均
        vals = []
        for k in keys:
            pi = all_runs[k][arm]["per_indicator"]
            vals.append(sum(v[field] for v in pi.values()) / len(pi))
        return sum(vals) / len(vals)

    print("\n=== gat-ablation 行（写入 ablation_metrics.csv，schema: "
          "protocol,model,MAE_z,RMSE_z,R2,raw_macro_MAE,raw_macro_RMSE）===")
    summary_rows = []
    for protocol, folds in FOLDS.items():
        keys = [(protocol, f, s) for f in folds for s in SEEDS if (protocol, f, s) in all_runs]
        if not keys:
            continue
        worse_mae = sum(all_runs[k]["ablated"]["MAE_z"] > all_runs[k]["full"]["MAE_z"] for k in keys)
        row = (f"{protocol},gat-ablation,"
               f"{macro('ablated','MAE_z',keys):.4f},{macro('ablated','RMSE_z',keys):.4f},"
               f"{macro('ablated','R2',keys):.4f},"
               f"{raw_macro('ablated','mae',keys):.4f},{raw_macro('ablated','rmse',keys):.4f}")
        summary_rows.append(row.split(","))
        print(row + f"    # {len(keys)} 折×种子; 切断邻居后 MAE_z 变差的占 {worse_mae}/{len(keys)}")

    summary_path = run_dir.parent / "ablation_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["protocol", "model", "MAE_z", "RMSE_z", "R2", "raw_macro_MAE", "raw_macro_RMSE"])
        writer.writerows(summary_rows)
    print(f"Saved ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
