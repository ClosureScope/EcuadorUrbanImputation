# Running the project

Run commands from the repository root. No sibling repository or external workspace is needed.

## Environment and quick check

```bash
uv sync --locked --extra dev
uv run pytest -q tests/test_reconstruction_experiment.py::test_gat_cpu_end_to_end_and_win_gate_schema
uv run pytest -q -m "not cuda"
```

The first test builds a small synthetic dataset and runs preparation, GAT training, baseline evaluation and report generation on CPU. It needs no census downloads.

The lockfile specifies Python 3.14 and PyTorch 2.13 with CUDA 13.0 binaries. CPU execution works without a GPU. GPU tests (`-m cuda`) require compatible hardware and drivers. Other Python/platform combinations have not been validated.

## Data and full experiment

```bash
uv run ecuador-evolution download
uv run ecuador-evolution prepare
uv run ecuador-evolution train --device cpu --parallel-backend serial
uv run ecuador-evolution evaluate --device cpu --parallel-backend serial
uv run ecuador-evolution report --device cpu --parallel-backend serial
```

The default config is `configs/reconstruction.toml`. The six official ZIP archives are downloaded to `data/raw/` and verified against SHA-256 values in `configs/sources.toml`. Full data downloads are approximately 1.7 GB. Intermediate tables go to `data/interim/`, model inputs to `data/processed/`, and generated experiment results to `artifacts/reconstruction/`.

Full training includes hyperparameter search and 11 folds × 3 seeds; it can take hours. Use `--device cuda` with a suitable parallel backend for GPU execution. A smaller CPU run on already prepared real data is available:

```bash
uv run ecuador-evolution train --config configs/smoke.toml
uv run ecuador-evolution evaluate --config configs/smoke.toml
```

This uses one seed, one parameter candidate and at most two epochs, writing to `artifacts/smoke/`. It still reads the full prepared data and is not a benchmark. For a check without downloads, use the synthetic test above.

## Spatial ablation

```bash
uv run python scripts/ablation.py --device cpu
```

Uses prepared inputs, the selected reference hyperparameters and all 11 folds × 3 seeds. Results go to `artifacts/ablation/`; caches are reused only when inputs, settings and code match. This does not overwrite the reference metrics.

## Export the data products

After downloading the raw inputs:

```bash
uv run python scripts/process_census.py --city quito
uv run python scripts/process_census.py --city cuenca
uv run python scripts/process_census.py --city guayaquil
uv run python scripts/merge_cities.py
```

These commands build the 2022 dataset with 19 indicators under `data/final/`. Per-city intermediates are stored in `data/interim/census/`. The pipeline's cross-year modeling panel has 13 harmonized indicators; export it after `ecuador-evolution prepare`:

```bash
uv run python scripts/export_panel.py
```

Generated datasets, environments and training artifacts are excluded from version control. The repository includes only source code, configuration, small metadata tables, reference metrics and selected maps.

## Reference results

`results/metrics.csv` and `assets/` preserve the existing experiment results. They are not outputs of the quick checks. The tables under `data/metadata/` describe that reference dataset, including observed quality statistics; they are not automatically recalculated for a changed dataset.

This reorganization has not rerun full raw-data preparation or the published multi-fold experiment. Data origins and the distinction between the reference work and this repository are recorded in [PROVENANCE.md](PROVENANCE.md).
