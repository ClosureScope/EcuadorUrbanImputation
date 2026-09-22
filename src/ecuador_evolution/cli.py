from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings
from .download import download_all, read_manifest
from .prepare import prepare
from .reporting import write_data_report
from .validation import validate
from .workflow import run_experiment


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ecuador-evolution")
    subcommands = root.add_subparsers(dest="command", required=True)
    for name in ("download", "prepare", "validate", "train", "evaluate", "report"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", default="configs/reconstruction.toml")
        command.add_argument("--data-root")
        command.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
        command.add_argument("--seed", type=int)
        command.add_argument("--parallel-backend", choices=("nvidia-mps", "process", "serial"))
        command.add_argument("--gpu-workers", help="GPU worker count or 'auto'")
        command.add_argument("--cpu-workers", type=int)
        calibration = command.add_mutually_exclusive_group()
        calibration.add_argument("--calibration", dest="calibration", action="store_true")
        calibration.add_argument("--no-calibration", dest="calibration", action="store_false")
        command.set_defaults(calibration=None)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    settings = load_settings(
        arguments.config,
        data_root=arguments.data_root,
        device=arguments.device,
        seed=arguments.seed,
        parallel_backend=arguments.parallel_backend,
        gpu_workers=arguments.gpu_workers,
        cpu_workers=arguments.cpu_workers,
        calibration=arguments.calibration,
    )
    if arguments.command == "download":
        records = download_all(read_manifest(settings.project_path("manifest")), settings.data_path("raw"))
        print(json.dumps(records, indent=2))
    elif arguments.command == "prepare":
        print(json.dumps(prepare(settings), indent=2))
    elif arguments.command == "validate":
        report = validate(settings)
        print(json.dumps({"passed": report.passed, "checks": report.checks}, indent=2))
    elif arguments.command == "report" and settings.values.get("task") == "shared":
        print(Path(write_data_report(settings)))
    else:
        result = run_experiment(arguments.command, settings)
        if result is not None:
            print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
