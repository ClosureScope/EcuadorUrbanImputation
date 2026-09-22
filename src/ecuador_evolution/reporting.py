from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .config import Settings


def write_data_report(settings: Settings) -> Path:
    processed = settings.data_path("processed")
    summary_path = processed / "preparation_summary.json"
    validation_path = settings.data_path("artifacts") / "validation-summary.json"
    if not summary_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run prepare and validate before report")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    coverage = pd.DataFrame(summary.get("coverage", []))
    lines = [
        "# Ecuador census preparation report",
        "",
        f"Prepared {summary['rows']:,} long-form rows for {summary['sectors']:,} target sectors, "
        f"{summary['indicators']} indicators, cities {', '.join(summary['cities'])}, and years {summary['years']}.",
        "",
        f"Mandatory validation: **{'PASS' if validation['passed'] else 'FAIL'}**.",
        "",
        "## Validation checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}: {check['detail']}"
        for check in validation["checks"]
    )
    if not coverage.empty:
        lines.extend(["", "## Indicator coverage", "", coverage.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "The result describes change across the 2010–2022 intercensal interval, not an annual trend. "
            "The 2010 de-facto and 2022 usual-residence concepts, anonymized 2022 geography, and area "
            "interpolation quality must accompany every substantive interpretation.",
            "",
        ]
    )
    output = settings.data_path("artifacts") / "data-report.md"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output
