from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .identifiers import canton_code


def csv_members(path: str | Path) -> list[str]:
    """List direct CSV/text members (kept for the public helper contract)."""
    with zipfile.ZipFile(path) as archive:
        return [name for name in archive.namelist() if name.lower().endswith((".csv", ".txt"))]


@contextmanager
def _nested_zip(outer: zipfile.ZipFile, member: str):
    """Open a nested ZIP without leaving an extracted archive in the data tree."""
    with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as spool:
        with outer.open(member) as source:
            shutil.copyfileobj(source, spool)
        spool.seek(0)
        with zipfile.ZipFile(spool) as nested:
            yield nested


def _member_streams(
    archive: zipfile.ZipFile,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, object]]:
    for name in archive.namelist():
        lower = name.lower()
        qualified = f"{prefix}!{name}" if prefix else name
        if lower.endswith((".csv", ".txt")):
            with archive.open(name) as raw:
                yield qualified, raw
        elif lower.endswith(".zip"):
            with _nested_zip(archive, name) as nested:
                yield from _member_streams(nested, prefix=qualified)


def iter_zip_csv_chunks(
    archive_path: str | Path,
    *,
    chunksize: int = 200_000,
    member_contains: str | None = None,
    encoding: str = "latin-1",
    separator: str | None = None,
    usecols: Sequence[str] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield chunks from direct or nested CSV members of an archive."""
    needle = member_contains.lower() if member_contains else None
    with zipfile.ZipFile(archive_path) as archive:
        for name, raw in _member_streams(archive):
            if needle and needle not in name.lower():
                continue
            text = io.TextIOWrapper(raw, encoding=encoding, errors="replace", newline="")
            options: dict[str, object]
            if separator:
                options = {"sep": separator}
            else:
                options = {"sep": None, "engine": "python"}
            reader = pd.read_csv(
                text,
                chunksize=chunksize,
                dtype=str,
                usecols=list(usecols) if usecols else None,
                **options,
            )
            for chunk in reader:
                yield name, chunk


def filtered_zip_csv_to_parquet(
    archive_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    city_codes: Iterable[str],
    chunksize: int = 200_000,
    province_field: str = "I01",
    canton_field: str = "I02",
    member_contains: str | None = None,
    encoding: str = "latin-1",
    separator: str | None = None,
    columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Stream ZIP members, filtering cantons before an atomic pruned Parquet write."""
    selected = {str(item).zfill(4) for item in city_codes}
    paths = [archive_path] if isinstance(archive_path, (str, Path)) else list(archive_path)
    writer: pq.ParquetWriter | None = None
    read_rows = kept_rows = members = 0
    member_names: list[str] = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    requested = (
        list(dict.fromkeys([*columns, province_field, canton_field]))
        if columns is not None
        else None
    )
    try:
        for path in paths:
            seen_in_archive: set[str] = set()
            for name, chunk in iter_zip_csv_chunks(
                path,
                chunksize=chunksize,
                member_contains=member_contains,
                encoding=encoding,
                separator=separator,
                usecols=requested,
            ):
                qualified = f"{Path(path).name}:{name}"
                if qualified not in seen_in_archive:
                    seen_in_archive.add(qualified)
                    members += 1
                    member_names.append(qualified)
                read_rows += len(chunk)
                if province_field not in chunk or canton_field not in chunk:
                    continue
                normalized = pd.Series(
                    [
                        canton_code(province, canton)
                        for province, canton in zip(chunk[province_field], chunk[canton_field], strict=True)
                    ],
                    index=chunk.index,
                )
                filtered = chunk.loc[normalized.isin(selected)].copy()
                if filtered.empty:
                    continue
                filtered["source_archive"] = Path(path).name
                filtered["source_member"] = name
                kept_rows += len(filtered)
                table = pa.Table.from_pandas(filtered, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
            temporary.replace(output)
        elif output.exists():
            output.unlink()
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    return {
        "members": members,
        "member_names": member_names,
        "rows_read": read_rows,
        "rows_kept": kept_rows,
    }
