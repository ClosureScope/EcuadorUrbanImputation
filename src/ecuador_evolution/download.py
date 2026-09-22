from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    bytes: int | None = None
    sha256: str | None = None


def read_manifest(path: str | Path) -> list[Source]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    sources = [Source(**item) for item in raw.get("source", [])]
    if not sources:
        raise ValueError(f"No [[source]] entries in {path}")
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Source names must be unique")
    return sources


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, source: Source) -> dict[str, object]:
    actual_size = path.stat().st_size
    if source.bytes and actual_size != source.bytes:
        raise IOError(f"{source.name}: expected {source.bytes} bytes, found {actual_size}")
    actual_sha = sha256_file(path)
    if source.sha256 and actual_sha.lower() != source.sha256.lower():
        raise IOError(f"{source.name}: SHA-256 mismatch ({actual_sha})")
    return {"bytes": actual_size, "sha256": actual_sha}


def download_one(source: Source, destination: Path, *, timeout: int = 60) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / source.filename
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists():
        metadata = verify(final_path, source)
        return {**asdict(source), **metadata, "path": str(final_path), "status": "verified-existing"}

    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(source.url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        # A server may ignore Range; restart rather than append a complete file.
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        total_header = int(response.headers.get("content-length", 0))
        total = offset + total_header if total_header else source.bytes
        with part_path.open(mode) as stream, tqdm(
            total=total, initial=offset, unit="B", unit_scale=True, desc=source.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    progress.update(len(chunk))
            stream.flush()
            os.fsync(stream.fileno())
    metadata = verify(part_path, source)
    part_path.replace(final_path)
    return {**asdict(source), **metadata, "path": str(final_path), "status": "downloaded"}


def download_all(sources: Iterable[Source], destination: Path) -> list[dict[str, object]]:
    records = [download_one(source, destination) for source in sources]
    manifest_path = destination / "download-metadata.json"
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records
