from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecuador_evolution.archive import filtered_zip_csv_to_parquet, iter_zip_csv_chunks
from ecuador_evolution.download import Source, read_manifest, verify
from ecuador_evolution.identifiers import add_2010_record_ids, canton_code, sector_id
from ecuador_evolution.indicators import category_counts, occupied_private_dwellings, safe_ratio
from ecuador_evolution.indicators import read_indicator_dictionary
from ecuador_evolution.transforms import ChangeTransformer, UnitAwareChangeTransformer


def test_identifier_formatting() -> None:
    assert canton_code(1, 1) == "0101"
    assert sector_id(17, 1, 2, 3, 4) == "170102003004"
    with pytest.raises(ValueError):
        canton_code("x", 1)


def test_indicator_recode_and_zero_denominator() -> None:
    numerator, denominator, value = category_counts(pd.Series([1, 2, 9, None]), numerator_categories={1}, denominator_categories={1, 2})
    assert (numerator, denominator, value) == (1, 2, 0.5)
    assert np.isnan(safe_ratio([1], [0])[0])


def test_dictionary_has_year_specific_source_contracts() -> None:
    dictionary = read_indicator_dictionary(Path(__file__).parents[1] / "configs" / "indicators.toml")
    assert len(dictionary) == 13
    assert all(item.source_2010 and item.source_2022 for item in dictionary)
    water = next(item for item in dictionary if item.name == "public_water_share")
    assert water.source_2022["numerator_categories"] == [1, 2]


def test_change_transform_round_trip_and_train_isolation() -> None:
    a = np.array([[10.0, 0.2], [20.0, 0.5], [30.0, 0.7]])
    b = np.array([[12.0, 0.3], [25.0, 0.4], [99.0, 0.1]])
    train = np.array([True, True, False])
    transformer = ChangeTransformer(("count", "rate")).fit(a, b, train)
    original_stats = (transformer.means.copy(), transformer.scales.copy())
    changed = b.copy(); changed[~train] = -999
    second = ChangeTransformer(("count", "rate")).fit(a, changed, train)
    assert np.allclose(original_stats[0], second.means)
    assert np.allclose(original_stats[1], second.scales)
    assert np.allclose(transformer.inverse(transformer.transform(a, b), a), b)


def test_unit_aware_change_transform_round_trip_and_bounds() -> None:
    a = np.array([[10.0, 0.2, 2.5], [20.0, 0.5, 3.5], [np.nan, 0.7, 4.0]])
    b = np.array([[12.0, 0.3, 2.8], [25.0, 0.4, 3.2], [30.0, 0.1, 4.5]])
    train = np.array([True, True, False])
    transformer = UnitAwareChangeTransformer(("count", "rate", "average")).fit(a, b, train)
    changed = b.copy(); changed[~train] = [9_999, 0.999, 99]
    second = UnitAwareChangeTransformer(("count", "rate", "average")).fit(a, changed, train)
    assert np.allclose(transformer.center, second.center)
    assert np.allclose(transformer.scale, second.scale)
    restored = transformer.inverse(transformer.transform(a, b), a)
    assert np.allclose(restored, b)
    assert (restored[:, 0] >= 0).all()
    assert ((restored[:, 1] > 0) & (restored[:, 1] < 1)).all()
    assert (restored[:, 2] > 0).all()


def test_manifest_verification(tmp_path: Path) -> None:
    payload = b"official fixture"
    file = tmp_path / "x.zip"; file.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    source = Source("x", "https://example.invalid/x", "x.zip", len(payload), digest)
    assert verify(file, source)["sha256"] == digest
    manifest = tmp_path / "sources.toml"
    manifest.write_text(f'[[source]]\nname="x"\nurl="https://example.invalid/x"\nfilename="x.zip"\nbytes={len(payload)}\nsha256="{digest}"\n')
    assert read_manifest(manifest) == [source]


def test_archive_filter_before_parquet(tmp_path: Path) -> None:
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("people.csv", "I01,I02,value\n01,01,keep\n09,01,keep2\n02,01,drop\n")
    parquet = tmp_path / "filtered.parquet"
    summary = filtered_zip_csv_to_parquet(archive, parquet, city_codes={"0101", "0901"}, chunksize=1, encoding="utf-8", separator=",")
    assert summary["rows_kept"] == 2
    assert set(pd.read_parquet(parquet)["value"]) == {"keep", "keep2"}


def test_nested_archive_streaming_and_column_pruning(tmp_path: Path) -> None:
    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w") as nested:
        nested.writestr("province_population.csv", "I01,I02,keep,drop\n01,01,a,x\n02,01,b,y\n")
    archive = tmp_path / "province.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        outer.writestr("population.zip", nested_bytes.getvalue())
    chunks = list(
        iter_zip_csv_chunks(
            archive,
            chunksize=1,
            member_contains="population",
            encoding="utf-8",
            separator=",",
            usecols=["I01", "I02", "keep"],
        )
    )
    assert len(chunks) == 2
    assert chunks[0][0] == "population.zip!province_population.csv"
    assert list(chunks[0][1].columns) == ["I01", "I02", "keep"]


def test_derived_2010_record_identifiers() -> None:
    frame = pd.DataFrame(
        {"I01": [1], "I02": [1], "I03": [2], "I04": [3], "I05": [4], "I09": [5], "I10": [1]}
    )
    result = add_2010_record_ids(frame)
    assert result.loc[0, "sector_id"] == "010102003004"
    assert result.loc[0, "dwelling_id"] == "010102003004005"
    assert result.loc[0, "household_id"] == "0101020030040051"


def test_linked_dwelling_eligibility_is_unique() -> None:
    households = pd.DataFrame({"dwelling_id": ["a", "a", "b", "c"]})
    dwellings = pd.DataFrame(
        {
            "dwelling_id": ["a", "b", "c", "d"],
            "private": [1, 9, 1, 1],
            "occupied": [1, 1, 2, 1],
        }
    )
    eligible = occupied_private_dwellings(
        households,
        dwellings,
        dwelling_key="dwelling_id",
        private_field="private",
        occupied_field="occupied",
        private_values={1},
        occupied_values={1},
    )
    assert eligible["dwelling_id"].tolist() == ["a"]
