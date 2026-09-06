"""Идентичность содержимого таблицы: порядок строк и parquet-кодек несущественны."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from collections.abc import Iterable

import pandas as pd


def _value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            return ["float", str(value)]
        numerator, denominator = value.as_integer_ratio()
        return numerator if denominator == 1 else ["number", numerator, denominator]
    if isinstance(value, datetime):
        normalized = value.astimezone(timezone.utc) if value.tzinfo else value
        return ["datetime", normalized.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("dataset_identity: ключи вложенного объекта должны быть строками")
        return ["object", [[key, _value(value[key])] for key in sorted(value)]]
    if isinstance(value, (tuple, list)):
        return ["list", [_value(item) for item in value]]
    if hasattr(value, "tolist"):
        return _value(value.tolist())
    if hasattr(value, "item"):
        return _value(value.item())
    raise ValueError(f"dataset_identity: неподдерживаемый тип {type(value).__name__}")


class DatasetFingerprint:
    """Инкрементальный SHA-256; в памяти только 32 байта хеша на строку и накладные расходы Python."""

    def __init__(self) -> None:
        self.columns: tuple[str, ...] | None = None
        self.rows: list[bytes] = []

    def add(self, columns: Iterable[str], rows: Iterable[tuple]) -> None:
        names = tuple(columns)
        if len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
            raise ValueError("dataset_identity: имена колонок должны быть уникальными строками")
        ordered = tuple(sorted(names))
        if self.columns is not None and self.columns != ordered:
            raise ValueError("dataset_identity: части выгрузки имеют разные колонки")
        self.columns = ordered
        positions = [names.index(name) for name in ordered]
        for row in rows:
            encoded = json.dumps([_value(row[i]) for i in positions], ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode()
            self.rows.append(sha256(encoded).digest())

    def add_frame(self, frame: pd.DataFrame) -> None:
        self.add(frame.columns, frame.itertuples(index=False, name=None))

    def hexdigest(self) -> str:
        digest = sha256(json.dumps(self.columns, ensure_ascii=False).encode())
        for row in sorted(self.rows):
            digest.update(row)
        return "laim-dataset.v1:" + digest.hexdigest()


def frame_identity(frame: pd.DataFrame) -> str:
    fingerprint = DatasetFingerprint()
    fingerprint.add_frame(frame)
    return fingerprint.hexdigest()


def parquet_identity(source: str) -> str:
    from glob import glob
    from pathlib import Path
    import pyarrow.parquet as pq

    path = Path(source)
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [Path(p) for p in sorted(glob(source))]
    if not paths:
        raise ValueError(f"dataset_identity: parquet не найден: {source}")
    fingerprint = DatasetFingerprint()
    for path in paths:
        with pq.ParquetFile(path, pre_buffer=False) as file:
            fingerprint.add(file.schema_arrow.names, ())
            for batch in file.iter_batches(batch_size=512, use_threads=False):
                values = batch.to_pydict()
                fingerprint.add(values, zip(*values.values()))
    return fingerprint.hexdigest()
