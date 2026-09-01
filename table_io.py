"""Транспорт таблиц, принимаемый dataframe-портом SberDS."""

from __future__ import annotations

import glob
import io
import os
from collections.abc import Callable

import pandas as pd

_TABLE_READERS: dict[bytes, Callable[..., pd.DataFrame]] = {
    b"PAR1": pd.read_parquet,
    b"PK\x03\x04": pd.read_excel,
}


def _reader_for(head: bytes) -> Callable[..., pd.DataFrame] | None:
    return next(
        (
            reader
            for signature, reader in _TABLE_READERS.items()
            if head.startswith(signature)
        ),
        None,
    )


def _signature(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(8)


def _read_file(path: str, port: str) -> pd.DataFrame:
    reader = _reader_for(_signature(path))
    if reader is None:
        raise ValueError(f"Порт {port}: файл {path} не parquet и не xlsx")
    return reader(path)


def read_table(value: object, port: str) -> pd.DataFrame:
    """Прочитать DataFrame, байты, файл или каталог с частями таблицы."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (bytes, bytearray)):
        blob = bytes(value)
        reader = _reader_for(blob[:8])
        if reader is None:
            raise ValueError(f"Порт {port}: байты не parquet и не xlsx")
        return reader(io.BytesIO(blob))
    if isinstance(value, str) and os.path.isfile(value):
        return _read_file(value, port)
    if isinstance(value, str) and os.path.isdir(value):
        parts = [
            path
            for path in sorted(
                glob.glob(os.path.join(value, "**", "*"), recursive=True)
            )
            if os.path.isfile(path) and _reader_for(_signature(path)) is not None
        ]
        if not parts:
            raise FileNotFoundError(
                f"Порт {port}: в каталоге {value} нет parquet или xlsx"
            )
        return pd.concat([_read_file(path, port) for path in parts], ignore_index=True)
    raise TypeError(f"Порт {port} отдал {type(value).__name__} — это не таблица")
