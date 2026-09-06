"""Транспорт таблиц, принимаемый dataframe-портом SberDS."""

from __future__ import annotations

import glob
import io
import os
from collections.abc import Callable

import pandas as pd
import pyarrow.parquet as pq
from pyarrow.types import is_signed_integer, is_unsigned_integer

from trace_dialogue import candidate_mask, state_payload_mask
from dataset_identity import DatasetFingerprint

_TABLE_READERS: dict[bytes, Callable[..., pd.DataFrame]] = {
    b"PAR1": pd.read_parquet,
    b"PK\x03\x04": pd.read_excel,
}


_TRACE_COLUMNS = (
    "agent_id", "solution_version", "session_id", "trace_id", "span_id",
    "aef_kind", "span_name", "start_time_ns", "input_text", "output_text",
)


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, [name for name in _TRACE_COLUMNS if name in frame]].copy()
    if {"aef_kind", "span_name", "input_text", "output_text"} <= set(result):
        needed = candidate_mask(result) | state_payload_mask(result)
        result.loc[~needed, ["input_text", "output_text"]] = ""
    return result


def _integer_dtype(arrow_type):
    if is_signed_integer(arrow_type):
        return pd.Int64Dtype()
    if is_unsigned_integer(arrow_type):
        return pd.UInt64Dtype()
    return None


def _parquet(source: object, fingerprint: DatasetFingerprint) -> pd.DataFrame:
    # ponytail: результат и метаданные остаются в памяти; при их росте нужен дисковый turn-store.
    with pq.ParquetFile(source, pre_buffer=False) as file:
        columns = file.schema_arrow.names
        fingerprint.add(columns, ())
        frames = []
        for batch in file.iter_batches(batch_size=512, use_threads=False):
            values = batch.to_pydict()
            fingerprint.add(values, zip(*values.values()))
            frame = batch.select([name for name in _TRACE_COLUMNS if name in columns]).to_pandas(types_mapper=_integer_dtype)
            frames.append(_compact(frame))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


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


def _read_file(path: str, port: str, fingerprint: DatasetFingerprint) -> pd.DataFrame:
    reader = _reader_for(_signature(path))
    if reader is None:
        raise ValueError(f"Порт {port}: файл {path} не parquet и не xlsx")
    if reader is pd.read_parquet:
        return _parquet(path, fingerprint)
    frame = reader(path)
    fingerprint.add_frame(frame)
    return _compact(frame)


def _read_table(value: object, port: str, fingerprint: DatasetFingerprint) -> pd.DataFrame:
    """Прочитать DataFrame, байты, файл или каталог с частями таблицы."""
    if isinstance(value, pd.DataFrame):
        fingerprint.add_frame(value)
        return _compact(value)
    if isinstance(value, (bytes, bytearray)):
        blob = bytes(value)
        reader = _reader_for(blob[:8])
        if reader is None:
            raise ValueError(f"Порт {port}: байты не parquet и не xlsx")
        if reader is pd.read_parquet:
            return _parquet(io.BytesIO(blob), fingerprint)
        frame = reader(io.BytesIO(blob))
        fingerprint.add_frame(frame)
        return _compact(frame)
    if isinstance(value, str) and os.path.isfile(value):
        return _read_file(value, port, fingerprint)
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
        return pd.concat([_read_file(path, port, fingerprint) for path in parts], ignore_index=True)
    raise TypeError(f"Порт {port} отдал {type(value).__name__} — это не таблица")


def read_table(value: object, port: str) -> pd.DataFrame:
    """Прочитать и сократить spans, сохранив идентичность полного исходного содержимого."""
    fingerprint = DatasetFingerprint()
    frame = _read_table(value, port, fingerprint)
    frame.attrs["source_dataset_id"] = fingerprint.hexdigest()
    return frame
