"""Идентичность связывает проверку с содержимым, а не числом строк или именем файла."""
from __future__ import annotations

import pandas as pd
import pytest

from dataset_identity import DatasetFingerprint, frame_identity, parquet_identity
from table_io import read_table


def test_identity_survives_order_codec_and_row_groups(tmp_path):
    frame = pd.DataFrame({'a': [1., None, 3.5], 'b': ['да', 'нет', 'да'],
                          'nested': [[1, 2], None, [3]]})
    expected = frame_identity(frame)
    assert frame_identity(frame.iloc[::-1, ::-1]) == expected
    for compression in ('snappy', 'zstd'):
        path = tmp_path / f'{compression}.parquet'
        frame.to_parquet(path, index=False, compression=compression, row_group_size=1)
        assert parquet_identity(str(path)) == expected
        assert read_table(str(path), 'monitoring_traces').attrs['source_dataset_id'] == expected


@pytest.mark.parametrize('change', ['value', 'duplicate', 'column'])
def test_identity_detects_content_and_multiplicity(change):
    frame = pd.DataFrame({'a': [1, 2]})
    other = frame.copy()
    if change == 'value':
        other.loc[0, 'a'] = 3
    elif change == 'duplicate':
        other = pd.concat([other, other.iloc[[0]]])
    else:
        other = other.rename(columns={'a': 'b'})
    assert frame_identity(frame) != frame_identity(other)


def test_parts_with_different_schema_are_rejected():
    fingerprint = DatasetFingerprint()
    fingerprint.add(['a'], [(1,)])
    with pytest.raises(ValueError, match='разные колонки'):
        fingerprint.add(['b'], [(1,)])


def test_empty_schema_and_nulls_are_stable(tmp_path):
    frame = pd.DataFrame({'a': pd.Series([], dtype='Int64')})
    path = tmp_path / 'empty.parquet'
    frame.to_parquet(path, index=False)
    assert frame_identity(frame) == parquet_identity(str(path))
    assert frame_identity(pd.DataFrame({'a': [None, 1.]})) == frame_identity(pd.DataFrame({'a': pd.Series([pd.NA, 1], dtype='Int64')}))


def test_nullable_nanosecond_identifiers_do_not_lose_precision(tmp_path):
    frame = pd.DataFrame({"start_time_ns": pd.Series([1700000000000000001, pd.NA], dtype="Int64")})
    path = tmp_path / 'nullable.parquet'
    frame.to_parquet(path, index=False)
    result = read_table(str(path), 'monitoring_traces')
    assert result.loc[0, 'start_time_ns'] == 1700000000000000001
    assert result.attrs['source_dataset_id'] == frame_identity(frame) == parquet_identity(str(path))


def test_raw_nested_nullable_integer_identity_precedes_pandas_conversion(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'nested.parquet'
    table = pa.table({'extra': pa.array([[2**60 + 1, None]], type=pa.list_(pa.int64())),
                      'trace_id': ['trace-1']})
    pq.write_table(table, path)
    result = read_table(str(path), 'monitoring_traces')
    assert result.attrs['source_dataset_id'] == parquet_identity(str(path))
    assert len(result) == 1
