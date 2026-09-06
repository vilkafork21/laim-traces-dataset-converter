"""Пакетное чтение сохраняет извлечение и учёт всех исходных spans."""
from __future__ import annotations

import pandas as pd
import pytest

from table_io import read_table
from trace_dialogue import ExtractionConfig, extract_turns
from test_trace_dialogue import _fipa_entry, _fipa_exit, _span


def _frame():
    rows = [_fipa_entry(), _fipa_exit()]
    rows.append(_span(trace_id='internal', span_id='i', input_payload={'unrelated': 'x' * 10000},
                      output_payload={}, start_time_ns=5, aef_kind='chain', span_name='internal'))
    return pd.DataFrame(rows).assign(solution_version='v1', unused_blob='x' * 10000)


@pytest.mark.parametrize('transport', ['frame', 'bytes', 'file', 'parts'])
def test_ingestion_keeps_all_rows_but_drops_unused_payload(tmp_path, transport):
    source = _frame()
    value = source
    if transport == 'bytes':
        value = source.to_parquet(index=False)
    elif transport == 'file':
        value = str(tmp_path / 'input.parquet')
        source.to_parquet(value, index=False, row_group_size=1)
    elif transport == 'parts':
        value = str(tmp_path)
        for index in range(len(source)):
            source.iloc[[index]].to_parquet(tmp_path / f'{index}.parquet', index=False)
    actual = read_table(value, 'monitoring_traces')
    assert len(actual) == len(source)
    assert 'unused_blob' not in actual
    assert actual.loc[2, 'input_text'] == ''
    assert source.loc[2, 'input_text'] != ''
    cfg = ExtractionConfig(observation_profile='fipa_external_reply_v1', external_party='agent_human')
    before, after = extract_turns(source, cfg), extract_turns(actual, cfg)
    pd.testing.assert_frame_equal(after.turns, before.turns)
    assert after.report == before.report


def test_internal_fipa_markers_are_preserved():
    source = _frame()
    source.loc[:1, ['aef_kind', 'span_name']] = ['chain', 'internal']
    actual = read_table(source, 'monitoring_traces')
    assert actual.loc[:1, 'input_text'].tolist() == source.loc[:1, 'input_text'].tolist()


def test_state_profile_preserves_terminal_state_in_internal_span():
    source = pd.DataFrame([_span(trace_id='state', span_id='s', start_time_ns=1,
        aef_kind='chain', span_name='internal', output_payload={},
        input_payload={'stage': 'exit', 'message_to_user': 'Ответ',
                       'messages': [{'role': 'user', 'content': 'Вопрос'}]})])
    cfg = ExtractionConfig(observation_profile='state_single_request_v1')
    original = extract_turns(source, cfg)
    actual = extract_turns(read_table(source, 'monitoring_traces'), cfg)
    assert len(original.turns) == 1
    pd.testing.assert_frame_equal(actual.turns, original.turns)
    assert actual.report == original.report
