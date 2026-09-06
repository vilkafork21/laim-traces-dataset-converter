"""Протокол, содержимое и полнота не зависят от правдоподобности текста."""
from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from main import main
from test_main import _fipa_pair, _metric, _trace_quality
from trace_dialogue import ExtractionConfig, extract_turns


def outgoing(row, **changes):
    payload = json.loads(row['output_text'])
    payload['outgoing'].update(changes)
    row['output_text'] = json.dumps(payload, ensure_ascii=False)


def convert(rows):
    frame = pd.DataFrame(rows)
    return main(frame, _metric(), traces_validation_result=_trace_quality(frame))


def test_explicit_foreign_conversation_cannot_inherit_incoming_id():
    entry, reply = _fipa_pair()
    outgoing(reply, conversation_id='other-conversation')
    result = convert([entry, reply])
    assert result['monitoring_umr'].empty
    assert result['processing_report']['status'] != 'complete'


@pytest.mark.parametrize('final_present', [False, True])
def test_agreement_is_not_a_terminal_answer(final_present):
    entry, reply = _fipa_pair()
    interim = deepcopy(reply)
    interim['span_id'] = 'interim'
    outgoing(interim, performative='agree', content={'message': [{'type': 'text', 'value': 'Обрабатываю запрос'}]})
    rows = [entry, interim, reply] if final_present else [entry, interim]
    result = convert(rows)
    if final_present:
        assert result['monitoring_umr'].output_answer.tolist() == ['Финальный ответ']
    else:
        assert result['monitoring_umr'].empty


def test_case_distinct_replies_are_conflicts_in_both_orders():
    entry, reply = _fipa_pair(answer='Token AbC')
    other = deepcopy(reply)
    other['span_id'] = 'other'
    outgoing(other, content={'message': [{'type': 'text', 'value': 'Token ABC'}]})
    for rows in ([entry, reply, other], [entry, other, reply]):
        result = convert(rows)
        assert result['monitoring_umr'].empty


def test_multipart_user_text_is_not_a_route_without_mapping():
    entry, reply = _fipa_pair()
    payload = json.loads(entry['input_text'])
    payload['message']['content']['message'] = [{'type': 'text', 'value': t} for t in ['Привет', 'Мне нужна помощь']]
    entry['input_text'] = json.dumps(payload, ensure_ascii=False)
    result = extract_turns(pd.DataFrame([entry, reply]), ExtractionConfig(
        observation_profile='fipa_external_reply_v1', external_party='agent_human', agent_id='CI00000001'))
    assert result.turns.input_query.tolist() == ['Привет\n\nМне нужна помощь']
    assert result.turns.route_label.tolist() == ['']


@pytest.mark.parametrize('position', [0, 2])
def test_route_message_part_is_selected_by_explicit_source(position):
    entry, reply = _fipa_pair()
    payload = json.loads(entry['output_text'])
    parts = ['Служебное пояснение', 'Вопрос']
    parts.insert(position, 'class X')
    payload['outgoing']['content']['message'] = [{'type': 'text', 'value': t} for t in parts]
    entry['output_text'] = json.dumps(payload, ensure_ascii=False)
    result = extract_turns(pd.DataFrame([entry, reply]), ExtractionConfig(
        observation_profile='fipa_external_reply_v1', external_party='agent_human', agent_id='CI00000001',
        route_source={'envelope': 'outgoing', 'field': 'message', 'part_index': position}))
    assert result.turns.route_label.tolist() == ['class X']
    assert result.turns.input_query.tolist() == ['Вопрос']


def test_missing_route_source_cannot_approve_route_evaluation():
    from measurement import approve_measurement, definition_id
    metric = _metric()
    metric['evaluation'].pop('route_source')
    metric['definition_id'] = definition_id(metric)
    metric = approve_measurement(metric, {'decision': 'approved', 'definition_id': metric['definition_id'],
        'reviewer': 'test-reviewer', 'approved_at': '2026-09-06T12:00:00Z'})
    frame = pd.DataFrame(_fipa_pair())
    result = main(frame, metric, traces_validation_result=_trace_quality(frame))
    assert not result['processing_report']['ready_for_scoring']
    assert not result['monitoring_umr'].evaluation_ready.any()


@pytest.mark.parametrize('performative', [None, 'subscribe', 'custom-progress'])
def test_unknown_or_absent_terminal_semantics_are_not_ready(performative):
    entry, reply = _fipa_pair()
    outgoing(reply, performative=performative)
    result = convert([entry, reply])
    assert result['monitoring_umr'].empty


@pytest.mark.parametrize('answer', ['ok', 'Запрос'])
def test_aef_boundary_keeps_short_and_echo_answers(answer):
    from test_main import _span
    row = _span('t', 's', 'Запрос', answer, 10)
    result = extract_turns(pd.DataFrame([row]), ExtractionConfig(
        observation_profile='aef_boundary_v1', agent_id='CI00000001'))
    assert result.turns.agent_response.tolist() == [answer]


def test_identical_state_replicas_have_stable_turn_identity():
    from test_trace_dialogue import _state_json_span
    first = _state_json_span(span_id='first', start_time_ns=10)
    last = _state_json_span(span_id='last', start_time_ns=20)
    config = ExtractionConfig(observation_profile='state_single_request_v1', agent_id='CI00000001')
    before = extract_turns(pd.DataFrame([first, last]), config).turns
    after = extract_turns(pd.DataFrame([last, first]), config).turns
    pd.testing.assert_frame_equal(before, after)


def test_empty_terminal_state_is_not_removed_from_candidate_count():
    from test_trace_dialogue import _state_json_span
    good = _state_json_span()
    failed = _state_json_span(trace_id='failed', answer='')
    result = extract_turns(pd.DataFrame([good, failed]), ExtractionConfig(
        observation_profile='state_single_request_v1', agent_id='CI00000001'))
    assert result.report['candidate_turn_keys'] == 2
    assert result.report['complete_turns'] == 1
    assert result.report['extraction_coverage'] == .5


def test_same_protocol_keys_in_different_sessions_are_not_replicas():
    first = _fipa_pair(trace_suffix='first', start_time=10)
    first[1]['start_time_ns'] = 100
    second = _fipa_pair(trace_suffix='second', start_time=12)
    for row in second:
        row['session_id'] = 'other-session'
    result = convert([*first, *second])
    assert result['monitoring_umr'].empty
    extraction = result['processing_report']['extraction']
    assert extraction['conflicting_entry_keys'] == extraction['conflicting_exit_keys'] == 1
