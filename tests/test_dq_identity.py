"""Чужой зелёный DQ не удостоверяет текущую выгрузку."""
from __future__ import annotations

import pandas as pd
import pytest

from main import main
from test_main import _fipa_pair, _metric, _trace_quality


@pytest.mark.parametrize('change', ['answer', 'unused_payload', 'missing_identity'])
def test_dq_must_identify_actual_source(change):
    frame = pd.DataFrame(_fipa_pair())
    frame['unused_payload'] = 'old'
    quality = _trace_quality(frame)
    if change == 'answer':
        frame.loc[1, 'output_text'] = frame.loc[1, 'output_text'].replace('Финальный ответ', 'Другой ответ')
    elif change == 'unused_payload':
        frame['unused_payload'] = 'new'
    else:
        quality.pop('source_dataset_id')
    result = main(frame, _metric(), traces_validation_result=quality)
    assert result['processing_report']['reason_code'] == 'dq_dataset_unverified'
    assert result['monitoring_umr'].empty


def test_current_dq_rule_counters_are_not_additive():
    frame = pd.DataFrame(_fipa_pair())
    quality = _trace_quality(frame)
    quality['quality'][0].update(rule_violations=7, blocking_rule_violations=3, advisory_rule_violations=2)
    result = main(frame, _metric(), traces_validation_result=quality)
    assert len(result['monitoring_umr']) == 1
    assert result['processing_report']['source_dataset_id'] == quality['source_dataset_id']


def test_schema_only_failure_is_a_verdict_not_a_parser_crash():
    frame = pd.DataFrame({'unknown': [1]})
    quality = _trace_quality(frame)
    quality.update(schema={'критичных нарушено': 1}, quality=None, criteria={}, readiness=[])
    result = main(frame, _metric(), traces_validation_result=quality)
    assert result['processing_report']['reason_code'] == 'dq_failed'
    assert result['processing_report']['data_readiness']['state'] == 'failed'


def test_foreign_red_dq_cannot_declare_current_source_bad():
    frame = pd.DataFrame(_fipa_pair())
    quality = _trace_quality(frame)
    quality['criteria']['K2']['tone'] = 'bad'
    frame.loc[1, 'output_text'] = 'другая выгрузка'
    result = main(frame, _metric(), traces_validation_result=quality)
    assert result['processing_report']['reason_code'] == 'dq_dataset_unverified'
    assert result['processing_report']['data_readiness']['state'] == 'insufficient'


def test_dq_execution_error_does_not_mean_bad_data():
    quality = {'contract_version': 'laim-traces-validation.v2', 'error': 'Ошибка чтения'}
    result = main(pd.DataFrame(), _metric(), traces_validation_result=quality)
    assert result['processing_report']['reason_code'] == 'dq_not_computable'
    assert result['processing_report']['data_readiness']['state'] == 'insufficient'
