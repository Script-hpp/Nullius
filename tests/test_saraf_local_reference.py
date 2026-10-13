import ast
import importlib.util
import numpy as np
import pytest
from pathlib import Path

from repro_agent.harness.code_check import find_violations
from repro_agent.harness.training_check import checkpoint_issues


def test_full_reference_adapter_is_syntactically_valid_and_passes_existing_product_guards():
    root=Path(__file__).resolve().parents[1]
    path=root/'scripts/review/saraf_contract_full.py'
    spec=importlib.util.spec_from_file_location('saraf_reference_driver',path)
    driver=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    code=driver.adapter()
    ast.parse(code)
    assert find_violations(code)==[]
    assert checkpoint_issues(code)==[]
    assert 'copy.deepcopy(model.state_dict())' in code
    assert "mode='valid'" in code and "mode='train'" in code
    assert 'for epoch in range(10)' in code


@pytest.mark.parametrize('indices', [np.array(8),np.array([8]),[8],[7,8]])
def test_last_partial_reference_batch_preserves_batch_dimension(indices):
    from scripts.review.saraf_contract_full import take_batch
    values=np.arange(9*720*7).reshape(9,720,7)
    result=take_batch(values,indices)
    assert result.ndim==3 and result.shape[1:]==(720,7)
    assert np.array_equal(result[-1],values[8])
