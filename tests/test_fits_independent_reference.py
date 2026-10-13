import ast
import importlib.util
from pathlib import Path
from repro_agent.harness.code_check import find_violations
from repro_agent.harness.training_check import checkpoint_issues


def test_fits_reference_uses_existing_guard_and_forecast_only_author_training_mode():
    path=Path(__file__).resolve().parents[1]/'scripts/review/fits_independent_reference.py'
    spec=importlib.util.spec_from_file_location('fits_reference_driver',path)
    driver=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    ADAPTER=driver.ADAPTER
    ast.parse(ADAPTER)
    assert find_violations(ADAPTER)==[] and checkpoint_issues(ADAPTER)==[]
    assert 'copy.deepcopy(model.state_dict())' in ADAPTER
    assert "length,horizon,batch=360,96,128" in ADAPTER
    assert 'cut_freq=106' in ADAPTER
    assert 'output[:,-horizon:]-y' in ADAPTER
    assert "def predict(history):" in ADAPTER
