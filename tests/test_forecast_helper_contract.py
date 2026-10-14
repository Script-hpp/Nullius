"""The agent-facing helper example must execute against the real API."""
import re
import sys

import numpy as np
import pytest

from repro_agent.tasks import etth1_task
from test_io_helper import _load_helper


@pytest.mark.parametrize('calendar',[False,True])
def test_agent_contract_example_uses_named_real_windows(tmp_path,monkeypatch,calendar):
    task=etth1_task().with_independent_selection().with_io_helper()
    blocks=re.findall(r'```python\n(.*?)```',task.code_notes,re.S)
    example=next((block for block in blocks if 'nio.forecast_data(length)' in block),None)
    assert example is not None,'Agent never receives the helper return schema/example'
    rows=np.arange(1800*7,dtype=np.float32).reshape(1800,7)
    data={'train':rows[:1200],'val':rows[1200:]}
    if calendar:data.update(train_calendar=np.zeros((1200,5)),val_calendar=np.zeros((600,5)))
    helper=_load_helper(task,tmp_path,monkeypatch,data,False)
    monkeypatch.setitem(sys.modules,'nullius_io',helper)   # the example imports it, as train.py does next to nullius_io.py
    namespace={'length':360}
    exec(example,namespace)
    assert namespace['train_x'].shape[1:]==(360,7)
    assert namespace['train_y'].shape[1:]==(96,7)
    assert namespace['val_x'].shape[1:]==(360,7)
    assert namespace['val_y'].shape[1:]==(96,7)
    np.testing.assert_array_equal(namespace['train_y'][0],rows[360:456])
    assert (namespace['train_calendar'] is not None)==calendar
    assert (namespace['val_calendar'] is not None)==calendar
    assert example in task.data_description and example in task.io_output
