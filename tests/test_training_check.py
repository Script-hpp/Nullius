import pytest
from repro_agent.harness.training_check import checkpoint_issues
from test_loop_fakes import make_orch, FakeSandbox
from repro_agent.tasks import cora_task


def program(snapshot):
    return f'for epoch in range(10):\n    optimizer.step()\n    best = {snapshot}\nmodel.load_state_dict(best)\n'


@pytest.mark.parametrize('snapshot', ['model.state_dict()', 'model.state_dict().copy()', 'dict(model.state_dict())', 'copy.copy(model.state_dict())'])
def test_shallow_best_checkpoints_are_rejected(snapshot):
    assert checkpoint_issues(program(snapshot))


@pytest.mark.parametrize('snapshot', ['copy.deepcopy(model.state_dict())', '{k: v.detach().clone() for k,v in model.state_dict().items()}'])
def test_independent_weights_are_allowed(snapshot):
    assert checkpoint_issues(program(snapshot)) == []


def test_immediate_serialization_and_nontraining_restore_are_allowed():
    assert checkpoint_issues('for epoch in range(10):\n    optimizer.step()\n    torch.save(model.state_dict(), "checkpoint.pt")') == []
    assert checkpoint_issues('best = model.state_dict()\nmodel.load_state_dict(best)') == []


def test_other_scopes_and_scheduler_steps_are_not_treated_as_parameter_updates():
    code='def save():\n    for epoch in range(10):\n        optimizer.step()\n        best=model.state_dict()\ndef restore():\n    model.load_state_dict(best)\n'
    assert checkpoint_issues(code)==[]
    assert checkpoint_issues(program('model.state_dict()').replace('optimizer.step()', 'scheduler.step()'))==[]


def test_bad_checkpoint_stops_before_any_sandbox_command():
    sb = FakeSandbox()
    orch, _ = make_orch(sb)
    _, out = orch._attempt('snapshot', program('model.state_dict().copy()'), 0, cora_task(), 'accuracy')
    assert out['status'] == 'ungueltig' and out['static']
    assert 'best epoch' in out['detail'] and not sb.cmds
