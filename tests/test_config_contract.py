import pytest
from repro_agent.spec import Spec, SpecItem
from repro_agent.spec_check import config_check


def spec():
    return Spec(model='M',family='forecast',items=[SpecItem('lr','0.001','q','paper','paper',0.001,enforce=True)])


@pytest.mark.parametrize('write',["CONFIG['lr']=0.2","CONFIG.update({'lr':0.2})","CONFIG.update(lr=0.2)"])
def test_later_configuration_overwrite_cannot_pass_as_enforced(write):
    code="CONFIG={'lr':0.001}\n"+write+"\noptimizer=Adam(lr=CONFIG['lr'])\n"
    _,report=config_check(code,spec())
    assert any(c['key']=='lr' and c['literal']==0.2 for c in report['conflicts'])


def test_a_write_does_not_prove_that_the_parameter_is_read():
    _,report=config_check("CONFIG={'lr':0.001}\nCONFIG['lr']=0.001\n",spec())
    assert report['unused']==['lr']


def test_consistent_configuration_and_runtime_schedule_are_preserved():
    code="CONFIG={'lr':0.001}\nCONFIG.update({'other':4})\noptimizer=Adam(lr=CONFIG['lr'])\nfor epoch in range(10):\n    CONFIG['lr']*=0.5\n"
    new,report=config_check(code,spec())
    assert new==code and report['conflicts']==[] and report['unused']==[]


@pytest.mark.parametrize('code', [
    "CONFIG={'lr':0.001}\nCONFIG['lr']=0.2\noptimizer=Adam(lr=CONFIG['lr'])\n",
    "CONFIG={'lr':0.001}\nCONFIG.update(lr=0.2)\noptimizer=Adam(lr=CONFIG['lr'])\n",
    "CONFIG={'lr':0.2}\noptimizer=Adam(lr=CONFIG['lr'])\n",
    "CONFIG={'lr':0.001}\noptimizer=Adam(lr=0.2)\n",
])
def test_unrepaired_numeric_conflict_cannot_enter_the_sandbox(code):
    from test_loop_fakes import make_orch, FakeSandbox
    from repro_agent.tasks import cora_task
    sb=FakeSandbox()
    o,_=make_orch(sb)
    o.allow_test_feedback=False
    o.spec_mode,o.spec=True,spec()
    _,result=o._attempt('snapshot',code,0,cora_task(),'accuracy')
    assert result['status']=='ungueltig' and result['static']
    assert 'reproduction spec: lr' in result['detail'] and sb.cmds==[]


def test_consistent_numeric_config_can_still_run():
    from test_loop_fakes import make_orch, FakeSandbox
    from repro_agent.tasks import cora_task
    o,_=make_orch(FakeSandbox())
    o.allow_test_feedback=False
    o.spec_mode,o.spec=True,spec()
    _,result=o._attempt('snapshot',"CONFIG={'lr':0.001}\noptimizer=Adam(lr=CONFIG['lr'])\n",0,cora_task(),'accuracy')
    assert result['status']=='ok'
