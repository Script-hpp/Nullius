"""SARAF-style placeholders must be rejected before executing generated code."""
import pytest

from repro_agent.method_guard import MethodImplementationBlocked, placeholder_issues
from repro_agent.dossier import Claim, Dossier
from repro_agent.tasks import cora_task
from test_loop_fakes import FakeSandbox, make_orch

REFERENCE = '''class RetrievalTool:
    def prepare_dataset(self, data):
        self.database = data.copy()
    def retrieve_all(self, data):
        return self.search(self.database, data)
'''
STUB = '''import torch
class RetrievalTool:
    def prepare_dataset(self, data):
        pass
    def retrieve_all(self, data):
        n = data.shape[0]
        return torch.zeros((n, 96, 7))
'''


def test_zero_retrieval_and_empty_preparation_are_detected_against_reference():
    issues = placeholder_issues(STUB, {"layers/Retrieval.py": REFERENCE})
    assert len(issues) == 2 and "retrieve_all" in issues[1]
    assert placeholder_issues(STUB, {}) == []  # no unsupported method-fidelity claim


def test_initialization_and_data_dependent_retrieval_are_allowed():
    code = '''import torch
class RetrievalTool:
    def prepare_dataset(self, data):
        self.database = data.copy()
    def retrieve_all(self, data):
        out = torch.zeros_like(data)
        for i, row in enumerate(data):
            out[i] = self.search(row)
        return out
'''
    assert placeholder_issues(code, {"layers/Retrieval.py": REFERENCE}) == []
    assert placeholder_issues('this is not valid python:', {"r.py": REFERENCE}) == []


def test_truncated_later_function_does_not_hide_complete_reference_functions():
    repo = REFERENCE + '    def later(self):\n        broken_call(\n'
    assert len(placeholder_issues(STUB, {"layers/Retrieval.py": repo})) == 2


def test_guard_rechecks_every_repair_without_starting_sandbox(tmp_path):
    o, calls = make_orch(FakeSandbox(), seeds=3, max_attempts=3, repairs=[STUB, STUB])
    o.graph_repair = True
    o._repo_files = {"layers/Retrieval.py": REFERENCE}
    o._emit = lambda *a, **kw: None
    o._attempt = lambda *a, **kw: pytest.fail("Placeholder reached sandbox execution")
    claim = Claim('c', 'SARAF', 'mse', .371, .01, 'paper', 'paper')
    dossier = Dossier('saraf', [claim])
    value, note, _ = o._run_seed(claim, cora_task(), 'snapshot', dossier, tmp_path, 'retrieve historical neighbors', STUB, 0)
    assert value is None and len(dossier.attempts) == 3 and len(calls['repair']) == 2
    assert all(a.measured is None and a.branch_id == 'not-started' for a in dossier.attempts)
    assert 'Incomplete method' in dossier.attempts[0].error_summary


def test_exhausted_placeholder_is_not_a_negative_paper_result(tmp_path):
    o, calls = make_orch(FakeSandbox(), seeds=3, max_attempts=1, write=STUB)
    o.graph_repair, o.n_candidates = True, 1
    o._repo_files = {"layers/Retrieval.py": REFERENCE}
    o._emit = lambda *a, **kw: None
    o._attempt = lambda *a, **kw: pytest.fail('No sandbox for a placeholder')
    claim = Claim('c', 'SARAF', 'mse', .371, .01, 'paper', 'paper')
    dossier = Dossier('saraf', [claim])
    with pytest.raises(MethodImplementationBlocked, match='paper is not assessable'):
        o.run_claim({'dataset': 'ETTh1', 'seeds': 3}, claim, cora_task(), 'snapshot', dossier, tmp_path)
    assert dossier.verdicts == []
    assert len(dossier.attempts) == 1


def test_a_repaired_non_placeholder_can_proceed_to_measurement(tmp_path):
    o, calls = make_orch(FakeSandbox(), max_attempts=2, repairs=[REFERENCE])
    o.graph_repair, o._repo_files = True, {'layers/Retrieval.py': REFERENCE}
    o._emit = lambda *a, **kw: None
    executed = []
    def attempt(snapshot, code, seed, task, metric, **kw):
        executed.append(code)
        assert code == REFERENCE
        return 'real-branch', {'status': 'ok', 'measured': .371}
    o._attempt = attempt
    claim = Claim('c', 'SARAF', 'mse', .371, .01, 'paper', 'paper')
    dossier = Dossier('saraf', [claim])
    measured, note, code = o._run_seed(claim, cora_task(), 'snapshot', dossier, tmp_path, 'retrieve neighbors', STUB, 0)
    assert measured == .371 and executed == [REFERENCE]
    assert dossier.attempts[0].measured is None and dossier.attempts[1].measured == .371
