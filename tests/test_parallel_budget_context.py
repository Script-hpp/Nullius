from concurrent.futures import ThreadPoolExecutor
import pytest
from repro_agent import spend_guard as sg
from repro_agent.tavily_usage import CountedClient
from repro_agent.costs import BudgetExceeded


def test_lazy_clients_in_parallel_capture_parent_job_budget():
    job=sg.JobBudget(0.5)
    token=sg.current_budget.set(job)
    try:
        with ThreadPoolExecutor(3) as pool:
            clients=list(sg.context_map(pool, lambda _: CountedClient(object()), range(6)))
        assert all(c._budget is job for c in clients)
        assert sg.current_budget.get() is job
    finally: sg.current_budget.reset(token)


def test_cancelled_job_cannot_continue_in_a_child_task():
    job=sg.JobBudget(0.5);job.cancelled.set()
    token=sg.current_budget.set(job)
    try:
        with ThreadPoolExecutor(2) as pool:
            with pytest.raises(BudgetExceeded): list(sg.context_map(pool, lambda _: sg.current_budget.get().check(), range(2)))
    finally: sg.current_budget.reset(token)
