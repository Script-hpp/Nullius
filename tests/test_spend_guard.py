"""Production admission regressions: mocked providers only, no paid calls."""
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from repro_agent import llm, spend_guard as guard, tavily_usage as tu
from repro_agent.config import LLMConfig, Settings
from repro_agent.costs import BudgetExceeded

MODEL = "nvidia/nemotron-3-super-120b-a12b"
MSG = [{"role": "user", "content": "hello"}]


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "COST_LOG_PATH", tmp_path / "costs.jsonl")
    monkeypatch.setattr(tu, "TAVILY_LOG_PATH", tmp_path / "tavily_usage.jsonl")
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(tu, "_sleep", lambda _: None)
    monkeypatch.setenv("LIVE_MIN_BUDGET_USD", "0.1")
    monkeypatch.setenv("MAX_DAILY_COST_USD", "1")
    monkeypatch.setenv("TAVILY_CREDITS_TOTAL", "302")
    monkeypatch.setenv("LIVE_MIN_TAVILY_CREDITS", "300")
    return Settings(LLMConfig("tokenfactory", "http://mock", "dummy", MODEL, MODEL, MODEL),
                    "local", "", .5, 1, False, vat_rate=0, budget_total_usd=10,
                    budget_since="2026-01-01T00:00Z")


def client(settings, handler, scope=None):
    token = guard.current_budget.set(scope)
    try:
        c = llm.LLMClient(settings)
    finally:
        guard.current_budget.reset(token)
    c._client.close()
    c._client = httpx.Client(base_url="http://mock", transport=httpx.MockTransport(handler))
    return c


def ok(cost=.001):
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": cost}})


def seed_cost(cost):
    import time
    llm.COST_LOG_PATH.write_text(json.dumps({"ts": time.time(), "cost_usd": cost}) + "\n")


def test_global_budget_reserves_next_call_before_network(configured):
    seed_cost(9.495)
    calls = []
    c = client(configured, lambda r: calls.append(r) or ok())
    with pytest.raises(BudgetExceeded, match="95%"):
        c.complete("extract", MSG)
    assert calls == []


@pytest.mark.parametrize("limit", [0, .005, float("nan")])
def test_job_cap_covers_direct_extraction_and_retries(configured, limit):
    calls = []
    c = client(configured, lambda r: calls.append(r) or ok(), guard.JobBudget(limit))
    with pytest.raises(BudgetExceeded):
        c.complete("extract", MSG)
    assert calls == []


def test_paid_empty_response_rechecks_job_cap(configured):
    calls = []
    def handler(r):
        calls.append(r)
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": .01}})
    c = client(configured, handler, guard.JobBudget(.02))
    with pytest.raises(BudgetExceeded):
        c.complete("code", MSG)
    assert len(calls) == 1
    assert c.total_cost_usd == .01


def test_unknown_price_blocks_before_network(configured):
    settings = dataclasses.replace(configured, llm=dataclasses.replace(configured.llm, model_code="unknown"))
    c = client(settings, lambda r: pytest.fail("network must not run"))
    with pytest.raises(BudgetExceeded, match="price"):
        c.complete("code", MSG)


def test_default_output_cap_and_settlement(configured):
    calls = []
    c = client(configured, lambda r: calls.append(json.loads(r.content)) or ok())
    c.complete("code", MSG)
    assert calls[0]["max_tokens"] == 16000
    assert guard.pending(llm.COST_LOG_PATH) == 0
    assert c.total_cost_usd == .001


def test_uncertain_timeout_reservation_survives_new_client(configured):
    scope = guard.JobBudget(.02)
    calls = []
    def timeout(r):
        calls.append(r)
        raise httpx.ReadTimeout("uncertain", request=r)
    c = client(configured, timeout, scope)
    with pytest.raises(BudgetExceeded):
        c.complete("code", MSG)
    assert len(calls) == 1
    assert guard.pending(llm.COST_LOG_PATH) > .014
    seed_cost(9.48)
    c2 = client(configured, lambda r: pytest.fail("pending spend must count"))
    with pytest.raises(BudgetExceeded):
        c2.complete("code", MSG)


def test_parallel_clients_cannot_reserve_same_remaining_budget(configured):
    seed_cost(9.48)
    def admit(_):
        try:
            return guard.reserve_model(llm.COST_LOG_PATH, configured, guard.JobBudget(.5), .015)
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(admit, range(4)))
    assert sum(r is not None for r in results) == 1


@pytest.mark.parametrize("daily,floor", [(".01", ".1"), ("1", "9.99")])
def test_live_daily_and_floor_include_next_request(configured, monkeypatch, daily, floor):
    monkeypatch.setenv("MAX_DAILY_COST_USD", daily)
    monkeypatch.setenv("LIVE_MIN_BUDGET_USD", floor)
    c = client(configured, lambda r: pytest.fail("live budget must block"), guard.JobBudget(.5, live=True))
    with pytest.raises(BudgetExceeded):
        c.complete("code", MSG)


def test_cancelled_job_blocks_both_providers(configured):
    scope = guard.JobBudget(.5, live=True)
    c = client(configured, lambda r: pytest.fail("cancelled model request"), scope)
    token = guard.current_budget.set(scope)
    try:
        search = tu.CountedClient(type("Fake", (), {"search": lambda *a, **k: pytest.fail("cancelled search")})())
    finally:
        guard.current_budget.reset(token)
    scope.cancelled.set()
    with pytest.raises(BudgetExceeded):
        c.complete("code", MSG)
    with pytest.raises(BudgetExceeded):
        search.search(query="x")


def test_tavily_retry_checks_floor_before_each_attempt(configured, monkeypatch):
    import requests
    monkeypatch.setenv("LLM_PROVIDER", "tokenfactory")
    calls = []
    class Fake:
        def search(self, **kwargs):
            calls.append(kwargs)
            raise requests.exceptions.Timeout("uncertain")
    search = tu.CountedClient(Fake())
    with pytest.raises(BudgetExceeded, match="credit floor"):
        search.search(query="x", search_depth="advanced")
    assert len(calls) == 1
    assert tu.used_credits() == 2


def test_tavily_parallel_calls_reserve_credits(configured, monkeypatch):
    import threading
    monkeypatch.setenv("LLM_PROVIDER", "tokenfactory")
    entered, finish = threading.Event(), threading.Event()
    class Fake:
        def search(self, **kwargs):
            entered.set()
            assert finish.wait(2)
            return {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(tu.CountedClient(Fake()).search, query="x", search_depth="advanced")
        try:
            assert entered.wait(2)
            with pytest.raises(BudgetExceeded):
                tu.CountedClient(Fake()).search(query="y")
        finally:
            finish.set()
        assert first.result() == {}
    assert tu.used_credits() == 2
    assert guard.pending(tu.TAVILY_LOG_PATH) == 0


def test_unwritable_ledger_blocks_before_network(configured, monkeypatch):
    llm.COST_LOG_PATH.parent.joinpath("blocked").write_text("file")
    monkeypatch.setattr(llm, "COST_LOG_PATH", llm.COST_LOG_PATH.parent / "blocked" / "costs.jsonl")
    c = client(configured, lambda r: pytest.fail("unavailable ledger"))
    with pytest.raises(BudgetExceeded, match="ledger"):
        c.complete("code", MSG)


def test_unexpected_provider_cost_stops_all_shared_clients(configured):
    scope = guard.JobBudget(.5)
    c = client(configured, lambda r: ok(.6), scope)
    c.complete("code", MSG)
    assert c.total_cost_usd == .6 and scope.cancelled.is_set()
    c2 = client(configured, lambda r: pytest.fail("unexpected spend must stop job"), scope)
    with pytest.raises(BudgetExceeded):
        c2.complete("extract", MSG)


def test_missing_usage_retains_reservation(configured):
    c = client(configured, lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    with pytest.raises(BudgetExceeded, match="usage"):
        c.complete("code", MSG)
    assert guard.pending(llm.COST_LOG_PATH) > 0


def test_cancelled_job_cannot_launch_sandbox_work(configured):
    from repro_agent.sandbox.nebius import NebiusSandbox
    sandbox = NebiusSandbox.__new__(NebiusSandbox)
    sandbox._budget = guard.JobBudget(.5, live=True)
    sandbox._budget.cancelled.set()
    sandbox._client = None  # Any attempted SDK access would fail the test.
    sandbox._sessions = {}
    with pytest.raises(BudgetExceeded):
        sandbox.base_snapshot("image", [])
    with pytest.raises(BudgetExceeded):
        sandbox.branch("snapshot")
    with pytest.raises(BudgetExceeded):
        sandbox.run("branch", "command", 5)


def test_begin_run_preserves_api_scope_and_resets_cli_cap(configured):
    scope = guard.JobBudget(.5, live=True)
    c = client(configured, lambda r: ok(), scope)
    c.begin_run()
    assert c._budget is scope
    cli = client(configured, lambda r: ok())
    previous = cli._budget
    cli.begin_run()
    assert cli._budget is not previous


def test_failed_reservation_sync_blocks_network(configured, monkeypatch):
    def fail(fd):
        raise OSError("disk sync failed")
    monkeypatch.setattr(guard.os, "fsync", fail)
    c = client(configured, lambda r: pytest.fail("reservation must be durable first"))
    with pytest.raises(BudgetExceeded, match="ledger"):
        c.complete("code", MSG)


def test_failed_booking_keeps_existing_reservation(configured):
    scope = guard.JobBudget(.5)
    token = guard.reserve_model(llm.COST_LOG_PATH, configured, scope, .02)
    def fail(response):
        raise OSError("booking failed")
    response = llm.LLMResponse("ok", MODEL, 1, 1, .001, False)
    with pytest.raises(BudgetExceeded, match="ledger"):
        guard.settle_model(llm.COST_LOG_PATH, configured, scope, token, response, fail)
    assert guard.pending(llm.COST_LOG_PATH) == .02
