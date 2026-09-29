import json

import httpx
import pytest

from repro_agent import costs
from repro_agent import llm as llm_mod
from repro_agent.config import LLMConfig, Settings
from repro_agent.costs import BudgetExceeded, budget_status, gross


def _client(tmp_path, monkeypatch, responses, budget=0.0, provider="tokenfactory"):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_mod, "COST_LOG_PATH", tmp_path / "costs.jsonl")
    settings = Settings(
        llm=LLMConfig(provider, "http://x", "k", "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-super-120b-a12b", "m"),
        sandbox_backend="local", tavily_api_key="", max_cost_per_run_usd=1.0,
        max_attempts_per_claim=1, llm_cache=False, vat_rate=0.19, budget_total_usd=budget,
    )
    c = llm_mod.LLMClient(settings)
    calls = []
    it = iter(responses)

    def handler(req):
        calls.append(req)
        return next(it)

    c._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    return c, calls


def _ok(pt=1000, ct=2000, reasoning=500):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "completion_tokens_details": {"reasoning_tokens": reasoning}},
    })


MSG = [{"role": "user", "content": "x"}]


def test_gross_and_net_cost(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch, [_ok(1_000_000, 1_000_000)])
    r = c.complete("code", MSG)
    assert r.cost_usd == pytest.approx(0.30 + 0.90)          # netto
    assert c.gross(r.cost_usd) == pytest.approx(1.20 * 1.19)
    assert gross(1.0, 0.19) == pytest.approx(1.19)


def test_log_entry_fields(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch, [_ok()])
    c.complete("code", MSG)
    e = json.loads((tmp_path / "costs.jsonl").read_text())
    for k in ("provider", "base_url", "model", "prompt_tokens", "completion_tokens", "cache_hit",
              "finish_reason", "cost_usd"):
        assert k in e
    assert (e["provider"], e["prompt_tokens"], e["completion_tokens"], e["reasoning_tokens"],
            e["finish_reason"]) == ("tokenfactory", 1000, 2000, 500, "stop")


def test_empty_paid_response_is_logged(tmp_path, monkeypatch):
    empty = httpx.Response(200, json={
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    })
    c, _ = _client(tmp_path, monkeypatch, [empty, _ok()])
    c.complete("code", MSG)
    rows = costs.read_log(tmp_path / "costs.jsonl")
    assert [r["status"] for r in rows] == ["bad_response", "ok"]
    assert c.total_cost_usd == pytest.approx(sum(r["cost_usd"] for r in rows))


def test_budget_status_thresholds():
    assert budget_status(10, 0) == "off"
    assert budget_status(79.9, 100) == "ok"
    assert budget_status(80, 100) == "warn"
    assert budget_status(94.9, 100) == "warn"
    assert budget_status(95, 100) == "abort"


def _seed_log(path, net_usd):
    path.write_text(json.dumps({"ts": 1.0, "model": "m", "prompt_tokens": 1, "completion_tokens": 1,
                                "cost_usd": net_usd}) + "\n")


def test_budget_abort_before_first_call(tmp_path, monkeypatch):
    _seed_log(tmp_path / "costs.jsonl", 8.0)      # brutto 9,52 von 10 = 95,2 %
    c, calls = _client(tmp_path, monkeypatch, [_ok()], budget=10.0)
    with pytest.raises(BudgetExceeded, match="Budget"):
        c.complete("code", MSG)
    assert calls == []


def test_budget_warning_at_80(tmp_path, monkeypatch, capsys):
    _seed_log(tmp_path / "costs.jsonl", 7.0)      # brutto 8,33 von 10 = 83 %
    c, _ = _client(tmp_path, monkeypatch, [_ok()], budget=10.0)
    assert c.complete("code", MSG).content == "hi"
    assert "WARNUNG" in capsys.readouterr().err


@pytest.mark.parametrize("resp", [
    httpx.Response(402, json={"error": {"message": "Billing: Suspended"}}),
    httpx.Response(401, text="unauthorized"),
    httpx.Response(403, text="forbidden"),
    httpx.Response(400, json={"error": {"message": "Account suspended, billing issue"}}),
    httpx.Response(200, json={"error": {"message": "Insufficient balance"}}),
])
def test_billing_error_no_retry_and_blocks_run(tmp_path, monkeypatch, resp):
    c, calls = _client(tmp_path, monkeypatch, [resp, _ok(), _ok()])
    with pytest.raises(llm_mod.BillingBlocked, match="Nebius-Abrechnung blockiert – Konsole prüfen"):
        c.complete("code", MSG)
    assert len(calls) == 1                       # kein Retry
    with pytest.raises(llm_mod.BillingBlocked):  # weitere Aufrufe (z. B. Reparatur) scheitern sofort
        c.complete("code", MSG)
    assert len(calls) == 1


def test_billing_error_skips_repair_loop(tmp_path, monkeypatch):
    from repro_agent.loop import Orchestrator

    c, calls = _client(tmp_path, monkeypatch, [httpx.Response(402, text="Billing: Suspended"), _ok()])
    orch = Orchestrator.__new__(Orchestrator)
    orch.llm, orch.settings = c, c.settings
    orch._cost_base, orch.details = 0.0, []
    import threading
    orch._lock = threading.Lock()
    with pytest.raises(llm_mod.BillingBlocked):
        orch._code_from_llm("sys", "user")       # faengt nur ValueError ab, keine 3 Versuche
    assert len(calls) == 1


def test_rate_limit_and_free_daily_limit_still_retry(tmp_path, monkeypatch):
    limit = httpx.Response(429, json={"error": {"message": "free-models-per-day. Add 10 credits to unlock"}})
    c, calls = _client(tmp_path, monkeypatch, [limit, httpx.Response(503, text="down"), _ok()])
    assert c.complete("code", MSG).content == "hi"
    assert len(calls) == 3


def test_is_billing_error():
    assert llm_mod.is_billing_error(402, "")
    assert not llm_mod.is_billing_error(500, "billing service down")
    assert not llm_mod.is_billing_error(429, "rate limited")


def test_aggregate_splits_cache_and_provider():
    rows = [
        {"ts": 1.0, "model": "a:free", "prompt_tokens": 1, "completion_tokens": 2, "cost_usd": 0.0},
        {"ts": 1.0, "model": "b", "prompt_tokens": 10, "completion_tokens": 20, "cost_usd": 1.0},
        {"ts": 1.0, "model": "b", "prompt_tokens": 10, "completion_tokens": 20, "cost_usd": 0.0, "cache_hit": True},
    ]
    agg = costs.aggregate(rows, 0.19)
    assert len(agg) == 3
    b = next(r for r in agg if r["model"] == "b" and not r["cache_hit"])
    assert (b["provider"], b["gross"]) == ("tokenfactory", pytest.approx(1.19))
