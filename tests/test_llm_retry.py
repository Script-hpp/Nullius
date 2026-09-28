import httpx
import pytest

from repro_agent import llm as llm_mod
from repro_agent.config import LLMConfig, Settings


def _client(responses):
    settings = Settings(
        llm=LLMConfig("openrouter", "http://x", "k", "m", "m", "m"), sandbox_backend="local",
        tavily_api_key="", max_cost_per_run_usd=1.0, max_attempts_per_claim=1, llm_cache=False,
    )
    c = llm_mod.LLMClient(settings)
    it = iter(responses)
    c._client = httpx.Client(
        transport=httpx.MockTransport(lambda req: next(it)), base_url="http://x"
    )
    return c


def test_retries_200_with_error_body(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_mod, "COST_LOG_PATH", llm_mod.REPO_ROOT / "logs" / "test_costs.jsonl")
    ok = httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {}})
    bad = httpx.Response(200, json={"error": {"message": "rate limited"}})
    resp = _client([bad, bad, ok]).complete("fast", [{"role": "user", "content": "x"}])
    assert resp.content == "hi"


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    bad = httpx.Response(200, json={"error": "x"})
    with pytest.raises(RuntimeError, match="fehlgeschlagen"):
        _client([bad] * 5).complete("fast", [{"role": "user", "content": "x"}])
