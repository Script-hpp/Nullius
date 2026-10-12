import pytest
import json
import os
from repro_agent.discovery import DiscoveryAgent, CandidatePaper, CACHE_VERSION
from repro_agent.tavily_tools import TavilyTools
from repro_agent.config import Settings

class FakeLLM:
    def __init__(self, claim_content=None):
        if claim_content is None:
            self.claim_content = '{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "acc", "target": 95}'
        else:
            self.claim_content = claim_content
    def complete(self, role, messages, temperature=0.0, **kwargs):
        class Resp:
            content = self.claim_content
        return Resp()

# Discovery verlangt lesbaren Volltext (MIN_FULLTEXT_CHARS), Abstract allein reicht nicht.
MISMATCH = '{"protocol": "passt_nicht", "protocol_quote": "GraphNet achieves 95% accuracy", "protocol_reason": "Protocol mismatch", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}'
FULLTEXT = "This is a sample abstract mentioning Cora dataset. GraphNet achieves 95% accuracy. " + "Filler sentence. " * 300


class FakeTavily:
    def __init__(self, repos=None):
        self.repos = repos or []
        self.on_call = None
        
    def read_claims(self, url):
        if self.on_call: self.on_call(2, "extract", url, 0)
        return {"results": [{"raw_content": FULLTEXT}]}
        
    def curate_extract(self, url, query):
        self.extract_queries = getattr(self, "extract_queries", []) + [query]
        return {"results": [{"url": url, "raw_content": FULLTEXT}]}

    def curate_repo(self, title):
        if self.on_call: self.on_call(8, "search", title, len(self.repos))
        return {"results": [{"url": r} for r in self.repos]}
        
    def curate_discussion(self, title):
        if self.on_call: self.on_call(9, "search", title, 0)
        return {"results": []}


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP Error")
    def json(self):
        return {"stargazers_count": 100}

class FakeSandbox:
    def base_snapshot(self, *args, **kwargs):
        return "snap_1"
    def branch(self, *args, **kwargs):
        return "branch_1"
    def run(self, *args, **kwargs):
        class Res:
            exit_code = 0
            timed_out = False
        return Res()

@pytest.fixture
def arxiv_xml():
    with open("tests/fixtures/arxiv_sample.xml", "r") as f:
        return f.read()


def test_no_claim_reject(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(url, *args, **kwargs):
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)

    
    
    # 1. protocol passt_nicht
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=MISMATCH), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "protocol" in e.get("reason", "").lower()]
    assert len(rejects) >= 1
    
    # 2. Empty quote
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "quote": "", "target": 95}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "no quote provided" in e.get("reason", "").lower()]
    assert len(rejects) >= 1

    # 3. Quote not in text
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "quote": "I made this up 95", "target": 95}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "quote not found in text" in e.get("reason", "").lower()]
    assert len(rejects) >= 1

    # 4. Target not in quote
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 99}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "target value not found numerically in quote" in e.get("reason", "").lower()]
    assert len(rejects) >= 1
    
    # 5. Success case
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    candidates = [e for e in events if e.get("type") == "candidate"]
    assert len(candidates) >= 1

def test_arxiv_error(monkeypatch, tmp_path):
    def mock_get(url, *args, **kwargs):
        raise Exception("Timeout")
    monkeypatch.setattr("httpx.get", mock_get)

    
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    errors = [e for e in events if e.get("type") == "error"]
    assert len(errors) == 1
    assert "arXiv" in errors[0]["message"]

def test_tavily_on_call_fires():
    class FakeClient:
        def search(self, **kwargs):
            return {"results": [{"url": "https://github.com/foo/bar"}]}

    class FakeSettings:
        tavily_api_key = "fake"

    tavily = TavilyTools(settings=FakeSettings())
    tavily._client = FakeClient()
    
    calls = []
    def record_call(role, endpoint, target, n):
        calls.append((role, endpoint, target, n))
    tavily.on_call = record_call
    
    res = tavily.curate_repo("test paper")
    assert len(calls) == 1
    assert calls[0][0] == 8  # Role 8
    assert calls[0][1] == "search"
    assert calls[0][2] == "test paper"
    assert calls[0][3] == 1  # 1 result

def test_tavily_mode_fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("TAVILY_MODE", "fixture")
    
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    errors = [e for e in events if e.get("type") == "error"]
    assert len(errors) == 1
    assert "Discovery braucht live" in errors[0]["message"]

def test_cache_hit(monkeypatch, tmp_path):
    # Setup cache
    cache_file = tmp_path / "cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "1234.5678": {
                "title": "Cached Paper",
                "arxiv_id": "1234.5678",
                "dataset": "Cora",
                "claim": {"method": "GraphNet", "metric": "accuracy", "target": 80},
                "reason": "Cached",
                "github_url": "https://github.com/foo/bar",
                "status": "ungeprueft",
                "cache_version": CACHE_VERSION,
                "alternatives": []
            }
        }, f)

    def mock_get(*args, **kwargs):
        raise RuntimeError("Should not hit network")
    monkeypatch.setattr("httpx.get", mock_get)

    class CountingTavily:
        def __init__(self):
            self.calls = 0
        def curate_repo(self, title):
            self.calls += 1
            return {"results": []}

    tavily = CountingTavily()
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=tavily, cache_path=str(cache_file))
    
    # We mock _search_arxiv so it returns a paper that is in cache
    agent._search_arxiv = lambda ds, max_results=5: [{
        "arxiv_id": "1234.5678",
        "title": "Cached Paper",
        "dataset": ds,
        "abstract": "No network."
    }]
    
    events = []
    candidates = agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    # Assert network was not hit (tavily calls = 0)
    assert tavily.calls == 0
    # Assert candidate is returned
    assert len(candidates) == 1
    assert candidates[0].title == "Cached Paper"

def test_cache_hit_on_reject(monkeypatch, tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "1234.5678": {
                "status": "rejected",
                "cache_version": 3,
                "reason": "Regex pre-filter failed"
            }
        }, f)

    def mock_get(*args, **kwargs):
        raise RuntimeError("Should not hit network")
    monkeypatch.setattr("httpx.get", mock_get)

    class CountingTavily:
        def __init__(self):
            self.calls = 0
        def curate_repo(self, title):
            self.calls += 1
            return {"results": []}
        def curate_extract(self, url, query):
            self.calls += 1
            return {"results": [{"raw_content": "No claims here."}]}

    tavily = CountingTavily()
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=tavily, cache_path=str(cache_file))
    
    # We mock _search_arxiv so it returns a paper that is in cache
    agent._search_arxiv = lambda ds, max_results=30: [{
        "arxiv_id": "1234.5678",
        "title": "Cached Paper",
        "dataset": ds,
        "abstract": "No network."
    }]
    
    events = []
    candidates = agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    # Assert network was not hit (tavily calls = 0)
    assert tavily.calls == 0
    # Reject is in cache, so it shouldn't produce a candidate
    assert len(candidates) == 0

def test_numerical_quote(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(url, *args, **kwargs):
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    cache_file = tmp_path / "c1.json"
    
    # Target 95.0, quote has 95%
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95.0}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    candidates = [e for e in events if e.get("type") == "candidate"]
    assert len(candidates) >= 1

    cache_file = tmp_path / "c2.json"
    # Target 0.95, quote has 95%
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 0.95}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    candidates = [e for e in events if e.get("type") == "candidate"]
    assert len(candidates) >= 1

def test_extract_limit(monkeypatch, arxiv_xml, tmp_path):
    # Simulate 4 papers returned from search
    def mock_search(*args, **kwargs):
        return [
            {"arxiv_id": f"100{i}", "title": f"Paper {i}", "dataset": "Cora", "abstract": "Has 95% accuracy on cora"}
            for i in range(4)
        ]
    
    class CountingTavily:
        def __init__(self):
            self.calls = 0
        def curate_repo(self, title):
            return {"results": []}
        def curate_extract(self, url, query):
            self.calls += 1
            raise Exception("Fail extract")
            
    tavily = CountingTavily()
    cache_file = tmp_path / "c3.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=MISMATCH), tavily=tavily, cache_path=str(cache_file))
    agent._search_arxiv = mock_search
    
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    # Ensure exactly 3 extracts were made despite 4 papers passing regex and failing extract
    assert tavily.calls == 3
    warnings = [e for e in events if e.get("type") == "warning" and "Max 3" in e.get("message")]
    assert len(warnings) == 1


def test_missing_target(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(*args, **kwargs): return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)

    # Missing target
    cache_file = tmp_path / "c1.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "quote": "95"}'), tavily=FakeTavily(), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any("Target missing" in e.get("reason", "") for e in events if e.get("type") == "reject")

    # Null target
    cache_file = tmp_path / "c2.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "quote": "95", "target": null}'), tavily=FakeTavily(), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any("Target missing" in e.get("reason", "") for e in events if e.get("type") == "reject")

    # String target with %
    cache_file = tmp_path / "c3.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95%", "metric": "accuracy", "target": "95%"}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any(e.get("type") == "candidate" for e in events)

    # String target bad parse
    cache_file = tmp_path / "c4.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "quote": "95%", "target": "ninety-five"}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any("Target missing" in e.get("reason", "") for e in events if e.get("type") == "reject")


def test_on_call_events(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(url, *args, **kwargs):
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    cache_file = tmp_path / "c99.json"
    
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95%", "metric": "accuracy", "target": "95%"}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    # Assert sequence numbers are strictly monotonic
    seqs = [e["seq"] for e in events if "seq" in e]
    assert seqs == list(range(len(seqs)))
    
    # Assert timestamps are present
    assert all("ts" in e for e in events)
    
    # Assert tavily_call event
    tavily_calls = [e for e in events if e.get("type") == "tavily_call"]
    assert len(tavily_calls) >= 2 # 1 extract, 1 repo search
    assert tavily_calls[0]["role"] == 2 # extract
    assert "duration_ms" in tavily_calls[0]
    
    # Assert llm_call event
    llm_calls = [e for e in events if e.get("type") == "llm_call"]
    assert len(llm_calls) == 1
    assert "cost_usd" in llm_calls[0]
    
    # Assert arxiv_call event
    arxiv_calls = [e for e in events if e.get("type") == "arxiv_call"]
    assert len(arxiv_calls) == 1


def test_L1_https(monkeypatch, tmp_path):
    import httpx
    called_urls = []
    def mock_get(url, *args, **kwargs):
        called_urls.append(url)
        return FakeResponse('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    monkeypatch.setattr("httpx.get", mock_get)
    
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=FakeTavily(), cache_path=str(tmp_path / "c.json"))
    agent.discover(["Cora"])
    assert any(url.startswith("https://export.arxiv.org") for url in called_urls)

def test_L2_llm_truncated(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(url, *args, **kwargs):
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    
    class TruncatedLLM(FakeLLM):
        def complete(self, *args, **kwargs):
            from repro_agent.llm import LLMResponse
            return LLMResponse(content="{}", model="fake", prompt_tokens=10, completion_tokens=10, cost_usd=0, finish_reason="length", cached=False)
            
    cache_file = tmp_path / "c_l2.json"
    agent = DiscoveryAgent(llm=TruncatedLLM(), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    # Assert event
    assert any(e.get("reason") == "llm_truncated" for e in events)
    # Assert not cached permanently
    import json
    import os
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            c = json.load(f)
            assert "2101.12345" not in c

def test_L3_short_text(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(url, *args, **kwargs):
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    
    class ShortTavily(FakeTavily):
        def curate_extract(self, url, query):
            return {"results": [{"raw_content": "Too short!"}]}
            
    cache_file = tmp_path / "c_l3.json"
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=ShortTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    assert any(e.get("reason") == "Volltext nicht lesbar" for e in events)
    # Assert not cached permanently
    import json
    import os
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            c = json.load(f)
            assert "2101.12345" not in c


def test_L6_extract_limit_order(monkeypatch, tmp_path):
    # Mock arXiv to return 4 papers
    def mock_get(url, *args, **kwargs):
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2101.0001</id><title>P1</title><summary>Cora 95%</summary><author><name>A</name></author></entry>
  <entry><id>http://arxiv.org/abs/2101.0002</id><title>P2</title><summary>Cora 95%</summary><author><name>A</name></author></entry>
  <entry><id>http://arxiv.org/abs/2101.0003</id><title>P3</title><summary>Cora 95%</summary><author><name>A</name></author></entry>
  <entry><id>http://arxiv.org/abs/2101.0004</id><title>P4</title><summary>Cora 95%</summary><author><name>A</name></author></entry>
</feed>'''
        return FakeResponse(xml)
    monkeypatch.setattr("httpx.get", mock_get)

    # protocol passt_nicht: kein Kandidat, sonst stoppt die Grenze von 2 Kandidaten vor dem Deckel
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=MISMATCH), tavily=FakeTavily(), cache_path=str(tmp_path / "c_l6.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    
    warnings = [e for e in events if e.get("type") == "warning"]
    extractings = [e for e in events if e.get("type") == "extracting"]
    
    assert len(warnings) == 1
    assert "Max 3 Tavily extracts reached" in warnings[0]["message"]
    # We should have exactly 3 extractings, not 4!
    assert len(extractings) == 3



def _read_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def test_L2_truncated_json_not_cached(monkeypatch, arxiv_xml, tmp_path):
    """Echter Fall aus dem Live-Lauf: abgeschnittener Denktext, kein gueltiges JSON."""
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))

    class TruncatedLLM:
        def complete(self, *args, **kwargs):
            from repro_agent.llm import LLMResponse
            return LLMResponse(content='We need to find the claim. The paper says {"quote": "It ach',
                               model="fake", prompt_tokens=15000, completion_tokens=8000,
                               cost_usd=0, cached=False, finish_reason="length")

    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=TruncatedLLM(), tavily=FakeTavily(), cache_path=cache_file)
    events = []
    agent.discover(["Cora"], on_event=events.append)
    assert any(e.get("reason") == "llm_truncated" for e in events)
    assert "2101.12345" not in _read_cache(cache_file)


def test_L2_unreadable_answer_not_cached(monkeypatch, arxiv_xml, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=FakeLLM(claim_content="I could not decide."), tavily=FakeTavily(), cache_path=cache_file)
    events = []
    agent.discover(["Cora"], on_event=events.append)
    assert any(e.get("reason") == "LLM-Antwort nicht lesbar" for e in events)
    assert "2101.12345" not in _read_cache(cache_file)


def test_L2_thinking_text_before_json(monkeypatch, arxiv_xml, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    content = ('Let me think. The text uses {braces} in prose. Final answer: '
               '{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}')
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=content), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    events = []
    agent.discover(["Cora"], on_event=events.append)
    assert any(e.get("type") == "candidate" for e in events)


def test_L3_extract_error_does_not_fall_back_to_abstract(monkeypatch, arxiv_xml, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))

    class FailingTavily(FakeTavily):
        def curate_extract(self, url, query):
            raise RuntimeError("extract failed")

    class CountingLLM(FakeLLM):
        calls = 0
        def complete(self, *a, **k):
            CountingLLM.calls += 1
            return super().complete(*a, **k)

    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=CountingLLM(), tavily=FailingTavily(), cache_path=cache_file)
    events = []
    agent.discover(["Cora"], on_event=events.append)
    assert any(e.get("reason") == "Volltext nicht lesbar" for e in events)
    assert CountingLLM.calls == 0          # Protokoll nicht aus dem Abstract beurteilen
    assert "2101.12345" not in _read_cache(cache_file)


def test_max_two_candidates_per_dataset(monkeypatch, tmp_path):
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}'),
                           tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse("# Cora\n"))
    agent._search_arxiv = lambda ds, max_results=30: [
        {"arxiv_id": f"2101.000{i}", "title": f"P{i}", "abstract": "Cora 95% accuracy", "authors": []} for i in range(4)]
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    assert len(result) == 2
    assert len([e for e in events if e.get("type") == "extracting"]) == 2


def test_L4_select_sections_keeps_experiments():
    from repro_agent.discovery import _select_sections
    text = ("# Title\n" + "intro " * 3000 + "\n## 2 Related Work\n" + "related " * 3000
            + "\n## 4 Experiments\nWe use the standard Planetoid split and reach 81.5% on Cora.\n"
            + "## 6 Conclusion\n" + "end " * 3000)
    out = _select_sections(text, "Our abstract.", limit=20000)
    assert "81.5% on Cora" in out
    assert out.startswith("Abstract: Our abstract.")
    assert "related related" not in out
    assert len(out) <= 20000


ATOM_ONE = open(os.path.join(os.path.dirname(__file__), "fixtures", "arxiv_sample.xml")).read()


class TavilyFinder(FakeTavily):
    """Tavily findet Papers selbst: arXiv-Treffer plus ein Treffer ohne arXiv-ID."""
    def __init__(self, repos=None):
        super().__init__(repos)
        self.paper_queries = []
        self.discussion_titles = []
    def curate_papers(self, query, max_results=10):
        self.paper_queries.append(query)
        return {"results": [
            {"url": "https://arxiv.org/html/2101.12345v1", "title": "Sample", "content": "Table 2: Cora 95% accuracy"},
            {"url": "https://openreview.net/pdf/abc.pdf", "title": "No arXiv id", "content": "Cora 81.5"},
        ]}
    def curate_discussion(self, title, max_results=5):
        self.discussion_titles.append(title)
        return {"results": [
            {"url": "https://github.com/foo/bar/issues/12", "title": "Cannot reproduce Sample Paper Title 95%", "content": "Seed dependent"},
            {"url": "https://arxiv.org/abs/2999.00001", "title": "fMRI dynamics", "content": "Stuart-Landau oscillators in the brain"}]}


def test_tavily_finds_papers_and_community(monkeypatch, tmp_path):
    urls = []
    def mock_get(url, *args, **kwargs):
        urls.append((url, kwargs.get("params")))
        return FakeResponse(ATOM_ONE)
    monkeypatch.setattr("httpx.get", mock_get)
    tavily = TavilyFinder(repos=["https://github.com/foo/bar"])
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=tavily, cache_path=str(tmp_path / "c.json"))
    events = []
    result = agent.discover(["Cora"], on_event=events.append)

    assert tavily.paper_queries and "Cora" in tavily.paper_queries[0]
    # Metadaten per id_list, nicht per Stichwortsuche
    arxiv_calls = [p for u, p in urls if "export.arxiv.org" in u]
    assert arxiv_calls and arxiv_calls[0]["id_list"] == "2101.12345"
    assert not any(e.get("message", "").startswith("No Tavily hits") for e in events)
    # Gezielter Extract mit Frage zum Protokoll
    assert "split" in tavily.extract_queries[0].lower()
    assert len(result) == 1
    c = result[0]
    assert c.found_by == "tavily" and c.source_url == "https://arxiv.org/html/2101.12345v1"
    assert c.community[0]["url"].endswith("/issues/12")
    cand = [e for e in events if e.get("type") == "candidate"][0]
    assert cand["found_by"] == "tavily" and cand["community_hits"] == 1
    roles = [(e["role"], e["endpoint"]) for e in events if e.get("type") == "tavily_call"]
    assert roles[0] == (8, "search") and (2, "extract") in roles


def test_falls_back_to_arxiv_without_tavily_hits(monkeypatch, arxiv_xml, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    class NoHits(FakeTavily):
        def curate_papers(self, query, max_results=10):
            return {"results": [{"url": "https://openreview.net/forum?id=x", "content": "Cora 80%"}]}
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=NoHits(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    assert any(e.get("message", "").startswith("No Tavily hits") for e in events)
    assert result and result[0].found_by == "arxiv"


def test_prefilter_uses_tavily_snippet(monkeypatch, tmp_path):
    """Abstract ohne Datensatz/Zahl, aber Tavily-Ausschnitt mit Tabelle: nicht vorab verwerfen."""
    atom = ATOM_ONE.replace("This is a sample abstract mentioning Cora dataset. GraphNet achieves 95% accuracy.",
                            "We propose a new graph model.")
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(atom))
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=TavilyFinder(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    events = []
    agent.discover(["Cora"], on_event=events.append)
    assert not any(e.get("stage") == "prefilter" for e in events)


def test_protocol_checked_against_all_dataset_tasks(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf 3: Paper mit Shchur-Splits wurde verworfen, obwohl Cora-LCC genau das prueft."""
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    prompts = []
    class RecordingLLM(FakeLLM):
        def complete(self, role, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)
    content = '{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora-LCC", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}'
    agent = DiscoveryAgent(llm=RecordingLLM(claim_content=content), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    result = agent.discover(["Cora"])
    assert "[Cora]" in prompts[0] and "[Cora-LCC]" in prompts[0]
    assert result and result[0].claim["matched_task"] == "Cora-LCC"


def test_short_extract_retried_once(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf 3: dieselbe URL lieferte erst 16, beim zweiten Aufruf 3531 Zeichen."""
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    class Flaky(FakeTavily):
        calls = 0
        def curate_extract(self, url, query):
            Flaky.calls += 1
            return {"results": [{"raw_content": "x" * 16 if Flaky.calls == 1 else FULLTEXT}]}
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=Flaky(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    assert len(agent.discover(["Cora"])) == 1
    assert Flaky.calls == 2


def test_matched_task_with_brackets_accepted(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf 4 (Token Factory): 3 von 6 Papers verworfen, weil das LLM '[Cora-LCC]' schrieb."""
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    content = '{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": " [Cora-LCC] ", "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}'
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=content), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    result = agent.discover(["Cora"])
    assert result and result[0].claim["matched_task"] == "Cora-LCC"


# ---- Dreistufiges Protokoll, Suche, Repo ----

def _run(monkeypatch, arxiv_xml, tmp_path, content, repos=("https://github.com/foo/bar",), tavily=None):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=content), tavily=tavily or FakeTavily(repos=list(repos)), cache_path=cache_file)
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    return result, events, cache_file


CLAIM = '"method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95'


def test_protocol_unklar_is_candidate(monkeypatch, arxiv_xml, tmp_path):
    content = '{"protocol": "unklar", "matched_task": null, ' + CLAIM + '}'
    result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert len(result) == 1 and result[0].protocol == "unklar"
    assert result[0].to_dict()["protocol"] == "unklar"
    cand = [e for e in events if e.get("type") == "candidate"][0]
    assert cand["protocol"] == "unklar"
    assert _read_cache(cache_file)["2101.12345"]["protocol"] == "unklar"


def test_protocol_passt_with_quote_and_task(monkeypatch, arxiv_xml, tmp_path):
    content = '{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora-LCC", ' + CLAIM + '}'
    result, events, _ = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert result[0].protocol == "passt" and result[0].claim["matched_task"] == "Cora-LCC"
    assert [e for e in events if e.get("type") == "candidate"][0]["protocol"] == "passt"


def test_passt_nicht_with_quote_rejected_and_cached(monkeypatch, arxiv_xml, tmp_path):
    content = ('{"protocol": "passt_nicht", "protocol_quote": "GraphNet achieves 95% accuracy", '
               '"protocol_reason": "60/20/20 split", ' + CLAIM + '}')
    result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert result == []
    assert any(e.get("type") == "reject" and e.get("stage") == "protocol" for e in events)
    assert _read_cache(cache_file)["2101.12345"]["status"] == "rejected"


@pytest.mark.parametrize("pq", ['"protocol_quote": "text the paper never wrote", ', ""])
def test_passt_nicht_without_valid_quote_is_unklar(monkeypatch, arxiv_xml, tmp_path, pq):
    content = '{"protocol": "passt_nicht", ' + pq + '"protocol_reason": "different split", ' + CLAIM + '}'
    result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert len(result) == 1 and result[0].protocol == "unklar"
    assert not any(e.get("stage") == "protocol" for e in events)


@pytest.mark.parametrize("task", ['"PubMed"', "null"])
def test_passt_without_valid_task_is_unklar(monkeypatch, arxiv_xml, tmp_path, task):
    content = ('{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": ' + task + ', ' + CLAIM + '}')
    result, events, _ = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert len(result) == 1 and result[0].protocol == "unklar"
    assert result[0].claim["matched_task"] is None


def test_passt_without_valid_quote_is_unklar(monkeypatch, arxiv_xml, tmp_path):
    content = ('{"protocol": "passt", "protocol_quote": "invented sentence", "matched_task": "Cora", ' + CLAIM + '}')
    result, _, _ = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert len(result) == 1 and result[0].protocol == "unklar"


def test_prompt_states_three_levels(monkeypatch, arxiv_xml, tmp_path):
    prompts = []
    class Rec(FakeLLM):
        def complete(self, role, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    DiscoveryAgent(llm=Rec(), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                   cache_path=str(tmp_path / "c.json")).discover(["Cora"])
    p = prompts[0]
    assert "passt_nicht" in p and "unklar" in p and "protocol_quote" in p
    assert "Not mentioned means" in p and "standard split" in p


def test_repo_score_zero_gives_no_repo(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf: fremdes Sammel-Repo mit Score 0/3 wurde uebernommen."""
    class ForeignReadme(FakeTavily):
        pass
    def mock_get(url, *a, **k):
        if "raw.githubusercontent.com" in url:
            return FakeResponse("collection of citeseer tools")   # Schluesselwort, aber weder Autor, Titel noch Cora
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=ForeignReadme(repos=["https://github.com/zoo/collection"]),
                           cache_path=str(tmp_path / "c.json"))
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    assert len(result) == 1
    assert result[0].github_url is None and result[0].reason == "kein passendes Repo gefunden"
    assert result[0].feasibility == "no_repo"
    assert not any(e.get("stage") == "repo" for e in events)


def test_no_repo_at_all_is_candidate(monkeypatch, arxiv_xml, tmp_path):
    result, events, _ = _run(monkeypatch, arxiv_xml, tmp_path, FakeLLM().claim_content, repos=())
    assert len(result) == 1 and result[0].github_url is None
    assert result[0].reason == "kein passendes Repo gefunden"
    assert result[0].feasibility == "no_repo"
    assert result[0].to_dict()["feasibility"] == "no_repo"


def test_queries_contain_protocol_terms():
    from repro_agent.discovery import DISCOVERY_QUERIES, EXTRACT_QUERIES, MAX_SEARCHES_PER_DATASET
    cora = " ".join(DISCOVERY_QUERIES["Cora"]).lower()
    assert "planetoid" in cora and "140" in cora and "shchur" in cora and "largest connected component" in cora
    etth1 = " ".join(DISCOVERY_QUERIES["ETTh1"]).lower()
    assert "input length" not in etth1 and "96 192 336 720" in etth1 and "12/4/4" in etth1
    assert all(len(q) <= MAX_SEARCHES_PER_DATASET for q in DISCOVERY_QUERIES.values())
    assert "planetoid" in EXTRACT_QUERIES["Cora"].lower() and "12/4/4" in EXTRACT_QUERIES["ETTh1"]


def test_max_searches_and_dedup(monkeypatch, tmp_path):
    class TwoQ(FakeTavily):
        def __init__(self, repos=None):
            super().__init__(repos)
            self.paper_queries = []
        def curate_papers(self, query, max_results=10):
            self.paper_queries.append(query)
            return {"results": [
                {"url": "https://arxiv.org/html/2101.12345v1", "title": "Sample", "content": "Cora 95%"},
                {"url": f"https://arxiv.org/abs/2202.0000{len(self.paper_queries)}", "title": "x", "content": "Cora 95%"},
            ]}
    urls = []
    def mock_get(url, *a, **k):
        urls.append(k.get("params"))
        return FakeResponse(ATOM_ONE)
    monkeypatch.setattr("httpx.get", mock_get)
    tavily = TwoQ(repos=["https://github.com/foo/bar"])
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=tavily, cache_path=str(tmp_path / "c.json"))
    agent.discover(["Cora"])
    from repro_agent.discovery import MAX_SEARCHES_PER_DATASET
    assert len(tavily.paper_queries) == MAX_SEARCHES_PER_DATASET == 3
    assert len(set(tavily.paper_queries)) == 3
    ids = urls[0]["id_list"].split(",")
    assert ids.count("2101.12345") == 1 and len(ids) == 4


def test_prompt_contains_protocol_hints(monkeypatch, arxiv_xml, tmp_path):
    prompts = []
    class Rec(FakeLLM):
        def complete(self, role, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    agent = DiscoveryAgent(llm=Rec(), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    agent._extract_claim("text", "ETTh1")
    agent._extract_claim("text", "Cora")
    assert "8545/2881/2881" in prompts[0]
    assert "140/500/1000" in prompts[1] and "largest connected component" in prompts[1]
    # E3: Hauptergebnis der vorgeschlagenen Methode, keine Ablation/Variante; E4: Eingabelaenge frei, nur Horizont 96
    assert all("never an ablation" in p and "frozen" in p for p in prompts)
    assert "'input_length'" in prompts[0] and "'input_length'" not in prompts[1]
    assert "input length 96" not in prompts[0]


def test_old_cache_entries_without_version_are_rechecked(monkeypatch, arxiv_xml, tmp_path):
    cache_file = tmp_path / "c.json"
    cache_file.write_text(json.dumps({"2101.12345": {"status": "rejected", "reason": "Protocol mismatch"}}))
    result, events, _ = _run_with_cache(monkeypatch, arxiv_xml, str(cache_file))
    assert len(result) == 1                       # alter Reject ignoriert, Paper neu geprueft
    assert any(e.get("type") == "extracting" for e in events)
    entry = _read_cache(str(cache_file))["2101.12345"]
    assert entry["cache_version"] == CACHE_VERSION and entry["status"] == "ungeprueft"


def test_pre_column_fix_cache_v3_candidates_are_rechecked(monkeypatch, arxiv_xml, tmp_path):
    cache_file = tmp_path / 'v3.json'
    cache_file.write_text(json.dumps({'2101.12345': {'status': 'ungeprueft', 'cache_version': 3,
        'title': 'Old extraction', 'arxiv_id': '2101.12345', 'dataset': 'Cora',
        'claim': {'method': 'WrongColumn', 'target': 100}, 'reason': 'old result'}}))
    result, events, _ = _run_with_cache(monkeypatch, arxiv_xml, str(cache_file))
    assert len(result) == 1 and result[0].claim['method'] != 'WrongColumn'
    assert any(e['type'] == 'extracting' for e in events)
    assert _read_cache(str(cache_file))['2101.12345']['cache_version'] == 4


def test_old_cached_candidate_without_version_is_rechecked(monkeypatch, arxiv_xml, tmp_path):
    cache_file = tmp_path / "c.json"
    cache_file.write_text(json.dumps({"2101.12345": {"title": "Old", "arxiv_id": "2101.12345", "dataset": "Cora",
                                                     "claim": {}, "reason": "x", "github_url": None, "status": "ungeprueft"}}))
    result, events, _ = _run_with_cache(monkeypatch, arxiv_xml, str(cache_file))
    assert any(e.get("type") == "extracting" for e in events)
    assert result[0].protocol == "passt"


def _run_with_cache(monkeypatch, arxiv_xml, cache_file):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=cache_file)
    events = []
    return agent.discover(["Cora"], on_event=events.append), events, cache_file


# ---- D: Repo-Beleg, E: Methode, F: messbare Metrik ----

def _repo_agent(monkeypatch, arxiv_xml, tmp_path, urls, readme, paper_text="", method="", readmes=None):
    searched = []

    def mock_get(url, *a, **k):
        if "raw.githubusercontent.com" in url:
            for part, text in (readmes or {}).items():
                if part in url:
                    return FakeResponse(text)
            return FakeResponse(readme) if not readmes else FakeResponse("", 404)
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    tavily = FakeTavily(repos=urls)
    real = tavily.curate_repo
    tavily.curate_repo = lambda title: (searched.append(title), real(title))[1]
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=tavily, cache_path=str(tmp_path / "c.json"))
    out = agent._find_repo_and_context("Sample Paper Title", "Cora", ["John Doe"], "2101.12345", paper_text=paper_text, method=method)
    _repo_agent.searched = searched
    return out


def test_repo_dataset_match_alone_is_no_evidence(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf: TTS-Repo und Sammel-Repo mit Cora-Erwaehnung, aber ohne Titel/ID."""
    url, reason, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path,
                                 ["https://github.com/InsomaniacElf/sg-tamil-tts-resources-"],
                                 "resources for cora and text to speech. credits: doe")
    assert url is None and reason == "kein passendes Repo gefunden"


def test_repo_author_surname_in_body_is_not_author_match(monkeypatch, arxiv_xml, tmp_path):
    url, _, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/zoo/collection"],
                            "cora benchmark. thanks to doe for the idea.")
    assert url is None


def test_repo_blob_url_normalized_and_paper_link(monkeypatch, arxiv_xml, tmp_path):
    url, _, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path,
                            ["https://github.com/DaoSword/Repo/blob/main/README.md"],
                            "official implementation: code for the paper sample paper title (cora)")
    assert url == "https://github.com/DaoSword/Repo"


def test_repo_arxiv_id_is_paper_link(monkeypatch, arxiv_xml, tmp_path):
    url, _, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/a/b/issues/3"],
                            "official code, see arxiv 2101.12345 (cora)")
    assert url == "https://github.com/a/b"


def test_unofficial_repo_is_never_returned_as_authors_code(monkeypatch, arxiv_xml, tmp_path):
    url, reason, alts = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/someone/sample-reimpl"],
                                    "unofficial pytorch re-implementation of sample paper title (cora), arxiv 2101.12345")
    assert url is None and "unofficial" in reason and reason.startswith("kein passendes Repo gefunden")
    assert alts[0]["url"] == "https://github.com/someone/sample-reimpl" and alts[0]["status"] == "unofficial"
    # Paper-Beleg ohne Autor-/Besitzer-Treffer und ohne "official": ebenfalls nur inoffiziell
    url2, _, alts2 = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/someone/else"], "implements sample paper title (cora)")
    assert url2 is None and alts2[0]["status"] == "unofficial"


def test_code_link_in_paper_text_beats_search_and_keeps_subfolder(monkeypatch, arxiv_xml, tmp_path):
    text = "Abstract ... The implementation is available at: https://github.com/big-lab/collection/tree/master/ tsm . 1 Introduction"
    url, reason, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/someone/sample-reimpl"], "",
                                 paper_text=text, method="TSM", readmes={"big-lab/collection/master/tsm/README.md": "TSM, see arXiv 2101.12345"})
    assert url == "https://github.com/big-lab/collection/tree/HEAD/tsm" and "linked in the paper text" in reason
    assert _repo_agent.searched == []   # offizieller Treffer aus dem Paper: keine Suche noetig


def test_wrong_repo_linked_in_paper_footnote_is_rejected(monkeypatch, arxiv_xml, tmp_path):
    """TSMixer-Fund: Fussnote verweist auf das M5-Wettbewerbs-Repo; README nennt weder Modell noch Titel noch arXiv-ID."""
    text = "we use the repository of the third place solution 3 https://github.com/devmofl/M5_Accuracy_3rd 21"
    url, reason, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, [], "", paper_text=text, method="TSM",
                                 readmes={"devmofl/M5_Accuracy_3rd/main/README.md": "M5 forecasting accuracy, 3rd place solution with LightGBM"})
    assert url is None and reason == "kein passendes Repo gefunden"


def test_repo_author_in_owner_plus_dataset_ok(monkeypatch, arxiv_xml, tmp_path):
    url, _, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/johndoe-lab/gnn"], "cora experiments")
    assert url == "https://github.com/johndoe-lab/gnn"


def test_claim_method_must_be_in_quote(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf: Claim gehoerte einer Baseline. Methode fehlt oder steht nicht im Zitat -> Reject, nicht gecacht."""
    for method in ('"method": "", ', '"method": "OtherNet", ', ""):
        content = ('{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", ' + method +
                   '"quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}')
        result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
        assert result == []
        assert any(e.get("stage") == "quote" and "Method" in e["reason"] for e in events)
        assert "2101.12345" not in _read_cache(cache_file)
        os.remove(cache_file) if os.path.exists(cache_file) else None


@pytest.mark.parametrize("metric", ["MSE reduction percent", "relative improvement", None])
def test_unmeasurable_metric_rejected(monkeypatch, arxiv_xml, tmp_path, metric):
    content = json.dumps({"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora",
                          "method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": metric, "target": 95})
    result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert result == []
    assert any(e.get("stage") == "number" and "Metrik nicht messbar" in e["reason"] for e in events)
    assert _read_cache(cache_file)["2101.12345"]["status"] == "rejected"


def test_metric_normalization_and_allowed_per_dataset():
    from repro_agent.discovery import _normalize_metric, ALLOWED_METRICS
    assert [_normalize_metric(m) for m in ("Accuracy", "acc", "ACC", "MSE", "mae")] == ["accuracy", "accuracy", "accuracy", "mse", "mae"]
    assert _normalize_metric("MSE reduction percent") is None
    assert ALLOWED_METRICS["Cora"] == {"accuracy"} and ALLOWED_METRICS["ETTh1"] == {"mse", "mae"}
    assert "mse" not in ALLOWED_METRICS["Cora"]


def test_metric_normalized_in_candidate_claim(monkeypatch, arxiv_xml, tmp_path):
    content = ('{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", '
               '"method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "ACC", "target": 95}')
    result, _, _ = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert result[0].claim["metric"] == "accuracy" and result[0].claim["method"] == "GraphNet"


def test_prompt_demands_proposed_method_absolute_metric(monkeypatch, arxiv_xml, tmp_path):
    prompts = []
    class Rec(FakeLLM):
        def complete(self, role, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    DiscoveryAgent(llm=Rec(), tavily=FakeTavily(), cache_path=str(tmp_path / "c.json"))._extract_claim("t", "Cora")
    assert "PROPOSED" in prompts[0] and "never a baseline" in prompts[0] and "'method'" in prompts[0]
    assert "relative improvement" in prompts[0]


# ---- G: Sammlungen, H: Horizont, I: Zitat-Normalisierung ----

def test_readme_collection_is_no_repo(monkeypatch, arxiv_xml, tmp_path):
    """Live-Lauf 7: Sammel-Repo, dessen README den Titel nennt (Paper-Liste)."""
    many_arxiv = "cora sample paper title\n" + "\n".join(f"- https://arxiv.org/abs/2101.{i:05d}" for i in range(6))
    many_github = "cora sample paper title\n" + "\n".join(f"- https://github.com/u/r{i}" for i in range(11))
    for readme in (many_arxiv, many_github):
        url, reason, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/DaoSword/Collection"], readme)
        assert url is None and reason == "kein passendes Repo gefunden"


def test_readme_few_links_is_repo(monkeypatch, arxiv_xml, tmp_path):
    readme = "official code: cora sample paper title\n" + "\n".join(f"- https://arxiv.org/abs/2101.{i:05d}" for i in range(5))
    url, _, _ = _repo_agent(monkeypatch, arxiv_xml, tmp_path, ["https://github.com/a/b"], readme)
    assert url == "https://github.com/a/b"


ETT_TEXT = ("GraphNet reaches MSE 0.439 at horizon 96 on ETTh1. Average MSE 0.512 over all horizons on ETTh1. "
            "| 96 | 0.3663\\pm 0.0015 | on ETTh1. " + "Filler sentence. " * 300)


def _run_etth1(monkeypatch, tmp_path, claim):
    atom = ATOM_ONE.replace("Cora", "ETTh1")
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(atom))
    class EttTavily(FakeTavily):
        def curate_extract(self, url, query):
            return {"results": [{"raw_content": ETT_TEXT}]}
    base = {"protocol": "passt", "protocol_quote": "GraphNet reaches MSE 0.439 at horizon 96", "matched_task": "ETTh1",
            "method": "GraphNet", "metric": "MSE"}
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=json.dumps({**base, **claim})),
                           tavily=EttTavily(repos=["https://github.com/foo/bar"]), cache_path=cache_file)
    events = []
    return agent.discover(["ETTh1"], on_event=events.append), events, cache_file


def test_etth1_horizon_96_accepted(monkeypatch, tmp_path):
    result, _, _ = _run_etth1(monkeypatch, tmp_path, {"quote": "GraphNet reaches MSE 0.439 at horizon 96", "target": 0.439, "horizon": 96})
    assert len(result) == 1 and result[0].claim["metric"] == "mse"


@pytest.mark.parametrize("claim", [
    {"quote": "Average MSE 0.512 over all horizons", "target": 0.512, "horizon": 96},        # Mittelwert, Zitat ohne 96
    {"quote": "GraphNet reaches MSE 0.439 at horizon 96", "target": 0.439, "horizon": 336},  # falscher Horizont
    {"quote": "GraphNet reaches MSE 0.439 at horizon 96", "target": 0.439},                  # Horizont fehlt
    {"quote": "GraphNet reaches MSE 0.439 at horizon 96", "target": 0.439, "horizon": "avg"},
])
def test_etth1_horizon_not_96_rejected_and_cached(monkeypatch, tmp_path, claim):
    result, events, cache_file = _run_etth1(monkeypatch, tmp_path, claim)
    assert result == []
    assert any(e.get("stage") == "number" and e["reason"] == "Horizont nicht 96" for e in events)
    assert _read_cache(cache_file)["2101.12345"]["status"] == "rejected"


def test_etth1_input_length_kept_only_if_in_text(monkeypatch, tmp_path):
    ok = {"quote": "GraphNet reaches MSE 0.439 at horizon 96", "target": 0.439, "horizon": 96}
    r, _, _ = _run_etth1(monkeypatch, tmp_path, {**ok, "input_length": 336})   # 336 nicht im Text
    assert r[0].claim["input_length"] is None
    from repro_agent.discovery import _valid_input_length
    assert _valid_input_length(512, "look-back window 512 used") == 512
    assert _valid_input_length("512", "look-back 512") == 512
    assert _valid_input_length(512, "value 1512.5") is None          # Teilzahl zaehlt nicht
    assert _valid_input_length(1000, "input 1000") is None           # ueber 720
    assert _valid_input_length(None, "x") is None and _valid_input_length("n/a", "x") is None
    assert _valid_input_length(True, "1") is None


def test_cora_claim_needs_no_horizon(monkeypatch, arxiv_xml, tmp_path):
    result, _, _ = _run(monkeypatch, arxiv_xml, tmp_path, FakeLLM().claim_content)
    assert len(result) == 1


def test_prompt_horizon_only_for_etth1():
    class Rec(FakeLLM):
        prompts = []
        def complete(self, role, messages, **kwargs):
            Rec.prompts.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)
    agent = DiscoveryAgent(llm=Rec(), tavily=FakeTavily())
    agent._extract_claim("t", "ETTh1")
    agent._extract_claim("t", "Cora")
    assert "'horizon'" in Rec.prompts[0] and "never an average" in Rec.prompts[0]
    assert "'horizon'" not in Rec.prompts[1]


def test_quote_normalization_table_row():
    from repro_agent.discovery import _norm_quote
    row = "| 96 | 0.3663\\pm 0.0015 |"
    assert _norm_quote("96 0.3663 ± 0.0015") in _norm_quote("Table 3:\n" + row + "\n| 192 | 0.4 |")
    assert _norm_quote("$0.3663 +- 0.0015$") == _norm_quote("0.3663 ± 0.0015") == _norm_quote("0.3663\\pm0.0015")


def test_quote_normalization_is_not_fuzzy():
    from repro_agent.discovery import _norm_quote
    assert _norm_quote("96 0.3664 ± 0.0015") not in _norm_quote("| 96 | 0.3663\\pm 0.0015 |")
    assert _norm_quote("96 0.3663 0.0015") not in _norm_quote("| 96 | 0.3663\\pm 0.0015 |")


# ---- J: Methode im Spaltenkopf, K: Community-Bezug ----

HEAD = "Table 3: results on ETTh1.\nModel | HaKAN | RLinear\n"
ROW = "| 96 | 0.366 | 0.386 |\n"


def _run_table(monkeypatch, tmp_path, text, quote="| 96 | 0.366 | 0.386 |"):
    atom = ATOM_ONE.replace("Cora", "ETTh1")
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(atom))
    class T(FakeTavily):
        def curate_extract(self, url, query):
            return {"results": [{"raw_content": text}]}
    claim = {"protocol": "passt", "protocol_quote": quote, "matched_task": "ETTh1", "method": "HaKAN", "metric": "MSE",
             "horizon": 96, "quote": quote, "target": 0.366}
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=json.dumps(claim)), tavily=T(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    events = []
    return agent.discover(["ETTh1"], on_event=events.append), events, str(tmp_path / "c.json")


def test_method_in_table_header_is_supported(monkeypatch, tmp_path):
    result, _, _ = _run_table(monkeypatch, tmp_path, HEAD + ROW + "Filler sentence. " * 300)
    assert len(result) == 1 and result[0].claim["method"] == "HaKAN"


def test_method_too_far_before_quote_rejected(monkeypatch, tmp_path):
    text = HEAD + "Filler sentence. " * 300 + ROW + "Filler sentence. " * 50     # ca. 5100 Zeichen dazwischen
    result, events, cache_file = _run_table(monkeypatch, tmp_path, text)
    assert result == []
    assert any(e.get("stage") == "quote" and "Method" in e["reason"] for e in events)
    assert "2101.12345" not in _read_cache(cache_file)


def test_method_after_quote_not_supported():
    from repro_agent.discovery import _method_supported
    assert not _method_supported("HaKAN", "| 96 | 0.366 |", "| 96 | 0.366 |\nlater HaKAN is mentioned")
    assert _method_supported("HaKAN", "| 96 | 0.366 |", "Model | HaKAN\n| 96 | 0.366 |")
    assert not _method_supported("", "q", "q")


def test_community_hits_without_relation_dropped(monkeypatch, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(ATOM_ONE))
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=TavilyFinder(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    assert [c["url"] for c in result[0].community] == ["https://github.com/foo/bar/issues/12"]
    assert [e for e in events if e.get("type") == "candidate"][0]["community_hits"] == 1


def test_community_related_rules():
    from repro_agent.discovery import _community_related
    t = "Stuart-Landau Graph Convolutional Networks for Node Classification"
    assert _community_related(t, "2511.08094", "x", "see arXiv 2511.08094 issue")                      # arXiv-ID
    assert _community_related(t, "2511.08094", "Stuart-Landau graph convolutional networks for node classification", "")  # Titel
    assert _community_related(t, "2511.08094", "Stuart Landau convolutional networks", "node")           # 5 von 7 Woertern
    assert not _community_related(t, "2511.08094", "fMRI dynamics", "Stuart-Landau oscillators in the brain")


# ---- UI-Felder: purpose, arxiv_ids, Cache-Kandidaten ----

def test_tavily_calls_carry_purpose_and_arxiv_ids(monkeypatch, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(ATOM_ONE))
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=TavilyFinder(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    events = []
    agent.discover(["Cora"], on_event=events.append)
    calls = [e for e in events if e.get("type") == "tavily_call"]
    by_purpose = {}
    for c in calls:
        by_purpose.setdefault(c["purpose"], []).append(c)
    assert set(by_purpose) == {"find_papers", "read_paper", "find_code", "community"}
    assert by_purpose["find_papers"][0]["arxiv_ids"] == ["2101.12345"]      # nur echte arXiv-IDs, dedupliziert
    assert all("arxiv_ids" not in c for c in calls if c["purpose"] != "find_papers")
    assert by_purpose["read_paper"][0]["endpoint"] == "extract"


def test_cache_hit_emits_candidate_event(monkeypatch, tmp_path):
    cache_file = tmp_path / "c.json"
    cache_file.write_text(json.dumps({"1234.5678": {
        "title": "Cached Paper", "arxiv_id": "1234.5678", "dataset": "Cora", "claim": {"method": "M", "target": 80.0},
        "reason": "Cached", "github_url": None, "status": "ungeprueft", "cache_version": CACHE_VERSION, "protocol": "passt",
        "community": [{"url": "u"}]}}))
    monkeypatch.setattr("httpx.get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=FakeTavily(), cache_path=str(cache_file))
    agent._search_arxiv = lambda ds, max_results=30: [{"arxiv_id": "1234.5678", "title": "Cached Paper", "abstract": "x", "authors": []}]
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    cand = [e for e in events if e.get("type") == "candidate"]
    assert len(result) == 1 and len(cand) == 1
    assert cand[0]["cached"] is True and cand[0]["title"] == "Cached Paper"
    assert cand[0]["claim"] == {"method": "M", "target": 80.0} and cand[0]["protocol"] == "passt"
    assert cand[0]["community_hits"] == 1 and cand[0]["dataset"] == "Cora"


def test_llm_call_event_cached_costs_nothing(monkeypatch, arxiv_xml, tmp_path):
    from repro_agent.llm import LLMResponse
    class CachedLLM:
        def complete(self, *a, **k):
            return LLMResponse(content=FakeLLM().claim_content, model="fake", prompt_tokens=100, completion_tokens=50,
                               cost_usd=0.00385, cached=True, finish_reason="stop")
    class PaidLLM(CachedLLM):
        def complete(self, *a, **k):
            r = super().complete()
            return LLMResponse(content=r.content, model="fake", prompt_tokens=100, completion_tokens=50,
                               cost_usd=0.00385, cached=False, finish_reason="stop")
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    seen = {}
    for name, llm in (("cached", CachedLLM()), ("paid", PaidLLM())):
        agent = DiscoveryAgent(llm=llm, tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / f"{name}.json"))
        events = []
        agent.discover(["Cora"], on_event=events.append)
        seen[name] = [e for e in events if e.get("type") == "llm_call"][0]
    assert seen["cached"]["cached"] is True and seen["cached"]["cost_usd"] == 0.0
    assert seen["paid"]["cached"] is False and seen["paid"]["cost_usd"] == 0.00385


def test_fresh_candidate_event_carries_card_fields(monkeypatch, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse("Official implementation of Sample Paper Title (cora)" if "raw.githubusercontent" in url else ATOM_ONE))
    agent = DiscoveryAgent(llm=FakeLLM(), tavily=TavilyFinder(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / "c.json"))
    events = []
    agent.discover(["Cora"], on_event=events.append)
    cand = [e for e in events if e.get("type") == "candidate"][0]
    assert cand["cached"] is False and cand["title"] == "Sample Paper Title" and cand["dataset"] == "Cora"
    assert cand["claim"]["method"] == "GraphNet" and cand["claim"]["matched_task"] == "Cora"
    assert cand["github_url"] == "https://github.com/foo/bar"
    assert cand["source_url"] == "https://arxiv.org/html/2101.12345v1"
    assert cand["protocol"] == "passt" and cand["community_hits"] == 1


def test_candidate_event_without_repo_has_null_github_url(monkeypatch, arxiv_xml, tmp_path):
    result, events, _ = _run(monkeypatch, arxiv_xml, tmp_path, FakeLLM().claim_content, repos=())
    cand = [e for e in events if e.get("type") == "candidate"][0]
    assert cand["github_url"] is None and cand["feasibility"] == "no_repo"


# ---- Y1 zeilenweise Zitate, Y2 Spaltenpruefung, Y3 zweiter LLM-Versuch ----

TABLE = ("Table 3: results on ETTh1.\n| Methods | HaKAN | RLinear |\n| Metric | MSE | MAE | MSE | MAE |\n"
         "| ETTh1 96 | 0.369 | 0.394 | 0.376 | 0.395 |\n")
ROW_LINE = "| ETTh1 96 | 0.369 | 0.394 | 0.376 | 0.395 |"


def _run_claim(monkeypatch, tmp_path, text, **claim):
    atom = ATOM_ONE.replace("Cora", "ETTh1")
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(atom))
    class T(FakeTavily):
        def curate_extract(self, url, query):
            return {"results": [{"raw_content": text + "Filler sentence. " * 300}]}
    base = {"protocol": "passt", "protocol_quote": "Table 3: results on ETTh1.", "matched_task": "ETTh1", "method": "HaKAN",
            "metric": "MSE", "horizon": 96, "quote": ROW_LINE, "target": 0.369}
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=json.dumps({**base, **claim})), tavily=T(repos=["https://github.com/foo/bar"]),
                           cache_path=cache_file)
    events = []
    return agent.discover(["ETTh1"], on_event=events.append), events, cache_file


def test_multiline_quote_header_far_before_row_ok(monkeypatch, tmp_path):
    """Kopfzeile steht 300 Zeichen vor der Werte-Zeile, nicht direkt davor."""
    text = ("| (ours) | HaKAN | RLinear |\n| Metric | MSE | MAE |\n" + "Unrelated sentence. " * 15
            + "\n| ETTh1 96 | 0.369 | 0.394 |\n")
    quote = "| (ours) | HaKAN | RLinear |\n| Metric | MSE | MAE |\n| ETTh1 96 | 0.369 | 0.394 |"
    result, _, _ = _run_claim(monkeypatch, tmp_path, text, quote=quote)
    assert len(result) == 1 and result[0].claim["column_check"] == "unchecked"


def test_multiline_quote_with_invented_line_rejected(monkeypatch, tmp_path):
    text = "| Metric | MSE | MAE |\n| ETTh1 96 | 0.369 | 0.394 |\n"
    quote = "| Metric | MSE | MAE |\n| HaKAN made up row | 1 | 2 |\n| ETTh1 96 | 0.369 | 0.394 |"
    result, events, cache_file = _run_claim(monkeypatch, tmp_path, text, quote=quote)
    assert result == []
    assert any(e.get("reason") == "Quote not found in text" for e in events)


def test_method_window_anchored_at_number_line():
    from repro_agent.discovery import _method_supported
    text = "| Model | HaKAN |\n" + "x " * 100 + "\n| ETTh1 96 | 0.369 |\n" + "y " * 3000 + "| HaKAN |"
    quote = "| Metric | MSE |\n| ETTh1 96 | 0.369 |"
    assert not _method_supported("HaKAN", quote, text)                       # Zitatanfang steht nicht im Text
    assert _method_supported("HaKAN", "| ETTh1 96 | 0.369 |", text, "| ETTh1 96 | 0.369 |")


def test_column_ok_when_number_in_method_column(monkeypatch, tmp_path):
    result, _, _ = _run_claim(monkeypatch, tmp_path, TABLE)
    assert len(result) == 1 and result[0].claim["column_check"] == "ok"


def test_column_mismatch_when_number_in_baseline_column(monkeypatch, tmp_path):
    result, events, cache_file = _run_claim(monkeypatch, tmp_path, TABLE, target=0.376)   # RLinear-MSE
    assert result == []
    rej = [e for e in events if e.get("type") == "reject"][0]
    assert rej["stage"] == "number" and rej["reason"].startswith("Number belongs to another column")
    assert _read_cache(cache_file)["2101.12345"]["status"] == "rejected"


def test_column_unchecked_without_header(monkeypatch, tmp_path):
    text = "Table 3: HaKAN results on ETTh1.\n| ETTh1 96 | 0.369 | 0.394 |\n"
    result, _, _ = _run_claim(monkeypatch, tmp_path, text, quote="| ETTh1 96 | 0.369 | 0.394 |")
    assert len(result) == 1 and result[0].claim["column_check"] == "unchecked"


class SeqLLM:
    """Antwortet der Reihe nach; merkt sich Temperaturen."""
    def __init__(self, contents):
        self.contents, self.temps = list(contents), []
    def complete(self, role, messages, temperature=0.0, **kwargs):
        from repro_agent.llm import LLMResponse
        self.temps.append(temperature)
        return LLMResponse(content=self.contents.pop(0), model="fake", prompt_tokens=1, completion_tokens=1,
                           cost_usd=0.001, cached=False, finish_reason="stop")


def _run_seq(monkeypatch, arxiv_xml, tmp_path, contents):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    llm = SeqLLM(contents)
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=llm, tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=cache_file)
    events = []
    result = agent.discover(["Cora"], on_event=events.append)
    return result, events, llm, cache_file


def test_empty_first_answer_retried_with_other_temperature(monkeypatch, arxiv_xml, tmp_path):
    result, events, llm, _ = _run_seq(monkeypatch, arxiv_xml, tmp_path, ["{}", FakeLLM().claim_content])
    assert len(result) == 1
    assert len([e for e in events if e.get("type") == "llm_call"]) == 2
    assert llm.temps == [0.0, 0.3]


def test_unreadable_first_answer_retried(monkeypatch, arxiv_xml, tmp_path):
    result, _, llm, _ = _run_seq(monkeypatch, arxiv_xml, tmp_path, ["no json at all", FakeLLM().claim_content])
    assert len(result) == 1 and len(llm.temps) == 2


def test_two_empty_answers_reject_and_cache(monkeypatch, arxiv_xml, tmp_path):
    result, events, llm, cache_file = _run_seq(monkeypatch, arxiv_xml, tmp_path, ["{}", "{}"])
    assert result == [] and len(llm.temps) == 2               # hoechstens eine Wiederholung
    assert any(e.get("reason") == "No numerical claim found" for e in events)
    assert len([e for e in events if e.get("type") == "llm_call"]) == 2
    assert _read_cache(cache_file)["2101.12345"]["status"] == "rejected"


def test_valid_first_answer_has_no_retry(monkeypatch, arxiv_xml, tmp_path):
    result, events, llm, _ = _run_seq(monkeypatch, arxiv_xml, tmp_path, [FakeLLM().claim_content])
    assert len(result) == 1 and llm.temps == [0.0]
    assert len([e for e in events if e.get("type") == "llm_call"]) == 1


# ---- Z1: passt_nicht gegen einen Task, obwohl ein anderer unterstuetzter Task passt ----

SHCHUR_REASON = "uses Shchur et al. 2018 random splits, which differ from the Planetoid split required for the Cora task"


def test_passt_nicht_naming_other_supported_protocol_is_unklar_not_cached(monkeypatch, arxiv_xml, tmp_path):
    content = ('{"protocol": "passt_nicht", "protocol_quote": "GraphNet achieves 95% accuracy", "protocol_reason": "' + SHCHUR_REASON
               + '", ' + CLAIM + '}')
    result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert len(result) == 1 and result[0].protocol == "unklar"
    assert not any(e.get("stage") == "protocol" for e in events)
    assert _read_cache(cache_file)["2101.12345"]["status"] == "ungeprueft"          # kein gecachter Reject
    assert "unterstuetztes Protokoll" in result[0].claim["protocol_reason"]


@pytest.mark.parametrize("reason", ["uses the largest connected component", "Planetoid public split not used",
                                    "random splits per seed"])
def test_passt_nicht_keywords_per_task(monkeypatch, arxiv_xml, tmp_path, reason):
    content = ('{"protocol": "passt_nicht", "protocol_quote": "GraphNet achieves 95% accuracy", "protocol_reason": "'
               + reason + '", ' + CLAIM + '}')
    result, _, _ = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert len(result) == 1 and result[0].protocol == "unklar"


def test_passt_nicht_with_foreign_protocol_still_rejected(monkeypatch, arxiv_xml, tmp_path):
    content = ('{"protocol": "passt_nicht", "protocol_quote": "GraphNet achieves 95% accuracy", '
               '"protocol_reason": "uses 60/20/20 splits with variable training ratios", ' + CLAIM + '}')
    result, events, cache_file = _run(monkeypatch, arxiv_xml, tmp_path, content)
    assert result == [] and any(e.get("stage") == "protocol" for e in events)
    assert _read_cache(cache_file)["2101.12345"]["status"] == "rejected"


def test_prompt_passt_nicht_only_if_no_supported_protocol_matches():
    class Rec(FakeLLM):
        prompts = []
        def complete(self, role, messages, **kwargs):
            Rec.prompts.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)
    DiscoveryAgent(llm=Rec(), tavily=FakeTavily())._extract_claim("t", "Cora")
    assert "matches NONE of the supported protocols" in Rec.prompts[0]
    assert "answer \"passt\" with that task" in Rec.prompts[0]


# ---- Machbarkeit (CPU) und Repo-Markierung ----

def _run_with_llm(monkeypatch, arxiv_xml, tmp_path, llm):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=llm, tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=cache_file)
    events = []
    return agent.discover(["Cora"], on_event=events.append), events, cache_file


def test_infeasible_answer_is_clean_reject_and_cached(monkeypatch, arxiv_xml, tmp_path):
    from repro_agent.discovery import INFEASIBLE_REASON
    llm = FakeLLM(claim_content='{"infeasible": "pretrained foundation model"}')
    result, events, cache_file = _run_with_llm(monkeypatch, arxiv_xml, tmp_path, llm)
    assert result == []
    rej = [e for e in events if e.get("type") == "reject"]
    assert any(e["reason"] == INFEASIBLE_REASON and e["stage"] == "feasibility" for e in rej)
    assert not any(e["reason"] in ("LLM-Antwort nicht lesbar", "No numerical claim found") for e in rej)
    cached = _read_cache(cache_file)
    assert cached["2101.12345"]["status"] == "rejected" and cached["2101.12345"]["reason"] == INFEASIBLE_REASON


def test_prompt_has_feasibility_rule(monkeypatch, arxiv_xml, tmp_path):
    class Spy(FakeLLM):
        def complete(self, role, messages, **kw):
            Spy.prompt = messages[0]["content"]
            return super().complete(role, messages, **kw)
    _run_with_llm(monkeypatch, arxiv_xml, tmp_path, Spy())
    assert "FEASIBILITY RULE" in Spy.prompt and "infeasible" in Spy.prompt


def test_old_hard_no_repo_reject_is_rechecked(monkeypatch, arxiv_xml, tmp_path):
    cache_file = tmp_path / "c.json"
    cache_file.write_text(json.dumps({"2101.12345": {"status": "rejected", "reason": "kein passendes Repo gefunden", "cache_version": 3}}))
    result, events, _ = _run(monkeypatch, arxiv_xml, tmp_path, FakeLLM().claim_content, repos=())
    assert len(result) == 1 and result[0].feasibility == "no_repo"


def test_papers_mentioning_code_come_first():
    papers = [{"abstract": "x", "snippet": ""}, {"abstract": "Code is available at https://github.com/a/b", "snippet": ""}]
    assert DiscoveryAgent._prefer_code(papers)[0] is papers[1]


def test_search_covers_all_datasets_and_stays_bounded():
    from repro_agent.discovery import DISCOVERY_QUERIES, MAX_SEARCHES_PER_DATASET
    assert set(DISCOVERY_QUERIES) == {"Cora", "CiteSeer", "ETTh1"}
    assert all(1 <= len(q) <= MAX_SEARCHES_PER_DATASET <= 3 for q in DISCOVERY_QUERIES.values())
    assert any("github" in q for q in DISCOVERY_QUERIES["Cora"])
    assert any("linear" in q.lower() for q in DISCOVERY_QUERIES["ETTh1"])


GOOD_CLAIM = ('{"protocol": "passt", "protocol_quote": "GraphNet achieves 95% accuracy", "matched_task": "Cora", '
              '"method": "GraphNet", "quote": "GraphNet achieves 95% accuracy", "metric": "accuracy", "target": 95}')


def _seed_agent(tmp_path, claim=GOOD_CLAIM, repos=None):
    cache_file = tmp_path / "seed_cache.json"
    return DiscoveryAgent(llm=FakeLLM(claim_content=claim), tavily=FakeTavily(repos=repos or ["https://github.com/foo/bar"]),
                          cache_path=str(cache_file)), cache_file


def test_seeded_found_by_and_cache(monkeypatch, arxiv_xml, tmp_path):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeResponse(arxiv_xml))
    agent, cache_file = _seed_agent(tmp_path)
    cands = agent.discover(["Cora"], seed_ids=["2101.12345"])
    assert [c.found_by for c in cands] == ["seeded"]
    cached = json.loads(cache_file.read_text())["2101.12345"]
    assert cached["found_by"] == "seeded"


def test_seeded_skips_search_but_keeps_checks(monkeypatch, arxiv_xml, tmp_path):
    # Abstract does not name the dataset: search pre-filter would reject, seeded skips it.
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeResponse(arxiv_xml.replace("Cora dataset", "a graph dataset")))
    agent, _ = _seed_agent(tmp_path)

    def boom(*a, **k):
        raise AssertionError("search must not run")
    agent.tavily.curate_papers = boom
    cands = agent.discover(["Cora"], seed_ids=["2101.12345"])
    assert len(cands) == 1

    # Quote check still applies
    agent2, cache2 = _seed_agent(tmp_path, claim=GOOD_CLAIM.replace('"quote": "GraphNet achieves 95% accuracy"', '"quote": "invented 95"'))
    cache2.unlink(missing_ok=True)
    events = []
    assert agent2.discover(["Cora"], seed_ids=["2101.12345"], on_event=events.append) == []
    assert any(e.get("type") == "reject" and e.get("stage") == "quote" for e in events)


def test_seeded_not_capped_by_search_limit(monkeypatch, tmp_path):
    entries = "".join(f"<entry><id>http://arxiv.org/abs/2101.0000{i}v1</id><title>T{i}</title><summary>s</summary>"
                      f"<author><name>A B</name></author></entry>" for i in range(5))
    xml = f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeResponse(xml))
    agent, _ = _seed_agent(tmp_path)
    cands = agent.discover(["Cora"], seed_ids=[f"2101.0000{i}" for i in range(5)])
    assert len(cands) == 5


def test_parse_abs_page():
    from repro_agent.discovery import _parse_abs_page
    html = ('<meta name="citation_title" content="My &amp; Paper" /><meta name="citation_author" content="Doe, John" />'
            '<meta property="og:description" content="Abstract text." />')
    p = _parse_abs_page("2101.00001", html)
    assert p["title"] == "My & Paper" and p["authors"] == ["John Doe"] and p["abstract"] == "Abstract text."
