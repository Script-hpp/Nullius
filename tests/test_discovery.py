import pytest
import json
import os
from repro_agent.discovery import DiscoveryAgent, CandidatePaper
from repro_agent.tavily_tools import TavilyTools
from repro_agent.config import Settings

class FakeLLM:
    def __init__(self, claim_content=None):
        if claim_content is None:
            self.claim_content = '{"protocol_match": true, "matched_task": "Cora", "quote": "95% accuracy", "metric": "acc", "target": 95}'
        else:
            self.claim_content = claim_content
    def complete(self, role, messages, temperature=0.0, **kwargs):
        class Resp:
            content = self.claim_content
        return Resp()

# Discovery verlangt lesbaren Volltext (MIN_FULLTEXT_CHARS), Abstract allein reicht nicht.
FULLTEXT = "This is a sample abstract mentioning Cora dataset. It achieves 95% accuracy. " + "Filler sentence. " * 300


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

    
    
    # 1. protocol_match false
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": false}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "protocol" in e.get("reason", "").lower()]
    assert len(rejects) >= 1
    
    # 2. Empty quote
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "", "target": 95}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "no quote provided" in e.get("reason", "").lower()]
    assert len(rejects) >= 1

    # 3. Quote not in text
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "I made this up 95", "target": 95}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "quote not found in text" in e.get("reason", "").lower()]
    assert len(rejects) >= 1

    # 4. Target not in quote
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "It achieves 95% accuracy", "target": 99}'), tavily=FakeTavily(), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    rejects = [e for e in events if e.get("type") == "reject" and "target value not found numerically in quote" in e.get("reason", "").lower()]
    assert len(rejects) >= 1
    
    # 5. Success case
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "It achieves 95% accuracy", "target": 95}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(tmp_path / f"cache_{__import__('uuid').uuid4()}.json"))
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
                "claim": "80% acc",
                "reason": "Cached",
                "github_url": "https://github.com/foo/bar",
                "status": "ungeprueft",
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
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "It achieves 95% accuracy", "target": 95.0}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    candidates = [e for e in events if e.get("type") == "candidate"]
    assert len(candidates) >= 1

    cache_file = tmp_path / "c2.json"
    # Target 0.95, quote has 95%
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "It achieves 95% accuracy", "target": 0.95}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
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
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": false}'), tavily=tavily, cache_path=str(cache_file))
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
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "95"}'), tavily=FakeTavily(), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any("Target missing" in e.get("reason", "") for e in events if e.get("type") == "reject")

    # Null target
    cache_file = tmp_path / "c2.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "95", "target": null}'), tavily=FakeTavily(), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any("Target missing" in e.get("reason", "") for e in events if e.get("type") == "reject")

    # String target with %
    cache_file = tmp_path / "c3.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "95%", "target": "95%"}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any(e.get("type") == "candidate" for e in events)

    # String target bad parse
    cache_file = tmp_path / "c4.json"
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "95%", "target": "ninety-five"}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
    events = []
    agent.discover(["Cora"], on_event=lambda e: events.append(e))
    assert any("Target missing" in e.get("reason", "") for e in events if e.get("type") == "reject")


def test_on_call_events(monkeypatch, arxiv_xml, tmp_path):
    def mock_get(url, *args, **kwargs):
        return FakeResponse(arxiv_xml)
    monkeypatch.setattr("httpx.get", mock_get)
    cache_file = tmp_path / "c99.json"
    
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "95%", "target": "95%"}'), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=str(cache_file))
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

    # protocol_match false: kein Kandidat, sonst stoppt die Grenze von 2 Kandidaten vor dem Deckel
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": false}'), tavily=FakeTavily(), cache_path=str(tmp_path / "c_l6.json"))
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
               '{"protocol_match": true, "matched_task": "Cora", "quote": "It achieves 95% accuracy", "target": 95}')
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
    agent = DiscoveryAgent(llm=FakeLLM(claim_content='{"protocol_match": true, "matched_task": "Cora", "quote": "It achieves 95% accuracy", "target": 95}'),
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
        return {"results": [{"url": "https://github.com/foo/bar/issues/12", "title": "Cannot reproduce 95%", "content": "Seed dependent"}]}


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
    atom = ATOM_ONE.replace("This is a sample abstract mentioning Cora dataset. It achieves 95% accuracy.",
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
    content = '{"protocol_match": true, "matched_task": "Cora-LCC", "quote": "It achieves 95% accuracy", "target": 95}'
    agent = DiscoveryAgent(llm=RecordingLLM(claim_content=content), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    result = agent.discover(["Cora"])
    assert "[Cora]" in prompts[0] and "[Cora-LCC]" in prompts[0]
    assert result and result[0].claim["matched_task"] == "Cora-LCC"


def test_unknown_matched_task_rejected_not_cached(monkeypatch, arxiv_xml, tmp_path):
    monkeypatch.setattr("httpx.get", lambda url, *a, **k: FakeResponse(arxiv_xml))
    content = '{"protocol_match": true, "matched_task": "PubMed", "quote": "It achieves 95% accuracy", "target": 95}'
    cache_file = str(tmp_path / "c.json")
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=content), tavily=FakeTavily(repos=["https://github.com/foo/bar"]), cache_path=cache_file)
    events = []
    assert agent.discover(["Cora"], on_event=events.append) == []
    assert any(e.get("stage") == "protocol" and "PubMed" in e["reason"] for e in events)
    assert "2101.12345" not in _read_cache(cache_file)


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
    content = '{"protocol_match": true, "matched_task": " [Cora-LCC] ", "quote": "It achieves 95% accuracy", "target": 95}'
    agent = DiscoveryAgent(llm=FakeLLM(claim_content=content), tavily=FakeTavily(repos=["https://github.com/foo/bar"]),
                           cache_path=str(tmp_path / "c.json"))
    result = agent.discover(["Cora"])
    assert result and result[0].claim["matched_task"] == "Cora-LCC"
