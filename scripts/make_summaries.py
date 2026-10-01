"""Plain-language paper summaries for Discover (web/public/data/discovery/summaries.json).

One Nemotron call per paper, generated once and stored; the web app never calls the model per page view.
Input is only the arXiv abstract (arxiv.org abstract page, no Tavily credits) and the claim found by Discovery
(method, metric, value, dataset). Output: two to three simple sentences for readers without ML background.

Usage: LLM_PROVIDER=tokenfactory .venv/bin/python scripts/make_summaries.py [--force]
"""

import html
import json
import re
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from repro_agent.llm import LLMClient  # noqa: E402

DATA = REPO_ROOT / "web" / "public" / "data" / "discovery"
OUT = DATA / "summaries.json"
ROLE = "code"   # Nemotron Super; the 'fast' model writes its reasoning into the answer
DATASETS = {
    "Cora": "Cora, a collection of scientific articles that cite each other",
    "Cora-LCC": "Cora, a collection of scientific articles that cite each other",
    "ETTh1": "ETTh1, hourly measurements from an electricity transformer",
}
SYSTEM = "You explain machine-learning papers to readers without any technical background. Keep reasoning brief."


def abstract(arxiv_id: str) -> str:
    """Abstract from the arXiv abstract page (the export API often answers 429 or times out)."""
    r = httpx.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=30, follow_redirects=True)
    r.raise_for_status()
    m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', r.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", m.group(1)) if m else ""
    return html.unescape(" ".join(text.split())).removeprefix("Abstract:").strip()


def prompt(c: dict, abs_text: str) -> str:
    claim = c["claim"]
    unit = "% accuracy" if claim["metric"] == "accuracy" else f" {claim['metric'].upper()} (lower is better)"
    return (
        f"Paper title: {c['title']}\nAbstract: {abs_text}\n\n"
        f"The claim we check: the method {claim['method']} reaches {claim['target']}{unit} on "
        f"{DATASETS.get(claim.get('matched_task') or c.get('dataset', ''), c.get('dataset', 'a benchmark'))}.\n\n"
        "Write exactly three short sentences for a reader who has never studied computer science: (1) what the paper "
        "proposes, in everyday words; (2) what the data is; (3) what the claimed number means (say what accuracy or "
        "forecast error means in plain words). Avoid jargon such as neural network, graph, oscillator, amplitude, phase, "
        "MSE, embedding; describe the idea instead. Each sentence at most 25 words. Do not judge whether the claim is true. "
        "Use only facts from the title, abstract and claim above. Style example: \"This paper proposes a new way for a "
        "computer to sort scientific articles into topics by looking at which articles cite each other. The data is "
        "Cora, a collection of about 2,700 articles. It claims the method puts 81.5% of the articles in the right topic.\" "
        'End your answer with one line of JSON: {"summary": "..."}'
    )


def parse(text: str) -> str | None:
    for m in reversed(list(re.finditer(r"\{[^{}]*\"summary\"[^{}]*\}", text, flags=re.S))):
        try:
            s = json.loads(m.group(0))["summary"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
        if 40 <= len(s) <= 700:
            return s
    return None


def main() -> None:
    force = "--force" in sys.argv
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    cands = [json.loads(x) for x in (DATA / "replay.jsonl").read_text().splitlines() if x.strip()]
    cands = [c for c in cands if c.get("type") == "candidate"]
    llm = LLMClient()
    for c in cands:
        aid = c["arxiv_id"]
        if aid in out and not out[aid].get("example") and not force:
            print(f"{aid}: vorhanden, uebersprungen")
            continue
        p = prompt(c, abstract(aid))
        summary = None
        for t in (0.2, 0.5):
            resp = llm.complete(ROLE, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                temperature=t, max_tokens=3000)
            summary = parse(resp.content or "")
            if summary:
                break
        if not summary:
            print(f"{aid}: keine gueltige Antwort, Beispieltext bleibt")
            continue
        out[aid] = {"summary": summary, "source": f"arXiv abstract ({aid}) and the claim found by Discovery",
                    "generated_by": "Nemotron Super (Token Factory)"}
        print(f"{aid}: {summary}")
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
