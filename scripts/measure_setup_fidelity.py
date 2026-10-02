"""v5 Schritt A: Dossier-Treue von Rolle 4 messen (Vorab-Festlegung: benchmark/experiments/v5_step_a.md, Soll-Liste und Muster:
benchmark/experiments/v5_step_a_soll.yaml). Nur Rolle 4 (Extraktion), Arm B, Tavily-Fixtures und Repo-Dateien aus Fixtures, LLM-Cache aus,
kein Training, keine Sandbox, kein Tavily-Aufruf.

  uv run python scripts/measure_setup_fidelity.py --dry-run                       # Einheiten, Fixtures, geschaetzte LLM-Aufrufe
  uv run python scripts/measure_setup_fidelity.py --check-soll                    # Soll-Punkte gegen die gespeicherten Quellen
  LLM_PROVIDER=tokenfactory uv run python scripts/measure_setup_fidelity.py --label basis --reps 3 [--units a,b] [--setup-v2]
  uv run python scripts/measure_setup_fidelity.py --score basis [--vs vorher]     # deterministisch, ohne LLM
Rohdaten: runs/v5_step_a/<label>/<unit>_<rep>.json (Angaben, Plan-Text, Kosten).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLL_FILE = REPO_ROOT / "benchmark" / "experiments" / "v5_step_a_soll.yaml"
PAPERS_YAML = REPO_ROOT / "benchmark" / "papers.yaml"
FIXTURES = REPO_ROOT / "benchmark" / "fixtures"
OUT_ROOT = REPO_ROOT / "runs" / "v5_step_a"
NORM_POINTS = ("feature_norm", "method")   # Ziel 1: Normalisierung bzw. Methode (NLinear)
GOAL1_MIN_UNITS, GOAL1_MIN_REPS, GOAL2_SHARE = 5, 2, 0.80
WINDOW = 80   # --check-soll: Stichwort und Wert muessen im Abstand von hoechstens WINDOW Zeichen stehen (naechster Treffer je Dokument)
DEFAULT_MAX_GROSS_USD = 3.0


class Forbidden:
    """Ersatz fuer Sandbox und Tavily: jeder Zugriff bricht ab (Messung darf weder klonen noch Tavily aufrufen)."""

    def __getattr__(self, name):
        raise RuntimeError(f"Zugriff {name} verboten: fehlende Fixture, nicht live holen (melden)")


def load_soll(path: Path = SOLL_FILE) -> dict:
    return yaml.safe_load(path.read_text())


def paper_for(unit: str) -> dict:
    papers = yaml.safe_load(PAPERS_YAML.read_text())["papers"]
    return next(p for p in papers if p["id"] == unit)


def claim_for(unit: str, claim_id: str) -> dict:
    return next(c for c in paper_for(unit)["claims"] if c["id"] == claim_id)


# ---- Quellen aus Fixtures (spiegelt Orchestrator._gather_docs, ohne Aufrufe) ------------------------------------------------
def fixture_sources(unit: str, fixtures: Path = FIXTURES) -> tuple[dict[str, str], list[str]]:
    """({Dokumentname: Text}, fehlende Fixtures). Reihenfolge wie im Lauf: Paper, README, Repo-Dateien."""
    from repro_agent.repo_source import _repo_key, find_repo_url

    missing: list[str] = []
    paper = paper_for(unit)
    arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
    docs: dict[str, str] = {}
    pf = fixtures / f"{arxiv_id}.json"
    if not pf.exists():
        return docs, [f"{pf.name} (Paper-Text)"]
    text = json.loads(pf.read_text())["text"]
    docs["paper"] = text
    url = find_repo_url(text)
    if not url:   # Produktpfad: Rolle 1, Suche (Fixture search_<id>.json)
        sf = fixtures / "repos" / f"search_{arxiv_id}.json"
        url = json.loads(sf.read_text()).get("url") if sf.exists() else None
        if not url:
            return docs, missing + ["kein Repo-Link im Paper und keine search_*-Fixture"]
    key = _repo_key(url)
    rf, df = fixtures / "repos" / f"{key}_readme.json", fixtures / "repos" / f"{key}.json"
    if rf.exists():
        docs["repo-readme"] = json.loads(rf.read_text())["text"]
    else:
        missing.append(f"repos/{rf.name} (README)")
    if df.exists():
        dump = json.loads(df.read_text())
        for path, t in dump["files"].items():
            docs[f"repo:{path}@{dump['commit'][:7]}"] = t
    else:
        missing.append(f"repos/{df.name} (Repo-Dateien, Klon noetig: melden, nicht live klonen)")
    return docs, missing


def estimated_calls(docs: dict[str, str]) -> int:
    from repro_agent.setup_details import _chunks

    return sum(len(_chunks(t)) for t in docs.values())


def dry_run(soll: dict, units: list[str], fixtures: Path = FIXTURES) -> str:
    lines = ["Einheit | Claim | Soll-Punkte | Dokumente | Fixtures fehlen | geschaetzte LLM-Aufrufe je Wiederholung"]
    total = 0
    for u in units:
        docs, missing = fixture_sources(u, fixtures)
        n = estimated_calls(docs)
        total += n
        lines.append(f"{u} | {soll['units'][u]['claim']} | {len(soll['soll'][u])} | {len(docs)} "
                     f"({', '.join(k.split('@')[0] for k in list(docs)[:3])}{', ...' if len(docs) > 3 else ''}) | "
                     f"{'; '.join(missing) or 'keine'} | {n}")
    reps = 3
    lines.append(f"Summe je Wiederholung: {total} Aufrufe; {reps} Wiederholungen: {total * reps} (Rolle 4: Super, Cache aus)")
    lines.append(f"Soll-Punkte gesamt: {sum(len(soll['soll'][u]) for u in units)}")
    return "\n".join(lines)


# ---- Belegbarkeit ------------------------------------------------------------------------------------------------------------
def check_soll(soll: dict, units: list[str], fixtures: Path = FIXTURES) -> list[dict]:
    """Je Soll-Punkt: Steht der Wert (value-Muster) im Abstand von hoechstens WINDOW Zeichen zum Thema (topic-Muster) in einer
    gespeicherten Quelle? Je Dokument zaehlt das naechste Paar; ausgegeben werden die zwei naechsten Treffer ueber alle Dokumente
    (Wert und Stichwort wortlich, der Reviewer entscheidet). Nichts wird gestrichen."""
    rows = []
    for u in units:
        docs, _ = fixture_sources(u, fixtures)
        for point, spec in soll["patterns"][u].items():
            topic, value = re.compile(spec["topic"], re.I), re.compile(spec["value"], re.I)
            cands = []
            for name, text in docs.items():
                topics = [(m.start(), m.end()) for m in topic.finditer(text)]
                best = None
                for m in value.finditer(text):
                    for ts, te in topics:
                        d = max(0, ts - m.end(), m.start() - te)
                        if d <= WINDOW and (best is None or d < best[0]):
                            best = (d, m.start(), m.end(), ts, te)
                if best:
                    d, vs, ve, ts, te = best
                    lo, hi = min(vs, ts), max(ve, te)
                    cands.append((d, name.split("@")[0], text[max(0, lo - 30): hi + 30].replace("\n", " ")))
            cands.sort(key=lambda c: c[0])
            rows.append({"unit": u, "point": point, "soll": soll["soll"][u][point], "where": cands[0][1] if cands else None,
                         "snippet": cands[0][2] if cands else None, "hits": [{"doc": c[1], "distance": c[0], "snippet": c[2]} for c in cands[:2]]})
    return rows


def format_check(rows: list[dict]) -> str:
    out = ["Einheit | Punkt | Soll | Beleg"]
    for r in rows:
        out.append(f"{r['unit']} | {r['point']} | {r['soll']} | "
                   + (" || ".join(f"belegbar in {h['doc']} (Abstand {h['distance']}): ...{h['snippet']}..." for h in r["hits"]) if r["where"] else "nicht gefunden"))
    n_ok = sum(1 for r in rows if r["where"])
    missing = [f"{r['unit']}.{r['point']}" for r in rows if not r["where"]]
    out.append(f"belegbar: {n_ok} von {len(rows)}; nicht gefunden: {missing}")
    return "\n".join(out)


# ---- Auswertung ----------------------------------------------------------------------------------------------------------------
SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a\u2212\u2013", "0123456789-+--")
SCI_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[\u00b7\u00d7x*\u22c5]|\\times|\\cdot)\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?")
E_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![\w.])")
FALSE_RE = re.compile(r"\s*(false|no|none|off)\s*", re.I)
TRUE_RE = re.compile(r"\s*(true|yes|on)\s*", re.I)
NORM_TOPIC_RE = re.compile(r"normali[sz]|standardi[sz]", re.I)


def canonical_value(value: str, topic: str = "") -> str:
    """Gleichwertige Wertformen auf eine Form bringen: 5e-4 = 5 \u00b7 10 \u22124 = 5\u00b710\u207b\u2074 = 0.0005 (die Dezimalform wird angehaengt),
    Schalter normalize_features: false/False -> "not normalised", true -> "normalised"."""
    t = value.translate(SUPERSCRIPTS)
    forms = [f"{float(m.group(1)) * 10 ** int(m.group(2)):.10g}" for m in SCI_RE.finditer(t)]
    forms += [f"{float(m.group(1)) * 10 ** int(m.group(2)):.10g}" for m in E_RE.finditer(t)]
    if NORM_TOPIC_RE.search(topic):
        if FALSE_RE.fullmatch(t):
            forms.append("not normalised")
        elif TRUE_RE.fullmatch(t):
            forms.append("normalised")
    return t + (" " + " ".join(forms) if forms else "")


def score_point(details: list[dict], spec: dict) -> tuple[str, dict]:
    """(korrekt | fehlt | falsch | unklar, Hinweise). Dossier-Angaben, deren Thema (topic + value) trifft, werden nach Wert beurteilt.
    Werte werden vorher kanonisiert (canonical_value); `quote_value` (optional) erkennt den Wert auch im Zitat einer Angabe mit anderem
    Thema; `wrong_unless` (optional) nimmt Werte von der falsch-Regel aus, die erkennbar ein anderes Protokoll beschreiben."""
    topic = re.compile(spec["topic"], re.I)
    val = lambda d: canonical_value(str(d.get("value", "")), str(d.get("topic", "")))
    hits = [d for d in details if topic.search(f"{d.get('topic', '')} {val(d)}")]
    quote_good = [d for d in details if spec.get("quote_value") and re.search(spec["quote_value"], str(d.get("quote", "")), re.I)]
    if not hits and not quote_good:
        return "fehlt", {}
    value = re.compile(spec["value"], re.I)
    wrong = re.compile(spec["wrong"], re.I) if spec.get("wrong") else None
    unless = re.compile(spec["wrong_unless"], re.I) if spec.get("wrong_unless") else None
    good = quote_good + [d for d in hits if value.search(val(d)) and d not in quote_good]
    bad = [d for d in hits if wrong and wrong.search(val(d)) and not value.search(val(d)) and not (unless and unless.search(val(d)))]
    if good:
        return "korrekt", {"conflict": bool(bad), "evidence": good[0]}
    if bad:
        return "falsch", {"evidence": bad[0]}
    return "unklar", {"evidence": (hits or quote_good)[0]}


CONFLICT_RE = re.compile(r"repository says|-> use the value|highest-ranked source", re.I)


def parse_plan_line(line: str) -> dict:
    """Zeile des Plan-Textes "- Thema: Wert  (Quelle: \"Zitat\")  [Markierung]" -> topic, value, src, conflict."""
    t = line.strip()
    out = {"topic": "", "value": t, "src": "", "conflict": bool(CONFLICT_RE.search(t)), "text": t}
    m = re.match(r"-\s*(.*?):\s+(.*)$", t)
    if not m:
        return out
    topic, rest = m.groups()
    idx = rest.find("  (")
    if idx < 0:
        return {**out, "topic": topic, "value": rest}
    tail = rest[idx + 3:]
    cut = tail.find(': "')
    return {**out, "topic": topic, "value": rest[:idx], "src": tail[:cut] if cut >= 0 else tail}


def rule_matches(spec: dict, row: dict) -> bool:
    """Trifft die Regel die geparste Zeile? kind line: ganze Zeile; field: Thema und Wert getrennt; source: Quelle der Zeile."""
    kind = spec.get("kind", "line")
    if row["conflict"] and kind != "line" and not spec.get("allow_conflict"):
        return False   # Konfliktzeilen nennen beide Werte nur, sie fuehren keinen als gueltig
    if spec.get("exclude_line") and re.search(spec["exclude_line"], row["text"], re.I):
        return False
    if spec.get("unless_line") and re.search(spec["unless_line"], row["text"], re.I):
        return False
    if kind == "line":
        return bool(re.search(spec["line"], row["text"], re.I))
    if kind == "source":
        return bool(re.search(spec["source"], row["src"], re.I))
    if kind == "field":
        if not re.search(spec["topic"], row["topic"], re.I) or (spec.get("not_topic") and re.search(spec["not_topic"], row["topic"], re.I)):
            return False
        if not re.search(spec["value"], row["value"], re.I) or (spec.get("not_value") and re.search(spec["not_value"], row["value"], re.I)):
            return False
        return True
    raise ValueError(f"unbekannte Regel-Art {kind!r}")


def misleading_hits(plan_text: str, specs: list[dict], unless: str) -> list[dict]:
    """Zeilen des Plan-Textes, die einen der bekannten irrefuehrenden Faelle als gueltig fuehren (status "hit"), und mehrdeutige Zeilen
    (status "ambiguous", zaehlen nicht als Treffer). Markierte Zeilen (optional advice, repository default ...) werden uebersprungen.
    Je Eintrag: id, rule (Regel im Wortlaut), line."""
    out = []
    skip = re.compile(unless, re.I)
    for line in plan_text.splitlines():
        if not line.strip() or skip.search(line):
            continue
        row = parse_plan_line(line)
        hits = [sp for sp in specs if sp.get("status") != "ambiguous" and rule_matches(sp, row)]
        for sp in hits:
            out.append({"id": sp["id"], "rule": sp["rule"], "line": row["text"][:300], "status": "hit"})
        if not hits:
            for sp in specs:
                if sp.get("status") == "ambiguous" and rule_matches(sp, row):
                    out.append({"id": sp["id"], "rule": sp["rule"], "line": row["text"][:300], "status": "ambiguous"})
                    break
    seen, dedup = set(), []
    for h in out:
        key = (h["id"], h["rule"], h["line"], h["status"])
        if key not in seen:
            seen.add(key)
            dedup.append(h)
    return dedup


def load_label(label: str, root: Path = OUT_ROOT) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for f in sorted((root / label).glob("*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(d["unit"], []).append(d)
    return runs


def evaluate(soll: dict, runs: dict[str, list[dict]]) -> dict:
    """Auswertung eines Labels. Rueckgabe: {units, totals, goals, doubts, misleading}."""
    units, doubts, mislead, ambiguous = {}, [], [], []
    tot = Counter()
    for u, reps in runs.items():
        if u not in soll["patterns"]:
            continue
        rows = {}
        for point, spec in soll["patterns"][u].items():
            statuses = []
            for r in reps:
                st, info = score_point(r["details"], spec)
                statuses.append(st)
                tot[st] += 1
                if st == "unklar":
                    ev = info["evidence"]
                    doubts.append({"unit": u, "rep": r["rep"], "point": point, "topic": ev.get("topic"), "value": ev.get("value")})
            rows[point] = statuses
        units[u] = {"reps": len(reps), "points": rows}
        for r in reps:
            for h in misleading_hits(r.get("plan_text", ""), soll["misleading"].get(u, []), soll["misleading_unless"]):
                (mislead if h["status"] == "hit" else ambiguous).append({"unit": u, "rep": r["rep"], **h})
    n = sum(tot.values())
    share = tot["korrekt"] / n if n else 0.0
    norm_units = [u for u in units if any(p in NORM_POINTS for p in units[u]["points"])]
    norm_ok = [u for u in norm_units
               if sum(1 for p, s in units[u]["points"].items() if p in NORM_POINTS for st in [s.count("korrekt")] if st >= GOAL1_MIN_REPS)]
    known = {u for u in soll["misleading"]}
    goals = {
        "ziel1": {"units_with_point": len(norm_units), "units_ok": len(norm_ok), "ok_units": norm_ok,
                  "erreicht": len(norm_ok) >= GOAL1_MIN_UNITS},
        "ziel2": {"korrekt": tot["korrekt"], "gesamt": n, "anteil": share, "erreicht_ohne_basis": share >= GOAL2_SHARE},
        "ziel3": {"treffer": len(mislead), "units": sorted(known & set(runs)), "erreicht": not mislead and known <= set(runs)},
    }
    return {"units": units, "totals": dict(tot), "goals": goals, "doubts": doubts, "misleading": mislead, "ambiguous": ambiguous}


def format_score(label: str, ev: dict, base: dict | None = None, base_label: str | None = None) -> str:
    out = [f"Auswertung {label}"]
    out.append("Einheit | Punkt | " + " | ".join(f"Wdh {i}" for i in range(max((v["reps"] for v in ev["units"].values()), default=0))))
    for u, v in ev["units"].items():
        for p, st in v["points"].items():
            out.append(f"{u} | {p} | " + " | ".join(st))
        c = Counter(s for st in v["points"].values() for s in st)
        out.append(f"{u} Summe: korrekt {c['korrekt']}, fehlt {c['fehlt']}, falsch {c['falsch']}, unklar {c['unklar']}")
    t, g = ev["totals"], ev["goals"]
    out.append(f"Gesamt: {t}")
    g1, g2, g3 = g["ziel1"], g["ziel2"], g["ziel3"]
    out.append(f"Ziel 1 (Normalisierung/Methode in >= {GOAL1_MIN_REPS} von 3 Wdh bei >= {GOAL1_MIN_UNITS} von {g1['units_with_point']} Einheiten): "
               f"{g1['units_ok']} Einheiten ({', '.join(g1['ok_units'])}) -> {'ja' if g1['erreicht'] else 'nein'}")
    base_share = None if base is None else base["goals"]["ziel2"]["anteil"]
    ziel2 = g2["erreicht_ohne_basis"] and (base_share is None or g2["anteil"] > base_share)
    out.append(f"Ziel 2 (korrekt >= {GOAL2_SHARE:.0%} und hoeher als Basis): {g2['korrekt']} von {g2['gesamt']} = {g2['anteil']:.1%}"
               + (f", Basis {base_label} {base_share:.1%}" if base_share is not None else ", Basis nicht angegeben (--vs)")
               + f" -> {'ja' if ziel2 else 'nein'}")
    out.append(f"Ziel 3 (keine irrefuehrende Angabe als gueltig im Plan-Text, {len(g3['units'])} von 4 bekannten Einheiten gemessen): "
               f"{g3['treffer']} Treffer -> {'ja' if g3['erreicht'] else 'nein'}")
    for m in ev["misleading"]:
        out.append(f"  irrefuehrend: {m['unit']} Wdh {m['rep']} [{m['id']}] Regel: {m['rule']} | Zeile: {m['line']}")
    out.append(f"Mehrdeutige Zeilen des Plan-Textes (zaehlen nicht als Treffer, Handpruefung): {len(ev['ambiguous'])}")
    for m in ev["ambiguous"]:
        out.append(f"  mehrdeutig: {m['unit']} Wdh {m['rep']} [{m['id']}] Regel: {m['rule']} | Zeile: {m['line']}")
    out.append(f"Zweifelsfaelle (Thema getroffen, Wert unklar; Handpruefung, zaehlen nicht als korrekt): {len(ev['doubts'])}")
    for d in ev["doubts"]:
        out.append(f"  {d['unit']} Wdh {d['rep']} {d['point']}: {d['topic']} = {d['value']}")
    return "\n".join(out)


def raw_trace(calls: list[dict], docs) -> list[dict]:
    """Je LLM-Aufruf: Rohantwort, finish_reason, Dokument des Chunks, geparste Eintraege und Verwurfsgruende (deterministisch
    nachgerechnet mit verify_raw; Aufrufe ohne Chunk-Dokument, z. B. Dateiauswahl, haben doc None)."""
    from repro_agent.setup_details import _parse_list, verify_raw

    by_doc = {d.name: d for d in docs}
    out = []
    for c in calls:
        head = re.search(r"EXCERPT[^\n]*:\n", c["user"])   # "EXCERPT:" (v1) oder "EXCERPT from <Quelle>:" (setup_v2)
        excerpt = c["user"][head.end():].strip() if head else ""
        doc = next((d for d in docs if excerpt and excerpt[:200] in d.text), None)
        items = _parse_list(c["response"]) if doc else []
        dropped = []
        if doc and items:
            _, dropped = verify_raw([{"doc": doc.name, "item": i} for i in items], [by_doc[doc.name]])
        out.append({"doc": doc.name if doc else None, "prompt_head": c["user"][:300], "response": c["response"],
                    "finish_reason": c["finish_reason"], "completion_tokens": c["completion_tokens"], "status": c["status"],
                    "n_items": len(items), "dropped": [{"reason": d["reason"], "item": d["item"]} for d in dropped]})
    return out


# ---- Messlauf (LLM) ------------------------------------------------------------------------------------------------------------
def collect(label: str, units: list[str], reps: int, setup_v2: bool, max_gross: float, root: Path = OUT_ROOT) -> None:
    import dataclasses

    from repro_agent.claims_source import PaperText
    from repro_agent.loop import Orchestrator
    from repro_agent.setup_details import extract_setup_details

    soll = load_soll()
    kw = {"setup_v2": True} if setup_v2 else {}
    orch = Orchestrator(use_tavily=True, fresh=True, setup_details=True, tavily_mode="fixture", graph_repair=True, llm_cache=False, **kw)
    orch.sandbox, orch._tavily = Forbidden(), Forbidden()
    out = root / label
    out.mkdir(parents=True, exist_ok=True)
    for u in units:
        paper = paper_for(u)
        claim = claim_for(u, soll["units"][u]["claim"])
        arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
        for rep in range(reps):
            f = out / f"{u}_{rep}.json"
            if f.exists():
                continue
            if orch.spent_gross_usd >= max_gross:
                print(f"ABBRUCH: {orch.spent_gross_usd:.2f} $ brutto >= Grenze {max_gross:.2f} $", flush=True)
                return
            calls: list[dict] = []
            real_complete = orch.llm.complete

            def recording(role, messages, **kw):   # Rohantwort je Chunk festhalten (Verwurfsgruende rechnet raw_trace nachtraeglich aus)
                resp = real_complete(role, messages, **kw)
                calls.append({"user": messages[-1]["content"], "response": resp.content, "finish_reason": resp.finish_reason,
                              "completion_tokens": resp.completion_tokens, "status": resp.status})
                return resp

            orch.llm.complete = recording
            orch.paper_text = PaperText(**json.loads((FIXTURES / f"{arxiv_id}.json").read_text()))
            orch._paper_title, orch._sources, orch._repo_files, orch.details = paper.get("title"), [], {}, []
            before, before_gross = orch.llm.total_cost_usd, orch.spent_gross_usd
            docs, repo = orch._gather_docs(claim["text"], paper["dataset"])
            info = None
            if setup_v2:
                from repro_agent.setup_details import extract_setup_details_v2

                details, dropped, missing, info = extract_setup_details_v2(orch.llm, docs, claim["text"], paper["dataset"])
                info = {**info, "missing": missing}
            else:
                details, dropped = extract_setup_details(orch.llm, docs, claim["text"], paper["dataset"])
            orch.details = details
            orch.llm.complete = real_complete
            raw_dir = out / "raw"
            raw_dir.mkdir(exist_ok=True)
            (raw_dir / f"{u}_{rep}.json").write_text(json.dumps(raw_trace(calls, docs), ensure_ascii=False, indent=1))
            plan_text = (orch._details_prompt() + "\n\n" + orch._conflicts_prompt()).strip()
            f.write_text(json.dumps({
                "unit": u, "rep": rep, "label": label, "claim_id": claim["id"], "setup_v2": setup_v2, "repo": repo,
                "n_docs": len(docs), "dropped": dropped, "v2_info": info, "details": [dataclasses.asdict(d) for d in details],
                "plan_text": plan_text, "cost_net_usd": orch.llm.total_cost_usd - before,
                "cost_gross_usd": orch.spent_gross_usd - before_gross,
            }, ensure_ascii=False, indent=1))
            print(f"{u} Wdh {rep}: {len(details)} Angaben, {dropped} verworfen, {orch.spent_gross_usd:.3f} $ brutto bisher", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-soll", action="store_true")
    ap.add_argument("--score", metavar="LABEL")
    ap.add_argument("--vs", metavar="LABEL", help="Basis-Label fuer Ziel 2")
    ap.add_argument("--runs-root", type=Path, default=OUT_ROOT, help="Verzeichnis mit den Labels (Standard runs/v5_step_a)")
    ap.add_argument("--label")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--units", help="kommagetrennt; Standard: alle acht")
    ap.add_argument("--setup-v2", action="store_true")
    ap.add_argument("--max-gross-usd", type=float, default=DEFAULT_MAX_GROSS_USD)
    a = ap.parse_args(argv)
    soll = load_soll()
    units = a.units.split(",") if a.units else list(soll["units"])
    if a.dry_run:
        print(dry_run(soll, units))
    elif a.check_soll:
        print(format_check(check_soll(soll, units)))
    elif a.score:
        ev = evaluate(soll, load_label(a.score, a.runs_root))
        print(format_score(a.score, ev, evaluate(soll, load_label(a.vs, a.runs_root)) if a.vs else None, a.vs))
    elif a.label:
        collect(a.label, units, a.reps, a.setup_v2, a.max_gross_usd)
    else:
        ap.error("--dry-run, --check-soll, --score oder --label angeben")


if __name__ == "__main__":
    sys.exit(main())
