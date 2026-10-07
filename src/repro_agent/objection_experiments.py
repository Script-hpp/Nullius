"""Tavily-Rolle 3, Stufe 2: ein belegter Einwand (Fund aus `community.py`) wird zu einem Experiment.

Das LLM ordnet jeden Fund einmal einem von vier Typen zu; danach entscheidet fester Code (Regeln unten, vorab festgelegt in
docs/tavily_experiments.md und benchmark/experiments/tavily_experiments_test.md). Grundsaetze:
- Nur Bericht: Urteil, Achse 1 und 2 und `dossier.attempts` bleiben unberuehrt; Ergebnisse stehen in `dossier.objection_checks`.
- Harness-Vertrag unveraendert: Agenten-Code liefert Vorhersagen, unser Harness misst, im eigenen Branch.
- Das LLM kann Werte nicht erfinden: Parameterwert und berichtete Zahl muessen im woertlichen Zitat stehen, der Parameter in der
  Liste des Tasks (`Task.tunable`), die Metrik muss die des Claims sein. Sonst wird der Typ auf `not_measurable` herabgestuft.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .claims_source import _norm

TYPES = ("seed_variance", "parameter_change", "reported_number", "not_measurable")
MAX_FINDINGS = 12            # Funde, die klassifiziert werden (Kosten: ein extract-Aufruf je Fund)
MAX_EXPERIMENTS = 3          # Experimente mit Sandbox pro Lauf; Schnellmodus (API) 2
QUICK_MAX_EXPERIMENTS = 2
SEED_VAR_K = 5               # zusaetzliche Seeds bei seed_variance; Schnellmodus 3
QUICK_SEED_VAR_K = 3
PARAM_SEEDS = 3              # Seeds bei parameter_change (Seed 0 und 2 weitere), gepaart mit den Hauptseeds
SEED_WORDS = re.compile(r"seed|random|initiali[sz]|varian|vary|varies|varied|stochastic|run[- ]to[- ]run|unstable|fluctuat", re.I)

RULE_SEED = ("seed_variance: K extra seeds with the final main code. bestaetigt if std(main+extra) >= tolerance or "
             "|mean(extra) - main mean| > tolerance; widerlegt if neither and at least ceil(K/2) extra seeds are valid; offen otherwise "
             "(fewer than 2 valid extra seeds count as offen).")
RULE_PARAM = ("parameter_change: one new code changes exactly the named setting; paired seeds with the main run. delta = new mean - main mean "
              "(same seeds), SE = sqrt((s_new^2 + s_main^2) / 2) / sqrt(n). bestaetigt if |delta| > max(tolerance, 2*SE); widerlegt if "
              "|delta| <= tolerance (at least 2 valid pairs); offen otherwise.")
RULE_REPORTED = ("reported_number: no new run. Needs the same setup (named in the quote). bestaetigt if |ours - reported| <= tolerance and "
                 "|reported - target| > tolerance; widerlegt if |ours - target| <= tolerance and |ours - reported| > tolerance; offen otherwise.")
RULE_NONE = "not_measurable: stays text."


@dataclass
class ObjectionCheck:
    id: str
    claim_id: str
    topic: str
    statement: str
    quote: str
    url: str
    kind: str
    type: str
    type_reason: str
    param: str | None = None
    to_value: str | None = None
    reported_value: float | None = None
    reported_metric: str | None = None
    same_setup: str = "unknown"
    status: str = "nicht pruefbar"          # bestaetigt | widerlegt | offen | nicht pruefbar
    rule: str = RULE_NONE
    measured: dict | None = None
    attempts: list = field(default_factory=list)   # {branch_id, seed, status, measured, code_hash, parent_snapshot}
    code_hash: str | None = None
    main_code_hash: str | None = None
    cost_usd: float = 0.0
    seconds: float = 0.0
    note: str = ""


# ---- Klassifikation -------------------------------------------------------------------------------------------------

CLASSIFY_SYSTEM = "You classify objections quoted from web pages. Answer only from the text you are given."


def classify_prompt(paper: str, dataset: str, claim, tunable: dict[str, str], finding: dict) -> str:
    settings = "\n".join(f"- {k}: {v}" for k, v in tunable.items()) or "(none: this task allows no setting to be changed)"
    return (
        f"Paper: {paper}. Dataset: {dataset}. Claim: {claim.text} (metric {claim.metric}, target {claim.target}).\n"
        "A public source contains the objection below. Choose exactly ONE type:\n"
        "- seed_variance: the quote itself says the result varies with random seed / initialisation / run.\n"
        "- parameter_change: the quote names a concrete setting AND a concrete value for it (for example an input window of 336 "
        "instead of 96) that could change the result.\n"
        "- reported_number: the quote contains a number that someone measured with the same metric as the claim.\n"
        "- not_measurable: anything else (fairness, novelty, wording, questions, bugs of the original code, other metrics).\n"
        "If in doubt choose not_measurable.\n"
        f"Settings this task allows to change (parameter_change only):\n{settings}\n"
        "For parameter_change give param = the key from the list; if the quote names a setting that is not in the list, give its name "
        "in a few words. to_value = the value exactly as written in the quote.\n"
        "For reported_number give reported_value = the number as written in the quote and reported_metric. same_setup is 'yes' only if "
        "the quote names the same dataset and the same horizon or split as the claim, 'unknown' if it does not say, 'no' if it differs.\n"
        'Answer with ONE JSON object {"type": ..., "reason": one sentence, "param": string or null, "to_value": string or null, '
        '"reported_value": number or null, "reported_metric": string or null, "same_setup": "yes"|"no"|"unknown"}. '
        "Keep your reasoning short and end with the JSON object.\n\n"
        f"OBJECTION: {finding['statement']}\nQUOTE: {finding['quote']}"
    )


def _parse_object(content: str) -> dict:
    """Letztes gueltiges JSON-Objekt mit Feld `type` (Modelle schreiben Denkprozess davor)."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", content or ""))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "type" in value:
            return value
    return {}


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace(",", "."))]


def validate(raw: dict, finding: dict, claim_metric: str, tunable: dict[str, str]) -> dict:
    """Feste Pruefung der LLM-Antwort. Rueckgabe: {type, reason, param, to_value, reported_value, reported_metric, same_setup, note}.
    Eine Behauptung, die das Zitat oder die Task-Liste nicht deckt, wird `not_measurable` mit Begruendung (sichtbar im Zeugnis)."""
    typ = str(raw.get("type", "")).strip()
    out = {"type": "not_measurable", "reason": str(raw.get("reason") or "").strip(), "param": None, "to_value": None,
           "reported_value": None, "reported_metric": None, "same_setup": "unknown", "note": ""}
    if typ not in TYPES:
        out["note"] = "classification missing or invalid" if not raw else f"unknown type {typ!r}"
        return out
    quote_n = _norm(finding["quote"])
    if typ == "seed_variance":
        if SEED_WORDS.search(finding["statement"] + " " + finding["quote"]):
            out["type"] = typ
        else:
            out["note"] = "no statement about seeds or run-to-run variation in the quote"
    elif typ == "parameter_change":
        param, to_value = str(raw.get("param") or "").strip(), str(raw.get("to_value") or "").strip()
        if param not in tunable:
            hint = " (the task fixes data and splits)" if re.search(r"split|data|label|test set", param, re.I) else ""
            out["note"] = f"setting {param or 'unnamed'!r} is not changeable in this task{hint}"
            out["param"] = param or None
        elif not to_value or _norm(to_value) not in quote_n:
            out["note"] = "the value is not written in the quote"
        else:
            out.update(type=typ, param=param, to_value=to_value)
    elif typ == "reported_number":
        metric = str(raw.get("reported_metric") or "").strip()
        try:
            value = float(raw.get("reported_value"))
        except (TypeError, ValueError):
            value = None
        if value is None or not any(math.isclose(value, x, abs_tol=1e-9) for x in _numbers(finding["quote"])):
            out["note"] = "the reported number is not written in the quote"
        elif metric.lower() != claim_metric.lower():
            out["note"] = f"the reported metric ({metric or 'unnamed'}) is not the claim metric ({claim_metric})"
        else:
            same = str(raw.get("same_setup") or "unknown").strip().lower()
            out.update(type=typ, reported_value=value, reported_metric=metric, same_setup=same if same in ("yes", "no") else "unknown")
    return out


def classify_finding(llm, paper: str, dataset: str, claim, tunable: dict[str, str], finding: dict) -> dict:
    resp = llm.complete("extract", [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": classify_prompt(paper, dataset, claim, tunable, finding)},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return validate(_parse_object(resp.content), finding, claim.metric, tunable)


# ---- Regeln (reine Funktionen, vorab festgelegt) ----------------------------------------------------------------------

def rule_seed_variance(main_vals: list[float], extra_vals: list[float], tol: float, k: int) -> tuple[str, dict, str]:
    """Rueckgabe: (Status, Messwerte, Hinweis)."""
    if not main_vals or len(extra_vals) < 2:
        return "offen", {"n": len(extra_vals)}, "fewer than 2 valid extra seeds" if main_vals else "no valid main measurement"
    pool = main_vals + extra_vals
    std_pool, mean_extra, mean_main = statistics.stdev(pool), statistics.mean(extra_vals), statistics.mean(main_vals)
    shift = abs(mean_extra - mean_main)
    meas = {"n_main": len(main_vals), "n_extra": len(extra_vals), "mean": mean_extra, "std": statistics.stdev(extra_vals),
            "std_pool": std_pool, "main_mean": mean_main, "delta": mean_extra - mean_main}
    if std_pool >= tol or shift > tol:
        return "bestaetigt", meas, ""
    if len(extra_vals) >= math.ceil(k / 2):
        return "widerlegt", meas, ""
    return "offen", meas, f"only {len(extra_vals)} of {k} extra seeds valid"


def rule_parameter_change(new: dict[int, float], main: dict[int, float], tol: float) -> tuple[str, dict, str]:
    seeds = sorted(set(new) & set(main))
    if len(seeds) < 2:
        return "offen", {"n": len(seeds)}, "fewer than 2 valid paired seeds"
    a, b = [new[s] for s in seeds], [main[s] for s in seeds]
    delta = statistics.mean(a) - statistics.mean(b)
    se = math.sqrt((statistics.stdev(a) ** 2 + statistics.stdev(b) ** 2) / 2) / math.sqrt(len(seeds))
    meas = {"n": len(seeds), "seeds": seeds, "mean": statistics.mean(a), "std": statistics.stdev(a), "main_mean": statistics.mean(b),
            "main_std": statistics.stdev(b), "delta": delta, "se": se}
    if abs(delta) > max(tol, 2 * se):
        return "bestaetigt", meas, ""
    if abs(delta) <= tol:
        return "widerlegt", meas, ""
    return "offen", meas, "effect larger than the tolerance but not separable from seed noise (|delta| <= 2*SE)"


def rule_reported(r: float, m: float | None, t: float, tol: float, same_setup: str) -> tuple[str, dict, str]:
    if m is None or math.isnan(m):
        return "offen", {}, "no valid main measurement"
    near_r, near_t = abs(m - r) <= tol, abs(m - t) <= tol
    supports = "both" if near_r and near_t else "reported" if near_r else "paper" if near_t else "neither"
    meas = {"mean": m, "reported": r, "target": t, "supports": supports}
    if same_setup != "yes":
        return "offen", meas, "the quote does not state the same setup as our task" if same_setup == "unknown" else "different setup"
    if near_r and abs(r - t) > tol:
        return "bestaetigt", meas, ""
    if near_t and not near_r:
        return "widerlegt", meas, ""
    return "offen", meas, {"both": "reported value and target lie within the tolerance of each other",
                           "neither": "our measurement supports neither value"}.get(supports, "")


# ---- Lauf ------------------------------------------------------------------------------------------------------------

def _hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:12]


def _plan(checks: list[ObjectionCheck], max_exp: int) -> tuple[list[ObjectionCheck], dict[str, str]]:
    """Welche Experimente laufen: parameter_change vor seed_variance, doppelte nur einmal, hoechstens `max_exp`."""
    order = [c for t in ("parameter_change", "seed_variance") for c in checks if c.type == t]
    run, first_of, dup = [], {}, {}
    for c in order:
        key = (c.type, c.param, (c.to_value or "").lower()) if c.type == "parameter_change" else (c.type,)
        if key in first_of:
            dup[c.id] = first_of[key]
        elif len(run) < max_exp:
            first_of[key] = c.id
            run.append(c)
        else:
            c.status, c.note = "offen", f"limit of {max_exp} experiments per run reached"
    return run, dup


def run_round(orch, dossier, paper: dict, quick: bool = False) -> None:
    """Nach dem Community-Check. Liest dossier.community-Funde, schreibt nur dossier.objection_checks und Ereignisse."""
    findings = list((getattr(dossier, "community") or {}).get("findings", []))[:MAX_FINDINGS]
    if not findings:
        return
    t_start, cost0 = time.time(), orch.spent_usd
    verdicts = {v.claim_id: v for v in dossier.verdicts}
    claim = next((c for c in dossier.claims if c.origin == "paper" and c.id in verdicts and c.id in getattr(orch, "_claim_ctx", {})
                  and not math.isnan(verdicts[c.id].measured_mean)), None)
    if claim is None:
        orch._emit("objection_round_end", skipped="no claim with a valid measurement", seconds=0, n_experiments=0)
        return
    ctx, verdict = orch._claim_ctx[claim.id], verdicts[claim.id]
    task, tunable = ctx["task"], dict(getattr(ctx["task"], "tunable", ()) or ())
    title, dataset = paper.get("title", ""), paper.get("dataset", "")
    checks: list[ObjectionCheck] = []
    for i, f in enumerate(findings, 1):
        info = classify_finding(orch.llm, title, dataset, claim, tunable, f)
        c = ObjectionCheck(
            id=f"obj{i}", claim_id=claim.id, topic=f.get("topic", ""), statement=f["statement"], quote=f["quote"], url=f["url"],
            kind=f.get("kind", ""), type=info["type"], type_reason=info["reason"], param=info["param"], to_value=info["to_value"],
            reported_value=info["reported_value"], reported_metric=info["reported_metric"], same_setup=info["same_setup"], note=info["note"])
        if c.type == "not_measurable":
            c.status = "nicht pruefbar"
        elif c.type == "reported_number":
            c.status, c.measured, c.note = rule_reported(
                c.reported_value, verdict.measured_mean, claim.target, claim.tolerance, c.same_setup)
            c.rule = RULE_REPORTED
        checks.append(c)
        orch._emit("objection_classified", id=c.id, claim_id=claim.id, otype=c.type, topic=c.topic, quote=c.quote, url=c.url,
                   param=c.param, to_value=c.to_value, reported_value=c.reported_value, note=c.note or None)
    run, dup = _plan(checks, QUICK_MAX_EXPERIMENTS if quick else MAX_EXPERIMENTS)
    by_id = {c.id: c for c in checks}
    k_extra = QUICK_SEED_VAR_K if quick else SEED_VAR_K
    for c in run:
        c0, t0 = orch.spent_usd, time.time()
        orch._emit("objection_experiment", id=c.id, otype=c.type, stage="start", param=c.param, to_value=c.to_value)
        try:
            if orch.spent_gross_usd >= orch.settings.max_cost_per_run_usd:
                c.status, c.note = "offen", "run budget reached"
            elif c.type == "seed_variance":
                _run_seed_variance(orch, dossier, claim, verdict, ctx, c, k_extra)
            else:
                _run_parameter_change(orch, dossier, claim, ctx, c)
        except Exception as exc:
            from .llm import BillingBlocked
            if isinstance(exc, BillingBlocked):
                raise
            c.status, c.note = "offen", f"experiment failed: {type(exc).__name__}: {str(exc)[:200]}"
            orch._warn(f"Einwand-Experiment {c.id} fehlgeschlagen: {type(exc).__name__}: {exc}")
        c.cost_usd, c.seconds = round(orch.spent_usd - c0, 4), round(time.time() - t0, 1)
        orch._emit("objection_experiment", id=c.id, otype=c.type, stage="end", status=c.status, seconds=c.seconds)
    for did, first in dup.items():   # gleiche Aenderung wie ein anderer Fund: Ergebnis uebernehmen, nichts neu messen
        d, f = by_id[did], by_id[first]
        d.status, d.measured, d.rule, d.attempts, d.note = f.status, f.measured, f.rule, [], f"same experiment as {f.id}"
    for c in checks:
        dossier.objection_checks.append(asdict(c))
        orch._emit("objection_checked", id=c.id, claim_id=c.claim_id, otype=c.type, status=c.status, rule=c.rule, measured=c.measured,
                   quote=c.quote, url=c.url, param=c.param, to_value=c.to_value, note=c.note or None)
    orch._emit("objection_round_end", seconds=round(time.time() - t_start, 1), n_findings=len(findings), n_experiments=len(run),
               quick=quick, cost_net_usd=round(orch.spent_usd - cost0, 4))


def _measure(orch, ctx, claim, code: str, seed: int, c: ObjectionCheck) -> tuple[float | None, dict]:
    """Ein Seed in einem frischen Branch (kein Reparaturversuch). Rueckgabe: (Messwert oder None, Versuchsdaten)."""
    t0 = time.time()
    branch, res = orch._attempt(ctx["snapshot"], code, seed, ctx["task"], claim.metric)
    value = res.get("measured") if res["status"] == "ok" else None
    rec = {"branch_id": branch, "seed": seed, "status": res["status"], "measured": value, "code_hash": _hash(code),
           "parent_snapshot": ctx["snapshot"], "detail": None if value is not None else str(res.get("detail", ""))[-300:]}
    orch._emit("objection_experiment", id=c.id, otype=c.type, stage="seed", seed=seed, branch_id=branch, parent_snapshot=ctx["snapshot"],
               code_hash=rec["code_hash"], status=res["status"], measured=value, seconds=round(time.time() - t0, 1))
    return value, rec


def _parallel(orch, ctx, claim, code, seeds, c) -> dict[int, float]:
    from .loop import MAX_PARALLEL
    if not seeds:
        return {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(seeds))) as pool:
        results = list(pool.map(lambda s: _measure(orch, ctx, claim, code, s, c), seeds))
    out = {}
    for s, (v, rec) in zip(seeds, results):
        c.attempts.append(rec)
        if v is not None:
            out[s] = v
    return out


def main_values(dossier, claim_id: str, code_hash: str) -> dict[int, float]:
    """Messwerte des Hauptlaufs je Seed fuer genau den Code, der das Urteil trug."""
    return {a.seed: a.measured for a in dossier.attempts if a.claim_id == claim_id and a.code_hash == code_hash and a.measured is not None}


def _run_seed_variance(orch, dossier, claim, verdict, ctx, c: ObjectionCheck, k: int) -> None:
    code = ctx["code"]
    c.rule, c.code_hash = RULE_SEED, _hash(code)
    c.main_code_hash = c.code_hash
    main = list(main_values(dossier, claim.id, c.code_hash).values())
    start = ctx["n_seeds"]
    extra = _parallel(orch, ctx, claim, code, list(range(start, start + k)), c)
    c.status, c.measured, c.note = rule_seed_variance(main, list(extra.values()), claim.tolerance, k)
    c.measured = dict(c.measured, seeds=sorted(extra), k=k)


def _run_parameter_change(orch, dossier, claim, ctx, c: ObjectionCheck) -> None:
    from .loop import CODE_SYSTEM
    c.rule, task, code0 = RULE_PARAM, ctx["task"], ctx["code"]
    c.main_code_hash = _hash(code0)
    main = main_values(dossier, claim.id, c.main_code_hash)
    user = (
        f"{task.data_description}\n\n{orch._details_prompt()}\n\nReproduction plan:\n{ctx['plan']}\n\n"
        f"This script works and was measured:\n```python\n{code0}\n```\n"
        f"A public source says: \"{c.quote}\"\nChange exactly this one setting: set {c.param} to {c.to_value}. Change nothing else "
        "(same model, same data handling, same training protocol, same seed handling, same SMOKE handling). Return the full script. "
        f"The script runs WITHOUT network access and must not download anything, spawn processes or use eval/exec. {task.code_notes}"
        f"{orch._product_notes(task)}"
    )
    code = orch._code_from_llm(CODE_SYSTEM, user)
    if _norm(c.to_value) not in _norm(code):
        c.status, c.note, c.code_hash = "offen", f"the new code does not contain the value {c.to_value}", _hash(code)
        return
    paired = sorted(main)[:PARAM_SEEDS]
    first = paired[0] if paired else 0
    v0 = None
    for n in range(1, orch.settings.max_attempts_per_claim + 1):
        v0, rec = _measure(orch, ctx, claim, code, first, c)
        c.attempts.append(rec)
        if v0 is not None or n == orch.settings.max_attempts_per_claim:
            break
        try:
            code = orch.repair_code(ctx["plan"], task, code, rec["detail"] or rec["status"])
        except ValueError:
            break
    c.code_hash = _hash(code)
    new = {} if v0 is None else {first: v0}
    new.update(_parallel(orch, ctx, claim, code, [s for s in paired if s != first] if v0 is not None else [], c))
    c.status, c.measured, c.note = rule_parameter_change(new, main, claim.tolerance)
    if c.measured and "mean" in c.measured:
        c.measured["within_target_tolerance"] = abs(c.measured["mean"] - claim.target) <= claim.tolerance
