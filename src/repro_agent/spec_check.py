"""Steckbrief im Code durchsetzen (S2) und einzeln pruefen (S3), siehe docs/recall_steckbrief_plan.md.

S2 ist deterministisch (ast, kein LLM): Der Code traegt einen literalen CONFIG-Block. Weicht ein erzwungener Wert ab oder fehlt
er, haengt der Pruefer direkt hinter den Block eine Zeile `CONFIG["lr"] = 0.005  # spec override` an. Liest der Code einen
Schluessel nie, oder steht im Optimizer-Aufruf ein anderes Literal, ist das ein Befund fuer eine gezielte Reparatur.

S3 stellt je nicht erzwungenem Eintrag eine kleine Ja/Nein-Frage an die Code-Rolle. Das Zitat der Antwort muss woertlich im
Code stehen, sonst gilt "nein". Keine Test- oder Zielwerte in einem Prompt.
"""

from __future__ import annotations

import ast
import json
import re

from .spec import Spec, SpecItem

# Schluesselwort-Argumente, deren Literal mit dem Steckbrief kollidieren kann (Optimizer, DataLoader, nn.Dropout)
KEYWORDS = {"lr": ("lr", "learning_rate"), "weight_decay": ("weight_decay",), "batch_size": ("batch_size",), "dropout": ("p", "dropout")}
CHECK_SYSTEM = "You review a Python training script against one item of a reproduction spec. Be strict. Keep reasoning brief."
MAX_CODE_CHARS = 40_000


def _config_node(tree: ast.Module) -> ast.Assign | None:
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "CONFIG" and isinstance(node.value, ast.Dict)):
            return node
    return None


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return float(a) == float(b)


def _used_keys(tree: ast.Module, cfg: ast.Assign) -> set[str]:
    """Schluessel, die ausserhalb des Blocks gelesen werden: CONFIG["k"], CONFIG.get("k"), Alias = CONFIG, oder **CONFIG mit .k."""
    inside = {id(n) for n in ast.walk(cfg)}
    aliases = {"CONFIG"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            aliases |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    used, star, attrs, strings = set(), False, set(), set()
    for node in ast.walk(tree):
        if id(node) in inside:
            continue
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            k = _literal(node.slice)
            if isinstance(k, str):
                used.add(k)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "setdefault", "pop")
              and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases and node.args):
            k = _literal(node.args[0])
            if isinstance(k, str):
                used.add(k)
        elif isinstance(node, ast.keyword) and node.arg is None and isinstance(node.value, ast.Name) and node.value.id in aliases:
            star = True
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    if star:   # SimpleNamespace(**CONFIG).lr, argparse.set_defaults(**CONFIG)
        used |= attrs | strings
    return used


DROPOUT_CALLEE_RE = re.compile(r"dropout", re.I)   # nn.Dropout, Dropout2d, AlphaDropout, F.dropout


def _callee(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _training_loader(call: ast.Call, targets: set[str]) -> bool:
    """E10: Batchgroesse eines Trainings-Loaders, nicht eines Retrieval-, Vorberechnungs- oder Test-Loaders. Training erkennt man am Namen
    (Ziel oder Argument mit 'train') oder an shuffle=True."""
    if any(kw.arg == "shuffle" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in call.keywords):
        return True
    names = targets | {x for a in call.args for x in _names(a)} | {x for kw in call.keywords if kw.arg != "batch_size" for x in _names(kw.value)}
    return any("train" in n.lower() for n in names)


def _keyword_applies(key: str, arg: str, call: ast.Call, targets: set[str]) -> bool:
    """Gehoert das Schluesselwort `arg` dieses Aufrufs zum Steckbrief-Schluessel `key`? Gleiche Namen anderer Funktionen zaehlen nicht
    (F.normalize(p=2) ist kein Dropout, DataLoader(memory, batch_size=1024) fuer ein Retrieval keine Trainingsbatchgroesse)."""
    if arg not in KEYWORDS.get(key, ()):
        return False
    if key == "dropout" and arg == "p":
        return bool(DROPOUT_CALLEE_RE.search(_callee(call)))
    if key == "batch_size":
        return _training_loader(call, targets)
    return True


def _keyword_conflicts(tree: ast.Module, spec: Spec) -> list[dict]:
    targets: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            targets[id(node.value)] = {x for t in tg for x in _names(t)}
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            for item in spec.enforced():
                if kw.arg and isinstance(kw.value, ast.Constant) and _keyword_applies(item.key, kw.arg, node, targets.get(id(node), set())):
                    v = kw.value.value
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and not _equal(v, item.number):
                        out.append({"key": item.key, "literal": v, "line": node.lineno})
    return out


def _module_level(body: list[ast.stmt]):
    """Anweisungen, die beim Laden des Moduls genau einmal laufen: auch in if/with/try, nie in Schleifen (Lernraten-Plan) oder Funktionen."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            yield from _module_level(node.body + node.orelse)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _module_level(node.body)
        elif isinstance(node, ast.Try):
            yield from _module_level(node.body + [s for h in node.handlers for s in h.body] + node.orelse + node.finalbody)


def _config_mutations(tree: ast.Module, cfg: ast.Assign, spec: Spec) -> list[dict]:
    """Known module-level overwrites must not undo the injected spec values (E11: also through an alias, a reassignment of the
    whole dict, augmented assignment, __setitem__/pop/del and inside if/with/try). Runtime learning-rate schedules inside
    loops/functions are a separate protocol, not a conflicting initial configuration. A write that cannot be evaluated counts as a
    conflict: the enforced value is then no longer provable."""
    enforced = {i.key: i for i in spec.enforced()}
    aliases = {"CONFIG"}
    out: list[dict] = []

    def is_alias(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in aliases

    def report(key, value, node):
        keys = list(enforced) if key is None else [key] if key in enforced else []
        for k in keys:
            lit = _literal(value) if isinstance(value, ast.AST) else value
            if isinstance(lit, (int, float)) and not isinstance(lit, bool) and _equal(lit, enforced[k].number):
                continue
            shown = lit if isinstance(lit, (int, float)) and not isinstance(lit, bool) else (
                ast.unparse(value) if isinstance(value, ast.AST) else str(value))
            out.append({"key": k, "literal": shown, "line": node.lineno, "kind": "config_overwrite"})

    for node in _module_level(tree.body):
        if node.lineno <= (cfg.end_lineno or cfg.lineno):
            continue
        if isinstance(node, ast.Assign):
            if is_alias(node.value):
                aliases |= {t.id for t in node.targets if isinstance(t, ast.Name)}
                continue
            for target in node.targets:
                if isinstance(target, ast.Subscript) and is_alias(target.value):
                    k = _literal(target.slice)
                    report(k if isinstance(k, str) else None, node.value, node)
                elif is_alias(target):   # ganzes Dict neu: erzwungene Werte muessen sichtbar gleich bleiben
                    if isinstance(node.value, ast.Dict):
                        given = {_literal(k): v for k, v in zip(node.value.keys, node.value.values) if k is not None}
                        keeps = any(k is None and is_alias(v) for k, v in zip(node.value.keys, node.value.values))
                        for key in enforced:
                            if key in given:
                                report(key, given[key], node)
                            elif not keeps:
                                report(key, "<removed by reassignment>", node)
                    else:
                        report(None, node.value, node)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript) and is_alias(node.target.value):
            k = _literal(node.target.slice)
            report(k if isinstance(k, str) else None, ast.unparse(node), node)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and is_alias(target.value):
                    k = _literal(target.slice)
                    report(k if isinstance(k, str) else None, "<deleted>", node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and is_alias(node.value.func.value)):
            call, attr = node.value, node.value.func.attr
            if attr == "update":
                if call.args and isinstance(call.args[0], ast.Dict):
                    for k, v in zip(call.args[0].keys, call.args[0].values):
                        if k is None:
                            report(None, v, node)
                        elif isinstance(_literal(k), str):
                            report(_literal(k), v, node)
                elif call.args:
                    report(None, call.args[0], node)
                for kw in call.keywords:
                    report(kw.arg, kw.value, node)
            elif attr == "__setitem__" and len(call.args) == 2:
                k = _literal(call.args[0])
                report(k if isinstance(k, str) else None, call.args[1], node)
            elif attr in ("pop", "__delitem__") and call.args:
                k = _literal(call.args[0])
                report(k if isinstance(k, str) else None, "<removed>", node)
            elif attr in ("clear", "popitem"):
                report(None, f"<{attr}>", node)
    return out


def config_check(code: str, spec: Spec) -> tuple[str, dict]:
    """S2. Rueckgabe (Code mit Override-Zeilen, Bericht). Bericht: has_config, overrides [{key, old, new}], unused, conflicts, error."""
    report: dict = {"has_config": False, "overrides": [], "unused": [], "conflicts": [], "error": ""}
    enf = spec.enforced()
    if not enf:
        return code, report
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report["error"] = f"syntax: {exc.msg}"
        return code, report
    cfg = _config_node(tree)
    if cfg is None:
        report["unused"] = [i.key for i in enf]
        return code, report
    report["has_config"] = True
    present = {}
    for k, v in zip(cfg.value.keys, cfg.value.values):
        key = _literal(k) if k is not None else None
        if isinstance(key, str):
            present[key] = _literal(v)
    lines = []
    for item in enf:
        old = present.get(item.key, "<missing>")
        if item.key not in present or not _equal(old, item.number):
            lines.append(f"CONFIG[{item.key!r}] = {item.number!r}  # spec override ({item.rank})")
            report["overrides"].append({"key": item.key, "old": old if old != "<missing>" else None, "new": item.number})
    used = _used_keys(tree, cfg)
    report["unused"] = [i.key for i in enf if i.key not in used]
    report["conflicts"] = _keyword_conflicts(tree, spec) + _config_mutations(tree, cfg, spec)
    if lines:
        src = code.splitlines(keepends=True)
        end = cfg.end_lineno or cfg.lineno
        if src and not src[end - 1].endswith("\n"):
            src[end - 1] += "\n"
        code = "".join(src[:end]) + "\n".join(lines) + "\n" + "".join(src[end:])
    return code, report


# ---- S3 -------------------------------------------------------------------------------------------------------------
def item_prompt(code: str, item: SpecItem, model: str) -> str:
    return (
        f"Spec item for a reproduction of {model}:\n  {item.key} = {item.value}\n"
        f"  source ({item.rank}): \"{' '.join(item.quote.split())[:300]}\"\n\n"
        "Does the script below implement exactly this item? Judge only this item. If the item describes a computation, the script "
        "must perform the same computation (another normalisation, another layer count or another formula is 'no'). Quote the "
        "lines of the script that implement it, copied verbatim (one to five lines).\n"
        'Answer with one JSON object {"implemented": true or false, "lines": "the verbatim lines or empty", "reason": "one short '
        'sentence"} and end with it.\n\n'
        f"```python\n{code[:MAX_CODE_CHARS]}\n```"
    )


def _parse_obj(content: str) -> dict | None:
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", content or ""))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "implemented" in value:
            return value
    return None


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def lines_in_code(lines: str, code: str) -> bool:
    """Jede nichtleere zitierte Zeile (ohne Auslassungspunkte) steht, ohne Leerraum verglichen, im Code; mindestens eine Zeile."""
    hay = _squash(code)
    parts = [ln.strip() for ln in re.split(r"\n|\.\.\.|\u2026", lines or "") if ln.strip()]
    parts = [p for p in parts if len(_squash(p)) >= 4]
    return bool(parts) and all(_squash(p) in hay for p in parts)


def judge_answer(content: str, code: str) -> tuple[bool, str, str]:
    """Rueckgabe (umgesetzt, Zeilen, Grund). 'ja' zaehlt nur mit woertlichem Zitat aus dem Code."""
    obj = _parse_obj(content)
    if obj is None:
        return False, "", "Antwort ohne JSON"
    lines, reason = str(obj.get("lines", "") or ""), str(obj.get("reason", "") or "")[:300]
    if obj.get("implemented") is not True:
        return False, lines, reason or "nicht umgesetzt"
    if not lines_in_code(lines, code):
        return False, lines, "Zitat steht nicht im Code"
    return True, lines, reason


def check_items(ask, code: str, spec: Spec) -> list[dict]:
    """S3: ask(system, user) -> Antworttext. Eine Frage je nicht erzwungenem Eintrag."""
    out = []
    for item in spec.checkable():
        ok, lines, reason = judge_answer(ask(CHECK_SYSTEM, item_prompt(code, item, spec.model)), code)
        out.append({"key": item.key, "value": item.value, "ok": ok, "lines": lines[:500], "reason": reason})
    return out


def failures(config: dict, items: list[dict]) -> list[dict]:
    out = []
    if config.get("error"):
        return out
    if not config.get("has_config") and config.get("unused"):
        out.append({"key": "CONFIG", "kind": "no_config"})
    else:
        out += [{"key": k, "kind": "unused"} for k in config.get("unused", [])]
    out += [{"key": c["key"], "kind": "overwrite" if c.get("kind") == "config_overwrite" else "literal", "literal": c["literal"],
             "line": c["line"]} for c in config.get("conflicts", [])]
    out += [{"key": r["key"], "kind": "item", "reason": r["reason"]} for r in items if not r["ok"]]
    return out


def score(spec: Spec, config: dict, items: list[dict]) -> tuple[int, int]:
    """(erfuellt, gesamt): erzwungene Schluessel zaehlen, wenn sie gelesen werden und kein Literal kollidiert; dazu S3-Eintraege."""
    enf = [i.key for i in spec.enforced()]
    bad = {c["key"] for c in config.get("conflicts", [])} | set(config.get("unused", []))
    ok_cfg = 0 if config.get("error") else len([k for k in enf if k not in bad])
    return ok_cfg + sum(1 for r in items if r["ok"]), len(enf) + len(items)


def repair_message(fails: list[dict], spec: Spec) -> str:
    """Gezielte Reparatur nur fuer die Befunde; Steckbrief-Eintrag mit Quelle. Keine Zahlen aus Messungen."""
    lines = []
    for f in fails:
        item = spec.get(f["key"])
        src = f" (source {item.rank}: \"{' '.join(item.quote.split())[:200]}\")" if item else ""
        if f["kind"] == "no_config":
            keys = ", ".join(f"\"{i.key}\": {i.number!r}" for i in spec.enforced())
            lines.append(f"- The script has no literal CONFIG dict. Add CONFIG = {{{keys}}} right after the imports and read these "
                         "hyperparameters only from it.")
        elif f["kind"] == "unused":
            lines.append(f"- CONFIG[\"{f['key']}\"] is never read. Use CONFIG[\"{f['key']}\"] where {f['key']} is applied{src}.")
        elif f["kind"] == "overwrite":
            lines.append(f"- Line {f['line']} changes CONFIG[\"{f['key']}\"] after the CONFIG block ({f['literal']}). Remove this write; "
                         f"the reference value must reach training unchanged{src}.")
        elif f["kind"] == "literal":
            lines.append(f"- Line {f['line']} hard-codes {f['key']} = {f['literal']!r}. Use CONFIG[\"{f['key']}\"] instead{src}.")
        else:
            lines.append(f"- {f['key']} must be: {item.value if item else '?'}{src}. Current script: {f.get('reason', 'not implemented')}.")
    return (
        "Your script runs, but it does not follow the reproduction spec in these points:\n" + "\n".join(lines) + "\n"
        "Fix exactly these points and keep everything else (data loading, output format, other settings) unchanged. "
        "Return the corrected full script."
    )
