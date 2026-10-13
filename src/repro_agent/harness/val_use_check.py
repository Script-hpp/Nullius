"""B1 (v5 Schritt B): Werden die Validierungslabels im Training genutzt? Zwei deterministische Teile, kein LLM.

1. `classify_val_mask`: Vergleich der Validierungs-Metrik des Harness mit und ohne Maske der Validierungslabels
   (zweiter Lauf, Schwellen wie `classify_ablation`).
2. `static_val_use`: statische Taint-Analyse des Agenten-Codes. Trennt "Validierungslabels fliessen in einen Gradienten-/Fit-Schritt"
   (`training`) von "nur Modellauswahl/Early Stopping" (`selection`). Nur ein Hinweis, nie allein ein Ausschlussgrund.

Taint-Quellen: der Schluessel 'val_idx' und Masken ueber alle sichtbaren Labels (`labels >= 0`, `!= -1`, `> -1`; Testlabels sind
-1, die Maske umfasst also Train + Val). Senken: `.backward()` auf einem getainteten Wert, `.fit(...)`/`lstsq`/`solve` mit
getainteten Argumenten. Ein Validierungs-Loss, der nie zurueckgerechnet wird (Early Stopping), ist keine Senke.
"""

from __future__ import annotations

import ast

USED_DROP = 0.10      # Abfall >= : Validierungslabels im Training genutzt
UNUSED_DROP = 0.02    # Abfall <= : nicht genutzt

FIT_ATTRS = {"fit", "partial_fit", "lstsq", "solve"}
MAX_DEPTH = 6
# Klassenzahl o. ae. (`labels[labels >= 0].max() + 1`, `torch.unique(labels[labels >= 0])`) ist keine Label-Nutzung im Training
SHAPE_FUNCS = {"max", "min", "unique", "numel", "len", "bincount", "nunique", "item", "tolist", "size", "shape"}


def classify_val_mask(val_full: float | None, val_masked: float | None) -> str:
    """Validierungslabels genutzt | nicht genutzt | unklar | nicht messbar (Accuracy, hoeher ist besser)."""
    if val_full is None or val_masked is None:
        return "nicht messbar"
    drop = round(val_full - val_masked, 9)
    if drop >= USED_DROP:
        return "Validierungslabels genutzt"
    if drop <= UNUSED_DROP:
        return "Validierungslabels nicht genutzt"
    return "unklar"


def _key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _key(node.value)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Subscript):
        return _key(node.value)
    return None


def _is_int(node: ast.AST, value: int) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == value
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and _is_int(node.operand, -value)


def _labelish(node: ast.AST) -> bool:
    """Linke Seite sieht nach einem Label-Vektor aus (labels, y, targets, lbl); schliesst `epoch >= 0` aus."""
    k = (_key(node) or "").lower().split(".")[-1]
    return k in {"y", "yy", "lab", "labs"} or any(w in k for w in ("label", "target", "lbl"))


def _label_mask(node: ast.Compare) -> bool:
    """`x >= 0`, `x > -1`, `x != -1` (alle sichtbaren Labels)."""
    if len(node.ops) != 1 or not _labelish(node.left):
        return False
    op, right = node.ops[0], node.comparators[0]
    return ((isinstance(op, ast.GtE) and _is_int(right, 0)) or (isinstance(op, ast.Gt) and _is_int(right, -1))
            or (isinstance(op, ast.NotEq) and _is_int(right, -1)))


class _Analyzer:
    def __init__(self, tree: ast.Module, forecasting: bool = False):
        self.forecasting = forecasting
        self.defs: dict[str, ast.FunctionDef] = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs.setdefault(n.name, n)
        self.tree = tree
        self.ignored: set[int] = set()   # Compare-Knoten innerhalb von Klassenzahl-Ausdruecken
        for n in ast.walk(tree):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Compare) and _key(n.slice.left) == _key(n.value):
                self.ignored.add(id(n.slice))   # `labels[labels >= 0]`: Liste der sichtbaren Labels (Klassenzahl), keine Maske fuer einen Loss
            if isinstance(n, ast.Call):
                f = n.func
                name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
                if name in SHAPE_FUNCS:
                    self.ignored |= {id(c) for c in ast.walk(n) if isinstance(c, ast.Compare)}
        self.hits: list[str] = []
        self.val_seen = False
        self._intrinsic: dict[str, bool] = {}
        self._active: set[tuple] = set()

    # ---- Ausdruecke ----
    def tainted(self, expr: ast.AST | None, env: set[str]) -> bool:
        if expr is None:
            return False
        for n in ast.walk(expr):
            if isinstance(n, ast.Constant) and (n.value == "val_idx" or (self.forecasting and n.value == "val")):
                self.val_seen = True
                return True
            if isinstance(n, ast.Compare) and id(n) not in self.ignored and _label_mask(n):
                return True
            if isinstance(n, (ast.Name, ast.Attribute)):
                k = _key(n)
                if k and k in env:
                    return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and self.intrinsic(n.func.id):
                return True
        return False

    def intrinsic(self, name: str) -> bool:
        """Gibt die lokale Funktion `name` schon ohne getaintete Argumente einen getainteten Wert zurueck?"""
        if name not in self.defs:
            return False
        if name not in self._intrinsic:
            self._intrinsic[name] = False   # Rekursionsschutz
            self._intrinsic[name] = self.run_def(self.defs[name], set(), set(), report=False)
        return self._intrinsic[name]

    # ---- Anweisungen ----
    def run_def(self, fn: ast.FunctionDef, params: set[str], outer: set[str], report: bool = True, depth: int = 0) -> bool:
        """Analysiert den Rumpf mit getainteten Parametern; Rueckgabe: Gibt die Funktion getaintete Werte zurueck?"""
        sig = (fn.name, frozenset(params), report)
        if sig in self._active or depth > MAX_DEPTH:
            return False
        self._active.add(sig)
        try:
            env = set(outer) | set(params)
            ret = [False]
            self.block(fn.body, env, ret, report, depth)
            return ret[0]
        finally:
            self._active.discard(sig)

    def block(self, body: list[ast.stmt], env: set[str], ret: list[bool], report: bool, depth: int, nested: bool = True) -> None:
        local_defs: dict[str, ast.FunctionDef] = {}
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_defs[st.name] = st
                continue
            self.stmt(st, env, ret, report, depth)
        for name, fn in local_defs.items():   # Closures (z. B. fuer LBFGS) laufen mit der Umgebung der Definitionsstelle
            if name not in self.defs or self.defs[name] is not fn:
                self.defs[name] = fn
            self.run_def(fn, set(), env if nested else set(), report, depth + 1)

    def stmt(self, st: ast.stmt, env: set[str], ret: list[bool], report: bool, depth: int) -> None:
        if isinstance(st, (ast.For, ast.While)):
            for _ in range(2):   # Schleifen zweimal: Zuweisung am Ende wirkt auf den Anfang
                if isinstance(st, ast.For):
                    self.assign(st.target, st.iter, env)
                else:
                    self.calls(st.test, env, report, depth)
                self.block(st.body, env, ret, report, depth)
            self.block(st.orelse, env, ret, report, depth)
        elif isinstance(st, ast.If):
            self.calls(st.test, env, report, depth)
            e1, e2 = set(env), set(env)
            self.block(st.body, e1, ret, report, depth)
            self.block(st.orelse, e2, ret, report, depth)
            env |= e1 | e2
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            for it in st.items:
                self.calls(it.context_expr, env, report, depth)
            self.block(st.body, env, ret, report, depth)
        elif isinstance(st, ast.Try):
            for b in (st.body, *[h.body for h in st.handlers], st.orelse, st.finalbody):
                self.block(b, env, ret, report, depth)
        elif isinstance(st, ast.Return):
            self.calls(st.value, env, report, depth)
            if self.tainted(st.value, env):
                ret[0] = True
        elif isinstance(st, ast.Assign):
            self.calls(st.value, env, report, depth)
            for t in st.targets:
                if isinstance(t, ast.Subscript) and (_key(t.value) or '').lower() not in {'y', 'labels', 'targets'}:
                    # Clamping model outputs to visible validation labels is not
                    # early stopping. The labels>=0 form includes validation nodes.
                    reads_labels = any(isinstance(n, ast.Subscript) and (_labelish(n.value) or
                                       (isinstance(n.slice, ast.Constant) and n.slice.value in {'labels', 'targets', 'y'}))
                                       for n in ast.walk(st.value))
                    uses_val = self.tainted(t.slice, env) or self.tainted(st.value, env)
                    uses_visible = any(isinstance(n, ast.Compare) and _label_mask(n) for n in ast.walk(t.slice))
                    if reads_labels and (uses_val or uses_visible):
                        self.hit(report, st, "Validation labels copied into prediction outputs")
                self.assign(t, st.value, env)
        elif isinstance(st, (ast.AugAssign, ast.AnnAssign)):
            if st.value is not None:
                self.calls(st.value, env, report, depth)
                target = st.target
                if isinstance(st, ast.AugAssign) and self.tainted(target, env):
                    pass   # bleibt getaintet
                else:
                    self.assign(target, st.value, env)
        elif isinstance(st, ast.Expr):
            self.calls(st.value, env, report, depth)
        else:
            for child in ast.iter_child_nodes(st):
                if isinstance(child, ast.expr):
                    self.calls(child, env, report, depth)

    def assign(self, target: ast.AST, value: ast.AST, env: set[str]) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            n = len(target.elts)
            items = None
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == n:
                items = value.elts
            elif isinstance(value, (ast.GeneratorExp, ast.ListComp)) and isinstance(value.generators[0].iter, (ast.Tuple, ast.List)) \
                    and len(value.generators[0].iter.elts) == n:
                items = value.generators[0].iter.elts   # `a, b = (f(k) for k in ("x", "y"))`: je Element
            if items is not None:
                for t, v in zip(target.elts, items):
                    self.assign(t, v, env)
                return
            for t in target.elts:   # Entpacken aus einem Aufruf: Heuristik, nur Namen mit "val" erben den Taint (sonst waere train_idx mit getaintet)
                k = _key(t) or ""
                if "val" in k.lower() and self.tainted(value, env):
                    env.add(k)
                elif _key(t):
                    env.discard(k)
            return
        k = _key(target)
        if not k:
            return
        if self.tainted(value, env):
            env.add(k)
        elif not isinstance(target, ast.Subscript):
            env.discard(k)

    # ---- Aufrufe: Senken und Funktionen mit getainteten Argumenten ----
    def calls(self, expr: ast.AST | None, env: set[str], report: bool, depth: int) -> None:
        if expr is None:
            return
        for n in ast.walk(expr):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == "backward" and self.tainted(f.value, env):
                self.hit(report, n, "backward() auf einem Wert, der aus Validierungslabels/-indizes stammt")
            attr = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if attr in FIT_ATTRS and any(self.tainted(a, env) for a in [*n.args, *[k.value for k in n.keywords]]):
                self.hit(report, n, f"{attr}() mit Validierungslabels/-indizes")
            if isinstance(f, ast.Name) and f.id in self.defs:
                fn = self.defs[f.id]
                names = [a.arg for a in fn.args.args]
                params = {names[i] for i, a in enumerate(n.args) if i < len(names) and self.tainted(a, env)}
                params |= {k.arg for k in n.keywords if k.arg and self.tainted(k.value, env)}
                if params:
                    self.run_def(fn, params, set(), report, depth + 1)

    def hit(self, report: bool, node: ast.AST, text: str) -> None:
        if report:
            msg = f"Zeile {node.lineno}: {text}"
            if msg not in self.hits:
                self.hits.append(msg)


def static_val_use(code: str, forecasting: bool = False) -> dict:
    """{"verdict": "training" | "selection" | "none" | "unklar", "hits": [...]}.
    training: Validierungslabels/-indizes fliessen in einen Gradienten- oder Fit-Schritt; selection: Validierungsknoten werden nur
    gelesen (Auswahl, Early Stopping); none: keine Validierungsknoten im Code; unklar: Code nicht parsebar."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"verdict": "unklar", "hits": []}
    a = _Analyzer(tree, forecasting)
    # Modul-Ebene: Funktionsdefinitionen werden bei Aufruf oder als Closure (Umgebung der Definitionsstelle) analysiert
    env: set[str] = set()
    a.block(tree.body, env, [False], True, 0, nested=False)
    mentions = a.val_seen or "val_idx" in code
    if a.hits:
        return {"verdict": "training", "hits": a.hits}
    return {"verdict": "selection" if mentions else "none", "hits": []}
