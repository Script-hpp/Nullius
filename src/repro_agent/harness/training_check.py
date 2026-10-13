"""Provable checkpoint aliasing errors, checked before sandbox execution.

No target, dataset values or model calls. Saving a state_dict immediately is
allowed. Restoring a live/shallow state captured in a training loop is not.
"""
from __future__ import annotations
import ast


def checkpoint_issues(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # syntax handling belongs to the existing checks

    def shallow(n):
        if isinstance(n, ast.DictComp):   # {k: v for k, v in model.state_dict().items()} haelt dieselben Tensoren (E18)
            gen = n.generators[0] if n.generators else None
            source = gen.iter.func.value if (gen and isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Attribute)
                                             and gen.iter.func.attr == 'items') else None
            return source is not None and shallow(source) and not independent(n.value)
        if not isinstance(n, ast.Call):
            return False
        if isinstance(n.func, ast.Attribute) and n.func.attr == 'state_dict':
            return True
        if isinstance(n.func, ast.Attribute) and n.func.attr == 'copy':
            return shallow(n.func.value) or bool(n.args and shallow(n.args[0]))
        name = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ''
        return name in ('dict', 'OrderedDict') and bool(n.args and shallow(n.args[0]))

    def independent(value):
        """Eigene Kopie eines Tensors: clone(), deepcopy, copy_(), numpy()/tolist() o. ae.; detach() allein teilt den Speicher."""
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ('clone', 'deepcopy', 'copy', 'numpy', 'tolist', 'cpu')
                   for c in ast.walk(value))

    def key(n):
        return ast.unparse(n) if isinstance(n, (ast.Name, ast.Attribute)) else None

    def training_step(n):
        """Parameter-Update in einer Schleife: backward() oder step() eines Optimierers (beliebiger Name ausser Scheduler)."""
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            return False
        if n.func.attr == 'backward':
            return True
        return n.func.attr == 'step' and 'sched' not in ast.unparse(n.func.value).lower()

    def local_nodes(root):
        yield root
        for child in ast.iter_child_nodes(root):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield from local_nodes(child)

    issues = []
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        nodes = list(local_nodes(scope))
        restores = set()
        for n in nodes:
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'load_state_dict':
                arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == 'state_dict'), None)
                if key(arg) if arg is not None else None:
                    restores.add(key(arg))
        for loop in nodes:
            if not isinstance(loop, (ast.For, ast.While)):
                continue
            loop_nodes = list(local_nodes(loop))
            if not any(training_step(n) for n in loop_nodes):
                continue
            for n in loop_nodes:
                if isinstance(n, ast.Assign) and shallow(n.value) and any(key(t) in restores for t in n.targets):
                    issue = f"Checkpoint in line {n.lineno} shares live parameter tensors. Use copy.deepcopy(model.state_dict()) or clone each tensor; otherwise the best epoch is overwritten by later training."
                    if issue not in issues:
                        issues.append(issue)
    return issues
