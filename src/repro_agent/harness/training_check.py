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
        if not isinstance(n, ast.Call):
            return False
        if isinstance(n.func, ast.Attribute) and n.func.attr == 'state_dict':
            return True
        if isinstance(n.func, ast.Attribute) and n.func.attr == 'copy':
            return shallow(n.func.value) or bool(n.args and shallow(n.args[0]))
        return isinstance(n.func, ast.Name) and n.func.id == 'dict' and bool(n.args and shallow(n.args[0]))

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
        restores = {n.args[0].id for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'load_state_dict' and n.args and isinstance(n.args[0], ast.Name)}
        for loop in nodes:
            if not isinstance(loop, (ast.For, ast.While)):
                continue
            loop_nodes = list(local_nodes(loop))
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'step'
                       and isinstance(n.func.value, ast.Name) and
                       (n.func.value.id in {'opt', 'optim', 'model_optim'} or 'optimizer' in n.func.value.id.lower()) for n in loop_nodes):
                continue
            for n in loop_nodes:
                if isinstance(n, ast.Assign) and shallow(n.value) and any(isinstance(t, ast.Name) and t.id in restores for t in n.targets):
                    issue = f"Checkpoint in line {n.lineno} shares live parameter tensors. Use copy.deepcopy(model.state_dict()) or clone each tensor; otherwise the best epoch is overwritten by later training."
                    if issue not in issues:
                        issues.append(issue)
    return issues
