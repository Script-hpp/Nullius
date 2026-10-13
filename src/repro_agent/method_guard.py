"""Reject provable retrieval placeholders; never certify general method fidelity.

No execution or model calls. Compare generated retrieval functions to corresponding
non-trivial functions in the repository. Ordinary zero initialization is allowed.
"""

import ast
import re
import textwrap


class MethodImplementationBlocked(RuntimeError):
    """Known incomplete generated method; no numeric paper verdict may be issued."""


ZERO_CALLS = ("zeros", "zeros_like", "new_zeros")
FULL_CALLS = ("full", "full_like", "new_full")
HELPER_PREFIXES = ("__init__", "init", "reset", "clear", "zero", "empty", "pad", "mask", "default")


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and not isinstance(node.value, bool) and node.value == 0


def _zero_expr(node: ast.AST | None, constant_names: set[str]) -> bool:
    """Ausdruck, dessen Wert unabhaengig von den Daten nur aus Nullen besteht (E17: auch ueber eine Variable oder `0 * x`)."""
    if isinstance(node, ast.Name):
        return node.id in constant_names
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ZERO_CALLS:
            return True
        if node.func.attr in FULL_CALLS and len(node.args) >= 2 and _is_zero(node.args[1]):
            return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _is_zero(node.left) or _is_zero(node.right)
    return False


def _constant_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Namen, die genau einmal mit Nullen belegt und danach nie veraendert werden (kein Element-/In-place-Schreiben, keine Weitergabe)."""
    assigned: dict[str, int] = {}
    zero: set[str] = set()
    touched: set[str] = set()
    for n in ast.walk(function):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = assigned.get(t.id, 0) + 1
                    if _zero_expr(n.value, set()):
                        zero.add(t.id)
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    touched.add(t.value.id)
        elif isinstance(n, ast.AugAssign):
            target = n.target.value if isinstance(n.target, ast.Subscript) else n.target
            if isinstance(target, ast.Name):
                touched.add(target.id)
        elif isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.attr.endswith("_"):
                touched.add(n.func.value.id)   # result.add_(...), out.copy_(...)
            for a in [*n.args, *[k.value for k in n.keywords]]:
                if isinstance(a, ast.Name):
                    touched.add(a.id)          # fill(out), torch.add(a, b, out=out)
        elif isinstance(n, (ast.For, ast.comprehension)):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    touched.add(t.id)
    return {name for name in zero if assigned.get(name) == 1 and name not in touched}


def _placeholder(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = [s for s in function.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                           and isinstance(s.value.value, str))]
    if body and all(isinstance(s, ast.Pass) or isinstance(s, ast.Return) and (s.value is None or
                   isinstance(s.value, ast.Constant) and s.value.value is None) for s in body):
        return True
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    constant = _constant_names(function)
    return bool(returns) and all(_zero_expr(n.value, constant) for n in returns)


def _reference_functions(text: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(text)
        return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except SyntaxError:
        # Repository excerpts can end inside a later function. Use only complete
        # function bodies; truncated functions provide no proof for this guard.
        functions = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            match = re.match(r"^(\s*)(?:async )?def \w+\(", line)
            if not match:
                continue
            indent = len(match[1])
            end = i + 1
            while end < len(lines):
                next_line = lines[end]
                if next_line.strip() and not next_line.lstrip().startswith('#') and len(next_line) - len(next_line.lstrip()) <= indent:
                    break
                end += 1
            try:
                node = ast.parse(textwrap.dedent('\n'.join(lines[i:end]))).body[0]
            except (SyntaxError, IndexError):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
        return functions


def placeholder_issues(code: str, repo_files: dict[str, str]) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # the existing syntax/smoke checks handle this
    reference_functions = [(path, function) for path, text in repo_files.items() for function in _reference_functions(text)
                           if not _placeholder(function)]
    references = {function.name for _, function in reference_functions}
    # E17: a renamed placeholder (fetch_context instead of retrieve) is compared with the repository's retrieval as a whole
    repo_retrieves = any('retriev' in (path + function.name).lower() for path, function in reference_functions)
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for function in node.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if 'retriev' not in (node.name + function.name).lower() or not _placeholder(function):
                continue
            if function.name in references:
                issues.append(f"{node.name}.{function.name} replaces a non-trivial repository function with pass/constant zeros. "
                              "Implement the retrieval mechanism; a placeholder cannot assess the paper.")
            elif (repo_retrieves and len(function.args.args) > 1 and not function.name.lower().startswith(HELPER_PREFIXES)
                  and any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(function))):
                issues.append(f"{node.name}.{function.name} returns constant zeros for its input although the repository implements a "
                              "retrieval mechanism. Implement the retrieval; a placeholder cannot assess the paper.")
    return issues
