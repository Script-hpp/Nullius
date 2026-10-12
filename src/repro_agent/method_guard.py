"""Reject provable retrieval placeholders; never certify general method fidelity.

No execution or model calls. Compare generated retrieval functions to corresponding
non-trivial functions in the repository. Ordinary zero initialization is allowed.
"""

import ast
import re
import textwrap


class MethodImplementationBlocked(RuntimeError):
    """Known incomplete generated method; no numeric paper verdict may be issued."""


def _placeholder(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = [s for s in function.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                           and isinstance(s.value.value, str))]
    if body and all(isinstance(s, ast.Pass) or isinstance(s, ast.Return) and (s.value is None or
                   isinstance(s.value, ast.Constant) and s.value.value is None) for s in body):
        return True
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    return bool(returns) and all(isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                                and n.value.func.attr in ("zeros", "zeros_like", "new_zeros") for n in returns)


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
    references = {function.name for text in repo_files.values() for function in _reference_functions(text)
                  if not _placeholder(function)}
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for function in node.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if 'retriev' not in (node.name + function.name).lower():
                continue
            if function.name in references and _placeholder(function):
                issues.append(f"{node.name}.{function.name} replaces a non-trivial repository function with pass/constant zeros. "
                              "Implement the retrieval mechanism; a placeholder cannot assess the paper.")
    return issues
