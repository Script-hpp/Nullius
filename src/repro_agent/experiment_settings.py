"""Read literal launch settings without running any author shell/Python code.

Only singleton shell bindings and a bounded arithmetic/boolean Python subset
are understood. Unresolved or conflicting values never become a fixed setting.
"""
from __future__ import annotations

import ast
import math
import operator
import re
import shlex

UNKNOWN = object()


def active_paper_text(text: str) -> str:
    """Ignore LaTeX comments for setup evidence, preserving original source/hash."""
    if not re.search(r'\\(?:documentclass|begin|textbf|section|multicolumn)\b', text):
        return text
    lines = []
    for line in text.splitlines(keepends=True):
        for match in re.finditer('%', line):
            prefix = line[:match.start()]
            backslashes = len(prefix) - len(prefix.rstrip('\\'))
            if backslashes % 2 == 0:
                line = prefix + ('\n' if line.endswith('\n') else '')
                break
        lines.append(line)
    return ''.join(lines)


_ARITH = re.compile(r'\$\(\((.+)\)\)')
_INT_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Mod: operator.mod}


def shell_arithmetic(value: str, bindings: dict[str, str]) -> str | None:
    """`$(( $label_len * 2 ))` with singleton integer bindings -> "672"; anything else -> None (stays unresolved).
    Bash integer arithmetic only: + - * / % and parentheses; division truncates toward zero."""
    match = _ARITH.fullmatch(value.strip().strip('"'))
    if not match:
        return None
    unresolved = False

    def bound(m):
        nonlocal unresolved
        name = m[1] or m[2] or m[3]
        if not re.fullmatch(r'-?\d+', bindings.get(name, '')):
            unresolved = True
            return '0'
        return f'({bindings[name]})'
    expr = re.sub(r'\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)|\b([A-Za-z_]\w*)\b', bound, match[1])
    if unresolved or not re.fullmatch(r'[\d\s()+\-*/%]+', expr):
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                if b == 0:
                    raise ArithmeticError
                return int(a / b)
            if isinstance(node.op, ast.Mod):
                if b == 0:
                    raise ArithmeticError
                return a - b * int(a / b)   # bash keeps the dividend's sign
            op = _INT_OPS.get(type(node.op))
            if op:
                return op(a, b)
        raise ValueError('unsupported shell arithmetic')
    try:
        result = ev(ast.parse(expr.strip(), mode='eval'))
    except (SyntaxError, ValueError, ArithmeticError, RecursionError):
        return None
    return str(result) if abs(result) < 1e9 else None


def shell_calls(text: str, identities: dict | None = None) -> list[tuple[str, str, dict[str, str]]]:
    """(original invocation, safely expanded invocation, singleton bindings)."""
    bindings: dict[str, str] = {}
    options: dict[str, set[str]] = {}
    conditional = 0
    # Single-quoted variables are literal; escaped dollar signs remain literal.
    variable = re.compile(r"'[^']*'|(?<!\\)\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
    out = []
    for line in re.sub(r'\\[ \t]*\r?\n', ' ', text).splitlines():
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not tokens:
            continue
        if tokens[0] in ('if', 'case'):
            conditional += 1
        elif tokens[0] in ('fi', 'esac'):
            conditional = max(0, conditional - 1)
        name, vals = None, []
        if len(tokens) >= 4 and tokens[0] == 'for' and tokens[2] == 'in':
            vals = tokens[3:]
            if vals[-1:] == ['do']:
                vals = vals[:-1]
            vals = [v.rstrip(';') for v in vals]
            name = tokens[1]
        else:
            assignments = tokens[1:] if tokens[0] == 'export' else tokens
            if len(assignments) == 1 and re.fullmatch(r'[A-Za-z_]\w*=.+', assignments[0]):
                name, value = assignments[0].split('=', 1)
                vals = [shell_arithmetic(value, bindings) or value]
        if name:
            choices = set(vals)
            options[name] = choices if not conditional else set()
            if not conditional and len(choices) == 1 and not re.search(r'[$`{}*?;]', next(iter(choices))):
                bindings[name] = next(iter(choices))
            else:
                bindings.pop(name, None)
        if line.lstrip().startswith('#') or not re.search(r'(?<![\w-])--\w', line):
            continue
        selected = dict(bindings)
        compatible = True
        for flag, value in re.findall(r'--([\w-]+)(?:\s+|=)([^\s]+)', line):
            wanted = (identities or {}).get(flag.lower().replace('-', '_'))
            match = re.fullmatch(r'"?\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))"?', value)
            if wanted is None or match is None:
                continue
            var = match[1] or match[2]
            if var not in options:
                continue  # really unresolved, not an enumerated identity loop
            def same(v):
                if isinstance(wanted, (int, float)):
                    try: return float(v) == wanted
                    except ValueError: return False
                return v.removesuffix('.csv').casefold() == str(wanted).casefold()
            hits = [v for v in options[var] if same(v)]
            if len(hits) != 1:
                compatible = False
                break
            selected[var] = hits[0]
        if compatible:
            expanded = variable.sub(lambda m: selected.get(m[1] or m[2], m[0]) if m[1] or m[2] else m[0], line)
            out.append((line, expanded, selected))
    return out


def entrypoint(call: str) -> str | None:
    try:
        tokens = shlex.split(call, comments=True)
    except ValueError:
        return None
    for i, token in enumerate(tokens):
        if re.fullmatch(r'(?:.*/)?python(?:3(?:\.\d+)?)?', token):
            for path in tokens[i+1:]:
                if path.startswith('-'):
                    continue
                return path.removeprefix('./') if path.endswith('.py') and '$' not in path else None
    return None


def _eval(node: ast.AST, values: dict):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'args':
        return values.get(node.attr, UNKNOWN)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, values)
        if v is UNKNOWN: return UNKNOWN
        if isinstance(node.op, ast.Not): return not v
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v if isinstance(node.op, ast.UAdd) else UNKNOWN
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, values), _eval(node.right, values)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e9 for v in (a, b)):
            return UNKNOWN
        op = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
              ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod}.get(type(node.op))
        if op:
            try:
                result = op(a,b)
                return result if math.isfinite(result) and abs(result)<1e9 else UNKNOWN
            except (ArithmeticError, ValueError): return UNKNOWN
    if isinstance(node, ast.BoolOp):
        val = UNKNOWN
        for part in node.values:
            val = _eval(part, values)
            if val is UNKNOWN: return UNKNOWN
            if isinstance(node.op, ast.And) and not val: return val
            if isinstance(node.op, ast.Or) and val: return val
        return val
    if isinstance(node, ast.Compare) and len(node.ops)==1:
        a,b=_eval(node.left,values),_eval(node.comparators[0],values)
        if a is UNKNOWN or b is UNKNOWN:return UNKNOWN
        op={ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt,
            ast.LtE:operator.le, ast.Gt:operator.gt, ast.GtE:operator.ge,
            ast.In:lambda a,b:a in b, ast.NotIn:lambda a,b:a not in b}.get(type(node.ops[0]))
        try:return op(a,b) if op else UNKNOWN
        except TypeError:return UNKNOWN
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('int','float') and len(node.args)==1 and not node.keywords:
        val=_eval(node.args[0],values)
        if isinstance(val,(int,float)) and not isinstance(val,bool):return int(val) if node.func.id=='int' else float(val)
    return UNKNOWN


def python_settings(text: str, call: str, imported_sources=()) -> tuple[dict, dict[str, str]]:
    """Literal argparse defaults + explicit flags + known post-parse mutations.

    Unknown branches invalidate fields they may write; they are never guessed.
    Functions/loops/imports are not executed or interpreted.
    """
    try: tree=ast.parse(text); tokens=shlex.split(call,comments=True)
    except (SyntaxError, ValueError):return {},{}
    # Namespace names are author choices (args/opt/config), not a model rule.
    namespaces = {'args'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr in ('parse_args', 'parse_known_args'):
            namespaces.update(t.id for target in n.targets for t in ast.walk(target) if isinstance(t, ast.Name))
    def normalize(root):
        roots = set(namespaces)
        for function in ast.walk(root):
            if isinstance(function, ast.FunctionDef) and function.name == '__init__':
                parameters = {a.arg for a in function.args.args}
                for n in ast.walk(function):
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in parameters and any(
                            isinstance(t, ast.Attribute) and t.attr == 'args' for t in n.targets):
                        n.value.id = 'args'  # constructor stores its incoming namespace, not a changed set of values
        while True:
            aliases = {t.id for n in ast.walk(root) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in roots
                       for t in n.targets if isinstance(t, ast.Name)}
            if aliases <= roots: break
            roots |= aliases
        class Normalize(ast.NodeTransformer):
            def visit_Name(self, n):
                if n.id in roots: n.id = 'args'
                return n
            def visit_Attribute(self, n):
                if n.attr == 'args': return ast.copy_location(ast.Name(id='args', ctx=n.ctx), n)
                return self.generic_visit(n)
        return Normalize().visit(root)
    tree = normalize(tree)
    values, evidence, actions = {}, {}, {}
    for n in ast.walk(tree):
        if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute) or n.func.attr!='add_argument':continue
        flags=[a.value for a in n.args if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--')]
        if not flags:continue
        kwargs={k.arg:k.value for k in n.keywords}
        dest=_eval(kwargs['dest'],{}) if 'dest' in kwargs else flags[0][2:].replace('-','_')
        if not isinstance(dest,str):continue
        value=_eval(kwargs['default'],{}) if 'default' in kwargs else UNKNOWN
        action=_eval(kwargs['action'],{}) if 'action' in kwargs else ''
        if 'default' not in kwargs and action in ('store_true', 'store_false'):
            value = action == 'store_false'
        for f in flags:actions[f]=(dest,action)
        if dest in values and values[dest]!=value: value=UNKNOWN
        values[dest]=value
        evidence[dest]=ast.get_source_segment(text,n) or ''
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'set_defaults':
            for kw in n.keywords:
                if kw.arg:
                    values[kw.arg] = _eval(kw.value, {})
                    evidence[kw.arg] = ast.get_source_segment(text, n) or ''
    i=0
    while i<len(tokens):
        token=tokens[i]; i+=1
        if token in ('|',';','&&','||'):break
        if not token.startswith('--'):continue
        flag,sep,raw=token.partition('=')
        if flag not in actions:
            matches = [f for f in actions if f.startswith(flag)]
            if len(matches) == 1:
                flag = matches[0]
        dest,action=actions.get(flag,(flag[2:].replace('-','_'),''))
        if action in ('store_true','store_false'):
            value=action=='store_true'
        else:
            if not sep:
                if i>=len(tokens) or tokens[i].startswith('--'): values[dest]=UNKNOWN;continue
                raw=tokens[i];i+=1
            if '$' in raw or '`' in raw: value=UNKNOWN
            else:
                try:value=ast.literal_eval(raw)
                except (ValueError,SyntaxError):value=raw
        values[dest]=value; evidence[dest]=token+(' '+raw if not sep and action not in ('store_true','store_false') else '')

    def target(n):
        return n.attr if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args' else None
    def invalidate(node):
        for n in ast.walk(node):
            if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and (key:=target(n)):
                values[key]=UNKNOWN
    def process(nodes):
        for n in nodes:
            if isinstance(n,ast.Assign):
                for t in n.targets:
                    if key:=target(t):
                        values[key]=_eval(n.value,values)
                        evidence[key]=(evidence.get(key,'')+'\n'+(ast.get_source_segment(text,n) or '')).strip()
            elif isinstance(n,ast.AugAssign) and (key:=target(n.target)):
                values[key]=_eval(ast.BinOp(n.target,n.op,n.value),values)
                evidence[key]=(evidence.get(key,'')+'\n'+(ast.get_source_segment(text,n) or '')).strip()
            elif isinstance(n,ast.If):
                cond=_eval(n.test,values)
                if cond is UNKNOWN:invalidate(n)
                else:process(n.body if cond else n.orelse)
            elif isinstance(n,(ast.For,ast.While,ast.Try,ast.With)):
                invalidate(n)
    process(tree.body)
    # Mutation inside a helper is outside the interpreted module-level subset.
    # Never treat the pre-helper defaults as its effective configuration.
    all_unknown = False
    audit_trees = [tree]
    for source in imported_sources:
        try: audit_trees.append(normalize(ast.parse(source)))
        except SyntaxError:
            values = {k: UNKNOWN for k in values}
    for node in (n for root in audit_trees for n in ast.walk(root)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            invalidate(node)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'setattr' and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'args':
                all_unknown = True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('update', '__setitem__', 'setdefault', 'pop', 'clear'):
                receiver = node.func.value
                if (isinstance(receiver, ast.Attribute) and isinstance(receiver.value, ast.Name) and receiver.value.id == 'args' and receiver.attr == '__dict__') or (
                    isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) and receiver.func.id == 'vars' and receiver.args and isinstance(receiver.args[0], ast.Name) and receiver.args[0].id == 'args'):
                    all_unknown = True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for target in node.targets
                                               for t in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])):
            parsed = (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ('parse_args', 'parse_known_args')) or (
                      isinstance(node.value, ast.Name) and node.value.id == 'args')
            if not parsed:
                all_unknown = True
    if all_unknown:
        values = {k: UNKNOWN for k in values}
    return {k:v for k,v in values.items() if v is not UNKNOWN}, evidence


def namespace_reads(text: str, values: dict) -> set[str]:
    """Argument reads in source, excluding declarations and inactive known branches."""
    try: tree = ast.parse(text)
    except SyntaxError: return set()
    out = set()
    def scan(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            return
        if isinstance(node, ast.If):
            # Trainer namespaces commonly use self.args; evaluate only their
            # literal conditions, never imports/functions.
            import copy
            test = copy.deepcopy(node.test)
            class Normalize(ast.NodeTransformer):
                def visit_Attribute(self, n):
                    if isinstance(n.value, ast.Name) and n.value.id == 'self' and n.attr == 'args':
                        return ast.Name(id='args', ctx=ast.Load())
                    return self.generic_visit(n)
            cond = _eval(Normalize().visit(test), values)
            scan(node.test)
            for n in node.body + node.orelse if cond is UNKNOWN else (node.body if cond else node.orelse): scan(n)
            return
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name) and node.value.id in ('args','opt','cfg','config'):
                out.add(node.attr)
            elif isinstance(node.value, ast.Attribute) and node.value.attr == 'args':
                out.add(node.attr)
        for child in ast.iter_child_nodes(node):scan(child)
    scan(tree)
    return out
