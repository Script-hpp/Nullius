"""Prototype for docs/gat_failure_analysis.md (analysis only, parses text with ast, never executes agent code): flag Python loops over node counts that contain tensor ops. Usage: python3 -W ignore -I speed_lint_prototype.py <recall_cg_gate_samples dir>"""
import ast, sys, glob, os, collections
NODE_NAMES = {"N","num_nodes","n_nodes","nnodes","nb_nodes"}
def bound_is_graph_size(expr):
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id in NODE_NAMES: return True
        if isinstance(n, ast.Attribute) and n.attr in ("num_nodes","n_nodes"): return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "size" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 0: return True
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "shape" and isinstance(n.slice, ast.Constant) and n.slice.value == 0: return True
    return False
def iter_is_loop_over_nodes(it):
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
        return len(it.args)==1 and bound_is_graph_size(it.args[0])
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("unique","tolist","nonzero"): return True
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate": return iter_is_loop_over_nodes(it.args[0]) if it.args else False
    return False
def body_has_tensor_ops(body):
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (n.func.attr in ("softmax","exp","sum","mean","max","index_add_","matmul","mm","cat","stack") or (isinstance(n.func.value, ast.Name) and n.func.value.id in ("torch","F"))): return True
    return False
def lint(path):
    try: tree = ast.parse(open(path).read())
    except SyntaxError: return ["syntax"]
    hits=[]
    for n in ast.walk(tree):
        if isinstance(n, ast.For) and iter_is_loop_over_nodes(n.iter) and body_has_tensor_ops(n.body):
            hits.append(n.lineno)
    return hits
if __name__ == "__main__":
    root = sys.argv[1]
    per = collections.defaultdict(lambda:[0,0])
    detail=[]
    for p in sorted(glob.glob(f"{root}/*/code_*.py")+glob.glob(f"{root}/*/final_code_*.py")+glob.glob(f"{root}/*/run_*/*.py")):
        u = p[len(root)+1:].split("/")[0]
        h = lint(p)
        per[u][0]+=1
        if h and h!=["syntax"]:
            per[u][1]+=1; detail.append((p[len(root)+1:],h[:3]))
    for u,(t,h) in sorted(per.items()): print(f"{u:24s} files={t:3d} flagged={h}")
    for d in detail[:40]: print("  ",d)
