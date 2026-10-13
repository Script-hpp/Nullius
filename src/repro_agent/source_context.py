"""Whole, bounded author sources for a code-grounded reproduction.

Never turn a file prefix into a purported implementation. Paper/text-only spec
extraction keeps its older compact context; author-code mode needs the actual
model, local dependencies and training protocol.
"""
from __future__ import annotations

import ast
from .method_guard import MethodImplementationBlocked

MAX_AUTHOR_CHARS = 160_000


class SourceContextBlocked(MethodImplementationBlocked):
    """A requested author implementation could not be read completely."""


def author_excerpt(docs, model: str, dataset: str, *, budget: int = MAX_AUTHOR_CHARS,
                   listing: list[str] | None = None) -> tuple[str, list[str]]:
    from .spec import _path, is_source_text, model_files, TRAIN_RE
    from .setup_details import START_RE, CONFIG_RE
    from .repo_source import imported_files

    repo = { _path(d): d for d in docs if d.name.startswith("repo:") and is_source_text(_path(d), d.text) }
    models = model_files(list(repo.values()), model)[:2]
    paths = listing if listing is not None else list(repo)
    core = {_path(d): d for d in models}
    while True:
        needed = imported_files({p: d.text for p, d in core.items()}, paths, relative=True)
        if not needed:
            break
        for path in needed:
            if path not in repo:
                raise SourceContextBlocked(f"Author source incomplete: required local module {path} was not read. No experiment was started.")
            core[path] = repo[path]
    for path, doc in core.items():
        try:
            ast.parse(doc.text)
        except SyntaxError as exc:
            raise SourceContextBlocked(f"Author source incomplete or unreadable: {path} ({exc.msg}). No experiment was started.") from exc

    # Include the effective CLI defaults and actual training loop, not only
    # layer declarations. Whole pieces are retained or explicitly rejected.
    protocol = [d for p, d in repo.items() if p not in core and (
        TRAIN_RE.search(d.text) or 'add_argument(' in d.text or CONFIG_RE.search(p) or
        START_RE.search(p) and (dataset.lower() in d.text.lower() or model.lower() in d.text.lower()))]
    pieces = list(core.values()) + protocol
    # Schedule/preprocessing helpers are often imported by a trainer. Resolve them transitively against the full listing, with the
    # same completeness rule as the model core (E15). Files next to the claimed model (a generic experiment registry such as
    # exp_basic imports every file in models/) are other model implementations: kept when read, never required.
    model_dirs = {_path(d).rsplit("/", 1)[0] for d in models if "/" in _path(d)}
    seen = {_path(d) for d in pieces}
    frontier = {_path(d): d.text for d in protocol}
    while frontier:
        needed = [p for p in imported_files(frontier, paths, relative=True) if p not in seen]
        frontier = {}
        for path in needed:
            if path not in repo:
                if "/" in path and path.rsplit("/", 1)[0] in model_dirs:
                    continue
                raise SourceContextBlocked(f"Author source incomplete: training dependency {path} was not read. No experiment was started.")
            seen.add(path)
            pieces.append(repo[path])
            if not ("/" in path and path.rsplit("/", 1)[0] in model_dirs):
                frontier[path] = repo[path].text   # Imports anderer Modelle werden nicht weiter verfolgt
    out, names, used = [], [], 0
    for doc in pieces:
        if doc.name in names:
            continue
        piece = f"=== {doc.name} ===\n{doc.text}"
        if used + len(piece) > budget:
            raise SourceContextBlocked(f"Author source context exceeds its {budget}-character limit at {_path(doc)}. "
                                       "The reference was not silently truncated; no experiment was started.")
        out.append(piece)
        names.append(doc.name)
        used += len(piece) + 2
    return '\n\n'.join(out), names
