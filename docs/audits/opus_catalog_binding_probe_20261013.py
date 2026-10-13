"""Run all catalog entries through the ceo target binder on fresh public LaTeX sources. No model/network calls."""
import json, sys, os, re
code, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, code + '/src')
from repro_agent import catalog
from repro_agent.claims_source import (PaperText, verify_selected_target, _source_table_target, ExtractionInvalid,
                                       _column_status, _norm)
SRC = ['/tmp/nullius-catalog-audit', os.path.dirname(out) + '/sources']
def load(aid):
    for d in SRC:
        p = f'{d}/{aid}.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if len(j.get('text', '')) > 2000:
                return PaperText(**{k: j.get(k) for k in PaperText.__dataclass_fields__}), d
    return None, None
rows = []
for e in catalog.load_catalog(catalog.CATALOG_YAML)['candidates'] if isinstance(catalog.load_catalog(catalog.CATALOG_YAML), dict) else catalog.load_catalog(catalog.CATALOG_YAML):
    r = {'id': e['id'], 'arxiv_id': e['arxiv_id'], 'status': e.get('status')}
    try:
        paper, claims = catalog.build_paper(e)
    except Exception as x:
        r['build'] = f'NotRunnable: {x}'; rows.append(r); continue
    spec = paper['claims'][0]['extract']; c = claims[0]
    r.update(target=c.target, input_length=paper.get('input_length'), input_length_quote=bool(paper.get('input_length_quote')),
             warn_length=any('Eingabelaenge' in w for w in paper['warnings']))
    src, where = load(e['arxiv_id'])
    if src is None:
        r['source'] = 'unreadable'; rows.append(r); continue
    r['source'] = f"{src.source_kind} {src.sha256[:12]} ({'codex' if 'nullius' in where else 'own'})"
    q = paper['discovery_quote']
    r['quote_in_source'] = _norm(q) in _norm(src.text)
    try:
        ev = verify_selected_target(src, spec['method'], spec['dataset'], spec['metric'], c.target, q, spec['percent'])
        r['binder'] = 'fallback (None)' if ev is None else f"verified {ev.value} {ev.table}"
    except ExtractionInvalid as x:
        r['binder'] = f'BLOCK: {x}'
    try:
        ev2 = _source_table_target(src, spec['method'], spec['dataset'], spec['metric'], spec['percent'])
        r['deterministic'] = None if ev2 is None else f"{ev2.value} ({ev2.column_check})"
    except ExtractionInvalid as x:
        r['deterministic'] = f'invalid: {x}'
    rows.append(r)
json.dump(rows, open(out, 'w'), indent=1)
for r in rows: print(json.dumps(r, ensure_ascii=False))
