"""Negative probes for ceo cached LaTeX source identity. No network: downloads are monkeypatched."""
import sys, json, gzip, hashlib, copy, dataclasses, pathlib, tempfile
code = sys.argv[1]
sys.path.insert(0, code + '/src')
import pytest
from repro_agent import claims_source as cs, discovery
from repro_agent.claims_source import PaperText, ExtractionInvalid, fetch_paper_text
mp = pytest.MonkeyPatch()
reserve = pathlib.Path(code) / 'benchmark/source_reserves/2307.03756v3.json.gz'
raw = json.loads(gzip.open(reserve, 'rt').read())
out = {'reserve': {k: (v if k != 'text' else len(v)) for k, v in raw.items()},
       'reserve_hash_ok': hashlib.sha256(raw['text'].encode()).hexdigest() == raw['sha256']}
def no_tavily():
    raise AssertionError('Tavily requested')
def setup(cache_files=(), reserve_files=(), download=''):
    tmp = pathlib.Path(tempfile.mkdtemp())
    c, r = tmp / 'cache', tmp / 'reserve'
    c.mkdir(); r.mkdir()
    for name, obj in cache_files:
        (c / name).write_text(json.dumps(obj))
    for name, obj in reserve_files:
        with gzip.open(r / name, 'wt') as f: f.write(json.dumps(obj))
    mp.setattr(cs, 'SOURCE_CACHE_DIR', c); mp.setattr(cs, 'SOURCE_RESERVE_DIR', r)
    mp.setattr(discovery, '_fetch_arxiv_source', lambda aid: download)
    mp.setattr(cs, 'latest_arxiv_version', lambda aid: None)
    return c
def run(name, version, expected=None, **kw):
    setup(**kw)
    try:
        p = fetch_paper_text('2307.03756', no_tavily, mode='live', require_latex=True, source_version=version, expected_sha256=expected)
        out[name] = f"source {p.arxiv_version} from_cache={p.from_cache} unknown={getattr(p,'current_version_unknown',None)} sha={p.sha256[:12]}"
    except ExtractionInvalid as x:
        out[name] = f"stopped: {x}"[:140]
    except AssertionError as x:
        out[name] = f"TAVILY: {x}"
good = dict(raw)
def variant(**changes):
    v = dict(good); v.update(changes); return v
new_text = good['text'] + '\n% revised'
run('reserve_used_for_v3', 'v3', reserve_files=[('2307.03756v3.json.gz', good)])
run('other_requested_version_v4', 'v4', reserve_files=[('2307.03756v3.json.gz', good)])
run('file_named_v3_but_content_v2', 'v3', cache_files=[('2307.03756v3.json', variant(arxiv_version='v2'))])
run('other_paper_id', 'v3', cache_files=[('2307.03756v3.json', variant(arxiv_id='2307.03757'))])
run('corrupt_text_hash_mismatch', 'v3', cache_files=[('2307.03756v3.json', variant(text=good['text'] + 'x'))])
run('pdf_kind', 'v3', cache_files=[('2307.03756v3.json', variant(source_kind='pdf'))])
run('truncated_json', 'v3', cache_files=[('2307.03756v3.json', 'x')])
(setup()/'2307.03756v3.json').write_text('{"arxiv_id": "2307.03756"')
try:
    fetch_paper_text('2307.03756', no_tavily, mode='live', require_latex=True, source_version='v3'); out['broken_json_file'] = 'returned'
except ExtractionInvalid as x: out['broken_json_file'] = f'stopped: {x}'
# A self-consistent but different text (rehashed) for the same version is accepted: hash is only self-checked.
forged = variant(text=new_text, sha256=hashlib.sha256(new_text.encode()).hexdigest())
run('rehashed_different_text_same_version_with_catalog_hash', 'v3', expected=good['sha256'], cache_files=[('2307.03756v3.json', forged)])
run('reserve_with_catalog_hash', 'v3', expected=good['sha256'], reserve_files=[('2307.03756v3.json.gz', good)])
run('unknown_version_name_v9_content_v3', None, cache_files=[('2307.03756v9.json', good)])
run('rehashed_without_expected_hash_now', 'v3', cache_files=[('2307.03756v3.json', forged)])
# Fresh download always wins over cache
run('fresh_download_wins', 'v3', cache_files=[('2307.03756v3.json', good)], download=discovery.ArxivSourceText(new_text, 'https://export.arxiv.org/e-print/2307.03756v3'))
# Unknown version: which stored version is chosen?
v1 = variant(arxiv_version='v1'); v10 = variant(arxiv_version='v10')
run('unknown_version_picks', None, cache_files=[('2307.03756v1.json', v1), ('2307.03756v3.json', good), ('2307.03756v10.json', v10)])
# Saved file after a fresh download: is the URL attribute serialisable and is from_cache false?
c = setup(download=discovery.ArxivSourceText(good['text'], 'https://arxiv.org/src/2307.03756v3'))
p = fetch_paper_text('2307.03756', no_tavily, mode='live', require_latex=True, source_version='v3')
saved = json.loads((c / '2307.03756v3.json').read_text())
out['saved_after_download'] = {'url': saved['url'], 'from_cache': saved['from_cache'], 'version': saved['arxiv_version'],
                               'asdict_deepcopy_ok': dataclasses.asdict(p)['text'] == good['text']}
# HTML error page and PDF bytes must not decode as LaTeX
out['html_page_decoded'] = repr(discovery._decode_arxiv_source(b'<!DOCTYPE html><html>\\begin{x} rate limited</html>')[:20])
out['pdf_bytes_decoded'] = repr(discovery._decode_arxiv_source(b'%PDF-1.5 binary')[:20])
setup(download=discovery.ArxivSourceText('', 'https://arxiv.org/src/2307.03756v3', 'HTTP 403 at https://arxiv.org/src/2307.03756v3'))
try:
    fetch_paper_text('2307.03756', no_tavily, mode='live', require_latex=True, source_version='v3'); out['error_reason']='returned'
except ExtractionInvalid as x: out['error_reason']=str(x)
mp.undo()
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
