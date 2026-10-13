import asyncio
import json
import uuid

import httpx

from repro_agent import api, llm
from test_llm_retry import _client
from test_loop_fakes import FakeSandbox, make_orch


def test_direct_client_calls_and_ask_calls_are_each_counted_once(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, 'COST_LOG_PATH', tmp_path / 'costs.jsonl')
    response = lambda: httpx.Response(200, json={'choices': [{'message': {'content': '[]'}}], 'usage': {}})
    client = _client([response(), response()])
    o, _ = make_orch(FakeSandbox())
    o.llm = client
    client.on_call = o._llm_completed
    events = []
    o._emit_live = lambda typ, **kw: events.append((typ, kw))
    client.complete('extract', [{'role': 'user', 'content': 'setup'}])
    o._ask('code', 'system', 'user')
    assert [item['role'] for typ, item in events if typ == 'llm_call'] == ['extract', 'code']
    client.close()


def test_completed_error_stream_survives_registry_pruning_and_restart(monkeypatch):
    jid = str(uuid.uuid4())
    job = api.ReproJob(jid, 'c', True, created=0, done=True)
    job.add({'ts': 1, 'type': 'error', 'message': 'budget exceeded'})
    api._archive_job(job)
    api.repro_jobs[jid] = job
    api._prune_jobs()
    assert jid not in api.repro_jobs
    restored = api._load_archived_job(jid)
    assert restored and restored.done and restored.events == job.events
    class Request:
        async def is_disconnected(self): return False
    async def read():
        return [chunk async for chunk in api._repro_events(restored, Request(), 0, api.time.monotonic())]
    assert 'budget exceeded' in ''.join(asyncio.run(read()))


def test_archive_rejects_paths_and_corrupt_files():
    assert api._load_archived_job('../secret') is None
    jid = str(uuid.uuid4())
    api.REPRO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    (api.REPRO_ARCHIVE_DIR / f'{jid}.json.gz').write_bytes(b'not gzip')
    assert api._load_archived_job(jid) is None
