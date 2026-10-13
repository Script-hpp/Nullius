import shutil
from pathlib import Path

import pytest

from repro_agent import config  # noqa: F401  laedt .env


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: braucht echte Nebius-Zugangsdaten")


@pytest.fixture(scope="session")
def nebius():
    import os

    if not (os.getenv("NEBIUS_API_KEY") and os.getenv("NEBIUS_PROJECT_ID")):
        pytest.skip("NEBIUS_API_KEY/NEBIUS_PROJECT_ID fehlen")
    from repro_agent.sandbox.nebius import NebiusSandbox

    sb = NebiusSandbox()
    return sb, sb.base_snapshot("python:3.11-slim", ["echo base > /base.txt"])


REAL_FIXTURES = Path(__file__).resolve().parents[1] / "benchmark" / "fixtures"


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="session", autouse=True)
def _fixtures_untouched():
    """Tests duerfen nie nach benchmark/fixtures schreiben (Live-/Fixture-Schreibpfade nutzen eine Kopie)."""
    before = _snapshot(REAL_FIXTURES)
    yield
    assert _snapshot(REAL_FIXTURES) == before, "Test hat benchmark/fixtures veraendert"


@pytest.fixture(autouse=True)
def _fixture_copy(tmp_path_factory, monkeypatch):
    """Standard-FIXTURE_DIR der Quellmodule zeigt auf eine Kopie (lesen geht, schreiben trifft das Original nicht)."""
    copy = tmp_path_factory.mktemp("fixtures") / "fixtures"
    shutil.copytree(REAL_FIXTURES, copy)
    monkeypatch.setattr("repro_agent.claims_source.FIXTURE_DIR", copy)
    monkeypatch.setattr("repro_agent.claims_source.SOURCE_CACHE_DIR", copy.parent / "paper_sources")
    monkeypatch.setattr("repro_agent.repo_source.FIXTURE_DIR", copy / "repos")


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    from repro_agent import web_guard

    web_guard.HITS.clear()
    yield


@pytest.fixture(autouse=True)
def _isolate_live_candidates(tmp_path, monkeypatch):
    from repro_agent import live_catalog

    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "live_candidates.json")
    monkeypatch.setattr(live_catalog, "CACHE_PATH", tmp_path / "discovery_cache.json")
    monkeypatch.setattr(live_catalog, "REPLAY_PATH", tmp_path / "discovery_replay.json")
    monkeypatch.setattr('repro_agent.api.REPRO_ARCHIVE_DIR', tmp_path / 'reproduction_jobs')
