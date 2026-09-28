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
