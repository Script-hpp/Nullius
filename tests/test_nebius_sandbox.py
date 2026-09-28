import pytest

pytestmark = pytest.mark.integration


def test_branches_are_isolated(nebius):
    sb, snap = nebius
    a, b = sb.branch(snap), sb.branch(snap)
    sb.put_file(a, "/tmp/x.txt", b"hello")
    assert sb.run(a, "cat /tmp/x.txt", 60).stdout == "hello"
    assert sb.run(b, "ls /tmp/x.txt", 60).exit_code != 0


def test_chained_runs_keep_state(nebius):
    sb, snap = nebius
    a = sb.branch(snap)
    sb.run(a, "echo mut > /base.txt", 60)
    assert sb.run(a, "cat /base.txt", 60).stdout == "mut\n"


def test_get_file_reads_branch_state(nebius):
    sb, snap = nebius
    a = sb.branch(snap)
    sb.put_file(a, "/tmp/x.txt", b"hello")
    assert sb.get_file(a, "/tmp/x.txt") == b"hello"
    with pytest.raises(FileNotFoundError):
        sb.get_file(a, "/tmp/nope")


def test_timeout_flagged(nebius):
    sb, snap = nebius
    r = sb.run(sb.branch(snap), "sleep 20", 3)
    assert r.timed_out


@pytest.mark.parametrize("mb", [1, 10, 50])
def test_get_file_large(nebius, mb):
    sb, snap = nebius
    a = sb.branch(snap)
    sb.run(a, f"head -c {mb * 1024 * 1024} /dev/urandom > /tmp/big.bin", 120)  # inkompressibel
    data = sb.get_file(a, "/tmp/big.bin")
    assert len(data) == mb * 1024 * 1024


def test_unshare_blocks_network(nebius):
    sb, snap = nebius
    b = sb.branch(snap)
    py = "import socket;s=socket.socket();s.settimeout(5);s.connect(('140.82.112.3',443));print('CONNECTED')"
    assert "CONNECTED" in sb.run(b, f'python -c "{py}"', 60).stdout  # ohne unshare: Netz da
    r = sb.run(b, f'unshare -n python -c "{py}"', 60)
    assert "CONNECTED" not in r.stdout and r.exit_code != 0
