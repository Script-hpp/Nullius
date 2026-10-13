"""Raw dataset identity shared by preparation and independent scoring."""
import hashlib
import json
import os
from pathlib import Path


def hashes(paths) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(map(Path, paths))}


def write_manifest(out_npz: Path, paths) -> None:
    Path(str(out_npz) + '.source.json').write_text(json.dumps(hashes(paths), sort_keys=True))


def verify(paths) -> None:
    expected = os.environ.get('NULLIUS_DATA_IDENTITY')
    if expected is not None and hashes(paths) != json.loads(expected):
        raise ValueError('Raw dataset changed since preparation; independent scoring refused')
