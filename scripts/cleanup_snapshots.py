"""Raeumt Nebius-Snapshots (Tag-Prefix `repro-snapshot-`) auf. Nur Tags werden entfernt.

Beispiele:
  uv run python scripts/cleanup_snapshots.py --older-than 24h --dry-run
  uv run python scripts/cleanup_snapshots.py --older-than 7d --keep demo- --keep bench-

Snapshots, deren Label mit einem --keep-Prefix beginnt, bleiben immer (Belege fuer
Demo/Benchmark, auf die der Bericht verweist). Ohne --older-than passiert nichts.
"""

from __future__ import annotations

import argparse
import re
from datetime import timedelta

from repro_agent import config  # noqa: F401  laedt .env
from repro_agent.sandbox.nebius import SNAPSHOT_PREFIX


def parse_age(text: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([hd])", text)
    if not m:
        raise argparse.ArgumentTypeError("Format: 24h oder 7d")
    n = int(m.group(1))
    return timedelta(hours=n) if m.group(2) == "h" else timedelta(days=n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--older-than", type=parse_age, required=True)
    ap.add_argument(
        "--since", type=parse_age, default=timedelta(days=14),
        help="Suchfenster nach hinten (Default 14d). Das Listing ist gross, Fenster klein halten.",
    )
    ap.add_argument("--keep", action="append", default=[], help="Label-Prefix, der bleibt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from contree_sdk import ContreeSync

    client = ContreeSync()
    images = client.images(
        number=None, tagged=True, since=args.since, until=args.older_than
    )
    removed = kept = 0
    for img in images:
        if not (img.tag or "").startswith(SNAPSHOT_PREFIX):
            continue
        label = img.tag[len(SNAPSHOT_PREFIX):]
        if any(label.startswith(k) for k in args.keep):
            kept += 1
            print(f"keep    {img.tag}")
            continue
        print(f"{'would untag' if args.dry_run else 'untag'} {img.tag}")
        if not args.dry_run:
            img.untag()
        removed += 1
    print(f"{removed} {'wuerden entfernt' if args.dry_run else 'entfernt'}, {kept} behalten")


if __name__ == "__main__":
    main()
