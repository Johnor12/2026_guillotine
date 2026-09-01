#!/usr/bin/env python3
"""Pull the live board, re-evaluate source matches, and re-rank between picks.

Three steps, each of which stands alone; this fixes the order and stops on the first
failure:

    1. draft/fetch_draft.py       Sleeper's draft API -> draft.json
    2. sources/investigate.py     provider boards + draft.json -> data_source_matches.json
    3. rank.py                    pool + draft + source matches -> rankings.json

The pool build and the provider-board fetch are deliberately not steps: they are re-run
only when their underlying projections or rankings change, and this loop applies the
existing snapshots to the current draft without hitting any ranking provider. Source
investigation precedes ranking because each simulated opponent consumes its latest match.

Usage:
    uv run refresh.py            # draft.json, data_source_matches.json, rankings.json
    uv run refresh.py --report   # + every step's validation summary on stderr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", action="store_true", help="pass --report to every step")
    args = ap.parse_args(argv)

    report = ["--report"] if args.report else []
    steps = [
        ("draft", ["draft/fetch_draft.py", *report]),
        ("investigate", ["sources/investigate.py", *report]),
        ("rank", ["rank.py", *report]),
    ]

    for number, (name, command) in enumerate(steps, start=1):
        print(f"\n=== [{number}/{len(steps)}] {name} ===", file=sys.stderr)
        started = time.monotonic()
        code = subprocess.run(["uv", "run", *command], cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"refresh failed at step '{name}' (exit {code})", file=sys.stderr)
            return code
        print(f"--- {name} ok in {time.monotonic() - started:.1f}s", file=sys.stderr)

    print("\nrefresh complete -> rankings.json + data_source_matches.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
