#!/usr/bin/env python3
"""Refresh the in-season desk: projections, league state, and this week's decisions.

Three steps, each of which stands alone; this fixes the order and stops on the first
failure:

    1. pool/fetch_projections.py   Sleeper season + DraftSharks weekly (remaining weeks)
    2. season/fetch_league.py      Sleeper rosters, FAAB, transactions, weekly projections -> league.json
    3. season.py                   the race, my lineup and claims, league odds -> season.json

No manual input: everything comes from Sleeper's and DraftSharks' public endpoints. Run
it Tuesday or Wednesday before claims process for the week's bids, and again after
lineups lock for the week's cut odds. Then `uv run serve.py` and open /season.html.

Usage:
    uv run refresh_season.py
    uv run refresh_season.py --report   # + each step's summary on stderr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="pass --report to season.py")
    args = ap.parse_args(argv)

    steps = [
        ("projections", ["pool/fetch_projections.py"]),
        ("league", ["season/fetch_league.py"]),
        ("season", ["season.py", *(["--report"] if args.report else [])]),
    ]
    for number, (name, command) in enumerate(steps, start=1):
        print(f"\n=== [{number}/{len(steps)}] {name} ===", file=sys.stderr)
        started = time.monotonic()
        code = subprocess.run(["uv", "run", *command], cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"refresh failed at step '{name}' (exit {code})", file=sys.stderr)
            return code
        print(f"--- {name} ok in {time.monotonic() - started:.1f}s", file=sys.stderr)
    print("\nrefresh complete -> league.json + season.json (serve.py, then /season.html)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
