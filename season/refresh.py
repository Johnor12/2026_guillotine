#!/usr/bin/env python3
"""Refresh the in-season desk: projections, league state, and this week's decisions.

Three steps, each of which stands alone; this fixes the order and stops on the first
failure:

    1. shared.fetch_projections   Sleeper season + DraftSharks weekly (remaining weeks)
    2. season.fetch_league        Sleeper rosters, FAAB, transactions, weekly projections -> league.json
    3. season.run                 the race, my lineup and claims, league odds -> season.json

No manual input: everything comes from Sleeper's and DraftSharks' public endpoints.
Run before the week's waiver deadline and before games start. The season step prints
recommended bids, drops, and the optimal lineup; these are recommendations to enter on
Sleeper. The dashboard at /season.html also reads the resulting season.json.

Usage:
    uv run -m season.refresh            # refresh + this week's bids and lineup
    uv run -m season.refresh --report   # detailed model diagnostics as well
    uv run -m season.refresh --sims 256 # a quick check with fewer simulated seasons
"""

from __future__ import annotations

import argparse
import sys
import time

from shared import fetch_projections

from . import fetch_league, run
from .race import RACE_SIMS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="include detailed season model diagnostics on stderr")
    ap.add_argument("--sims", type=int, default=RACE_SIMS, help=f"simulated seasons (default {RACE_SIMS})")
    args = ap.parse_args(argv)

    steps = [
        ("projections", fetch_projections.main, []),
        ("league", fetch_league.main, []),
        ("season", run.main, ["--sims", str(args.sims), *(["--report"] if args.report else [])]),
    ]
    for number, (name, step, argv) in enumerate(steps, start=1):
        print(f"\n=== [{number}/{len(steps)}] {name} ===", file=sys.stderr)
        started = time.monotonic()
        code = step(*([argv] if argv else []))
        if code != 0:
            print(f"refresh failed at step '{name}' (exit {code})", file=sys.stderr)
            return code
        print(f"--- {name} ok in {time.monotonic() - started:.1f}s", file=sys.stderr)
    print("\nrefresh complete -> league.json + season.json (shared.serve, then /season.html)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
