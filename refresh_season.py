#!/usr/bin/env python3
"""Refresh the in-season desk: projections, league state, and this week's decisions.

Three steps, each of which stands alone; this fixes the order and stops on the first
failure:

    1. pool/fetch_projections.py   Sleeper season + DraftSharks weekly (remaining weeks)
    2. season/fetch_league.py      Sleeper rosters, FAAB, transactions, weekly projections -> league.json
    3. season.py                   the race, my lineup and claims, league odds -> season.json

No manual input: everything comes from Sleeper's and DraftSharks' public endpoints.
Run before the week's waiver deadline and before games start. Prints recommended
bids, drops, and the optimal lineup; these are recommendations to enter on Sleeper.
The dashboard at /season.html also reads the resulting season.json.

Usage:
    uv run refresh_season.py            # refresh + this week's bids and lineup
    uv run refresh_season.py --report   # detailed model diagnostics as well
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def report_decisions(payload: dict) -> None:
    me = payload["me"]
    claims = payload["claims"]
    print(f"\nWeek {payload['week']} — {me['name']} — FAAB remaining: ${me['faab_left']}")
    if claims["activate"]:
        print(f"\nNo longer reserve-eligible, move to the active roster: {', '.join(claims['activate'])}")
    if claims["forced_cuts"]:
        print(f"Roster over capacity, cut before any add below: {', '.join(c['name'] for c in claims['forced_cuts'])}")
    if not payload["waivers_ran"]:
        print("\nFAAB recommendations:")
    else:
        print("\nRecommendations (this week's waivers processed; players dropped since need a claim until they clear):")
    candidates = [c for c in claims["candidates"] if c["title_at_optimal"] > 0][:12]
    if candidates:
        print("Each bid is evaluated separately; treat these as alternatives.")
        for c in candidates:
            drop = c["drop"]["name"] if c["drop"] else "no drop needed"
            if c["to_reserve"]:
                drop += f"; to IR: {', '.join(c['to_reserve'])}"
            if payload["waivers_ran"] and not c["waiver_clears"]:
                action, terms = "free", "add now"
            else:
                action = f"${c['optimal_bid']}"
                terms = f"wins {c['p_win_at_optimal']:.0%}, worth up to ${c['break_even_bid']}"
                if c["waiver_clears"]:
                    clears = dt.datetime.fromisoformat(c["waiver_clears"]).astimezone()
                    terms = f"claim clears ~{clears:%a %H:%M %Z}; {terms}"
            print(
                f"  {action:<5} {c['name']} ({c['position']}) — {terms}; "
                f"drop: {drop}; this week's gain: {c['gain_this_week']:+.1f} pts"
            )
    else:
        print("  No pickups improve the modeled title odds.")

    lineup = me["lineup"]
    print(f"\nOptimal lineup — {lineup['total']:.1f} projected points (currently set: {lineup['current_total']:.1f}):")
    for slot in lineup["slots"]:
        print(f"  {slot['slot']:<5} {slot.get('name') or '(empty)':<24} {slot.get('points', 0):>5.1f}")
    for label, names in (("Start", lineup["start"]), ("Sit", lineup["sit"])):
        if names:
            print(f"  {label}: {', '.join(names)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="include detailed season model diagnostics on stderr")
    args = ap.parse_args(argv)

    steps = [
        ("projections", ["pool/fetch_projections.py"]),
        ("league", ["season/fetch_league.py"]),
        ("season", ["season.py", *(["--report"] if args.report else [])]),
    ]
    for number, (name, command) in enumerate(steps, start=1):
        print(f"\n=== [{number}/{len(steps)}] {name} ===", file=sys.stderr)
        started = time.monotonic()
        code = subprocess.run([sys.executable, *command], cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"refresh failed at step '{name}' (exit {code})", file=sys.stderr)
            return code
        print(f"--- {name} ok in {time.monotonic() - started:.1f}s", file=sys.stderr)
    report_decisions(json.loads((REPO_ROOT / "season.json").read_text(encoding="utf-8")))
    print("\nrefresh complete -> league.json + season.json (serve.py, then /season.html)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
