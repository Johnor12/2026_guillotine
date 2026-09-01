#!/usr/bin/env python3
"""Attach DraftSharks' native per-week projections as weekly_points. Stage 5 of the build.

This stage adds ``weekly_points``: DraftSharks' own per-week projection for league
weeks 1-17, taken directly from ``data/weekly_projections.json``
(``fetch_weekly_projections.py`` captures it, already scored with the league's live
Sleeper scoring settings). A week DraftSharks does not project is a week the player
does not play and gets 0 — byes, suspensions, injury ramps, and other known absences
are explicit zero weeks. Week 18 is simply ignored: the league ends with the week
16-17 championship. The weekly numbers are the value input downstream; the Sleeper
season ``points`` column stays as the pool-membership filter and a season-level
reference, so the two columns are different providers' projections and their totals
differ by a few percent.

A pool player DraftSharks' weekly page does not carry (a handful of deep stashes)
falls back to a uniform 1/17th of his Sleeper season points per week, skipping his
``bye_week`` — a cross-provider approximation that is fine at that depth. Fallbacks
are reported loudly so a projectable player missing from the join is noticed.

Re-running is idempotent: ``weekly_points`` is recomputed from the weekly file each time.

Usage:
    uv run pool_pipeline/apply_weekly_shape.py [pool.json] [--weekly FILE] [--report]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import paths

LEAGUE_WEEKS = 17  # weeks 1-15 regular season + the 16-17 championship
WEEKLY_FIELD = "weekly_points"

WEEKLY_DESCRIPTION = (
    "DraftSharks' per-week projection for league weeks 1-17 (index 0 = week 1), scored "
    "with the league's live Sleeper scoring settings. A week DraftSharks does not "
    "project is 0.0 (bye or known absence); week 18 is ignored — the league ends after "
    "week 17. Native DraftSharks numbers: the sum is this provider's season total and "
    "differs from the Sleeper-priced `points` column by a few percent."
)


def native_weeks(ds_weeks: dict[str, dict]) -> list[float]:
    """DraftSharks' league-scored points for weeks 1-17; a missing week is 0."""
    return [
        round(max(float((ds_weeks.get(str(w)) or {}).get("points", 0.0)), 0.0), 2)
        for w in range(1, LEAGUE_WEEKS + 1)
    ]


def uniform_weeks(points: float, bye_week: int | None) -> list[float]:
    """Flat 1/17th-per-game weeks with only the bye zeroed — the sourceless fallback."""
    per_week = round(points / LEAGUE_WEEKS, 2)
    return [
        0.0 if bye_week is not None and w == bye_week else per_week
        for w in range(1, LEAGUE_WEEKS + 1)
    ]


def apply(document: dict, weekly: dict) -> list[dict]:
    """Write weekly_points onto every pool row, in place. Returns the fallback rows."""
    by_id = {p["player_id"]: p["weeks"] for p in weekly["players"]}
    fallbacks: list[dict] = []
    for row in document["players"]:
        ds_weeks = by_id.get(row["player_id"])
        if ds_weeks is None:
            fallbacks.append(row)
            row[WEEKLY_FIELD] = uniform_weeks(row["points"], row.get("bye_week"))
            continue
        row[WEEKLY_FIELD] = native_weeks(ds_weeks)

    document["fields"][WEEKLY_FIELD] = WEEKLY_DESCRIPTION
    document["weekly_source"] = {
        "file": paths.WEEKLY_PROJECTIONS.name,
        "season": weekly.get("season"),
        "fetched_at": weekly.get("fetched_at"),
        "uniform_fallbacks": len(fallbacks),
    }
    return fallbacks


def report(document: dict, fallbacks: list[dict]) -> None:
    out = sys.stderr
    rows = document["players"]
    with_zero_weeks = sum(
        1 for r in rows if any(v == 0.0 for v in r[WEEKLY_FIELD])
    )
    ratios = [sum(r[WEEKLY_FIELD]) / r["points"] for r in rows if r["points"]]
    print(
        f"\nweekly: {len(rows)} players priced natively; {with_zero_weeks} have at "
        f"least one zero week (bye or absence); mean DraftSharks-weekly total over "
        f"Sleeper season points: {sum(ratios) / len(ratios):.3f}",
        file=out,
    )
    # A player whose zero weeks go beyond one bye is carrying absence information —
    # the whole reason this stage exists. Show the biggest names among them.
    absent = [
        (r, sum(1 for v in r[WEEKLY_FIELD] if v == 0.0))
        for r in rows
        if sum(1 for v in r[WEEKLY_FIELD] if v == 0.0) > 1
    ]
    absent.sort(key=lambda t: -t[0]["points"])
    for r, zeros in absent[:8]:
        weeks = [w + 1 for w, v in enumerate(r[WEEKLY_FIELD]) if v == 0.0]
        print(
            f"  {r['name']:<22} {r['position']} {r['points']:>6} pts, "
            f"out weeks {weeks}",
            file=out,
        )
    print(file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pool", nargs="?", default=paths.POOL, type=Path)
    ap.add_argument(
        "--weekly",
        default=paths.WEEKLY_PROJECTIONS,
        type=Path,
        help="DraftSharks weekly projections (fetch_weekly_projections.py writes it)",
    )
    ap.add_argument("--report", action="store_true", help="print a validation summary to stderr")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent; 0 for compact")
    args = ap.parse_args(argv)

    for path in (args.pool, args.weekly):
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
    with args.pool.open(encoding="utf-8") as handle:
        document = json.load(handle)
    with args.weekly.open(encoding="utf-8") as handle:
        weekly = json.load(handle)

    fallbacks = apply(document, weekly)
    for row in sorted(fallbacks, key=lambda r: -r["points"]):
        print(
            f"warning: no DraftSharks weekly projection for {row['name']} "
            f"({row['position']} {row['team']}, {row['points']} pts) — uniform weeks",
            file=sys.stderr,
        )

    with args.pool.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=args.indent or None, ensure_ascii=False)
        handle.write("\n")

    print(
        f"weekly: {len(document['players'])} players priced from "
        f"{paths.display(args.weekly)} ({len(fallbacks)} uniform fallbacks) "
        f"-> {paths.display(args.pool)}",
        file=sys.stderr,
    )
    if args.report:
        report(document, fallbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
