#!/usr/bin/env python3
"""Spread each pool player's season points across weeks 1-17. Stage 5 of the build.

``apply_sleeper_points.py`` leaves every pool player priced with a one-season Sleeper
projection. This stage adds ``weekly_points``: that same season total distributed over
the league's 17 playable weeks in proportion to DraftSharks' per-week projections from
``data/weekly_projections.json`` (``fetch_weekly_projections.py`` captures it). The
weekly shape is what carries byes, suspensions, injury ramps, and other known
absences — a week DraftSharks does not project is a week the player does not play and
gets 0. Totals stay in the Sleeper currency: the weekly numbers are Sleeper's season
points reshaped, never DraftSharks' own totals, so the two point scales still never mix.

Normalization runs over DraftSharks' full 18 published weeks, then week 18 is dropped:
the league ends with the week 16-17 championship, so production landing in week 18 is
genuinely worthless here and each player quietly loses that game's share (~1/17).

A pool player DraftSharks' weekly page does not carry (a handful of deep stashes) falls
back to a uniform 1/17th of his season points per week, skipping his ``bye_week`` — the
same per-game rate the reshape produces, minus any absence information. Fallbacks are
reported loudly so a projectable player missing from the join is noticed.

Re-running is idempotent: ``weekly_points`` is recomputed from ``points`` each time.

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
    "This player's season `points` spread over league weeks 1-17 (index 0 = week 1) in "
    "proportion to DraftSharks' per-week projections, which encode byes and known "
    "absences as missing weeks (-> 0.0 here). Normalized over DraftSharks' 18 weeks, "
    "so a week-18 game's share is dropped: the league ends after week 17. Sums to "
    "slightly less than `points` for that reason."
)


def weekly_shape(ds_weeks: dict[str, dict]) -> list[float] | None:
    """DraftSharks' week profile as nonnegative weights over weeks 1-18, or None."""
    raw = [max(float((ds_weeks.get(str(w)) or {}).get("points", 0.0)), 0.0) for w in range(1, 19)]
    return raw if sum(raw) > 0 else None


def uniform_weeks(points: float, bye_week: int | None) -> list[float]:
    """Flat 1/17th-per-game weeks with only the bye zeroed — the shapeless fallback."""
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
        shape = None
        ds_weeks = by_id.get(row["player_id"])
        if ds_weeks is not None:
            shape = weekly_shape(ds_weeks)
        if shape is None:
            fallbacks.append(row)
            row[WEEKLY_FIELD] = uniform_weeks(row["points"], row.get("bye_week"))
            continue
        total = sum(shape)
        row[WEEKLY_FIELD] = [
            round(row["points"] * shape[w] / total, 2) for w in range(LEAGUE_WEEKS)
        ]

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
    used = [sum(r[WEEKLY_FIELD]) / r["points"] for r in rows if r["points"]]
    print(
        f"\nweekly: {len(rows)} players shaped; {with_zero_weeks} have at least one "
        f"zero week (bye or absence); mean share of season points kept in weeks 1-17: "
        f"{sum(used) / len(used):.3f}",
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
        f"weekly: {len(document['players'])} players shaped from "
        f"{paths.display(args.weekly)} ({len(fallbacks)} uniform fallbacks) "
        f"-> {paths.display(args.pool)}",
        file=sys.stderr,
    )
    if args.report:
        report(document, fallbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
