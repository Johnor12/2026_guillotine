#!/usr/bin/env python3
"""Re-price pool.json from Sleeper's projections. Stage 4 of the build.

``build_pool.py`` leaves the pool priced with DraftSharks points and
``match_sleeper.py`` leaves every player carrying a ``sleeper_id``. This stage swaps
the value column: each player's ``points`` becomes the Sleeper/Rotowire season
projection from ``data/sleeper_projections.json`` (already scored with this league's
own settings by ``fetch_sleeper_projections.py``), joined on ``sleeper_id``. Identity
fields and the DraftSharks ADP are untouched — DraftSharks still says who is in the
pool, Sleeper now says what they will score.

The two point scales must never mix (Sleeper prices Josh Allen at 351.5 where
DraftSharks says 379, and a mixed column would corrupt every replacement-level
comparison), so a pool player with no Sleeper projection is dropped, not left at his
DraftSharks number. ``sleeper_projections.json`` carries Sleeper's top 250 by points,
so the pool narrows to the players both sources know — ~217 of 417, still ~100 past
the 120-pick draft. Raise ``TOP_N`` in ``fetch_sleeper_projections.py`` if a deeper
tail is ever needed.

Sleeper players the pool cannot represent (no DraftSharks row, or an id the matcher
never assigned) are printed loudly: a missing high scorer would silently distort the
board. DEF rows are skipped — the pool has no DEF position.

Ranks are re-derived from the new points; ties keep the previous pool order.
Re-running is idempotent: already-priced rows re-join to the same values, and the
dropped count accumulates instead of resetting.

Usage:
    uv run pool_pipeline/apply_sleeper_points.py [pool.json] [--sleeper FILE] [--report]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import paths

POINTS_FIELD = "points"

POINTS_DESCRIPTION = (
    "One-season projected fantasy points under this league's own Sleeper scoring "
    "settings: the Sleeper/Rotowire season projection, joined on sleeper_id by "
    "apply_sleeper_points.py."
)


def reprice(document: dict, sleeper: dict) -> tuple[list[dict], list[dict]]:
    """Overwrite each pool row's points from Sleeper's, in place. Returns
    (dropped pool rows, unrepresented non-DEF Sleeper rows)."""
    projections = {
        row["sleeper_id"]: row for row in sleeper["players"] if row["position"] != "DEF"
    }

    kept: list[dict] = []
    dropped: list[dict] = []
    joined: set[str] = set()
    for row in document["players"]:
        source = projections.get(row.get("sleeper_id"))
        if source is None:
            dropped.append(row)
            continue
        row[POINTS_FIELD] = source[POINTS_FIELD]
        joined.add(row["sleeper_id"])
        kept.append(row)

    # Old rank breaks ties, keeping the previous (provider) order deterministic.
    kept.sort(key=lambda r: (-r[POINTS_FIELD], r["rank"]))
    seen: collections.Counter[str] = collections.Counter()
    for rank, row in enumerate(kept, start=1):
        seen[row["position"]] += 1
        row["rank"] = rank
        row["positional_rank"] = seen[row["position"]]

    document["players"] = kept
    document["player_count"] = len(kept)
    document["excluded"]["no_sleeper_projection"] = (
        document["excluded"].get("no_sleeper_projection", 0) + len(dropped)
    )
    document["scoring_scheme"]["points_copied_from"] = paths.SLEEPER_PROJECTIONS.name
    document["fields"][POINTS_FIELD] = POINTS_DESCRIPTION
    document["points_source"] = {
        "file": paths.SLEEPER_PROJECTIONS.name,
        "season": sleeper.get("season"),
        "fetched_at": sleeper.get("fetched_at"),
    }

    unrepresented = [
        row for sid, row in projections.items() if sid not in joined
    ]
    return dropped, unrepresented


def report(document: dict, dropped: list[dict]) -> None:
    out = sys.stderr
    rows = document["players"]
    counts = collections.Counter(row["position"] for row in rows)
    print(
        "\npool: " + ", ".join(f"{pos} {n}" for pos, n in counts.most_common())
        + f" = {len(rows)}",
        file=out,
    )
    if dropped:
        best = sorted(dropped, key=lambda r: -r[POINTS_FIELD])[:5]
        print(
            f"  dropped {len(dropped)} without a Sleeper projection; best by their "
            "old DraftSharks points: "
            + ", ".join(f"{r['name']} {r[POINTS_FIELD]}" for r in best),
            file=out,
        )

    ranks_ok = [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    monotone = all(
        rows[i][POINTS_FIELD] >= rows[i + 1][POINTS_FIELD] for i in range(len(rows) - 1)
    )
    per_pos: collections.Counter[str] = collections.Counter()
    positional_ok = True
    for row in rows:
        per_pos[row["position"]] += 1
        positional_ok &= row["positional_rank"] == per_pos[row["position"]]
    ids = {row["sleeper_id"] for row in rows}
    print(
        f"  rank 1..{len(rows)} gap-free: {ranks_ok}; monotone in {POINTS_FIELD}: "
        f"{monotone}; positional ranks consistent: {positional_ok}; "
        f"unique sleeper_ids: {len(ids) == len(rows)}",
        file=out,
    )

    print("\ntop 5 and the last 2 in the pool", file=out)
    for row in rows[:5] + rows[-2:]:
        print(
            f"  {row['rank']:>3} {row['name']:<22} {row['position']}"
            f"{row['positional_rank']:<3} {row[POINTS_FIELD]:>6} pts   adp {row['adp']}",
            file=out,
        )
    print(file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pool", nargs="?", default=paths.POOL, type=Path)
    ap.add_argument(
        "--sleeper",
        default=paths.SLEEPER_PROJECTIONS,
        type=Path,
        help="Sleeper projections (fetch_sleeper_projections.py writes it)",
    )
    ap.add_argument("--report", action="store_true", help="print a validation summary to stderr")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent; 0 for compact")
    args = ap.parse_args(argv)

    for path in (args.pool, args.sleeper):
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
    with args.pool.open(encoding="utf-8") as handle:
        document = json.load(handle)
    with args.sleeper.open(encoding="utf-8") as handle:
        sleeper = json.load(handle)

    if not any(row.get("sleeper_id") for row in document["players"]):
        print(
            "error: no pool player carries a sleeper_id — run fetch_sleeper.py and "
            "match_sleeper.py first",
            file=sys.stderr,
        )
        return 1

    dropped, unrepresented = reprice(document, sleeper)
    if not document["players"]:
        print("error: no pool player has a Sleeper projection", file=sys.stderr)
        return 1
    for row in sorted(unrepresented, key=lambda r: -r[POINTS_FIELD]):
        print(
            f"warning: Sleeper projects {row['name']} ({row['position']} {row['team']}, "
            f"{row[POINTS_FIELD]} pts) but the pool cannot represent him",
            file=sys.stderr,
        )

    with args.pool.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=args.indent or None, ensure_ascii=False)
        handle.write("\n")

    print(
        f"points: {len(document['players'])} players re-priced from "
        f"{paths.display(args.sleeper)}, {len(dropped)} dropped without a projection "
        f"-> {paths.display(args.pool)}",
        file=sys.stderr,
    )
    if args.report:
        report(document, dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
