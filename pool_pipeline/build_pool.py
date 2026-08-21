#!/usr/bin/env python3
"""Cut projections.json down to this league's draft pool: one row, one value column.

Stage 2 of the build. ``parse_projections.py`` produces the full provider export —
900 players x 8 scoring schemes x 4 horizons, ~2.9 MB, most of it irrelevant to a
10-team 0.5 PPR redraft. This narrows it to what a draft board actually
consumes and drops everything else:

    900 players  ->  QB/RB/WR/TE only         (K and IDP have no roster slot; the
                                               drafted D/ST has no source rows at all)
                 ->  a usable one-season point total (~417 players; all of them kept)

    8 schemes x 4 horizons  ->  one column: one-season points in this league's scoring
    8 ADP columns           ->  one column: 1QB ADP, as an overall pick number

**Scoring.** The league is 0.5/rec with no tight end premium, one QB. The provider's
``half_ppr`` column prices exactly that for every position, so cells are copied from
it, never computed. The 1QB family is used because this league has no superflex; the
provider's ADP responds only to 1QB vs superflex, so ``half_ppr`` ADP is the 1QB ADP.

The one-season point total is the only value column kept. The saved page is the
provider's dynasty export, but its 1-year projection is an ordinary season projection;
the multi-year horizons and 3D value are simply not carried. 3D value is deliberately
excluded either way: it is a provider-scaled ordinal (best player pinned at 100) that
bakes in someone else's roster assumptions, is not in points, and so cannot enter a
points-based expected-lineup value — which is how ``rank.py`` prices the pool.

**ADP** is the 1QB ADP, copied. The source encodes it as round.pick for a 12-team
draft — "2.03" is round 2 pick 3, not a decimal — which is both a different league
size than ours and a trap for anything that sorts numerically, so it is decoded to an
integer overall pick. Its deep tail is provider noise: a pick number in the thousands
means "effectively undrafted", not a real slot.

**Ranking.** The pool is ordered by one-season points descending, ties broken by the
provider's overall rank. So the emitted ``rank`` is verifiable from the emitted
``points`` column and the file references no quantity it does not contain.

**Cleanups applied.** Players with a 0 or missing one-season projection are dropped
rather than ranked last (the source uses 0 where a null belongs). The teamless players
carry ``bye_week: 18``, a sentinel — real byes run weeks 5-14 — so their bye is nulled.
Stale printed ranks, undocumented percent_low/percent_high, hidden_row, analyst
comments and profile paths are not carried: recover them from projections.json,
which this script only reads.

Usage:
    uv run build_pool.py [projections.json] [-o pool.json] [--limit 1000] [--report]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEME = "half_ppr"
HORIZON = "1yr"
POINTS_FIELD = "points"

#: Roster slots exist for these only; K and IDP (DB/DL/LB) are dropped whole.
POSITIONS = ("QB", "RB", "WR", "TE")

#: Effectively no cut: only ~417 of the 900 source rows are offensive players with a
#: usable projection, and the draft takes 120 of them (the D/ST round takes none).
RANK_LIMIT = 1000

#: The published 1QB column that prices this league's 0.5/rec for every position.
POINTS_COLUMN = "half_ppr"

#: ADP is roster-format dependent only, so the 1QB family is this league's ADP.
#: ADP_COLUMN is the one read, the rest are cross-checked by --report.
ADP_COLUMN = "half_ppr"
ADP_FAMILY = ("standard", "half_ppr", "ppr", "te_premium")

#: The source's ADP is round.pick for a 12-team draft. That is the provider's format,
#: not this league's size, and it is decoded rather than reinterpreted.
TEAMS_PER_ROUND = 12
PLAUSIBLE_ROUNDS = 45  # ~490 ranked players / 12 per round; beyond this is tail noise

#: Sentinels the source uses for unsigned players: no team, so no real bye week.
NO_TEAM = frozenset({"UNS", "RK"})
PLACEHOLDER_BYE = 18

FIELD_DEFINITIONS = {
    "rank": (
        f"Pool rank, 1..N, by {POINTS_FIELD} descending (ties broken by the "
        "provider's overall rank). Unique and gap-free."
    ),
    "positional_rank": "Rank within position under the same ordering.",
    "player_id": "Provider player id. The only unique key — names collide.",
    "name": "Player name.",
    "position": "QB, RB, WR or TE.",
    "team": "NFL team abbreviation; 'UNS'/'RK' mean unsigned.",
    "age": "Age in years.",
    "bye_week": "Team bye week; null for unsigned players.",
    "is_rookie": "True for 2026 rookies.",
    POINTS_FIELD: (
        "One-season projected fantasy points under this league's scoring: 0.5/rec, "
        "no TE premium, copied from the provider's half_ppr column."
    ),
    "adp": (
        "1QB ADP as an overall pick number in the source's 12-team draft "
        f"(its round.pick value decoded: 2.03 -> 15). Past pick "
        f"{PLAUSIBLE_ROUNDS * TEAMS_PER_ROUND} the source's tail is noise, i.e. "
        "'effectively undrafted' rather than a real slot."
    ),
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def points_of(record: dict, horizon: str = HORIZON) -> int | None:
    """This league's point total: a copy of the column that prices its reception rate."""
    return record["projections"][horizon].get(POINTS_COLUMN)


def decode_adp(value: float, teams: int = TEAMS_PER_ROUND) -> int:
    """``2.03`` (round 2, pick 3) -> overall pick 15."""
    rnd = int(value)
    return (rnd - 1) * teams + round((value - rnd) * 100)


def adp_of(record: dict) -> int | None:
    value = record["adp"].get(ADP_COLUMN)
    return None if value is None else decode_adp(value)


def provider_rank(record: dict) -> float:
    """The provider's own overall rank, used only to break point ties."""
    rank = record.get("rank_by_3d_value")
    return math.inf if rank is None else rank


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select(records: list[dict], limit: int = RANK_LIMIT) -> tuple[list[dict], dict]:
    """Filter to the relevant pool and order it. Returns (kept records, stats).

    Three cuts, in order: position, then a usable point total, then the rank limit.
    The rank cut has to come last — it is defined on the points column, so it is only
    meaningful once the rows without one are gone.
    """
    dropped_position: collections.Counter[str] = collections.Counter()
    unusable: list[str] = []
    eligible: list[dict] = []

    for record in records:
        position = record.get("position")
        if position not in POSITIONS:
            dropped_position[position or "?"] += 1
            continue
        points = points_of(record)
        if not points or points <= 0:
            unusable.append(record["name"])
            continue
        eligible.append(record)

    eligible.sort(key=lambda r: (-points_of(r), provider_rank(r), r["player_id"]))
    kept, cut = eligible[:limit], eligible[limit:]

    stats = {
        "dropped_position": dict(dropped_position.most_common()),
        "dropped_unusable": sorted(unusable),
        "right_position": len(eligible) + len(unusable),
        "eligible": len(eligible),
        "cut_by_rank": len(cut),
        "cut_at_points": points_of(kept[-1]) if kept else None,
        "best_points_cut": points_of(cut[0]) if cut else None,
    }
    return kept, stats


def build_rows(kept: list[dict]) -> list[dict]:
    """One flat, minimal record per player, in pool order."""
    seen: collections.Counter[str] = collections.Counter()
    rows = []
    for rank, record in enumerate(kept, start=1):
        position = record["position"]
        seen[position] += 1
        team = record.get("team")
        bye = record.get("bye_week")
        rows.append(
            {
                "rank": rank,
                "positional_rank": seen[position],
                "player_id": record["player_id"],
                "name": record["name"],
                "position": position,
                "team": team,
                "age": record.get("age"),
                "bye_week": None if team in NO_TEAM or bye == PLACEHOLDER_BYE else bye,
                "is_rookie": bool(record.get("is_rookie")),
                POINTS_FIELD: points_of(record),
                "adp": adp_of(record),
            }
        )
    return rows


def build_document(
    source: dict, source_path: Path, rows: list[dict], stats: dict
) -> dict:
    """The output file: a short provenance header plus the rows."""
    return {
        "source_file": source_path.name,
        "source_player_count": source.get("player_count", len(source["players"])),
        "scoring_scheme": {
            "name": SCHEME,
            "description": (
                "0.5 points per reception for every position, no tight end premium, "
                "one-QB roster format."
            ),
            "reception_points": {"all_positions": 0.5},
            "points_copied_from": POINTS_COLUMN,
            "adp_copied_from": ADP_COLUMN,
        },
        "horizon": HORIZON,
        "positions": list(POSITIONS),
        "player_count": len(rows),
        "excluded": {
            "by_position": stats["dropped_position"],
            "no_usable_projection": len(stats["dropped_unusable"]),
            "below_rank_limit": stats["cut_by_rank"],
        },
        "fields": FIELD_DEFINITIONS,
        "players": rows,
    }


def check_sources(document: dict) -> None:
    """Fail loudly if a column this reads is gone. Only the ones read are required —
    the rest of ADP_FAMILY is a cross-check, and --report just compares fewer columns."""
    published = set(document.get("scoring_schemes") or [])
    required = {POINTS_COLUMN, ADP_COLUMN}
    missing = sorted(required - published)
    if missing:
        raise ValueError(
            f"source scheme(s) {', '.join(missing)} absent from scoring_schemes — "
            "re-run parse_projections.py"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(source: dict, kept: list[dict], rows: list[dict], stats: dict) -> None:
    out = sys.stderr
    total = len(source["players"])
    dropped = stats["dropped_position"]

    print(f"\ninput: {total} players", file=out)
    print(
        f"  positions {'/'.join(POSITIONS)}: kept {stats['right_position']}, "
        f"dropped {sum(dropped.values())} ("
        + ", ".join(f"{pos} {n}" for pos, n in dropped.items())
        + ")",
        file=out,
    )
    unusable = stats["dropped_unusable"]
    print(
        f"  usable {HORIZON} projection: kept {stats['eligible']}, dropped {len(unusable)}"
        + (f" (0 or missing: {', '.join(unusable[:4])}...)" if unusable else ""),
        file=out,
    )
    print(
        f"  top {len(rows)} by {POINTS_FIELD}: dropped {stats['cut_by_rank']} "
        f"(cut at {stats['cut_at_points']} pts"
        + (f"; best excluded {stats['best_points_cut']})" if stats["cut_by_rank"] else ")"),
        file=out,
    )
    counts = collections.Counter(row["position"] for row in rows)
    print(
        "pool: " + ", ".join(f"{pos} {counts[pos]}" for pos in POSITIONS)
        + f" = {len(rows)}, {sum(row['is_rookie'] for row in rows)} rookies",
        file=out,
    )

    # -- points: is the copied column really this league's scoring? --------
    print(f"\n{POINTS_FIELD}  [all positions -> {POINTS_COLUMN} (0.5/rec)]", file=out)

    def flat(record: dict) -> dict:
        """The record's source cells for this horizon, all schemes."""
        return record["projections"][HORIZON]

    copied = sum(
        1 for row, r in zip(rows, kept) if row[POINTS_FIELD] == flat(r)[POINTS_COLUMN]
    )
    print(f"  emitted == source column: {copied}/{len(rows)} — exact", file=out)
    qbs = [r for r in kept if r["position"] == "QB"]
    reception_blind = sum(
        1 for r in qbs if flat(r)["standard"] == flat(r)["ppr"] == flat(r)[POINTS_COLUMN]
    )
    print(
        f"  QB sanity: standard == half_ppr == ppr for {reception_blind}/{len(qbs)} "
        "(QB points are reception-blind)",
        file=out,
    )

    # -- adp ---------------------------------------------------------------
    print(
        f"\nadp  [{ADP_COLUMN}, round.pick -> overall pick, {TEAMS_PER_ROUND}-team source]",
        file=out,
    )
    family = [s for s in ADP_FAMILY if s in (source.get("scoring_schemes") or [])]
    agree = sum(1 for r in kept if len({r["adp"].get(s) for s in family}) == 1)
    print(
        f"  all {len(family)} 1QB styles identical for {agree}/{len(kept)} — "
        "the source ADP responds to roster format only",
        file=out,
    )
    picks = [row["adp"] for row in rows if row["adp"] is not None]
    print(
        f"  present {len(picks)}/{len(rows)}, distinct {len(set(picks))}"
        + ("" if len(set(picks)) == len(picks) else "  <- COLLISIONS"),
        file=out,
    )
    roundtrip = sum(
        1
        for r in kept
        if r["adp"].get(ADP_COLUMN) is not None
        and decode_adp(r["adp"][ADP_COLUMN]) == adp_of(r)
    )
    tail = [pick for pick in picks if pick > PLAUSIBLE_ROUNDS * TEAMS_PER_ROUND]
    print(
        f"  decode round-trips for {roundtrip}/{len(kept)}; range "
        f"{min(picks, default=0)}..{max(picks, default=0)}, {len(tail)} past round "
        f"{PLAUSIBLE_ROUNDS} (provider tail noise, not a real slot)",
        file=out,
    )

    # -- integrity ---------------------------------------------------------
    print("\nintegrity", file=out)
    ranks = [row["rank"] for row in rows]
    ids = {row["player_id"] for row in rows}
    ordered = all(
        rows[i][POINTS_FIELD] >= rows[i + 1][POINTS_FIELD] for i in range(len(rows) - 1)
    )
    per_pos = collections.Counter()
    positional_ok = True
    for row in rows:
        per_pos[row["position"]] += 1
        positional_ok &= row["positional_rank"] == per_pos[row["position"]]
    print(
        f"  rank 1..{len(rows)} gap-free: {ranks == list(range(1, len(rows) + 1))}; "
        f"unique player_ids: {len(ids) == len(rows)}; "
        f"monotone in {POINTS_FIELD}: {ordered}; positional ranks consistent: {positional_ok}",
        file=out,
    )
    nulled = sum(1 for row in rows if row["bye_week"] is None)
    print(
        f"  bye_week nulled for {nulled} unsigned players (source sentinel "
        f"{PLACEHOLDER_BYE}); age/team present for all {len(rows)}",
        file=out,
    )
    print(f"  fields per player: {len(rows[0])} ({', '.join(rows[0])})", file=out)

    # -- spot check --------------------------------------------------------
    print("\ntop 5 and the last 2 in the pool", file=out)
    for row in rows[:5] + rows[-2:]:
        print(
            f"  {row['rank']:>3} {row['name']:<22} {row['position']}{row['positional_rank']:<3} "
            f"{row[POINTS_FIELD]:>5} pts   adp {row['adp']}",
            file=out,
        )
    print(file=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", nargs="?", default=paths.PROJECTIONS_JSON, type=Path)
    ap.add_argument("-o", "--output", default=paths.POOL, type=Path)
    ap.add_argument(
        "--limit", type=int, default=RANK_LIMIT, help=f"keep this many players (default {RANK_LIMIT})"
    )
    ap.add_argument("--report", action="store_true", help="print a validation summary to stderr")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent; 0 for compact")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 1
    with args.input.open(encoding="utf-8") as handle:
        source = json.load(handle)
    if not source.get("players"):
        print(f"error: no players in {args.input}", file=sys.stderr)
        return 1

    try:
        check_sources(source)
        kept, stats = select(source["players"], args.limit)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not kept:
        print("error: no players survived the filters", file=sys.stderr)
        return 1

    rows = build_rows(kept)
    document = build_document(source, args.input, rows, stats)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=args.indent or None, ensure_ascii=False)
        handle.write("\n")

    print(
        f"pool: {len(rows)} of {len(source['players'])} players, "
        f"{POINTS_FIELD} + adp -> {paths.display(args.output)}",
        file=sys.stderr,
    )
    if args.report:
        report(source, kept, rows, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
