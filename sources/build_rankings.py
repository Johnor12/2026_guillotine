#!/usr/bin/env python3
"""Normalize the provider snapshots into one boards file, data/boards.json.

    uv run sources/build_rankings.py
    uv run sources/build_rankings.py --report

Eight boards are parsed from the raw snapshots fetch_rankings.py saved: FantasyCalc,
KeepTradeCut, FantasyFootballCalculator ADP and FantasyPros ECR in their ordinary 1QB
form, and the superflex/2QB variant of each (KeepTradeCut's sits in the same page).
Six more are derived: DraftSharks 1QB ADP from pool.json; Sleeper half-PPR ADP,
Sleeper 2QB ADP and the league-scored projected-points order (the one board that
prices the TE premium) from the pool's Sleeper projections snapshot; a consensus
board averaging each pool player's rank across every other source; and the cold-start
room prior the ranker gives an opponent who has not picked yet: Sleeper's half-PPR
ADP, the list this league's draft room displays and autopick drafts from, blended
30% toward the mean of the format-adjusted boards for the savvy minority. The
consensus board completes each provider's uncovered tail in the ranker.

Every row is resolved onto the pool's Sleeper id (identity.PlayerResolver): by supplied
id, then normalized name and position, then a conservative last-name/team/first-name-
prefix tier. A row the pool does not carry keeps sleeper_id null; the ranker ignores
it and the investigator falls back to name matching for it.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

from identity import PlayerResolver, normalized_name

PROCESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROCESS_DIR.parent
RAW = PROCESS_DIR / "data" / "raw"
BOARDS = PROCESS_DIR / "data" / "boards.json"
POOL = REPO_ROOT / "pool.json"
SLEEPER_PROJECTIONS = REPO_ROOT / "pool" / "data" / "sleeper_projections.json"

POSITIONS = {"QB", "RB", "WR", "TE"}


# --- provider parsers -------------------------------------------------------------


def js_json(text: str, marker: str):
    """The JSON literal assigned right after ``marker`` in a page's inline script."""
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing JavaScript marker {marker!r}")
    start += len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def player(rank, name, position, team, value, sleeper_id=None) -> dict:
    return {
        "rank": int(rank),
        "name": name.strip(),
        "position": position.strip().upper(),
        "team": team.strip().upper() if team and team.strip() else None,
        "sleeper_id": str(sleeper_id) if sleeper_id not in (None, "") else None,
        "value": value,
    }


def parse_fantasycalc(path: Path) -> list[dict]:
    return [
        player(
            row["overallRank"],
            row["player"]["name"],
            row["player"]["position"],
            row["player"].get("maybeTeam"),
            row.get("value"),
            row["player"].get("sleeperId"),
        )
        for row in json.loads(path.read_text())
        if row["player"]["position"] in POSITIONS
    ]


def parse_keeptradecut(path: Path, value_set: str = "oneQBValues") -> list[dict]:
    # KTC's fantasy page carries its 1QB and superflex value sets in one payload (there
    # is no TE-premium redraft variant); each set is parsed as its own board.
    ranked = []
    for raw in js_json(path.read_text(), "var playersArray ="):
        if raw["position"] not in POSITIONS:
            continue
        values = raw[value_set]
        ranked.append(
            player(values["rank"], raw["playerName"], raw["position"], raw.get("team"), values.get("value"))
        )
    # KTC has emitted duplicate rank numbers before; value is what drives its board.
    ranked.sort(key=lambda row: (-row["value"], row["rank"], row["name"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def parse_ffcalculator(path: Path) -> list[dict]:
    rows = [
        raw
        for raw in json.loads(path.read_text()).get("players", [])
        if raw.get("position") in POSITIONS
    ]
    rows.sort(key=lambda raw: (raw["adp"], raw["name"]))
    return [
        player(rank, raw["name"], raw["position"], raw.get("team"), raw["adp"])
        for rank, raw in enumerate(rows, start=1)
    ]


def parse_fantasypros(path: Path) -> list[dict]:
    return [
        player(
            raw["rank_ecr"],
            raw["player_name"],
            raw["player_position_id"],
            raw.get("player_team_id"),
            float(raw["rank_ave"]),
        )
        for raw in js_json(path.read_text(), "var ecrData =")["players"]
        if raw["player_position_id"] in POSITIONS
    ]


PARSERS = {
    "fantasycalc": ("FantasyCalc", "fantasycalc.json", parse_fantasycalc),
    "fantasycalc_sf": ("FantasyCalc SF", "fantasycalc_sf.json", parse_fantasycalc),
    "keeptradecut": ("KeepTradeCut", "keeptradecut.html", parse_keeptradecut),
    "keeptradecut_sf": (
        "KeepTradeCut SF",
        "keeptradecut.html",
        lambda path: parse_keeptradecut(path, "superflexValues"),
    ),
    "ffcalculator": ("FF Calculator ADP", "ffcalculator.json", parse_ffcalculator),
    "ffcalculator_2qb": ("FF Calculator 2QB ADP", "ffcalculator_2qb.json", parse_ffcalculator),
    "fantasypros": ("FantasyPros ECR", "fantasypros.html", parse_fantasypros),
    "fantasypros_sf": ("FantasyPros SF ECR", "fantasypros_sf.html", parse_fantasypros),
}


# --- derived boards ---------------------------------------------------------------


def draftsharks_adp(pool: dict) -> list[dict]:
    # Ties (none in the current pool) break on pool order, i.e. Sleeper season points.
    ranked = sorted(
        (row for row in pool["players"] if row.get("adp") is not None),
        key=lambda row: row["adp"],
    )
    return [
        player(rank, raw["name"], raw["position"], raw.get("team"), raw["adp"], raw["sleeper_id"])
        for rank, raw in enumerate(ranked, start=1)
    ]


def sleeper_adp(projections: dict, field: str = "adp") -> list[dict]:
    """Sleeper's redraft ADP order: ``adp`` is the half-PPR list the draft room shows,
    ``adp_2qb`` the 2QB-league list."""
    rows = [
        raw
        for raw in projections["players"]
        if raw["position"] in POSITIONS
        and raw.get(field) is not None
        and raw[field] < 999.0  # Sleeper's "undrafted" sentinel
    ]
    rows.sort(key=lambda raw: (raw[field], raw["name"]))
    return [
        player(rank, raw["name"], raw["position"], raw.get("team"), raw[field], raw["sleeper_id"])
        for rank, raw in enumerate(rows, start=1)
    ]


def sleeper_points(projections: dict) -> list[dict]:
    """League-scored projected points, the sort Sleeper's own league players page shows
    every manager. The one board that prices the +1.0/rec TE premium."""
    rows = [raw for raw in projections["players"] if raw["position"] in POSITIONS]
    rows.sort(key=lambda raw: (-raw["points"], raw.get("adp") or 999.0, raw["name"]))
    return [
        player(rank, raw["name"], raw["position"], raw.get("team"), raw["points"], raw["sleeper_id"])
        for rank, raw in enumerate(rows, start=1)
    ]


def consensus(sources: list[dict], pool: dict) -> list[dict]:
    """Mean provider rank per pool player over the sources that rank him. Every pool
    player carries a DraftSharks ADP, so the board covers the whole pool."""
    ranks: dict[str, list[int]] = {}
    for source in sources:
        for row in source["players"]:
            if row["sleeper_id"] is not None:
                ranks.setdefault(row["sleeper_id"], []).append(row["rank"])
    by_id = {str(row["sleeper_id"]): row for row in pool["players"]}
    averaged = sorted(
        ((statistics.fmean(found), sleeper_id) for sleeper_id, found in ranks.items()),
        key=lambda item: (item[0], by_id[item[1]]["name"]),
    )
    return [
        player(
            rank,
            by_id[sid]["name"],
            by_id[sid]["position"],
            by_id[sid].get("team"),
            round(mean, 2),
            sid,
        )
        for rank, (mean, sid) in enumerate(averaged, start=1)
    ]


# The format-adjusted boards a savvy drafter plausibly finds or approximates: the
# superflex/2QB variants plus the league-scored points list (the one TE-premium board).
SAVVY_SOURCE_IDS = (
    "fantasycalc_sf",
    "keeptradecut_sf",
    "ffcalculator_2qb",
    "fantasypros_sf",
    "sleeper_2qb",
    "sleeper_points",
)
# Most of this room drafts off the list Sleeper displays (a league mock showed the
# draft room's numbers matching adp_half_ppr to the decimal, and autopick follows the
# same list); roughly a 30% minority is expected to find real boards or adjust.
SAVVY_WEIGHT = 0.3


def cold_start_blend(sources: list[dict], pool: dict) -> list[dict]:
    """Rank-space blend, (1 - w) * Sleeper half-PPR ADP + w * mean savvy-board rank. A
    player only one side ranks takes that rank alone; a player neither ranks is left
    to the ranker's consensus tail completion."""
    by_id = {source["id"]: source for source in sources}
    default_ranks = {
        row["sleeper_id"]: row["rank"]
        for row in by_id["sleeper_adp"]["players"]
        if row["sleeper_id"] is not None
    }
    savvy_ranks: dict[str, list[int]] = {}
    for source_id in SAVVY_SOURCE_IDS:
        for row in by_id[source_id]["players"]:
            if row["sleeper_id"] is not None:
                savvy_ranks.setdefault(row["sleeper_id"], []).append(row["rank"])
    pool_by_id = {str(row["sleeper_id"]): row for row in pool["players"]}
    scored = []
    for sleeper_id in pool_by_id:
        default = default_ranks.get(sleeper_id)
        savvy = statistics.fmean(savvy_ranks[sleeper_id]) if sleeper_id in savvy_ranks else None
        if default is None and savvy is None:
            continue
        if default is None or savvy is None:
            score = default if savvy is None else savvy
        else:
            score = (1 - SAVVY_WEIGHT) * default + SAVVY_WEIGHT * savvy
        scored.append((score, sleeper_id))
    scored.sort(key=lambda item: (item[0], pool_by_id[item[1]]["name"]))
    return [
        player(
            rank,
            pool_by_id[sid]["name"],
            pool_by_id[sid]["position"],
            pool_by_id[sid].get("team"),
            round(score, 2),
            sid,
        )
        for rank, (score, sid) in enumerate(scored, start=1)
    ]


# --- validation -------------------------------------------------------------------


def drop_ambiguous_identities(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Split off every row of a same-name-same-position collision. FantasyPros' 2026
    ECR lists two distinct WRs named Isaiah Williams; a name-keyed identity cannot tell
    them apart, so neither is joinable and dropping the pair beats failing the source."""
    counts: collections.Counter = collections.Counter(
        (normalized_name(row["name"], drop_suffix=True), row["position"]) for row in rows
    )
    kept, dropped = [], []
    for row in rows:
        ident = (normalized_name(row["name"], drop_suffix=True), row["position"])
        (kept if counts[ident] == 1 else dropped).append(row)
    return kept, sorted(f"{row['name']} ({row['position']}, rank {row['rank']})" for row in dropped)


def validate(source_id: str, rows: list[dict]) -> list[str]:
    problems = []
    if len(rows) < 50:
        problems.append(f"{source_id}: only {len(rows)} players (expected at least 50)")
    ranks = [row["rank"] for row in rows]
    if any(rank < 1 for rank in ranks):
        problems.append(f"{source_id}: ranks must be positive")
    if len(ranks) != len(set(ranks)):
        problems.append(f"{source_id}: duplicate ranks")
    bad_positions = sorted({row["position"] for row in rows} - POSITIONS)
    if bad_positions:
        problems.append(f"{source_id}: unsupported positions {bad_positions}")
    return problems


# --- CLI --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    pool = json.loads(POOL.read_text())
    resolver = PlayerResolver(pool)
    meta_path = RAW / "fetch_meta.json"
    fetch_meta = json.loads(meta_path.read_text()).get("sources", {}) if meta_path.exists() else {}

    def resolved(rows: list[dict]) -> list[dict]:
        for row in rows:
            row["sleeper_id"] = resolver.resolve(row)
        return rows

    def source(source_id: str, name: str, fmt: str | None, fetched_at, rows: list[dict]) -> dict:
        return {
            "id": source_id,
            "name": name,
            "format": fmt,
            "fetched_at": fetched_at,
            "player_count": len(rows),
            "matched_to_sleeper": sum(row["sleeper_id"] is not None for row in rows),
            "players": rows,
        }

    sources: list[dict] = []
    failures: list[str] = []
    for source_id, (name, filename, parser) in PARSERS.items():
        path = RAW / filename
        if not path.exists():
            failures.append(f"{source_id}: missing {path}; run fetch_rankings.py")
            continue
        try:
            rows = parser(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{source_id}: {exc}")
            continue
        rows.sort(key=lambda row: row["rank"])
        rows, ambiguous = drop_ambiguous_identities(rows)
        if ambiguous:
            print(f"{source_id}: dropped unjoinable duplicate names: {', '.join(ambiguous)}", file=sys.stderr)
        problems = validate(source_id, rows)
        if problems:
            failures.extend(problems)
            continue
        meta = fetch_meta.get(source_id, {})
        sources.append(source(source_id, name, meta.get("format"), meta.get("fetched_at"), resolved(rows)))

    sources.append(
        source("draftsharks_adp", "DraftSharks ADP", "1QB half-PPR ADP from pool.json", None, resolved(draftsharks_adp(pool)))
    )
    projections = json.loads(SLEEPER_PROJECTIONS.read_text())
    for source_id, name, fmt, rows in (
        (
            "sleeper_adp",
            "Sleeper ADP",
            "half-PPR redraft ADP from the pool's Sleeper projections snapshot",
            sleeper_adp(projections),
        ),
        (
            "sleeper_2qb",
            "Sleeper 2QB ADP",
            "2QB redraft ADP from the same snapshot",
            sleeper_adp(projections, "adp_2qb"),
        ),
        (
            "sleeper_points",
            "Sleeper League Points",
            "league-scored projected points (0.5 PPR, +1.0/rec TE premium): the sort "
            "Sleeper's league players page shows",
            sleeper_points(projections),
        ),
    ):
        sources.append(source(source_id, name, fmt, projections.get("fetched_at"), resolved(rows)))
    if failures:
        print("ranking normalization failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    # Built last so the average spans every other source.
    sources.append(
        source(
            "consensus",
            "Consensus Average",
            f"mean provider rank across the other {len(sources)} sources",
            None,
            consensus(sources, pool),
        )
    )
    # After consensus, so the blend never feeds back into the average.
    sources.append(
        source(
            "cold_start",
            "Cold-Start Room Prior",
            f"{1 - SAVVY_WEIGHT:.0%} Sleeper half-PPR ADP + {SAVVY_WEIGHT:.0%} mean rank of "
            "the format-adjusted boards",
            None,
            cold_start_blend(sources, pool),
        )
    )
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source_count": len(sources),
        "sources": sources,
    }
    BOARDS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.report:
        for src in sources:
            print(
                f"  {src['id']}: {src['player_count']} players, {src['matched_to_sleeper']} matched to Sleeper",
                file=sys.stderr,
            )
    print(f"normalized {len(sources)} sources -> {BOARDS.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
