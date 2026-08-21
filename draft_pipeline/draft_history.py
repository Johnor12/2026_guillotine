#!/usr/bin/env python3
"""Parse a hand-pasted ESPN draft-room pick history into made picks.

ESPN's league read API does not surface live-draft picks while the draft runs (a pick
made in the room stayed invisible to `mDraftDetail` for minutes in testing, and may
only sync when the draft completes), so on draft day the pick history is copied out of
the draft room by hand into ``draft_history.txt`` at the repo root. This module turns
that paste into the made-pick shapes ``fetch_draft.py`` builds from the API, and
``fetch_draft.py`` overlays them on the board it already fetched — ESPN-reported picks
win wherever both exist.

The paste is the draft room's pick history copied as text. Each round starts with a
``Round N`` line, and each pick contributes, in order: its overall pick number (the
displayed Pick column counts across rounds; the ``Round N`` header and league size
only cross-check it), the player's name, an
injury designation (``Q``, ``O``, ...) if the player carries one, NFL team, and
position, then a tail of columns (fantasy team, points, rank) this parser never
reads. Records are anchored on the position line — the NFL team sits directly above
it, then an optional status token, then name and pick number — so junk in the tail
cannot break the parse, and a malformed record raises instead of being skipped: it
is a mis-paste to fix.

Players are matched to the pool pipeline's cached Sleeper dump by normalized name,
position always required to agree and team used as a tiebreaker — a trimmed copy of
``pool_pipeline/match_sleeper.py``'s tiers 1 and 2 (the pipelines deliberately share
no code, and the paste has no age or alternate spelling for the last-name tier to
lean on). D/ST picks skip name matching: Sleeper keys team defenses by abbreviation.

Usage:
    uv run draft_pipeline/draft_history.py --teams 4      # parse + match draft_history.txt
    uv run draft_pipeline/draft_history.py --teams 4 some_paste.txt
    uv run draft_pipeline/draft_history.py --selftest
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import paths

#: The position tokens ESPN prints in the history; each one anchors a pick record.
POSITIONS = {"QB", "RB", "WR", "TE", "K", "D/ST"}

#: ESPN NFL abbreviation -> Sleeper's, where they differ.
TEAM_ALIASES = {"WSH": "WAS", "JAC": "JAX", "LVR": "LV"}

#: ESPN injury designations, pasted as their own line between name and NFL team.
STATUSES = {"Q", "D", "O", "IR", "PUP", "SSPD", "NA"}

SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")

ROUND_RE = re.compile(r"^Round (\d+)$")
TEAM_RE = re.compile(r"^[A-Z]{2,3}$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse(text: str, teams: int) -> list[dict]:
    """Pick records from the pasted history: {pick_no, name, team, position}."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[dict] = []
    round_no = None

    for i, line in enumerate(lines):
        header = ROUND_RE.match(line)
        if header:
            round_no = int(header.group(1))
            continue
        if line not in POSITIONS:
            continue
        context = " / ".join(lines[max(0, i - 4) : i + 1])
        if round_no is None:
            raise ValueError(f"pick before any 'Round N' line: {context}")
        j = i - 2  # line above the team: an injury status, or already the name
        if j >= 0 and lines[j] in STATUSES:
            j -= 1
        if j < 1:
            raise ValueError(f"position line with no pick above it: {context}")
        number, name, team = lines[j - 1], lines[j], lines[i - 1]
        if not number.isdigit() or int(number) < 1:
            raise ValueError(f"expected an overall pick number, got {number!r}: {context}")
        pick_no = int(number)
        if (pick_no - 1) // teams + 1 != round_no:
            raise ValueError(
                f"pick {pick_no} is not in round {round_no} of a {teams}-team league: {context}"
            )
        if not TEAM_RE.match(team):
            raise ValueError(f"expected an NFL team abbreviation, got {team!r}: {context}")
        records.append({"pick_no": pick_no, "name": name, "team": team, "position": line})

    if not records:
        raise ValueError("no picks found — is this the draft room's pick history?")
    duplicates = [
        n for n, count in collections.Counter(r["pick_no"] for r in records).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"pick number(s) {duplicates[:5]} appear more than once — double paste?")
    return sorted(records, key=lambda record: record["pick_no"])


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def strip_suffix(normalized: str) -> str:
    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            return normalized[: -len(suffix)]
    return normalized


def positions_of(player: dict) -> set[str]:
    listed = player.get("fantasy_positions") or []
    return {p for p in [player.get("position"), *listed] if p}


def index_players(players: dict) -> tuple[dict, dict]:
    """(normalized name -> players, suffix-stripped name -> players)."""
    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    by_base: dict[str, list[dict]] = collections.defaultdict(list)
    for player in players.values():
        if not isinstance(player, dict) or not player.get("position"):
            continue
        name = player.get("search_full_name") or norm(
            player.get("full_name")
            or f"{player.get('first_name', '')} {player.get('last_name', '')}"
        )
        if name:
            by_name[name].append(player)
            by_base[strip_suffix(name)].append(player)
    return by_name, by_base


def narrow(record: dict, candidates: list[dict]) -> list[dict]:
    """Position must agree; team and active status break ties, never disqualify."""
    survivors = [p for p in candidates if record["position"] in positions_of(p)]
    if len(survivors) <= 1:
        return survivors
    team = TEAM_ALIASES.get(record["team"], record["team"])
    on_team = [p for p in survivors if p.get("team") == team]
    if on_team:
        survivors = on_team
    if len(survivors) <= 1:
        return survivors
    active = [p for p in survivors if p.get("active")]
    return active or survivors


def match_player(record: dict, by_name: dict, by_base: dict, players: dict) -> dict | None:
    if record["position"] == "D/ST":
        return players.get(TEAM_ALIASES.get(record["team"], record["team"]))
    name = norm(record["name"])
    for candidates in (by_name.get(name, []), by_base.get(strip_suffix(name), [])):
        survivors = narrow(record, candidates)
        if len(survivors) == 1:
            return survivors[0]
        if survivors:
            return None  # ambiguous — left unmatched rather than guessed
    return None


def as_picks(records: list[dict], players: dict) -> tuple[list[dict], list[str]]:
    """The made-pick shapes ``adapt()`` builds, plus one warning per unmatched player.

    No ``roster_id`` here — ``fetch_draft.py`` attaches ownership from ESPN's
    laid-out board, and the derived owner covers any pick the layout misses.
    """
    by_name, by_base = index_players(players)
    picks: list[dict] = []
    warnings: list[str] = []
    for record in records:
        player = match_player(record, by_name, by_base, players)
        if player is None:
            warnings.append(
                f"pick {record['pick_no']} {record['name']} ({record['position']} "
                f"{record['team']}) has no unambiguous Sleeper match — sleeper_id stays null"
            )
        first, _, last = record["name"].partition(" ")
        picks.append(
            {
                "pick_no": record["pick_no"],
                "player_id": player["player_id"] if player else None,
                "is_keeper": False,
                "metadata": {
                    "first_name": player.get("first_name") if player else first,
                    "last_name": player.get("last_name") if player else (last or None),
                    "position": player.get("position") if player else record["position"],
                    "team": player.get("team") if player else record["team"],
                },
            }
        )
    return picks, warnings


# ---------------------------------------------------------------------------
# Selftest — the paste format, locked to a real draft-room copy
# ---------------------------------------------------------------------------

SAMPLE = """\
Round 1
Pick
Player
Team
2025 PTS
PROJ PTS
RK
1

Jahmyr Gibbs
DET
RB
John's Supreme Team3
366.9
364.9
1
2

Bijan Robinson
ATL
RB
John's Stout Team4
370.8
352.8
2
3

Puka Nacua
Q
LAR
WR
John's Scary Team
375
356.3
4
Round 2
Pick
Player
Team
2025 PTS
PROJ PTS
RK
5

Jaxon Smith-Njigba
SEA
WR
John's Stout Team4
359.9
326.7
5
"""


def selftest() -> int:
    records = parse(SAMPLE, teams=4)
    expected = [
        {"pick_no": 1, "name": "Jahmyr Gibbs", "team": "DET", "position": "RB"},
        {"pick_no": 2, "name": "Bijan Robinson", "team": "ATL", "position": "RB"},
        {"pick_no": 3, "name": "Puka Nacua", "team": "LAR", "position": "WR"},
        {"pick_no": 5, "name": "Jaxon Smith-Njigba", "team": "SEA", "position": "WR"},
    ]
    assert records == expected, records

    dump = {
        "1": {
            "player_id": "1", "first_name": "Jahmyr", "last_name": "Gibbs",
            "full_name": "Jahmyr Gibbs", "position": "RB", "team": "DET", "active": True,
        },
        "DEN": {
            "player_id": "DEN", "first_name": "Denver", "last_name": "Broncos",
            "position": "DEF", "team": "DEN",
        },
    }
    dst = {"pick_no": 4, "name": "Broncos D/ST", "team": "DEN", "position": "D/ST"}
    picks, warnings = as_picks(records + [dst], dump)
    assert [p["player_id"] for p in picks] == ["1", None, None, None, "DEN"], picks
    assert picks[4]["metadata"]["position"] == "DEF", picks[4]
    assert len(warnings) == 3 and "Bijan Robinson" in warnings[0], warnings

    for bad in (
        "",
        "Round 1\n1\nJahmyr Gibbs",
        SAMPLE.replace("Round 1", ""),
        SAMPLE.replace("Round 2", "Round 3"),  # pick 5 can't be in round 3
    ):
        try:
            parse(bad, teams=4)
        except ValueError:
            continue
        raise AssertionError(f"parse accepted {bad[:40]!r}")

    print("draft_history selftest ok", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI — check a paste by eye before a refresh consumes it
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", nargs="?", default=paths.DRAFT_HISTORY, type=Path)
    ap.add_argument(
        "--teams", type=int, help="league size, to cross-check pick numbers against round headers"
    )
    ap.add_argument("--selftest", action="store_true", help="check the parser offline, then exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.teams:
        ap.error("--teams is required (to check pick numbers against their Round headers)")
    if not args.input.is_file():
        print(f"error: {paths.display(args.input)} not found", file=sys.stderr)
        return 1

    try:
        records = parse(args.input.read_text(encoding="utf-8"), args.teams)
    except ValueError as exc:
        print(f"error: {paths.display(args.input)}: {exc}", file=sys.stderr)
        return 1
    with paths.SLEEPER_PLAYERS.open(encoding="utf-8") as handle:
        players = json.load(handle)

    picks, warnings = as_picks(records, players)
    for record, pick in zip(records, picks):
        meta = pick["metadata"]
        matched = (
            f"sleeper {pick['player_id']}  {meta['first_name']} {meta['last_name']} "
            f"{meta['position']} {meta['team']}"
            if pick["player_id"]
            else "UNMATCHED"
        )
        print(
            f"{pick['pick_no']:>3}  {record['name']:<26} {record['position']:<5} "
            f"{record['team']:<3} -> {matched}"
        )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"{len(picks)} picks, {len(picks) - len(warnings)} matched", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
