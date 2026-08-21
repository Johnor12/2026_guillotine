#!/usr/bin/env python3
"""Fetch the ESPN league's live draft and publish the complete made-and-pending board.

This is a network-only, on-demand pipeline. Board geometry and the output contract live
in `draft_board.py`; this entry point fetches ESPN's undocumented v3 league API and
adapts its response into the shapes `draft_board.py` consumes, so `draft.json` keeps
the exact contract the ranker already reads.

One request does everything: the league endpoint with the `mDraftDetail` (picks),
`mSettings` (draft order and roster shape) and `mTeam` (teams and members) views.
ESPN identifies players by its own ids; picks are joined back to `pool.json`'s
`sleeper_id` through the Sleeper player dump the pool pipeline already caches, which
carries `espn_id` for every fantasy-relevant player. D/ST picks use ESPN's fixed
`-16000 - proTeamId` id form and Sleeper's team-abbreviation DEF ids.

The read API lags the live draft (a made pick stays `playerId` -1 for minutes, maybe
until the draft completes), so during a draft the room's pick history is hand-pasted
into `draft_history.txt` at the repo root; when that file has content, its picks are
parsed by `draft_history.py` and overlaid on the fetched board, ESPN-reported picks
winning.

Assumptions, stated instead of defended: the league drafts a plain snake (ESPN has no
third-round reversal), picks are never traded (ESPN live drafts do not support it, so
`traded_picks` is always empty), and the auth cookies below are current — ESPN answers
a private league with cookies from any logged-in browser session.

Usage:
    uv run draft_pipeline/fetch_draft.py
    uv run draft_pipeline/fetch_draft.py --report
    uv run draft_pipeline/fetch_draft.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import draft_history
import paths
import report as draft_report
from draft_board import Board, build_document, index_users, pick_number_problems, pick_rows
from selftest import selftest as run_selftest

LEAGUE_ID = "1900421374"
SEASON = 2026

# My ESPN account. SWID doubles as my member id; both cookies come from a logged-in
# browser session and expire on ESPN's schedule — refresh them here when requests 401.
MY_SWID = "{98AC871B-86C0-43C5-B552-2A87B01A415C}"
ESPN_S2 = (
    "AEBd9cRJUO0X%2BPMM1kvvMPiS4aOR5Wgc0Ykbg5XMCffgidDvLzwZ6InpnOXCmd%2Fzo8G%2F2noSPG1"
    "uOOBpdVeZV%2BDK02PkomPiVgkUfY9%2FJTxcdvcZGz5P2wp0%2Fy3ynLT0L61JI0mgkNnlgwRXH45pSu"
    "vgWbcZ3ouz7XRjzPzM%2F%2F3P5NPUNm7A6o0HSli8ZEj1leS%2B7z3vfrGmyYKZ2hdpj%2B%2FMRFs%2"
    "Fd2GHYKXAkKK613DOK17T2MTrVc648mofeGj1dsBoWwijU5VvCU8GK%2FmG0NH0LEixgu5jB8darzUg2r"
    "WaCg%3D%3D"
)

API = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
VIEWS = ("mDraftDetail", "mSettings", "mTeam")
TIMEOUT_SECONDS = 30

#: ESPN's IR lineup slot — the one roster slot a live draft does not fill.
IR_SLOT = "21"

#: ESPN proTeamId -> the abbreviation Sleeper uses as its DEF player id.
ESPN_TEAM_ABBREV = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN",
    17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}


def league_url(league_id: str, season: int) -> str:
    views = "&".join(f"view={view}" for view in VIEWS)
    return f"{API}/seasons/{season}/segments/0/leagues/{league_id}?{views}"


def fetch(league_id: str, season: int, timeout: int = TIMEOUT_SECONDS) -> dict:
    request = urllib.request.Request(
        league_url(league_id, season),
        headers={
            "Accept": "application/json",
            "Cookie": f"espn_s2={ESPN_S2}; SWID={MY_SWID}",
            # ESPN's edge rejects urllib's default agent.
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        league = json.loads(response.read())
    if not isinstance(league, dict) or str(league.get("id")) != str(league_id):
        raise ValueError(f"response is not league {league_id} — got {str(league)[:200]}")
    return league


# ---------------------------------------------------------------------------
# ESPN -> the Sleeper-shaped inputs draft_board.py consumes
# ---------------------------------------------------------------------------


def sleeper_index(players_path: Path) -> tuple[dict, dict[str, dict]]:
    """(espn_id -> sleeper player, the full sleeper dump keyed by sleeper id)."""
    with players_path.open(encoding="utf-8") as handle:
        players = json.load(handle)
    by_espn = {
        str(player["espn_id"]): player
        for player in players.values()
        if isinstance(player, dict) and player.get("espn_id")
    }
    return by_espn, players


def sleeper_player_for(espn_player_id: int, by_espn: dict, players: dict) -> dict | None:
    if str(espn_player_id) in by_espn:
        return by_espn[str(espn_player_id)]
    if espn_player_id <= -16001:  # D/ST: -16000 - proTeamId
        abbrev = ESPN_TEAM_ABBREV.get(-(espn_player_id + 16000))
        return players.get(abbrev) if abbrev else None
    return None


def draft_status(draft_detail: dict) -> str:
    if draft_detail.get("drafted"):
        return "complete"
    if draft_detail.get("inProgress"):
        return "drafting"
    return "pre_draft"


def adapt(league: dict, by_espn: dict, players: dict) -> dict:
    """The fetched-shapes dict fetch_draft historically built from Sleeper's four calls."""
    settings = league.get("settings") or {}
    draft_settings = settings.get("draftSettings") or {}
    draft_detail = league.get("draftDetail") or {}
    teams = league.get("teams") or []
    members = league.get("members") or []

    slot_counts = (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    rounds = sum(int(n) for slot, n in slot_counts.items() if slot != IR_SLOT)

    # pickOrder is the round-1 team order; a team's primaryOwner stands in for
    # Sleeper's user id, so member SWIDs flow through user_id everywhere.
    pick_order = [int(team_id) for team_id in (draft_settings.get("pickOrder") or [])]
    primary_owner = {int(team["id"]): team.get("primaryOwner") for team in teams}
    slot_to_roster = {slot: team_id for slot, team_id in enumerate(pick_order, start=1)}
    draft_order = {
        primary_owner[team_id]: slot
        for slot, team_id in slot_to_roster.items()
        if primary_owner.get(team_id)
    }

    draft = {
        "draft_id": str(league["id"]),
        "league_id": str(league["id"]),
        "metadata": {
            "name": settings.get("name"),
            "scoring_type": (settings.get("scoringSettings") or {}).get("scoringType"),
        },
        "season": str(league.get("seasonId") or SEASON),
        "status": draft_status(draft_detail),
        "start_time": draft_settings.get("date"),
        "last_picked": None,  # ESPN does not report a last-pick timestamp
        "type": str(draft_settings.get("type") or "").lower() or "snake",
        "settings": {
            "teams": int(settings.get("size") or len(teams)),
            "rounds": rounds,
            "reversal_round": 0,
        },
        "draft_order": draft_order,
        "slot_to_roster_id": slot_to_roster,
    }

    unmatched: list[int] = []
    picks = []
    for pick in draft_detail.get("picks") or []:
        espn_player_id = int(pick["playerId"])
        # ESPN lays out the whole board up front; an unmade pick is playerId -1.
        # Real ids are positive, except D/ST at -16000 - proTeamId.
        if -16000 < espn_player_id <= 0:
            continue
        player = sleeper_player_for(espn_player_id, by_espn, players)
        if player is None:
            unmatched.append(espn_player_id)
        picks.append(
            {
                "pick_no": int(pick["overallPickNumber"]),
                "roster_id": int(pick["teamId"]),
                "player_id": player["player_id"] if player else None,
                "is_keeper": bool(pick.get("keeper")),
                "metadata": {
                    "first_name": player.get("first_name") if player else "espn",
                    "last_name": player.get("last_name") if player else str(espn_player_id),
                    "position": player.get("position") if player else None,
                    "team": player.get("team") if player else None,
                },
            }
        )

    def team_name(team: dict) -> str | None:
        name = (team.get("name") or "").strip()
        located = f"{team.get('location') or ''} {team.get('nickname') or ''}".strip()
        return name or located or None

    owner_team = {
        owner: team for team in teams for owner in (team.get("owners") or [])
    }
    users = [
        {
            "user_id": member["id"],
            "display_name": member.get("displayName"),
            "metadata": {
                "team_name": team_name(owner_team.get(member["id"], {})),
            },
        }
        for member in members
        if member.get("id")
    ]

    warning = None
    if unmatched:
        warning = (
            f"{len(unmatched)} pick(s) have no Sleeper match for ESPN id(s) "
            f"{unmatched[:5]} — they keep sleeper_id null and join nothing in the pool"
        )
    return {"draft": draft, "picks": picks, "traded": [], "users": users, "warning": warning}


def overlay_history(text: str, fetched: dict, league: dict, board: Board, players: dict) -> str:
    """Merge hand-pasted history picks into ``fetched['picks']``; ESPN picks win.

    ESPN's read API lags the live draft, so the draft room's pick history is pasted
    into ``draft_history.txt`` and carried in from there. Ownership still comes from
    ESPN's laid-out board (unmade picks carry ``teamId`` too), so the derivation
    check keeps running against what ESPN reports; the derived owner covers any pick
    the layout misses. Raises ValueError on a malformed paste.
    """
    records = draft_history.parse(text, board.teams)
    txt_picks, warnings = draft_history.as_picks(records, players)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    layout = {
        int(pick["overallPickNumber"]): int(pick["teamId"])
        for pick in (league.get("draftDetail") or {}).get("picks") or []
    }
    reported = {pick["pick_no"] for pick in fetched["picks"]}
    added = 0
    for pick in txt_picks:
        if pick["pick_no"] in reported:
            continue
        if pick["pick_no"] in layout:
            pick["roster_id"] = layout[pick["pick_no"]]
        fetched["picks"].append(pick)
        added += 1
    fetched["picks"].sort(key=lambda pick: pick["pick_no"])
    return (
        f"history: {len(txt_picks)} picks in {paths.display(paths.DRAFT_HISTORY)}, "
        f"{added} overlaid, {len(reported)} already reported by ESPN"
    )


def resolve_me(board: Board, by_user: dict[str, dict]) -> dict:
    """My slot and roster, straight from the SWID — no name matching needed."""
    slot = next((s for s, uid in board.slot_to_user.items() if uid == MY_SWID), None)
    return {
        "username": (by_user.get(MY_SWID) or {}).get("username"),
        "user_id": MY_SWID if MY_SWID in by_user else None,
        "draft_slot": slot,
        "roster_id": board.slot_to_roster.get(slot) if slot else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--output", default=paths.DRAFT, type=Path)
    ap.add_argument("--report", action="store_true", help="print a validation summary to stderr")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="check the board geometry offline (no network), then exit",
    )
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest()

    source = league_url(LEAGUE_ID, SEASON)
    print(f"GET {source}", file=sys.stderr)
    try:
        league = fetch(LEAGUE_ID, SEASON)
    except urllib.error.HTTPError as exc:
        hint = " — cookies expired or wrong league?" if exc.code in (401, 403) else ""
        print(f"error: ESPN answered {exc.code} {exc.reason}{hint}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"error: request failed: {exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        by_espn, players = sleeper_index(paths.SLEEPER_PLAYERS)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {paths.display(paths.SLEEPER_PLAYERS)}: {exc}", file=sys.stderr)
        return 1

    try:
        fetched = adapt(league, by_espn, players)
        board = Board(fetched["draft"], fetched["traded"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: league {LEAGUE_ID} is not shaped as expected: {exc!r}", file=sys.stderr)
        return 1
    fatal = board.problems()

    if not fatal and paths.DRAFT_HISTORY.is_file():
        history_text = paths.DRAFT_HISTORY.read_text(encoding="utf-8")
        # A blank file is the pre-draft state, not a mis-paste — nothing to overlay.
        if history_text.strip():
            try:
                note = overlay_history(history_text, fetched, league, board, players)
            except ValueError as exc:
                print(f"error: {paths.display(paths.DRAFT_HISTORY)}: {exc}", file=sys.stderr)
                return 1
            print(note, file=sys.stderr)

    problems, notable = pick_number_problems(fetched["picks"], board)
    fatal += problems
    if fatal:
        for problem in fatal:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    for note in notable:
        print(f"warning: {note}", file=sys.stderr)
    if fetched["warning"]:
        print(f"warning: {fetched['warning']}", file=sys.stderr)
    if not board.slot_to_user:
        print(
            "warning: ESPN has not published a draft order yet — pick owners will be null",
            file=sys.stderr,
        )

    by_user = index_users(fetched["users"])
    me = resolve_me(board, by_user)
    if me["user_id"] is None:
        print(
            f"warning: no league member with SWID {MY_SWID} — no pick will be marked is_mine",
            file=sys.stderr,
        )
    elif me["draft_slot"] is None:
        print(f"warning: {me['username']} has no slot in the draft order", file=sys.stderr)

    rows, checks = pick_rows(board, fetched["picks"], by_user, me["user_id"])
    if checks["mismatches"]:
        print(
            f"warning: the derived pick order disagrees with ESPN on "
            f"{len(checks['mismatches'])} of {checks['made_picks_checked']} made picks — "
            "pending picks may be attributed to the wrong team; run --report",
            file=sys.stderr,
        )

    document = build_document(fetched, board, rows, checks, me, API)
    document["source"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    clock = document["on_the_clock"]
    mine_next = document["my_next_pick"]
    print(
        f"draft: {document['picks_made']}/{document['pick_count']} picks made, "
        f"status {document['status']}"
        + (
            f"; on the clock #{clock['pick_no']} ({clock['slot']}) "
            f"{clock['username'] or clock['user_id']}"
            if clock
            else "; complete"
        )
        + (
            f"; mine #{mine_next['pick_no']} ({mine_next['slot']}) in "
            f"{mine_next.get('picks_away')}"
            if mine_next
            else ""
        )
        + f" -> {paths.display(args.output)}",
        file=sys.stderr,
    )
    if args.report:
        draft_report.report(document, rows, board, paths.POOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
