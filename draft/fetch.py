#!/usr/bin/env python3
"""Fetch Sleeper's live draft and publish the complete made-and-pending board.

Board geometry and the output contract live in `geometry.py`; this entry point
coordinates I/O and prints the --report diagnostics.

Sleeper's draft API is public and real-time — a pick made in the draft room shows up on
the very next fetch, so there is no manual overlay step and nothing is cached between
runs. The default draft is this league's; `--draft-id` points a run at another draft,
e.g. a league mock, which carries the same shape (a mock's `league_id` moves into its
`metadata`, so the league user list still resolves).

Usage:
    uv run -m draft.fetch
    uv run -m draft.fetch --draft-id 1400304132081893376
    uv run -m draft.fetch --report
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.error
from pathlib import Path

from shared.paths import DRAFT, POOL
from shared.sleeper import API, MY_USER_ID, get_json

from .geometry import (
    Board,
    build_document,
    index_users,
    pick_number_problems,
    pick_rows,
    resolve_me,
    round_pick,
)
from .settings import DRAFT_ID

TIMEOUT_SECONDS = 30


def fetch(draft_id: str, api: str = API, timeout: int = TIMEOUT_SECONDS) -> dict:
    """The draft, its picks, its traded picks, and the league's users.

    The first three are load-bearing and a failure is fatal — half a board is worse
    than none, and a missing traded_picks would silently misattribute pending picks.
    The user list only supplies display names, so it degrades to a warning.
    """
    draft = get_json(f"{api}/draft/{draft_id}", timeout)
    if not isinstance(draft, dict) or not draft.get("draft_id"):
        raise ValueError(
            f"no draft {draft_id} at {api} — check the id in the draft URL"
        )

    picks = get_json(f"{api}/draft/{draft_id}/picks", timeout) or []
    traded = get_json(f"{api}/draft/{draft_id}/traded_picks", timeout) or []
    if not isinstance(picks, list) or not isinstance(traded, list):
        raise ValueError("picks/traded_picks did not come back as lists")

    users, warning = [], None
    # A league mock carries its league in metadata rather than league_id.
    league_id = draft.get("league_id") or (draft.get("metadata") or {}).get("league_id")
    if league_id:
        try:
            users = get_json(f"{api}/league/{league_id}/users", timeout) or []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            warning = f"could not read league users ({exc}) — names will be null"
    else:
        warning = "draft has no league_id — names will be null"

    return {
        "draft": draft,
        "picks": picks,
        "traded": traded,
        "users": users,
        "warning": warning,
    }


def pool_join(rows: list[dict], pool_path: Path) -> dict | None:
    """How the made picks land in pool.json — the join this file exists to enable."""
    if not pool_path.is_file():
        return None
    try:
        with pool_path.open(encoding="utf-8") as handle:
            players = json.load(handle).get("players") or []
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    # pool.json is ordered by season projection, so the index is the pool rank.
    by_id = {str(p["sleeper_id"]): (rank, p) for rank, p in enumerate(players, start=1)}
    made = [row for row in rows if row["status"] == "made" and row["sleeper_id"]]
    hits = [(row, *by_id[row["sleeper_id"]]) for row in made if row["sleeper_id"] in by_id]
    return {
        "pool_size": len(players),
        "pool_with_id": len(by_id),
        "made": len(made),
        "matched": len(hits),
        "outside_pool": [row for row in made if row["sleeper_id"] not in by_id],
        "top_50_gone": sum(1 for _, rank, _ in hits if rank <= 50),
        # A position mismatch on a joined id would mean the pool build joined the
        # wrong player, the one failure mode a name-based join can hide.
        "disagreements": [
            (row, player) for row, _, player in hits if row["position"] != player["position"]
        ],
    }


def report(document: dict, rows: list[dict], board: Board, pool_path: Path) -> None:
    """Derivation checks against what Sleeper reported, the latest picks, and the pool join."""
    out = sys.stderr
    fmt = document["format"]
    print(
        f"\ndraft {document['draft_id']} — {document.get('league_name')} {document.get('season')}, "
        f"status {document['status']}: {fmt['type']}, {fmt['teams']} teams x {fmt['rounds']} rounds"
        + (f", reversal at round {fmt['reversal_round']}" if fmt["reversal_round"] else "")
        + f"; last pick {document['last_picked_at']}",
        file=out,
    )
    check = document["board_derivation"]
    print(
        f"  slot and roster agree with Sleeper on {check['slot_and_roster_agree']}/"
        f"{check['checked_against_made_picks']} made picks",
        file=out,
    )
    for bad in check["mismatches"][:10]:
        print(f"  ^ pick {bad['pick_no']}: reported {bad['reported']} vs derived {bad['derived']}", file=out)
    made = [row for row in rows if row["status"] == "made"]
    for row in made[-12:]:
        print(
            f"  {round_pick(row['round'], row['pick_in_round']):>6} #{row['pick_no']:<4} "
            f"{(row['username'] or row['user_id'] or '?'):<16} {(row['name'] or '?'):<24} "
            f"{row['position'] or '?':<3} {row['team'] or '?':<4}{'  <- mine' if row['is_mine'] else ''}",
            file=out,
        )
    drafted = [row["sleeper_id"] for row in made if row["sleeper_id"]]
    repeats = [i for i, n in collections.Counter(drafted).items() if n > 1]
    unowned = [row["pick_no"] for row in rows if row["user_id"] is None]
    print(
        f"  drafted twice: {repeats or 'none'}; picks without an owner: {len(unowned)}; "
        f"made picks without a player: {len(made) - len(drafted)}",
        file=out,
    )
    join = pool_join(rows, pool_path)
    if join is None:
        print("  pool.json not readable — join skipped", file=out)
        return
    print(
        f"  pool join: {join['matched']}/{join['made']} made picks are in the pool "
        f"({join['pool_with_id']}/{join['pool_size']} pool players carry a sleeper_id); "
        f"pool top 50 already gone: {join['top_50_gone']}",
        file=out,
    )
    for row in join["outside_pool"][:15]:
        print(f"  ^ outside the pool: #{row['pick_no']} {row['name']} {row['position']} {row['team']}", file=out)
    for row, player in join["disagreements"][:10]:
        print(
            f"  ^ position disagrees on sleeper_id {row['sleeper_id']}: sleeper {row['name']} "
            f"{row['position']} vs pool {player['name']} {player['position']}",
            file=out,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--draft-id", default=DRAFT_ID, help=f"default: {DRAFT_ID} (the league draft)"
    )
    ap.add_argument(
        "--report", action="store_true", help="print a validation summary to stderr"
    )
    args = ap.parse_args(argv)

    print(
        f"GET {API}/draft/{args.draft_id} (+picks, traded_picks, league users)",
        file=sys.stderr,
    )
    try:
        fetched = fetch(args.draft_id)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"error: request failed: {exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if fetched["warning"]:
        print(f"warning: {fetched['warning']}", file=sys.stderr)

    try:
        board = Board(fetched["draft"], fetched["traded"])
        fatal, notable = pick_number_problems(fetched["picks"], board)
        fatal = board.problems() + fatal
    except (TypeError, ValueError) as exc:
        print(
            f"error: draft {args.draft_id} is not shaped as expected: {exc}",
            file=sys.stderr,
        )
        return 1
    if fatal:
        for problem in fatal:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    for note in notable:
        print(f"warning: {note}", file=sys.stderr)
    if not board.slot_to_user:
        print(
            "warning: draft_order is empty — pick owners will be null", file=sys.stderr
        )

    by_user = index_users(fetched["users"])
    me = resolve_me(MY_USER_ID, board, by_user)
    if me["draft_slot"] is None:
        print(
            f"warning: user {MY_USER_ID} has no slot in this draft's order — "
            "no pick will be marked is_mine",
            file=sys.stderr,
        )

    rows, checks = pick_rows(board, fetched["picks"], by_user, MY_USER_ID)
    if checks["mismatches"]:
        print(
            f"warning: the derived pick order disagrees with Sleeper on "
            f"{len(checks['mismatches'])} of {checks['made_picks_checked']} made picks — "
            "pending picks may be attributed to the wrong team; run --report",
            file=sys.stderr,
        )

    document = build_document(fetched, board, rows, checks, me, API)
    DRAFT.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

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
        + f" -> {DRAFT.name}",
        file=sys.stderr,
    )
    if args.report:
        report(document, rows, board, POOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
