#!/usr/bin/env python3
"""Fetch the league's in-season state from Sleeper and publish league.json.

Everything the in-season model needs that is not a projection, plus Sleeper's own
weekly projections for the weeks still to play:

    week            Sleeper's current NFL week (state.week); the week the model decides for
    teams           every roster: owner, players, current starters, reserve (IR), FAAB used,
                    points so far; `alive` is false once the commissioner has emptied an
                    eliminated roster (Sleeper has no guillotine flag, so an empty roster
                    after week 1 is the elimination signal)
    transactions    every waiver claim and free-agent move so far, bid included, so the
                    market model can be checked against what the room actually paid
    players         name / position / NFL team / injury status for every player referenced
                    by a roster, a transaction, or a weekly projection
    projections     Sleeper (Rotowire) weekly projections in this league's scoring for the
                    current week through week 17, keyed by week then sleeper_id; a player
                    with no game that week has no entry (bye)

Usage:
    uv run season/fetch_league.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEAGUE_JSON = REPO_ROOT / "league.json"

LEAGUE_ID = "1397662420398247936"
MY_USER_ID = "1127785716420898816"  # johnor
API = "https://api.sleeper.app/v1"
PROJECTIONS = "https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&" + "&".join(
    f"position[]={p}" for p in ("QB", "RB", "WR", "TE")
)
LAST_WEEK = 17


def get_json(url: str):
    # Sleeper's CDN caches for minutes; a unique query param forces origin.
    bust = f"{'&' if '?' in url else '?'}nocache={time.time_ns()}"
    request = urllib.request.Request(
        url + bust, headers={"Accept": "application/json", "User-Agent": "curl/8.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def score(stats: dict, scoring: dict) -> float:
    return round(sum(scoring[k] * v for k, v in stats.items() if k in scoring and v), 2)


def main() -> int:
    state = get_json(f"{API}/state/nfl")
    league = get_json(f"{API}/league/{LEAGUE_ID}")
    week = int(state["week"])
    season = str(league["season"])
    if state.get("season") != season:
        raise SystemExit(f"error: NFL state is season {state.get('season')}, league is {season}")
    if not 1 <= week <= LAST_WEEK:
        raise SystemExit(f"error: NFL week {week} is outside the league's weeks 1-{LAST_WEEK}")
    scoring = league["scoring_settings"]

    users = {u["user_id"]: u for u in get_json(f"{API}/league/{LEAGUE_ID}/users")}
    rosters = get_json(f"{API}/league/{LEAGUE_ID}/rosters")
    matchups = {m["roster_id"]: m for m in get_json(f"{API}/league/{LEAGUE_ID}/matchups/{week}")}
    if len(rosters) != league["total_rosters"]:
        raise SystemExit(f"error: {len(rosters)} rosters for {league['total_rosters']} teams")

    teams = []
    for r in rosters:
        user = users.get(r.get("owner_id")) or {}
        settings = r.get("settings") or {}
        players = list(r.get("players") or [])
        teams.append(
            {
                "roster_id": r["roster_id"],
                "user_id": r.get("owner_id"),
                "username": user.get("display_name"),
                "team_name": ((user.get("metadata") or {}).get("team_name") or "").strip() or None,
                "is_mine": r.get("owner_id") == MY_USER_ID,
                "alive": bool(players) or week == 1,
                "players": players,
                "starters": list(r.get("starters") or []),
                "reserve": list(r.get("reserve") or []),
                "faab_used": int(settings.get("waiver_budget_used") or 0),
                "waiver_position": settings.get("waiver_position"),
                "points_for": float(settings.get("fpts") or 0) + float(settings.get("fpts_decimal") or 0) / 100,
                "points_this_week": float((matchups.get(r["roster_id"]) or {}).get("points") or 0.0),
            }
        )
    if sum(t["is_mine"] for t in teams) != 1:
        raise SystemExit(f"error: user {MY_USER_ID} owns {sum(t['is_mine'] for t in teams)} rosters")

    transactions = []
    for leg in range(1, week + 1):
        for tx in get_json(f"{API}/league/{LEAGUE_ID}/transactions/{leg}") or []:
            if tx.get("type") not in ("waiver", "free_agent"):
                continue
            transactions.append(
                {
                    "week": leg,
                    "type": tx["type"],
                    "status": tx.get("status"),
                    "roster_id": (tx.get("roster_ids") or [None])[0],
                    "adds": tx.get("adds") or {},
                    "drops": tx.get("drops") or {},
                    "bid": (tx.get("settings") or {}).get("waiver_bid"),
                    "note": (tx.get("metadata") or {}).get("notes"),
                    "created": dt.datetime.fromtimestamp(
                        tx["created"] / 1000, dt.timezone.utc
                    ).isoformat(timespec="seconds"),
                }
            )
    transactions.sort(key=lambda t: t["created"])

    projections: dict[str, dict[str, float]] = {}
    projected_players: dict[str, dict] = {}
    for w in range(week, LAST_WEEK + 1):
        rows = get_json(PROJECTIONS.format(season=season, week=w))
        col: dict[str, float] = {}
        for row in rows:
            if not row.get("date"):
                continue  # no game this week: bye
            points = score(row.get("stats") or {}, scoring)
            if points <= 0:
                continue
            col[row["player_id"]] = points
            who = row.get("player") or {}
            projected_players[row["player_id"]] = {
                "name": f"{who.get('first_name', '')} {who.get('last_name', '')}".strip(),
                "position": who.get("position"),
                "team": row.get("team"),
                "injury_status": who.get("injury_status"),
            }
        if len(col) < 250:
            raise SystemExit(f"error: week {w} has only {len(col)} projected QB/RB/WR/TE")
        projections[str(w)] = col
        print(f"week {w}: {len(col)} projected players", file=sys.stderr)

    referenced = {p for t in teams for p in t["players"]}
    for tx in transactions:
        referenced.update(tx["adds"])
        referenced.update(tx["drops"])
    directory = get_json(f"{API}/players/nfl")
    players = {}
    for pid in referenced | set(projected_players):
        row = directory.get(pid) or {}
        fallback = projected_players.get(pid, {})
        players[pid] = {
            "name": row.get("full_name") or fallback.get("name") or f"Sleeper #{pid}",
            "position": row.get("position") or fallback.get("position"),
            "team": row.get("team") or fallback.get("team"),
            "injury_status": row.get("injury_status") or fallback.get("injury_status"),
            "status": row.get("status"),
        }

    LEAGUE_JSON.write_text(
        json.dumps(
            {
                "source": f"{API}/league/{LEAGUE_ID}",
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "league_id": LEAGUE_ID,
                "league_name": league["name"],
                "season": season,
                "status": league["status"],
                "week": week,
                "season_start_date": state.get("season_start_date"),
                "roster_positions": league["roster_positions"],
                "reserve_slots": league["settings"].get("reserve_slots", 0),
                "faab_budget": league["settings"]["waiver_budget"],
                "me": {"user_id": MY_USER_ID, "roster_id": next(t["roster_id"] for t in teams if t["is_mine"])},
                "teams": teams,
                "transactions": transactions,
                "players": players,
                "projections": projections,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    alive = sum(t["alive"] for t in teams)
    print(
        f"week {week}: {alive}/{len(teams)} rosters alive, {len(transactions)} transactions, "
        f"{len(players)} players, projections for weeks {week}-{LAST_WEEK} -> {LEAGUE_JSON.name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
