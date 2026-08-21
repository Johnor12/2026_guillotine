#!/usr/bin/env python3
"""Capture Sleeper's season projections and ADP. **Manual step — not a pipeline stage.**

    uv run pool_pipeline/fetch_sleeper_projections.py

The league players page (sleeper.com/leagues/<id>/players) is backed by a public
projections endpoint — no login required. It serves Rotowire's full-season stat
projections plus ADP for every player. The points shown on the page are *not* the
endpoint's precomputed ``pts_half_ppr`` (that assumes Sleeper's default scoring,
e.g. -1 per pass INT where this league uses -2); the page scores the raw stat
projections with the league's own ``scoring_settings``. This script does the same
dot product, which reproduces the page's numbers exactly.

Writes ``data/sleeper_projections.json``: the top 250 players by projected points
across QB/RB/WR/TE/DEF (the page's sort), each with ``sleeper_id``, ``name``,
``position``, ``team``, ``points`` (league scoring), and ``adp`` (half-PPR
redraft, the league's format; 999.0 is Sleeper's "undrafted" sentinel).
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

import paths

LEAGUE_ID = "1395974104053448704"
POSITIONS = ("QB", "RB", "WR", "TE", "DEF")  # no kickers in this league
TOP_N = 250

LEAGUE_URL = f"https://api.sleeper.app/v1/league/{LEAGUE_ID}"
PROJECTIONS_URL = (
    "https://api.sleeper.com/projections/nfl/{season}?season_type=regular&"
    + "&".join(f"position[]={p}" for p in POSITIONS)
)


def fetch_json(url: str):
    # api.sleeper.com rejects urllib's default Python-urllib/x.y agent with a 403
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "curl/8.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    league = fetch_json(LEAGUE_URL)
    scoring = league["scoring_settings"]
    season = league["season"]

    rows = fetch_json(PROJECTIONS_URL.format(season=season))
    players = []
    for row in rows:
        stats = row.get("stats") or {}
        points = sum(scoring[k] * v for k, v in stats.items() if k in scoring and v)
        if points <= 0:
            continue
        who = row["player"]
        players.append(
            {
                "sleeper_id": row["player_id"],
                "name": f"{who['first_name']} {who['last_name']}",
                "position": who["position"],
                "team": row["team"],
                "points": round(points, 2),
                "adp": stats.get("adp_half_ppr"),
            }
        )

    if len(players) < TOP_N:
        raise SystemExit(
            f"error: only {len(players)} projected players, expected >= {TOP_N} — "
            "truncated or off-season response?"
        )
    players.sort(key=lambda p: (-p["points"], p["adp"] or 999.0))

    out = {
        "source": PROJECTIONS_URL.format(season=season),
        "league_id": LEAGUE_ID,
        "season": season,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "players": players[:TOP_N],
    }
    with paths.SLEEPER_PROJECTIONS.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")

    print(
        f"wrote top {TOP_N} of {len(players)} projected players to "
        f"{paths.display(paths.SLEEPER_PROJECTIONS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
