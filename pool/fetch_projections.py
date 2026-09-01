#!/usr/bin/env python3
"""Refresh the two projection snapshots build_pool.py reads. Manual step; hits the network.

    uv run pool/fetch_projections.py

Both files are scored with the league's live Sleeper ``scoring_settings`` (the same dot
product the league players page uses), so the -2 pass INT, the +1.0 TE reception
premium and 6-point return TDs are all applied:

    data/sleeper_projections.json  Sleeper/Rotowire season projections: the top 500
                                   QB/RB/WR/TE by projected points, with half-PPR ADP
                                   (999.0 is Sleeper's "undrafted" sentinel). Gates pool
                                   membership and supplies the sleeper_id join key.
    data/weekly_projections.json   DraftSharks per-week stat projections for weeks 1-18,
                                   from the weekly-rankings page's public load-rows
                                   endpoint. A player absent from a week has no game that
                                   week (bye or known absence). DraftSharks publishes no
                                   weekly fumble or 2-pt projections, so those terms are
                                   absent: a small optimistic bias, largest for QBs.

Refetch when projections should move (injury news, depth-chart changes), then rebuild
the pool. Neither file is touched by the live-draft refresh loop.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SLEEPER_PROJECTIONS = DATA / "sleeper_projections.json"
WEEKLY_PROJECTIONS = DATA / "weekly_projections.json"

LEAGUE_ID = "1397662420398247936"
POSITIONS = ("QB", "RB", "WR", "TE")  # no kicker or D/ST slots in this league
# 32 teams x 8 rounds = 256 all-offense picks, so the priced tail must run well past
# the draft; 500 covers every DraftSharks pool player that could plausibly match.
TOP_N = 500
WEEKS = range(1, 19)

LEAGUE_URL = f"https://api.sleeper.app/v1/league/{LEAGUE_ID}"
SEASON_URL = (
    "https://api.sleeper.com/projections/nfl/{season}?season_type=regular&"
    + "&".join(f"position[]={p}" for p in POSITIONS)
)
WEEKLY_PAGE = "https://www.draftsharks.com/weekly-rankings/1/flex/half-ppr"
WEEKLY_ROWS = (
    "https://www.draftsharks.com/weekly-rankings/load-rows"
    "?offset=0&limit=1000&fantasyPosition=flex&pprSuperflexSlug=half-ppr"
    "&researchDepth=projections&week={week}"
)

# DraftSharks data-attribute -> Sleeper scoring_settings key. Attributes that never
# score in this league (kicking, D/ST, IDP, return yardage) are not captured.
STAT_TO_SCORING = {
    "pass_yds": "pass_yd",
    "pass_tds": "pass_td",
    "pass_int": "pass_int",
    "rush_yds": "rush_yd",
    "rush_tds": "rush_td",
    "rec_catch": "rec",
    "rec_yds": "rec_yd",
    "rec_tds": "rec_td",
    "return_touchdowns": "st_td",
}


def fetch_json(url: str):
    # Sleeper's hosts reject urllib's default Python-urllib/x.y agent with a 403.
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "curl/8.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


# --- Sleeper season projections ---------------------------------------------------


def fetch_sleeper(scoring: dict, season: str) -> None:
    rows = fetch_json(SEASON_URL.format(season=season))
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
            f"error: only {len(players)} projected players, expected >= {TOP_N}; "
            "truncated or off-season response?"
        )
    players.sort(key=lambda p: (-p["points"], p["adp"] or 999.0))
    write(
        SLEEPER_PROJECTIONS,
        {
            "source": SEASON_URL.format(season=season),
            "league_id": LEAGUE_ID,
            "season": season,
            "fetched_at": now(),
            "players": players[:TOP_N],
        },
    )
    print(
        f"wrote top {TOP_N} of {len(players)} Sleeper projections -> {SLEEPER_PROJECTIONS.name}",
        file=sys.stderr,
    )


# --- DraftSharks weekly projections ------------------------------------------------


class RowParser(HTMLParser):
    """Per-player stat values from one week's load-rows HTML: each player is a
    ``<tbody data-player-row>`` whose ``<td>`` cells carry data-attribute/data-value
    pairs, with the team abbreviation in a ``player-details-group__team-name`` span."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict] = []
        self.current: dict | None = None
        self.in_team_span = False

    def handle_starttag(self, tag, attrs) -> None:
        got = dict(attrs)
        if tag == "tbody" and "data-player-row" in got:
            self.current = {
                "player_id": int(got["data-key"]),
                "name": got["data-player-name"],
                "position": got["data-fantasy-position"],
                "team": "",
                "values": {},
            }
            self.rows.append(self.current)
        elif tag == "td" and self.current is not None and "data-attribute" in got:
            self.current["values"][got["data-attribute"]] = got.get("data-value")
        elif tag == "span" and self.current is not None:
            if "player-details-group__team-name" in (got.get("class") or ""):
                self.in_team_span = True

    def handle_data(self, data: str) -> None:
        if self.in_team_span and self.current is not None:
            self.current["team"] += data.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            self.in_team_span = False


def stat_value(raw: str | None) -> float:
    if not raw:
        return 0.0
    return float(re.sub(r"[^0-9.-]", "", raw) or 0)


def fetch_week(week: int) -> list[dict]:
    request = urllib.request.Request(
        WEEKLY_ROWS.format(week=week), headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    parser = RowParser()
    parser.feed(body)
    rows = [r for r in parser.rows if r["position"] in POSITIONS]
    # Bye weeks remove up to 6 teams' players; anything below ~3/4 of the ~430-player
    # offense pool means a truncated or reshaped response.
    if len(rows) < 300:
        raise SystemExit(
            f"error: week {week} returned only {len(rows)} QB/RB/WR/TE rows; "
            "truncated response or page layout change?"
        )
    return rows


def score(values: dict[str, str | None], position: str, scoring: dict) -> float:
    points = sum(
        scoring.get(key, 0.0) * stat_value(values.get(attr))
        for attr, key in STAT_TO_SCORING.items()
    )
    if position == "TE":
        points += scoring.get("bonus_rec_te", 0.0) * stat_value(values.get("rec_catch"))
    return round(points, 2)


def fetch_weekly(scoring: dict, season: str) -> None:
    players: dict[int, dict] = {}
    for week in WEEKS:
        rows = fetch_week(week)
        for row in rows:
            player = players.setdefault(
                row["player_id"],
                {
                    "player_id": row["player_id"],
                    "name": row["name"],
                    "position": row["position"],
                    "team": row["team"],
                    "weeks": {},
                },
            )
            player["weeks"][str(week)] = {"points": score(row["values"], row["position"], scoring)}
        print(f"week {week}: {len(rows)} players", file=sys.stderr)
        time.sleep(0.5)
    write(
        WEEKLY_PROJECTIONS,
        {
            "source": WEEKLY_PAGE,
            "league_id": LEAGUE_ID,
            "season": season,
            "fetched_at": now(),
            "scoring": {
                key: scoring.get(key, 0.0)
                for key in (*STAT_TO_SCORING.values(), "bonus_rec_te")
            },
            "players": sorted(
                players.values(),
                key=lambda p: -sum(w["points"] for w in p["weeks"].values()),
            ),
        },
    )
    print(
        f"wrote {len(players)} players x {len(WEEKS)} weeks -> {WEEKLY_PROJECTIONS.name}",
        file=sys.stderr,
    )


def main() -> int:
    league = fetch_json(LEAGUE_URL)
    scoring, season = league["scoring_settings"], league["season"]
    fetch_sleeper(scoring, season)
    fetch_weekly(scoring, season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
