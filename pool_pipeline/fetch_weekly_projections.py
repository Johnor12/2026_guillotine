#!/usr/bin/env python3
"""Capture DraftSharks' weekly stat projections, scored with league settings.
**Manual step — not a pipeline stage.**

    uv run pool_pipeline/fetch_weekly_projections.py

The weekly-rankings page (draftsharks.com/weekly-rankings/1/flex/half-ppr) loads
its rows from a public ``load-rows`` endpoint — no login required. With
``researchDepth=projections`` each row carries the projections-tab stat
breakdown as ``data-value`` attributes: passing att/cmp/yds/TDs/INTs, rushing
yds/TDs, receiving catches/yds/TDs, and return TDs, per player per week. The
position filter is client-side, so one request per week returns the entire
player pool; a player absent from a week's response has no game that week (bye).

Each stat line is scored with the league's live Sleeper ``scoring_settings``
(same dot product as ``fetch_sleeper_projections.py``), which bakes in the
league's quirks: -2 per pass INT, the +1.0 TE reception premium, and 6-point
return TDs. DraftSharks does not publish weekly fumble or 2-pt-conversion
projections, so the ``fum_lost``/``*_2pt`` terms are necessarily absent — a
small (< ~0.3 pt/wk) optimistic bias, largest for QBs. DraftSharks' own
half-PPR weekly total is kept alongside as ``ds_points`` for sanity checks;
it is not comparable to ``points`` and is never mixed with it.

Writes ``data/weekly_projections.json``: every QB/RB/WR/TE (no K or D/ST slots
in this league) with a per-week map of league-scored ``points``, ``ds_points``,
and the raw stat line. ``player_id`` is the DraftSharks id, the same id
``pool.json`` carries, so the join is direct.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser

import paths

LEAGUE_ID = "1397662420398247936"
POSITIONS = ("QB", "RB", "WR", "TE")  # no kickers or D/ST slots in this league
WEEKS = range(1, 19)

LEAGUE_URL = f"https://api.sleeper.app/v1/league/{LEAGUE_ID}"
PAGE_URL = "https://www.draftsharks.com/weekly-rankings/1/flex/half-ppr"
ROWS_URL = (
    "https://www.draftsharks.com/weekly-rankings/load-rows"
    "?offset=0&limit=1000&fantasyPosition=flex&pprSuperflexSlug=half-ppr"
    "&researchDepth=projections&week={week}"
)

# DraftSharks data-attribute -> Sleeper scoring_settings key. Attributes that
# exist in the table but never score in this league (kicking, D/ST buckets,
# IDP, return yardage) are not captured.
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
# Captured for granularity even though their scoring weight is 0.
EXTRA_STATS = ("pass_att", "pass_cmp")


class RowParser(HTMLParser):
    """Pull per-player stat values out of one week's load-rows HTML.

    Each player is a ``<tbody data-player-row ...>`` whose ``<td>`` cells carry
    ``data-attribute``/``data-value`` pairs; the team abbreviation lives in a
    ``player-details-group__team-name`` span.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict] = []
        self.current: dict | None = None
        self.in_team_span = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


def fetch_league_scoring() -> tuple[dict, str]:
    # api.sleeper.app rejects urllib's default Python-urllib/x.y agent with a 403
    request = urllib.request.Request(
        LEAGUE_URL, headers={"Accept": "application/json", "User-Agent": "curl/8.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        league = json.load(response)
    return league["scoring_settings"], league["season"]


def fetch_week_rows(week: int) -> list[dict]:
    request = urllib.request.Request(
        ROWS_URL.format(week=week), headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    parser = RowParser()
    parser.feed(body)
    rows = [r for r in parser.rows if r["position"] in POSITIONS]
    # Bye weeks remove up to 6 teams' players; anything below ~3/4 of the
    # ~430-player offense pool means a truncated or reshaped response.
    if len(rows) < 300:
        raise SystemExit(
            f"error: week {week} returned only {len(rows)} QB/RB/WR/TE rows — "
            "truncated response or page layout change?"
        )
    return rows


def stat_value(raw: str | None) -> float:
    if not raw:
        return 0.0
    return float(re.sub(r"[^0-9.-]", "", raw) or 0)


def score(stats: dict[str, float], position: str, scoring: dict) -> float:
    points = sum(
        scoring.get(key, 0.0) * stats.get(attr, 0.0)
        for attr, key in STAT_TO_SCORING.items()
    )
    if position == "TE":
        points += scoring.get("bonus_rec_te", 0.0) * stats.get("rec_catch", 0.0)
    return round(points, 2)


def main() -> int:
    scoring, season = fetch_league_scoring()

    players: dict[int, dict] = {}
    for week in WEEKS:
        rows = fetch_week_rows(week)
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
            stats = {
                attr: stat_value(row["values"].get(attr))
                for attr in (*STAT_TO_SCORING, *EXTRA_STATS)
            }
            player["weeks"][str(week)] = {
                "points": score(stats, row["position"], scoring),
                "ds_points": stat_value(row["values"].get("weekly3dPts")),
                "stats": {k: v for k, v in stats.items() if v},
            }
        print(f"week {week}: {len(rows)} players", file=sys.stderr)
        time.sleep(0.5)

    out = {
        "source": PAGE_URL,
        "league_id": LEAGUE_ID,
        "season": season,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "scoring": {
            key: scoring.get(key, 0.0)
            for key in (*STAT_TO_SCORING.values(), "bonus_rec_te")
        },
        "players": sorted(
            players.values(),
            key=lambda p: -sum(w["points"] for w in p["weeks"].values()),
        ),
    }
    with paths.WEEKLY_PROJECTIONS.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")

    print(
        f"wrote {len(players)} players x {len(WEEKS)} weeks to "
        f"{paths.display(paths.WEEKLY_PROJECTIONS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
