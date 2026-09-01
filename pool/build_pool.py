#!/usr/bin/env python3
"""Build pool.json, the ranker's player pool, from the three snapshots in data/.

    uv run pool/build_pool.py            # -> ../pool.json
    uv run pool/build_pool.py --report   # + join diagnostics on stderr

Three inputs, one output:

    data/projections.html          hand-saved DraftSharks rankings page (the site's
                                   dynasty export; only identity, age, bye, rookie flag,
                                   1QB ADP and the one-season projection are read)
    data/sleeper_projections.json  Sleeper's top-500 season projections (fetch_projections.py)
    data/weekly_projections.json   DraftSharks per-week projections (fetch_projections.py)

A pool player is a DraftSharks QB/RB/WR/TE with a positive one-season projection who
also has a Sleeper season projection. Sleeper is the league host, so its id is what
draft.json and every provider board join on; a player Sleeper does not project could
never be recognised once drafted, so he is dropped rather than carried without an id.

There is no shared id between DraftSharks and Sleeper, so the join is by name in three
tiers, each requiring exactly one survivor at the same position (an ambiguous name is
left unmatched and dropped): full normalized name; name without a Jr./III-style suffix;
last name plus NFL team, for editorial first names (Cam/Cameron, Tank/Nathaniel). Team
codes differ between the two (JAC/JAX, LVR/LV) and are translated; DraftSharks' UNS/RK
sentinels mean unsigned. --report prints every tier-3 join for eyeballing, since that
is the tier where a wrong join would be plausible enough to slip through.

Each player carries:
    weekly_points  DraftSharks' league-scored points for weeks 1-17 (index 0 = week 1);
                   a week DraftSharks does not project is 0.0, a bye or a known
                   absence. Week 18 is dropped: the league ends with the week 16-17
                   final. This is the ranker's value input. The handful of deep players
                   missing from the weekly page fall back to a flat 1/17th of the Sleeper
                   season projection per week with only the bye zeroed, and are reported.
    points         Sleeper's season projection in the league's scoring: the membership
                   gate and the pool's sort order. The ranker values rosters from
                   weekly_points, whose sum is DraftSharks' season total instead.
    adp            DraftSharks' 1QB ADP as an overall pick number in the source's
                   12-team draft (its round.pick notation decoded: 2.03 -> 15). Past
                   ~pick 540 the tail is provider noise, not a real slot.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
PROJECTIONS_HTML = DATA / "projections.html"
SLEEPER_PROJECTIONS = DATA / "sleeper_projections.json"
WEEKLY_PROJECTIONS = DATA / "weekly_projections.json"
POOL = DATA.parent.parent / "pool.json"

POSITIONS = ("QB", "RB", "WR", "TE")
LEAGUE_WEEKS = 17
# The source's ADP is round.pick for a 12-team draft: the provider's format, not this
# league's size, so it is decoded to an overall pick rather than reinterpreted.
ADP_TEAMS = 12
# Provider team code -> Sleeper team code; everything else is already identical.
TEAM_ALIASES = {"JAC": "JAX", "LVR": "LV"}
NO_TEAM = frozenset({"UNS", "RK"})
PLACEHOLDER_BYE = 18  # what the source prints for unsigned players; real byes run 5-14
SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")


# --- DraftSharks page -------------------------------------------------------------


def to_number(raw: str | None) -> int | float | None:
    if raw is None:
        return None
    text = raw.strip().replace(",", "")
    if not text or text in {"-", "--", "N/A"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() and "." not in text else value


class PageParser(HTMLParser):
    """One record per ``<tbody data-player-row>``. Every scoring scheme's value sits on
    each cell as a data-scoring-value-* attribute (the page is a server-rendered Alpine
    table), so the half-PPR 1QB columns are read straight from the static HTML."""

    #: data-attribute -> (record key, attribute holding the value)
    CELLS = {
        "adp": ("adp", "data-scoring-value-half-ppr"),
        "fantasy_points": ("ds_points", "data-scoring-value-half-ppr"),
        "player.age": ("age", "data-value"),
        "player.team.bye": ("bye_week", "data-value"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.players: list[dict] = []
        self._row: dict | None = None
        self._in_player_row = False
        self._capture_team = False

    def handle_starttag(self, tag, attrs):
        got = {name: value or "" for name, value in attrs}
        if tag == "tbody":
            if "data-player-row" in got:
                self._row = {
                    "player_id": int(got["data-key"]),
                    "name": got["data-player-name"].strip(),
                    "position": got["data-fantasy-position"],
                    "team": "",
                    "is_rookie": got.get("data-is-rookie") == "true",
                }
            return
        if self._row is None:
            return
        if tag == "tr":
            self._in_player_row = "player-row" in got.get("class", "").split()
        elif not self._in_player_row:
            return  # the collapsed detail-view row repeats the same values
        elif tag == "td" and got.get("data-attribute") in self.CELLS:
            key, attribute = self.CELLS[got["data-attribute"]]
            self._row[key] = to_number(got.get(attribute))
        elif tag == "span" and "player-details-group__team-name" in got.get("class", ""):
            self._capture_team = True

    def handle_data(self, data):
        if self._capture_team and self._row is not None:
            self._row["team"] += data.strip()

    def handle_endtag(self, tag):
        if tag == "span":
            self._capture_team = False
        elif tag == "tr":
            self._in_player_row = False
        elif tag == "tbody" and self._row is not None:
            self.players.append(self._row)
            self._row = None


def decode_adp(value: float | None) -> int | None:
    """``2.03`` (round 2, pick 3) -> overall pick 15."""
    if value is None:
        return None
    rnd = int(value)
    return (rnd - 1) * ADP_TEAMS + round((value - rnd) * 100)


def parse_page(path: Path) -> list[dict]:
    parser = PageParser()
    with path.open(encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), ""):
            parser.feed(chunk)
    parser.close()
    if not parser.players:
        raise SystemExit("error: no player rows in projections.html; did the page layout change?")
    rows = []
    for p in parser.players:
        # The source uses 0 where a null belongs: no projection means no pool row.
        if p["position"] not in POSITIONS or not p.get("ds_points") or p["ds_points"] <= 0:
            continue
        unsigned = p["team"] in NO_TEAM
        bye = p.get("bye_week")
        rows.append(
            {
                **p,
                "bye_week": None if unsigned or bye == PLACEHOLDER_BYE else bye,
                "adp": decode_adp(p.get("adp")),
            }
        )
    return rows


# --- Sleeper join -----------------------------------------------------------------


def norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def name_words(name: str) -> list[str]:
    """Normalized words with a trailing suffix dropped: ``Dont'e Thornton Jr.`` ->
    ``['donte', 'thornton']``."""
    words = [w for w in (norm(word) for word in name.split()) if w] or [norm(name)]
    while len(words) > 1 and words[-1] in SUFFIXES:
        words.pop()
    return words


def sleeper_team(code: str | None) -> str | None:
    if not code or code in NO_TEAM:
        return None
    return TEAM_ALIASES.get(code, code)


class SleeperIndex:
    def __init__(self, rows: list[dict]):
        self.by_name = collections.defaultdict(list)
        self.by_base = collections.defaultdict(list)
        self.by_last = collections.defaultdict(list)
        for row in rows:
            key = row["position"]
            self.by_name[(norm(row["name"]), key)].append(row)
            self.by_base[("".join(name_words(row["name"])), key)].append(row)
            self.by_last[(name_words(row["name"])[-1], key)].append(row)


def match(row: dict, index: SleeperIndex) -> tuple[dict | None, str]:
    """(Sleeper row or None, tier). Team breaks a same-name tie but never vetoes a lone
    candidate: the two providers disagree about who plays where in the off-season."""
    team = sleeper_team(row["team"])
    position = row["position"]
    for tier, candidates in (
        ("name", index.by_name.get((norm(row["name"]), position), [])),
        ("suffix", index.by_base.get(("".join(name_words(row["name"])), position), [])),
    ):
        if len(candidates) > 1 and team:
            candidates = [c for c in candidates if c["team"] == team] or candidates
        if len(candidates) == 1:
            return candidates[0], tier
        if candidates:
            return None, f"ambiguous {tier}"
    # Tier 3: the first name is editorial (Tank/Nathaniel), so the team carries the
    # join. A Sleeper row with no team yet (a fresh signing) has nothing to confirm it,
    # so it only counts when the first names are prefixes (Jam/Jamarion).
    if team:
        words = name_words(row["name"])
        candidates = [
            c
            for c in index.by_last.get((words[-1], position), [])
            if c["team"] == team
            or (
                c["team"] is None
                and (
                    words[0].startswith(name_words(c["name"])[0])
                    or name_words(c["name"])[0].startswith(words[0])
                )
            )
        ]
        if len(candidates) == 1:
            return candidates[0], "last_name_team"
        if candidates:
            return None, "ambiguous last_name_team"
    return None, "none"


# --- weekly shape -----------------------------------------------------------------


def native_weeks(weeks: dict[str, dict]) -> list[float]:
    return [
        round(max(float((weeks.get(str(w)) or {}).get("points", 0.0)), 0.0), 2)
        for w in range(1, LEAGUE_WEEKS + 1)
    ]


def uniform_weeks(points: float, bye_week: int | None) -> list[float]:
    per_week = round(points / LEAGUE_WEEKS, 2)
    return [0.0 if w == bye_week else per_week for w in range(1, LEAGUE_WEEKS + 1)]


# --- build ------------------------------------------------------------------------


def build(report: bool) -> dict:
    sleeper = json.loads(SLEEPER_PROJECTIONS.read_text())
    weekly = json.loads(WEEKLY_PROJECTIONS.read_text())
    page = parse_page(PROJECTIONS_HTML)
    index = SleeperIndex([r for r in sleeper["players"] if r["position"] in POSITIONS])
    weeks_by_id = {p["player_id"]: p["weeks"] for p in weekly["players"]}

    players: list[dict] = []
    tiers: collections.Counter[str] = collections.Counter()
    tier3: list[tuple[dict, dict]] = []
    unmatched: list[tuple[dict, str]] = []
    fallbacks: list[dict] = []
    for row in page:
        found, tier = match(row, index)
        tiers[tier] += 1
        if found is None:
            unmatched.append((row, tier))
            continue
        if tier == "last_name_team":
            tier3.append((row, found))
        weeks = weeks_by_id.get(row["player_id"])
        player = {
            "player_id": row["player_id"],
            "sleeper_id": found["sleeper_id"],
            "name": row["name"],
            "position": row["position"],
            "team": row["team"],
            "age": row.get("age"),
            "bye_week": row["bye_week"],
            "is_rookie": row["is_rookie"],
            "points": found["points"],
            "adp": row["adp"],
            "weekly_points": (
                native_weeks(weeks)
                if weeks is not None
                else uniform_weeks(found["points"], row["bye_week"])
            ),
        }
        if weeks is None:
            fallbacks.append(player)
        players.append(player)

    duplicates = [
        sid for sid, n in collections.Counter(p["sleeper_id"] for p in players).items() if n > 1
    ]
    if duplicates:
        raise SystemExit(f"error: sleeper id(s) matched more than one pool player: {duplicates}")
    players.sort(key=lambda p: (-p["points"], p["player_id"]))

    for player in sorted(fallbacks, key=lambda p: -p["points"]):
        print(
            f"warning: no DraftSharks weekly projection for {player['name']} "
            f"({player['position']} {player['team']}, {player['points']} pts); uniform weeks",
            file=sys.stderr,
        )
    if report:
        print(
            f"\nDraftSharks page: {len(page)} QB/RB/WR/TE with a projection; matched "
            f"{len(players)} to Sleeper's {len(sleeper['players'])} ("
            + ", ".join(f"{tier} {n}" for tier, n in tiers.items())
            + ")",
            file=sys.stderr,
        )
        print(f"\ntier 3, last name + team ({len(tier3)}):", file=sys.stderr)
        for row, found in tier3:
            print(
                f"  {row['name']:<24} ({row['position']} {row['team']}) -> "
                f"{found['name']:<20} {found['position']} {found['team']} id={found['sleeper_id']}",
                file=sys.stderr,
            )
        print(f"\nunmatched, dropped ({len(unmatched)}):", file=sys.stderr)
        for row, tier in sorted(unmatched, key=lambda r: -r[0]["ds_points"]):
            print(
                f"  {row['name']:<24} {row['position']} {row['team']:<4} {row['ds_points']:>4} pts"
                f"{'  ' + tier if tier != 'none' else ''}",
                file=sys.stderr,
            )
        counts = collections.Counter(p["position"] for p in players)
        absent = [p for p in players if sum(v == 0.0 for v in p["weekly_points"]) > 1]
        print(
            "\npool: " + ", ".join(f"{pos} {counts[pos]}" for pos in POSITIONS)
            + f" = {len(players)}; {len(fallbacks)} uniform weekly fallbacks; "
            f"{len(absent)} players with zero weeks beyond a bye, e.g. "
            + ", ".join(
                f"{p['name']} {[w + 1 for w, v in enumerate(p['weekly_points']) if v == 0.0]}"
                for p in sorted(absent, key=lambda p: -p["points"])[:5]
            ),
            file=sys.stderr,
        )
        print(file=sys.stderr)

    return {
        "sources": {
            "identity_and_adp": PROJECTIONS_HTML.name,
            "points": {"file": SLEEPER_PROJECTIONS.name, "fetched_at": sleeper.get("fetched_at")},
            "weekly_points": {"file": WEEKLY_PROJECTIONS.name, "fetched_at": weekly.get("fetched_at")},
        },
        "player_count": len(players),
        "players": players,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", action="store_true", help="join diagnostics on stderr")
    args = ap.parse_args(argv)

    document = build(args.report)
    POOL.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {document['player_count']} players -> {POOL.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
