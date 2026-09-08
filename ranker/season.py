"""In-season state: the player universe, the 32 rosters, and the weekly lineup solver.

Value input for the season is per-week and per-player: DraftSharks' weekly projection
(pool/data/weekly_projections.json, joined to Sleeper ids through pool.json) blended 2:1
with Sleeper's weekly projection for the same week (league.json), the same weighting the
draft used on season totals. A player only Sleeper projects — anyone outside the draft
pool who has since become relevant — carries Sleeper's number alone. Weeks already
played are zero: nothing in the season model looks backwards.

The weekly lineup is the greedy optimum: dedicated slots take each position's best
bodies, the flex seats take the best pooled RB/WR/TE leftovers. That greedy fill is
exact for this slot chain (league.SLOT_CHAIN), so the current week's recommendation is
the true optimum under the projections. The race simulator scores every team-week the
same way, with surprise inactives left to the score noise rather than the availability
cascade the draft valuation used — the race runs tens of thousands of team-weeks and the
cascade is too slow for that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .league import (
    FAAB_BUDGET,
    POSITIONS,
    WEEK_ROSTER_SIZE,
    WEEKLY_SHAPES,
    WEEKS,
)
from .projections import DRAFTSHARKS_WEIGHT

POS_CODE = {pos: i for i, pos in enumerate(POSITIONS)}
# Sleeper's roster_positions vocabulary -> the shape keys WEEKLY_SHAPES uses.
_SLEEPER_SLOT = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "FLEX": "FLEX", "SUPER_FLEX": "QB"}


@dataclass(slots=True)
class SeasonPlayer:
    index: int
    sleeper_id: str
    name: str
    position: str
    team: str | None
    injury_status: str | None
    weekly: tuple[float, ...]  # per league week, past weeks zero
    source: str  # "blend" (DraftSharks + Sleeper) or "sleeper"


@dataclass(slots=True)
class Team:
    roster_id: int
    name: str
    username: str | None
    is_mine: bool
    alive: bool
    roster: list[int]  # player indexes
    starters: list[str | None]  # Sleeper's current lineup, sleeper ids in slot order
    reserve: list[int]
    unknown: list[str]  # rostered ids with no projection anywhere (roster filler)
    faab_left: int
    points_for: float
    points_this_week: float


@dataclass(slots=True)
class SeasonState:
    week: int  # 1-based; the week being decided
    fetched_at: str
    league_name: str
    players: list[SeasonPlayer]
    teams: list[Team]  # roster_id order
    me: int  # index into teams
    free_agents: list[int]  # player indexes on no roster
    waivers_ran: bool  # this week's claims already processed (or week 1's free agency)
    transactions: list[dict]
    problems: list[str] = field(default_factory=list)

    @property
    def my_team(self) -> Team:
        return self.teams[self.me]


def _shape_from_sleeper(positions: list[str]) -> dict[str, int]:
    shape = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0}
    bench = 0
    for slot in positions:
        if slot == "BN":
            bench += 1
        elif slot in _SLEEPER_SLOT:
            shape[_SLEEPER_SLOT[slot]] += 1
        else:
            raise ValueError(f"roster slot {slot!r} has no place in this league's lineup")
    return shape | {"BN": bench}


def load_season(pool_path: Path, weekly_path: Path, league_path: Path) -> SeasonState:
    pool = json.loads(pool_path.read_text())
    weekly_raw = json.loads(weekly_path.read_text())
    league = json.loads(league_path.read_text())
    week = int(league["week"])
    w0 = week - 1
    problems: list[str] = []

    live = _shape_from_sleeper(league["roster_positions"])
    want = WEEKLY_SHAPES[w0] | {"BN": WEEK_ROSTER_SIZE[w0] - sum(WEEKLY_SHAPES[w0].values())}
    if live != want:
        problems.append(
            f"Sleeper's week-{week} roster shape {live} disagrees with league.py {want}; "
            "edit WEEKLY_SHAPES / WEEK_ROSTER_SIZE"
        )
    if league["faab_budget"] != FAAB_BUDGET:
        problems.append(f"Sleeper says the FAAB budget is {league['faab_budget']}, league.py {FAAB_BUDGET}")

    # DraftSharks weekly by sleeper id, through the pool's id join.
    sleeper_of_ds = {p["player_id"]: p["sleeper_id"] for p in pool["players"] if p.get("sleeper_id")}
    ds_weekly: dict[str, dict[str, float]] = {}
    for row in weekly_raw["players"]:
        sid = sleeper_of_ds.get(row["player_id"])
        if sid:
            ds_weekly[sid] = {w: v["points"] for w, v in row["weeks"].items()}
    sl_weekly = league["projections"]  # week -> sleeper_id -> points
    directory = league["players"]

    def profile(sid: str) -> tuple[tuple[float, ...], str] | None:
        ds = ds_weekly.get(sid)
        out = [0.0] * WEEKS
        any_points = False
        for w in range(w0, WEEKS):
            sl = sl_weekly.get(str(w + 1), {}).get(sid, 0.0)
            if ds is not None:
                v = DRAFTSHARKS_WEIGHT * ds.get(str(w + 1), 0.0) + (1 - DRAFTSHARKS_WEIGHT) * sl
            else:
                v = sl
            out[w] = round(v, 2)
            any_points |= v > 0
        if not any_points:
            return None
        return tuple(out), ("blend" if ds is not None else "sleeper")

    rostered = {sid for t in league["teams"] for sid in t["players"]}
    candidates = set(directory) | rostered
    players: list[SeasonPlayer] = []
    index: dict[str, int] = {}
    for sid in sorted(candidates):
        meta = directory.get(sid) or {}
        position = meta.get("position")
        if position not in POSITIONS:
            continue
        got = profile(sid)
        if got is None:
            continue
        weekly, source = got
        index[sid] = len(players)
        players.append(
            SeasonPlayer(
                index=len(players),
                sleeper_id=sid,
                name=meta.get("name") or f"Sleeper #{sid}",
                position=position,
                team=meta.get("team"),
                injury_status=meta.get("injury_status"),
                weekly=weekly,
                source=source,
            )
        )

    teams: list[Team] = []
    for t in sorted(league["teams"], key=lambda t: t["roster_id"]):
        roster = [index[sid] for sid in t["players"] if sid in index]
        unknown = [sid for sid in t["players"] if sid not in index]
        teams.append(
            Team(
                roster_id=t["roster_id"],
                name=t.get("team_name") or t.get("username") or f"roster {t['roster_id']}",
                username=t.get("username"),
                is_mine=t["is_mine"],
                alive=t["alive"],
                roster=roster,
                starters=list(t["starters"]),
                reserve=[index[sid] for sid in t["reserve"] if sid in index],
                unknown=unknown,
                faab_left=FAAB_BUDGET - t["faab_used"],
                points_for=t["points_for"],
                points_this_week=t["points_this_week"],
            )
        )
    me = next(i for i, t in enumerate(teams) if t.is_mine)
    on_roster = {i for t in teams for i in t.roster}
    free_agents = [p.index for p in players if p.index not in on_roster]

    transactions = league["transactions"]
    waivers_ran = week == 1 or any(
        tx["type"] == "waiver" and tx["week"] == week for tx in transactions
    )
    return SeasonState(
        week=week,
        fetched_at=league["fetched_at"],
        league_name=league["league_name"],
        players=players,
        teams=teams,
        me=me,
        free_agents=free_agents,
        waivers_ran=waivers_ran,
        transactions=transactions,
        problems=problems,
    )


# --- weekly lineup -------------------------------------------------------------------


def lineup(
    roster: list[int], points: list[float], positions: list[int], w: int
) -> tuple[float, list[tuple[str, int | None]]]:
    """The optimal legal lineup for week index `w`: (total, [(slot, player index)]).

    `points[i]` is player i's projection for the week, `positions[i]` his POS_CODE.
    Slots come back in Sleeper's display order for the week's shape; an unfilled slot
    carries None.
    """
    shape = WEEKLY_SHAPES[w]
    by_pos: list[list[tuple[float, int]]] = [[], [], [], []]
    for i in roster:
        by_pos[positions[i]].append((points[i], -i))
    for col in by_pos:
        col.sort(reverse=True)
    total = 0.0
    slots: list[tuple[str, int | None]] = []
    leftovers: list[tuple[float, int]] = []
    for pos in POSITIONS:
        col = by_pos[POS_CODE[pos]]
        n = shape[pos]
        for k in range(n):
            if k < len(col):
                total += col[k][0]
                slots.append((pos, -col[k][1]))
            else:
                slots.append((pos, None))
        if pos != "QB":
            leftovers.extend(col[n:])
    leftovers.sort(reverse=True)
    for k in range(shape["FLEX"]):
        if k < len(leftovers):
            total += leftovers[k][0]
            slots.append(("FLEX", -leftovers[k][1]))
        else:
            slots.append(("FLEX", None))
    return total, slots


def lineup_points(roster: list[int], points: list[float], positions: list[int], w: int) -> float:
    """`lineup` without the slot list: the race's inner loop."""
    shape = WEEKLY_SHAPES[w]
    qb: list[float] = []
    rb: list[float] = []
    wr: list[float] = []
    te: list[float] = []
    cols = (qb, rb, wr, te)
    for i in roster:
        cols[positions[i]].append(points[i])
    for col in cols:
        col.sort(reverse=True)
    nrb, nwr, nte = shape["RB"], shape["WR"], shape["TE"]
    total = sum(qb[: shape["QB"]]) + sum(rb[:nrb]) + sum(wr[:nwr]) + sum(te[:nte])
    flex = rb[nrb:] + wr[nwr:] + te[nte:]
    if len(flex) > shape["FLEX"]:
        flex.sort(reverse=True)
        return total + sum(flex[: shape["FLEX"]])
    return total + sum(flex)


def thresholds(roster: list[int], points: list[float], positions: list[int], w: int) -> tuple[float, float, float, float]:
    """Per position, the points a free agent must beat to enter the week-`w` lineup: the
    weaker of the last dedicated starter and the last flex starter (0 for an empty seat).
    A player above his position's threshold adds exactly (his points - threshold)."""
    shape = WEEKLY_SHAPES[w]
    cols: list[list[float]] = [[], [], [], []]
    for i in roster:
        cols[positions[i]].append(points[i])
    for col in cols:
        col.sort(reverse=True)
    nq, nrb, nwr, nte, nflex = shape["QB"], shape["RB"], shape["WR"], shape["TE"], shape["FLEX"]
    qb_last = cols[0][nq - 1] if len(cols[0]) >= nq else 0.0
    flex = cols[1][nrb:] + cols[2][nwr:] + cols[3][nte:]
    flex.sort(reverse=True)
    flex_last = flex[nflex - 1] if len(flex) >= nflex else 0.0

    def last(col: list[float], n: int) -> float:
        return min(col[n - 1] if len(col) >= n else 0.0, flex_last)

    return qb_last, last(cols[1], nrb), last(cols[2], nwr), last(cols[3], nte)
