"""In-season state: the player universe, the 32 rosters, and the weekly lineup solver.

Value input for the season is per-week and per-player: DraftSharks' weekly projection
(pool/data/weekly_projections.json, joined to Sleeper ids through pool.json, or by name
against league.json's directory for a player outside the draft pool) blended 2:1 with
Sleeper's weekly projection for the same week (league.json), the same weighting the
draft used on season totals. A player only Sleeper projects carries Sleeper's number
alone. Weeks already played are zero: nothing in the season model looks backwards.

The weekly lineup is the greedy optimum: dedicated slots take each position's best
bodies, the flex seats take the best pooled RB/WR/TE leftovers. That greedy fill is
exact for this slot chain (league.SLOT_CHAIN), so the current week's recommendation is
the true optimum under the projections. The race simulator scores every team-week the
same way, with surprise inactives left to the score noise rather than the availability
cascade the draft valuation used — the race runs tens of thousands of team-weeks and the
cascade is too slow for that.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .league import (
    FAAB_BUDGET,
    POSITIONS,
    RESERVE_SLOTS,
    RESERVE_STATUSES,
    WAIVER_CLEAR_DAYS,
    WAIVER_CLEAR_HOURS,
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
    ir_until: int  # first week index he needs a regular roster spot; reserve-eligible before it


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
    # After the weekly run: free agents dropped since who are still on waivers -> when they clear
    on_waivers: dict[int, str] = field(default_factory=dict)

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


_SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")


def _name_key(name: str) -> str:
    """build_pool.py's normalization: lowercase alphanumerics, generational suffix dropped."""
    words = [w for w in (re.sub(r"[^a-z0-9]", "", part.lower()) for part in name.split()) if w]
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    return "".join(words)


def draftsharks_by_sleeper(pool: dict, weekly_raw: dict, directory: dict) -> dict[str, dict[str, float]]:
    """DraftSharks weekly rows keyed by sleeper id: the pool's verified join first, then a
    row outside the pool (a backup who became a starter after the pool was built) joins on
    normalized name and position when the directory holds exactly one such player."""
    sleeper_of_ds = {p["player_id"]: p["sleeper_id"] for p in pool["players"] if p.get("sleeper_id")}
    by_name: dict[tuple[str, str | None], list[str]] = {}
    for sid, meta in directory.items():
        by_name.setdefault((_name_key(meta["name"]), meta.get("position")), []).append(sid)
    out: dict[str, dict[str, float]] = {}
    unpooled = []
    for row in weekly_raw["players"]:
        sid = sleeper_of_ds.get(row["player_id"])
        if sid is None:
            unpooled.append(row)
        else:
            out[sid] = {w: v["points"] for w, v in row["weeks"].items()}
    for row in unpooled:
        found = by_name.get((_name_key(row["name"]), row["position"]), [])
        if len(found) == 1 and found[0] not in out:
            out[found[0]] = {w: v["points"] for w, v in row["weeks"].items()}
    return out


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
    if league["reserve_slots"] != RESERVE_SLOTS:
        problems.append(f"Sleeper says there are {league['reserve_slots']} reserve slots, league.py {RESERVE_SLOTS}")
    if league["waiver_clear_days"] != WAIVER_CLEAR_DAYS:
        problems.append(f"Sleeper says waivers clear after {league['waiver_clear_days']} day(s), league.py "
                        f"{WAIVER_CLEAR_DAYS}; re-observe WAIVER_CLEAR_HOURS")

    directory = league["players"]
    ds_weekly = draftsharks_by_sleeper(pool, weekly_raw, directory)
    sl_weekly = league["projections"]  # week -> sleeper_id -> points

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
        ir_until = w0
        if meta.get("injury_status") in RESERVE_STATUSES:
            # Reserve-eligible while his projection stays zero; a stale Out label on a
            # player projected to play opens no slot.
            ir_until = next((w for w in range(w0, WEEKS) if weekly[w] > 0), WEEKS)
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
                ir_until=ir_until,
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
        tx["type"] == "waiver" and tx["week"] == week and tx["status"] in ("complete", "failed")
        for tx in transactions
    )
    free = set(free_agents)
    # Before the weekly run every free agent is in it; after, only the recently dropped need a claim.
    on_waivers = {
        index[sid]: clears for sid, clears in waiver_clears(transactions, league["fetched_at"]).items()
        if sid in index and index[sid] in free
    } if waivers_ran else {}
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
        on_waivers=on_waivers,
    )


def waiver_clears(transactions: list[dict], fetched_at: str) -> dict[str, str]:
    """Sleeper id -> when he clears waivers (UTC), for every player whose latest drop is
    still inside the clearing window at `fetched_at`."""
    dropped: dict[str, dt.datetime] = {}
    for tx in transactions:
        if tx["status"] == "complete":
            at = dt.datetime.fromisoformat(tx["processed_at"])
            for sid in tx["drops"]:
                dropped[sid] = max(dropped.get(sid, at), at)
    now = dt.datetime.fromisoformat(fetched_at)
    window = dt.timedelta(hours=WAIVER_CLEAR_HOURS)
    return {sid: (at + window).isoformat(timespec="minutes") for sid, at in dropped.items() if at + window > now}


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


def _columns(roster: list[int], points: list[float], positions: list[int]) -> list[list[float]]:
    cols: list[list[float]] = [[], [], [], []]
    for i in roster:
        cols[positions[i]].append(points[i])
    for col in cols:
        col.sort(reverse=True)
    return cols


def _solve(cols: list[list[float]], shape: dict[str, int]) -> tuple[float, tuple[float, float, float, float], float]:
    """(lineup total, per-position entry thresholds, last flex starter) from sorted
    position columns."""
    nq, nrb, nwr, nte, nflex = shape["QB"], shape["RB"], shape["WR"], shape["TE"], shape["FLEX"]
    flex = cols[1][nrb:] + cols[2][nwr:] + cols[3][nte:]
    flex.sort(reverse=True)
    flex_last = flex[nflex - 1] if len(flex) >= nflex else 0.0
    total = sum(cols[0][:nq]) + sum(cols[1][:nrb]) + sum(cols[2][:nwr]) + sum(cols[3][:nte]) + sum(flex[:nflex])

    def last(col: list[float], n: int) -> float:
        return min(col[n - 1] if len(col) >= n else 0.0, flex_last)

    qb_last = cols[0][nq - 1] if len(cols[0]) >= nq else 0.0
    return total, (qb_last, last(cols[1], nrb), last(cols[2], nwr), last(cols[3], nte)), flex_last


def thresholds(roster: list[int], points: list[float], positions: list[int], w: int) -> tuple[float, float, float, float]:
    """Per position, the points a free agent must beat to enter the week-`w` lineup: the
    weaker of the last dedicated starter and the last flex starter (0 for an empty seat).
    A player above his position's threshold adds exactly (his points - threshold)."""
    return _solve(_columns(roster, points, positions), WEEKLY_SHAPES[w])[1]


def lineup_without_each(
    roster: list[int], points: list[float], positions: list[int], w: int
) -> tuple[float, tuple[float, float, float, float], dict[int, tuple[float, tuple[float, float, float, float]]]]:
    """The week-`w` lineup total and thresholds of `roster`, and for each body the pair
    without him. Removing a body outside the lineup changes neither, so only starters
    (and bodies tied with one) are recomputed: the bidding context's hot path."""
    shape = WEEKLY_SHAPES[w]
    cols = _columns(roster, points, positions)
    total, floors, flex_last = _solve(cols, shape)
    counts = (shape["QB"], shape["RB"], shape["WR"], shape["TE"])
    dedicated = [col[n - 1] if len(col) >= n else 0.0 for col, n in zip(cols, counts)]
    without = {}
    for i in roster:
        pos, val = positions[i], points[i]
        if val < dedicated[pos] and (pos == 0 or val < flex_last):
            without[i] = (total, floors)
            continue
        rest = list(cols)
        rest[pos] = cols[pos][:]
        rest[pos].remove(val)
        without[i] = _solve(rest, shape)[:2]
    return total, floors, without
