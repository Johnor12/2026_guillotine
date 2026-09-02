"""Guillotine-weighted expected lineup points and waiver-wire levels.

A roster is valued week by week: for each of the league's 17 weeks, the expectation,
over position-wide Bernoulli availability, of the best legal lineup under that week's
starting shape (lineups expand in-season — league.WEEKLY_SHAPES) using that week's
projections (byes and known absences are zero weeks). The weekly optimum decomposes
exactly: each position fills its dedicated slots with its best available bodies, then
the FLEX seats take the best pooled RB/WR/TE leftovers. A QB never takes a flex seat;
the week-14+ superflex is modeled as a second dedicated QB slot. Each expectation is
closed form — a Bernoulli cascade per position for dedicated slots, a layer-cake
integral of the pooled marginal-count law for the flex seats — and --selftest checks
single weeks against brute-force enumeration of every availability subset.

The 17 weekly values are combined by `Levels.weights`, the converged guillotine week
weights: each regular-season week's weight is the marginal effect of a weekly point on
log P(surviving that week's cut), and the championship weeks' weight is its effect on
log P(winning the week 16-17 final) (see guillotine.py). Weights are normalized to sum
to 1, so a roster value reads as guillotine-weighted expected weekly lineup points.
Scaling never changes any argmax, so the pick policy is unaffected by normalization.

The waiver wire is week-specific: `Levels.wire` holds, per position per week, the
bodies a team could sign that week for roughly nothing. The free-agent pool is the
undrafted tail plus every eliminated roster so far (`Levels.dropped`, measured by
guillotine.py), and it is shared: with S teams alive, the top S bodies at a position
are one claim each, the next S the second claim, so a survivor's j-th waiver body is
the mean of the j-th tier. One good free agent is worth a thirty-second of himself in
week 1, not a starter for all thirty-two rosters. Each in-season add a surviving roster
holds contributes one such body (league.WEEK_WIRE_BODIES — one per position early, more
as the roster expands over half-waiver late rosters); a body can fill one lineup job,
never several simultaneous holes.

That equal split is the league wire (`Levels.league_wire`): it values the opponents and
so sets the elimination bars. My own roster is valued against `Levels.wire`, the same
pool under my FAAB policy (league.FAAB_HOLD_WEEKS / FAAB_SPEND_WEEK): the undrafted
tail alone while I hold the budget, the equal split through the lineup expansion, and
the top half of every tier once the saved budget is spent.

This is one objective with one set of units: guillotine-weighted expected weekly lineup
points. There is no role threshold and no separate bench bonus, and roster value stays
monotone when a projection improves or a player is added.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache

from . import league
from .league import (
    POSITIONS,
    REGULAR_WEEKS,
    SLOT_CHAIN,
    STARTING_SLOTS,
    UNAVAILABLE_RATE,
    WEEK_WIRE_BODIES,
    WEEKLY_SHAPES,
    WEEKS,
)
from .pool import Player, by_position

# One sort key, bound once: the lineup solver runs inside every marginal-value
# evaluation, so per-call lambda construction is worth avoiding.
_KEY = lambda q: (-q.points, q.player_id)  # noqa: E731

# Per-position dedicated-slot counts and flex seats by week, unpacked once.
WEEK_DEDICATED = {pos: tuple(s[pos] for s in WEEKLY_SHAPES) for pos in POSITIONS}
WEEK_FLEX = tuple(s["FLEX"] for s in WEEKLY_SHAPES)

# How many undrafted players per position the weekly wire scan reads. A week's tiers
# need at most survivors x bodies = 48 values, and weekly profiles are near-flat, so
# the top of the position's season order holds them; 64 leaves margin for bye weeks.
_WIRE_SCAN = 64

@dataclass(frozen=True, slots=True)
class Levels:
    """The converged league levels a roster is valued against.

    weights: guillotine week weights, length WEEKS, normalized to sum to 1.
    wire: per POSITIONS-index, per week, the tuple of waiver-body point values my
    roster holds that week under my FAAB policy (WEEK_WIRE_BODIES bodies, best first —
    each marginal add is a worse player, so the tiers decay).
    league_wire: the same, as the survivors' equal split of the pool — what an
    opponent holds, and so what the elimination bars are measured against.
    dropped: per POSITIONS-index, per week, the eliminated rosters' best weekly values
    (`pool_size` of them, best first, zero-padded) — kept apart from the undrafted
    tail so a rollout can re-measure that tail from its own final board and re-pool
    it with these (`refresh_wire`).
    """

    weights: tuple[float, ...]
    wire: tuple[tuple[float, ...], ...]
    league_wire: tuple[tuple[float, ...], ...]
    dropped: tuple[tuple[float, ...], ...]


def pos_sorted(players: list[Player]) -> dict[str, list[Player]]:
    """Per-position lists, each sorted by points descending."""
    return {k: sorted(v, key=_KEY) for k, v in by_position(players).items()}


def sorted_roster(roster: list[Player]) -> list[Player]:
    """A roster pre-sorted by season points — the form `team_value` consumes.

    The weekly solver canonicalizes per position internally, so the order is a
    stable container convention rather than a correctness requirement.
    """
    return sorted(roster, key=_KEY)


def insert_sorted(seq: list[Player], p: Player) -> list[Player]:
    """A new list with `p` merged into an already-sorted list."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        q = seq[mid]
        if q.points > p.points or (q.points == p.points and q.player_id < p.player_id):
            lo = mid + 1
        else:
            hi = mid
    out = seq.copy()
    out.insert(lo, p)
    return out


# --- single-week closed forms -----------------------------------------------------


@lru_cache(maxsize=131_072)
def _position_expected_values(
    projections: tuple[tuple[int, float], ...],
    wire_bodies: tuple[float, ...],
    unavailable: float,
    max_starters: int,
) -> tuple[float, ...]:
    """All starter-count values for one week's depth chart; cached across branches."""
    available = 1.0 - unavailable
    entries = [
        (points / available, points, available, player_id)
        for player_id, points in projections
    ]
    # Actual free agents, not an unlimited scalar that can fill every open slot: one
    # body per waiver add the roster holds that week, each tier a worse player.
    entries.extend(
        (points, points, 1.0, -1 - i) for i, points in enumerate(wire_bodies)
    )
    entries.sort(key=lambda row: (-row[0], row[3]))

    out = [0.0]
    for starter_count in range(1, max_starters + 1):
        # dist[k] = P(exactly k higher bodies are active), with the final cell
        # collecting saturated states that cannot leave a job for a lower body.
        dist = [1.0] + [0.0] * starter_count
        total = 0.0
        for _, expected_points, active_probability, _ in entries:
            total += expected_points * sum(dist[:starter_count])
            next_dist = [0.0] * (starter_count + 1)
            for active_higher, probability in enumerate(dist):
                next_dist[active_higher] += probability * (1.0 - active_probability)
                next_dist[min(active_higher + 1, starter_count)] += (
                    probability * active_probability
                )
            dist = next_dist
        out.append(total)
    return tuple(out)


def position_expected_value(
    projections: list[tuple[int, float]],
    wire_bodies: tuple[float, ...],
    unavailable: float,
    starter_count: int,
) -> float:
    """Expected points from one position-week with a unique always-available wire body.

    Projections are unconditional (player_id, points) pairs. Dividing by availability
    orders players by their points when active; multiplying that active rate by their
    own availability returns the original projection. The running Bernoulli
    distribution supplies the probability that fewer than ``starter_count`` higher
    bodies are active, which is the probability this depth-chart entry is called on.
    """
    return _position_expected_values(
        tuple(sorted(projections)), wire_bodies, unavailable, starter_count
    )[starter_count]


@lru_cache(maxsize=1024)
def _binomial_pmf(k: int, p: float) -> tuple[float, ...]:
    """P(j of k iid bodies are available), j = 0..k."""
    pmf = [1.0]
    for _ in range(k):
        nxt = [0.0] * (len(pmf) + 1)
        for j, prob in enumerate(pmf):
            nxt[j] += prob * (1.0 - p)
            nxt[j + 1] += prob * p
        pmf = nxt
    return tuple(pmf)


@lru_cache(maxsize=131_072)
def _extra_count_table(
    projections: tuple[tuple[int, float], ...],
    wire_bodies: tuple[float, ...],
    unavailable: float,
    dedicated: int,
    cap: int,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Piecewise law of this position's marginal-body count above a value threshold.

    A marginal body is an available body ranked below the position's top-`dedicated`
    available ones — what a flex seat could use. Only bodies sorted at index >=
    `dedicated` can ever be marginal, so the law changes only at their values.
    Returns (breaks, rows), breaks ascending: rows[i] applies for thresholds in
    [breaks[i-1] (or 0), breaks[i]) and gives P(count = 0..cap), last cell >= cap.
    Past the last break the count is surely zero (omitted).
    """
    available = 1.0 - unavailable
    ws = [points / available for _, points in projections if points > 0]
    ws.extend(wire_bodies)  # the always-available waiver bodies, tiered
    ws.sort(reverse=True)
    breaks = sorted({w for w in ws[dedicated:] if w > 0})
    rows = []
    for b in breaks:
        wire_above = sum(1 for points in wire_bodies if points >= b)
        players_above = sum(1 for w in ws if w >= b) - wire_above
        row = [0.0] * (cap + 1)
        for j, prob in enumerate(_binomial_pmf(players_above, available)):
            row[min(max(j + wire_above - dedicated, 0), cap)] += prob
        rows.append(tuple(row))
    return tuple(breaks), tuple(rows)


@lru_cache(maxsize=262_144)
def _flex_expected_value(flex_tables: tuple[tuple, ...], seats: int) -> float:
    """E[points from the week's FLEX seats], by a layer-cake integral over thresholds.

    Weekly, those seats take the top-`seats` pooled RB/WR/TE marginals. With N(>t)
    the pooled marginal count — independent across positions, so a small capped
    convolution:
        E[top-F sum] = integral of E[min(F, N(>t))] dt
                     = integral of F - sum_{k<F} (F-k) * P(N=k) dt
    The integrand is piecewise constant between body values and zero past the last.
    Each position's rows carry cells 0..seats (last cell = ">= seats"), so only the
    pooled cells below `seats` are ever needed.
    """
    (a_brks, a_rows), (b_brks, b_rows), (c_brks, c_rows) = flex_tables
    all_breaks = sorted({*a_brks, *b_brks, *c_brks})
    na, nb, nc = len(a_brks), len(b_brks), len(c_brks)
    ia = ib = ic = 0
    one = (1.0,) + (0.0,) * seats
    total = 0.0
    prev = 0.0
    for x in all_breaks:
        while ia < na and a_brks[ia] < x:
            ia += 1
        while ib < nb and b_brks[ib] < x:
            ib += 1
        while ic < nc and c_brks[ic] < x:
            ic += 1
        arow = a_rows[ia] if ia < na else one
        brow = b_rows[ib] if ib < nb else one
        crow = c_rows[ic] if ic < nc else one
        expected = float(seats)
        for k in range(seats):
            pooled = 0.0
            for i in range(k + 1):
                for j in range(k - i + 1):
                    pooled += arow[i] * brow[j] * crow[k - i - j]
            expected -= (seats - k) * pooled
        total += (x - prev) * expected
        prev = x
    return total


def week_value(
    week_projections: dict[str, list[tuple[int, float]]],
    wire: dict[str, tuple[float, ...]],
    shape: dict[str, int],
) -> float:
    """One week's expected optimal lineup value — the unit --selftest brute-forces."""
    total = 0.0
    tables = {}
    for pos in POSITIONS:
        projections = tuple(sorted(week_projections.get(pos, [])))
        dedicated = shape[pos]
        total += _position_expected_values(
            projections, wire[pos], UNAVAILABLE_RATE[pos], dedicated
        )[dedicated]
        if pos != "QB":
            tables[pos] = _extra_count_table(
                projections, wire[pos], UNAVAILABLE_RATE[pos], dedicated, shape["FLEX"]
            )
    return total + _flex_expected_value(
        (tables["RB"], tables["WR"], tables["TE"]), shape["FLEX"]
    )


# --- whole-season valuation --------------------------------------------------------


@lru_cache(maxsize=65_536)
def _position_week_tables(
    entries: tuple[tuple[int, tuple[float, ...]], ...],
    wire_col: tuple[tuple[float, ...], ...],
    pos: str,
) -> tuple[tuple[float, ...], tuple]:
    """A position's dedicated values and flex tables for every week, cached as a unit.

    `entries` is the depth chart as (player_id, weekly points) sorted by id — a
    canonical key, so candidate branches that reach the same chart share the work.
    Zero weeks (bye, absence) drop out of that week's projections entirely.
    """
    unavailable = UNAVAILABLE_RATE[pos]
    dedicated = WEEK_DEDICATED[pos]
    ded = []
    tables = []
    for w in range(WEEKS):
        week_proj = tuple(
            (pid, weeks[w]) for pid, weeks in entries if weeks[w] > 0.0
        )
        d = dedicated[w]
        ded.append(
            _position_expected_values(week_proj, wire_col[w], unavailable, d)[d]
        )
        tables.append(
            None
            if pos == "QB"
            else _extra_count_table(
                week_proj, wire_col[w], unavailable, d, WEEK_FLEX[w]
            )
        )
    return tuple(ded), tuple(tables)


def _entries(players: list[Player]) -> tuple[tuple[int, tuple[float, ...]], ...]:
    return tuple(sorted((p.player_id, p.weekly) for p in players))


def weekly_team_values(roster: list[Player], levels: Levels) -> tuple[float, ...]:
    """Expected best lineup points for each league week, in closed form."""
    by_pos = {pos: [] for pos in POSITIONS}
    for player in roster:
        by_pos[player.position].append(player)
    ded = {}
    tables = {}
    for i, pos in enumerate(POSITIONS):
        ded[pos], tables[pos] = _position_week_tables(
            _entries(by_pos[pos]), levels.wire[i], pos
        )
    out = []
    for w in range(WEEKS):
        out.append(
            ded["QB"][w]
            + ded["RB"][w]
            + ded["WR"][w]
            + ded["TE"][w]
            + _flex_expected_value(
                (tables["RB"][w], tables["WR"][w], tables["TE"][w]), WEEK_FLEX[w]
            )
        )
    return tuple(out)


def starting_positions(positions: list[str]) -> list[str]:
    """Which of these rostered positions fill the week-1 slots when a team must field
    everyone it can. Validation uses this to check every simulated roster, off-pool
    picks included, can field a full opening lineup."""
    caps = dict(STARTING_SLOTS)
    out: list[str] = []
    for position in positions:
        for slot in SLOT_CHAIN[position]:
            if caps[slot]:
                caps[slot] -= 1
                out.append(position)
                break
    return out


def team_value(
    roster_sorted: list[Player],
    levels: Levels,
    extra: Player | None = None,
) -> float:
    """Guillotine-weighted expected weekly lineup points for a roster."""
    roster = roster_sorted if extra is None else [*roster_sorted, extra]
    weekly = weekly_team_values(roster, levels)
    weights = levels.weights
    return sum(weights[w] * weekly[w] for w in range(WEEKS))


def team_values_with_candidates(
    roster: list[Player],
    levels: Levels,
    candidates: list[Player],
) -> tuple[float, dict[int, float]]:
    """Roster value and each one-player addition, sharing the unchanged positions.

    A candidate alters one position's depth charts. Computing the other three
    positions' weekly tables once avoids rebuilding them for every option.
    """
    by_pos = {pos: [] for pos in POSITIONS}
    for player in roster:
        by_pos[player.position].append(player)

    base_entries = {pos: _entries(by_pos[pos]) for pos in POSITIONS}
    ded = {}
    tables = {}
    for i, pos in enumerate(POSITIONS):
        ded[pos], tables[pos] = _position_week_tables(
            base_entries[pos], levels.wire[i], pos
        )
    ded_sum = [sum(ded[pos][w] for pos in POSITIONS) for w in range(WEEKS)]
    base_flex = [
        _flex_expected_value(
            (tables["RB"][w], tables["WR"][w], tables["TE"][w]), WEEK_FLEX[w]
        )
        for w in range(WEEKS)
    ]
    weights = levels.weights
    baseline = sum(
        weights[w] * (ded_sum[w] + base_flex[w]) for w in range(WEEKS)
    )

    values: dict[int, float] = {}
    for candidate in candidates:
        pos = candidate.position
        cand_entries = tuple(
            sorted(base_entries[pos] + ((candidate.player_id, candidate.weekly),))
        )
        cded, ctables = _position_week_tables(
            cand_entries, levels.wire[POSITIONS.index(pos)], pos
        )
        total = 0.0
        for w in range(WEEKS):
            week_total = ded_sum[w] - ded[pos][w] + cded[w]
            if pos == "QB":
                week_total += base_flex[w]
            else:
                merged = {**{p: tables[p][w] for p in ("RB", "WR", "TE")}, pos: ctables[w]}
                week_total += _flex_expected_value(
                    (merged["RB"], merged["WR"], merged["TE"]), WEEK_FLEX[w]
                )
            total += weights[w] * week_total
        values[candidate.player_id] = total
    return baseline, values


# --- wire levels --------------------------------------------------------------------


def wire_replacement(
    taken: set[int], pos: dict[str, list[Player]]
) -> dict[str, float]:
    """The best player at each position actually left undrafted, in season points.

    Reporting and convergence tracing only; valuation uses the per-week wire.
    """
    out: dict[str, float] = {}
    for position, players in pos.items():
        out[position] = max(
            (p.points for p in players if p.player_id not in taken), default=0.0
        )
    return out


def survivors(w: int) -> int:
    """Teams alive entering week w+1: cut rounds 1..min(w, 15) have run."""
    return league.TEAMS - 2 * min(w, REGULAR_WEEKS)


def pool_size(pos: str, w: int) -> int:
    """Free agents that can matter in a week: one tier of survivors per waiver body."""
    return survivors(w) * WEEK_WIRE_BODIES[pos][w]


def top_values(values: Iterable[float], pos: str, w: int) -> tuple[float, ...]:
    """The `pool_size` largest values, best first, zero-padded to that length."""
    best = heapq.nlargest(pool_size(pos, w), values)
    best.extend([0.0] * (pool_size(pos, w) - len(best)))
    return tuple(best)


def tier_bodies(
    values: tuple[float, ...], pos: str, w: int, share: float = 1.0
) -> tuple[float, ...]:
    """A survivor's waiver bodies from a free-agent pool: body j is the mean of the
    j-th tier of S values, S the teams alive, so the pool is shared rather than
    handed whole to every roster. `share` < 1 reads the mean of only the best
    fraction of each tier — the claims a bigger budget wins."""
    s = survivors(w)
    best = top_values(values, pos, w)
    n = max(1, round(s * share))
    return tuple(
        sum(best[j * s : j * s + n]) / n for j in range(WEEK_WIRE_BODIES[pos][w])
    )


def weekly_undrafted(
    taken: set[int], pos: dict[str, list[Player]]
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Per position, per week: the undrafted tail's best weekly points, `pool_size`
    of them, best first.

    Reads the top-_WIRE_SCAN undrafted players by season points — deeper players
    cannot hold a weekly tier under near-flat weekly profiles, and the bound keeps
    rollout playouts cheap.
    """
    out = []
    for position in POSITIONS:
        scanned: list[tuple[float, ...]] = []
        for p in pos[position]:
            if p.player_id in taken:
                continue
            scanned.append(p.weekly)
            if len(scanned) == _WIRE_SCAN:
                break
        out.append(
            tuple(
                top_values((weeks[w] for weeks in scanned), position, w)
                for w in range(WEEKS)
            )
        )
    return tuple(out)


def combine_wire(
    undrafted: tuple[tuple[tuple[float, ...], ...], ...],
    dropped: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """One free-agent pool per position-week — undrafted tail plus eliminated
    rosters — tiered across the survivors: the league wire."""
    return tuple(
        tuple(
            tier_bodies(u + d, position, w)
            for w, (u, d) in enumerate(zip(undrafted_col, dropped_col))
        )
        for position, undrafted_col, dropped_col in zip(POSITIONS, undrafted, dropped)
    )


def my_wire(
    undrafted: tuple[tuple[tuple[float, ...], ...], ...],
    dropped: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """The same pool under my FAAB policy: no claims on the eliminated rosters while
    I hold the budget, the equal split through the lineup expansion, the top half of
    every tier once the saved budget is spent (league.FAAB_HOLD_WEEKS/FAAB_SPEND_WEEK)."""

    def bodies(u, d, position, w):
        if w < league.FAAB_HOLD_WEEKS:
            return tier_bodies(u, position, w)
        if w < league.FAAB_SPEND_WEEK - 1:
            return tier_bodies(u + d, position, w)
        return tier_bodies(u + d, position, w, share=0.5)

    return tuple(
        tuple(
            bodies(u, d, position, w)
            for w, (u, d) in enumerate(zip(undrafted_col, dropped_col))
        )
        for position, undrafted_col, dropped_col in zip(POSITIONS, undrafted, dropped)
    )


def refresh_wire(
    levels: Levels, taken: set[int], pos: dict[str, list[Player]]
) -> Levels:
    """Levels with the undrafted tail re-measured from a specific final board.

    Rollout playouts end with different players undrafted than the converged draft
    did; the eliminated rosters and week weights are slow league-level quantities
    and carry over unchanged. Only my wire is refreshed: the league wire prices
    opponents, whom a playout never values.
    """
    return replace(levels, wire=my_wire(weekly_undrafted(taken, pos), levels.dropped))


def seed_levels(players: list[Player]) -> Levels:
    """Iteration-0 levels from pure slot counting, no draft behaviour assumed.

    Assign the top of the pool to the league's opening starting slots the way a
    perfectly efficient market would — dedicated slots by positional rank, then the
    flex slots to the best players still eligible — and read the weekly wire off the
    players who did not earn a slot. Week weights seed flat across the regular season
    with the championship at half weight; convergence replaces both with measured
    levels (guillotine.py).
    """
    caps = {slot: n * league.TEAMS for slot, n in STARTING_SLOTS.items()}
    assigned: set[int] = set()
    for p in sorted(players, key=_KEY):
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot]:
                caps[slot] -= 1
                assigned.add(p.player_id)
                break
    raw = [1.0] * REGULAR_WEEKS + [0.5] * (WEEKS - REGULAR_WEEKS)
    total = sum(raw)
    nothing_dropped = tuple(
        tuple(top_values((), pos, w) for w in range(WEEKS)) for pos in POSITIONS
    )
    undrafted = weekly_undrafted(assigned, pos_sorted(players))
    return Levels(
        weights=tuple(w / total for w in raw),
        wire=my_wire(undrafted, nothing_dropped),
        league_wire=combine_wire(undrafted, nothing_dropped),
        dropped=nothing_dropped,
    )
