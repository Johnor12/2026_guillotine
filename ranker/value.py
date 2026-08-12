"""Expected lineup points and wire-level measurement.

A roster is valued as the expectation, over position-wide Bernoulli availability, of the
best legal lineup each week: the composition is re-chosen per availability draw, so a
flex job vacated by an unavailable RB can be refilled by the best remaining body at any
flex position (expectation-of-max, not max-of-expectations — the latter overvalues depth
behind many locked slots).

The weekly optimum decomposes exactly: each position fills its dedicated slots with its
best available bodies, then the FLEX seats go to the best pooled RB/WR/TE leftovers.
The QB slot is dedicated only — no flex seat can take a second QB. With F flex seats:
    dedicated tops  +  top-F pooled non-QB marginals.
Each expectation is computed in closed form: the dedicated term by a small Bernoulli
cascade per position, the pooled term by a layer-cake integral of the marginal-count
distribution over value thresholds (positions are independent, so the pooled count is a
tiny convolution). The best waiver player at each position is inserted once as an
always-available body, so one free agent can fill one lineup job, never several
simultaneous holes.

This is one objective with one set of units: expected lineup points. There is no role
threshold and no separate bench bonus. A better projection can retain every role a worse
projection could fill, which makes roster value monotone when a player improves, is
replaced by a better same-position player, or is simply added.
"""

from __future__ import annotations

from functools import lru_cache

from .league import (
    DEDICATED_SLOTS,
    POSITIONS,
    SLOT_CHAIN,
    STARTING_SLOTS,
    TEAMS,
    UNAVAILABLE_RATE,
)
from .pool import Player, by_position

# One sort key, bound once: the lineup solver runs inside every marginal-value
# evaluation, so per-call lambda construction is worth avoiding.
_KEY = lambda q: (-q.points, q.player_id)  # noqa: E731


def pos_sorted(players: list[Player]) -> dict[str, list[Player]]:
    """Per-position lists, each sorted by points descending."""
    return {k: sorted(v, key=_KEY) for k, v in by_position(players).items()}


def sorted_roster(roster: list[Player]) -> list[Player]:
    """A roster pre-sorted by points — the form `team_value` consumes.

    Valuing a candidate means re-valuing the roster with him added, thousands of times
    per pick, and re-sorting to place 1 player was the single largest cost in the
    simulation. So the roster is sorted once per pick and each candidate is merged in
    by `insert_sorted`.
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


# --- expected lineup solver -------------------------------------------------------

# The position-flexible starting seats beyond the dedicated slots. The unrolled
# integrand in `_extra_expected_value` assumes exactly two flex seats and no superflex.
_FLEX_SLOTS = STARTING_SLOTS["FLEX"]
assert _FLEX_SLOTS == 2 and "SF" not in STARTING_SLOTS


@lru_cache(maxsize=32_768)
def _position_expected_values(
    projections: tuple[tuple[int, float], ...],
    wire_points: float,
    unavailable: float,
    max_starters: int,
) -> tuple[float, ...]:
    """All starter-count values for one depth chart; cached across candidate branches."""
    available = 1.0 - unavailable
    entries = [
        (points / available, points, available, player_id)
        for player_id, points in projections
    ]
    # One actual free agent, not an unlimited scalar that can fill every open slot.
    entries.append((wire_points, wire_points, 1.0, -1))
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
    players: list[Player], wire_points: float, starter_count: int
) -> float:
    """Expected points from one position with a unique, always-available wire body.

    Player projections are unconditional. Dividing by availability orders players by
    their points when active; multiplying that active rate by their own availability
    returns the original projection. The running Bernoulli distribution supplies the
    probability that fewer than ``starter_count`` higher bodies are active, which is the
    probability this depth-chart entry is called on.
    """
    unavailable = UNAVAILABLE_RATE[players[0].position] if players else 0.0
    projections = tuple(sorted((p.player_id, p.points) for p in players))
    return _position_expected_values(
        projections, wire_points, unavailable, starter_count
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


@lru_cache(maxsize=32_768)
def _extra_count_table(
    projections: tuple[tuple[int, float], ...],
    wire_points: float,
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
    ws.append(wire_points)  # the unique always-available wire body
    ws.sort(reverse=True)
    breaks = sorted({w for w in ws[dedicated:] if w > 0})
    wire_w = wire_points
    rows = []
    for b in breaks:
        players_above = sum(1 for w in ws if w >= b) - (1 if wire_w >= b else 0)
        wire_above = 1 if wire_w >= b else 0
        row = [0.0] * (cap + 1)
        for j, prob in enumerate(_binomial_pmf(players_above, available)):
            row[min(max(j + wire_above - dedicated, 0), cap)] += prob
        rows.append(tuple(row))
    return tuple(breaks), tuple(rows)


@lru_cache(maxsize=65_536)
def _extra_expected_value(flex_tables: tuple[tuple, ...]) -> float:
    """E[points from the two FLEX seats], by a layer-cake integral over thresholds.

    Weekly, those seats take the top-2 pooled RB/WR/TE marginals. With N(>t) the pooled
    marginal count — independent across positions, so a small capped convolution:
        E[top-2 sum] = integral of E[min(2, N(>t))]
    The integrand is piecewise constant between body values and zero past the last.
    The convolution is unrolled for this league's two seats: only pooled cells 0-1 are
    needed, since E[min(2, N)] = p1 + 2(1 - p0 - p1) = 2 - 2*p0 - p1.
    """
    (a_brks, a_rows), (b_brks, b_rows), (c_brks, c_rows) = flex_tables
    all_breaks = sorted({*a_brks, *b_brks, *c_brks})
    na, nb, nc = len(a_brks), len(b_brks), len(c_brks)
    ia = ib = ic = 0
    one = (1.0, 0.0, 0.0)
    total = 0.0
    prev = 0.0
    for x in all_breaks:
        while ia < na and a_brks[ia] < x:
            ia += 1
        while ib < nb and b_brks[ib] < x:
            ib += 1
        while ic < nc and c_brks[ic] < x:
            ic += 1
        a0, a1, _ = a_rows[ia] if ia < na else one
        b0, b1, _ = b_rows[ib] if ib < nb else one
        c0, c1, _ = c_rows[ic] if ic < nc else one
        p0 = a0 * b0 * c0
        p1 = a0 * b0 * c1 + a0 * b1 * c0 + a1 * b0 * c0
        total += (x - prev) * (2.0 - 2.0 * p0 - p1)
        prev = x
    return total


def _position_tables(
    projections: tuple[tuple[int, float], ...], wire_points: float, pos: str
) -> tuple[float, tuple | None]:
    """A position's two closed-form pieces: dedicated-slot value and marginal law.

    A QB can never take a flex seat, so its marginal law is not needed.
    """
    dedicated = DEDICATED_SLOTS[pos]
    unavailable = UNAVAILABLE_RATE[pos]
    ded_value = _position_expected_values(
        projections, wire_points, unavailable, dedicated
    )[dedicated]
    if pos == "QB":
        return ded_value, None
    table = _extra_count_table(
        projections, wire_points, unavailable, dedicated, _FLEX_SLOTS
    )
    return ded_value, table


def expected_lineup_value(roster: list[Player], wire: dict[str, float]) -> float:
    """Expected best weekly lineup over availability draws, in closed form."""
    by_pos = {pos: [] for pos in POSITIONS}
    for player in roster:
        by_pos[player.position].append(player)
    total = 0.0
    tables = {}
    for pos in POSITIONS:
        projections = tuple(sorted((p.player_id, p.points) for p in by_pos[pos]))
        ded_value, tables[pos] = _position_tables(projections, wire[pos], pos)
        total += ded_value
    return total + _extra_expected_value((tables["RB"], tables["WR"], tables["TE"]))


def starting_positions(roster: list[Player]) -> list[str]:
    """Which positions fill the 8 slots when a team must field everyone it can.

    Validation uses this to check every simulated roster can field a full lineup.
    """
    caps = dict(STARTING_SLOTS)
    out: list[str] = []
    for p in sorted(roster, key=_KEY):
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot]:
                caps[slot] -= 1
                out.append(p.position)
                break
    return out


def team_value(
    roster_sorted: list[Player],
    wire: dict[str, float],
    extra: Player | None = None,
) -> float:
    """Expected optimal lineup points for a pre-sorted roster."""
    return expected_lineup_value(
        roster_sorted if extra is None else insert_sorted(roster_sorted, extra),
        wire,
    )


def team_values_with_candidates(
    roster: list[Player],
    wire: dict[str, float],
    candidates: list[Player],
) -> tuple[float, dict[int, float]]:
    """Roster value and each one-player addition, sharing the unchanged positions.

    A candidate alters one position. Computing the other three depth charts once
    avoids rebuilding them for every personal-strategy option during rollouts.
    """
    by_pos = {pos: [] for pos in POSITIONS}
    for player in roster:
        by_pos[player.position].append(player)

    point_rows = {
        pos: tuple(sorted((player.player_id, player.points) for player in by_pos[pos]))
        for pos in POSITIONS
    }
    ded = {}
    tables = {}
    for pos in POSITIONS:
        ded[pos], tables[pos] = _position_tables(point_rows[pos], wire[pos], pos)
    baseline = sum(ded.values()) + _extra_expected_value(
        (tables["RB"], tables["WR"], tables["TE"])
    )
    values: dict[int, float] = {}
    for candidate in candidates:
        pos = candidate.position
        candidate_rows = tuple(
            sorted(point_rows[pos] + ((candidate.player_id, candidate.points),))
        )
        cded, ctable = _position_tables(candidate_rows, wire[pos], pos)
        merged = {**tables, pos: ctable}
        values[candidate.player_id] = (
            sum(ded[p] for p in POSITIONS if p != pos)
            + cded
            + _extra_expected_value((merged["RB"], merged["WR"], merged["TE"]))
        )
    return baseline, values


# --- wire levels --------------------------------------------------------------------


def _rep_at_rank(pos_players: list[Player], rank: int) -> float:
    """Points of the rank-th best player at a position (1-based), clamped to the pool."""
    if not pos_players:
        return 0.0
    return pos_players[min(max(rank, 1), len(pos_players)) - 1].points


def seed_wire(players: list[Player]) -> dict[str, float]:
    """Iteration-0 wire levels from pure slot counting, no draft behaviour assumed.

    Assign the top of the pool to the league's 80 starting slots the way a perfectly
    efficient market would — dedicated slots by positional rank, then the 20 flex slots
    to the best players still eligible — and take the best player at each position who
    did not earn a slot. Convergence replaces this with the best player actually left
    undrafted (`wire_replacement`).
    """
    pos = pos_sorted(players)
    caps = {slot: n * TEAMS for slot, n in STARTING_SLOTS.items()}
    used = {p: 0 for p in POSITIONS}
    for p in sorted(players, key=_KEY):
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot]:
                caps[slot] -= 1
                used[p.position] += 1
                break
    return {k: _rep_at_rank(pos[k], used[k] + 1) for k in POSITIONS}


def wire_replacement(
    taken: set[int], pos: dict[str, list[Player]]
) -> dict[str, float]:
    """The best player at each position actually left undrafted.

    Far below starting caliber once the draft has run: it answers "how much better is
    he than what I could sign for nothing after the draft", and contributes one unique,
    always-available body per position to expected lineup value.
    """
    out: dict[str, float] = {}
    for position, players in pos.items():
        out[position] = max(
            (p.points for p in players if p.player_id not in taken), default=0.0
        )
    return out
