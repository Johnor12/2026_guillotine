"""Guillotine bidding: a published price prior, roster needs, and observed managers.

Charchian's early-season guide supplies the scale, not player-specific predictions:
https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2
Elite / ordinary starters / depth: 15-20% / 2.5-5% / 0.1-1% of $1,000. Our positional
rank curve, season-long roster valuation and uncertainty priors are modeling assumptions
that adapt that 18-team guide to our scoring and expanding lineups.

A claim is a pickup and a drop together: for each candidate the drop is the body whose
loss leaves the best remaining-season roster with the candidate on it, so a backup QB
goes when a better QB arrives and a bench RB goes for a receiver. The claim's gain is
that swap's net lineup points through week 17. The pickup may be held only for its
useful weeks, after which the vacated spot is refilled from the wire the room leaves
untaken (`Bidding.objective`): the drop is charged what that refill cannot restore.
With no replacement the drop's whole remaining season is charged, so a returning
starter is not a free placeholder. `weights` value each week's points (opponents:
every week alike; mine: alike or by title leverage, whichever replays better,
claims.title_objective), and opponents also multiply the week they are bidding for by
ROOM_CURRENT_WEEK_WEIGHT. Guide prices rank players by points per game played,
weighted the same way, since the gain already prorates missed weeks. The two reserve
slots hold Out/IR/PUP bodies while their projection is zero and stop holding them when
it resumes (state.SeasonPlayer.ir_until).

The arithmetic is the hot path of both the race and the replays, so it is compiled
(numba): one `Context` per (roster, week) holds every drop's loss profile and post-drop
lineup thresholds over the remaining weeks, and candidates are priced against the drops
in one call.
"""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from shared.league import POSITIONS, REGULAR_WEEKS, RESERVE_SLOTS, WEEK_ROSTER_SIZE, WEEKLY_SHAPES, WEEKS
from shared.noise import SIGMA_WEEK, TEAM_SEASON_SIGMA, cdf

from .state import lineup_points

GUIDE_URL = "https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2"
BID_SIGMA = 0.8  # per-bid log noise before any bid is observed
# Opponents value the week they bid for this many times each later week, in claim gains
# and guide ranks. An ex-ante backtest of the week-3 auction (fit on week 2) improved with
# the weight up to 32-64x on bid sizes, who got claimed and clearing prices; 32 predicted
# who got claimed best, and valuing this week alone did worse.
ROOM_CURRENT_WEEK_WEIGHT = 32.0
# Guide price share of the budget by positional rank: 20% for the elite anchors
# (QB4 / RB6 / WR6 / TE3), then an inverse-square curve.
ELITE_RANK = (4, 6, 6, 3)
REFILL_DEPTH = 3  # untaken bodies ranked per position, in case the best are already held
CONTEXT_CACHE = 8192  # (roster, week) contexts kept per Bidding before the cache is cleared

# Opponents' saving habits: desired cash entering each week (zero-based), one plan per
# manager for the season, a uniform prior rather than a fit to one auction. The patient
# plan keeps half its cash for week-14 superflex. My own agent does not use these: it
# bids a multiple of the guide ceiling that the replays choose (race.SPENDING).
SAVING_PLANS = {
    "value": ((0, 0.0), (17, 0.0)),
    "balanced": ((0, 1.0), (4, 0.9), (8, 0.75), (12, 0.25), (13, 0.2), (15, 0.05), (17, 0.0)),
    "patient": ((0, 1.0), (4, 0.95), (8, 0.85), (12, 0.55), (13, 0.5), (15, 0.15), (17, 0.0)),
}

# Dedicated slots per position and flex seats, by week, for the lineup solver.
_DEDICATED = np.array([[shape[pos] for shape in WEEKLY_SHAPES] for pos in POSITIONS], dtype=np.int64)
_FLEX = np.array([shape["FLEX"] for shape in WEEKLY_SHAPES], dtype=np.int64)
_NONE = -1  # "no drop" in the compiled core's integer arrays


def reserve_fraction(policy: str, w: int) -> float:
    for (w0, r0), (w1, r1) in zip(SAVING_PLANS[policy], SAVING_PLANS[policy][1:]):
        if w0 <= w <= w1:
            return r0 + (r1 - r0) * (w - w0) / (w1 - w0)
    raise ValueError(f"week index {w} outside the saving plan")


def spending_allowance(budget: int, initial_budget: int, start: int, w: int,
                       policy: str, risk: float) -> int:
    """An opponent's cash available to this auction under its saving plan."""
    if policy == "value":
        return budget
    reserve = initial_budget * reserve_fraction(policy, w + 1) / reserve_fraction(policy, start)
    # Survival emergencies can override saving; ordinary weekly cut risk cannot.
    emergency = min(1.0, max(0.0, (risk - 0.25) / 0.25))
    return min(budget, max(0, int(budget - reserve * (1.0 - emergency))))


def projected_bar(rosters, alive, points, positions, w):
    """Expected cut score from projected lineups; bids cannot see realized scores."""
    means = [lineup_points(r, points, positions, w) for r, live in zip(rosters, alive) if live]
    sigma = math.hypot(TEAM_SEASON_SIGMA, SIGMA_WEEK[min(w, REGULAR_WEEKS - 1)])
    low, high = min(means) - 4 * sigma, max(means) + 4 * sigma
    for _ in range(20):
        mid = (low + high) / 2
        if sum(cdf((mid - mu) / sigma) for mu in means) < min(2, len(means) / 2):
            low = mid
        else:
            high = mid
    return (low + high) / 2


def projected_risk(roster, points, positions, w, bar):
    if w >= REGULAR_WEEKS:
        return 0.0
    mu = lineup_points(roster, points, positions, w)
    return cdf((bar - mu) / math.hypot(TEAM_SEASON_SIGMA, SIGMA_WEEK[w]))


# Participation prior Beta(2/3, 1/3), worth one auction: the method-of-moments fit to the
# week-2 and week-3 auctions, where 15 of 16 week-2 bidders bid again and 5 of 11 quiet
# managers started.
ACTIVE_PRIOR = 2.0 / 3.0


def guide_reference(bidding, player: int, ceiling: float, budget: int, w: int) -> float:
    """The guide price a bid is measured against. Managers also speculate on depth; a small
    market-value floor keeps a pickup the guide ceiling barely values priced."""
    return max(budget * bidding.shares[w][player] * 0.25, ceiling, 1.0)


@dataclass(frozen=True)
class PriceCurve:
    """The room's bid for a guide reference: log bid = intercept + slope * log reference.
    This room is flatter than the guide (slope below 1): depth and fill-ins sell for
    several times their guide price, stars for less."""
    intercept: float = 0.0
    slope: float = 1.0
    sigma: float = BID_SIGMA  # residual log noise of one submitted bid

    def price(self, reference: float) -> float:
        return math.exp(self.intercept + self.slope * math.log(reference))


@dataclass(frozen=True)
class Manager:
    activity: float = 0.5
    bid_weeks: int = 0
    bids: int = 0

    def participation(self, w: int, start: int) -> float:
        # No permanent inactive class: quiet survivors can return as the field shrinks.
        progress = (w - start) / max(1, WEEKS - 1 - start)
        return self.activity + (1.0 - self.activity) * progress


@dataclass(frozen=True)
class Offer:
    player: int
    drop: int | None
    gain: float  # net weighted lineup points per remaining week
    ceiling: float  # the guide's price for it



# --- compiled core ------------------------------------------------------------------


@njit(cache=True)
def _lineups(points, pos, dedicated, flex):
    """Every remaining week's greedy lineup on a roster, and each body's removal.

    points (span, n) by week; pos (n,); dedicated (4, span) slots; flex (span,) seats.
    Returns total (span,), floors (4, span), totals_without (n, span), floors_without
    (n, 4, span). A floor is the points a free agent must beat to enter that week's
    lineup at that position: the weaker of the last dedicated starter and the last flex.
    A dedicated starter's removal promotes his position's first leftover, which leaves
    the flex (if it held him) to the next pooled body; a flex starter's removal seats
    that next body; a bench body's removal changes nothing.
    """
    span, n = points.shape
    total = np.zeros(span)
    floors = np.zeros((4, span))
    totals_without = np.zeros((n, span))
    floors_without = np.zeros((n, 4, span))
    column = np.empty(n)
    order = np.empty(n, np.int64)
    rank = np.empty(n, np.int64)
    pooled = np.empty(n)
    last = np.zeros(4)
    promoted = np.zeros(4)
    has_promoted = np.zeros(4, np.bool_)
    for i in range(span):
        week_total = 0.0
        npooled = 0
        for p in range(4):
            k = 0
            for d in range(n):
                if pos[d] == p:
                    column[k] = points[i, d]
                    order[k] = d
                    k += 1
            for a in range(1, k):  # sort descending, carrying the body ids
                x = column[a]
                o = order[a]
                b = a - 1
                while b >= 0 and column[b] < x:
                    column[b + 1] = column[b]
                    order[b + 1] = order[b]
                    b -= 1
                column[b + 1] = x
                order[b + 1] = o
            for a in range(k):
                rank[order[a]] = a
            slots = dedicated[p, i]
            for a in range(min(slots, k)):
                week_total += column[a]
            last[p] = column[slots - 1] if k >= slots else 0.0
            has_promoted[p] = k > slots
            promoted[p] = column[slots] if k > slots else 0.0
            if p > 0:
                for a in range(slots, k):
                    pooled[npooled] = column[a]
                    npooled += 1
        for a in range(1, npooled):
            x = pooled[a]
            b = a - 1
            while b >= 0 and pooled[b] < x:
                pooled[b + 1] = pooled[b]
                b -= 1
            pooled[b + 1] = x
        seats = flex[i]
        for a in range(min(seats, npooled)):
            week_total += pooled[a]
        flex_last = pooled[seats - 1] if npooled >= seats else 0.0
        flex_next = pooled[seats] if npooled > seats else 0.0
        total[i] = week_total
        floors[0, i] = last[0]
        for p in range(1, 4):
            floors[p, i] = min(last[p], flex_last)
        for d in range(n):
            p = pos[d]
            x = points[i, d]
            flex_after = flex_last
            last_after = last[p]
            if rank[d] < dedicated[p, i]:
                if has_promoted[p]:
                    if p > 0 and promoted[p] >= flex_last:
                        without = week_total - x + flex_next
                        flex_after = flex_next
                    else:
                        without = week_total - x + promoted[p]
                    last_after = promoted[p]
                else:
                    without = week_total - x
                    last_after = 0.0
            elif p > 0 and x >= flex_last:
                without = week_total - x + flex_next
                flex_after = flex_next
            else:
                without = week_total
            totals_without[d, i] = without
            for q in range(4):
                base = last_after if q == p else last[q]
                floors_without[d, q, i] = base if q == 0 else min(base, flex_after)
    return total, floors, totals_without, floors_without


@njit(cache=True)
def _refill(lost, rest_floors, weights, refill_points, refill_pos):
    """What the untaken wire restores once a pickup leaves a vacated spot.

    lost (n, span) per drop per week; rest_floors (n, 4, span); refill_points (m, span-1)
    the refill bodies' points from next week on; refill_pos (m,). Returns after (n, span),
    the loss the best refill cannot restore once a pickup held through week offset i
    leaves, cost (n,), the cheapest way to vacate the spot for at least one week, and
    refillable (n,).
    """
    n, span = lost.shape
    m = refill_points.shape[0]
    after = np.zeros((n, span))
    cost = np.zeros(n)
    refillable = np.zeros(n, np.bool_)
    restored = np.zeros(span - 1)
    for d in range(n):
        best = 0.0
        best_k = -1
        for k in range(m):
            gained = 0.0
            for i in range(span - 1):
                diff = refill_points[k, i] - rest_floors[d, refill_pos[k], i + 1]
                if diff > 0.0:
                    gained += weights[i + 1] * diff
            if gained > best:
                best = gained
                best_k = k
        held = 0.0
        if best_k < 0:
            for i in range(span):
                held += lost[d, i]
            cost[d] = held
            continue
        refillable[d] = True
        for i in range(span - 1):
            diff = refill_points[best_k, i] - rest_floors[d, refill_pos[best_k], i + 1]
            restored[i] = weights[i + 1] * diff if diff > 0.0 else 0.0
        tail = 0.0
        for h in range(span - 1, 0, -1):
            tail += lost[d, h] - restored[h - 1]
            after[d, h - 1] = tail if tail > 0.0 else 0.0
        cheapest = np.inf
        for h in range(span):
            held += lost[d, h]
            if held + after[d, h] < cheapest:
                cheapest = held + after[d, h]
        cost[d] = cheapest
    return after, cost, refillable


@njit(cache=True)
def _evaluate(candidates, points, pos, ir_until, ros, w, weights, floors, floor_min, size, reserve_slots,
              eligible, drops, drop_eligible, loss, cost, rest_floors, lost, after, refillable, prune):
    """For each candidate: (gain per remaining week, drop, fits without a drop), plus his
    net against every drop when `prune` is off (for ranking drops).

    Options are in cost order: no swap nets more than the candidate's gain against the
    kindest thresholds less the drop's vacate cost, so the loop stops once that bound
    cannot beat the best swap (with slack for rounding, so ties settle by the same key).
    Among equal nets the drop with the lower rest-of-season points, then index, goes.
    """
    span = weights.shape[0]
    c = candidates.shape[0]
    n = drops.shape[0]
    gains = np.zeros(c)
    chosen = np.full(c, -1, np.int64)
    fits = np.zeros(c, np.bool_)
    nets = np.full((n, c), -np.inf)
    for ci in range(c):
        j = candidates[ci]
        p = pos[j]
        eligible_j = 1 if ir_until[j] > w else 0
        free_gain = 0.0
        most = 0.0
        for i in range(span):
            x = points[w + i, j]
            if x > floors[p, i]:
                free_gain += weights[i] * (x - floors[p, i])
            if x > floor_min[p, i]:
                most += weights[i] * (x - floor_min[p, i])
        if n + 1 - min(reserve_slots, eligible + eligible_j) <= size:
            fits[ci] = True
            gains[ci] = free_gain / span
            continue
        if most <= 0.0:
            continue
        best = -np.inf
        best_o = -1
        for o in range(n):
            if prune and best_o >= 0 and most - cost[o] < best - 1e-9:
                break
            eligible_d = 1 if drop_eligible[o] else 0
            if n - min(reserve_slots, eligible - eligible_d + eligible_j) > size:
                continue
            if refillable[o]:
                held = 0.0
                net = -np.inf
                for i in range(span):
                    x = points[w + i, j]
                    if x > rest_floors[o, p, i]:
                        held += weights[i] * (x - rest_floors[o, p, i])
                    held -= lost[o, i]
                    if held - after[o, i] > net:
                        net = held - after[o, i]
            else:
                net = -loss[o]
                for i in range(span):
                    x = points[w + i, j]
                    if x > rest_floors[o, p, i]:
                        net += weights[i] * (x - rest_floors[o, p, i])
            nets[o, ci] = net
            if best_o < 0 or net > best or (net == best and (
                    ros[drops[o]] < ros[drops[best_o]]
                    or (ros[drops[o]] == ros[drops[best_o]] and drops[o] < drops[best_o]))):
                best = net
                best_o = o
        if best_o >= 0 and best > 0.0:
            gains[ci] = best / span
            chosen[ci] = drops[best_o]
    return gains, chosen, fits, nets


@dataclass(slots=True)
class Context:
    """One roster before one week's claims, shared by every candidate evaluated on it.

    Week offset i is week w + i. The drop arrays are in cost order, cheapest to vacate
    first, so the compiled evaluation can stop early.
    """
    roster: tuple[int, ...]
    w: int
    eligible: int  # reserve-eligible bodies on the roster this week
    floors: np.ndarray  # (4, span) per-position lineup entry thresholds with no drop
    floor_min: np.ndarray  # (4, span) the same under the kindest drop
    drops: np.ndarray  # (n,) player indexes
    drop_eligible: np.ndarray  # (n,) reserve-eligible this week
    loss: np.ndarray  # (n,) weighted lineup points lost over the season by dropping him
    cost: np.ndarray  # (n,) the cheapest way to vacate his spot for at least one week
    lost: np.ndarray  # (n, span) the loss per week
    rest_floors: np.ndarray  # (n, 4, span) thresholds after the drop
    after: np.ndarray  # (n, span) loss the wire cannot restore once a pickup held i + 1 weeks leaves
    refillable: np.ndarray  # (n,) whether an untaken wire body refills him at all
    memo: dict[int, tuple[float, int | None]] = field(default_factory=dict)


class Bidding:
    def __init__(self, positions, weekly, ros, ir_until, current_weight: float = 1.0):
        self.positions = positions
        self.pos = np.asarray(positions, dtype=np.int64)
        self.points = np.asarray(weekly, dtype=np.float64)  # (WEEKS, players)
        self.ros = np.asarray(ros, dtype=np.float64)
        self.ir_until = np.asarray(ir_until, dtype=np.int64)
        self.weights = np.ones(WEEKS)
        self.current_weight = current_weight  # multiplies the week being decided
        self.refills = [((),) * 4] * WEEKS  # [w][pos] -> refill candidates, best first
        self.contexts: dict[tuple[tuple[int, ...], int], Context] = {}
        # Guide price share by positional rank of weighted points per game played from
        # each week on, so a returning player is not charged twice for missed weeks.
        self.shares = []
        for w in range(WEEKS):
            per_game = []
            for j in range(len(positions)):
                played = [(current_weight if v == w else 1.0, weekly[v][j]) for v in range(w, WEEKS) if weekly[v][j] > 0]
                total = sum(k for k, _ in played)
                per_game.append(sum(k * p for k, p in played) / total if total else 0.0)
            shares = [0.0] * len(positions)
            for p, elite in enumerate(ELITE_RANK):
                ordered = sorted((j for j, q in enumerate(positions) if q == p), key=lambda j: (-per_game[j], j))
                for rank, j in enumerate(ordered, 1):
                    shares[j] = 0.20 * min(1.0, elite / rank) ** 2
            self.shares.append(shares)

    def objective(self, weights, replacements) -> "Bidding":
        """A copy valuing week v's points at weights[v] whose drops can be refilled from
        `replacements` once a pickup has served its weeks."""
        other = copy.copy(self)
        other.weights = np.asarray(weights, dtype=np.float64)
        other.contexts = {}
        replacements = np.asarray(sorted(replacements), dtype=np.int64)
        refills = []
        for w in range(WEEKS):
            value = other.weights[w + 1:] @ self.points[w + 1:, replacements]
            cols = []
            for p in range(4):
                mine = self.pos[replacements] == p
                ids = replacements[mine]
                order = ids[np.lexsort((ids, -value[mine]))]
                cols.append(tuple(int(r) for r in order[:REFILL_DEPTH]))
            refills.append(tuple(cols))
        other.refills = refills
        return other

    def week_weights(self, w: int) -> np.ndarray:
        """Weights on weeks w onward for a decision before week `w`'s games."""
        weights = self.weights[w:].copy()
        weights[0] *= self.current_weight
        return weights

    def reserved(self, roster, w: int) -> int:
        """Bodies the reserve slots hold before week `w`'s games."""
        return min(RESERVE_SLOTS, sum(1 for i in roster if self.ir_until[i] > w))

    def active(self, roster, w: int) -> int:
        """Bodies needing a regular roster spot in week `w`."""
        return len(roster) - self.reserved(roster, w)

    def fits(self, roster, player: int, w: int) -> bool:
        """Whether `player` joins the roster before week `w` without a drop."""
        return self.active([*roster, player], w) <= WEEK_ROSTER_SIZE[w]

    def crunch(self, roster, w: int) -> int | None:
        """The body to cut when the roster does not fit week `w` (a reserve body whose
        projection resumed needs a regular spot), or None when it fits. The cut is the
        cheapest remaining-season lineup loss among bodies whose removal frees a spot;
        cutting a reserve body frees nothing unless the reserve slots are oversubscribed."""
        if self.active(roster, w) <= WEEK_ROSTER_SIZE[w]:
            return None
        ctx = self.context(roster, w)
        frees = ~ctx.drop_eligible | (ctx.eligible > RESERVE_SLOTS)
        order = np.lexsort((ctx.drops, self.ros[w, ctx.drops], np.where(frees, ctx.loss, np.inf)))
        return int(ctx.drops[order[0]])

    def context(self, roster, w: int) -> Context:
        key = (tuple(sorted(roster)), w)
        ctx = self.contexts.get(key)
        if ctx is None:
            if len(self.contexts) >= CONTEXT_CACHE:
                self.contexts.clear()
            ctx = self.contexts[key] = self._build(key[0], w)
        return ctx

    def _build(self, roster: tuple[int, ...], w: int) -> Context:
        weights = self.week_weights(w)
        ids = np.asarray(roster, dtype=np.int64)
        total, floors, totals_without, floors_without = _lineups(
            self.points[w:, ids], self.pos[ids], _DEDICATED[:, w:], _FLEX[w:]
        )
        lost = weights * (total - totals_without)
        # The wire's refill of a vacated spot, from next week on: the best untaken body at
        # each position not already held.
        held = set(roster)
        refill = [r for r in (next((r for r in col if r not in held), None) for col in self.refills[w]) if r is not None]
        refill = np.asarray(refill, dtype=np.int64)
        after, cost, refillable = _refill(lost, floors_without, weights, self.points[w + 1:, refill].T, self.pos[refill])
        order = np.lexsort((ids, self.ros[w, ids], cost))
        return Context(
            roster, w, int((self.ir_until[ids] > w).sum()), floors, floors_without.min(0),
            ids[order], self.ir_until[ids][order] > w, lost.sum(1)[order], cost[order], lost[order],
            floors_without[order], after[order], refillable[order],
        )

    def _core(self, ctx: Context, candidates: list[int], prune: bool):
        return _evaluate(
            np.asarray(candidates, dtype=np.int64), self.points, self.pos, self.ir_until, self.ros[ctx.w],
            ctx.w, self.week_weights(ctx.w), ctx.floors, ctx.floor_min, WEEK_ROSTER_SIZE[ctx.w], RESERVE_SLOTS,
            ctx.eligible, ctx.drops, ctx.drop_eligible, ctx.loss, ctx.cost, ctx.rest_floors, ctx.lost, ctx.after,
            ctx.refillable, prune,
        )

    def evaluate(self, roster, candidates, w: int) -> list[tuple[float, int | None]]:
        """(gain per remaining week, drop) for each candidate: the drop leaving the best
        remaining-season roster with him on it, or no drop when he fits; (0, None) when
        nothing improves the roster. Memoized per (roster, week)."""
        ctx = self.context(roster, w)
        fresh = [j for j in candidates if j not in ctx.memo]
        if fresh:
            gains, drops, _, _ = self._core(ctx, fresh, True)
            for j, gain, drop in zip(fresh, gains.tolist(), drops.tolist()):
                ctx.memo[j] = (gain, None if drop == _NONE else drop)
        return [ctx.memo[j] for j in candidates]

    def _offer(self, j: int, drop: int | None, gain: float, budget: int, w: int, risk: float,
               spending: float) -> Offer:
        if w == WEEKS - 1:
            return Offer(j, drop, gain, budget)  # Unspent FAAB has no value after the final game.
        scale = spending * budget * (WEEKS - 1) / (WEEKS - w) * (1.0 + 2.0 * risk)
        return Offer(j, drop, gain, min(budget, scale * self.shares[w][j] * min(1.5, gain / 5.0)))

    def offers(self, roster, candidates, budget: int, w: int, risk: float = 0.0,
               spending: float = 1.0) -> list[Offer]:
        """An offer for every candidate that improves the roster, with the drop that goes,
        at `spending` times the guide's price."""
        owned = set(roster)
        wanted = [j for j in candidates if j not in owned]
        return [
            self._offer(j, drop, gain, budget, w, risk, spending)
            for j, (gain, drop) in zip(wanted, self.evaluate(roster, wanted, w))
            if gain > 0.0
        ]

    def swaps(self, roster, j: int, budget: int, w: int, risk: float, k: int) -> list[Offer]:
        """Up to `k` improving offers for `j`, one per drop, best gain first. An open spot
        needs no drop."""
        ctx = self.context(roster, w)
        span = WEEKS - w
        gains, _, fits, nets = self._core(ctx, [j], False)
        if fits[0]:
            return [self._offer(j, None, float(gains[0]), budget, w, risk, 1.0)] if gains[0] > 0 else []
        order = np.lexsort((ctx.drops, self.ros[w, ctx.drops], -nets[:, 0]))
        return [
            self._offer(j, int(ctx.drops[o]), float(nets[o, 0]) / span, budget, w, risk, 1.0)
            for o in order[:k] if nets[o, 0] > 0.0
        ]


def submitted_bids(state):
    """One submitted amount per team/player/week, including losing and invalid claims."""
    seen = {}
    for tx in sorted(state.transactions, key=lambda tx: tx["created"]):
        if tx["type"] != "waiver" or tx["status"] not in ("complete", "failed") or tx["bid"] is None:
            continue
        for sid in tx["adds"]:
            seen[tx["week"], tx["roster_id"], sid] = tx
    return list(seen.values())


def bid_observations(state, bidding):
    """Each submitted bid against the roster and budget its manager held before that
    week's claims, rolled back from the live rosters through the completed transactions.

    Past projections are not archived; the current remaining-season projection is the
    explicit proxy for older bids.
    """
    indexes = {p.sleeper_id: p.index for p in state.players}
    teams = {t.roster_id: t for t in state.teams}
    rosters = {t.roster_id: set(t.roster) for t in state.teams}
    budgets = {t.roster_id: t.faab_left for t in state.teams}
    bids = submitted_bids(state)
    observations = []
    w = state.week - 1
    for week in sorted({tx["week"] for tx in bids}, reverse=True):
        completed = [tx for tx in state.transactions if tx["week"] == week and tx["status"] == "complete"]
        for tx in sorted(completed, key=lambda tx: tx.get("processed_at", tx["created"]), reverse=True):
            team = tx["roster_id"]
            rosters[team].difference_update(indexes[sid] for sid in tx["adds"] if sid in indexes)
            rosters[team].update(indexes[sid] for sid in tx["drops"] if sid in indexes)
            budgets[team] += tx["bid"] or 0
        bar = projected_bar([rosters[t.roster_id] for t in state.teams], [t.alive for t in state.teams],
                            bidding.points[w], bidding.positions, w)
        for tx in bids:
            team = tx["roster_id"]
            if tx["week"] != week or teams[team].is_mine:
                continue
            j = indexes.get(next(iter(tx["adds"])))
            if j is None:
                continue
            risk = projected_risk(rosters[team], bidding.points[w], bidding.positions, w, bar)
            offers = bidding.offers(rosters[team], [j], budgets[team], w, risk)
            reference = guide_reference(bidding, j, offers[0].ceiling if offers else 0.0, budgets[team], w)
            observations.append({"week": week, "team": team, "player": j, "bid": tx["bid"],
                                 "reference": reference, "budget": budgets[team]})
    return observations


def fit_price_curve(observations) -> PriceCurve:
    """Least squares of log bid on log guide reference over positive submitted bids, with
    the residual spread as bid noise; the guide itself when there is nothing to fit.

    One curve for the room: managers' levels around it did not persist from the week-2 to
    the week-3 auction (correlation -0.24 over 15 managers), and per-manager offsets fit on
    week 2 predicted week 3 worse than the curve alone."""
    points = [(math.log(o["reference"]), math.log(o["bid"])) for o in observations if o["bid"] > 0]
    if len(points) < 3 or len({x for x, _ in points}) < 2:
        return PriceCurve()
    mx = statistics.fmean(x for x, _ in points)
    my = statistics.fmean(y for _, y in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    intercept = my - slope * mx
    sigma = math.sqrt(sum((y - intercept - slope * x) ** 2 for x, y in points) / (len(points) - 2))
    return PriceCurve(intercept, slope, sigma)


def off_cycle_share(state) -> float:
    """Opponents claiming off-cycle (processed after the week's run, on players dropped
    since) per opponent claiming in that run, pooled over completed weeks. Mid-week
    attention is thinner: week 2 had 5 off-cycle bidders against 15 in the run. With no
    completed run to observe yet (week 1), nobody is assumed to be watching."""
    mine = state.my_team.roster_id
    claims = [tx for tx in state.transactions if tx["type"] == "waiver" and tx["status"] in ("complete", "failed")
              and tx["week"] < state.week and tx["roster_id"] != mine]
    in_run = off_cycle = 0
    for week in {tx["week"] for tx in claims}:
        rows = [tx for tx in claims if tx["week"] == week]
        run = min(tx["processed_at"] for tx in rows)  # the weekly run processes every claim at once
        in_run += len({tx["roster_id"] for tx in rows if tx["processed_at"] == run})
        off_cycle += len({tx["roster_id"] for tx in rows if tx["processed_at"] > run})
    return off_cycle / in_run if in_run else 0.0


def fit_managers(state, observations):
    """The room's price curve, and each manager's participation, which does persist."""
    weeks = {o["week"] for o in observations}
    out = []
    for team in state.teams:
        rows = [o for o in observations if o["team"] == team.roster_id]
        active = len({o["week"] for o in rows})
        out.append(Manager((ACTIVE_PRIOR + active) / (1.0 + len(weeks)), active, len(rows)))
    return fit_price_curve(observations), out


def _errors(state, rows, curve):
    """Log errors of the fitted and guide predictions for positive bids in `rows`."""
    fitted, prior, predictions = [], [], []
    for row in rows:
        if row["bid"] <= 0:
            continue
        predicted = min(row["budget"], curve.price(row["reference"]))
        fitted.append(math.log(row["bid"] / predicted))
        prior.append(math.log(row["bid"] / row["reference"]))
        predictions.append({"player": state.players[row["player"]].name, "roster_id": row["team"],
                            "actual": row["bid"], "predicted": round(predicted)})
    return fitted, prior, predictions


def _mae(errors):
    return round(statistics.fmean(abs(e) for e in errors), 3) if errors else None


def calibration(state, observations):
    """Leave an entire player's bids out, then predict them from other players' bids; and
    with two or more auctions, predict the latest from the earlier ones alone."""
    errors, prior_errors, predictions = [], [], []
    for j in sorted({o["player"] for o in observations}):
        curve = fit_price_curve([o for o in observations if o["player"] != j])
        got = _errors(state, [o for o in observations if o["player"] == j], curve)
        for acc, new in zip((errors, prior_errors, predictions), got):
            acc.extend(new)
    weeks = sorted({o["week"] for o in observations})
    temporal = None
    if len(weeks) > 1:
        earlier = [o for o in observations if o["week"] < weeks[-1]]
        fitted, prior, _ = _errors(state, [o for o in observations if o["week"] == weeks[-1]],
                                   fit_price_curve(earlier))
        temporal = {"test_week": weeks[-1], "positive_bids_tested": len(fitted),
                    "prior_log_mae": _mae(prior), "fitted_log_mae": _mae(fitted),
                    "fitted_mean_log_ratio": round(statistics.fmean(fitted), 3) if fitted else None}
    curve = fit_price_curve(observations)
    return {"method": "leave-one-player-out; current projections proxy historical player value",
            "bid_weeks": len(weeks), "submitted_bids": len(observations),
            "positive_bids_tested": len(errors),
            "price_curve": {"intercept": round(curve.intercept, 3), "slope": round(curve.slope, 3),
                            "sigma": round(curve.sigma, 3)},
            "prior_log_mae": _mae(prior_errors), "fitted_log_mae": _mae(errors),
            "latest_week_holdout": temporal, "predictions": predictions}
