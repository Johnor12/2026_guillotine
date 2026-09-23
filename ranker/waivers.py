"""Guillotine bidding: a published price prior, roster needs, and observed managers.

Charchian's early-season guide supplies the scale, not player-specific predictions:
https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2
Elite / ordinary starters / depth: 15–20% / 2.5–5% / 0.1–1% of $1,000.
Our positional-rank curve, season-long roster valuation, and uncertainty priors are modeling
assumptions. They adapt that 18-team guide to our scoring and expanding lineups.

A claim is a pickup and a drop together: for each candidate the drop is the body whose
loss leaves the best remaining-season roster with the candidate on it, so a backup QB
goes when a better QB arrives and a bench RB goes for a receiver. The claim's gain is that
swap's net lineup points through Week 17. The pickup may be held only for its useful
weeks, after which the vacated spot is refilled from `replacements`, the wire the room
leaves untaken: the drop is charged what that refill cannot restore. With no
replacement, the drop's whole remaining season is charged, so a returning starter is
not a free placeholder. `weights` value each week's points (opponents: every week alike;
mine: alike or by title leverage, whichever replays better, claims.title_objective).
Guide prices rank players by points per game played, since the gain already prorates
missed weeks.
The two reserve slots hold Out/IR/PUP bodies while their projection is zero and stop
holding them when it resumes (league.RESERVE_SLOTS, season.SeasonPlayer.ir_until).
"""

from __future__ import annotations

import copy
import heapq
import math
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from operator import itemgetter

from .league import (
    CLAIM_FULL_BUDGET_GAIN, CLAIM_GAIN_EXPONENT, CLAIM_CONSERVATION_FLOOR,
    CLAIM_CONSERVATION_FULL_WEEK, REGULAR_WEEKS, RESERVE_SLOTS, TEAM_SEASON_SIGMA,
    WEEK_ROSTER_SIZE, WEEKS,
)
from .season import lineup_points, lineup_without_each, thresholds
from .guillotine import SIGMA_WEEK, _cdf

GUIDE_URL = "https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2"
BID_SIGMA = 0.8

# Desired cash entering each week (zero-based). These are strategy priors, not fits
# to one auction. The patient plan keeps half its cash for week-14 superflex.
SAVING_PLANS = {
    "value": ((0, 0.0), (17, 0.0)),
    "balanced": ((0, 1.0), (4, 0.9), (8, 0.75), (12, 0.25), (13, 0.2), (15, 0.05), (17, 0.0)),
    "patient": ((0, 1.0), (4, 0.95), (8, 0.85), (12, 0.55), (13, 0.5), (15, 0.15), (17, 0.0)),
}


def reserve_fraction(policy: str, w: int) -> float:
    for (w0, r0), (w1, r1) in zip(SAVING_PLANS[policy], SAVING_PLANS[policy][1:]):
        if w0 <= w <= w1:
            return r0 + (r1 - r0) * (w - w0) / (w1 - w0)
    raise ValueError(f"week index {w} outside the saving plan")


def spending_allowance(budget: int, initial_budget: int, start: int, w: int,
                       policy: str, risk: float) -> int:
    if policy == "value":
        return budget
    reserve = initial_budget * reserve_fraction(policy, w + 1) / reserve_fraction(policy, start)
    # Survival emergencies can override saving; ordinary weekly cut risk cannot.
    emergency = min(1.0, max(0.0, (risk - 0.25) / 0.25))
    return min(budget, max(0, int(budget - reserve * (1.0 - emergency))))


def projected_bar(rosters, alive, points, positions, w):
    means = [lineup_points(r, points, positions, w) for r, live in zip(rosters, alive) if live]
    sigma = math.hypot(TEAM_SEASON_SIGMA, SIGMA_WEEK[min(w, REGULAR_WEEKS - 1)])
    low, high = min(means) - 4 * sigma, max(means) + 4 * sigma
    for _ in range(20):
        mid = (low + high) / 2
        if sum(_cdf((mid - mu) / sigma) for mu in means) < min(2, len(means) / 2):
            low = mid
        else:
            high = mid
    return (low + high) / 2


def projected_risk(roster, points, positions, w, bar):
    if w >= REGULAR_WEEKS:
        return 0.0
    mu = lineup_points(roster, points, positions, w)
    return _cdf((bar - mu) / math.hypot(TEAM_SEASON_SIGMA, SIGMA_WEEK[w]))


@dataclass(frozen=True)
class Manager:
    activity: float = 0.5
    log_scale: float = 0.0
    scale_sd: float = 0.6
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
    gain: float
    ceiling: float


@dataclass(slots=True)
class Context:
    """One roster before one week's claims, shared by every candidate evaluated on it."""
    roster: tuple[int, ...]
    w: int
    eligible: int  # reserve-eligible bodies on the roster this week
    floors: tuple[tuple[float, ...], ...]  # [pos] -> per-week entry threshold, no drop
    # Per drop, cheapest to vacate first: (player, reserve-eligible, season lineup loss,
    # cheapest vacate cost holding a pickup at least one week, [pos] -> per-week
    # threshold, per-week loss, [i] -> loss after a pickup's first i + 1 weeks that the
    # refill cannot restore, or None when nothing on the wire refills him).
    options: list[tuple[int, bool, float, float, tuple[tuple[float, ...], ...], list[float], list[float] | None]]
    floor_min: tuple[tuple[float, ...], ...]  # [pos] -> per-week threshold under the kindest drop
    memo: dict[int, tuple[float, int | None]] = field(default_factory=dict)


REFILL_DEPTH = 3  # untaken bodies ranked per position, in case the best are already held


class Bidding:
    def __init__(self, positions, weekly, ros, ir_until):
        self.positions = positions
        self.weekly = weekly
        self.ros = ros
        self.ir_until = ir_until
        self.weights = (1.0,) * WEEKS
        self.refills = [((),) * 4] * WEEKS  # [w][pos] -> refill candidates, best first
        self.shares = []
        for w in range(WEEKS):
            points = []
            for j in range(len(positions)):
                played = [weekly[v][j] for v in range(w, WEEKS) if weekly[v][j] > 0]
                points.append(sum(played) / len(played) if played else 0.0)
            shares = [0.0] * len(positions)
            for pos, elite in enumerate((4, 6, 6, 3)):
                ordered = sorted((j for j, p in enumerate(positions) if p == pos), key=lambda j: (-points[j], j))
                for rank, j in enumerate(ordered, 1):
                    shares[j] = 0.20 * min(1.0, elite / rank) ** 2
            self.shares.append(shares)

    def objective(self, weights, replacements) -> "Bidding":
        """A copy valuing week v's points at weights[v] whose drops can be refilled from
        `replacements` once a pickup has served its weeks."""
        other = copy.copy(self)
        other.weights = tuple(weights)
        other.refills = [
            tuple(
                tuple(heapq.nlargest(
                    REFILL_DEPTH, (r for r in replacements if self.positions[r] == pos),
                    key=lambda r: (sum(other.weights[v] * self.weekly[v][r] for v in range(w + 1, WEEKS)), -r)))
                for pos in range(4)
            )
            for w in range(WEEKS)
        ]
        return other

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
        """The body to cut when the roster does not fit week `w` — a reserve body whose
        projection resumed needs a regular spot — or None when it fits. The cut is the
        cheapest remaining-season lineup loss among bodies whose removal frees a spot."""
        if self.active(roster, w) <= WEEK_ROSTER_SIZE[w]:
            return None
        ctx = self.context(tuple(sorted(roster)), w)
        # Cutting a reserve body frees nothing unless the reserve slots are oversubscribed.
        # A cut leaves no spot for a refill, so it is charged its whole loss.
        return min((o for o in ctx.options if not o[1] or ctx.eligible > RESERVE_SLOTS),
                   key=lambda o: (o[2], self.ros[w][o[0]], o[0]))[0]

    @lru_cache(maxsize=8192)
    def context(self, roster: tuple[int, ...], w: int) -> Context:
        per_week = [lineup_without_each(roster, self.weekly[v], self.positions, v) for v in range(w, WEEKS)]
        weights = self.weights[w:]
        span = len(weights)
        floors = tuple(zip(*(f for _, f, _ in per_week)))
        owned = set(roster)
        # The best untaken body at each position not already held; he arrives next week
        # at the earliest.
        refills = [r for r in (next((r for r in col if r not in owned), None) for col in self.refills[w])
                   if r is not None]
        later = [list(map(itemgetter(r), self.weekly[w + 1:])) for r in refills]
        options = []
        for d in roster:
            lost = [k * (total - without[d][0]) for k, (total, _, without) in zip(weights, per_week)]
            loss = sum(lost)
            rest_floors = tuple(zip(*(without[d][1] for _, _, without in per_week)))
            restored, refilled = None, 0.0
            for r, points in zip(refills, later):
                gains = [k * (p - f) if p > f else 0.0
                         for k, p, f in zip(weights[1:], points, rest_floors[self.positions[r]][1:])]
                if sum(gains) > refilled:
                    restored, refilled = gains, sum(gains)
            if restored is None:
                options.append((d, self.ir_until[d] > w, loss, loss, rest_floors, lost, None))
                continue
            after, tail = [0.0] * span, 0.0
            for h in range(span - 1, 0, -1):
                tail += lost[h] - restored[h - 1]
                after[h - 1] = tail if tail > 0.0 else 0.0
            held, cost = 0.0, math.inf
            for held_loss, unrestored in zip(lost, after):
                held += held_loss
                if held + unrestored < cost:
                    cost = held + unrestored
            options.append((d, self.ir_until[d] > w, loss, cost, rest_floors, lost, after))
        options.sort(key=lambda o: (o[3], self.ros[w][o[0]], o[0]))
        floor_min = tuple(tuple(min(col) for col in zip(*(o[4][pos] for o in options))) for pos in range(4))
        eligible = sum(1 for i in roster if self.ir_until[i] > w)
        return Context(roster, w, eligible, floors, options, floor_min)

    def _evaluate(self, ctx: Context, j: int) -> tuple[float, int | None]:
        """(gain per remaining week, drop) for adding `j`: the drop leaving the best
        remaining-season roster with `j` on it, and that swap's net weighted lineup points,
        holding `j` for the most valuable number of weeks before refilling the spot."""
        w, size, n = ctx.w, WEEK_ROSTER_SIZE[ctx.w], len(ctx.roster)
        pos = self.positions[j]
        weights = self.weights[w:]
        points = list(map(itemgetter(j), self.weekly[w:]))
        eligible_j = self.ir_until[j] > w
        if n + 1 - min(RESERVE_SLOTS, ctx.eligible + eligible_j) <= size:
            return sum(k * (p - f) for k, p, f in zip(weights, points, ctx.floors[pos]) if p > f) / len(points), None
        # No swap nets more than j's gain against the kindest thresholds less the drop's
        # cheapest vacate cost, so drops in that order stop once the bound cannot beat the
        # best swap (with slack for rounding, so ties are settled by the same key as
        # without pruning).
        most = sum(k * (p - f) for k, p, f in zip(weights, points, ctx.floor_min[pos]) if p > f)
        if most <= 0.0:
            return 0.0, None
        best = None
        for d, eligible_d, loss, cost, floors, lost, after in ctx.options:
            if best is not None and most - cost < best[0] - 1e-9:
                break
            if n - min(RESERVE_SLOTS, ctx.eligible - eligible_d + eligible_j) > size:
                continue
            if after is None:
                net = sum(k * (p - f) for k, p, f in zip(weights, points, floors[pos]) if p > f) - loss
            else:
                held, net = 0.0, -math.inf
                for k, p, f, held_loss, unrestored in zip(weights, points, floors[pos], lost, after):
                    held += (k * (p - f) if p > f else 0.0) - held_loss
                    if held - unrestored > net:
                        net = held - unrestored
            key = (net, -self.ros[w][d], -d)
            if best is None or key > best:
                best = key
        if best is None or best[0] <= 0.0:
            return 0.0, None
        return best[0] / len(points), -best[2]

    def offers(self, roster, candidates, budget: int, w: int, risk: float = 0.0):
        ctx = self.context(tuple(sorted(roster)), w)
        scale = budget * (WEEKS - 1) / (WEEKS - w) * (1.0 + 2.0 * risk)
        owned = set(roster)
        out = []
        for j in candidates:
            if j in owned:
                continue
            got = ctx.memo.get(j)
            if got is None:
                got = ctx.memo[j] = self._evaluate(ctx, j)
            gain, drop = got
            if gain <= 0.0:
                continue
            ceiling = min(budget, scale * self.shares[w][j] * min(1.5, gain / 5.0))
            if w == WEEKS - 1:
                ceiling = budget  # Unspent FAAB has no value after the final game.
            out.append(Offer(j, drop, gain, ceiling))
        return out


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
    """Use each week's pre-claim roster/budget; never fit a winner against his new roster.

    Past projections are not archived; the current remaining-season projection is
    the explicit proxy for older bids. Evaluation reports this limitation.
    """
    indexes = {p.sleeper_id: p.index for p in state.players}
    teams = {t.roster_id: t for t in state.teams}
    rosters = {t.roster_id: set(t.roster) for t in state.teams}
    budgets = {t.roster_id: t.faab_left for t in state.teams}
    bids = submitted_bids(state)
    observations = []
    weeks = sorted({tx["week"] for tx in bids}, reverse=True)
    for week in weeks:
        completed = [tx for tx in state.transactions if tx["week"] == week and tx["status"] == "complete"]
        for tx in sorted(completed, key=lambda tx: tx.get("processed_at", tx["created"]), reverse=True):
            team = tx["roster_id"]
            rosters[team].difference_update(indexes[sid] for sid in tx["adds"] if sid in indexes)
            rosters[team].update(indexes[sid] for sid in tx["drops"] if sid in indexes)
            budgets[team] += tx["bid"] or 0
        w = state.week - 1
        bar = projected_bar([rosters[t.roster_id] for t in state.teams], [t.alive for t in state.teams],
                            bidding.weekly[w], bidding.positions, w)
        for tx in bids:
            if tx["week"] != week:
                continue
            team = tx["roster_id"]
            if teams[team].is_mine:
                continue
            j = indexes.get(next(iter(tx["adds"])))
            if j is None:
                continue
            risk = projected_risk(rosters[team], bidding.weekly[w], bidding.positions, w, bar)
            offers = bidding.offers(rosters[team], [j], budgets[team], w, risk)
            # Managers also speculate on depth; a small market-value floor allows that.
            reference = max(budgets[team] * bidding.shares[w][j] * 0.25,
                            offers[0].ceiling if offers else 0.0, 1.0)
            gain = bidding.ros[w][j] - thresholds(rosters[team], bidding.ros[w], bidding.positions, w)[bidding.positions[j]]
            observations.append({"week": week, "team": team, "player": j, "bid": tx["bid"],
                                 "reference": reference, "budget": budgets[team], "gain": gain})
    return observations


def fit_managers(state, observations, exclude_player: int | None = None):
    obs = [o for o in observations if o["player"] != exclude_player]
    weeks = {o["week"] for o in observations}
    positive = [math.log(o["bid"] / o["reference"]) for o in obs if o["bid"] > 0]
    # Partial pooling limits what one auction can say about a manager's habits.
    room = sum(positive) / (len(positive) + 8)
    out = []
    for team in state.teams:
        rows = [o for o in obs if o["team"] == team.roster_id]
        active = len({o["week"] for o in rows})
        ratios = [math.log(o["bid"] / o["reference"]) for o in rows if o["bid"] > 0]
        strength = 3.0
        mean = (strength * room + sum(ratios)) / (strength + len(ratios))
        out.append(Manager((2.0 + active) / (4.0 + len(weeks)), mean,
                           BID_SIGMA / math.sqrt(strength + len(ratios)), active, len(rows)))
    return out


def calibration(state, observations):
    """Leave an entire player's bids out, then predict them from other players' bids."""
    errors, prior_errors, legacy_errors = [], [], []
    by_team = {t.roster_id: i for i, t in enumerate(state.teams)}
    predictions = []
    for j in sorted({o["player"] for o in observations}):
        fitted = fit_managers(state, observations, exclude_player=j)
        for row in observations:
            if row["player"] != j or row["bid"] <= 0:
                continue
            predicted = min(row["budget"], row["reference"] * math.exp(fitted[by_team[row["team"]]].log_scale))
            errors.append(abs(math.log(row["bid"] / predicted)))
            prior_errors.append(abs(math.log(row["bid"] / row["reference"])))
            ramp = min(1.0, (row["week"] - 1) / (CLAIM_CONSERVATION_FULL_WEEK - 1))
            legacy = row["budget"] * min(1.0, max(0.0, row["gain"]) / CLAIM_FULL_BUDGET_GAIN) ** CLAIM_GAIN_EXPONENT
            legacy *= CLAIM_CONSERVATION_FLOOR + (1 - CLAIM_CONSERVATION_FLOOR) * ramp
            legacy_errors.append(abs(math.log(row["bid"] / max(1.0, legacy))))
            predictions.append({"player": state.players[j].name, "roster_id": row["team"],
                                "actual": row["bid"], "predicted": round(predicted)})
    return {"method": "leave-one-player-out; current projections proxy historical player value",
            "bid_weeks": len({o["week"] for o in observations}), "submitted_bids": len(observations),
            "positive_bids_tested": len(errors),
            "prior_log_mae": round(statistics.fmean(prior_errors), 3) if errors else None,
            "legacy_log_mae": round(statistics.fmean(legacy_errors), 3) if errors else None,
            "fitted_log_mae": round(statistics.fmean(errors), 3) if errors else None,
            "predictions": predictions}
