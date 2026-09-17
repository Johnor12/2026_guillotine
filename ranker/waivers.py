"""Guillotine bidding: a published price prior, roster needs, and observed managers.

Charchian's early-season guide supplies the scale, not player-specific predictions:
https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2
Elite / ordinary starters / depth: 15–20% / 2.5–5% / 0.1–1% of $1,000.
Our positional-rank curve, season-long roster valuation, and uncertainty priors are modeling
assumptions. They adapt that 18-team guide to our scoring and expanding lineups.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from functools import lru_cache

from .league import (
    CLAIM_FULL_BUDGET_GAIN, CLAIM_GAIN_EXPONENT, CLAIM_CONSERVATION_FLOOR,
    CLAIM_CONSERVATION_FULL_WEEK, REGULAR_WEEKS, TEAM_SEASON_SIGMA, WEEK_ROSTER_SIZE, WEEKS,
)
from .season import lineup_points, thresholds
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


class Bidding:
    def __init__(self, positions, weekly, ros):
        self.positions = positions
        self.weekly = weekly
        self.ros = ros
        self.shares = []
        for points in ros:
            shares = [0.0] * len(positions)
            for pos, elite in enumerate((4, 6, 6, 3)):
                ordered = sorted((j for j, p in enumerate(positions) if p == pos), key=lambda j: (-points[j], j))
                for rank, j in enumerate(ordered, 1):
                    shares[j] = 0.20 * min(1.0, elite / rank) ** 2
            self.shares.append(shares)

    @lru_cache(maxsize=8192)
    def context(self, roster: tuple[int, ...], w: int, extra: int):
        weeks = tuple(range(w, WEEKS))
        totals = [lineup_points(roster, self.weekly[v], self.positions, v) for v in weeks]
        drop = None
        if len(roster) >= WEEK_ROSTER_SIZE[w] + extra:
            # Include every remaining bye and expansion, including week-14 superflex.
            def lost(j):
                rest = [p for p in roster if p != j]
                loss = sum(total - lineup_points(rest, self.weekly[v], self.positions, v)
                           for v, total in zip(weeks, totals))
                return loss, self.ros[w][j], j
            drop = min(roster, key=lost)
        rest = [p for p in roster if p != drop]
        losses = tuple(total - lineup_points(rest, self.weekly[v], self.positions, v) for v, total in zip(weeks, totals))
        floors = tuple(thresholds(rest, self.weekly[v], self.positions, v) for v in weeks)
        return weeks, drop, losses, floors

    def offers(self, roster, candidates, budget: int, w: int, extra: int, risk: float = 0.0):
        weeks, drop, losses, floors = self.context(tuple(sorted(roster)), w, extra)
        scale = budget * (WEEKS - 1) / (WEEKS - w) * (1.0 + 2.0 * risk)
        owned = set(roster)
        out = []
        for j in candidates:
            if j in owned:
                continue
            pos = self.positions[j]
            gain = sum(max(0.0, self.weekly[v][j] - floor[pos]) - loss
                       for v, loss, floor in zip(weeks, losses, floors)) / len(weeks)
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
            offers = bidding.offers(rosters[team], [j], budgets[team], w, len(teams[team].reserve), risk)
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
