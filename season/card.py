"""This week's bidding card: the claims to enter on Sleeper, optimized together.

Sleeper processes the room's claims highest bid first (a team's equal bids in the order
it lists them) and checks each against the roster as its earlier wins left it: a claim
fails when its player is already mine, its drop is gone, no regular spot is left for
him, or the cash is gone. A claimed body lands in a regular spot, never straight onto
reserve: the reserve holds what I put there before the run, so a drop from reserve frees
him nothing, and a body won earlier in the run keeps his regular spot until it is over.
So claims naming the same drop are alternatives (the first to win takes the spot), and
a player claimed several times with different drops has fallbacks for when an earlier
win used his drop. A free agent after the run is added now, ahead of every claim, by
hand, with the reserve reshuffled between adds.

The card is scored per recorded opponent season (race.simulate without me): each claim
wins or loses at that season's price, and the roster and cash the card leaves are valued
by that season's replay (race.replay), the same records the single claims are priced on.
Every roster a card reaches is replayed once, at the cash it leaves; the value of cash
around that point, and a roster not yet replayed, come from the base plus each of the
roster's swaps' single effect. That additive estimate is not good enough to decide on:
on the few seasons a rare roster is reached (a cheap win beside a lost one) it misses by
more than the differences between cards, so no card is judged on it.
The card grows greedily: every claim option (a player and one of his drops) is screened
by the additive estimate with its own bid optimized, alone and, when its drop is one a
claim on the card names, ahead of that claim together with the claim's fallback on
another of his drops (so a better drop for one player need not cost another player);
then the options with the best estimates are tried in turn, each of their extensions
replayed with every bid re-optimized by coordinate ascent over the clearing prices, and
the first whose gain over the card so far exceeds CONFIDENCE paired standard errors
joins; the card is done when none of the SCREENED best options' extensions does.
Claims that no longer help are pruned, and an equal bid ahead of a claim that depends on
it is raised a dollar so the order does not rest on Sleeper's tie rule.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from shared.league import RESERVE_SLOTS

SCREENED = 4  # options tried per step, best additive estimate first, until an extension joins
CONFIDENCE = 2.0  # paired standard errors an extension must gain to join the card
BID_LEVELS = 32  # clearing prices tried per claim while the card grows; the last pass tries every one


@dataclass(frozen=True)
class Claim:
    player: int
    drop: int | None
    bid: int
    free: bool = False  # a free agent after the run: added now, ahead of every claim

    @property
    def swap(self) -> tuple[int, int | None]:
        return self.player, self.drop


def depends(earlier: Claim, later: Claim) -> bool:
    """Whether `later`'s fate turns on `earlier` being processed first."""
    return earlier.player == later.player or (earlier.drop is not None and earlier.drop == later.drop)


class Values:
    """Per-season title values of my roster variants.

    `grids[roster]` holds a roster replayed at every budget of the grid, (budgets,
    seasons); `points[roster][budget]` one replayed at that budget alone. A roster is
    estimated as the base plus each of its swaps' single effect (`singles`, on the grid;
    claims.py lends a swap replayed at the full budget only its player's replayed drop's
    budget profile), and where it was replayed at one budget, that point corrects the
    estimate's level: the value of cash is the estimate's shape, the roster's worth the
    replay's.
    """

    def __init__(self, base: tuple[int, ...], budgets: list[int], base_grid: np.ndarray):
        self.base = base
        self.budgets = np.asarray(budgets, dtype=np.float64)  # ascending, the full budget last
        self.grids: dict[tuple[int, ...], np.ndarray] = {base: base_grid}
        self.points: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
        self.singles: dict[tuple[int, int | None], np.ndarray] = {}
        self.estimates: dict[tuple[int, ...], np.ndarray] = {}

    def learn(self, roster: tuple[int, ...], budget: int, values: np.ndarray) -> None:
        self.points.setdefault(roster, {})[budget] = values

    def known(self, roster: tuple[int, ...]) -> bool:
        """Replayed at some budget: the level is the replay's, only the shape of cash is estimated."""
        return roster in self.grids or roster in self.points

    def single(self, swap: tuple[int, int | None], effect: np.ndarray) -> None:
        self.singles[swap] = effect

    def roster_after(self, swaps) -> tuple[int, ...]:
        roster = set(self.base)
        for player, drop in swaps:
            roster.discard(drop)
            roster.add(player)
        return tuple(sorted(roster))

    def grid(self, roster: tuple[int, ...], swaps) -> np.ndarray:
        got = self.grids.get(roster)
        if got is None:
            got = self.estimates.get(roster)
            if got is None:
                got = self.grids[self.base].copy()
                for swap in swaps:
                    got += self.singles[swap]
                self.estimates[roster] = got
        return got

    def _interpolate(self, grid: np.ndarray, budget: float, idx: np.ndarray) -> np.ndarray:
        if len(self.budgets) == 1:
            return grid[0, idx]
        k = min(max(int(np.searchsorted(self.budgets, budget, side="right")) - 1, 0), len(self.budgets) - 2)
        t = (budget - self.budgets[k]) / (self.budgets[k + 1] - self.budgets[k])
        return grid[k, idx] * (1.0 - t) + grid[k + 1, idx] * t

    def at(self, roster: tuple[int, ...], swaps, budget: int, idx: np.ndarray) -> np.ndarray:
        """The seasons `idx` holding `roster` with `budget` left."""
        grid = self.grid(roster, swaps)
        value = self._interpolate(grid, budget, idx)
        points = self.points.get(roster)
        if points and roster not in self.grids:
            nearest = min(points, key=lambda b: abs(b - budget))
            value = value + points[nearest][idx] - self._interpolate(grid, nearest, idx)
        return value


class Scorer:
    """A card against the recorded seasons: Sleeper's processing per season, then the
    replayed value of whatever roster and cash it leaves.

    `outcomes[player]` is each season's clearing price (-1 untaken, -2 - k the k-th free
    pickup); a player absent from it was wanted by nobody. `eligible` holds the players
    the reserve slots may hold this week; `size` is the week's regular roster size.
    """

    def __init__(self, values: Values, outcomes: dict[int, np.ndarray], n: int, budget: int, size: int,
                 eligible: set[int]):
        self.values = values
        self.outcomes = outcomes
        self.n = n
        self.budget = budget
        self.size = size
        self.eligible = eligible
        self.count0 = len(values.base)
        self.elig0 = sum(1 for i in values.base if i in eligible)
        self._levels: dict[tuple[int, int | None], list[int]] = {}

    def order(self, card: list[Claim]) -> list[int]:
        """Processing order: free adds as listed, then bids high to low, equal bids as listed."""
        return sorted(range(len(card)), key=lambda k: (not card[k].free, -card[k].bid, k))

    def resolve(self, card: list[Claim]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per claim and season, whether it was still valid when reached and whether it
        won; per season, the cash left.

        Every add lands in a regular spot. The free adds come first, by hand: between them
        the reserve is refilled from the eligible bodies held, so an add fits when the
        others leave him a regular spot. The paid claims then process in one run with the
        reserve as it stood entering it, min(RESERVE_SLOTS, eligible) bodies, so a drop
        from reserve frees nothing: an eligible drop counts as active only while the
        eligible bodies outnumber the slots (the ones the card drops are left active
        first; which of several eligible drops is the active one is not tracked).
        """
        n = self.n
        has: dict[int, np.ndarray] = {}
        left = np.full(n, self.budget, dtype=np.int64)
        count = np.full(n, self.count0, dtype=np.int64)
        elig = np.full(n, self.elig0, dtype=np.int64)
        active = overflow = None  # regular spots filled and eligible bodies beyond the reserve, frozen at the run
        won = np.zeros((len(card), n), dtype=bool)
        reached = np.zeros((len(card), n), dtype=bool)
        for k in self.order(card):
            claim = card[k]
            mine = has.setdefault(claim.player, np.zeros(n, dtype=bool))
            live = ~mine & (left >= claim.bid)
            dropped = None
            if claim.drop is not None:
                dropped = has.setdefault(claim.drop, np.ones(n, dtype=bool))
                live &= dropped
            gained = claim.player in self.eligible
            lost = claim.drop is not None and claim.drop in self.eligible
            if claim.free:
                others = count - (claim.drop is not None)
                live &= others + 1 - np.minimum(RESERVE_SLOTS, elig - lost) <= self.size
            else:
                if active is None:
                    held = np.minimum(RESERVE_SLOTS, elig)
                    active, overflow = count - held, elig - held
                frees = overflow > 0 if lost else np.full(n, claim.drop is not None)
                after = active + 1 - frees
                live &= after <= self.size
            reached[k] = live
            if claim.free:
                ok = live
            else:
                prices = self.outcomes.get(claim.player)
                if prices is None:
                    ok = live.copy()  # nobody wanted him: any bid lands him
                else:
                    ok = live & ((prices < claim.bid) if claim.bid > 0 else (prices == -1))
            won[k] = ok
            mine |= ok
            if dropped is None:
                count += ok
            else:
                dropped &= ~ok
            elig += ok * (gained - lost)
            if not claim.free:
                active = np.where(ok, after, active)
                if lost:
                    overflow -= ok & frees
            left -= claim.bid * ok
        return won, reached, left

    def evaluate(self, card: list[Claim]) -> tuple[float, np.ndarray, dict]:
        """Mean title odds, each season's, and the resolution: what each claim did, the
        cash left, and the rosters reached with the seasons reaching them."""
        won, reached, left = self.resolve(card)
        codes = np.zeros(self.n, dtype=np.int64)
        for k in range(len(card)):
            codes |= won[k].astype(np.int64) << k
        titles = np.empty(self.n)
        rosters = {}
        for code in np.unique(codes):
            idx = np.flatnonzero(codes == code)
            swaps = [card[k].swap for k in range(len(card)) if code >> k & 1]
            roster = self.values.roster_after(swaps)
            budget = int(left[idx[0]])  # the same claims won: the same cash left
            titles[idx] = self.values.at(roster, swaps, budget, idx)
            rosters[int(code)] = (roster, swaps, budget, idx)
        return float(titles.mean()), titles, {"won": won, "reached": reached, "left": left, "rosters": rosters}

    def levels(self, player: int, thin: int | None) -> list[int]:
        """Bids worth trying: title odds are flat between clearing prices, so one above
        each; `thin` keeps that many, evenly spaced through the price distribution."""
        got = self._levels.get((player, thin))
        if got is None:
            prices = self.outcomes.get(player)
            levels = [0] if prices is None else sorted({0, 1} | {int(p) + 1 for p in np.unique(prices) if 0 <= p < self.budget})
            if thin is not None and len(levels) > thin:
                picks = np.linspace(0, len(levels) - 1, thin).round().astype(int)
                levels = [levels[i] for i in dict.fromkeys(picks.tolist())]
            got = self._levels[player, thin] = levels
        return got

    def optimize(self, card: list[Claim], which=None, thin: int | None = BID_LEVELS) -> list[Claim]:
        """Coordinate ascent on the bids of the claims `which` (all by default)."""
        card = list(card)
        which = range(len(card)) if which is None else which
        best = self.evaluate(card)[0]
        improved = True
        while improved:
            improved = False
            for k in which:
                if card[k].free:
                    continue
                for bid in self.levels(card[k].player, thin):
                    if bid == card[k].bid:
                        continue
                    trial = card[:]
                    trial[k] = dataclasses.replace(card[k], bid=bid)
                    value = self.evaluate(trial)[0]
                    if value > best + 1e-12:
                        best, card, improved = value, trial, True
        return card


def paired(titles: np.ndarray, against: np.ndarray) -> tuple[float, float]:
    """Mean gain of one card over another across the seasons, and its standard error."""
    diff = titles - against
    se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return float(diff.mean()), se


def _replay_reached(scorer: Scorer, cards: list[list[Claim]], replay) -> None:
    """Replay every roster these cards reach that has not been, at the cash it most often leaves."""
    values = scorer.values
    wanted: dict[tuple[int, ...], int] = {}
    for card in cards:
        reached = scorer.evaluate(card)[2]["rosters"].values()
        for roster, _, budget, idx in sorted(reached, key=lambda item: -len(item[3])):
            if not values.known(roster):
                wanted.setdefault(roster, budget)
    if wanted:
        for (roster, budget), titles in replay(sorted(wanted.items())).items():
            values.learn(roster, budget, titles)


def _extensions(card: list[Claim], option: Claim, options: list[Claim]):
    """Cards adding the option, with the indexes of what they add: the option after the
    card, and, for each claim on the card whose drop it takes, ahead of the card (so an
    equal bid processes it first) with that claim's fallback on another of his drops
    appended at the same bid."""
    yield card + [option], [len(card)]
    taken = {claim.swap for claim in card}
    for claim in card:
        if claim.drop is not None and claim.drop == option.drop:
            for other in options:
                if other.player == claim.player and other.swap != claim.swap and other.swap not in taken:
                    yield [option] + card + [dataclasses.replace(other, bid=claim.bid)], [0, len(card) + 1]


def _confirmed(scorer: Scorer, card: list[Claim], trial: list[Claim], replay) -> list[Claim] | None:
    """The trial with its bids settled on replayed values, if it gains on the card with confidence."""
    trial = scorer.optimize(trial)
    _replay_reached(scorer, [trial], replay)
    trial = scorer.optimize(trial)
    gain, se = paired(scorer.evaluate(trial)[1], scorer.evaluate(card)[1])
    return trial if gain > CONFIDENCE * se else None


def _extend(scorer: Scorer, options: list[Claim], replay) -> list[Claim]:
    card: list[Claim] = []
    while True:
        value = scorer.evaluate(card)[0]
        taken = {claim.swap for claim in card}
        screened = []  # each option's extensions, best estimate first, ranked by that best
        for k, option in enumerate(options):
            if option.swap in taken:
                continue
            trials = [scorer.optimize(trial, which=added) for trial, added in _extensions(card, option, options)]
            trials = sorted(((scorer.evaluate(trial)[0], trial) for trial in trials), key=lambda row: -row[0])
            screened.append((trials[0][0], k, trials))
        screened.sort(key=lambda row: (-row[0], row[1]))
        candidates = [trial for _, _, trials in screened[:SCREENED] for estimate, trial in trials if estimate > value]
        for trial in candidates:
            confirmed = _confirmed(scorer, card, trial, replay)
            if confirmed is not None:
                card = confirmed
                break
        else:
            return card


def _prune(scorer: Scorer, card: list[Claim]) -> list[Claim]:
    """Drop any claim the card does no worse without."""
    pruned = True
    while pruned and card:
        pruned = False
        value = scorer.evaluate(card)[0]
        for k in range(len(card)):
            without = card[:k] + card[k + 1:]
            if scorer.evaluate(without)[0] >= value:
                card, pruned = without, True
                break
    return card


def _break_ties(scorer: Scorer, card: list[Claim]) -> list[Claim]:
    """A dollar more on an equal bid ahead of a claim that depends on it, so Sleeper
    processes them in the card's order whatever its tie rule; cash permitting."""
    card = list(card)
    changed = True
    while changed:
        changed = False
        order = scorer.order(card)
        for a, k in enumerate(order):
            for later in order[a + 1:]:
                earlier, next_claim = card[k], card[later]
                if (not earlier.free and not next_claim.free and earlier.bid == next_claim.bid
                        and earlier.bid < scorer.budget and depends(earlier, next_claim)):
                    card[k] = dataclasses.replace(earlier, bid=earlier.bid + 1)
                    changed = True
                    break
            if changed:
                break
    return card


def build(scorer: Scorer, options: list[Claim], replay) -> list[Claim]:
    """The card from the claim `options` (each at any bid): grown greedily while a
    replayed extension gains with confidence, pruned, every bid settled over every
    clearing price, and ties broken. `replay(rosters and budgets)` returns each pair's
    per-season title values."""
    card = _prune(scorer, _extend(scorer, options, replay))
    if card:
        card = scorer.optimize(card, thin=None)
        _replay_reached(scorer, [card], replay)
        card = _break_ties(scorer, card)
    return [card[k] for k in scorer.order(card)]
