"""The in-season elimination race with a waiver market: rosters, budgets, and cuts.

One simulated season from the live state: each week every alive team fields its greedy
optimal lineup on that week's projections, scores it plus a persistent projection bias
(TEAM_SEASON_SIGMA) and weekly noise (WEEKLY_SIGMA, floored at SCORE_FLOOR_Z, the same
noise model as the draft's guillotine.py), and the two lowest are cut. The cut rosters
join the free-agent pool, and before the next week's games the survivors bid on it.

Bids use season-long roster values, guide ceilings and manager behavior from
waivers.py. Opponents have per-manager participation and the room's price curve, both
learned from submitted bids, and persistent sampled saving habits; my agent submits an
offer for every candidate improving my objective (claims.title_objective) within its
guide ceiling and saving allowance. Claims naming the same drop are alternatives, open
spots and remaining cash are checked as claims resolve, and a claim that no longer
improves the roster after an earlier win is passed over. The reserve slots hold
Out/IR/PUP bodies while their projection is zero; when one resumes, the team cuts its
least valuable body to make room before that week's claims. Week 1 is free agency.
After this week's run, the players dropped since are still on waivers: they are
auctioned off-cycle, among opponents at the observed mid-week share of their
participation (waivers.off_cycle_share), and everyone else stays free.

Two runs of the same race answer two questions. Excluding me (`exclude_me`), the 31
opponents race among themselves and the record carries, per week, the elimination bar
(the second-lowest surviving opponent: beat it and I survive, whoever I am), the
finalist's championship score, and the market: which free agents were in play and what
each cleared for. `replay` then walks my roster through that record under my own claim
policy, so any roster-and-budget variant is priced against the same seasons, in closed
form per week (Phi over the bar) exactly as the draft valuation did. Including me, all
32 race and every team's elimination and title odds are just frequencies.
"""

from __future__ import annotations

import heapq
import math
import multiprocessing
import random
from dataclasses import dataclass, field

from shared.league import REGULAR_WEEKS, WEEKS
from shared.noise import SCORE_FLOOR_Z, SIGMA_CHAMP, SIGMA_WEEK, TEAM_SEASON_SIGMA, cdf, phi
from shared.workers import worker_count

from .state import POS_CODE, SeasonState, lineup_points
from .waivers import (ROOM_CURRENT_WEEK_WEIGHT, SAVING_PLANS, Bidding, Manager, PriceCurve, bid_observations,
                      calibration, fit_managers, guide_reference, off_cycle_share, projected_bar,
                      projected_risk, spending_allowance)

RACE_SIMS = 2048
CLAIMS_PER_TEAM = 3  # an opponent's claims per auction
CLAIM_CANDIDATES = 80  # free agents in play each week, by rest-of-season points
POLICIES = tuple(SAVING_PLANS)


@dataclass(slots=True)
class RaceInputs:
    week0: int  # index of the week being decided
    positions: list[int]
    weekly: list[list[float]]  # [week][player]
    ros: list[list[float]]  # [week][player]: mean projection over that week and the rest
    rosters: list[list[int]]
    budgets: list[int]
    alive: list[bool]
    me: int
    free_agents: list[int]
    waivers_ran: bool
    bidding: Bidding
    managers: list[Manager]
    market_fit: dict
    opening_budgets: list[int]  # actual cash entering the current week, before completed claims
    policy: str = "value"
    my_bidding: Bidding | None = None  # mine, without the room's current-week weight; claims.title_objective sets its weeks
    price_curve: PriceCurve = PriceCurve()  # the room's bid for a guide reference
    # After the weekly run: players still on waivers -> when they clear, and opponents'
    # participation in that off-cycle auction relative to a weekly run's.
    on_waivers: dict[int, str] = field(default_factory=dict)
    off_cycle_share: float = 0.0

    def __post_init__(self):
        if self.my_bidding is None:
            self.my_bidding = self.bidding

    def bidding_for(self, team: int) -> Bidding:
        return self.my_bidding if team == self.me else self.bidding


def race_inputs(state: SeasonState) -> RaceInputs:
    w0 = state.week - 1
    positions = [POS_CODE[p.position] for p in state.players]
    weekly = [[p.weekly[w] for p in state.players] for w in range(WEEKS)]
    ros = [[round(sum(p.weekly[w:]) / (WEEKS - w), 2) for p in state.players] for w in range(WEEKS)]
    ir_until = [p.ir_until for p in state.players]
    bidding = Bidding(positions, weekly, ros, ir_until, ROOM_CURRENT_WEEK_WEIGHT)
    observations = bid_observations(state, bidding)
    curve, managers = fit_managers(state, observations)
    spent = {t.roster_id: 0 for t in state.teams}
    for tx in state.transactions:
        if tx["week"] == state.week and tx["type"] == "waiver" and tx["status"] == "complete":
            spent[tx["roster_id"]] += tx["bid"] or 0
    return RaceInputs(
        week0=w0,
        positions=positions,
        weekly=weekly,
        ros=ros,
        rosters=[list(t.roster) for t in state.teams],
        budgets=[t.faab_left for t in state.teams],
        alive=[t.alive for t in state.teams],
        me=state.me,
        free_agents=list(state.free_agents),
        waivers_ran=state.waivers_ran,
        bidding=bidding,
        my_bidding=Bidding(positions, weekly, ros, ir_until),
        managers=managers,
        market_fit=calibration(state, observations),
        opening_budgets=[t.faab_left + spent[t.roster_id] for t in state.teams],
        price_curve=curve,
        on_waivers=dict(state.on_waivers),
        off_cycle_share=off_cycle_share(state) if state.on_waivers else 0.0,
    )


def fit_roster(bidding: Bidding, roster: list[int], w: int, free: set[int] | None = None) -> None:
    """Cut until the roster fits week `w`: a reserve body whose projection resumed needs a
    regular spot, and the cuts (the cheapest lineup losses) hit the wire when `free` is given."""
    while (drop := bidding.crunch(roster, w)) is not None:
        roster.remove(drop)
        if free is not None:
            free.add(drop)


def forecast_bar(inputs, w, rosters, alive):
    """Expected cut score from projected lineups; bids cannot see realized scores."""
    return projected_bar(rosters, alive, inputs.weekly[w], inputs.positions, w)


def cut_risk(inputs, roster, w, bar):
    return projected_risk(roster, inputs.weekly[w], inputs.positions, w, bar)


def claim_plan(inputs, bidding, roster, candidates, budget, w, risk, rng, opponent=False,
               policy="value", initial_budget=None):
    """(bid, offer) pairs, largest bid first, and the auction's spending allowance. My
    agent offers on every improving candidate at its ceiling; an opponent picks its best
    few by noisy gain and bids the room's price for each."""
    offers = bidding.offers(roster, candidates, budget, w, risk)
    allowance = spending_allowance(budget, budget if initial_budget is None else initial_budget,
                                   inputs.week0, w, policy, risk)
    if not opponent:
        plan = [(min(allowance, int(o.ceiling)), o) for o in offers]
        allowance = max((bid for bid, _ in plan), default=0)
    else:
        chosen = heapq.nlargest(CLAIMS_PER_TEAM, offers,
                               key=lambda o: o.gain * math.exp(rng.gauss(0, 0.5)))
        curve = inputs.price_curve
        plan = [(min(allowance, int(curve.price(guide_reference(bidding, o.player, o.ceiling, budget, w))
                                    * math.exp(rng.gauss(0, curve.sigma)))), o)
                for o in chosen]
    if w == 0:
        plan = [(0, offer) for _, offer in plan]
    return sorted(plan, key=lambda item: (-item[0], -item[1].gain, item[1].player)), allowance


def apply_offer(roster, offer):
    if offer.drop is not None:
        roster.remove(offer.drop)
    roster.append(offer.player)


def _resolve(bidding, roster, plan, budget, allowance, w, won) -> tuple[int, int]:
    """Walk one team's plan in bid order: `won(bid, offer)` says whether the bid wins its
    player; a win is applied unless it no longer fits or improves the roster as it stands
    after earlier wins. Returns the cash and allowance left."""
    still_improves = None  # None until the roster changes; then the players that still gain
    for k, (bid, offer) in enumerate(plan):
        j = offer.player
        if bid > min(budget, allowance) or j in roster:
            continue
        if offer.drop is not None and offer.drop not in roster:
            continue  # Claims naming the same drop are alternatives, as on Sleeper.
        if offer.drop is None and not bidding.fits(roster, j, w):
            continue
        if still_improves is not None and j not in still_improves:
            continue
        if not won(bid, offer):
            continue
        apply_offer(roster, offer)
        budget -= bid
        allowance -= bid
        later = [o.player for _, o in plan[k + 1:]]
        still_improves = {p for p, (gain, _) in zip(later, bidding.evaluate(roster, later, w)) if gain > 0.0}
    return budget, allowance


def _auction(inputs, w, rosters, budgets, alive, free, rng, skip, bar, policies, pool=None, attention=1.0):
    """One week's claims, then free pickups. An off-cycle auction limits the players in
    play to `pool` and scales opponents' participation by `attention`."""
    candidates = heapq.nlargest(CLAIM_CANDIDATES, free if pool is None else pool, key=lambda i: (inputs.ros[w][i], -i))
    bids = []
    allowances = list(budgets)
    active = []
    for team, roster in enumerate(rosters):
        if not alive[team] or team == skip:
            continue
        mine = team == inputs.me
        if not mine and rng.random() >= attention * inputs.managers[team].participation(w, inputs.week0):
            continue
        active.append(team)
        plan, allowances[team] = claim_plan(
            inputs, inputs.bidding_for(team), roster, candidates, budgets[team], w,
            cut_risk(inputs, roster, w, bar), rng, not mine,
            policies[team], inputs.budgets[team])
        for bid, offer in plan:
            bids.append((bid, rng.random(), team, offer))
    bids.sort(key=lambda row: (-row[0], row[1]))
    winning = {j: -1 for j in candidates}
    wins = []
    # Sleeper processes the whole room's claims in bid order; a team's later claims are
    # checked against its roster as its earlier wins left it.
    queue: dict[int, list[int]] = {}
    for _, _, team, offer in bids:
        queue.setdefault(team, []).append(offer.player)
    still_improves: dict[int, set[int]] = {}
    for bid, _, team, offer in bids:
        j = offer.player
        queue[team].remove(j)
        bidding = inputs.bidding_for(team)
        if winning[j] != -1 or j not in free or bid > min(budgets[team], allowances[team]):
            continue
        if offer.drop is not None and offer.drop not in rosters[team]:
            continue  # Claims naming the same drop are alternatives, as on Sleeper.
        if offer.drop is None and not bidding.fits(rosters[team], j, w):
            continue
        if team in still_improves and j not in still_improves[team]:
            continue
        apply_offer(rosters[team], offer)
        budgets[team] -= bid
        allowances[team] -= bid
        free.remove(j)
        if offer.drop is not None:
            free.add(offer.drop)
        winning[j] = bid
        wins.append((j, team, bid))
        later = [p for p in queue[team] if winning[p] == -1]
        still_improves[team] = {p for p, (gain, _) in zip(later, bidding.evaluate(rosters[team], later, w)) if gain > 0.0}
    rng.shuffle(active)
    taken = 0
    for team in active:
        offers = inputs.bidding_for(team).offers(rosters[team], [j for j in candidates if winning[j] == -1],
                                                 budgets[team], w)
        if not offers:
            continue
        offer = max(offers, key=lambda o: (o.gain, -o.player))
        apply_offer(rosters[team], offer)
        free.discard(offer.player)
        if offer.drop is not None:
            free.add(offer.drop)
        winning[offer.player] = -2 - taken
        taken += 1
        wins.append((offer.player, team, 0))
    return candidates, [winning[j] for j in candidates], wins


def simulate(inputs: RaceInputs, seed: int, exclude_me: bool) -> dict:
    """One season. With `exclude_me` the record is the market and bars I would face;
    otherwise every team's fate, budgets and claims."""
    rng = random.Random(seed)
    n = len(inputs.rosters)
    rosters = [list(r) for r in inputs.rosters]
    budgets = list(inputs.budgets)
    alive = list(inputs.alive)
    skip = inputs.me if exclude_me else None
    if exclude_me:
        alive[inputs.me] = False
    free = set(inputs.free_agents)
    bias = [rng.gauss(0.0, TEAM_SEASON_SIGMA) for _ in range(n)]
    w0 = inputs.week0
    positions = inputs.positions

    # A manager keeps one saving strategy for the season, independent of bid noise.
    policies = [inputs.policy if i == inputs.me else rng.choice(POLICIES) for i in range(n)]
    forecast_bars = [0.0] * WEEKS
    cut_week = [None if live else w0 - 1 for live in alive]
    alive_count: list[int] = [0] * WEEKS
    bars: list[float] = [0.0] * WEEKS
    auctions: list[tuple[list[int], list[int]] | None] = [None] * WEEKS
    claims: list[list[tuple[int, int, int]]] = [[] for _ in range(WEEKS)]
    budget_path: list[list[int]] = []
    budget_after_claims: list[list[int]] = []
    championship = [0.0] * n
    for w in range(w0, WEEKS):
        for i in range(n):
            if alive[i]:
                fit_roster(inputs.bidding_for(i), rosters[i], w, free)
        budget_path.append(list(inputs.opening_budgets if w == w0 else budgets))
        forecast_bars[w] = forecast_bar(inputs, w, rosters, alive)
        # Once this week's run is done, only players dropped since are auctioned, as they clear.
        off_cycle = w == w0 and inputs.waivers_ran
        if not off_cycle or inputs.on_waivers:
            candidates, winning, wins = _auction(
                inputs, w, rosters, budgets, alive, free, rng, skip, forecast_bars[w], policies,
                list(inputs.on_waivers) if off_cycle else None, inputs.off_cycle_share if off_cycle else 1.0)
            auctions[w] = (candidates, winning)
            claims[w] = wins
        budget_after_claims.append(list(budgets))
        alive_count[w] = sum(alive)
        if w >= REGULAR_WEEKS:
            for i in range(n):
                if alive[i]:
                    championship[i] += lineup_points(rosters[i], inputs.weekly[w], positions, w)
            continue
        sigma = SIGMA_WEEK[w]
        floor = SCORE_FLOOR_Z * math.hypot(TEAM_SEASON_SIGMA, sigma)
        points = inputs.weekly[w]
        scored = sorted(
            (lineup_points(rosters[i], points, positions, w) + max(bias[i] + rng.gauss(0.0, sigma), floor), i)
            for i in range(n)
            if alive[i]
        )
        opponents = [s for s, i in scored if i != inputs.me]
        bars[w] = opponents[1] if len(opponents) > 1 else opponents[0]
        for _, i in scored[:2]:
            alive[i] = False
            cut_week[i] = w
            free.update(rosters[i])
            rosters[i] = []

    finalists = [i for i in range(n) if alive[i]]
    totals = {i: championship[i] + 2.0 * bias[i] + rng.gauss(0.0, SIGMA_CHAMP) for i in finalists}
    champion = max(finalists, key=lambda i: (totals[i], -i))
    return {
        "seed": seed,
        "my_bias": bias[inputs.me],
        "forecast_bars": forecast_bars,
        "bars": bars,
        "alive": alive_count,
        "champ_bar": totals[champion] if exclude_me else None,
        "auctions": auctions,
        "claims": claims,
        "cut_week": cut_week,
        "champion": champion,
        "budget_path": budget_path,
        "budget_after_claims": budget_after_claims,
    }


# --- my replay through a recorded race ---------------------------------------------


def replay(records: list[dict], inputs: RaceInputs, roster: list[int], budget: int, policy: str) -> dict:
    """P(title) and the weekly survival profile for a roster-and-budget variant of my
    team, against every recorded opponent race. The variant already reflects this
    week's claim outcome, so my agent bids only from next week on, under `policy`
    (POLICIES); each week's survival is Phi over that record's bar. `leverage` is
    d log P(title) / d(my expected points) per week, as the draft's week weights."""
    bidding = inputs.my_bidding
    w0 = inputs.week0
    positions = inputs.positions
    title = 0.0
    reach = 0.0
    alive_by_week = [0.0] * REGULAR_WEEKS
    hazard = [0.0] * REGULAR_WEEKS  # P(cut in week w and alive entering it)
    budget_left = [0.0] * WEEKS
    budget_after_claims = [0.0] * WEEKS
    budget_weight = [0.0] * WEEKS
    titles: list[float] = []
    leverage = [0.0] * WEEKS
    for rec in records:
        mine = list(roster)
        left = budget
        surv = 1.0
        champ = 0.0
        rates = [0.0] * WEEKS  # d log P(title in this record) / d(points in week w)
        for w in range(w0, WEEKS):
            fit_roster(bidding, mine, w)
            budget_left[w] += surv * (inputs.opening_budgets[inputs.me] if w == w0 else left)
            budget_weight[w] += surv
            auction = rec["auctions"][w]
            if w > w0 and auction is not None:
                # Reseeded per week so roster variants share their bid draws: the
                # difference between two variants is then roster, not dice.
                rng = random.Random(rec["seed"] * WEEKS + w)
                # My place in the free-pickup queue: the k-th free pickup of the record
                # is still there for me if I am ahead of the team that took him.
                place = rng.randrange(rec["alive"][w] + 1)
                outcomes = dict(zip(*auction))

                def open_to_me(outcome: int) -> bool:
                    return outcome == -1 or (outcome <= -2 and -2 - outcome >= place)

                def won(bid: int, offer) -> bool:
                    outcome = outcomes[offer.player]
                    # A paid claim beats any free pickup; a $0 claim is a free pickup.
                    return (bid > outcome) if outcome >= 0 else (bid > 0 or open_to_me(outcome))

                plan, allowance = claim_plan(
                    inputs, bidding, mine, auction[0], left, w,
                    cut_risk(inputs, mine, w, rec["forecast_bars"][w]), rng,
                    policy=policy, initial_budget=budget)
                left, _ = _resolve(bidding, mine, plan, left, allowance, w, won)
                offers = bidding.offers(mine, [j for j, outcome in outcomes.items()
                                               if outcome < 0 and open_to_me(outcome)], left, w)
                if offers:
                    apply_offer(mine, max(offers, key=lambda o: (o.gain, -o.player)))
            budget_after_claims[w] += surv * left
            if w >= REGULAR_WEEKS:
                champ += lineup_points(mine, inputs.weekly[w], positions, w)
                continue
            mu = lineup_points(mine, inputs.weekly[w], positions, w)
            # The same floored deviation the race gives every team: a blowup week
            # bottoms out at SCORE_FLOOR_Z, so a margin wider than that is safe.
            sigma = SIGMA_WEEK[w]
            if mu + SCORE_FLOOR_Z * math.hypot(TEAM_SEASON_SIGMA, sigma) >= rec["bars"][w]:
                p = 1.0
            else:
                z = (mu + rec["my_bias"] - rec["bars"][w]) / sigma
                p = cdf(z)
                if p > 0.0:
                    rates[w] = phi(z) / (sigma * p)
            hazard[w] += surv * (1.0 - p)
            surv *= p
            alive_by_week[w] += surv
        champ += 2.0 * rec["my_bias"]
        reach += surv
        z = (champ - rec["champ_bar"]) / SIGMA_CHAMP
        p = cdf(z)
        if p > 0.0:
            for w in range(max(w0, REGULAR_WEEKS), WEEKS):
                rates[w] = phi(z) / (SIGMA_CHAMP * p)
        won_title = surv * p
        title += won_title
        titles.append(won_title)
        for w in range(w0, WEEKS):
            leverage[w] += won_title * rates[w]
    n = len(records)
    return {
        "p_title": title / n,
        "title_by_record": titles,
        "leverage": [x / title if title > 0.0 else 0.0 for x in leverage[w0:]],
        "p_reach_final": reach / n,
        "p_cut_now": hazard[w0] / n if w0 < REGULAR_WEEKS else 0.0,
        "p_alive_by_week": [a / n for a in alive_by_week[w0:]],
        "budget_by_week": [b / weight if weight else None
                           for b, weight in zip(budget_left[w0:], budget_weight[w0:])],
        "budget_after_claims": [b / weight if weight else None
                                for b, weight in zip(budget_after_claims[w0:], budget_weight[w0:])],
    }


# --- worker pool -----------------------------------------------------------------------

_INPUTS: RaceInputs | None = None
_RECORDS: list[dict] | None = None


def _init(inputs: RaceInputs, records: list[dict] | None) -> None:
    global _INPUTS, _RECORDS
    _INPUTS = inputs
    _RECORDS = records


def _simulate_task(task: tuple[int, bool]) -> dict:
    seed, exclude_me = task
    return simulate(_INPUTS, seed, exclude_me)


def _replay_task(task: tuple[tuple[int, ...], int, str]) -> dict:
    roster, budget, policy = task
    return replay(_RECORDS, _INPUTS, list(roster), budget, policy)


def run_race(inputs: RaceInputs, sims: int, seed: int, exclude_me: bool) -> list[dict]:
    with multiprocessing.Pool(worker_count(), initializer=_init, initargs=(inputs, None)) as pool:
        return pool.map(_simulate_task, [(seed + s, exclude_me) for s in range(sims)], chunksize=32)


def run_replays(
    inputs: RaceInputs, records: list[dict], variants: list[tuple[tuple[int, ...], int, str]]
) -> list[dict]:
    """One replay per (roster, budget, policy) variant, in parallel."""
    if not variants:
        return []
    with multiprocessing.Pool(worker_count(), initializer=_init, initargs=(inputs, records)) as pool:
        return pool.map(_replay_task, variants, chunksize=1)
