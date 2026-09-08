"""The in-season elimination race with a waiver market: rosters, budgets, and cuts.

One simulated season from the live state: each week every alive team fields its greedy
optimal lineup on that week's projections, scores it plus a persistent projection bias
(TEAM_SEASON_SIGMA) and weekly noise (WEEKLY_SIGMA, floored at SCORE_FLOOR_Z, the same
noise model as the draft's guillotine.py), and the two lowest are cut. The cut rosters
join the free-agent pool, and before the next week's games the survivors bid on it:

  * Each team prices the top CLAIM_CANDIDATES free agents by rest-of-season points as
    the per-week lineup gain over the starter each would displace (season.thresholds),
    and claims its CLAIMS_PER_TEAM best gains at league.py's bid rule — a share of its
    remaining budget, convex in that gain, tempered early in the season, with lognormal
    noise.
  * Claims resolve highest bid first. A claim fails if the player is gone or the budget
    no longer covers the bid. A win on a full roster drops the team's weakest
    rest-of-season player back into the pool.
  * Unclaimed players are free once claims clear: each survivor, in random order, then
    adds the best remaining candidate that still improves its lineup for $0.
  * Week 1 is free agency: every bid is $0 and ties are a coin flip.

Two runs of the same race answer two questions. Excluding me (`exclude_me`), the 31
opponents race among themselves and the record carries, per week, the elimination bar
(the second-lowest surviving opponent — beat it and I survive, whoever I am), the
finalist's championship score, and the market: which free agents were in play and what
each cleared for. `replay` then walks my roster through that record under my own claim
policy, so any roster-and-budget variant is priced against the same 2048 seasons, in
closed form per week (Phi over the bar) exactly as the draft valuation did. Including
me, all 32 race and every team's elimination and title odds are just frequencies.
"""

from __future__ import annotations

import heapq
import math
import multiprocessing
import os
import random
from dataclasses import dataclass

from .guillotine import SIGMA_CHAMP, SIGMA_WEEK, _cdf
from .league import (
    CLAIM_CANDIDATES,
    FAAB_HOLD_WEEKS,
    FAAB_SPEND_WEEK,
    CLAIM_CONSERVATION_FLOOR,
    CLAIM_CONSERVATION_FULL_WEEK,
    CLAIM_FULL_BUDGET_GAIN,
    CLAIM_GAIN_EXPONENT,
    CLAIM_NOISE_SIGMA,
    CLAIMS_PER_TEAM,
    REGULAR_WEEKS,
    SCORE_FLOOR_Z,
    TEAM_SEASON_SIGMA,
    WEEK_ROSTER_SIZE,
    WEEKS,
)
from .season import POS_CODE, SeasonState, lineup_points, thresholds


@dataclass(slots=True)
class RaceInputs:
    week0: int  # index of the week being decided
    positions: list[int]
    weekly: list[list[float]]  # [week][player]
    ros: list[list[float]]  # [week][player]: mean projection over that week and the rest
    rosters: list[list[int]]
    budgets: list[int]
    alive: list[bool]
    capacity_extra: list[int]  # reserve (IR) bodies each team carries beyond the roster size
    me: int
    free_agents: list[int]
    waivers_ran: bool


def race_inputs(state: SeasonState) -> RaceInputs:
    w0 = state.week - 1
    positions = [POS_CODE[p.position] for p in state.players]
    weekly = [[p.weekly[w] for p in state.players] for w in range(WEEKS)]
    ros = []
    for w in range(WEEKS):
        span = WEEKS - w
        ros.append([round(sum(p.weekly[w:]) / span, 2) for p in state.players])
    return RaceInputs(
        week0=w0,
        positions=positions,
        weekly=weekly,
        ros=ros,
        rosters=[list(t.roster) for t in state.teams],
        budgets=[t.faab_left for t in state.teams],
        alive=[t.alive for t in state.teams],
        capacity_extra=[len(t.reserve) for t in state.teams],
        me=state.me,
        free_agents=list(state.free_agents),
        waivers_ran=state.waivers_ran,
    )


def bid_for(gain: float, budget: int, w: int, rng: random.Random) -> int:
    """A team's claim for a free agent worth `gain` points a week, from `budget`,
    before week index `w`'s games. Week 1 is free agency."""
    if w == 0 or budget <= 0:
        return 0
    share = min(1.0, gain / CLAIM_FULL_BUDGET_GAIN) ** CLAIM_GAIN_EXPONENT
    ramp = min(1.0, w / (CLAIM_CONSERVATION_FULL_WEEK - 1))
    conserve = CLAIM_CONSERVATION_FLOOR + (1.0 - CLAIM_CONSERVATION_FLOOR) * ramp
    return min(budget, int(budget * share * conserve * math.exp(rng.gauss(0.0, CLAIM_NOISE_SIGMA))))


def my_bid_for(gain: float, budget: int, w: int, rng: random.Random) -> int:
    """My FAAB policy (league.FAAB_HOLD_WEEKS / FAAB_SPEND_WEEK): free pickups only while
    I hold the budget, the room's rule through the lineup expansion, and the room's rule
    without its early-season tempering once the saved budget is meant to go."""
    if w < FAAB_HOLD_WEEKS:
        return 0
    if w < FAAB_SPEND_WEEK - 1:
        return bid_for(gain, budget, w, rng)
    share = min(1.0, gain / CLAIM_FULL_BUDGET_GAIN) ** CLAIM_GAIN_EXPONENT
    return min(budget, int(budget * share * math.exp(rng.gauss(0.0, CLAIM_NOISE_SIGMA))))


POLICIES = {"room": bid_for, "hold": my_bid_for}


def _add_player(
    roster: list[int], player: int, w: int, ros_w: list[float], extra: int
) -> int | None:
    """Add to a roster before week `w`, dropping the weakest rest-of-season body when
    the roster is over the week's size. Returns the dropped player, if any."""
    roster.append(player)
    if len(roster) <= WEEK_ROSTER_SIZE[w] + extra:
        return None
    drop = min((i for i in roster if i != player), key=lambda i: (ros_w[i], i))
    roster.remove(drop)
    return drop


def _auction(
    inputs: RaceInputs,
    w: int,
    rosters: list[list[int]],
    budgets: list[int],
    alive: list[bool],
    free: set[int],
    rng: random.Random,
    skip: int | None,
) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    """One week's claims among the alive teams (except `skip`). Returns the candidates
    in play, the outcome on each (the winning bid; -1 if nobody took him; -2 - k if he
    went as the k-th free pickup), and (player, team, bid) wins."""
    ros_w = inputs.ros[w]
    positions = inputs.positions
    candidates = heapq.nlargest(CLAIM_CANDIDATES, free, key=lambda i: (ros_w[i], -i))
    bids: list[tuple[int, float, int, int]] = []
    for team in range(len(rosters)):
        if not alive[team] or team == skip:
            continue
        thr = thresholds(rosters[team], ros_w, positions, w)
        budget = budgets[team]
        gains = [(ros_w[j] - thr[positions[j]], j) for j in candidates]
        for gain, j in heapq.nlargest(CLAIMS_PER_TEAM, gains):
            if gain > 0.0:
                bids.append((bid_for(gain, budget, w, rng), rng.random(), team, j))
    bids.sort(reverse=True)
    winning = {j: -1 for j in candidates}
    wins: list[tuple[int, int, int]] = []
    for bid, _, team, j in bids:
        if winning[j] >= 0 or bid > budgets[team]:
            continue
        winning[j] = bid
        budgets[team] -= bid
        free.discard(j)
        dropped = _add_player(rosters[team], j, w, ros_w, inputs.capacity_extra[team])
        if dropped is not None:
            free.add(dropped)
        wins.append((j, team, bid))
    # Unclaimed candidates are free agents now: one free pickup per team, random order.
    teams = [t for t in range(len(rosters)) if alive[t] and t != skip]
    rng.shuffle(teams)
    taken = 0
    for team in teams:
        thr = thresholds(rosters[team], ros_w, positions, w)
        best, best_gain = None, 0.0
        for j in candidates:
            if winning[j] == -1:
                gain = ros_w[j] - thr[positions[j]]
                if gain > best_gain:
                    best, best_gain = j, gain
        if best is not None:
            winning[best] = -2 - taken
            taken += 1
            free.discard(best)
            dropped = _add_player(rosters[team], best, w, ros_w, inputs.capacity_extra[team])
            if dropped is not None:
                free.add(dropped)
            wins.append((best, team, 0))
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

    cut_week = [None] * n
    alive_count: list[int] = [0] * WEEKS
    bars: list[float] = [0.0] * WEEKS
    auctions: list[tuple[list[int], list[int]] | None] = [None] * WEEKS
    claims: list[list[tuple[int, int, int]]] = [[] for _ in range(WEEKS)]
    budget_path: list[list[int]] = []
    for w in range(w0, WEEKS):
        if not (w == w0 and inputs.waivers_ran):
            candidates, winning, wins = _auction(inputs, w, rosters, budgets, alive, free, rng, skip)
            auctions[w] = (candidates, winning)
            claims[w] = wins
        budget_path.append(list(budgets))
        alive_count[w] = sum(alive)
        if w >= REGULAR_WEEKS:
            continue
        sigma = SIGMA_WEEK[w]
        floor = SCORE_FLOOR_Z * math.hypot(TEAM_SEASON_SIGMA, sigma)
        points = inputs.weekly[w]
        scored = sorted(
            (
                lineup_points(rosters[i], points, positions, w)
                + max(bias[i] + rng.gauss(0.0, sigma), floor),
                i,
            )
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
    totals = {
        i: lineup_points(rosters[i], inputs.weekly[REGULAR_WEEKS], positions, REGULAR_WEEKS)
        + lineup_points(rosters[i], inputs.weekly[REGULAR_WEEKS + 1], positions, REGULAR_WEEKS + 1)
        + 2.0 * bias[i]
        + rng.gauss(0.0, SIGMA_CHAMP)
        for i in finalists
    }
    champion = max(finalists, key=lambda i: (totals[i], -i))
    return {
        "seed": seed,
        "my_bias": bias[inputs.me],
        "bars": bars,
        "alive": alive_count,
        "champ_bar": totals[champion] if exclude_me else None,
        "auctions": auctions,
        "claims": claims,
        "cut_week": cut_week,
        "champion": champion,
        "budget_path": budget_path,
    }


# --- my replay through a recorded race ---------------------------------------------


def replay(
    records: list[dict],
    inputs: RaceInputs,
    roster: list[int],
    budget: int,
    policy: str,
) -> dict:
    """P(title) and the weekly survival profile for a roster-and-budget variant of my
    team, against every recorded opponent race. The variant already reflects this
    week's claim outcome, so my agent bids only from next week on, under `policy`
    (POLICIES); each week's survival is Phi over that record's bar."""
    bid_rule = POLICIES[policy]
    w0 = inputs.week0
    positions = inputs.positions
    extra = inputs.capacity_extra[inputs.me]
    title = 0.0
    reach = 0.0
    alive_by_week = [0.0] * REGULAR_WEEKS
    hazard = [0.0] * REGULAR_WEEKS  # P(cut in week w and alive entering it)
    budget_left = [0.0] * WEEKS
    titles: list[float] = []
    for rec in records:
        mine = list(roster)
        left = budget
        surv = 1.0
        for w in range(w0, WEEKS):
            auction = rec["auctions"][w]
            if w > w0 and auction is not None:
                # Reseeded per week so roster variants share their bid draws: the
                # difference between two variants is then roster, not dice.
                rng = random.Random(rec["seed"] * WEEKS + w)
                ros_w = inputs.ros[w]
                # My place in the free-pickup queue: the k-th free pickup of the record
                # is still there for me if I am ahead of the team that took him.
                place = rng.randrange(rec["alive"][w] + 1)

                def open_to_me(outcome: int) -> bool:
                    return outcome == -1 or (outcome <= -2 and -2 - outcome >= place)

                thr = thresholds(mine, ros_w, positions, w)
                gains = [(ros_w[j] - thr[positions[j]], j, outcome) for j, outcome in zip(*auction)]
                for gain, j, outcome in heapq.nlargest(CLAIMS_PER_TEAM, gains):
                    if gain <= 0.0 or j in mine:
                        continue
                    bid = bid_rule(gain, left, w, rng)
                    # A paid claim beats any free pickup; a $0 claim is a free pickup.
                    if (bid > outcome) if outcome >= 0 else (bid > 0 or open_to_me(outcome)):
                        left -= bid
                        _add_player(mine, j, w, ros_w, extra)
                thr = thresholds(mine, ros_w, positions, w)
                free_gain, free_pick = 0.0, None
                for gain, j, outcome in gains:
                    if outcome < 0 and j not in mine and open_to_me(outcome):
                        gain = ros_w[j] - thr[positions[j]]
                        if gain > free_gain:
                            free_gain, free_pick = gain, j
                if free_pick is not None:
                    _add_player(mine, free_pick, w, ros_w, extra)
            budget_left[w] += left
            if w >= REGULAR_WEEKS:
                continue
            mu = lineup_points(mine, inputs.weekly[w], positions, w)
            # The same floored deviation the race gives every team: a blowup week
            # bottoms out at SCORE_FLOOR_Z, so a margin wider than that is safe.
            sigma = SIGMA_WEEK[w]
            if mu + SCORE_FLOOR_Z * math.hypot(TEAM_SEASON_SIGMA, sigma) >= rec["bars"][w]:
                p = 1.0
            else:
                p = _cdf((mu + rec["my_bias"] - rec["bars"][w]) / sigma)
            hazard[w] += surv * (1.0 - p)
            surv *= p
            alive_by_week[w] += surv
        champ = (
            lineup_points(mine, inputs.weekly[REGULAR_WEEKS], positions, REGULAR_WEEKS)
            + lineup_points(mine, inputs.weekly[REGULAR_WEEKS + 1], positions, REGULAR_WEEKS + 1)
            + 2.0 * rec["my_bias"]
        )
        reach += surv
        won = surv * _cdf((champ - rec["champ_bar"]) / SIGMA_CHAMP)
        title += won
        titles.append(won)
    n = len(records)
    return {
        "p_title": title / n,
        "title_by_record": titles,
        "p_reach_final": reach / n,
        "p_cut_now": hazard[w0] / n,
        "p_alive_by_week": [a / n for a in alive_by_week[w0:]],
        "budget_by_week": [b / n for b in budget_left[w0:]],
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


def _workers() -> int:
    return max(1, min(16, os.cpu_count() or 1))


def run_races(inputs: RaceInputs, sims: int, seed: int) -> tuple[list[dict], list[dict]]:
    """(records excluding me, records of the full 32-team race), `sims` seasons each."""
    tasks = [(seed + s, True) for s in range(sims)] + [(seed + s, False) for s in range(sims)]
    with multiprocessing.Pool(_workers(), initializer=_init, initargs=(inputs, None)) as pool:
        out = pool.map(_simulate_task, tasks, chunksize=32)
    return out[:sims], out[sims:]


def run_replays(
    inputs: RaceInputs, records: list[dict], variants: list[tuple[tuple[int, ...], int, str]]
) -> list[dict]:
    """One replay per (roster, budget, policy) variant, in parallel."""
    with multiprocessing.Pool(_workers(), initializer=_init, initargs=(inputs, records)) as pool:
        return pool.map(_replay_task, variants, chunksize=1)
