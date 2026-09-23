"""This week's decisions for my roster: the lineup and the waiver claims.

Lineup: the greedy optimum on this week's projections (season.lineup), compared with
the starters Sleeper currently has set for me.

Objective: a drop's spot can be refilled from the wire the room leaves untaken, and my
bidding values each week's points alike or by d log P(title) / d(points), whichever
replays to better title odds (title_objective). Screening, drops, ceilings and my
future policy use it; this week's bids are chosen by replayed title odds.

Claims: choose the bid that maximizes replayed title odds anywhere in the budget. The
guide-based ceiling caps only my future bids inside the replays (and the room's): for
this week's claim it undersold players the room buys (a depth upgrade capped at $0 that
the room claims nine times in ten at a median $25).
Each candidate's drop is chosen by replay among the few that waivers.Bidding ranks best:
the heuristic prices a fixed roster, where a future lineup expansion is an empty seat,
so its favorite can give up a starter today for depth the wire would supply anyway.
The chosen roster variant replays the same opponent seasons at several budgets under
spending and saving plans. After this week's run, only players dropped since are still
on waivers and take a bid; everyone else is a free add.
A variant that only fits with a body moved onto reserve names that move. A record
contributes the acquired roster's title value if the bid wins, and standing pat
otherwise. This keeps the dependence between prices and future opportunities. Odds are
relative to standing pat; replay levels are approximate because opponents retain
players acquired by our variant.
"""

from __future__ import annotations

import dataclasses
import heapq
import statistics
from collections import Counter

from .league import CLAIM_CANDIDATES, WEEK_ROSTER_SIZE, WEEKS
from .race import POLICIES, RaceInputs, apply_offer, cut_risk, fit_roster, run_replays
from .season import SeasonState, lineup

BUDGET_STEPS = (0, 25, 50, 100, 200, 400, 700)
DROP_CHOICES = 3  # drops per candidate compared by replay, best heuristic gain first
CANDIDATES_BY_WEEK = 10  # streamers: free agents that would start for me this week
PRICED_CANDIDATES = 12  # free agents whose value is priced across the whole bid range
UNTAKEN = 0.5  # a free agent the room takes in fewer of its seasons refills a dropped spot


def title_objective(inputs: RaceInputs, records: list[dict]) -> tuple[RaceInputs, dict | None]:
    """`inputs` with my bidding refilling drops from the untaken wire and weighting weeks
    by whichever objective replays the standing roster to better title odds, plus a
    summary of that choice.

    The refill pool is today's free agents the room takes (claim or free pickup) in
    fewer than UNTAKEN of its seasons by the next weekly auction, counting any off-cycle
    auction before it; they are assumed to stay available. Title weights come from the
    points objective's replay, normalized to average 1 so gains keep the guide ceiling's
    points-a-week scale. They are a first-order fit computed once, so a policy on them
    can give up points in weeks that only look safe; the replay decides.
    """
    if not inputs.alive[inputs.me]:
        return inputs, None
    w = inputs.week0
    pool = []
    weekly = w + 1 if inputs.waivers_ran else w
    auction = next((v for v in range(weekly, WEEKS) if records[0]["auctions"][v] is not None), None)
    if auction is not None:
        taken = Counter(j for rec in records for v in range(w, auction + 1) if rec["auctions"][v] is not None
                        for j, outcome in zip(*rec["auctions"][v]) if outcome != -1)
        pool = [j for j in inputs.free_agents if taken[j] < UNTAKEN * len(records)]
    roster, budget = tuple(inputs.rosters[inputs.me]), inputs.budgets[inputs.me]

    def best_run(weights):
        variant = dataclasses.replace(inputs, my_bidding=inputs.my_bidding.objective(weights, pool))
        runs = run_replays(variant, records, [(roster, budget, pol) for pol in POLICIES])
        return variant, max(runs, key=lambda run: run["p_title"])

    points, points_run = best_run(inputs.my_bidding.weights)
    scale = len(points_run["leverage"]) / sum(points_run["leverage"])
    weights = [0.0] * w + [x * scale for x in points_run["leverage"]]
    titled, titled_run = best_run(weights)
    chosen = "title_weighted" if titled_run["p_title"] > points_run["p_title"] else "points"
    return (titled if chosen == "title_weighted" else points), {
        "chosen": chosen,
        "p_title": {"points": round(points_run["p_title"], 4), "title_weighted": round(titled_run["p_title"], 4)},
        "title_weights": [{"week": v + 1, "weight": round(weights[v], 3)} for v in range(w, WEEKS)],
        "refill_pool": len(pool),
    }


def my_lineup(state: SeasonState, inputs: RaceInputs) -> dict:
    """This week's optimal lineup versus the one set on Sleeper."""
    w = inputs.week0
    me = state.my_team
    points = inputs.weekly[w]
    total, slots = lineup(me.roster, points, inputs.positions, w)
    starting = {i for _, i in slots if i is not None}
    by_sleeper = {p.sleeper_id: p.index for p in state.players}
    current = [by_sleeper.get(sid) for sid in me.starters if sid]
    current_total = sum(points[i] for i in current if i is not None)

    def row(i: int | None) -> dict | None:
        if i is None:
            return None
        p = state.players[i]
        return {
            "player_id": p.sleeper_id,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "points": points[i],
            "ros_per_week": inputs.ros[w][i],
            "injury_status": p.injury_status,
            "source": p.source,
        }

    return {
        "total": round(total, 1),
        "slots": [{"slot": slot, **(row(i) or {"name": None})} for slot, i in slots],
        "bench": [row(i) for i in me.roster if i not in starting],
        "unvalued": me.unknown,
        "current_total": round(current_total, 1),
        "current_matches": set(i for i in current if i is not None) == starting,
        "start": [row(i)["name"] for i in starting if i not in current],
        "sit": [row(i)["name"] for i in current if i is not None and i not in starting],
    }


def _win_probability(outcomes: list[int], bid: int) -> float:
    """Share of recorded auctions this bid wins: a paid claim beats every lower claim
    and any free pickup; a $0 claim only lands a player nobody wanted."""
    if not outcomes:
        return 1.0
    if bid == 0:
        return sum(1 for o in outcomes if o == -1) / len(outcomes)
    return sum(1 for o in outcomes if o < bid) / len(outcomes)


def _interpolate(grid: list[tuple[int, float]], budget: int) -> float:
    """Piecewise-linear value at `budget` from (budget, value) points, budget-sorted."""
    for (b0, v0), (b1, v1) in zip(grid, grid[1:]):
        if b0 <= budget <= b1:
            return v0 if b1 == b0 else v0 + (v1 - v0) * (budget - b0) / (b1 - b0)
    return grid[0][1] if budget < grid[0][0] else grid[-1][1]


def best_replays(inputs, records, variants):
    """Choose a continuation plan by its mean outcome, never by a future record."""
    runs = run_replays(inputs, records, [(roster, budget, policy)
                                        for roster, budget in variants for policy in POLICIES])
    return [max(runs[k:k + len(POLICIES)], key=lambda run: run["p_title"])
            for k in range(0, len(runs), len(POLICIES))]


def claims(
    state: SeasonState, inputs: RaceInputs, records: list[dict], race_title: float
) -> dict:
    """Optimal bids on this week's free agents, plus the value of budget itself."""
    w = inputs.week0
    me = state.my_team
    budget = me.faab_left
    # A reserve body that lost eligibility needs a regular spot before any claim, as the
    # race's fit_roster assumes; otherwise no one-for-one swap fits the roster.
    fitted = list(me.roster)
    fit_roster(inputs.my_bidding, fitted, w)
    roster = tuple(fitted)
    positions = inputs.positions
    ros_w = inputs.ros[w]
    points = inputs.weekly[w]
    base_total, _ = lineup(list(roster), points, positions, w)

    # Candidates: the market's list plus anyone who would start for me this week.
    by_ros = heapq.nlargest(CLAIM_CANDIDATES, inputs.free_agents, key=lambda i: (ros_w[i], -i))
    by_week = heapq.nlargest(
        CANDIDATES_BY_WEEK, inputs.free_agents, key=lambda i: (points[i], -i)
    )
    candidates = list(dict.fromkeys(by_ros + by_week))
    risk = cut_risk(inputs, roster, w, statistics.fmean(r["forecast_bars"][w] for r in records))
    choices = {j: inputs.my_bidding.swaps(roster, j, budget, w, risk, DROP_CHOICES) for j in candidates}
    candidates = [j for j in candidates if choices[j]]
    auction = records[0]["auctions"][w] if records else None
    outcomes: dict[int, list[int]] = {j: [] for j in candidates}
    if auction is not None:
        for rec in records:
            cands, outs = rec["auctions"][w]
            seen = dict(zip(cands, outs))
            for j in candidates:
                # Absent from a record's list means nobody there wanted him.
                outcomes[j].append(seen.get(j, -1))
    pending = auction is not None

    # Compare full-season spending and saving outcomes for every cash/roster state.
    budgets = sorted({max(0, budget - step) for step in BUDGET_STEPS} | {0, budget}) if pending else [budget]
    baseline_runs = run_replays(
        inputs, records, [(roster, b, pol) for pol in POLICIES for b in budgets]
    )
    baseline = {
        pol: list(zip(budgets, baseline_runs[k * len(budgets) : (k + 1) * len(budgets)]))
        for k, pol in enumerate(POLICIES)
    }
    policy = max(POLICIES, key=lambda pol: baseline[pol][-1][1]["p_title"])
    base = baseline[policy][-1][1]
    v0 = base["p_title"]

    def with_offer(offer) -> tuple[int, ...]:
        mine = list(roster)
        apply_offer(mine, offer)
        return tuple(mine)

    # Every candidate with each of his drops at the full budget (his value as a free
    # pickup) keeps the drop that replays best; then the price grid only for the ones
    # worth paying for: a player who does not help for free does not help for money.
    swaps = [offer for j in candidates for offer in choices[j]]
    free_value, offers = {}, {}
    for offer, run in zip(swaps, best_replays(inputs, records, [(with_offer(o), budget) for o in swaps])):
        if offer.player not in free_value or run["p_title"] > free_value[offer.player]["p_title"]:
            free_value[offer.player], offers[offer.player] = run, offer
    variant_rosters = {j: with_offer(offers[j]) for j in candidates}
    drops = {j: offers[j].drop for j in candidates}
    priced = [j for j in candidates if not inputs.waivers_ran or j in inputs.on_waivers]
    paid = [j for j in sorted(priced, key=lambda j: -free_value[j]["p_title"]) if free_value[j]["p_title"] > v0][:PRICED_CANDIDATES]
    priced_budgets = {j: [b for b in budgets if b < budget] for j in paid}
    tasks = [(j, b) for j in paid for b in priced_budgets[j]]
    paid_runs = dict(zip(tasks, best_replays(
        inputs, records, [(variant_rosters[j], b) for j, b in tasks]
    )))
    per_candidate: dict[int, list[tuple[int, float]]] = {}
    record_grids = {}
    for j in candidates:
        grid = [(budget, free_value[j]["p_title"])]
        if j in paid:
            grid += [(b, paid_runs[j, b]["p_title"]) for b in priced_budgets[j]]
        per_candidate[j] = sorted(grid)
        runs = [(budget, free_value[j]["title_by_record"])]
        if j in paid:
            runs += [(b, paid_runs[j, b]["title_by_record"]) for b in priced_budgets[j]]
        record_grids[j] = sorted(runs)

    def paired_se(runs: dict) -> float:
        """Standard error of the relative title change against standing pat."""
        diffs = [a - b for a, b in zip(runs["title_by_record"], base["title_by_record"])]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        return (var / n) ** 0.5 / v0 * 100 if v0 > 0 else 0.0

    def player(i: int) -> dict:
        p = state.players[i]
        return {
            "player_id": p.sleeper_id,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "injury_status": p.injury_status,
            "source": p.source,
            "points_this_week": points[i],
            "ros_per_week": ros_w[i],
        }

    def reserve_moves(variant: tuple[int, ...]) -> list[str]:
        """Bodies to move onto reserve, beyond those already there, for the variant to fit."""
        held = sum(1 for i in variant if i in me.reserve)
        needed = len(variant) - WEEK_ROSTER_SIZE[w] - held
        movable = [i for i in variant if inputs.bidding.ir_until[i] > w and i not in me.reserve]
        return [state.players[i].name for i in movable[:max(0, needed)]]

    rows = []
    for j in candidates:
        grid = per_candidate[j]
        value_at = lambda b: _interpolate(grid, b)  # noqa: E731
        outs = outcomes[j]
        # Title odds are flat between clearing prices, so the best bid is one above one of them.
        bids = sorted({0, 1, *(o + 1 for o in outs if 0 <= o < budget)}) if pending and j in paid else [0]
        curve = []
        best = None
        for b in bids:
            p_win = _win_probability(outs, b) if pending else 1.0
            if pending:
                values = _record_values(record_grids[j], budget - b)
                # Price and future opportunity come from the same simulated season.
                ev = statistics.fmean(v if (o == -1 if b == 0 else o < b) else base_v
                                      for o, v, base_v in zip(outs, values, base["title_by_record"]))
            else:
                ev = value_at(budget)
            curve.append((b, p_win, ev))
            if best is None or ev > best[0] + 1e-12:
                best = (ev, b, p_win)
        break_even = max((b for b in range(budget + 1) if value_at(budget - b) >= v0), default=0) if j in paid else 0
        claimed = [o for o in outs if o >= 0]
        this_week_total, _ = lineup(list(variant_rosters[j]), points, positions, w)
        drop = drops[j]
        rows.append(
            {
                **player(j),
                "waiver_clears": inputs.on_waivers.get(j),
                "drop": player(drop) if drop is not None else None,
                "to_reserve": reserve_moves(variant_rosters[j]),
                "gain_this_week": round(this_week_total - base_total, 1),
                "title_if_free": _relative(value_at(budget), v0),
                "title_if_free_se": round(paired_se(free_value[j]), 1),
                "optimal_bid": best[1],
                "planning_gain": round(offers[j].gain, 2),
                "p_win_at_optimal": round(best[2], 3),
                "title_at_optimal": _relative(best[0], v0),
                "break_even_bid": break_even,
                "clearing": {
                    "p_claimed": round(len(claimed) / len(outs), 3) if outs else 0.0,
                    "p_free_pickup": round(sum(1 for o in outs if o <= -2) / len(outs), 3) if outs else 0.0,
                    "mean": round(statistics.fmean(claimed), 0) if claimed else None,
                    "p50": statistics.median(claimed) if claimed else None,
                    "p90": _quantile(claimed, 0.9) if claimed else None,
                },
                "curve": [
                    {"bid": b, "p_win": round(p, 3), "title": _relative(ev, v0)}
                    for b, p, ev in _thin(curve)
                ],
            }
        )
    rows.sort(key=lambda r: (-r["title_at_optimal"], -r["title_if_free"], -r["ros_per_week"], r["name"]))

    return {
        "pending": pending,
        "policy": policy,
        "budget": budget,
        "race_title": race_title,
        "baseline": {
            pol: [
                {
                    "budget": b,
                    "p_title": round(res["p_title"], 4),
                    "p_reach_final": round(res["p_reach_final"], 4),
                    "relative": _relative(res["p_title"], v0),
                }
                for b, res in baseline[pol]
            ]
            for pol in POLICIES
        },
        "base": {
            "p_title": round(v0, 4),
            "p_reach_final": round(base["p_reach_final"], 4),
            "p_cut_now": round(base["p_cut_now"], 4),
            "p_alive_by_week": [round(x, 4) for x in base["p_alive_by_week"]],
            "budget_by_week": [round(x) if x is not None else None for x in base["budget_by_week"]],
            "budget_after_claims": [round(x) if x is not None else None for x in base["budget_after_claims"]],
        },
        "roster_size": WEEK_ROSTER_SIZE[w] + inputs.bidding.reserved(roster, w),
        "forced_cuts": [player(i) for i in me.roster if i not in roster],
        "activate": [state.players[i].name for i in me.reserve if inputs.bidding.ir_until[i] <= w],
        "candidates": rows,
    }


def _relative(value: float, base: float) -> float:
    """Title odds relative to standing pat, as a percentage change."""
    return round((value / base - 1.0) * 100, 1) if base > 0 else 0.0


def _record_values(grid, budget):
    for (b0, v0), (b1, v1) in zip(grid, grid[1:]):
        if b0 <= budget <= b1:
            weight = (budget - b0) / (b1 - b0)
            return [a + weight * (b - a) for a, b in zip(v0, v1)]
    return grid[0][1] if budget < grid[0][0] else grid[-1][1]


def _quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _thin(curve: list[tuple[int, float, float]], keep: int = 24) -> list[tuple[int, float, float]]:
    if len(curve) <= keep:
        return curve
    step = len(curve) / keep
    picked = [curve[int(i * step)] for i in range(keep)]
    return picked + [curve[-1]] if picked[-1] is not curve[-1] else picked
