"""This week's decisions for my roster: the lineup and the waiver claims.

Lineup: the greedy optimum on this week's projections (season.lineup), compared with
the starters Sleeper currently has set for me.

Claims: choose the best modeled bid within the guide-based, roster-specific ceiling.
Roster variants use all remaining weeks for drops and replay the same opponent
seasons at several budgets under spending and saving plans. A record contributes the
acquired roster's title value if the bid wins, and standing pat otherwise. This keeps
the dependence between prices and future opportunities. Odds are relative to standing pat; replay
levels are approximate because opponents retain players acquired by our variant.
"""

from __future__ import annotations

import heapq
import statistics

from .league import CLAIM_CANDIDATES, WEEK_ROSTER_SIZE
from .race import POLICIES, RaceInputs, apply_offer, cut_risk, run_replays
from .season import SeasonState, lineup

BUDGET_STEPS = (0, 25, 50, 100, 200, 400, 700)
CANDIDATES_BY_WEEK = 10  # streamers: free agents that would start for me this week
PRICED_CANDIDATES = 12  # free agents whose value is priced across the whole bid range


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
    roster = tuple(me.roster)
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
    extra = inputs.capacity_extra[state.me]
    risk = cut_risk(inputs, roster, w, statistics.fmean(r["forecast_bars"][w] for r in records))
    offers = {o.player: o for o in inputs.bidding.offers(roster, candidates, budget, w, extra, risk)}
    candidates = [j for j in candidates if j in offers]
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

    variant_rosters: dict[int, tuple[int, ...]] = {}
    drops: dict[int, int | None] = {}
    for j in candidates:
        mine = list(roster)
        drops[j] = offers[j].drop
        apply_offer(mine, offers[j])
        variant_rosters[j] = tuple(mine)
    # Every candidate at the full budget (his value as a free pickup), then the price
    # grid only for the ones worth paying for: a player who does not help for free
    # does not help for money.
    free_runs = best_replays(inputs, records, [(variant_rosters[j], budget) for j in candidates])
    free_value = dict(zip(candidates, free_runs))
    paid = [j for j in sorted(candidates, key=lambda j: -free_value[j]["p_title"]) if free_value[j]["p_title"] > v0][:PRICED_CANDIDATES]
    # Keep the original interpolation knots bracketing legal bids; prices above the
    # ceiling cannot affect any recommendation or reported break-even value.
    priced_budgets = {}
    for j in paid:
        lower = max(b for b in budgets if b <= budget - int(offers[j].ceiling)) if pending else budget
        priced_budgets[j] = [b for b in budgets if lower <= b < budget]
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

    rows = []
    for j in candidates:
        grid = per_candidate[j]
        value_at = lambda b: _interpolate(grid, b)  # noqa: E731
        outs = outcomes[j]
        ceiling = int(offers[j].ceiling)
        bids = sorted({0, 1, ceiling, *(o + 1 for o in outs if 0 <= o < ceiling)} & set(range(ceiling + 1))) if pending and j in paid else [0]
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
        break_even = max((b for b in range(ceiling + 1) if value_at(budget - b) >= v0), default=0) if j in paid else 0
        claimed = [o for o in outs if o >= 0]
        this_week_total, _ = lineup(list(variant_rosters[j]), points, positions, w)
        drop = drops[j]
        rows.append(
            {
                **player(j),
                "drop": player(drop) if drop is not None else None,
                "gain_this_week": round(this_week_total - base_total, 1),
                "title_if_free": _relative(value_at(budget), v0),
                "title_if_free_se": round(paired_se(free_value[j]), 1),
                "optimal_bid": best[1],
                "bid_ceiling": ceiling,
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
        "roster_size": WEEK_ROSTER_SIZE[w] + extra,
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
