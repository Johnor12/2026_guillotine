"""This week's decisions for my roster: the lineup and the waiver claims.

Lineup: the greedy optimum on this week's projections (state.lineup), compared with the
starters Sleeper currently has set for me.

Objective: my roster is valued as it will be managed: from next week on its lineup may
start the wire's best body at each position, what the room leaves untaken in the
recorded seasons (wire_stream), and my bidding values each week's points alike or by
d log P(title) / d(points), whichever replays to better title odds (title_objective).
Screening, drops and my future bidding use it; this week's bids are chosen by replayed
title odds.

Claims: every candidate's drop is chosen by replay among the few that waivers.Bidding
ranks best (the heuristic prices a fixed roster, where a future lineup expansion is an
empty seat, so its favorite can give up a starter today for depth the wire would supply
anyway), and his bid alone is whatever maximizes replayed title odds anywhere in the
budget. The standing roster is replayed at several budgets under every future bid price
(race.PRICES, per point of a claim's season gain); that grid is the value of cash, and
the price that replays best at each budget is what every variant at that budget is
priced under, so saving or spending is whatever the seasons reward. After this week's run, only players
dropped since are still on waivers and take a bid; everyone else is a free add. A
variant that only fits with a body moved onto reserve names that move. A record
contributes the acquired roster's title value if the bid wins, and standing pat
otherwise, which keeps the dependence between prices and future opportunities. Odds
are relative to standing pat. A player the variant holds comes off the recorded
roster of the opponent who bought him (race.replay); what that opponent would have
done with the cash instead is not replayed.

Card: the recommendation is the whole set of claims to enter (card.py), built from
those single valuations: each claim option (a candidate and one of his drops, or a free
add) may join at any bid, claims naming the same drop are alternatives and a player may
be claimed again with another drop as a fallback, and the card is scored per recorded
season by walking it the way Sleeper processes claims. Only my side bids this way; the
room and my replayed future self keep race.py's one offer per candidate.
"""

from __future__ import annotations

import dataclasses
import heapq
import statistics

import numpy as np

from shared.league import POSITIONS, WEEK_ROSTER_SIZE, WEEKS

from .card import Claim, Scorer, Values, build, paired
from .race import CLAIM_CANDIDATES, PRICES, Market, RaceInputs, apply_offer, cut_risk, fit_roster, run_replays
from .state import SeasonState, lineup

BUDGET_STEPS = (0, 25, 50, 100, 200, 400, 700)
DROP_CHOICES = 3  # drops per candidate compared by replay, best heuristic gain first
CANDIDATES_BY_WEEK = 10  # streamers: free agents that would start for me this week
PRICED_CANDIDATES = 12  # free agents whose value is priced across the whole bid range


def wire_stream(inputs: RaceInputs, records: list[dict]) -> np.ndarray:
    """Expected points, by position and week, of the best body the room leaves on the
    wire that week: listed at that week's auctions (in play, race.CLAIM_CANDIDATES by
    rest-of-season points) and untaken, which is what my replayed agent can pick up and
    what my bidding lets its lineups start from next week on (waivers.Bidding.objective).
    The mean over the recorded seasons of each season's best, so a wire that always
    holds some starter counts even when it is a different one every season. A free
    agent the auctions never list is not counted, since the replay never offers him."""
    points = np.asarray(inputs.weekly)
    pos = np.asarray(inputs.positions)
    stream = np.zeros((4, WEEKS))
    for rec in records:
        for w in range(inputs.week0, WEEKS):
            available = np.zeros(len(pos), dtype=bool)
            for candidates, winning in rec["auctions"][w]:
                listed, outcome = np.asarray(candidates, dtype=np.int64), np.asarray(winning)
                available[listed[outcome == -1]] = True
            for p in range(4):
                mask = available & (pos == p)
                if mask.any():
                    stream[p, w] += points[w][mask].max()
    return stream / len(records)


def title_objective(inputs: RaceInputs, records: list[dict]) -> tuple[RaceInputs, dict | None]:
    """`inputs` with the recorded market my future bids are shaded against, and my
    bidding starting the recorded wire's best bodies from next week on and weighting
    weeks by whichever objective replays the standing roster to better title odds, plus
    a summary of that choice.

    Title weights come from the points objective's replay at its best price, normalized
    to average 1 so gains keep their points-a-week scale. They are a first-order fit
    computed once, so a policy on them can give up points in weeks that only look safe;
    the replay decides.
    """
    inputs = dataclasses.replace(inputs, market=Market(records))
    if not inputs.alive[inputs.me]:
        return inputs, None
    w = inputs.week0
    stream = wire_stream(inputs, records)
    roster, budget = tuple(inputs.rosters[inputs.me]), inputs.budgets[inputs.me]
    points = dataclasses.replace(inputs, my_bidding=inputs.my_bidding.objective(inputs.my_bidding.weights, stream))
    runs = run_replays(points, records, [(roster, budget, s) for s in PRICES])
    price, points_run = max(zip(PRICES, runs), key=lambda item: item[1]["p_title"])
    scale = len(points_run["leverage"]) / sum(points_run["leverage"])
    weights = [0.0] * w + [x * scale for x in points_run["leverage"]]
    titled = dataclasses.replace(inputs, my_bidding=points.my_bidding.objective(weights, stream))
    titled_run = run_replays(titled, records, [(roster, budget, price)])[0]
    chosen = "title_weighted" if titled_run["p_title"] > points_run["p_title"] else "points"
    return (titled if chosen == "title_weighted" else points), {
        "chosen": chosen,
        "p_title": {"points": round(points_run["p_title"], 4), "title_weighted": round(titled_run["p_title"], 4)},
        "title_weights": [{"week": v + 1, "weight": round(weights[v], 3)} for v in range(w, WEEKS)],
        "wire": [{"week": v + 1, **{p: round(float(stream[k, v]), 1) for k, p in enumerate(POSITIONS)}}
                 for v in range(w, WEEKS)],
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


def claims(state: SeasonState, inputs: RaceInputs, records: list[dict]) -> dict:
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
    by_week = heapq.nlargest(CANDIDATES_BY_WEEK, inputs.free_agents, key=lambda i: (points[i], -i))
    candidates = list(dict.fromkeys(by_ros + by_week))
    risk = cut_risk(inputs, roster, w, statistics.fmean(r["forecast_bars"][w] for r in records))
    choices = {j: inputs.my_bidding.swaps(roster, j, budget, w, risk, DROP_CHOICES) for j in candidates}
    candidates = [j for j in candidates if choices[j]]
    # This week's run (or the off-cycle auction on the players still on waivers) is the
    # round my claims enter; the cascade rounds after it are the room's alone.
    pending = bool(records) and bool(records[0]["auctions"][w])
    outcomes: dict[int, list[int]] = {j: [] for j in candidates}
    if pending:
        for rec in records:
            seen = dict(zip(*rec["auctions"][w][0]))
            for j in candidates:
                outcomes[j].append(seen.get(j, -1))  # absent from a record's list: nobody there wanted him

    # The standing roster at every budget under every future bid price: the value of
    # cash, and the price every variant at that budget is priced under.
    budgets = sorted({max(0, budget - step) for step in BUDGET_STEPS} | {0, budget}) if pending else [budget]
    baseline_runs = run_replays(inputs, records, [(roster, b, s) for s in PRICES for b in budgets])
    baseline = {s: list(zip(budgets, baseline_runs[k * len(budgets):(k + 1) * len(budgets)]))
                for k, s in enumerate(PRICES)}
    price_at = {b: max(PRICES, key=lambda s: baseline[s][k][1]["p_title"]) for k, b in enumerate(budgets)}
    price = price_at[budget]
    base = baseline[price][-1][1]
    v0 = base["p_title"]

    def with_offer(offer) -> tuple[int, ...]:
        mine = list(roster)
        apply_offer(mine, offer)
        return tuple(sorted(mine))  # the card's Values keys rosters sorted

    # Every candidate with each of his drops at the full budget (his value as a free
    # pickup) keeps the drop that replays best; then the price grid only for the ones
    # worth paying for: a player who does not help for free does not help for money.
    swaps = [offer for j in candidates for offer in choices[j]]
    single_runs = {(o.player, o.drop): {} for o in swaps}
    free_value, offers = {}, {}
    for offer, run in zip(swaps, run_replays(inputs, records, [(with_offer(o), budget, price) for o in swaps])):
        single_runs[offer.player, offer.drop][budget] = run
        if offer.player not in free_value or run["p_title"] > free_value[offer.player]["p_title"]:
            free_value[offer.player], offers[offer.player] = run, offer
    variant_rosters = {j: with_offer(offers[j]) for j in candidates}
    priced = [j for j in candidates if not inputs.waivers_ran or j in inputs.on_waivers]
    paid = [j for j in sorted(priced, key=lambda j: -free_value[j]["p_title"]) if free_value[j]["p_title"] > v0][:PRICED_CANDIDATES]
    tasks = [(j, b) for j in paid for b in budgets if b < budget]
    for (j, b), run in zip(tasks, run_replays(inputs, records, [(variant_rosters[j], b, price_at[b]) for j, b in tasks])):
        single_runs[offers[j].player, offers[j].drop][b] = run
    grids: dict[int, list[tuple[int, float]]] = {}
    record_grids = {}
    for j in candidates:
        runs = sorted(single_runs[offers[j].player, offers[j].drop].items())
        grids[j] = [(b, run["p_title"]) for b, run in runs]
        record_grids[j] = [(b, run["title_by_record"]) for b, run in runs]

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

    def reserve_moves(variant: tuple[int, ...], added: int) -> list[str]:
        """Bodies to move onto reserve before the add, beyond those already there, so the
        added body has a regular spot (Sleeper never adds straight onto reserve)."""
        eligible = [i for i in variant if i != added and inputs.bidding.ir_until[i] > w]
        needed = len(variant) - WEEK_ROSTER_SIZE[w] - sum(1 for i in eligible if i in me.reserve)
        movable = [i for i in eligible if i not in me.reserve]
        if needed > len(movable):
            raise ValueError(f"{state.players[added].name} has no regular spot on {variant}")
        return [state.players[i].name for i in movable[:max(0, needed)]]

    rows = []
    for j in candidates:
        grid = grids[j]
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
                ev = _interpolate(grid, budget)
            curve.append((b, p_win, ev))
            if best is None or ev > best[0] + 1e-12:
                best = (ev, b, p_win)
        break_even = max((b for b in range(budget + 1) if _interpolate(grid, budget - b) >= v0), default=0) if j in paid else 0
        claimed = [o for o in outs if o >= 0]
        this_week_total, _ = lineup(list(variant_rosters[j]), points, positions, w)
        drop = offers[j].drop
        rows.append(
            {
                **player(j),
                "waiver_clears": inputs.on_waivers.get(j),
                "drop": player(drop) if drop is not None else None,
                "to_reserve": reserve_moves(variant_rosters[j], j),
                "gain_this_week": round(this_week_total - base_total, 1),
                "title_if_free": _relative(_interpolate(grid, budget), v0),
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

    # The card: every swap that beats standing pat for free is a claim option (a free
    # add after the run, or a bid on a player still on waivers); the room's clearing
    # prices and the replayed grids score any card the way Sleeper processes it.
    table = Values(roster, budgets, np.array([baseline[price_at[b]][k][1]["title_by_record"] for k, b in enumerate(budgets)]))
    base_full = table.grids[roster][-1]
    for j in candidates:
        best = (offers[j].player, offers[j].drop)
        for offer in sorted(choices[j], key=lambda o: (o.player, o.drop) != best):
            swap = (offer.player, offer.drop)
            runs = single_runs[swap]
            if len(runs) == len(budgets):
                grid = np.array([runs[b]["title_by_record"] for b in budgets])
                table.grids[with_offer(offer)] = grid
                table.single(swap, grid - table.grids[roster])
            else:
                # Replayed at the full budget only: his player's replayed drop lends the budget profile.
                full = np.array(runs[budget]["title_by_record"])
                table.learn(with_offer(offer), budget, full)
                effect = np.tile(full - base_full, (len(budgets), 1))
                if swap != best:
                    effect += table.singles[best] - table.singles[best][-1]
                table.single(swap, effect)
    options = [Claim(o.player, o.drop, 0, free=o.player not in priced) for j in paid + [j for j in candidates if j not in priced]
               for o in choices[j] if single_runs[o.player, o.drop][budget]["p_title"] > v0]
    if not pending:
        options = [dataclasses.replace(o, free=True) for o in options]
    eligible = {i for i in range(len(positions)) if inputs.bidding.ir_until[i] > w}
    scorer = Scorer(table, {j: np.array(outs) for j, outs in outcomes.items() if outs}, len(records), budget,
                    WEEK_ROSTER_SIZE[w], eligible)

    replayed = []

    def replay_rosters(variants: list[tuple[tuple[int, ...], int]]) -> dict[tuple[tuple[int, ...], int], np.ndarray]:
        """Each (roster, cash) pair's per-season title values, priced as the nearest grid budget is."""
        tasks = [(r, b, price_at[min(budgets, key=lambda g: abs(g - b))]) for r, b in variants]
        replayed.extend(variants)
        return {v: np.array(run["title_by_record"]) for v, run in zip(variants, run_replays(inputs, records, tasks))}

    card = build(scorer, options, replay_rosters)
    card_value, card_titles, resolution = scorer.evaluate(card)
    _, card_se = paired(card_titles, base_full)
    outcomes_reached = sorted(resolution["rosters"].values(), key=lambda item: -len(item[3]))
    claims_out = []
    for k, claim in enumerate(card):
        without = scorer.evaluate(card[:k] + card[k + 1:])[0]
        offer = next(o for o in choices[claim.player] if (o.player, o.drop) == claim.swap)
        claims_out.append({
            "order": k + 1,
            "bid": None if claim.free else claim.bid,
            "player": player(claim.player),
            "waiver_clears": inputs.on_waivers.get(claim.player),
            "drop": player(claim.drop) if claim.drop is not None else None,
            "to_reserve": reserve_moves(with_offer(offer), claim.player),
            "p_reached": round(float(resolution["reached"][k].mean()), 3),
            "p_win": round(float(resolution["won"][k].mean()), 3),
            "title_without": _relative(without, v0),
        })

    return {
        "pending": pending,
        "card": {
            "claims": claims_out,
            "p_title": round(card_value, 4),
            "relative": _relative(card_value, v0),
            "relative_se": round(card_se / v0 * 100, 1) if v0 > 0 else 0.0,
            "expected_spend": round(float(budget - resolution["left"].mean())),
            "p_any_win": round(float(resolution["won"].any(0).mean()), 3),
            "replays": len(replayed),
            "outcomes": [
                {
                    "adds": [state.players[j].name for j, _ in swaps_won],
                    "drops": [state.players[d].name for _, d in swaps_won if d is not None],
                    "spend": int(budget - resolution["left"][idx[0]]),
                    "p": round(len(idx) / len(records), 3),
                    "title": _relative(float(card_titles[idx].mean()), float(base_full[idx].mean())),
                }
                for _, swaps_won, _, idx in outcomes_reached[:8]
            ],
        },
        "price": price,
        "budget": budget,
        "baseline": [
            {
                "price": s,
                "budgets": [
                    {
                        "budget": b,
                        "p_title": round(res["p_title"], 4),
                        "p_reach_final": round(res["p_reach_final"], 4),
                        "relative": _relative(res["p_title"], v0),
                    }
                    for b, res in baseline[s]
                ],
            }
            for s in PRICES
        ],
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
