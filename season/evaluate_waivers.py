#!/usr/bin/env python3
"""Validate bid sizes with held-out players and compare saving plans on new seeds.

    uv run -m season.evaluate_waivers            # -> bidding_evaluation.json
    uv run -m season.evaluate_waivers --sims 256

The plan comparison uses paired opponent seasons, including a sensitivity run where
every opponent participates immediately. Absolute replay odds are approximate because
opponents retain players our replay buys. A few observed auctions cannot validate future
activity; held-out bid-size errors (per player, and the latest auction from earlier ones)
are conditional on a manager submitting a positive bid, using current projections for
historical values. The pre-auction counterfactual reconstructs this week's rosters
before the run and prices claims with the learned model; it uses observed bids, so it is
a diagnostic rather than an ex-ante backtest.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys

from shared.noise import SEED
from shared.paths import BIDDING_EVALUATION, LEAGUE, POOL, WEEKLY_PROJECTIONS

from .claims import claims, title_objective
from .race import POLICIES, RACE_SIMS, race_inputs, run_race, run_replays
from .state import load_season


def counterfactual(state, sims):
    """Reconstruct today's pre-auction rosters; learned bids make this a diagnostic."""
    indexes = {p.sleeper_id: p.index for p in state.players}
    rosters = {t.roster_id: set(t.roster) for t in state.teams}
    budgets = {t.roster_id: t.faab_left for t in state.teams}
    completed = [tx for tx in state.transactions if tx["week"] == state.week and tx["status"] == "complete"]
    for tx in sorted(completed, key=lambda tx: tx["processed_at"], reverse=True):
        team = tx["roster_id"]
        rosters[team].difference_update(indexes[sid] for sid in tx["adds"] if sid in indexes)
        rosters[team].update(indexes[sid] for sid in tx["drops"] if sid in indexes)
        budgets[team] += tx["bid"] or 0
    teams = [dataclasses.replace(t, roster=sorted(rosters[t.roster_id]), faab_left=budgets[t.roster_id]) for t in state.teams]
    owned = set().union(*rosters.values())
    before = dataclasses.replace(state, teams=teams, waivers_ran=False,
                                 free_agents=[p.index for p in state.players if p.index not in owned],
                                 transactions=[t for t in state.transactions if t["week"] < state.week])
    learned = race_inputs(state)
    inputs = dataclasses.replace(race_inputs(before), managers=learned.managers, price_curve=learned.price_curve)
    print(f"simulating pre-auction reconstruction: {sims} seasons", file=sys.stderr, flush=True)
    records = run_race(inputs, sims, SEED + 200_000, exclude_me=True)
    decisions = claims(before, title_objective(inputs, records)[0], records)
    assert decisions["pending"]
    assert all(0 <= c["optimal_bid"] <= decisions["budget"] for c in decisions["candidates"])
    return {"note": "Current projections and learned opponent bids; a counterfactual, not an ex-ante backtest.",
            "candidates": [{k: c[k] for k in ("name", "optimal_bid", "break_even_bid", "p_win_at_optimal",
                                             "title_at_optimal", "gain_this_week")}
                           for c in decisions["candidates"]]}


def compare(inputs, records, roster, budget):
    runs = run_replays(inputs, records, [(tuple(roster), budget, p) for p in POLICIES])
    output = {}
    for policy, run in zip(POLICIES, runs):
        path = run["budget_by_week"]
        output[policy] = {"p_title": run["p_title"], "p_reach_final": run["p_reach_final"],
                          "budget_by_week": [{"week": inputs.week0 + k + 1, "budget": round(b) if b is not None else None}
                                             for k, b in enumerate(path)],
                          "mean_week9_spend": round(path[8 - inputs.week0] - run["budget_after_claims"][8 - inputs.week0])
                          if inputs.week0 <= 8 and path[8 - inputs.week0] is not None else None}
    for policy, run in zip(POLICIES[1:], runs[1:]):
        differences = [a - b for a, b in zip(runs[0]["title_by_record"], run["title_by_record"])]
        delta = statistics.fmean(differences)
        se = statistics.stdev(differences) / math.sqrt(len(differences))
        output[policy]["value_advantage_percentage_points"] = round(delta * 100, 3)
        output[policy]["paired_95_percent_interval"] = [round((delta + sign * 1.96 * se) * 100, 3) for sign in (-1, 1)]
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=RACE_SIMS, help=f"simulated seasons per scenario (default {RACE_SIMS})")
    args = ap.parse_args(argv)
    state = load_season(POOL, WEEKLY_PROJECTIONS, LEAGUE)
    inputs = race_inputs(state)
    result = {"fetched_at": state.fetched_at, "simulations_per_scenario": args.sims,
              "seed": SEED + 100_000, "bid_calibration": inputs.market_fit, "scenarios": {}}
    for name in ("fitted_activity", "all_opponents_active"):
        if name == "all_opponents_active":
            inputs = dataclasses.replace(inputs, managers=[dataclasses.replace(m, activity=1.0) for m in inputs.managers])
        print(f"simulating {name}: {args.sims} seasons", file=sys.stderr, flush=True)
        records = run_race(inputs, args.sims, SEED + 100_000, exclude_me=True)
        result["scenarios"][name] = compare(title_objective(inputs, records)[0], records,
                                            state.my_team.roster, state.my_team.faab_left)
    result["limitations"] = ["Few observed auctions; the reactivation rate is not validated.",
                              "Bid-size validation holds out all bids on a player, and the latest auction; "
                              "current projections proxy historical values.",
                              "Replay odds are paired approximations, not calibrated championship probabilities."]
    result["pre_auction_counterfactual"] = counterfactual(state, args.sims)
    BIDDING_EVALUATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {BIDDING_EVALUATION.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
