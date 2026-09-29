#!/usr/bin/env python3
"""In-season decisions and odds for this league, from the live Sleeper state.

    uv run -m season.run              # pool.json + weekly projections + league.json -> season.json
    uv run -m season.run --report     # + this week's lineup, claims and league odds on stderr
    uv run -m season.run --sims 256   # fewer simulated seasons, for a quick check

Prints this week's recommendations (bids, drops, reserve moves, lineup); --report adds
the model diagnostics and league odds on stderr.

The value input is per-week: DraftSharks' weekly projection blended 2:1 with Sleeper's
for the same week (season/state.py). The engine is an agent-based race (season/race.py):
every alive team fields its optimal lineup each week under the draft model's noise, the
bottom two are cut, their players hit the wire, and the survivors bid: opponents at
learned prices around a guide-based reference (season/waivers.py), my agent at a price
per point of season gain the replays choose. Run once without me it is the
market and the elimination bars I face, which price my roster variants
(season/claims.py: this week's lineup, and for each free agent the bid that maximizes
modeled title odds, with the drop that replays best among those the bidding heuristic
ranks highest); run with all 32, my team entering this week's card, it is every team's
chance of being cut this week, of reaching the final, and of the title, along with its
expected spend and budget path.

Deterministic: the race is seeded.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import statistics
import sys
import time

from shared.league import REGULAR_WEEKS, WEEK_ROSTER_SIZE, WEEKS
from shared.noise import SEED, TEAM_SEASON_SIGMA, WEEKLY_SIGMA
from shared.paths import LEAGUE, POOL, SEASON, WEEKLY_PROJECTIONS

from .card import Claim
from .claims import claims, my_lineup, title_objective
from .race import CASCADE_ROUNDS, CLAIM_CANDIDATES, CLAIMS_PER_TEAM, RACE_SIMS, race_inputs, run_race
from .state import lineup_points, load_season
from .waivers import GUIDE_URL, SAVING_PLANS


def league_odds(state, inputs, records: list[dict]) -> list[dict]:
    """Every team's fate from the full race: cut-this-week, survival, title, spend."""
    w0 = inputs.week0
    n = len(records)
    out = []
    for i, team in enumerate(state.teams):
        cuts = [r["cut_week"][i] for r in records]
        alive_by_week = [
            sum(1 for c in cuts if team.alive and (c is None or c > w)) / n
            for w in range(w0, REGULAR_WEEKS)
        ]

        def budget_entering(week: int) -> float | None:
            w = week - 1
            if w < w0 or not team.alive:
                return None
            alive = [r["budget_path"][w - w0][i] for r in records if r["cut_week"][i] is None or r["cut_week"][i] >= w]
            return round(statistics.fmean(alive)) if alive else None

        spend_now = statistics.fmean(inputs.budgets[i] - r["budget_after_claims"][0][i] for r in records)
        out.append(
            {
                "roster_id": team.roster_id,
                "name": team.name,
                "username": team.username,
                "is_mine": team.is_mine,
                "alive": team.alive,
                "faab_left": team.faab_left,
                "points_for": round(team.points_for, 1),
                "points_this_week": round(team.points_this_week, 1),
                "roster_size": len(team.roster) + len(team.unknown),
                "projected_now": round(lineup_points(team.roster, inputs.weekly[w0], inputs.positions, w0), 1) if team.alive else None,
                "p_cut_now": round(sum(1 for c in cuts if team.alive and c == w0) / n, 4),
                "p_alive_by_week": [round(x, 4) for x in alive_by_week],
                "p_reach_final": round(sum(1 for c in cuts if team.alive and c is None) / n, 4),
                "p_title": round(sum(1 for r in records if r["champion"] == i) / n, 4),
                "spend_now": round(spend_now),
                "p_claim_now": round(sum(1 for r in records if any(t == i for _, t, _ in r["claims"][w0])) / n, 3),
                "budget_entering_week9": budget_entering(9),
                "budget_entering_week13": budget_entering(13),
                "players": [
                    {"name": state.players[j].name, "position": state.players[j].position, "points": inputs.weekly[w0][j], "ros_per_week": inputs.ros[w0][j]}
                    for j in sorted(team.roster, key=lambda j: -inputs.ros[w0][j])
                ],
            }
        )
    return out


def market(state, inputs, excluded: list[dict], full: list[dict]) -> dict:
    """Bars, budgets and this week's simulated claims, plus what the room actually paid."""
    w0 = inputs.week0
    n = len(excluded)
    bars = [[r["bars"][w] for r in excluded] for w in range(w0, REGULAR_WEEKS)]
    budget_by_week = []
    for k, w in enumerate(range(w0, WEEKS)):
        alive = [
            r["budget_path"][k][i]
            for r in full
            for i in range(len(state.teams))
            if state.teams[i].alive and (r["cut_week"][i] is None or r["cut_week"][i] >= w)
        ]
        budget_by_week.append(round(statistics.fmean(alive)) if alive else None)
    names = {p.sleeper_id: p.name for p in state.players}
    team_names = {t.roster_id: t.name for t in state.teams}
    observed = [
        {
            "week": tx["week"],
            "type": tx["type"],
            "status": tx["status"],
            "team": team_names.get(tx["roster_id"]),
            "adds": [names.get(sid, sid) for sid in tx["adds"]],
            "drops": [names.get(sid, sid) for sid in tx["drops"]],
            "bid": tx["bid"],
            "note": tx["note"],
            "created": tx["created"],
        }
        for tx in reversed(state.transactions)
    ]
    # The room's claims in this week's run (or the pending off-cycle auction), as a check
    # against the observed ones: how often each player in play is taken, by a paid claim
    # or a free pickup, and what a paid claim wins him for. The cascade rounds are left
    # out, as are the winning bids' extremes: the highest of thousands of seasons' draws
    # says nothing about the price to expect.
    outcomes: dict[int, list[int]] = {}
    for r in excluded:
        if r["auctions"][w0]:
            for j, outcome in zip(*r["auctions"][w0][0]):
                outcomes.setdefault(j, []).append(outcome)
    simulated = []
    for j, outs in outcomes.items():
        paid = sorted(o for o in outs if o >= 0)
        simulated.append(
            {
                "name": state.players[j].name,
                "position": state.players[j].position,
                "p_taken": round(sum(1 for o in outs if o != -1) / n, 3),
                "p_claimed": round(len(paid) / n, 3),
                "p50": paid[len(paid) // 2] if paid else None,
                "p90": paid[min(len(paid) - 1, int(0.9 * len(paid)))] if paid else None,
            }
        )
    simulated.sort(key=lambda r: (-r["p_taken"], -r["p_claimed"], -(r["p50"] or 0), r["name"]))
    simulated = simulated[:40]
    return {
        "calibration": inputs.market_fit,
        "managers": [
            {"roster_id": t.roster_id, "name": t.name, "activity": round(m.activity, 3),
             "bid_weeks": m.bid_weeks, "bids": m.bids}
            for t, m in zip(state.teams, inputs.managers) if t.alive and not t.is_mine
        ],
        "bars_by_week": [
            {
                "week": w + 1,
                "mean": round(statistics.fmean(col), 1),
                "p10": round(sorted(col)[int(0.1 * n)], 1),
                "p90": round(sorted(col)[int(0.9 * n)], 1),
            }
            for w, col in zip(range(w0, REGULAR_WEEKS), bars)
        ],
        "champion_bar_mean": round(statistics.fmean(r["champ_bar"] for r in excluded), 1),
        "mean_budget_by_week": [{"week": w + 1, "budget": b} for w, b in zip(range(w0, WEEKS), budget_by_week)],
        "simulated_claims_now": simulated,
        "observed": observed,
    }


def report_decisions(payload: dict) -> None:
    me = payload["me"]
    claims = payload["claims"]
    print(f"\nWeek {payload['week']} — {me['name']} — FAAB remaining: ${me['faab_left']}")
    if claims["activate"]:
        print(f"\nNo longer reserve-eligible, move to the active roster: {', '.join(claims['activate'])}")
    if claims["forced_cuts"]:
        print(f"Roster over capacity, cut before any add below: {', '.join(c['name'] for c in claims['forced_cuts'])}")
    card = payload["claims"]["card"]
    if not card["claims"]:
        print("\nNo pickups improve the modeled title odds.")
    elif all(c["bid"] is None for c in card["claims"]):
        print("\nFree adds, in this order:")
    else:
        print("\nBidding card — enter every claim. Sleeper processes the highest bid first; keep equal bids in this order:")
    for c in card["claims"]:
        drop = c["drop"]["name"] if c["drop"] else "no drop needed"
        if c["to_reserve"]:
            drop += f"; to IR: {', '.join(c['to_reserve'])}"
        if c["bid"] is None:
            action, terms = "free", "add now"
        else:
            action = f"${c['bid']}"
            terms = f"wins {c['p_win']:.0%}"
            if c["p_reached"] < 0.995:
                terms += f" (still valid when reached in {c['p_reached']:.0%})"
            if c["waiver_clears"]:
                clears = dt.datetime.fromisoformat(c["waiver_clears"]).astimezone()
                terms = f"claim clears ~{clears:%a %H:%M %Z}; {terms}"
        print(
            f"  {c['order']:>2}. {action:<5} {c['player']['name']} ({c['player']['position']}) — {terms}; "
            f"drop: {drop}; card without it: {c['title_without']:+.1f}%"
        )
    if card["claims"]:
        print(f"Card: title odds {card['relative']:+.1f}% (±{card['relative_se']}) against standing pat; "
              f"expected spend ${card['expected_spend']}; something lands {card['p_any_win']:.0%} of the time.")
        outcomes = [o for o in card["outcomes"] if o["adds"]][:4]
        if outcomes:
            print("Likely outcomes: " + "; ".join(
                f"{' + '.join(o['adds'])} {o['p']:.0%} ({o['title']:+.1f}%, ${o['spend']})" for o in outcomes))

    lineup = me["lineup"]
    print(f"\nOptimal lineup — {lineup['total']:.1f} projected points (currently set: {lineup['current_total']:.1f}):")
    for slot in lineup["slots"]:
        print(f"  {slot['slot']:<5} {slot.get('name') or '(empty)':<24} {slot.get('points', 0):>5.1f}")
    for label, names in (("Start", lineup["start"]), ("Sit", lineup["sit"])):
        if names:
            print(f"  {label}: {', '.join(names)}")


def report(payload: dict) -> None:
    me = payload["me"]
    print(
        f"\nweek {payload['week']}: {payload['alive']} alive; {me['name']} FAAB ${me['faab_left']}; "
        f"P(cut this week) {me['p_cut_now']:.1%}, P(final) {me['p_reach_final']:.1%}, "
        f"P(title) {me['p_title']:.1%} (replay {payload['claims']['base']['p_title']:.1%}, "
        f"future bids at {payload['claims']['price']} weeks of cash per point per week)",
        file=sys.stderr,
    )
    lu = me["lineup"]
    print(f"lineup {lu['total']} vs bar {me['bar_now']['mean']} (p90 {me['bar_now']['p90']}):", file=sys.stderr)
    for s in lu["slots"]:
        print(f"  {s['slot']:<5} {s.get('name') or '(empty)':<24} {s.get('points', 0):>5}", file=sys.stderr)
    if lu["start"] or lu["sit"]:
        print(f"  start {lu['start']}, sit {lu['sit']}", file=sys.stderr)
    objective = payload["claims"]["objective"]
    if objective:
        print(f"objective {objective['chosen']} (replay P(title): points {objective['p_title']['points']:.1%}, "
              f"title-weighted {objective['p_title']['title_weighted']:.1%})", file=sys.stderr)
    print("Budget value by future bid price, weeks of cash per point per week of gain (relative to standing pat):",
          file=sys.stderr)
    for level in payload["claims"]["baseline"]:
        print(f"  {level['price']}: " + ", ".join(f"${r['budget']} {r['relative']:+.0f}%" for r in level["budgets"]),
              file=sys.stderr)
    mode = ("weekly run pending" if not payload["waivers_ran"] else
            "off-cycle: players dropped since the run need a claim" if payload["claims"]["pending"] else "free agents only")
    card = payload["claims"]["card"]
    print(f"card: {len(card['claims'])} claims, {card['replays']} replays beyond the single valuations", file=sys.stderr)
    print(f"claims ({mode}), each player alone:", file=sys.stderr)
    for c in payload["claims"]["candidates"][:12]:
        cl = c["clearing"]
        print(
            f"  {c['name']:<22} {c['position']} ros {c['ros_per_week']:>4} wk {c['points_this_week']:>4} "
            f"bid ${c['optimal_bid']:<4} win {c['p_win_at_optimal']:.0%} title {c['title_at_optimal']:+.1f}% "
            f"(free {c['title_if_free']:+.1f}%, even ${c['break_even_bid']}, market p50 {cl['p50']} p90 {cl['p90']} "
            f"claimed {cl['p_claimed']:.0%}) drop {c['drop']['name'] if c['drop'] else '-'}"
            + (f" IR {', '.join(c['to_reserve'])}" if c["to_reserve"] else "")
            + (f" clears {c['waiver_clears']}" if c["waiver_clears"] else ""),
            file=sys.stderr,
        )
    print("league (P cut now / P final / P title / FAAB / spend now):", file=sys.stderr)
    for t in sorted(payload["teams"], key=lambda t: -t["p_title"])[:10]:
        print(
            f"  {t['name'][:22]:<22} {t['p_cut_now']:>5.1%} {t['p_reach_final']:>5.1%} {t['p_title']:>5.1%} "
            f"${t['faab_left']:<5} ${t['spend_now']}",
            file=sys.stderr,
        )
    observed = [o for o in payload["market"]["observed"] if o["type"] == "waiver"][:8]
    fit = payload["market"]["calibration"]
    if fit["positive_bids_tested"]:
        print(f"bid-size validation: {fit['submitted_bids']} opponent bids, {fit['bid_weeks']} auction week(s); "
              f"held-out log MAE: guide {fit['prior_log_mae']}, fitted {fit['fitted_log_mae']}", file=sys.stderr)
        if fit["latest_week_holdout"]:
            t = fit["latest_week_holdout"]
            print(f"  week {t['test_week']} predicted from earlier weeks only: "
                  f"guide {t['prior_log_mae']}, fitted {t['fitted_log_mae']}", file=sys.stderr)
    if observed:
        print("observed waiver claims (latest):", file=sys.stderr)
        for o in observed:
            print(f"  wk{o['week']} {o['team']}: {o['adds']} ${o['bid']} {o['status']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="summary on stderr")
    ap.add_argument("--sims", type=int, default=RACE_SIMS, help=f"simulated seasons (default {RACE_SIMS})")
    args = ap.parse_args(argv)

    state = load_season(POOL, WEEKLY_PROJECTIONS, LEAGUE)
    inputs = race_inputs(state)
    t0 = time.perf_counter()
    excluded = run_race(inputs, args.sims, SEED, exclude_me=True)
    if args.report:
        print(f"[opponent races {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    t0 = time.perf_counter()
    inputs, objective = title_objective(inputs, excluded)
    if args.report:
        print(f"[title objective {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    t0 = time.perf_counter()
    decisions = claims(state, inputs, excluded)
    decisions["objective"] = objective
    if args.report:
        print(f"[claims {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    # The full race plays this week's card for my team; my agent bids from next week on.
    by_id = {p.sleeper_id: p.index for p in state.players}
    card = [Claim(by_id[c["player"]["player_id"]], by_id[c["drop"]["player_id"]] if c["drop"] else None,
                  c["bid"] or 0, free=c["bid"] is None) for c in decisions["card"]["claims"]]
    inputs = dataclasses.replace(inputs, price=decisions["price"], my_card=card)
    t0 = time.perf_counter()
    full = run_race(inputs, args.sims, SEED, exclude_me=False)
    teams = league_odds(state, inputs, full)
    mine = next(t for t in teams if t["is_mine"])
    decisions["race_title"] = mine["p_title"]
    if args.report:
        print(f"[full races {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    w0 = inputs.week0
    bars_now = sorted(r["bars"][w0] for r in excluded)
    n = len(bars_now)
    me = state.my_team
    outlook = []
    for w in range(w0, WEEKS):
        outlook.append(
            {
                "week": w + 1,
                "my_points": round(lineup_points(me.roster, inputs.weekly[w], inputs.positions, w), 1),
                "bar_mean": round(statistics.fmean(r["bars"][w] for r in excluded), 1) if w < REGULAR_WEEKS else None,
                "p_alive": decisions["base"]["p_alive_by_week"][w - w0] if w < REGULAR_WEEKS else None,
                "budget": decisions["base"]["budget_by_week"][w - w0],
            }
        )
    payload = {
        "league_name": state.league_name,
        "week": state.week,
        "fetched_at": state.fetched_at,
        "alive": sum(t.alive for t in state.teams),
        "waivers_ran": state.waivers_ran,
        "value_input": (
            "DraftSharks weekly projections blended 2:1 with Sleeper's weekly projections "
            "in this league's scoring; Sleeper alone for players DraftSharks does not project"
        ),
        "method": (
            "Agent-based season race: optimal weekly lineups under the draft model's noise, "
            "bottom two cut, survivors bid FAAB on the wire; see season/run.py and season/race.py."
        ),
        "me": {
            "roster_id": me.roster_id,
            "name": me.name,
            "faab_left": me.faab_left,
            "p_cut_now": mine["p_cut_now"],
            "p_reach_final": mine["p_reach_final"],
            "p_title": mine["p_title"],
            "rank_by_title": 1 + sum(1 for t in teams if t["p_title"] > mine["p_title"]),
            "bar_now": {
                "mean": round(statistics.fmean(bars_now), 1),
                "p10": round(bars_now[int(0.1 * n)], 1),
                "p50": round(bars_now[n // 2], 1),
                "p90": round(bars_now[int(0.9 * n)], 1),
            },
            "lineup": my_lineup(state, inputs),
            "outlook": outlook,
        },
        "claims": decisions,
        "teams": teams,
        "market": market(state, inputs, excluded, full),
        "model": {
            "bid_guide": GUIDE_URL,
            "bid_lookahead_weeks": WEEKS - w0,
            "my_future_bid_price_weeks_of_cash_per_point_per_week": decisions["price"],
            "opponent_saving_plans": {
                plan: {"probability": 1 / len(SAVING_PLANS),
                       "reserve_fraction_entering_week": {str(w + 1): share for w, share in anchors}}
                for plan, anchors in SAVING_PLANS.items()
            },
            "race_sims": args.sims,
            "weekly_sigma": WEEKLY_SIGMA,
            "team_season_sigma": TEAM_SEASON_SIGMA,
            "off_cycle_share": round(inputs.off_cycle_share, 3),
            "cascade_rounds": CASCADE_ROUNDS,
            "claims_per_team": CLAIMS_PER_TEAM,
            "claim_candidates": CLAIM_CANDIDATES,
            "roster_size_by_week": list(WEEK_ROSTER_SIZE),
        },
        "validation": {"problems": state.problems, "ok": not state.problems},
    }
    SEASON.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        report(payload)
    report_decisions(payload)
    for problem in state.problems:
        print(f"  - {problem}", file=sys.stderr)
    print(f"wrote {SEASON.name} (week {state.week}, {len(state.players)} players)", file=sys.stderr)
    return 1 if state.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
