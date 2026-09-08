#!/usr/bin/env python3
"""In-season decisions and odds for this league, from the live Sleeper state.

    uv run season.py             # pool.json + weekly projections + league.json -> season.json
    uv run season.py --report    # + this week's lineup, claims and league odds on stderr

The value input is per-week: DraftSharks' weekly projection blended 2:1 with Sleeper's
for the same week (ranker/season.py). The engine is an agent-based race
(ranker/race.py): every alive team fields its optimal lineup each week under the
draft model's noise, the bottom two are cut, their players hit the wire, and the
survivors bid FAAB from their remaining budgets under league.py's claim rule. Run once
without me it is the market and the elimination bars I face, which price my roster
variants (ranker/claims.py: this week's lineup, and for each free agent the bid that
maximizes my title odds against what he clears for); run with all 32 it is every
team's chance of being cut this week, of reaching the final, and of the title, along
with its expected spend and budget path.

Python stdlib only. Deterministic: the race is seeded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from ranker import league
from ranker.claims import claims, my_lineup
from ranker.league import RACE_SIMS, REGULAR_WEEKS, SEED, WEEKS
from ranker.race import race_inputs, run_races
from ranker.season import lineup_points, load_season

REPO_ROOT = Path(__file__).resolve().parent
POOL = REPO_ROOT / "pool.json"
WEEKLY = REPO_ROOT / "pool/data/weekly_projections.json"
LEAGUE = REPO_ROOT / "league.json"
SEASON = REPO_ROOT / "season.json"


def league_odds(state, inputs, records: list[dict]) -> list[dict]:
    """Every team's fate from the full race: cut-this-week, survival, title, spend."""
    w0 = inputs.week0
    n = len(records)
    out = []
    for i, team in enumerate(state.teams):
        cuts = [r["cut_week"][i] for r in records]
        alive_by_week = [
            sum(1 for c in cuts if c is None or c > w) / n for w in range(w0, REGULAR_WEEKS)
        ]

        def budget_entering(week: int) -> float | None:
            w = week - 1
            if w < w0:
                return None
            alive = [r["budget_path"][w - w0][i] for r in records if r["cut_week"][i] is None or r["cut_week"][i] >= w]
            return round(statistics.fmean(alive)) if alive else None

        spend_now = statistics.fmean(inputs.budgets[i] - r["budget_path"][0][i] for r in records)
        won_now = [b for r in records for j, t, b in r["claims"][w0] if t == i]
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
                "p_cut_now": round(sum(1 for c in cuts if c == w0) / n, 4),
                "p_alive_by_week": [round(x, 4) for x in alive_by_week],
                "p_reach_final": round(sum(1 for c in cuts if c is None) / n, 4),
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
            if r["cut_week"][i] is None or r["cut_week"][i] >= w
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
    # This week's simulated claims by the room, as a check against the observed ones.
    claimed: dict[int, list[int]] = {}
    for r in excluded:
        for j, _, bid in r["claims"][w0]:
            claimed.setdefault(j, []).append(bid)
    simulated = sorted(
        (
            {
                "name": state.players[j].name,
                "position": state.players[j].position,
                "p_taken": round(len(bids) / n, 3),
                "mean_bid": round(statistics.fmean(bids)),
                "max_bid": max(bids),
            }
            for j, bids in claimed.items()
        ),
        key=lambda r: (-r["mean_bid"], -r["p_taken"]),
    )[:40]
    return {
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


def report(payload: dict) -> None:
    me = payload["me"]
    print(
        f"\nweek {payload['week']}: {payload['alive']} alive; {me['name']} FAAB ${me['faab_left']}; "
        f"P(cut this week) {me['p_cut_now']:.1%}, P(final) {me['p_reach_final']:.1%}, "
        f"P(title) {me['p_title']:.1%} (replay {payload['claims']['base']['p_title']:.1%}, "
        f"policy {payload['claims']['policy']})",
        file=sys.stderr,
    )
    lu = me["lineup"]
    print(f"lineup {lu['total']} vs bar {me['bar_now']['mean']} (p90 {me['bar_now']['p90']}):", file=sys.stderr)
    for s in lu["slots"]:
        print(f"  {s['slot']:<5} {s.get('name') or '(empty)':<24} {s.get('points', 0):>5}", file=sys.stderr)
    if lu["start"] or lu["sit"]:
        print(f"  start {lu['start']}, sit {lu['sit']}", file=sys.stderr)
    print("FAAB policy check (title odds relative to standing pat):", file=sys.stderr)
    for pol, rows in payload["claims"]["baseline"].items():
        print("  " + pol + ": " + ", ".join(f"${r['budget']} {r['relative']:+.0f}%" for r in rows), file=sys.stderr)
    print(f"claims ({'pending' if payload['claims']['pending'] else 'not pending: free agents only'}):", file=sys.stderr)
    for c in payload["claims"]["candidates"][:12]:
        cl = c["clearing"]
        print(
            f"  {c['name']:<22} {c['position']} ros {c['ros_per_week']:>4} wk {c['points_this_week']:>4} "
            f"bid ${c['optimal_bid']:<4} win {c['p_win_at_optimal']:.0%} title {c['title_at_optimal']:+.1f}% "
            f"(free {c['title_if_free']:+.1f}%, even ${c['break_even_bid']}, market p50 {cl['p50']} p90 {cl['p90']} "
            f"claimed {cl['p_claimed']:.0%}) drop {c['drop']['name'] if c['drop'] else '-'}",
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
    if observed:
        print("observed waiver claims (latest):", file=sys.stderr)
        for o in observed:
            print(f"  wk{o['week']} {o['team']}: {o['adds']} ${o['bid']} {o['status']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="summary on stderr")
    args = ap.parse_args(argv)

    state = load_season(POOL, WEEKLY, LEAGUE)
    inputs = race_inputs(state)
    t0 = time.perf_counter()
    excluded, full = run_races(inputs, RACE_SIMS, SEED)
    if args.report:
        print(f"[races {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    t0 = time.perf_counter()
    teams = league_odds(state, inputs, full)
    mine = next(t for t in teams if t["is_mine"])
    decisions = claims(state, inputs, excluded, mine["p_title"])
    if args.report:
        print(f"[claims {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
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
            "in this league's scoring; Sleeper alone for players outside the draft pool"
        ),
        "method": (
            "Agent-based season race: optimal weekly lineups under the draft model's noise, "
            "bottom two cut, survivors bid FAAB on the wire; see season.py and ranker/race.py."
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
            "race_sims": RACE_SIMS,
            "weekly_sigma": league.WEEKLY_SIGMA,
            "team_season_sigma": league.TEAM_SEASON_SIGMA,
            "claim_full_budget_gain": league.CLAIM_FULL_BUDGET_GAIN,
            "claim_gain_exponent": league.CLAIM_GAIN_EXPONENT,
            "claim_conservation_floor": league.CLAIM_CONSERVATION_FLOOR,
            "claim_conservation_full_week": league.CLAIM_CONSERVATION_FULL_WEEK,
            "claim_noise_sigma": league.CLAIM_NOISE_SIGMA,
            "claims_per_team": league.CLAIMS_PER_TEAM,
            "claim_candidates": league.CLAIM_CANDIDATES,
            "faab_hold_weeks": league.FAAB_HOLD_WEEKS,
            "faab_spend_week": league.FAAB_SPEND_WEEK,
            "roster_size_by_week": list(league.WEEK_ROSTER_SIZE),
        },
        "validation": {"problems": state.problems, "ok": not state.problems},
    }
    SEASON.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        report(payload)
    for problem in state.problems:
        print(f"  - {problem}", file=sys.stderr)
    print(f"wrote {SEASON.name} (week {state.week}, {len(state.players)} players)", file=sys.stderr)
    return 1 if state.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
