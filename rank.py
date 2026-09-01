#!/usr/bin/env python3
"""Roster-aware draft board for this league, with an optimal-drafter draft simulation.

    uv run rank.py               # pool.json + draft.json + source boards -> rankings.json
    uv run rank.py --report      # + board, convergence and recommendation on stderr
    uv run rank.py --selftest    # verify solver, opponents, planning, and board loader

Scope is this league and nothing else; the league constants and strategy knobs live in
ranker/league.py. The value input is `weekly_points` from `pool.json`: DraftSharks'
per-week projections in this league's 0.5 PPR + TE premium scoring for weeks 1-17, with
byes and known absences as zero weeks, each player's season total blended equally with
Sleeper's projection and the market's rank-matched level so the draft does not chase
one source's outliers (`ranker/market.py`). The method, in one breath: this is a guillotine
league, so a roster is valued week by week as expected optimal lineup points under that
week's starting shape and position-wide availability, and the weeks are combined by
converged guillotine weights, each week's weight being the marginal effect of a weekly
point on log P(surviving that week's cut), with the week 16-17 championship entering
through log P(winning the final) (`ranker/guillotine.py`). What the waiver wire holds
each week is an *outcome* of how the league drafts and who gets eliminated, so the
weekly wire (the undrafted tail early, then the survivors' split of every eliminated
roster, which by the final is other teams' first-round picks) is measured from the
converged draft and feeds valuation as tiered waiver bodies per position per week
(`ranker/value.py`). `draft.json`, the live board, is the
simulation's starting state, not a filter (`ranker/board.py`). Only my slot uses the
projection-based roster objective. Every opponent uses the external provider board most
associated with its prior picks, loaded from the source investigator; unfilled dedicated
starters softly adjust that order, and its observed adherence controls Monte Carlo
choice noise. Opponent picks never use my projections or board.

The board's headline `lineup_gain` is the decision metric itself: the player's marginal
guillotine-weighted weekly lineup value on my current roster at the converged levels.
The `my_next_picks` block is the direct answer to "who should I draft next": on top of
that same gain it prices opponent demand and what my following picks keep. Its first
decision searches target plans across my next four held picks, then plays each
first-candidate plan out to the end of the draft (`ranker/planning.py`).

Python stdlib only. Deterministic: every tie breaks on player_id and the RNG is seeded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ranker.board import load_board
from ranker.convergence import converge
from ranker.league import NOISE, ROLLOUT_SIMS, SEED, SIMS
from ranker.market import blend_to_market
from ranker.opponents import load_opponent_strategies
from ranker.output import build_payload
from ranker.planning import (
    apply_option_redraw,
    apply_rollout,
    apply_survival_floor,
    broaden_first_pick,
    candidate_survival,
    four_pick_lookahead,
    monte_carlo,
    option_redraw,
    rollout,
)
from ranker.pool import load_pool
from ranker.rankings import build_rankings
from ranker.report import report_board, report_summary
from ranker.selftest import selftest
from ranker.validate import validate

REPO_ROOT = Path(__file__).resolve().parent
POOL = REPO_ROOT / "pool.json"
DRAFT = REPO_ROOT / "draft.json"
RANKINGS = REPO_ROOT / "rankings.json"
SOURCE_MATCHES = REPO_ROOT / "data_source_matches.json"
SOURCE_BOARDS = REPO_ROOT / "sources/data/boards.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", action="store_true", help="validation summary on stderr")
    ap.add_argument("--selftest", action="store_true", help="offline checks, then exit")
    args = ap.parse_args(argv)

    players, pool_meta = load_pool(POOL)
    if args.selftest:
        return selftest(players)
    blend_to_market(players, SOURCE_BOARDS)

    try:
        draft_raw = json.loads(DRAFT.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {DRAFT.name}: {exc}; run draft/fetch_draft.py", file=sys.stderr)
        return 1
    board, board_problems = load_board(draft_raw, players, DRAFT.name)
    try:
        opponents = load_opponent_strategies(players, board, draft_raw, SOURCE_MATCHES, SOURCE_BOARDS)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"cannot build opponent strategies: {exc}; run sources/investigate.py",
            file=sys.stderr,
        )
        return 1

    if args.report:
        print(
            f"pool: {len(players)} players "
            + ", ".join(f"{k} {v}" for k, v in pool_meta["by_position"].items()),
            file=sys.stderr,
        )
        report_board(board)
        print("opponent source strategies:", file=sys.stderr)
        for slot in sorted(opponents):
            strategy = opponents[slot]
            print(
                f"  slot {slot:>2} {strategy.username or 'unknown':<20} "
                f"{strategy.source_name:<25} fit {strategy.fit_score:>4.1f}, "
                f"loss {strategy.mean_log2_loss:.3f}",
                file=sys.stderr,
            )
        print("converging levels:", file=sys.stderr)

    t0 = time.perf_counter()

    def stage(name: str) -> None:
        nonlocal t0
        if args.report:
            print(f"  [{name} {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
        t0 = time.perf_counter()

    levels, draft, history, guillotine = converge(players, board, args.report, opponents)
    stage("converge")
    draft = broaden_first_pick(draft, players, board, levels, opponents)

    def first_pick_candidates():
        return [c for _, _, c in draft.my_decisions.get(board.my_picks[0], [])] if board.my_picks else []

    candidates = first_pick_candidates()
    if args.report and candidates:
        print(
            f"candidate survival: {SIMS} banned-me redraws x {len(candidates)} live-board candidates",
            file=sys.stderr,
        )
    survival = candidate_survival(players, board, levels, candidates, SIMS, NOISE, SEED, opponents)
    stage("survival")
    draft = apply_survival_floor(draft, board, survival)
    candidates = first_pick_candidates()
    if args.report and candidates:
        print(
            f"next-pick options: {SIMS} conditional opponent redraws x {len(candidates)} candidates",
            file=sys.stderr,
        )
    options = option_redraw(players, board, levels, candidates, SIMS, NOISE, SEED, opponents)
    stage("options")
    draft = apply_option_redraw(draft, options, players, board, levels, opponents)
    candidates = first_pick_candidates()
    if args.report and candidates:
        print(
            f"four-pick lookahead: broadened {len(candidates)}-candidate first-pick pool",
            file=sys.stderr,
        )
    lookahead = four_pick_lookahead(
        players, board, levels, candidates, survival, opponents, NOISE, SEED
    )
    stage("lookahead")
    if args.report and candidates:
        print(
            f"rollout: {ROLLOUT_SIMS} full-draft playouts x {len(candidates)} four-pick plan sets",
            file=sys.stderr,
        )
    rolled = rollout(
        players, board, levels, candidates, ROLLOUT_SIMS, NOISE, SEED, opponents, lookahead
    )
    draft = apply_rollout(draft, rolled, players, board, levels, opponents)
    stage("rollout")

    if args.report:
        print(f"monte carlo: {SIMS} source-strategy drafts (noise multiplier={NOISE})", file=sys.stderr)
    picks = monte_carlo(players, board, levels, SIMS, NOISE, SEED, opponents)
    stage("monte carlo")
    rows = build_rankings(players, levels, draft, picks, SIMS, board)

    problems = board_problems + validate(rows, players, levels, draft, board, history)
    payload = build_payload(
        players, pool_meta, board, levels, draft, history, rows, problems,
        opponents, guillotine, options, rolled, survival,
    )
    RANKINGS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.report:
        report_summary(rows, draft, board, rolled, survival, guillotine)
    if problems:
        print(f"\n{len(problems)} VALIDATION PROBLEM(S):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    print(f"wrote {RANKINGS.name} ({len(rows)} undrafted of {len(players)})", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
