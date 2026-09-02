"""Every-run invariants over the board, the simulation and the emitted rows.

These run on every invocation and land in `validation.problems`; a non-empty list is a
non-zero exit. `--selftest` (selftest.py) covers the states a real draft.json cannot
currently reach.
"""

from __future__ import annotations

from . import league
from .board import Board
from .league import (
    DEDICATED_SLOTS,
    MAX_ITERS,
    POSITIONS,
    STARTING_SLOTS,
    WEEKS,
    draft_order,
    pick_label,
    picks_for_slot,
)
from .pool import Player
from .simulation import Draft
from .value import Levels, starting_positions, tier_bodies


def validate(
    rows: list[dict],
    players: list[Player],
    levels: Levels,
    draft: Draft,
    board: Board,
    history: dict,
) -> list[str]:
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    # The static snake is the yardstick even on a live board, pinned to the README.
    full = picks_for_slot(board.my_slot, draft_order())
    readme = ["1.20", "2.13", "3.13", "4.20", "5.13", "6.20", "7.13", "8.20"]
    labels = [pick_label(p) for p in full]
    check(labels == readme, f"draft order {labels} != README {readme}")
    check(
        len(full) == league.ROUNDS,
        f"{len(full)} picks for slot {board.my_slot}, want {league.ROUNDS}",
    )
    check(
        len(board.order) == len(board.pick_nos),
        f"board has {len(board.order)} owners for {len(board.pick_nos)} pending picks",
    )
    accounted = board.picks_made + len(board.order)
    check(
        accounted == league.TOTAL_PICKS,
        f"board accounts for {accounted} picks, want {league.TOTAL_PICKS}",
    )
    check(
        sum(board.owed_size(s) for s in range(1, league.TEAMS + 1)) == league.TOTAL_PICKS,
        "the picks each team owns do not sum to the board",
    )

    if board.live:
        first = board.pick_nos[0] if board.pick_nos else league.TOTAL_PICKS + 1
        check(
            board.pick_nos == list(range(first, league.TOTAL_PICKS + 1)),
            "pending picks are not the contiguous tail of the board",
        )
        clock = (board.live.get("on_the_clock") or {}).get("pick_no")
        check(
            clock is None or clock == first,
            f"on the clock is pick {clock}, board resumes at {first}",
        )
        mine = (board.live.get("next_pick_of_mine") or {}).get("pick_no")
        check(
            mine is None or board.my_picks[:1] == [mine],
            f"draft.json says my next pick is {mine}, board says {board.my_picks[:1]}",
        )
        if not board.live.get("traded_picks"):
            want = [n for n in full if n >= first]
            check(
                board.my_picks == want,
                f"my {len(board.my_picks)} remaining picks are not the snake's tail "
                f"({len(want)} picks) even though no picks were traded",
            )

    want_taken = sum(len(r) for r in board.rosters) + len(board.order)
    check(
        len(draft.taken) == want_taken,
        f"{len(draft.taken)} unique pool players drafted, want {want_taken}",
    )
    for i, roster in enumerate(draft.rosters, start=1):
        made = board.rosters[i - 1]
        off = board.off_pool[i - 1]
        # Made picks are facts: they must come through the simulation untouched, in order.
        check(
            roster[: len(made)] == made,
            f"slot {i} lost or reordered one of its {len(made)} made picks",
        )
        want = len(made) + board.picks_left[i - 1]
        check(len(roster) == want, f"slot {i} ends with {len(roster)} players, want {want}")
        # Off-pool picks fill slots too: a live pick on an unranked QB is still a QB.
        starters = starting_positions(
            [p.position for p in roster] + [o["position"] for o in off if o.get("position") in POSITIONS]
        )
        check(
            len(starters) == sum(STARTING_SLOTS.values()),
            f"slot {i} cannot field a full lineup "
            f"({len(starters)}/{sum(STARTING_SLOTS.values())})",
        )
        for pos, need in DEDICATED_SLOTS.items():
            have = sum(1 for p in roster if p.position == pos)
            have += sum(1 for o in off if o.get("position") == pos)
            check(have >= need, f"slot {i} has {have} {pos}, needs {need}")
    gains = [r["lineup_gain"] for r in rows]
    check(gains == sorted(gains, reverse=True), "rows are not sorted by lineup gain descending")
    source_ids = {strategy.source_id for strategy in draft.opponents.values()}
    # Before any opponent has picked, all of them share the cold-start board.
    if any(board.rosters[slot - 1] for slot in draft.opponents):
        check(len(source_ids) >= 2, f"opponents use only {len(source_ids)} distinct source board(s)")
    check(
        any(row["opponent_rank_delta"] != 0 for row in rows),
        "opponent consensus order does not diverge from my board order",
    )
    want_rows = len(players) - len(board.taken)
    check(
        len(rows) == want_rows,
        f"{len(rows)} rows for {want_rows} undrafted players in a {len(players)}-player pool",
    )
    check(
        not (board.taken & {r["player_id"] for r in rows}),
        "a player already drafted in draft.json is on the emitted board",
    )
    # Level invariants: the week weights are a normalized distribution over league
    # weeks, and the weekly wire stays inside the pool's weekly range while never
    # dipping below the eliminated-roster floor it was combined with.
    check(
        abs(sum(levels.weights) - 1.0) < 1e-9,
        f"week weights sum to {sum(levels.weights):.6f}, want 1",
    )
    check(
        len(levels.weights) == WEEKS and all(w >= 0.0 for w in levels.weights),
        "week weights are not a nonnegative length-17 distribution",
    )
    for i, k in enumerate(POSITIONS):
        pool_max = max(
            (max(p.weekly) for p in players if p.position == k), default=0.0
        )
        wire_col = levels.wire[i]
        check(
            len(wire_col) == WEEKS
            and all(
                0.0 <= v <= pool_max + 1e-6 for bodies in wire_col for v in bodies
            ),
            f"{k} weekly wire outside the pool's weekly range [0, {pool_max:.1f}]",
        )
        check(
            all(
                v + 1e-9 >= f
                for w, (bodies, dropped) in enumerate(zip(wire_col, levels.dropped[i]))
                for v, f in zip(bodies, tier_bodies(dropped, k, w))
            ),
            f"{k} weekly wire dips below what the eliminated rosters alone supply",
        )
    check(
        history["iterations_run"] < MAX_ITERS,
        f"no limit cycle found within {MAX_ITERS} iterations",
    )
    return problems
