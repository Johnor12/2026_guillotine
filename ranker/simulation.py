"""Deterministic draft state and pick policies.

`Draft` resumes an immutable live board, applies roster legality, values my slot, and
uses each opponent's inferred provider order. Fixed-point level convergence is in
`convergence.py`; stochastic redraws and deeper plans are in `planning.py`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from functools import lru_cache

from .board import Board
from .league import (
    DEDICATED_SLOTS,
    LOOKAHEAD_PER_POS,
    OPPONENT_BALANCE_STRENGTH,
    OPPONENT_DEPTH_PENALTY,
    OPPONENT_DEPTH_TARGETS,
    OPPONENT_INTEL,
    OPPONENT_POSITION_TILT,
    POSITIONS,
    STARTING_SLOTS,
    SURVIVAL_SIGMA,
)
from .opponents import OpponentStrategy
from .pool import Player
from .value import (
    Levels,
    insert_sorted,
    pos_sorted,
    sorted_roster,
    starting_positions,
    team_value,
    team_values_with_candidates,
)

# --- availability model -----------------------------------------------------------


class Fenwick:
    """Availability counter over the consensus-sorted pool, for O(log n) 'how many better'."""

    __slots__ = ("n", "tree")

    def __init__(self, n: int) -> None:
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        """Count of set entries strictly before index i."""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def _starts_week1(p: Player) -> bool:
    """Projected to score in week 1 — the depth-chart starter test the opponents use."""
    return p.weekly[0] > 0.0


@lru_cache(maxsize=64)
def _rank_weights(q: float, n: int) -> tuple[float, ...]:
    """rank**-q for ranks 1..n (index 0 unused): the opponents' choice law by source rank."""
    return (0.0,) + tuple(rank**-q for rank in range(1, n + 1))


def survival(better_available: int, gap: int) -> float:
    """P(a player is still there `gap` picks later), given how many rank above him.

    If drafters followed the consensus order exactly he lasts iff more than `gap` players rank
    above him. Softened into a logistic because they follow their own roster needs, not a
    single global list.
    """
    if gap <= 0:
        return 1.0
    return 1.0 / (1.0 + math.exp((gap - better_available) / SURVIVAL_SIGMA))


# --- draft simulation -------------------------------------------------------------


class Draft:
    """One simulated draft from a Board's starting state.

    `noise` > 0 perturbs the other teams' scores only. Everything the board supplies is
    copied, never mutated, so the same Board seeds every iteration of the fixed point and
    all `--sims` noisy redraws.
    """

    def __init__(
        self,
        players: list[Player],
        levels: Levels,
        board: Board,
        noise: float = 0.0,
        rng: random.Random | None = None,
        opponents: dict[int, OpponentStrategy] | None = None,
        forced: dict[int, Player] | None = None,
        targets: dict[int, Player] | None = None,
        noise_from: int = 0,
        my_ban: int | None = None,
    ) -> None:
        self.players = players
        self.levels = levels
        self.board = board
        self.order = board.order
        self.pick_nos = board.pick_nos
        self.my_slot = board.my_slot
        self.noise = noise
        self.rng = rng
        self.opponents = opponents or {}
        missing = set(self.order) - {self.my_slot} - set(self.opponents)
        if missing:
            raise ValueError(f"no source strategy for opponent slot(s) {sorted(missing)}")
        self.by_id = {p.player_id: p for p in players}
        # The fast survival approximation for my later picks needs one league-wide order.
        # Average the 31 opponents' source ranks, counting a repeated source once per
        # manager. This is only an approximation of the slot-specific policies; the
        # decision in front of me is priced by branch-specific Monte Carlo redraws.
        consensus_score = {
            p.player_id: sum(s.ranks[p.player_id] for s in self.opponents.values())
            / len(self.opponents)
            for p in players
        }
        # Pool sorted by opponent consensus drives my fast survival model; per-position
        # lists sorted by my projections drive only my candidate generation.
        self.opponent_sorted = sorted(
            players, key=lambda p: (consensus_score[p.player_id], p.player_id)
        )
        self.opponent_consensus_rank = {
            p.player_id: i for i, p in enumerate(self.opponent_sorted, start=1)
        }
        for i, p in enumerate(self.opponent_sorted):
            p.availability_index = i
        # Already-drafted players stay in the sorted structures (their rank is still a fact
        # about the pool) but carry no availability bit, so `better_available` counts only
        # players who can actually be taken ahead of a candidate.
        self.taken: set[int] = set(board.taken)
        self.avail_bits = Fenwick(len(players))
        for p in players:
            if p.player_id not in self.taken:
                self.avail_bits.add(p.availability_index, 1)
        self.pos_lists = pos_sorted(players)
        self.heads = {pos: 0 for pos in POSITIONS}
        self.rosters: list[list[Player]] = [list(r) for r in board.rosters]
        self.off_pool = board.off_pool  # read-only here; nothing is ever added
        self.pick_of: dict[int, int] = {}
        # My picks' scored candidates as (value_now, next_pick_ev, player), best first.
        # Recorded on deterministic runs only — this is what "who should I draft next"
        # reads off the final draft.
        self.my_decisions: dict[int, list[tuple[float, float, Player]]] = {}
        self.next_pick = self._next_pick_table()
        self.picks_left = list(board.picks_left)
        # Each source order resolved to Player objects once, keyed by the order tuple's
        # identity (the strategies outlive the draft); most opponents share a board.
        self._order_players: dict[int, tuple[Player, ...]] = {}
        # Rollout hooks (see `rollout`): picks dictated by the caller instead of chosen,
        # and the first pick index where the other teams' noise applies — everything
        # before it plays deterministically, so every playout branches from one state.
        self.forced = forced or {}
        # A target yields to an opponent taking him first, then the normal policy chooses.
        # Unlike `forced`, this is a draft plan rather than an assertion about availability.
        self.targets = targets or {}
        self.noise_from = noise_from
        # My slot never drafts this player (candidate_survival's counterfactual redraws).
        self.my_ban = my_ban
        # The QB run so far, for the opponents' scarcity rule (`_qb_run_forces`).
        self.picks_made = board.picks_made
        self.qb_taken = sum(
            1 for roster in self.rosters for p in roster if p.position == "QB"
        ) + sum(1 for off in self.off_pool for row in off if row.get("position") == "QB")
        self.qb_starters_left = sum(
            1
            for p in players
            if p.position == "QB" and _starts_week1(p) and p.player_id not in self.taken
        )
        self.qb_left = sum(
            1 for p in players if p.position == "QB" and p.player_id not in self.taken
        )
        # OPPONENT_INTEL by slot, names resolved to pool players. Synthetic boards
        # (the selftest) carry no real usernames and so no intel.
        by_name = {p.name: p for p in players}
        self.intel: dict[int, list[list[Player] | str]] = {
            strategy.slot: [
                step if isinstance(step, str) else [by_name[name] for name in step]
                for step in OPPONENT_INTEL[strategy.username]
            ]
            for strategy in self.opponents.values()
            if strategy.username in OPPONENT_INTEL
        }

    def _next_pick_table(self) -> list[int | None]:
        """For each pick index, the slot's following pick index (None if it is their last)."""
        nxt: list[int | None] = [None] * len(self.order)
        last_seen: dict[int, int] = {}
        for i in range(len(self.order) - 1, -1, -1):
            slot = self.order[i]
            nxt[i] = last_seen.get(slot)
            last_seen[slot] = i
        return nxt

    # -- availability helpers

    def _advance(self, pos: str) -> None:
        lst = self.pos_lists[pos]
        i = self.heads[pos]
        while i < len(lst) and lst[i].player_id in self.taken:
            i += 1
        self.heads[pos] = i

    def candidates(
        self,
        roster: list[Player],
        per_pos: int = 1,
        picks_left: int | None = None,
        off: Sequence[dict] = (),
    ) -> list[Player]:
        """Best available players at each legal position, honouring the mandatory slots.

        A lineup needs 1 QB, 1 RB, 2 WR, 1 TE and two flex, so a manager cannot spend
        every pick on the best name available and end up without a quarterback, or hoard
        quarterbacks on a one-bench roster. Once the picks remaining are exactly the
        unfilled starting slots, candidates narrow to the positions that fill one. This is the honest way to stop
        a team punting a position; an earlier attempt distorted the value of an empty slot
        instead, which broke the board.

        `off` is the team's already-drafted players the pool cannot value. They count here
        and only here: they occupy a roster spot and they answer a mandatory position, so a
        team that spent a live pick on an unranked quarterback is not made to draft another.
        """
        for positions in self._eligibility_scenarios(roster, picks_left, off):
            out: list[Player] = []
            for pos in positions:
                self._advance(pos)
                found = 0
                for p in self.pos_lists[pos][self.heads[pos] :]:
                    if p.player_id in self.taken:
                        continue
                    out.append(p)
                    found += 1
                    if found == per_pos:
                        break
            if out:
                return out
        return []

    def _eligibility_scenarios(
        self,
        roster: list[Player],
        picks_left: int | None,
        off: Sequence[dict],
    ) -> tuple[tuple[str, ...], ...]:
        """Roster-legality filters shared by my board and the opponent boards: once the
        remaining picks are no more than the unfilled starting slots (flex included, so
        a one-bench roster cannot carry a third QB), only positions that add a starter,
        else every position. The second scenario is the fallback when a position is
        exhausted, which a 370-player pool never reaches."""
        eligible = POSITIONS
        if picks_left is not None:
            positions = [p.position for p in roster] + [
                o["position"] for o in off if o.get("position") in POSITIONS
            ]
            filled = len(starting_positions(positions))
            if picks_left <= sum(STARTING_SLOTS.values()) - filled:
                eligible = tuple(
                    pos
                    for pos in POSITIONS
                    if len(starting_positions(positions + [pos])) > filled
                ) or POSITIONS
        return (eligible, POSITIONS)

    def opponent_candidates(self, pick_index: int, slot: int) -> list[Player]:
        """All legal players, ordered only by this opponent's inferred source board.

        A drafter with OPPONENT_INTEL follows the plan while it lasts, falling back
        to the board when the planned players are gone or the position is exhausted.
        A team still without a quarterback is assumed to be at least this rational
        about the one position every roster must fill from a supply of 32 week-1
        starters: it never spends the pick on a QB who does not start week 1 while a
        starter is on the board, and once the run has left no starter likely to survive
        to its next pick it takes one now (`_qb_run_forces`). Whether a QB starts week
        1 is public depth-chart information, not a projection edge.
        """
        strategy = self.opponents[slot]
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        order_players = self._order_players.get(id(strategy.order))
        if order_players is None:
            order_players = self._order_players[id(strategy.order)] = tuple(
                self.by_id[player_id] for player_id in strategy.order
            )
        taken = self.taken
        candidates: list[Player] = []
        for positions in self._eligibility_scenarios(
            roster, self.picks_left[slot - 1], off
        ):
            if positions == POSITIONS:
                candidates = [p for p in order_players if p.player_id not in taken]
            else:
                candidates = [
                    p
                    for p in order_players
                    if p.player_id not in taken and p.position in positions
                ]
            if candidates:
                break
        plan = self.intel.get(slot)
        if plan is not None and len(roster) + len(off) < len(plan):
            step = plan[len(roster) + len(off)]
            if isinstance(step, str):
                wanted = [p for p in candidates if p.position == step]
            else:
                wanted = [p for p in step if p.player_id not in self.taken][:1]
            if wanted:
                return wanted
        if self.position_counts(roster, off)["QB"] == 0:
            starters = [p for p in candidates if p.position == "QB" and _starts_week1(p)]
            if starters:
                if self._qb_run_forces(pick_index, self.qb_starters_left):
                    return starters
                candidates = [
                    p for p in candidates if p.position != "QB" or _starts_week1(p)
                ]
            else:
                # The starters are gone; the same rule now guards the backups, since
                # 42 pool QBs do not cover 32 teams once ten of them carry two.
                backups = [p for p in candidates if p.position == "QB"]
                if backups and self._qb_run_forces(pick_index, self.qb_left):
                    return backups
        return candidates

    def _qb_run_forces(self, pick_index: int, supply: int) -> bool:
        """Whether none of the `supply` QBs still on the board is expected to survive
        to this slot's next pick, at the rate quarterbacks have gone in this draft or
        given the QB-less teams picking before then. A pseudo-count of 4 QBs per 32 picks stands in before the room has shown
        its rate, so nothing is forced at the open even though the starter supply
        exactly matches the 32 teams."""
        nxt = self.next_pick[pick_index]
        if nxt is None:
            return True
        gap = self.order[pick_index + 1 : nxt]
        rate = (self.qb_taken + 4) / (self.picks_made + 32)
        # Every QB-less team picking in between is under this same rule, so the
        # visible demand bounds the draw from below; the run's rate covers the rest.
        qbless = {
            slot
            for slot in set(gap)
            if self.position_counts(self.rosters[slot - 1], self.off_pool[slot - 1])["QB"] == 0
        }
        return supply - max(rate * len(gap), len(qbless)) < 1.0

    def target_is_legal(self, target: Player, slot: int) -> bool:
        """Whether a surviving plan target belongs to the first viable legal scenario."""
        if target.player_id in self.taken:
            return False
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        for positions in self._eligibility_scenarios(
            roster, self.picks_left[slot - 1], off
        ):
            exists = any(
                p.player_id not in self.taken and p.position in positions
                for p in self.players
            )
            if exists:
                return target.position in positions
        return False

    @staticmethod
    def position_counts(roster: Sequence[Player], off: Sequence[dict]) -> dict[str, int]:
        have = {position: 0 for position in POSITIONS}
        for player in roster:
            have[player.position] += 1
        for row in off:
            if row.get("position") in have:
                have[row["position"]] += 1
        return have

    @staticmethod
    def opponent_depth_penalty(
        roster: Sequence[Player], off: Sequence[dict], position: str
    ) -> float:
        """Opponent preference penalty for an already-deep position."""
        have = Draft.position_counts(roster, off)[position]
        excess_depth = max(have - OPPONENT_DEPTH_TARGETS[position] + 1, 0)
        return OPPONENT_DEPTH_PENALTY**excess_depth

    def opponent_position_adjustments(self, slot: int) -> dict[str, float]:
        """Source-rank multipliers for starter needs and excessive bench depth."""
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        have = self.position_counts(roster, off)
        out: dict[str, float] = {}
        for position, required in DEDICATED_SLOTS.items():
            starter_boost = (
                1.0
                + OPPONENT_BALANCE_STRENGTH
                * max(required - have[position], 0)
                / required
            )
            depth_penalty = self.opponent_depth_penalty(roster, off, position)
            tilt = OPPONENT_POSITION_TILT.get(position, 1.0)
            out[position] = tilt * depth_penalty / starter_boost
        return out

    def better_available(self, p: Player) -> int:
        return self.avail_bits.prefix(p.availability_index)

    # -- valuation

    def lookahead(
        self,
        roster: list[Player],
        roster_sorted: list[Player],
        taking: Player,
        value_cache: dict[tuple[int, ...], float],
        gap: int,
        left: int,
        off: Sequence[dict] = (),
    ) -> float:
        """E[value of the best player still available at this team's next pick].

        The roster objective already includes one unique waiver body at each position.
        Order statistic over the plausible next-pick candidates: the best surviving one
        is candidate i if i survives and everyone better does not. Candidates are treated
        as independent, which slightly understates the chance that a whole position gets
        cleared out between picks.
        """
        future_roster = roster + [taking]
        future_sorted = insert_sorted(roster_sorted, taking)
        base_key = (taking.player_id,)
        base = value_cache.get(base_key)
        if base is None:
            base = team_value(future_sorted, self.levels)
            value_cache[base_key] = base
        scored: list[tuple[float, float]] = []
        for cand in self.candidates(
            future_roster, per_pos=LOOKAHEAD_PER_POS, picks_left=left, off=off
        ):
            if cand.player_id == taking.player_id:
                continue
            a, b = taking.player_id, cand.player_id
            pair_key = (a, b) if a < b else (b, a)
            pair_value = value_cache.get(pair_key)
            if pair_value is None:
                pair_value = team_value(future_sorted, self.levels, cand)
                value_cache[pair_key] = pair_value
            gain = pair_value - base
            scored.append((gain, survival(self.better_available(cand), gap)))
        scored.sort(key=lambda t: -t[0])
        expected = 0.0
        mass = 1.0
        for gain, surv in scored:
            expected += mass * surv * gain
            mass *= 1.0 - surv
            if mass < 1e-4:
                break
        return expected

    def score_my_candidates(
        self, pick_index: int, per_pos: int = 1, extra: Sequence[Player] = ()
    ) -> list[tuple[float, float, Player]]:
        """Roster and next-pick value for my legal candidates at one draft state.

        `extra` adds named players to the shortlist whether or not this state still has
        them: the live shortlist scores long shots the survival redraws then prune."""
        slot = self.my_slot
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        roster_sorted = sorted_roster(roster)
        nxt = self.next_pick[pick_index]
        gap = None if nxt is None else nxt - pick_index - 1
        # Taking A then B values the same roster as B then A.
        lookahead_values: dict[tuple[int, ...], float] = {}
        left = self.picks_left[slot - 1]
        cands = self.candidates(roster, per_pos=per_pos, picks_left=left, off=off)
        target = self.targets.get(pick_index)
        if target is not None and self.target_is_legal(target, slot):
            # Planned later targets can be interior players that the bulk candidate shortcut
            # would omit. Score them too so the reported decision remains inspectable.
            if target.player_id not in {c.player_id for c in cands}:
                cands.append(target)
        for p in extra:
            if p.player_id not in {c.player_id for c in cands}:
                cands.append(p)
        if self.my_ban is not None and slot == self.my_slot:
            # The ban yields to roster legality: if he is my only legal candidate, take him.
            cands = [c for c in cands if c.player_id != self.my_ban] or cands
        base, candidate_values = team_values_with_candidates(roster, self.levels, cands)
        detail: list[tuple[float, float, Player]] = []
        for cand in cands:
            now = candidate_values[cand.player_id] - base
            later = (
                0.0
                if gap is None
                else self.lookahead(
                    roster, roster_sorted, cand, lookahead_values, gap, left - 1, off,
                )
            )
            detail.append((now, later, cand))
        assert detail, "pool exhausted"
        return sorted(detail, key=lambda t: (-(t[0] + t[1]), t[2].player_id))

    def choose(self, pick_index: int, slot: int) -> Player:
        if slot != self.my_slot:
            return self.choose_opponent(pick_index, slot)

        detail = self.score_my_candidates(pick_index)

        if slot == self.my_slot and self.rng is None:
            self.my_decisions[self.pick_nos[pick_index]] = detail

        return detail[0][2]

    def choose_opponent(self, pick_index: int, slot: int) -> Player:
        """Choose from a balance-adjusted source board, without consulting my valuation."""
        candidates = self.opponent_candidates(pick_index, slot)
        assert candidates, "pool exhausted"
        adjustments = self.opponent_position_adjustments(slot)
        if not self.noise or self.rng is None or pick_index < self.noise_from:
            return min(
                (
                    (rank * adjustments[player.position], rank, player)
                    for rank, player in enumerate(candidates, start=1)
                ),
                key=lambda row: (row[0], row[1], row[2].player_id),
            )[2]

        # The investigator measures log2(rank among available) for each real pick.
        # rank_power calibrates source adherence: P(pick) is proportional to the
        # balance-adjusted rank to the power -rank_power/noise, so a nearby player at an
        # unfilled starter position gets better odds without a guaranteed pick, and
        # `noise` scales the random variation around that preference. One inverse-CDF
        # draw per pick; the adjusted rank factors into a rank table and a position term.
        q = self.opponents[slot].rank_power / self.noise
        rank_weight = _rank_weights(q, len(self.players))
        position_weight = {pos: adj**-q for pos, adj in adjustments.items()}
        weights = [
            rank_weight[rank] * position_weight[player.position]
            for rank, player in enumerate(candidates, start=1)
        ]
        u = self.rng.random() * sum(weights)
        for player, weight in zip(candidates, weights):
            u -= weight
            if u < 0.0:
                return player
        return candidates[-1]  # rounding left a sliver past the last weight

    def run(
        self,
        until_taken: int | None = None,
        stop_before: int | None = None,
    ) -> int | None:
        """Play pending picks, optionally stopping at a player or before a pick index."""
        for i, slot in enumerate(self.order):
            if i == stop_before:
                break
            pick = self.forced.get(i)
            if pick is None:
                target = self.targets.get(i)
                if target is not None and self.target_is_legal(target, slot):
                    # Deterministic replays still score the ordinary choice for reporting;
                    # noisy rollout workers only need to exercise the planned target.
                    if self.rng is None and slot == self.my_slot:
                        self.choose(i, slot)
                    pick = target
                else:
                    pick = self.choose(i, slot)
            else:
                assert pick.player_id not in self.taken, f"forced pick {pick.name} already taken"
            self.taken.add(pick.player_id)
            self.avail_bits.add(pick.availability_index, -1)
            self.rosters[slot - 1].append(pick)
            self.picks_left[slot - 1] -= 1
            self.pick_of[pick.player_id] = self.pick_nos[i]
            self.picks_made += 1
            if pick.position == "QB":
                self.qb_taken += 1
                self.qb_left -= 1
                self.qb_starters_left -= _starts_week1(pick)
            if pick.player_id == until_taken:
                return self.pick_nos[i]
        return None
