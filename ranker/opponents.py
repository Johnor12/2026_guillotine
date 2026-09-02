"""Opponent boards inferred from this league's completed picks.

Each opponent uses the provider board that best fits their picks in
``data_source_matches.json``. Provider rows join the pool by the Sleeper id the source
build already resolved against this same pool; a row it could not resolve is a player
the pool does not carry. Before an opponent has picked, they follow the source build's
``cold_start`` board: Sleeper's half-PPR ADP, the list this league's draft room
displays (a league mock matched ``adp_half_ppr`` to the decimal) and autopick drafts
from, blended 30% toward the format-adjusted boards for the savvy minority. A
provider's uncovered tail follows the consensus board (the mean rank across every
other source), so every mandatory pick remains possible without ever falling back to
this ranker's projections or board.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .board import Board
from .pool import Player

COLD_START_SOURCE_ID = "cold_start"
COLD_START_LOG2_LOSS = 1.5
# An owner whose 1-2 observed picks happen to sit exactly on a source board gets a
# fitted loss near 0, which calibrates to a near-deterministic policy (replayed picks
# then cost 30-80 bits when such an owner deviates). Shrink the fitted loss toward the
# cold-start prior as if that prior had been observed on this many extra picks.
LOSS_SHRINK_PSEUDO_PICKS = 2


@dataclass(frozen=True, slots=True)
class OpponentStrategy:
    slot: int
    roster_id: int
    username: str | None
    source_id: str
    source_name: str
    source_format: str | None
    fit_score: float
    confidence: str
    mean_log2_loss: float
    rank_power: float
    primary_players: int
    ranks: dict[int, int]
    order: tuple[int, ...]

    def public(self) -> dict:
        return {
            "draft_slot": self.slot,
            "roster_id": self.roster_id,
            "username": self.username,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_format": self.source_format,
            "fit_score": self.fit_score,
            "confidence": self.confidence,
            "mean_log2_loss": self.mean_log2_loss,
            "rank_power": round(self.rank_power, 3),
            "players_ranked_by_source": self.primary_players,
        }


def expected_log2_rank(power: float, choices: int) -> float:
    """Expected log2 choice rank under P(rank=k) proportional to k**-power."""
    weights = [k**-power for k in range(1, choices + 1)]
    total = sum(weights)
    return sum(w * math.log2(k) for k, w in enumerate(weights, start=1)) / total


def rank_power(mean_log2_loss: float, choices: int) -> float:
    """Calibrate source-rank choice noise to the investigator's observed loss."""
    target = min(max(mean_log2_loss, 0.0), expected_log2_rank(0.0, choices))
    lo, hi = 0.0, 32.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if expected_log2_rank(mid, choices) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _source_order(source: dict, by_sleeper: dict[str, Player]) -> list[int]:
    """The provider's pool players in its rank order, by the resolved Sleeper id."""
    order: list[int] = []
    seen: set[int] = set()
    for row in source["players"]:
        player = by_sleeper.get(str(row.get("sleeper_id")))
        if player is not None and player.player_id not in seen:
            seen.add(player.player_id)
            order.append(player.player_id)
    return order


def _complete_order(
    source: dict,
    fallback_order: list[int],
    by_sleeper: dict[str, Player],
) -> tuple[int, ...]:
    primary = _source_order(source, by_sleeper)
    primary_ids = set(primary)
    return tuple(
        primary + [player_id for player_id in fallback_order if player_id not in primary_ids]
    )


def build_opponent_strategies(
    players: list[Player],
    board: Board,
    draft: dict,
    matches: dict,
    rankings: dict,
) -> dict[int, OpponentStrategy]:
    """Build one complete external player order for every opposing draft slot."""

    draft_id = draft.get("draft_id")
    if matches.get("draft", {}).get("draft_id") != draft_id:
        raise ValueError(
            f"source matches describe draft {matches.get('draft', {}).get('draft_id')}, "
            f"not {draft_id}"
        )
    snapshot = matches.get("ranking_snapshot", {}).get("generated_at")
    if snapshot != rankings.get("generated_at"):
        raise ValueError(
            f"source matches used ranking snapshot {snapshot}, but rankings are "
            f"from {rankings.get('generated_at')}"
        )

    by_sleeper = {str(p.sleeper_id): p for p in players if p.sleeper_id is not None}
    if len(by_sleeper) != len(players):
        raise ValueError("every pool player needs a unique sleeper_id for opponent boards")

    sources = {source["id"]: source for source in rankings["sources"]}
    fallback = sources.get("consensus")
    if fallback is None:
        raise ValueError("rankings have no consensus tail fallback")
    fallback_order = _source_order(fallback, by_sleeper)
    if len(fallback_order) != len(players):
        raise ValueError(
            f"consensus covers {len(fallback_order)}/{len(players)} pool players"
        )

    owner_matches = {owner["roster_id"]: owner for owner in matches["owners"]}
    slots = {slot["draft_slot"]: slot for slot in draft.get("slots", [])}
    strategies: dict[int, OpponentStrategy] = {}
    for slot in range(1, len(slots) + 1):
        if slot == board.my_slot:
            continue
        slot_row = slots.get(slot)
        if slot_row is None or slot_row.get("roster_id") is None:
            raise ValueError(f"draft slot {slot} has no roster_id")
        roster_id = slot_row["roster_id"]
        owner = owner_matches.get(roster_id)
        if owner is None:
            inferred = {
                "source_id": COLD_START_SOURCE_ID,
                "fit_score": 0.0,
                "mean_log2_loss": COLD_START_LOG2_LOSS,
            }
            confidence = "insufficient"
        else:
            inferred = owner["inferred_source"]
            confidence = owner["confidence"]
        source_id = inferred["source_id"]
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"inferred source {source_id!r} is absent from rankings")

        primary = _source_order(source, by_sleeper)
        order = _complete_order(source, fallback_order, by_sleeper)
        if len(order) != len(players):
            raise ValueError(
                f"opponent slot {slot}'s {source_id} board covers "
                f"{len(order)}/{len(players)} players"
            )
        picks_seen = owner["pick_count"] if owner else 0
        loss = (
            picks_seen * float(inferred["mean_log2_loss"])
            + LOSS_SHRINK_PSEUDO_PICKS * COLD_START_LOG2_LOSS
        ) / (picks_seen + LOSS_SHRINK_PSEUDO_PICKS)
        strategies[slot] = OpponentStrategy(
            slot=slot,
            roster_id=roster_id,
            username=owner.get("username") if owner else slot_row.get("username"),
            source_id=source_id,
            source_name=source["name"],
            source_format=source.get("format"),
            fit_score=float(inferred["fit_score"]),
            confidence=confidence,
            mean_log2_loss=loss,
            rank_power=rank_power(loss, len(players)),
            primary_players=len(primary),
            ranks={player_id: rank for rank, player_id in enumerate(order, start=1)},
            order=order,
        )

    expected_slots = set(range(1, len(slots) + 1)) - {board.my_slot}
    if set(strategies) != expected_slots:
        raise ValueError(
            f"built opponent strategies for slots {sorted(strategies)}, "
            f"want {sorted(expected_slots)}"
        )
    return strategies


def load_opponent_strategies(
    players: list[Player],
    board: Board,
    draft: dict,
    matches_path: Path,
    rankings_path: Path,
) -> dict[int, OpponentStrategy]:
    """Load source artifacts and build every opposing draft-slot policy."""
    matches = json.loads(matches_path.read_text())
    rankings = json.loads(rankings_path.read_text())
    return build_opponent_strategies(players, board, draft, matches, rankings)
