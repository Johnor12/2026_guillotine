"""Every file the processes exchange, anchored to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared inputs and the pool built from them.
PROJECTIONS_HTML = ROOT / "shared/data/projections.html"  # hand-saved DraftSharks page
SLEEPER_PROJECTIONS = ROOT / "shared/data/sleeper_projections.json"
WEEKLY_PROJECTIONS = ROOT / "shared/data/weekly_projections.json"
POOL = ROOT / "pool.json"

# Draft.
DRAFT = ROOT / "draft.json"
BOARDS = ROOT / "draft/sources/data/boards.json"
RAW_BOARDS = ROOT / "draft/sources/data/raw"
SOURCE_MATCHES = ROOT / "data_source_matches.json"
RANKINGS = ROOT / "rankings.json"

# Season.
LEAGUE = ROOT / "league.json"
SEASON = ROOT / "season.json"
BIDDING_EVALUATION = ROOT / "bidding_evaluation.json"
