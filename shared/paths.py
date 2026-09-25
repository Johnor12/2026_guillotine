"""Every file the processes exchange, anchored to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Generated artifacts and the dashboards that read them; shared/serve.py serves this
# directory, so the dashboards fetch the JSON by bare filename.
OUT = ROOT / "out"

# Shared inputs and the pool built from them.
PROJECTIONS_HTML = ROOT / "shared/data/projections.html"  # hand-saved DraftSharks page
SLEEPER_PROJECTIONS = ROOT / "shared/data/sleeper_projections.json"
WEEKLY_PROJECTIONS = ROOT / "shared/data/weekly_projections.json"
POOL = OUT / "pool.json"

# Draft.
DRAFT = OUT / "draft.json"
BOARDS = ROOT / "draft/sources/data/boards.json"
RAW_BOARDS = ROOT / "draft/sources/data/raw"
SOURCE_MATCHES = OUT / "data_source_matches.json"
RANKINGS = OUT / "rankings.json"

# Season.
LEAGUE = OUT / "league.json"
SEASON = OUT / "season.json"
BIDDING_EVALUATION = OUT / "bidding_evaluation.json"
