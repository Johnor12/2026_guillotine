# Pool pipeline

This independent, offline pipeline turns a saved DraftSharks projections page into the
league's `pool.json`. It is rerun when projections change, not during every live-draft
refresh.

## Stages

```text
data/projections.html
  -> parse_projections.py -> data/projections.json
  -> build_pool.py        -> ../pool.json
  -> match_sleeper.py     -> ../pool.json with sleeper_id
```

`pipeline.py` runs those stages in order and stops on the first failure. Every stage is
also a standalone CLI:

```bash
uv run pool_pipeline/pipeline.py --report
uv run pool_pipeline/pipeline.py --only pool
uv run pool_pipeline/parse_projections.py in.html -o out.json
uv run pool_pipeline/build_pool.py --limit 450 -o big.json
uv run pool_pipeline/match_sleeper.py --report
uv run pool_pipeline/fetch_sleeper.py
uv run pool_pipeline/fetch_sleeper_projections.py
```

`fetch_sleeper.py` is manual and is not a pipeline stage. Sleeper's player dump is about
14 MB and should not be downloaded more than once per day. It is cached under `data/`;
the small metadata file records when it was fetched.

`fetch_sleeper_projections.py` is also manual: it writes `data/sleeper_projections.json`,
the top 250 players by Sleeper/Rotowire season projection (no kickers) with half-PPR ADP,
scored with the league's own settings so the numbers match the league players page.

## File contracts

`projections.json` is a faithful provider export: identity fields, eight scoring
schemes, four horizons, displayed and derived ranks, ADP, and analysis text. The printed
ranks in the saved HTML are stale; consumers use the gap-free ranks derived from 3D value.
The saved page is the provider's dynasty export; this league consumes only its one-season
projection, which is an ordinary season projection.

`pool.json` is the narrow draft input: every usable QB/RB/WR/TE player, with 11
fields per player. The ranker uses projected points, not DraftSharks' provider-scaled 3D
value.

- `points`: one-season points in this league's scoring (0.5/rec, no TE premium),
  copied from the provider's `half_ppr` column for every position
- `adp`: overall 1QB ADP decoded from the provider's 12-team round.pick notation
- `sleeper_id`: the join key used by the live draft and investigator
- `rank`: descending `points`, with the provider's overall rank breaking ties

## Sleeper matching

There is no shared provider id, so `match_sleeper.py` uses three conservative name
tiers: full normalized name; name without suffix; then last name plus team and nearby
age. Position must always agree, ambiguity is left unmatched, and duplicate Sleeper ids
are fatal. Re-running is idempotent because ids are dropped and re-derived.
