# Pool pipeline

This independent, offline pipeline turns a saved DraftSharks projections page into the
league's `pool.json`. It is rerun when projections change, not during every live-draft
refresh.

## Stages

```text
data/projections.html
  -> parse_projections.py     -> data/projections.json
  -> build_pool.py            -> ../pool.json
  -> match_sleeper.py         -> ../pool.json with sleeper_id
  -> apply_sleeper_points.py  -> ../pool.json re-priced from data/sleeper_projections.json
```

`pipeline.py` runs those stages in order and stops on the first failure. Every stage is
also a standalone CLI:

```bash
uv run pool_pipeline/pipeline.py --report
uv run pool_pipeline/pipeline.py --only pool
uv run pool_pipeline/parse_projections.py in.html -o out.json
uv run pool_pipeline/build_pool.py --limit 450 -o big.json
uv run pool_pipeline/match_sleeper.py --report
uv run pool_pipeline/apply_sleeper_points.py --report
uv run pool_pipeline/fetch_sleeper.py
uv run pool_pipeline/fetch_sleeper_projections.py
```

`fetch_sleeper.py` is manual and is not a pipeline stage. Sleeper's player dump is about
14 MB and should not be downloaded more than once per day. It is cached under `data/`;
the small metadata file records when it was fetched.

`fetch_sleeper_projections.py` is also manual: it writes `data/sleeper_projections.json`,
the top 500 QB/RB/WR/TE players by Sleeper/Rotowire season projection with half-PPR ADP,
scored with the league's own settings (TE premium included) so the numbers match the
league players page.
`apply_sleeper_points.py` (stage 4) then reads that committed file — refetch it when the
projections should move.

## File contracts

`projections.json` is a faithful provider export: identity fields, eight scoring
schemes, four horizons, displayed and derived ranks, ADP, and analysis text. The printed
ranks in the saved HTML are stale; consumers use the gap-free ranks derived from 3D value.
The saved page is the provider's dynasty export; this league consumes only its one-season
projection, which is an ordinary season projection.

`pool.json` is the narrow draft input: every QB/RB/WR/TE player both sources know
(~370 — DraftSharks' pool intersected with Sleeper's top-500 projection list), with 11
fields per player. DraftSharks supplies identity and ADP; Sleeper supplies the value
column. The ranker uses projected points, not DraftSharks' provider-scaled 3D value.

- `points`: one-season points in this league's own Sleeper scoring settings, joined
  from `data/sleeper_projections.json` on `sleeper_id` by stage 4. Stage 2 first fills
  the column from the provider's `half_ppr`, but stage 4 replaces it — the two point
  scales never mix, so players without a Sleeper projection are dropped there
- `adp`: overall 1QB ADP decoded from the provider's 12-team round.pick notation
- `sleeper_id`: the join key used by stage 4, the live draft, and the investigator
- `rank`: descending `points`; ties keep the previous pool order

## Sleeper matching

There is no shared provider id, so `match_sleeper.py` uses three conservative name
tiers: full normalized name; name without suffix; then last name plus team and nearby
age. Position must always agree, ambiguity is left unmatched, and duplicate Sleeper ids
are fatal. Re-running is idempotent because ids are dropped and re-derived.
