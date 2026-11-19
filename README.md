# wset_pipeline

Deterministic pipeline turning free-text wine reviews into partial WSET SAT vectors. The card/lexicon follow the WSET Level 4 Systematic
Approach to Tasting Wine, Issue 1.2 (Aug 2023); the earlier Level 3 (3W v2.0, 2016) version is archived in lexicon/archive/. (c) WSET; the SAT documents are not stored here, compared with Jev (TypeSafe System One),
BERT/sentence embeddings, geography, vintage and growing-season weather. See `reports/REVIEW_vs_WSET_L3_SAT.md`.

- `lexicon/` aromas.yaml (18 SAT clusters in 3 tiers, 99 SAT descriptors + non-SAT cluster-level extras), structure.yaml (SAT scales incl. 6-level sweetness, mousse, nose intensity;
  categorical colour/clarity/condition; readiness), overrides.yaml (audit corrections). Cue words and cue-to-level mappings are project heuristics.
- `src/wsetkit.py` matcher: longest-match lookup, bound adjectives, negation, nose/palate scope, wine-colour resolution of white/red tertiary clusters.
  `dashboard/matcher.js` is a parity-tested browser port (`node dashboard/parity.js`).
- `src/jev_pilot.py` + `src/compare_jev.py`: 238 SAT-derived typed questions per review (every lexicon descriptor asked separately for nose and palate); key from `.env` (gitignored); `src/key_status.py` monitors key health.
- Quality is a PROXY from critic points (SAT has 6 levels; corpus minimum is 80 pts).
- Data/outputs are git-ignored; source reviews are scraped Wine Enthusiast text (reportedly CC BY-NC-SA). The deployed dashboard carries aggregates only.
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Weather: NASA POWER. Geocoding: OpenStreetMap Nominatim (validated in `src/places.py`).

## Terroir tab
`python src/terroir.py` (about 3 min; needs the vectors, weather cache and embeddings) writes `dashboard/data/terroir.json` (aggregates only, safe to deploy): variance explained, climate response curves, vintage explorer, taste map with linked selection, and geography-vs-taste distance decay. See reports/REVIEW_vs_WSET_L3_SAT.md section 7.

## Guess model, blind rounds, causal lab (added 2026-11-19)

**Guess (deterministic, statistical).** From the description alone, `src/guess_model.py` guesses grape, place, producer, vintage, price and critic points.
Features: the SAT vector computed *without* the wine-colour label (`run_matcher.py all --nocolour`) plus note length and year hints (`src/guess_features.py`, 170 features; hits-path and vector-path
featurisers are tested identical). Models: softmax regression (variety, country, place, stacked on grape probabilities), ridge with residual-quantile intervals (vintage, log price, points),
producer = P(winery | place, grape) tables x a style kernel. Tiers: `lite` (SAT only; exported to JSON and run in the browser: `src/export_guess.py`, `dashboard/guess.js`, parity test `node dashboard/guess_parity.js`)
and `full` (SAT + MiniLM embedding, `python src/embed.py minilm all` first). `python src/guess_model.py dev [full]` = development split (not a blind round).

**Blind rounds** (`src/blind.py run [--n 500] [--dry]`): sample 500 notes never tested (seed 1000+round, minus `experiments/blind_tasting/ledger.csv`), seal the key, train on everything else
(earlier revealed rounds included), predict, freeze predictions (sha256 in the manifest), reveal, score, record, append the ids to the ledger. Both tiers are scored on the same notes.
Results: `experiments/blind_tasting/round_NNN/` and `dashboard/data/blind_rounds.json` (metrics only; `blind_examples.json` has review text and stays local).

**Causal lab**: `lexicon/dag.yaml` (base graph) + `lexicon/dags/*.yaml` (variants), `src/causal.py` (d-separation, back-door/front-door, adjustment sets, the three do-calculus rules; JS port `dashboard/causal.js`,
parity `node dashboard/causal_parity.js`), `src/causal_estimate.py` (adjustment gallery on the real data, placebo tests; needs `requirements/causal.txt`). Dashboard tab "Causal".

**Environment**: `Dockerfile` targets `core`, `causal`, `embed`, `serve`; `docker-compose.yml`; `Makefile`; pinned `requirements/*.txt`; `scripts/smoke.py`. Data is mounted, never baked in.
