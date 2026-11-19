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
