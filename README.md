# wset_pipeline

Deterministic pipeline turning free-text wine reviews into partial WSET-style (SAT) vectors,
then comparing them with BERT/sentence embeddings, geography, vintage and growing-season weather.

- `lexicon/` hand-written WSET-style lexicon (aromas, structure cues, overrides). Heuristic; tuned by audit.
- `src/` code. `outputs/`, `cache/`, data files are git-ignored (source reviews are scraped Wine Enthusiast text, reportedly CC BY-NC-SA).
- Quality band assumption: points <=82 acceptable, 83-86 good, 87-90 very good, 91+ outstanding.
- Determinism: pinned model revisions, fixed seeds, CPU. Rerun hashes must match.
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

Source data lives outside the repo at `../datasets/` (see parent folder).
