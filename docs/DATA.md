# Getting the data and reproducing the results

The repository ships **code, the derived data the dashboard needs, and the experiment records**. It does **not** ship the review corpus, the embeddings of all 169k notes, the weather cache or the WSET documents. Read [DISCLAIMER.md](../DISCLAIMER.md) and [DATA_LICENSES.md](../DATA_LICENSES.md) first: the review data are CC BY-NC-SA 4.0, so use them **non-commercially**.

## 0. Just look at the dashboard (no data needed)
```bash
cd dashboard && python3 -m http.server 8765      # open http://localhost:8765
```
All derived files the public site needs are in `dashboard/data/`. Panels that show review text ("Examples", revealed blind-round notes, the note under a recommended wine) need the local-only files below and show a message without them.

## 1. Download the reviews (not redistributed here)
Create a `datasets/` folder **next to** this repository (`../datasets/` from the repo root) and place:
- `kaggle_wine_reviews/winemag-data-130k-v2.csv` and `winemag-data_first150k.csv`: Kaggle dataset "Wine Reviews" (search for *zynicide wine-reviews* on kaggle.com; you need a free Kaggle account). Licence as stated there: CC BY-NC-SA 4.0.
- optional: `hf_gronlp_winemag/{train,validation,test}.csv`, a Hugging Face mirror; it adds no unique review.

## 2. Build the corpus
```bash
python src/build_corpus.py          # -> ../datasets/wine_reviews_deduped.parquet (169,428 unique reviews)
```
This reproduces the corpus used for the published results: the same 169,428 reviews, with each column identical to the original at 99.5–100% (differences come from tie-breaking when two sources disagree on a field).

## 3. Run the pipeline (in this order)
```bash
python src/normalise.py                                  # grapes, vintages, wine colour -> outputs/reviews_norm.parquet
python src/run_matcher.py all                            # SAT vectors (rule-based reader)
python src/run_matcher.py all --nocolour                 # description-only features for the guess model
python src/geocode.py && python src/geocode_retry.py     # OpenStreetMap Nominatim, 1 request/second, cached (hours the first time)
python src/weather.py                                    # NASA POWER 2000-2017 per grid cell, cached (hours the first time)
python src/embed.py minilm all                           # MiniLM embeddings of all notes (about 45 min on a laptop CPU)
python src/export_lexicon.py                             # browser lexicon + parity fixture
python src/terroir.py && python src/place_attrs.py       # terroir analysis for the dashboard
python src/guess_model.py dev && python src/guess_model.py dev full
python src/blind.py run --n 500                          # one blind-tasting round (records go to experiments/)
python src/recsys.py build && python src/recsys.py eval  # recommender index and evaluation
python src/causal.py export && python src/causal_estimate.py
python src/build_dashboard.py 20000                      # remaining dashboard JSON files
```
Or use Docker: `make build`, then `make matcher`, `make guess-dev`, `make blind-round`, `make causal` (data folders are mounted, never baked into images).

## 4. Local-only files (never commit or deploy)
`dashboard/data/examples.json`, `recsys_text.json`, `blind_examples.json` and `key_status.json` contain review text or operational data. They are in `.gitignore` and `dashboard/.vercelignore`.

## 5. Local-only full views (optional)
The public site keeps things light: an 8,000-review sample in the vector-space plot and a 25,000-wine index in the recommender. Two scripts build **local-only** versions with every review; their output goes to `dashboard/data/local/`, which is git-ignored and excluded from deployment (`.vercelignore`):
```bash
python src/spaces_full.py          # PCA + UMAP of all 169,428 reviews, WSET and MiniLM spaces (about 5-10 minutes)
python src/recsys.py build --full  # int8 index of all wines (about 95 MB) + the notes (about 42 MB)
```
When those files exist, the dashboard adds an "ALL reviews (local copy)" option to the vector-space page and automatically searches all wines in the recommender. Without them it silently uses the public samples.
