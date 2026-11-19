# Plan v2 for review: guess model + blind-tasting rounds, causal DAG + data plan, clearer map, packaging

Status: PLAN. Already done in this round: the draft DAG (`lexicon/dag.yaml`, checked for cycles and adjustment sets), the Docker/environment files, and a feasibility baseline. The guess model, the blind-tasting machinery and the dashboard panels are NOT built yet.

---

## A. The guess model (deterministic, statistical) and blind-tasting rounds

### A1. Idea
From the description alone, predict **grape, place, producer, vintage, price** with deterministic statistical models. No LLM and no judges in the loop: inputs are the WSET features from our matcher and/or the text embedding. Same text in, same guess out, so every round is reproducible from (code commit, data hash, seed).

### A2. Features (three families, switchable, so each can be ablated)
1. **SAT features**: 113 descriptor flags, cluster/tier flags, 8 scales (+ missing flags), tannin nature, texture, colour cues, readiness/year mentions (e.g. "drink 2020–2030"), coverage.
2. **Embedding**: MiniLM 384-d (BERT optional). Currently computed for 20k notes; all 169k needed (one-off, ~1 h CPU in the `embed` image, cached).
3. **Names in text** (gazetteer): literal mentions of grape, region, winery and 4-digit years. The variety name appears in 20% of notes, country 20%, winery 6%, region 6%, a year 16%. Reported as an ablation: "names on" vs "names masked" (a masked copy of the text goes through the same pipeline).

### A3. Models, per target
| Target | Model | Output |
|---|---|---|
| Grape (47 groups, then 738 varieties) | multinomial logistic regression on SAT + embedding, blended with gradient boosting on SAT; temperature-scaled; hierarchical (group → variety) | top-5 with calibrated probabilities |
| Place | hierarchical country → province → region (conditional multinomial LR), with the **grape probabilities fed in as features** (stacking, cross-fitted so no leakage) | top-5 regions, posterior over places, expected map point and an uncertainty radius (km holding 50/80% of the probability) |
| Producer (19k wineries) | retrieval + prior: P(winery) ∝ P(winery | predicted place, grape) × text-similarity to that winery's notes (embedding/SAT kNN); wineries with ≥5 notes (87% of reviews) | top-5 + "other" mass |
| Vintage | gradient boosting with quantile loss (10/50/90) + explicit year-mention rule features; split-conformal calibration | point + 80% interval |
| Price | same recipe on log(price) | point + 80% interval |
| (bonus) Critic points | same recipe | point + interval |

Fixed seeds; stacking is cross-fitted (5 folds); calibration (temperature, conformal) is fitted on out-of-fold predictions, never on test data.

### A4. Feasibility check already run (SAT features only, 20k notes, simple models, 20% test)
| Target | Model | Baseline (guess the most common / median) |
|---|---|---|
| Grape group (47) | top-1 **35.5%**, top-3 **58.9%** | 10.2% |
| Country | top-1 **56.5%**, top-3 **83.3%** | 41.4% |
| Province | top-1 **36.9%**, top-3 **57.1%** | 28.5% |
| Vintage | MAE 2.43 yr | 2.79 yr (skill 0.13) |
| Price | MAE 0.446 in log price (about ±56%) | 0.538 (skill 0.17) |
So the SAT vector alone carries real but modest signal. The full version should do clearly better with all 169k notes, embeddings, stacking and name features; producer will stay a long shot (we'll report skill against the prior, not hope for a high hit rate). These numbers are a starting point, not the model.

### A5. Blind-tasting rounds (500 at a time)
- A round: sample 500 notes never tested before (seed 1000+r, from the corpus minus the ledger), **seal the answers**, train the models on everything else, predict the 500, reveal, score, record, and append the 500 ids to the ledger.
- The sealed 500 are excluded from training, calibration and model selection for that round (hard assertion in code: no sealed id may appear in any training matrix).
- **"Add them back"**: after the reveal the 500 stay in the corpus, flagged `blind_tested`, are excluded from every future draw, and join the training pool for the next round's model. That is how the model grows round by round.
- Development discipline: model design and tuning use cross-validation on the pool only; a blind round is the unbiased audit. A **frozen v1 model** is also scored on every round so improvements are visible and not hindsight. Model changes happen only between rounds and are logged in a changelog.
- Statistics: with 500 notes the 95% interval on a hit rate is about ±2–4 points; differences smaller than that between rounds are noise. Rounds are also pooled.
- Conditions reported each round: names on vs masked; notes where the name was in the text vs not.

### A6. Scoring
Grape: top-1/top-3, colour family right. Place: country/province/region exact, hierarchical credit, km error from the expected point, whether the true place is inside the 50/80% region. Producer: exact (normalised), top-3, skill vs prior, scored on wineries with ≥5 notes and overall. Vintage: MAE, within ±1, interval coverage. Price: MAE of log price, within ±25%, Spearman, interval coverage. Everything also as **skill vs a baseline** (most frequent grape/country/winery, median vintage, median price by grape × country) and **calibration** (Brier, reliability plot, interval coverage).

### A7. Recording
```
experiments/blind_tasting/
  ledger.csv                id, round, tested_at
  CHANGELOG.md              model versions and why
  round_001/
    manifest.json           seed, sampler version, git commit, requirements lock hash, corpus and embedding hashes, hyperparameters, timings
    items.parquet           ids, masked text flag, leak flags
    key.parquet             true labels (sealed until scoring)
    predictions.parquet     every guess with probabilities (model vN and frozen v1)
    scores.json  report.md
```
Scripts: `blind.py sample | train | predict | score | close`.

### A8. Dashboard: tab "Guess" (replaces the earlier "Blind tasting" idea)
1. **Describe a wine (live)**: a text box. It runs the matcher in the browser, applies the exported model weights and shows: grape top-5 bars, place top-5 plus a map with probability bubbles and an uncertainty circle, producer top-5, vintage and price intervals, and **why** (which descriptors pushed each guess, highlighted in the text). Works fully offline on a static site (see E).
2. **Example guesses**: cards showing a note, the model's guess and the revealed truth, side by side, with what it got right and wrong (from revealed rounds; **local only**, since it contains review text; the public build uses a small set of notes we write ourselves).
3. **Rounds**: results per round and pooled, trend across rounds, calibration plots, grape confusion matrix, map of place errors.

### A9. Effort and order
Embeddings for all 169k → feature store → grape and place models → vintage/price → producer → `blind.py` and round 1 → dashboard panels. First measurable milestone: grape + place with a round-1 dry run on 100 notes.

---

## B. Causal DAG (draft ready for review) and data plan

- **Files**: `lexicon/dag.yaml` (23 variables, 50 edges, with a reason and a strength on each edge) and the diagram shown with this plan.
- **Checked by code** (networkx): acyclic; for the effect of **season weather on the note and on points**, adjusting for {Year, Place} is already sufficient (Grape optional, Producer optional); adjusting for price, quality, ripeness, viticulture or winemaking would be **wrong** (they are descendants of the weather); the declared independences are testable. One entry in my first draft was wrong (a price/weather independence the graph does not imply) and is fixed.
- **Estimands**: E1 total effect of weather on the note/points (main); E2 controlled grape/place contrasts; E3 mediation through ripeness, only where Brix/harvest data exist.
- **Design for E1**: place × grape and year fixed effects (variant: same winery across vintages), clustering by country-year, negative controls (next-year and post-harvest weather), sensitivity (E-value), heterogeneity by grape and site climate. A new "Causal" tab with the interactive DAG, open/closed paths, and a results table.

### B1. New variables to add, and how to get them (priority order)
| # | Variable | Source | Effort | Unlocks |
|---|---|---|---|---|
| 1 | Region polygons (appellations) | US AVA GeoJSON (UC Davis/TTB); France INAO AOC areas (data.gouv.fr, open licence); OpenStreetMap polygons via Nominatim; national GI shapefiles for Italy, Spain, Portugal, Australia, NZ, Chile | medium, ~1 week for the top 8 countries (>90% of reviews) | better place precision for everything below; fixes the 75% region-centroid limitation |
| 2 | Elevation, slope, aspect | Copernicus GLO-30 / SRTM tiles (free), summarised over each polygon | low | site climate, new DAG node |
| 3 | Soil (clay, sand, pH, organic carbon, coarse fragments) | SoilGrids 2.0 (ISRIC, 250 m, free) sampled over each polygon | medium | soil node; coarse, treated as a regional descriptor |
| 4 | Better climate | WorldClim 2.1 / CHELSA 1 km normals; ERA5-Land daily 0.1° (Copernicus CDS, free account); phenology-window metrics (flowering heat, veraison temperature, harvest rain) | medium | sharper weather treatment than the current 0.5°×0.625° grid |
| 5 | Ripeness and harvest | USDA NASS California Grape Crush Report (district × variety × year: Brix, tons, price); Wine Australia vintage report; NZ Winegrowers; Oregon/Washington grower reports; France Agreste/CIVC/BIVB; Chile SAG | medium-high; California first (largest share of reviews) | E3 mediation; also validates the weather index (does heat predict Brix?) |
| 6 | Grape traits (ripening epoch, colour, parentage) | VIVC, Wikidata, INRAE Pl@ntGrape | low (~50 grape groups) | moderator of weather effects |
| 7 | Producer traits | keyword flags from descriptions and `designation` (already in the raw data, 70% filled: single-vineyard/reserve cues); Wikidata; organic/biodynamic registers | medium, partial coverage | producer and cuvee nodes |
| 8 | Taster | `taster_name` already in the raw data (56% filled, 19 tasters) | low | confounder: tasters cover regions and have a style |
| 9 | **Review date** | not in any file we hold; winemag.com pages show it but scraping is subject to the site's terms | blocked: needs your call | bottle age at tasting, a key confounder of tertiary notes |
| 10 | Published vintage charts | public rating tables, entered as a small hand-checked table | low | external validation only (copyright: no scraping) |

---

## C. Making map ⇄ embedding legible (unchanged from v1, my recommendation: do first)
Colour key and axis end-labels on the plot; ~25 labelled landmark regions and country hulls; auto-named taste clusters; "taste twins" (click a region, lines on the map to its 8 nearest taste neighbours with km); biplot arrows of the main descriptors; colour by one attribute with a colourbar; hover-link plus preset comparisons; shape by wine colour.

---

## D. Packaging: own environment, Docker, Vercel

**Done now**: `Dockerfile` (Python 3.11, pinned requirements, three targets: `core`, `embed`, `serve`), `docker-compose.yml` (data, outputs, cache and experiments are mounted, never baked in), `.dockerignore`, `requirements/{core,embed,serve,causal}.txt`, and `scripts/smoke.py` (imports + matcher check + parity against the reference results). The corpus and review text stay outside the image.

**Vercel**: the dashboard is already static. The "Guess" tab runs the **lite** model in the browser: the JS matcher turns the text into SAT features and the models are exported as JSON weights (a few MB), with Python ↔ JS parity tests like the matcher has. The **full** model (embedding) runs in the `serve` container (FastAPI); the dashboard calls it if its URL is configured and otherwise falls back to lite. Hosting the container is separate (Vercel cannot hold PyTorch). A stretch option is MiniLM in the browser (transformers.js, ~25 MB) so even the full model is static.

---

## E. Decisions I still need
1. **Embeddings for all 169k notes** (≈1 h CPU, one-off, in Docker): OK?
2. **Public build** shows no review text: the example panel uses ~20 notes we write ourselves. OK? (Your local build still shows real revealed notes.)
3. **Lite producer guess** in the browser = nearest style-centroid among wineries with ≥20 notes (~3,500), with a prior; the full model does proper retrieval. Acceptable?
4. **Place resolution** to predict: region (1,330 classes) with province and country rolled up. OK?
5. **Review date**: do you want to decide on scraping terms, or approximate bottle age without it?
6. **DAG**: edit/confirm the variables and edges; confirm E1 as the first estimand; which data items in B1 do you want first?
7. **Installs** for the causal module (statsmodels, pyfixest): OK to add to `requirements/causal.txt` and the image?
