# Review of the pipeline against the WSET Level 3 Systematic Approach to Tasting Wine (3W v2.0, June 2016)

Reference: WSET L3 SAT grid + L3 Wine-Lexicon (copyright WSET; not stored in this repo - fetch from wsetglobal.com).
Scope: lexicon (aromas.yaml, structure.yaml, overrides.yaml), matcher, vectors, dashboard labels, and the Jev prompt (src/jev_pilot.py).

## 1. What matches the SAT
- Five-point scales with WSET level names for acidity, tannin, body, flavour intensity, finish (low..high, light..full, short..long, light..pronounced).
- Alcohol: 3 levels (low/medium/high). Development: youthful / developing / fully developed / tired.
- Most primary clusters (floral, green fruit, citrus, stone, tropical, red, black, herbaceous, herbal) and their core descriptors.

## 2. Divergences found (ordered by impact)
1. **Aroma taxonomy is not the SAT's.** SAT has 20 clusters in 3 tiers: primary (12: incl. dried/cooked fruit, pungent spice, other), secondary (yeast, MLF, oak), tertiary (deliberate oxidation, fruit development white/red, bottle age white/red).
   Mine has 14 clusters. Specific errors:
   - `other_primary` is a grab-bag holding notes the SAT files as tertiary/secondary: nuts/almond (deliberate oxidation), leather/game, earth/mushroom (bottle age red), petrol, honey (bottle age white), caramel/toffee, coffee/chocolate, smoke (oak). Clearly-tertiary ones alone appear in ~24% of reviews; including caramel, coffee, smoke and candy ~47%.
   - Yeast and MLF are merged (`lees_malolactic`); the five tertiary clusters are merged into `oxidative_aged`, which also contains non-aroma words (mature, old, aged).
   - `dried_cooked_fruit` is tertiary in my file; the SAT lists it under PRIMARY (jamminess, stewed/baked, prune, raisin).
   - Pungent spice should be its own primary cluster (pepper, liquorice); clove/nutmeg are oak, cinnamon/ginger are bottle age (white). Liquorice/anise sits in my `herbal`.
   - White vs red matters for fruit development and bottle age; I have wine colour per review (variety) but do not use it.
2. **Sweetness scale has 5 cue levels but the SAT has 6** (dry, off-dry, medium-dry, medium-sweet, sweet, luscious). Bug: level labels are looked up as levels[k-1], so my level 3 displays as "medium-dry", 4 as "medium-sweet", 5 as "sweet"; "luscious" is unreachable. Affects the matcher output labels, dashboard Mappings/Examples tabs, and the Jev prompt (5 merged levels).
3. **Nose and palate are merged.** SAT records aroma characteristics (nose) and flavour characteristics (palate) separately, and nose intensity vs flavour intensity separately. 39% of reviews contain a nose word, 69% a palate word, 33% both, so attribution is feasible.
4. **Fields in the grid not modelled at all:**
   - Appearance (clarity, intensity, colour, other): colour words appear in ~6.5% of reviews, so it is sparse, not absent (I previously said "never").
   - Nose condition (clean/unclean): fault words ~0.8%.
   - Mousse (delicate/creamy/aggressive): 22% of sparkling reviews mention bubbles/mousse.
5. **Quality:** SAT has 6 levels (faulty, poor, acceptable, good, very good, outstanding), judged on balance/length/intensity/complexity. I derive 4 bands from critic points (data min 80, so faulty/poor can never occur). This is a proxy and should be labelled so. The Jev prompt has no quality question.
6. **Readiness:** SAT has 4-5 options (can drink now but has potential / drink now, not for ageing / too young / too old). I have two flags (drink_now, can_age) plus year range; "too young" and "too old" are not separate outcomes.
7. **Jev prompt (jev_pilot.py)** inherits all of the above: 14 non-SAT clusters with my own glosses, 5-level sweetness, no appearance/condition/mousse/quality, 2 readiness yes/no questions, no nose-vs-palate split. The pilot results (kappa, agreement) were measured on that non-SAT scheme and must be re-run after realignment (cost ~ $0.05 per 500 reviews).
8. **Documentation:** README/diagram say "WSET-style"; the cue-to-level mapping (e.g. "crisp" -> acidity 4) is a project heuristic, not part of the SAT.

## 3. Proposed realignment (not yet done)
1. Rebuild aromas.yaml on the SAT's 20 clusters; multi-assign ambiguous words (toast, chocolate, nutmeg, cheese) and disambiguate white/red tertiary using wine colour.
2. Fix sweetness to 6 levels and the label bug; add mousse scale; add appearance (colour group + depth) and condition fields as sparse optional fields.
3. Split nose vs palate with clause-scoped cues; separate nose intensity.
4. Readiness as one ordinal outcome; label quality as a points-derived proxy.
5. Rewrite the Jev question set from the SAT (Choice for colour/readiness/quality, Score for 5/6-level scales, Noul per SAT cluster), rerun the 500-review pilot, rebuild vectors/dashboard.

---
## 4. Resolution (realignment completed 2026-11-18)
All eight divergences above were addressed:
1. Aroma lexicon rebuilt on the SAT's 20 clusters / 3 tiers (130 descriptors). Ambiguous words are kept under every SAT cluster that lists them;
   white/red tertiary clusters are resolved with wine colour; tier flags use unambiguous evidence only (tertiary reads 35% of reviews, still inflated
   by words like "earthy"/"leather" that occur in young wines).
2. Sweetness now has the SAT's 6 levels and the label bug is gone; mousse, nose intensity, colour, clarity, colour depth and condition added (all sparse).
3. Nose vs palate scope assigned from trigger words in the same sentence (39% of reviews have nose-scoped, 60% palate-scoped aroma). Heuristic, unvalidated.
4. Readiness is one outcome (too_old / drink_now_potential / drink_now / not_ready_potential); quality is labelled a points proxy.
5. Jev questions rewritten from the SAT (68 questions, ~4,000 tokens/review, ~$28.6 for the full corpus); pilot rerun on 497 reviews ($0.084).
6. Browser matcher ported and parity-tested (6,000/6,000 identical incl. colour, scope, categorical fields).

### Jev vs deterministic (497 reviews; agreement, not accuracy)
Aroma clusters: mean kappa 0.61 (primary 0.67, secondary 0.58, tertiary 0.50). Weak: the SAT "other" cluster (Jev uncertain 97% - the question wording is
ambiguous), herbaceous, herbal, pungent spice. Scope agreement is low (nose kappa 0.34, palate 0.42). Structure levels agree well where both state them
(sweetness rho 0.88 on 6 levels; development 0.75; body 0.65) while Jev states finish/development/flavour intensity in 2-3x more reviews than the keyword rules.
Jev quality score vs critic points: Spearman 0.74.

### Known remaining limits
- Nothing has been validated by a WSET-qualified taster; Jev is not ground truth.
- Embedding-mapped terms (68) are ~80% precise on held-out lexicon words and visibly lower by eye (e.g. meld->melon); they are tagged `embed` and excluded from analyses.
- Appearance/condition/mousse are sparse in reviews (colour 3.8%, fault 0.4%, mousse 0.9% of all reviews).
- Weather analysis: reanalysis at 0.5 x 0.625 degree cells, associational; n = place x grape x vintage units.

---
## 5. Moved to the Level 4 card (Issue 1.2, Aug 2023) - 2026-11-18
Reason: the user supplied the newer L4 card. Differences vs L3 (2016) and what changed:
- Clusters: 18 (12 primary incl. new "fruit ripeness" and renamed "spice"; 3 secondary; tertiary grouped as red wine / white wine / deliberately oxidised).
  Descriptor lists changed (e.g. oak lost nutmeg/butterscotch/toast; yeast gained yogurt, acetaldehyde, flor; spice gained cinnamon).
- New card lines: tannin nature (ripe, soft, smooth, unripe, green, coarse, stalky, chalky, fine-grained); other observations (texture: oily/creamy/austere/luscious; pétillance; appearance: legs/tears, deposit, bubbles); rosé colours pink / pink-orange / orange.
- Dropped from the card: development, mousse, 4-way readiness (L3-only; still computed in the vectors). Conclusions are now quality + bottle ageing (suitable / not suitable); "drink now" without ageing cues is read as not suitable (implied).
- Jev prompt: 238 typed questions - every lexicon descriptor asked separately for NOSE and PALATE (the earlier cluster-level prompt could not say "cloves").
  Redo cost $0.58 for 1,500 reviews (full corpus ~$65). Comparison: descriptor level - both 4,431 mentions, Jev only 2,588, deterministic only 1,033.
  Jev answers "ripe fruit" for 83% of reviews (question probably too loose); finish/flavour intensity/tannin nature/texture are stated far more often by Jev.

## 6. One-of fields and assumption A1 (2026-11-19)

The L4 card states: *"For lines where the entries are separated by a hyphen you must select one and only one of these options."*
Hyphen lines: clarity, intensity, colour, nose condition, nose intensity, sweetness, acidity, tannin level, alcohol, body,
flavour intensity, finish, quality, bottle ageing. All were already single-valued in the Jev prompt except **nose condition**,
which was a yes/no question and is now a single choice (clean / unclean / not stated).

**A1 (our assumption, not WSET's):** the Wine-Lexicon prints *fruit ripeness* with commas (a prompt list), but unripe, ripe, dried
and cooked are successive states of the same fruit, so we treat it as one-of per note.
- Deterministic: fruit_ripeness hits plus tertiary dried/cooked fruit (same words); "ripe fruit" is the generic baseline and
  loses to a specific state; two or more specific states = `mixed`, no value set (0.05% of reviews). Columns `fruit_ripeness`, `fruit_ripeness_mixed`.
- Jev: 8 yes/no questions replaced by one 5-way choice (`ripeness`), with explicit "not unripe merely because tart/green apple" and
  "not cooked merely because candied/burnt" exclusions.
- Result on 1,500 reviews: both name a single state in 406 reviews, exact agreement 0.93, kappa 0.87 (before: Jev said unripe for 95 reviews the matcher did not; now 9).
- Still e.g.-lists (several allowed): aroma descriptors, tannin nature, texture, other appearance observations, pétillance.

## 7. Terroir analysis (`src/terroir.py` -> `dashboard/data/terroir.json`, Terroir tab; replaces the Weather tab)

Why: on a single review, grape, place, vintage and weather explain very little of the WSET profile (adjusted R2: grape 4.1%, place 4.3%,
producer 4.6%, place x grape 8.5%, vintage 0.3%, within-place season weather 0.05%; text embeddings: place x grape 13.5%), so a raw
correlation grid on reviews is unreadable. Method: units = place x grape x vintage; anomalies vs each grid cell's own 2000-2017 climate
(NASA POWER daily data, all 18 seasons, not only vintages with reviews); within-group demeaning; cluster bootstrap over place x grape groups.

Findings (associational): +1 SD warmer season -> +0.21 critic points (90% CI 0.17..0.25; Pinot Noir +0.37) and -0.05 on the 1-5 acidity cue;
wetter season -> fewer points. A warm vs cool vintage cannot be recognised from the note profile (leave-vintage-out PLS R2 = -0.05).
Regions <25 km apart taste ~56% as different as an average pair (net of grape); rank correlation of taste vs km is 0.31 within 1,600 km (Mantel p = 0.007).

Panels: (1) variance explained, (2) climate response curves and a per-attribute table by grape, (3) vintage explorer per place x grape
(100 combinations with >= 6 vintages, >= 150 notes), (4) taste map: place profiles (>= 40 notes, shrunk) projected with PCA/UMAP from the WSET vector
(raw or net of grape) or text embeddings (20k sample), coloured by position in the plot, linked by box-selection to the world map; plus distance decay.
Caveats: critic vocabulary, region-level coordinates for 75% of reviews, 0.5x0.625 degree weather grid, many attributes tested.
