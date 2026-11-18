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
