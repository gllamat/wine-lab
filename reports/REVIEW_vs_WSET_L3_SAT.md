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
