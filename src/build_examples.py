"""Local-only review examples for the dashboard (contains review TEXT -> gitignored + .vercelignore'd).
Deterministic stratified selection of cases that show how the matcher behaves. Usage: build_examples.py [N]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id

n = sys.argv[1] if len(sys.argv) > 1 else "20000"
O = ROOT / "outputs"
vec = pd.read_parquet(O / f"wset_vectors_{n}.parquet")
nm = pd.read_parquet(O / "reviews_norm.parquet").set_index("id")
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
raw["id"] = raw.description.map(review_id)
raw = raw.drop_duplicates("id").set_index("id")
d = vec.set_index("id").join(nm[["country", "province", "region_1", "variety_norm", "colour", "points", "price", "vintage"]]).join(raw)
ev = d.evidence.map(json.loads)
scales = [c for c in d if c.startswith("s_") and c != "s_texture"]
d["n_scales"] = d[scales].notna().sum(axis=1)
cases = {
    "negation (e.g. 'no oak')": d.n_neg > 0,
    "adjective bound across comma/and": ev.map(lambda h: any(x[5] == "bound-x" for x in h)),
    "adjective + noun (soft tannins)": ev.map(lambda h: any(x[5] == "bound" for x in h)),
    "texture cue": d.s_texture.notna(),
    "drink window / year range": d.r_year_from.notna(),
    "needs ageing (can_age)": d.r_can_age == 1,
    "drink now": d.r_drink_now == 1,
    "rich: 4+ structure fields": d.n_scales >= 4,
    "sparse: aroma only": (d.n_scales == 0) & (d.n_hits >= 1),
    "low token coverage (<10%)": d.coverage < 0.10,
    "sweet / dessert": d.s_sweetness >= 4,
    "tertiary / aged notes": d.c_oxidative_aged == 1,
}
rng = np.random.RandomState(11)
chosen, tag = {}, {}
for name, mask in cases.items():
    ids = d.index[mask.values]
    pick = rng.choice(ids, min(22, len(ids)), replace=False) if len(ids) else []
    for i in pick:
        chosen.setdefault(i, []).append(name)
for i in rng.choice(d.index, 40, replace=False):
    chosen.setdefault(i, []).append("random")
out = []
for i, cs in sorted(chosen.items()):
    r = d.loc[i]
    out.append(dict(text=r.description, variety=r.variety_norm, country=r.country, region=None if pd.isna(r.region_1) else r.region_1,
                    points=None if pd.isna(r.points) else int(r.points), price=None if pd.isna(r.price) else float(r.price),
                    vintage=None if pd.isna(r.vintage) else int(r.vintage), colour=r.colour, quality=r.quality_band, cases=cs))
json.dump(out, open(ROOT / "dashboard/data/examples.json", "w"), separators=(",", ":"))
print("examples", len(out), {k: int(m.sum()) for k, m in cases.items()})
