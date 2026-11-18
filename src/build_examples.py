"""Local-only examples for the side-by-side view (contains review TEXT -> gitignored + .vercelignore'd).
Items: every review mapped by BOTH methods (Jev pilot 500 + Jev new 1000) + curated deterministic-only cases.
Usage: build_examples.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id

O = ROOT / "outputs"
vec = pd.read_parquet(O / "wset_vectors_all.parquet")
nm = pd.read_parquet(O / "reviews_norm.parquet").set_index("id")
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
raw["id"] = raw.description.map(review_id)
raw = raw.drop_duplicates("id").set_index("id")


def compact(ans):
    out = {}
    for k, a in ans.items():
        t = a["type"]
        out[k] = round(a["noul"], 2) if t == "noul" else [round(a["score"], 2), round(a["confidence"], 2)] if t == "score" else [a["choice"], round(a["confidence"], 2)]
    return out


jev, batch = {}, {}
for f, b in (("jev_pilot.jsonl", "pilot-500"), ("jev_1000.jsonl", "new-1000")):
    p = O / f
    if p.exists():
        for l in open(p):
            j = json.loads(l); jev[j["id"]] = compact(j["answers"]); batch[j["id"]] = b
d = vec.set_index("id").join(nm[["country", "province", "region_1", "variety_norm", "colour", "points", "price", "vintage"]]).join(raw)
ev = d.evidence.map(json.loads)
scales = [c for c in d if c.startswith("s_") and c != "s_texture"]
d["n_scales"] = d[scales].notna().sum(axis=1)
cn = [c for c in d if c.startswith("cn_")]; cp = [c for c in d if c.startswith("cp_")]
cases = {
    "negation (e.g. 'no oak')": d.n_neg > 0,
    "adjective bound across comma/and": ev.map(lambda h: any(x[5] == "bound-x" for x in h)),
    "adjective + noun (soft tannins)": ev.map(lambda h: any(x[5] == "bound" for x in h)),
    "texture cue": d.s_texture.notna(),
    "drink window / year range": d.r_year_from.notna(),
    "too old / past best": d.readiness == "too_old",
    "needs ageing (can_age)": d.r_can_age == 1,
    "drink now": d.r_drink_now == 1,
    "rich: 4+ structure fields": d.n_scales >= 4,
    "sparse: aroma only": (d.n_scales == 0) & (d.n_hits >= 1),
    "low token coverage (<10%)": d.coverage < 0.10,
    "sweet / dessert (level 5-6)": d.s_sweetness >= 5,
    "tertiary notes (unambiguous)": d.t_tertiary == 1,
    "nose-scoped aroma": d[cn].max(axis=1) == 1,
    "palate-scoped flavour": d[cp].max(axis=1) == 1,
    "colour / appearance stated": d.cat_colour.notna() | d.cat_appearance_intensity.notna(),
    "fault / unclean": d.cat_condition.notna(),
    "mousse (sparkling)": d.s_mousse.notna(),
    "ambiguous word, colour-resolved": ev.map(lambda h: any(x[0] == "aroma" and x[7] > 1 for x in h)),
    "readiness: too young (potential)": d.readiness == "not_ready_potential",
}
rng = np.random.RandomState(11)
chosen = {}
for i in jev:
    chosen[i] = ["with Jev (side by side)"]
for name, mask in cases.items():
    ids = d.index[mask.values]
    for i in rng.choice(ids, min(22, len(ids)), replace=False):
        chosen.setdefault(i, []).append(name)
    for i in ids[np.isin(ids, list(jev))]:
        chosen[i].append(name) if name not in chosen[i] else None
for i in rng.choice(d.index, 30, replace=False):
    chosen.setdefault(i, []).append("random")
out = []
for i, cs in sorted(chosen.items()):
    r = d.loc[i]
    out.append(dict(text=r.description, variety=r.variety_norm, country=r.country, region=None if pd.isna(r.region_1) else r.region_1,
                    points=None if pd.isna(r.points) else int(r.points), price=None if pd.isna(r.price) else float(r.price),
                    vintage=None if pd.isna(r.vintage) else int(r.vintage), colour=r.colour, cases=cs, jev=jev.get(i), batch=batch.get(i)))
json.dump(out, open(ROOT / "dashboard/data/examples.json", "w"), separators=(",", ":"))
print("examples", len(out), "| with Jev:", sum(1 for o in out if o["jev"]), "| size KB", round((ROOT / "dashboard/data/examples.json").stat().st_size / 1024))
