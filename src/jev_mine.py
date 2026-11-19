"""Mine disagreements between Jev (typed answers) and the deterministic rules for CANDIDATE lexicon additions. Nothing is changed automatically:
candidates go to outputs/jev_candidates.csv for a human (ideally a WSET-qualified taster) to accept or reject.
Method: for each descriptor, notes where Jev says yes (p>=0.5, nose or palate) and the rules found nothing vs notes where neither fired.
An n-gram is a candidate when, among notes WITHOUT a rule hit, it co-occurs with a Jev "yes" at least `minp` of the time (and >= `minn` notes).
Usage: jev_mine.py"""
import sys, json, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon, review_id, tokenize

J = []
for f in ("outputs/jev_pilot.jsonl", "outputs/jev_1000.jsonl"): J += [json.loads(l) for l in open(ROOT / f)]
ids = [j["id"] for j in J]
V = pd.read_parquet(ROOT / "outputs/wset_vectors_all.parquet").set_index("id").loc[ids]
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"]); raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id")
texts = raw.loc[ids, "description"].tolist()
STOP = set("the a an and of with in on to is it its this that has have for as by at from but are was be than more very quite some good great nice wine wines flavor flavors flavour flavours aroma aromas note notes nose palate finish drink now will can into just also show shows offers offer there their these those which while along".split())
def grams(text):
    t = [x for x in tokenize(text) if x not in ("|", "#")]
    g = set(w for w in t if w.isalpha() and len(w) > 2 and w not in STOP)
    g |= {a + " " + b for a, b in zip(t, t[1:]) if a.isalpha() and b.isalpha() and a not in STOP and b not in STOP}
    return g
G_ = [grams(x) for x in texts]
lex = Lexicon(); known = set(" ".join(k) for k in lex.entries)
rows = []; tot_jev_only = 0; covered = 0
for cl, d in lex.descriptors:
    if d.endswith("_generic") or cl == "fruit_ripeness": continue
    kn, kp = f"nose__{cl}__{d}", f"palate__{cl}__{d}"
    jv = np.array([max(j["answers"][kn]["noul"], j["answers"][kp]["noul"]) >= .5 for j in J]); rl = V[f"a_{cl}__{d}"].values == 1
    jo = jv & ~rl; tot_jev_only += int(jo.sum())
    if jo.sum() < 3: continue
    c_all, c_jo = Counter(), Counter()
    for i in np.where(~rl)[0]:
        for g in G_[i]: c_all[g] += 1
    for i in np.where(jo)[0]:
        for g in G_[i]: c_jo[g] += 1
    cands = [(g, n, c_all[g], n / c_all[g]) for g, n in c_jo.items() if n >= 3 and n / c_all[g] >= 0.6 and g not in known]
    cands.sort(key=lambda x: (-x[1], -x[3]))
    hit = set()
    for g, n, na, p in cands[:6]:
        ex = next((texts[i][:110] for i in np.where(jo)[0] if g in G_[i]), "")
        rows.append(dict(cluster=cl, descriptor=d, ngram=g, jev_only_notes=n, notes_without_rule_hit=na, precision_vs_jev=round(p, 2), example=ex))
        hit |= {i for i in np.where(jo)[0] if g in G_[i]}
    covered += len(hit)
df = pd.DataFrame(rows); df.to_csv(ROOT / "outputs/jev_candidates.csv", index=False)
print("descriptors with candidates:", df.groupby(["cluster", "descriptor"]).ngram.count().shape[0])
print(f"Jev-only descriptor mentions in 1,500 notes: {tot_jev_only} | covered by the top candidates (upper bound, overlapping counted once per descriptor): {covered} ({100*covered/max(1,tot_jev_only):.0f}%)")
print(df.sort_values("jev_only_notes", ascending=False).head(28)[["cluster", "descriptor", "ngram", "jev_only_notes", "notes_without_rule_hit", "precision_vs_jev"]].to_string(index=False))
