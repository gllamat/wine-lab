"""Compare Jev pilot answers with the deterministic matcher on the same reviews (no review text used)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT

J = [json.loads(l) for l in open(ROOT / "outputs/jev_pilot.jsonl")]
vec = pd.read_parquet(ROOT / "outputs/wset_vectors_20000.parquet").set_index("id")
ids = [j["id"] for j in J]; V = vec.loc[ids]
A = lambda k: np.array([j["answers"][k].get("noul", np.nan) for j in J])
S = lambda k: np.array([j["answers"][k]["score"] for j in J]); C = lambda k: np.array([j["answers"][k]["confidence"] for j in J])
print(f"reviews {len(J)} | models {set(j['model'] for j in J)}")
rows = []
for c in [c[2:] for c in vec if c.startswith("c_")]:
    k = f"aroma__{c}"
    if k not in J[0]["answers"]: continue
    p, m = A(k), V["c_" + c].values
    jv = p >= 0.5
    tp = int((jv & (m == 1)).sum())
    rows.append(dict(cluster=c, jev_rate=jv.mean(), matcher_rate=m.mean(), agree=(jv == (m == 1)).mean(),
                     kappa=cohen_kappa_score(jv, m == 1), matcher_recall_vs_jev=tp / max(1, jv.sum()), matcher_precision_vs_jev=tp / max(1, (m == 1).sum()),
                     jev_uncertain=((p > .2) & (p < .8)).mean()))
r = pd.DataFrame(rows).round(2)
print("\nAROMA CLUSTERS (Jev Noul>=0.5 vs deterministic matcher)\n", r.to_string(index=False))
print(f"\nmean kappa {r.kappa.mean():.2f} | mean agreement {r.agree.mean():.2f} | matcher recall vs Jev {r.matcher_recall_vs_jev.mean():.2f} | precision {r.matcher_precision_vs_jev.mean():.2f}")
rows = []
for s in ["acidity", "tannin", "body", "sweetness", "finish", "flavour_intensity", "alcohol", "development"]:
    st = A(f"stated__{s}"); sc = S(f"level__{s}") + 1; cf = C(f"level__{s}")
    m = V["s_" + s].values; mf = ~np.isnan(m); jv = st >= 0.5
    both = mf & jv
    row = dict(scale=s, jev_stated=jv.mean(), matcher_filled=mf.mean(), both=int(both.sum()), jev_only=int((jv & ~mf).sum()), matcher_only=int((~jv & mf).sum()))
    if both.sum() >= 8:
        row["spearman"] = spearmanr(sc[both], m[both]).correlation
        row["exact_level"] = float((np.rint(sc[both]) == np.rint(m[both])).mean())
        row["within_1"] = float((np.abs(sc[both] - m[both]) <= 1).mean())
        row["jev_conf_mean"] = float(cf[both].mean())
    rows.append(row)
print("\nSTRUCTURE (stated = Jev 'explicitly described' >=0.5; level = Jev score+1 vs matcher level 1-5)\n", pd.DataFrame(rows).round(2).to_string(index=False))
for k, col in (("drink_now", "r_drink_now"), ("can_age", "r_can_age")):
    p, m = A(f"ready__{k}") >= 0.5, V[col].values == 1
    print(f"\nREADINESS {k}: Jev {p.mean():.2f} vs matcher {m.mean():.2f} | agree {(p == m).mean():.2f} | kappa {cohen_kappa_score(p, m):.2f}")
# examples of biggest disagreements (ids only; no text)
d = []
for s in ["acidity", "tannin", "body"]:
    st = A(f"stated__{s}"); m = V["s_" + s].values
    for i in np.where((st >= .9) & np.isnan(m))[0][:3]: d.append((ids[i], s, "Jev says stated, matcher found no cue"))
print("\nsample ids where Jev finds a structure statement the matcher missed:", d[:6])
r.to_csv(ROOT / "outputs/jev_vs_matcher_aroma.csv", index=False)
