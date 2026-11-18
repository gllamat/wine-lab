"""Compare Jev (SAT prompt) with the deterministic matcher on the same reviews. No review text used."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon

J = [json.loads(l) for l in open(ROOT / "outputs/jev_pilot.jsonl")]
vec = pd.read_parquet(ROOT / "outputs/wset_vectors_20000.parquet").set_index("id")
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet").set_index("id")
ids = [j["id"] for j in J]; V = vec.loc[ids]; N = nm.loc[ids]
lex = Lexicon()
A = lambda k: np.array([j["answers"][k]["noul"] for j in J])
S = lambda k: np.array([j["answers"][k]["score"] for j in J])
C = lambda k: np.array([j["answers"][k]["choice"] for j in J])
CF = lambda k: np.array([j["answers"][k]["confidence"] for j in J])
print(f"reviews {len(J)} | model {set(j['model'] for j in J)}\n")


def agree_row(name, p, m):
    jv = p >= 0.5; mv = m == 1
    tp = int((jv & mv).sum())
    return dict(field=name, jev=jv.mean(), matcher=mv.mean(), agree=(jv == mv).mean(),
                kappa=cohen_kappa_score(jv, mv) if jv.any() and mv.any() else np.nan,
                matcher_recall=tp / max(1, jv.sum()), matcher_precision=tp / max(1, mv.sum()), jev_uncertain=((p > .2) & (p < .8)).mean())


rows = []
for cl, tier in lex.tier.items():
    pn, pp = A(f"nose__{cl}"), A(f"palate__{cl}"); pa = np.maximum(pn, pp)
    rows.append(dict(agree_row(f"{cl}", pa, V["c_" + cl].values), tier=tier))
r = pd.DataFrame(rows).round(2)
print("AROMA/FLAVOUR CLUSTERS (any scope): Jev max(nose,palate)>=0.5 vs matcher\n", r.to_string(index=False))
print(f"mean kappa {r.kappa.mean():.2f} | agreement {r.agree.mean():.2f} | matcher recall {r.matcher_recall.mean():.2f} | precision {r.matcher_precision.mean():.2f}")
print("by tier:\n", r.groupby("tier")[["jev", "matcher", "agree", "kappa"]].mean().round(2).to_string())
sc = []
for cl in lex.tier:
    sc.append(dict(agree_row(f"{cl} [nose]", A(f"nose__{cl}"), V["cn_" + cl].values)))
    sc.append(dict(agree_row(f"{cl} [palate]", A(f"palate__{cl}"), V["cp_" + cl].values)))
sc = pd.DataFrame(sc)
print(f"\nSCOPE: nose-only kappa {sc[sc.field.str.contains('nose')].kappa.mean():.2f} | palate-only kappa {sc[sc.field.str.contains('palate')].kappa.mean():.2f}  (matcher scope = nearest nose/palate cue word in the sentence)")
rows = []
for s in lex.scales:
    if f"level__{s}" not in J[0]["answers"] or "s_" + s not in V: continue
    st = A(f"stated__{s}"); sco = S(f"level__{s}") + 1; m = V["s_" + s].values
    mask = np.ones(len(J), bool)
    if s == "mousse": mask = (N.colour == "sparkling").values
    mf = ~np.isnan(m); jv = st >= 0.5; both = mf & jv & mask
    row = dict(scale=s, levels=len(lex.scales[s]), jev_stated=jv[mask].mean(), matcher_filled=mf[mask].mean(), both=int(both.sum()), jev_only=int((jv & ~mf & mask).sum()), matcher_only=int((~jv & mf & mask).sum()))
    if both.sum() >= 8:
        row["spearman"] = spearmanr(sco[both], m[both]).correlation
        row["exact"] = float((np.rint(sco[both]) == np.rint(m[both])).mean()); row["within1"] = float((np.abs(sco[both] - m[both]) <= 1).mean())
    rows.append(row)
print("\nSTRUCTURE SCALES (Jev stated>=0.5; Jev score+1 vs matcher level where both state it; mousse only on sparkling)\n", pd.DataFrame(rows).round(2).to_string(index=False))
print("\nAPPEARANCE / CONDITION")
jc = C("app__colour"); mc = V["cat_colour"].values
print(f"  colour: Jev states {np.mean(jc != 'not_stated'):.3f} | matcher {pd.notna(mc).mean():.3f} | both {int(((jc != 'not_stated') & pd.notna(mc)).sum())} -> exact agreement {np.mean([a == b for a, b in zip(jc[(jc != 'not_stated') & pd.notna(mc)], mc[(jc != 'not_stated') & pd.notna(mc)])]) if ((jc != 'not_stated') & pd.notna(mc)).any() else float('nan'):.2f}")
for f, k in (("clarity", "app__clarity"), ("appearance_intensity", "app__intensity")):
    jj = C(k); mm = V["cat_" + f].values; b = (jj != "not_stated") & pd.notna(mm)
    print(f"  {f}: Jev states {np.mean(jj != 'not_stated'):.3f} | matcher {pd.notna(mm).mean():.3f} | both {int(b.sum())} -> agreement {np.mean([x == y for x, y in zip(jj[b], mm[b])]) if b.any() else float('nan'):.2f}")
pc = A("nose__condition"); mcnd = V["cat_condition"].notna().values
print(f"  condition(unclean): Jev {np.mean(pc >= .5):.3f} | matcher {mcnd.mean():.3f} | agree {np.mean((pc >= .5) == mcnd):.2f}")
jr = C("readiness"); mr = V["readiness"].fillna("not_stated").values
print("\nREADINESS: Jev vs matcher outcome (rows=Jev)\n", pd.crosstab(pd.Series(jr, name="jev"), pd.Series(mr, name="matcher")).to_string())
q = S("quality") ; pts = N.points.values
print(f"\nQUALITY: Jev quality score vs critic points: Spearman {spearmanr(q, pts).correlation:.2f} | mean Jev score by points band:", pd.Series(q).groupby(V['quality_band'].values).mean().round(2).to_dict())
r.to_csv(ROOT / "outputs/jev_vs_matcher_clusters.csv", index=False)
