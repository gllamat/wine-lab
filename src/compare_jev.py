"""Compare Jev (L4-card prompt, every descriptor) with the deterministic matcher on the same reviews. No review text used."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon

SLOTS = {"dried_fruit"}   # primary fruit_ripeness descriptor also matches tertiary red/white dried_fruit
J = []
for f in ("outputs/jev_pilot.jsonl", "outputs/jev_1000.jsonl"):
    J += [json.loads(l) for l in open(ROOT / f)]
vec = pd.read_parquet(ROOT / "outputs/wset_vectors_all.parquet").set_index("id")
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet").set_index("id")
ids = [j["id"] for j in J]; V = vec.loc[ids]; N = nm.loc[ids]
lex = Lexicon()
get = lambda k, f: np.array([j["answers"][k][f] for j in J])
P = lambda k: get(k, "noul")
RIP = np.array([j["answers"]["ripeness"]["choice"] for j in J])   # assumption A1: one-of fruit ripeness
print(f"reviews {len(J)} | model {set(j['model'] for j in J)} | questions per review {len(J[0]['answers'])}\n")


def kap(jv, mv):
    return cohen_kappa_score(jv, mv) if jv.any() and mv.any() and not (jv.all() and mv.all()) else np.nan


# ---- descriptor level (Jev nose OR palate >=0.5 vs deterministic any scope)
rows = []
for cl, d in lex.descriptors:
    if d.endswith("_generic"): continue
    pj = (RIP == d) if cl == "fruit_ripeness" else np.maximum(P(f"nose__{cl}__{d}"), P(f"palate__{cl}__{d}")) >= 0.5; mv = (V[[c for c in V if c.startswith("a_") and c.endswith("__" + d) and (c[2:].split("__")[0] == cl or d in SLOTS)]] == 1).any(axis=1).values
    rows.append(dict(cluster=cl, tier=lex.tier[cl], descriptor=d, jev=pj.mean(), det=mv.mean(), both=int((pj & mv).sum()), jev_only=int((pj & ~mv).sum()), det_only=int((~pj & mv).sum()), kappa=kap(pj, mv)))
dd = pd.DataFrame(rows)
print("DESCRIPTOR LEVEL (99 SAT descriptors): total mentions found by both / Jev only / deterministic only")
print(dd.groupby("tier")[["both", "jev_only", "det_only"]].sum().to_string()); print(f"overall: both {dd.both.sum()} | Jev only {dd.jev_only.sum()} | deterministic only {dd.det_only.sum()} | median kappa {dd.kappa.median():.2f}")
print("most Jev-only descriptors:", dd.sort_values("jev_only", ascending=False).head(8)[["cluster", "descriptor", "jev_only", "both"]].to_string(index=False, header=False).replace("\n", " | "))
print("most deterministic-only:", dd.sort_values("det_only", ascending=False).head(8)[["cluster", "descriptor", "det_only", "both"]].to_string(index=False, header=False).replace("\n", " | "))
# ---- cluster level
rows = []
for cl, tier in lex.tier.items():
    ds = [d for c, d in lex.descriptors if c == cl and not d.endswith("_generic")]
    pj = (RIP != "not_stated") if cl == "fruit_ripeness" else np.max([np.maximum(P(f"nose__{cl}__{d}"), P(f"palate__{cl}__{d}")) for d in ds], axis=0) >= 0.5; mv = V["c_" + cl].values == 1
    rows.append(dict(cluster=cl, tier=tier, jev=pj.mean(), det=mv.mean(), agree=(pj == mv).mean(), kappa=kap(pj, mv)))
cc = pd.DataFrame(rows); print("\nCLUSTER LEVEL\n", cc.round(2).to_string(index=False)); print(f"mean kappa {cc.kappa.mean():.2f} | agreement {cc.agree.mean():.2f}")
# ---- scales
rows = []
for s in ["sweetness", "acidity", "tannin", "alcohol", "body", "flavour_intensity", "finish", "nose_intensity"]:
    st = P(f"stated__{s}"); sco = get(f"level__{s}", "score") + 1; m = V["s_" + s].values
    mf = ~np.isnan(m); jv = st >= .5; both = mf & jv
    row = dict(scale=s, levels=len(lex.scales[s]), jev_stated=jv.mean(), det_stated=mf.mean(), both=int(both.sum()), jev_only=int((jv & ~mf).sum()), det_only=int((~jv & mf).sum()))
    if both.sum() >= 8:
        row.update(spearman=spearmanr(sco[both], m[both]).correlation, exact=float((np.rint(sco[both]) == np.rint(m[both])).mean()), within1=float((np.abs(sco[both] - m[both]) <= 1).mean()))
    rows.append(row)
print("\nSCALES\n", pd.DataFrame(rows).round(2).to_string(index=False))
# ---- L4-only fields
print("\nTANNIN NATURE / TEXTURE / PETILLANCE / APPEARANCE")
for name, vals, col in (("tannin nature", ["ripe", "soft", "smooth", "unripe", "green", "coarse", "stalky", "chalky", "fine-grained"], "cat_tannin_nature"), ("texture", ["oily", "creamy", "austere", "luscious"], "cat_texture")):
    pre = "tnature__" if "tannin" in name else "texture__"
    jv = np.array([[P(pre + v)[i] >= .5 for v in vals] for i in range(len(J))]); mv = np.array([[v in str(x).split("|") for v in vals] for x in V[col].values])
    print(f"  {name}: Jev states in {jv.any(axis=1).mean():.3f} of reviews | deterministic {mv.any(axis=1).mean():.3f} | per-value both {int((jv & mv).sum())}, Jev only {int((jv & ~mv).sum())}, det only {int((~jv & mv).sum())}")
jp, dp = P("palate__petillance") >= .5, V["cat_petillance"].notna().values
print(f"  petillance: Jev {jp.mean():.3f} | det {dp.mean():.3f} | both {int((jp & dp).sum())}")
jc = np.array([j["answers"]["app__colour"]["choice"] for j in J]); mc = V["cat_colour"].values
b = (jc != "not_stated") & pd.notna(mc)
print(f"  colour: Jev states {np.mean(jc != 'not_stated'):.3f} | det {pd.notna(mc).mean():.3f} | both {int(b.sum())} -> agree {np.mean([x in str(y).split('|') for x, y in zip(jc[b], mc[b])]) if b.any() else float('nan'):.2f}")
cj = np.array([j["answers"]["condition"]["choice"] for j in J]); print(f"  condition (one-of): Jev clean {np.mean(cj == 'clean'):.3f} / unclean {np.mean(cj == 'unclean'):.3f} / not stated {np.mean(cj == 'not_stated'):.3f} | det unclean {V['cat_condition'].notna().mean():.3f} | both unclean {int(((cj == 'unclean') & V['cat_condition'].notna().values).sum())}")
print("\nFRUIT RIPENESS, one of per note (assumption A1) rows = Jev, cols = deterministic")
dr = V["fruit_ripeness"].fillna("none").where(V["fruit_ripeness_mixed"] == 0, "mixed").values
print(pd.crosstab(pd.Series(RIP, name="jev"), pd.Series(dr, name="det")).to_string())
bk = (RIP != "not_stated") & (dr != "none") & (dr != "mixed"); print(f"both state a single value: {int(bk.sum())} -> exact agreement {np.mean(RIP[bk] == dr[bk]):.2f}, kappa {cohen_kappa_score(RIP[bk], dr[bk]):.2f}")
ja = np.array([j["answers"]["ageing"]["choice"] for j in J]); da = V["bottle_ageing"].fillna("not_stated").values
print("\nBOTTLE AGEING (rows = Jev)\n", pd.crosstab(pd.Series(ja, name="jev"), pd.Series(da, name="det")).to_string())
q = get("quality", "score"); print(f"\nQUALITY: Jev score vs critic points Spearman {spearmanr(q, N.points.values).correlation:.2f}")
dd.to_csv(ROOT / "outputs/jev_vs_det_descriptors.csv", index=False)
