"""Does the recommender make sense? An offline benchmark that uses Jev's WSET answers as the yardstick (NOT used by the recommender itself).
For queries that Jev has labelled, every technique returns 10 wines; for the neighbours that Jev has also labelled we can ask the taster's questions:
  * same colour group? red <-> white clash? sparkling <-> still clash?                       (all wines: metadata)
  * sweetness / body / acidity / tannin more than 2 levels apart (a WSET-card clash)?        (labelled pairs only)
  * do the two wines share the same descriptors (Jev yes-answers)?  taste similarity?        (labelled pairs only)
Writes results.json (metrics with 95% intervals) and sanity_lists.json (30 hand-picked-by-rule queries with their lists, for the preview panel).
Usage: python experiments/recsys_sanity/benchmark.py [n_queries]
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import recsys as RS
from wsetkit import review_id

OUT = Path(__file__).resolve().parent
NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
K = 10
TECH = ["random", "raw", "zw", "latent", "emb", "hybrid", "rrf", "mmr"]
LEVELS = ["level__sweetness", "level__body", "level__acidity", "level__tannin"]

R = RS.Rec(); d = R.d
meta = json.load(open(ROOT / "dashboard/data/recsys_meta.json")); ids = np.array(meta["ids"]); n = R.n
colour, group, country = d["colour"].astype(str), d["group"].astype(str), d["country"].astype(str)

# ---- Jev's answers for the wines of the index that have been labelled
J = {}
for f in ("jev_pilot.jsonl", "jev_1000.jsonl", "jev_more.jsonl"):
    for line in open(ROOT / "outputs" / f):
        r = json.loads(line); J[r["id"]] = r["answers"]
lab = np.array([i in J for i in ids]); L_idx = np.where(lab)[0]
print(f"index {n} wines | {lab.sum()} have Jev answers ({100 * lab.mean():.0f}%)")
first = next(iter(J.values())); BIN = [q for q, a in first.items() if a["type"] == "noul"]
Bm = np.full((n, len(BIN)), np.nan, np.float32); LV = np.full((n, len(LEVELS)), np.nan, np.float32)
for i in L_idx:
    a = J[ids[i]]; Bm[i] = [a[q]["noul"] for q in BIN]; LV[i] = [a[q]["score"] for q in LEVELS]
mu, sd = np.nanmean(Bm[L_idx], 0), np.nanstd(Bm[L_idx], 0) + 1e-3
Zj = (Bm - mu) / sd                                                    # z-scored Jev yes/no vector: taste similarity that does not let common descriptors dominate
Zn = Zj / np.maximum(np.linalg.norm(np.nan_to_num(Zj), axis=1, keepdims=True), 1e-9)
Yes = np.nan_to_num(Bm) >= 0.5

# ---- Jev-aligned taste vectors (jev_vector.py): predicted from the words, out-of-fold for labelled wines
JV = np.load(OUT / "jev_vectors.npy") if (OUT / "jev_vectors.npy").exists() else None
CUSTOM = {}
if JV is not None:
    nb = len(BIN); z = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-6)
    Zb = z(JV[:, :nb]); Zl = z(JV[:, nb:])
    def unit(M): return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    LW = 3.0                                                              # weight of the 8 structure levels relative to one yes/no answer (they carry sweetness, body, acidity...)
    M_desc, M_full = unit(Zb), unit(np.hstack([Zb, LW * Zl]))
    zs_ = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    def rank_(s, i): s = s.copy(); s[i] = -1e9; idx = np.argpartition(-s, K)[:K]; return idx[np.argsort(-s[idx])]
    def rrf_(ss, k0=60):
        sc = np.zeros(n)
        for s in ss: r = np.empty(n); r[np.argsort(-s)] = np.arange(1, n + 1); sc += 1.0 / (k0 + r)
        return sc
    E_ = R.M["emb"]
    CUSTOM = {"jev_desc": lambda i: rank_(M_desc @ M_desc[i], i),
              "jev": lambda i: rank_(M_full @ M_full[i], i),
              "jev_hybrid": lambda i: rank_(0.5 * zs_(M_full @ M_full[i]) + 0.5 * zs_(E_ @ E_[i]), i),
              "jev_rrf": lambda i: rank_(rrf_([M_full @ M_full[i], E_ @ E_[i]]), i)}

# ---- the taster's rules: candidates must be the same colour family (so no red for white, no sparkling for still) and not more than SWMAX sweetness levels apart (predicted levels, so it also works for unlabelled wines)
SWMAX = 1.0
SW = JV[:, len(BIN)] if JV is not None else None
def allowed(i):
    m = colour == colour[i]
    if SW is not None: m = m & (np.abs(SW - SW[i]) < SWMAX)
    m[i] = False; return m
def topk_masked(sc, i, mask):
    sc = np.where(mask, sc, -1e9); sc[i] = -1e9; idx = np.argpartition(-sc, K)[:K]; return idx[np.argsort(-sc[idx])]
SCORE = {"hybrid": lambda i: R.hybrid(R.query(i), 0.5), "rrf": lambda i: R.rrf(R.query(i))}
if JV is not None:
    SCORE["jev"] = lambda i: M_full @ M_full[i]
    SCORE["jev_hybrid"] = lambda i: 0.5 * zs_(M_full @ M_full[i]) + 0.5 * zs_(E_ @ E_[i])
for t_, f_ in SCORE.items(): CUSTOM[t_ + "+rules"] = (lambda f: lambda i: topk_masked(f(i), i, allowed(i)))(f_)
def why(i, j):
    if colour[i] != colour[j]: return f"{colour[j]} wine for a {colour[i]} query"
    if SW is not None and abs(SW[i] - SW[j]) >= SWMAX: return f"sweetness {SW[j]:.1f} vs {SW[i]:.1f}"
    return None
TECH = TECH + list(CUSTOM)
rng = np.random.RandomState(11); Q = rng.choice(L_idx, min(NQ, len(L_idx)), replace=False)
STILL = lambda c: c != "sparkling"


def ci(p, m):                                   # 95% normal interval for a rate
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / max(m, 1))


# choose SWMAX on DIFFERENT queries than the ones we report on
if JV is not None:
    tq = np.random.RandomState(99).choice(np.setdiff1d(L_idx, Q), 600, replace=False)
    print("tuning the sweetness limit on 600 other queries (jev_hybrid + rules):")
    for sw_ in (1.0, 1.5, 2.0, 2.5, 99):
        SWMAX = sw_; Lt = np.array([CUSTOM["jev_hybrid+rules"](int(i)) for i in tq]); m_ = lab[Lt]; qi_ = np.repeat(tq, K).reshape(Lt.shape)[m_]; ni_ = Lt[m_]
        g_ = np.abs(LV[qi_] - LV[ni_]); print(f"  limit {sw_:>4}: sweetness gap>=2 {(g_[:, 0] >= 2).mean():.3f} | sweet-or-body gap {((g_[:, 0] >= 2) | (g_[:, 1] >= 2)).mean():.3f} | taste cos {(Zn[qi_] * Zn[ni_]).sum(1).mean():.3f}")
    SWMAX = 1.0
res, lists = {}, {}
for t in TECH:
    Lst = np.array([(CUSTOM[t](int(i)) if t in CUSTOM else R.technique(t, int(i), K)) for i in Q]); lists[t] = Lst
    qc, nc = colour[Q][:, None], colour[Lst]
    rw = ((qc == "red") & (nc == "white")) | ((qc == "white") & (nc == "red"))
    sp = (qc == "sparkling") != (nc == "sparkling")
    same_grape = group[Lst] == group[Q][:, None]; same_cty = country[Lst] == country[Q][:, None]
    # labelled pairs only
    m = lab[Lst]; q_idx = np.repeat(Q, K).reshape(Lst.shape)
    qi, ni = q_idx[m], Lst[m]; npairs = int(m.sum())
    gap = np.abs(LV[qi] - LV[ni])                                           # (pairs, 4) level distance, in WSET levels
    sweet_clash = float((gap[:, 0] >= 2).mean()); body_clash = float((gap[:, 1] >= 2).mean()); acid_clash = float((gap[:, 2] >= 2).mean())
    red_pairs = np.isin(colour[qi], ["red"]) & np.isin(colour[ni], ["red"]); tan_clash = float((gap[red_pairs, 3] >= 2).mean()) if red_pairs.sum() else np.nan
    any_style = float(((gap[:, 0] >= 2) | (gap[:, 1] >= 2)).mean())
    cos = (Zn[qi] * Zn[ni]).sum(1)
    inter = (Yes[qi] & Yes[ni]).sum(1); union = np.maximum((Yes[qi] | Yes[ni]).sum(1), 1); jac = inter / union
    lists_hard = float((rw | sp).any(axis=1).mean())                          # share of 10-wine lists with at least one red<->white or sparkling<->still wine
    cnt = np.bincount(Lst.ravel(), minlength=n)
    res[t] = dict(colour_same=float((qc == nc).mean()), red_white_clash=float(rw.mean()), red_white_clash_ci=float(ci(rw.mean(), rw.size)), sparkling_still_clash=float(sp.mean()), lists_with_hard_clash=lists_hard,
                  grape_group=float(same_grape.mean()), country=float(same_cty.mean()), labelled_pairs=npairs,
                  sweetness_clash=sweet_clash, sweetness_clash_ci=float(ci(sweet_clash, npairs)), body_clash=body_clash, acidity_clash=acid_clash, tannin_clash_red=tan_clash, style_clash_any=any_style,
                  taste_cosine=float(cos.mean()), taste_cosine_ci=float(1.96 * cos.std() / np.sqrt(max(npairs, 1))), descriptor_jaccard=float(jac.mean()),
                  coverage=float((cnt > 0).mean()), never_shown=float((cnt == 0).mean()), hub_max=int(cnt.max()))
    if t.endswith('+rules'):
        off = lists[t[:-6]]; res[t]['wines_swapped_per_list'] = float(np.mean([(~allowed(int(q))[off[r]]).sum() for r, q in enumerate(Q)]))
    r_ = res[t]
    print(f"{t:7s} red<->white {r_['red_white_clash']:.3f} spark/still {r_['sparkling_still_clash']:.3f} | sweet>=2 lvl {r_['sweetness_clash']:.3f} body>=2 {r_['body_clash']:.3f} any {r_['style_clash_any']:.3f} | taste cos {r_['taste_cosine']:.3f} jacc {r_['descriptor_jaccard']:.3f} | grape {r_['grape_group']:.3f} | lists with a hard clash {r_['lists_with_hard_clash']:.2f} | never {r_['never_shown']:.2f}")

json.dump(dict(n_index=n, n_labelled=int(lab.sum()), queries=int(len(Q)), k=K, note="queries and pairs are restricted to wines Jev has labelled; metrics on labelled pairs use Jev's answers as the yardstick",
               techniques=res), open(OUT / "results.json", "w"), indent=1)

# ---- 30 sanity queries: stratified by colour / grape group so that reds, whites, rose, sparkling and sweet wines are all there
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description", "title", "variety", "country", "points", "price", "winery"])
raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id")
def card(i):
    r = raw.loc[ids[i]]
    sw = None if np.isnan(LV[i, 0]) else round(float(LV[i, 0]), 1)
    return dict(i=int(i), title=(str(r.title) if pd.notna(r.title) else f"{r.winery} {r.variety}"), variety=str(r.variety), country=str(r.country), colour=str(colour[i]), points=int(r.points), price=None if pd.isna(r.price) else float(r.price),
                note=str(r.description)[:170], sweetness=sw, body=None if np.isnan(LV[i, 1]) else round(float(LV[i, 1]), 1))
rng2 = np.random.RandomState(5); picks = []
strata = [("red", 10), ("white", 10), ("rose", 3), ("sparkling", 4)]
for c, k in strata:
    pool = [i for i in L_idx if colour[i] == c and 160 <= len(raw.loc[ids[i], "description"]) <= 420]; rng2.shuffle(pool); picks += pool[:k]
sweet = [i for i in L_idx if LV[i, 0] >= 2.5][:200]; rng2.shuffle(sweet); picks += [i for i in sweet if i not in picks][:3]
out = []
for i in picks:
    entry = dict(query=card(i), lists={})
    for t in ("random", "raw", "zw", "emb", "hybrid", "rrf") + tuple(CUSTOM):
        nb = CUSTOM[t](int(i)) if t in CUSTOM else R.technique(t, int(i), K); items = []
        for j in nb:
            c = card(j); qc_, nc_ = colour[i], colour[j]
            c["flags"] = dict(colour_clash=bool({qc_, nc_} == {"red", "white"}), sparkling_still=bool((qc_ == "sparkling") != (nc_ == "sparkling")),
                              sweetness_gap=None if (np.isnan(LV[i, 0]) or np.isnan(LV[j, 0])) else round(float(abs(LV[i, 0] - LV[j, 0])), 1),
                              same_grape=bool(group[i] == group[j]), same_country=bool(country[i] == country[j]))
            items.append(c)
        entry["lists"][t] = items
    entry["removed"] = {}
    for t in SCORE:
        off = CUSTOM[t](int(i)) if False else (R.technique(t, int(i), K) if t in ("hybrid", "rrf") else (rank_(SCORE[t](int(i)), int(i))))
        entry["removed"][t] = [dict(card(j), why=why(int(i), int(j))) for j in off if why(int(i), int(j))]
    out.append(entry)
json.dump(out, open(OUT / "sanity_lists.json", "w"), indent=0, default=str)
print("wrote results.json and sanity_lists.json with", len(out), "queries")
