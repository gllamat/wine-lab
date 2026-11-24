"""Learned phrases for typed notes. Mines (phrase -> Jev yes/no question) rules from the labelled notes and turns a typed description into a Jev-aligned vector WITHOUT any model:
  score(question) = precision of the strongest mined phrase found in the text (0 if none), floored at the index average so that 'no phrase' is neutral;
  the 8 structure levels come from the existing WSET cue matcher (the s_ cues) when the text states them, else the index average.
Evaluation (clean): rules mined ONLY from labelled notes that are not in the 25,000-wine index; queries = typed descriptions (the notes) of 1,200 labelled index wines.
Compared: cue vector (zw) | learned phrases | the wine's own predicted vector (what the index uses, for reference).
Ships: dashboard/data/phrases.json (rules mined from ALL labelled notes), phrases_eval.json.
Usage: python experiments/recsys_sanity/mine_phrases.py"""
import sys, json, re, unicodedata, time
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import recsys as RS

OUT = Path(__file__).resolve().parent
K = 10; NB = 216; MIN_SUPPORT = int(sys.argv[1]) if len(sys.argv) > 1 else 20; MIN_PREC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6


def lemma_(w):                                                              # keep identical to the dashboard (dashboard/phrases.js)
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "zes", "sses")): return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")): return w[:-1]
    return w


def phrases(text):
    t = unicodedata.normalize("NFKD", str(text).lower()); t = "".join(c for c in t if not unicodedata.combining(c)).replace("'", "").replace("-", " ")
    out = []
    for clause in re.split(r"[.;:,!?()]", t):
        w = [lemma_(x) for x in re.findall(r"[a-z]+", clause)]
        out += w + [w[i] + " " + w[i + 1] for i in range(len(w) - 1)]
    return out


R = RS.Rec(); d = R.d; n = R.n
meta = json.load(open(ROOT / "dashboard/data/recsys_meta.json")); ids = np.array(meta["ids"]); names = meta["names"]; colour = d["colour"].astype(str)
JV = np.load(OUT / "jev_vectors.npy"); schema = json.load(open(ROOT / "distill/pack/wset_distill_schema.json")); BIN = schema["binary"]
SC = ["sweetness", "acidity", "tannin", "alcohol", "body", "flavour_intensity", "finish", "nose_intensity"]
NL = {s: len(schema["groups"]["level__" + s]) for s in SC}
pack = pd.read_parquet(ROOT / "distill/pack/wset_distill_pack.parquet"); pos = {i: k for k, i in enumerate(pack.id)}
in_idx = pack.id.isin(set(ids)).values; Yb = pack[["b::" + q for q in BIN]].to_numpy(np.float32)
print(f"labelled notes {len(pack)} | of them in the index {int(in_idx.sum())} | mining on {int((~in_idx).sum())} for the clean evaluation")


# level classes: one column per level of each of the 8 structure scales (the class is Jev's most likely level)
YL = np.concatenate([np.eye(NL[s], dtype=np.float32)[np.array(pack["g::level__" + s].tolist(), dtype=np.float32).argmax(1)] for s in SC], axis=1); LOFF = np.cumsum([0] + [NL[s] for s in SC])


def mine(rows, min_support, min_prec, levels=False):
    vec = CountVectorizer(analyzer=phrases, min_df=5, binary=True, dtype=np.float32); X = vec.fit_transform(pack.description.iloc[rows]).tocsc()
    V = np.array(vec.get_feature_names_out()); Y = YL[rows] if levels else (Yb[rows] >= 0.5).astype(np.float32)
    n_with = np.asarray(X.sum(0)).ravel(); both = np.asarray((X.T @ sp.csr_matrix(Y)).todense()); prec = both / np.maximum(n_with[:, None], 1)
    ok = (both >= min_support) & (prec >= min_prec)
    IDX = {w: i for i, w in enumerate(V)}                                     # prune: a two-word rule is redundant when one of its words is already as precise a rule
    for i, w in enumerate(V):
        if " " in w:
            for u in w.split(" "):
                j = IDX.get(u)
                if j is not None:
                    red = ok[j] & (prec[j] >= prec[i] - 0.05); ok[i] &= ~red
    return V, ok, prec.astype(np.float32)


t0 = time.time(); V, ok, prec = mine(np.where(~in_idx)[0], MIN_SUPPORT, MIN_PREC); VL, okl, precl = mine(np.where(~in_idx)[0], MIN_SUPPORT, MIN_PREC, True)
print(f"clean rules (support >= {MIN_SUPPORT}, precision >= {MIN_PREC}): {int(ok.sum()):,} phrase-question rules over {int(ok.any(1).sum()):,} distinct phrases, questions with a rule {int(ok.any(0).sum())}/{NB} ({time.time() - t0:.0f}s)")

# ---- the typed-note path (Python reference of what the browser does)
mu_b, sd_b = JV[:, :NB].mean(0), JV[:, :NB].std(0) + 1e-6; mu_l, sd_l = JV[:, NB:].mean(0), JV[:, NB:].std(0) + 1e-6
PHR = {w: i for i, w in enumerate(V)}; P = np.where(ok, prec, 0); PHRL = {w: i for i, w in enumerate(VL)}; PL = np.where(okl, precl, 0)
cue_ix = {nme: j for j, nme in enumerate(names)}


def typed_vector(text, xrow):
    sc = np.zeros(NB, np.float32)
    for p in set(phrases(text)):
        j = PHR.get(p)
        if j is not None: sc = np.maximum(sc, P[j])
    sc = np.maximum(sc, mu_b)                                                 # no phrase found -> neutral (the index average)
    cl = np.zeros(LOFF[-1], np.float32)
    for p in set(phrases(text)):
        j = PHRL.get(p)
        if j is not None: cl = np.maximum(cl, PL[j])
    lv = np.zeros(len(SC), np.float32)
    for k, s in enumerate(SC):
        w = cl[LOFF[k]:LOFF[k + 1]]
        if w.sum() > 0: lv[k] = (w * np.arange(NL[s])).sum() / w.sum()                         # level = precision-weighted vote of the level phrases found
        elif ("s_" + s + "_na") in cue_ix and xrow[cue_ix["s_" + s + "_na"]] < 0.5: lv[k] = xrow[cue_ix["s_" + s]] * (NL[s] - 1)      # else the WSET cue matcher, if it found a stated level
        else: lv[k] = mu_l[k]                                                                   # else the index average
    return np.concatenate([(sc - mu_b) / sd_b, 3.0 * (lv - mu_l) / sd_l])


unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
Zidx = np.hstack([(JV[:, :NB] - mu_b) / sd_b, 3.0 * (JV[:, NB:] - mu_l) / sd_l]); Mj = unit(Zidx).astype(np.float32)

# ---- evaluation on labelled index wines
J = {}
for f in ("jev_pilot.jsonl", "jev_1000.jsonl", "jev_more.jsonl"):
    for line in open(ROOT / "outputs" / f):
        r = json.loads(line); J[r["id"]] = r["answers"]
lab = np.array([i in J for i in ids]); L_idx = np.where(lab)[0]
LV = np.full((n, 2), np.nan, np.float32); Bm = np.full((n, NB), np.nan, np.float32)
for i in L_idx: a = J[ids[i]]; LV[i] = [a["level__sweetness"]["score"], a["level__body"]["score"]]; Bm[i] = [a[q]["noul"] for q in BIN]
Zj = (Bm - np.nanmean(Bm[L_idx], 0)) / (np.nanstd(Bm[L_idx], 0) + 1e-3); Zn = Zj / np.maximum(np.linalg.norm(np.nan_to_num(Zj), axis=1, keepdims=True), 1e-9)
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
from wsetkit import review_id
raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id"); text = {i: raw.loc[ids[i], "description"] for i in L_idx}
Q = np.random.RandomState(21).choice(L_idx, 1200, replace=False)


def metrics(L, Qs):
    m = lab[L]; qi = np.repeat(Qs, K).reshape(L.shape)[m]; ni = L[m]; g = np.abs(LV[qi] - LV[ni]); qc, nc = colour[Qs][:, None], colour[L]
    rw = ((qc == "red") & (nc == "white")) | ((qc == "white") & (nc == "red")) | ((qc == "sparkling") != (nc == "sparkling"))
    return dict(style_gap=float(((g[:, 0] >= 2) | (g[:, 1] >= 2)).mean()), taste=float((Zn[qi] * Zn[ni]).sum(1).mean()), lists_with_hard_clash=float(rw.any(1).mean()), pairs=int(m.sum()))


def top(s, i): s = s.copy(); s[i] = -1e9; idx = np.argpartition(-s, K)[:K]; return idx[np.argsort(-s[idx])]


res = {}
Lc = np.array([top(R.cos("zw", R.query(int(i))), int(i)) for i in Q]); res["typed: WSET cues (zw)"] = metrics(Lc, Q)
Tv = np.array([typed_vector(text[i], R.Xd[i]) for i in Q]); Mt = unit(Tv).astype(np.float32)
Lp = np.array([top(Mj @ Mt[r], int(i)) for r, i in enumerate(Q)]); res["typed: learned phrases"] = metrics(Lp, Q)
Lo = np.array([top(Mj @ Mj[i], int(i)) for i in Q]); res["reference: the wine's own Jev vector"] = metrics(Lo, Q)
Lr = np.array([np.random.RandomState(int(i)).choice(np.where(colour == colour[i])[0], K, replace=False) for i in Q]); res["random, same colour"] = metrics(Lr, Q)
for k, v in res.items(): print(f"{k:42s} style gap {v['style_gap']:.3f} | taste {v['taste']:.3f} | lists with a colour clash {v['lists_with_hard_clash']:.2f}")
json.dump(dict(min_support=MIN_SUPPORT, min_precision=MIN_PREC, queries=len(Q), techniques=res, rules_clean=int(ok.sum())), open(OUT / "phrases_eval.json", "w"), indent=1)

# ---- ship: rules mined from ALL labelled notes, as compact json
V2, ok2, prec2 = mine(np.arange(len(pack)), MIN_SUPPORT, MIN_PREC); V2l, ok2l, prec2l = mine(np.arange(len(pack)), MIN_SUPPORT, MIN_PREC, True)
keep = np.where(ok2.any(1))[0]
out = dict(min_support=MIN_SUPPORT, min_precision=MIN_PREC, questions=BIN, phrases=[V2[i] for i in keep], rules=[[[int(np.searchsorted(keep, i)), int(round(100 * prec2[i, j]))] for i in np.where(ok2[:, j])[0]] for j in range(NB)],
           mu=[round(float(x), 4) for x in mu_b], sd=[round(float(x), 4) for x in sd_b], lmu=[round(float(x), 4) for x in mu_l], lsd=[round(float(x), 4) for x in sd_l], scales=SC, levels=[NL[s] for s in SC], n_rules=int(ok2.sum()) + int(ok2l.sum()), n_labelled=len(pack),
           lphrases=[V2l[i] for i in np.where(ok2l.any(1))[0]], lrules=[[[int(np.searchsorted(np.where(ok2l.any(1))[0], i)), int(round(100 * prec2l[i, j]))] for i in np.where(ok2l[:, j])[0]] for j in range(YL.shape[1])])
json.dump(out, open(ROOT / "dashboard/data/phrases.json", "w"), separators=(",", ":"))
print("wrote dashboard/data/phrases.json", round((ROOT / "dashboard/data/phrases.json").stat().st_size / 1024), "KB |", out["n_rules"], "rules |", len(keep), "phrases")
