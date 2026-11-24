"""Hubness: do the same few wines show up in everybody's list? Every wine of the 25,000-wine index is used as a query (250,000 list slots).
Measures, per technique: share of wines that appear at least once, share that never appear, how much of all list slots the top 1% of wines take, the maximum number of lists one wine is in, and the Gini coefficient of exposure.
Fix tested: CSLS-style hub correction: score(q, c) - lam * r_c, where r_c = mean cosine of wine c to its 10 nearest wines (a wine that is close to everything gets a penalty). The offsets r_c are one number per wine per space.
lam is chosen on 600 OTHER labelled queries (taste similarity and style gaps must not get worse).
Writes hubness.json (metrics, hub list, histogram) and hub_offsets.json (the r_c per space, for the dashboard).
Usage: python experiments/recsys_sanity/benchmark_hub.py"""
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import recsys as RS

OUT = Path(__file__).resolve().parent
K = 10; SWMAX = 1.0; BLOCK = 1000
R = RS.Rec(); d = R.d; n = R.n
meta = json.load(open(ROOT / "dashboard/data/recsys_meta.json")); ids = np.array(meta["ids"]); colour = d["colour"].astype(str)
JV = np.load(OUT / "jev_vectors.npy"); NB = 216
z = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-6); unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
Mj = unit(np.hstack([z(JV[:, :NB]), 3.0 * z(JV[:, NB:])])).astype(np.float32); Me = R.M["emb"].astype(np.float32); SW = JV[:, NB]

# ---- the offsets: r_c = mean of the 10 largest cosines of wine c to the OTHER wines
t0 = time.time(); off = {}
for name, M in (("jev", Mj), ("emb", Me)):
    r = np.zeros(n, np.float32)
    for s in range(0, n, BLOCK):
        S = M[s:s + BLOCK] @ M.T; S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1; r[s:s + BLOCK] = np.sort(np.partition(S, -K, axis=1)[:, -K:], axis=1).mean(1)
    off[name] = r
print("offsets", {k: (round(float(v.mean()), 3), round(float(v.std()), 3)) for k, v in off.items()}, round(time.time() - t0), "s")
json.dump({k: [round(float(x), 4) for x in v] for k, v in off.items()}, open(OUT / "hub_offsets.json", "w"), separators=(",", ":"))

zs = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def topk(rows, tech, lam, rules):
    """top-K lists for the given query rows"""
    Sj = Mj[rows] @ Mj.T - lam * off["jev"][None, :]; Se = Me[rows] @ Me.T - lam * off["emb"][None, :]
    S = {"jev": Sj, "emb": Se, "jev_hybrid": 0.5 * zs(Sj) + 0.5 * zs(Se)}[tech]
    if rules: S = np.where((colour[rows][:, None] == colour[None, :]) & (np.abs(SW[rows][:, None] - SW[None, :]) < SWMAX), S, -1e9)
    S[np.arange(len(rows)), rows] = -1e9
    idx = np.argpartition(-S, K, axis=1)[:, :K]; return idx


def exposure(tech, lam, rules):
    cnt = np.zeros(n, np.int64)
    for s in range(0, n, BLOCK):
        L = topk(np.arange(s, min(n, s + BLOCK)), tech, lam, rules); cnt += np.bincount(L.ravel(), minlength=n)
    srt = np.sort(cnt)[::-1]; tot = cnt.sum(); cum = np.cumsum(np.sort(cnt)); gini = float(1 - 2 * (cum / tot).sum() / n + 1 / n)
    return cnt, dict(shown=float((cnt > 0).mean()), never=float((cnt == 0).mean()), top1pct_share=float(srt[:n // 100].sum() / tot), max_lists=int(cnt.max()), gini=gini)


# ---- quality on 600 other labelled queries (do the fixes hurt the taster's questions?)
J = {}
for f in ("jev_pilot.jsonl", "jev_1000.jsonl", "jev_more.jsonl"):
    for line in open(ROOT / "outputs" / f):
        r = json.loads(line); J[r["id"]] = r["answers"]
lab = np.array([i in J for i in ids]); L_idx = np.where(lab)[0]; first = next(iter(J.values())); BIN = [q for q, a in first.items() if a["type"] == "noul"]
LV = np.full((n, 2), np.nan, np.float32); Bm = np.full((n, len(BIN)), np.nan, np.float32)
for i in L_idx: a = J[ids[i]]; LV[i] = [a["level__sweetness"]["score"], a["level__body"]["score"]]; Bm[i] = [a[q]["noul"] for q in BIN]
Zj = (Bm - np.nanmean(Bm[L_idx], 0)) / (np.nanstd(Bm[L_idx], 0) + 1e-3); Zn = Zj / np.maximum(np.linalg.norm(np.nan_to_num(Zj), axis=1, keepdims=True), 1e-9)
tq = np.random.RandomState(99).choice(L_idx, 600, replace=False)


def quality(tech, lam, rules):
    L = topk(tq, tech, lam, rules); m = lab[L]; qi = np.repeat(tq, K).reshape(L.shape)[m]; ni = L[m]; g = np.abs(LV[qi] - LV[ni])
    return dict(style_gap=float(((g[:, 0] >= 2) | (g[:, 1] >= 2)).mean()), taste=float((Zn[qi] * Zn[ni]).sum(1).mean()))


res = {}
for tech in ("jev", "jev_hybrid"):
    for lam in (0.0, 0.5, 1.0, 1.5):
        cnt, e = exposure(tech, lam, True); q = quality(tech, lam, True); res[f"{tech}|{lam}"] = dict(e, **q)
        print(f"{tech:10s} lam {lam:.1f} +rules | shown {e['shown']:.3f} never {e['never']:.3f} top1% {e['top1pct_share']:.3f} max {e['max_lists']:5d} gini {e['gini']:.3f} | style gap {q['style_gap']:.3f} taste {q['taste']:.3f}", flush=True)
cnt0, e0 = exposure("jev_hybrid", 0.0, True)
LAM = 0.5
cnt1, e1 = exposure("jev_hybrid", LAM, True)
_, e_plain = exposure("jev_hybrid", 0.0, False)
_, e_emb = exposure("emb", 0.0, False)
top = np.argsort(-cnt0)[:12]
import pandas as pd
from wsetkit import review_id
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description", "title", "variety", "winery"]); raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id")
nm = lambda i: str(raw.loc[ids[i], "title"]) if str(raw.loc[ids[i], "title"]) != "None" else f"{raw.loc[ids[i], 'winery']} {raw.loc[ids[i], 'variety']}"
hist_bins = [0, 1, 2, 5, 10, 20, 50, 100, 1000000]
hist = lambda c: [int(((c >= a) & (c < b)).sum()) for a, b in zip(hist_bins[:-1], hist_bins[1:])]
json.dump(dict(n_wines=n, slots=n * K, lam=LAM, plain_emb=e_emb, plain_jev_hybrid=e_plain, rules=e0, rules_fix=e1, sweep=res, hist_bins=hist_bins, hist_rules=hist(cnt0), hist_rules_fix=hist(cnt1),
               top_hubs=[dict(title=nm(i), lists=int(cnt0[i]), lists_after_fix=int(cnt1[i]), hub_offset=float(off["jev"][i])) for i in top]), open(OUT / "hubness.json", "w"), indent=1)
print("wrote hubness.json; plain emb", e_emb, "| jev_hybrid", e_plain, "| +rules", e0, "| +rules +fix", e1)
