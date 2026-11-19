"""Recommender lab: "if you like this wine, you'll like ...". Content-based only (the corpus has critics' reviews, not consumers' ratings).
Builds a 25k-wine index for the browser (dashboard/data/recsys_*.{json,bin}), implements every technique in numpy, runs an offline evaluation (recsys_eval.json)
and writes a Python-vs-JS parity fixture. Techniques, in the order the dashboard teaches them:
  raw    cosine on the raw SAT cue vector             (common cues dominate)
  zw     cosine on the standardised vector            (rare cues count more: IDF-like)
  latent cosine in a 32-dim SVD/PCA 'taste space'     (denoised, smooth)
  emb    cosine on MiniLM text embeddings             (meaning of the whole sentence)
  hybrid z-scored score fusion of zw and emb, weight alpha
  rrf    reciprocal rank fusion of zw and emb
  mmr    maximal marginal relevance re-ranking of the hybrid list (relevance vs redundancy)
Usage: recsys.py build | eval | fixture"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT
import guess_model as G

N_ITEMS, SX, NLAT, SEED = 25000, 40.0, 32, "recsys1"
DASH = ROOT / "dashboard/data"


def l2(M): return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


def load_index():
    d = np.load(ROOT / "outputs/recsys_index.npz", allow_pickle=True); return {k: d[k] for k in d.files}


def build():
    ids, X, spec, nm = G.corpus("lite"); E = np.load(ROOT / "outputs/emb_minilm_all.npy").astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    order = np.argsort([hashlib.md5((SEED + i).encode()).hexdigest() for i in ids])[:N_ITEMS]; sel = np.sort(order)
    Xs, Es = X[sel], E[sel]; Xq = np.clip(np.round(Xs * SX), -127, 127).astype(np.int8)
    emax = float(np.abs(Es).max()); SE = 127.0 / emax; Eq = np.clip(np.round(Es * SE), -127, 127).astype(np.int8)
    Xd = Xq.astype(np.float32) / SX; Ed = Eq.astype(np.float32) / SE            # dequantised = what the browser sees
    Z = (Xd - mu) / sd; Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc[::2], full_matrices=False); comps = Vt[:NLAT].astype(np.float32)
    t = G.targets(nm); xy, _ = G.place_coords(t); coords = xy.reindex(t.triple.values[sel]).values
    names = spec["names"]; vint = nm.vintage.astype(float).values[sel]; prc = nm.price.astype(float).values[sel]; pts_ = nm.points.astype(float).values[sel]
    cat = lambda s: pd.Series(s).astype("category"); out = {}
    meta_cols = dict(variety=nm.variety_norm.values[sel], group=nm.variety_group.astype(str).values[sel], country=nm.country.astype(str).values[sel], province=nm.province.astype(str).values[sel],
                     region=nm.region_1.fillna("").astype(str).values[sel], winery=nm.winery.astype(str).values[sel], colour=nm.colour.astype(str).values[sel])
    dicts, codes = {}, {}
    for k, v in meta_cols.items(): c = cat(v); dicts[k] = [str(x) for x in c.cat.categories]; codes[k] = c.cat.codes.astype(int).tolist()
    meta = dict(n=int(len(sel)), ids=[str(i) for i in ids[sel]], dicts=dicts, codes=codes, vintage=[None if np.isnan(v) else int(v) for v in vint],
                price=[None if np.isnan(v) else float(v) for v in prc], points=[int(v) for v in pts_],
                lat=[None if np.isnan(v) else round(float(v), 3) for v in coords[:, 0]], lon=[None if np.isnan(v) else round(float(v), 3) for v in coords[:, 1]],
                names=names, mu=[round(float(v), 5) for v in mu], sd=[round(float(v), 5) for v in sd], sx=SX, se=SE, comps=np.round(comps, 5).tolist(), zmean=np.round(Z.mean(0), 5).tolist(), nlat=NLAT, dim_x=len(names), dim_e=int(E.shape[1]))
    json.dump(meta, open(DASH / "recsys_meta.json", "w"), separators=(",", ":")); Xq.tofile(DASH / "recsys_x.bin"); Eq.tofile(DASH / "recsys_e.bin")
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"]); from wsetkit import review_id
    raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id"); json.dump([raw.loc[i, "description"] for i in ids[sel]], open(DASH / "recsys_text.json", "w"), separators=(",", ":"))
    np.savez(ROOT / "outputs/recsys_index.npz", Xd=Xd, Ed=Ed, mu=mu, sd=sd, comps=comps, sel=sel, **{k: np.array(v) for k, v in meta_cols.items()}, vintage=vint, price=prc, points=pts_,
             lat=coords[:, 0], lon=coords[:, 1])
    print("index", len(sel), "| x.bin MB", round(Xq.nbytes / 1e6, 1), "| e.bin MB", round(Eq.nbytes / 1e6, 1), "| meta KB", round((DASH / "recsys_meta.json").stat().st_size / 1024))


class Rec:
    def __init__(self):
        d = load_index(); self.d = d; self.Xd, self.Ed = d["Xd"], d["Ed"]; self.mu, self.sd, self.comps = d["mu"], d["sd"], d["comps"]
        self.Z = (self.Xd - self.mu) / self.sd; self.Zc = self.Z - self.Z.mean(0); self.zmean = self.Z.mean(0)
        self.Lat = (self.Z - self.zmean) @ self.comps.T          # centred projection on the SVD components
        self.M = dict(raw=l2(self.Xd), zw=l2(self.Z), latent=l2(self.Lat), emb=l2(self.Ed)); self.n = len(self.Xd)

    def query(self, i):
        return dict(raw=self.M["raw"][i], zw=self.M["zw"][i], latent=self.M["latent"][i], emb=self.M["emb"][i])

    def cos(self, space, q): return self.M[space] @ q[space]

    def rank(self, s, k, excl=(), mask=None):
        s = s.copy(); s[list(excl)] = -1e9
        if mask is not None: s[~mask] = -1e9
        idx = np.argpartition(-s, k)[:k]; return idx[np.argsort(-s[idx])]

    def zs(self, s): return (s - s.mean()) / (s.std() + 1e-9)

    def hybrid(self, q, alpha=0.5): return alpha * self.zs(self.cos("zw", q)) + (1 - alpha) * self.zs(self.cos("emb", q))

    def rrf(self, q, k0=60):
        sc = np.zeros(self.n)
        for sp in ("zw", "emb"):
            r = np.empty(self.n); r[np.argsort(-self.cos(sp, q))] = np.arange(1, self.n + 1); sc += 1.0 / (k0 + r)
        return sc

    def mmr(self, rel, k, lam=0.7, pool=100, excl=()):
        cand = list(self.rank(rel, pool, excl)); chosen = []
        while len(chosen) < k and cand:
            if not chosen: best = cand[0]
            else:
                S = self.M["emb"][chosen]; sim = S @ self.M["emb"][cand].T                    # (chosen x cand)
                r = rel[cand]; r = (r - r.min()) / (r.max() - r.min() + 1e-9)
                best = cand[int(np.argmax(lam * r - (1 - lam) * sim.max(0)))]
            chosen.append(best); cand.remove(best)
        return np.array(chosen)

    def technique(self, name, i, k=10, **kw):
        q = self.query(i); ex = (i,)
        if name in ("raw", "zw", "latent", "emb"): return self.rank(self.cos(name, q), k, ex)
        if name == "hybrid": return self.rank(self.hybrid(q, kw.get("alpha", 0.5)), k, ex)
        if name == "rrf": return self.rank(self.rrf(q), k, ex)
        if name == "mmr": return self.mmr(self.hybrid(q, kw.get("alpha", 0.5)), k, kw.get("lam", 0.7), 100, ex)
        if name == "random":
            rng = np.random.RandomState(i); col = self.d["colour"] == self.d["colour"][i]; return rng.choice(np.where(col)[0][np.where(col)[0] != i], k, replace=False)
        raise ValueError(name)


TECH = ["random", "raw", "zw", "latent", "emb", "hybrid", "rrf", "mmr"]


def evaluate(nq=3000, k=10):
    R = Rec(); d = R.d; rng = np.random.RandomState(11); Q = rng.choice(R.n, nq, replace=False); out = {}; lists = {}
    price, pts = d["price"].astype(float), d["points"].astype(float)
    for t in TECH:
        L = np.array([R.technique(t, int(i), k) for i in Q]); lists[t] = L
        grp = (d["group"][L] == d["group"][Q][:, None]).mean(); cty = (d["country"][L] == d["country"][Q][:, None]).mean(); prv = (d["province"][L] == d["province"][Q][:, None]).mean()
        win = (d["winery"][L] == d["winery"][Q][:, None]).mean(); dp = np.abs(pts[L] - pts[Q][:, None]).mean()
        pr = np.nanmedian(price[L] / price[Q][:, None]); E = R.M["emb"]
        ild = np.mean([1 - ((E[l] @ E[l].T).sum() - k) / (k * (k - 1)) for l in L]); nct = np.mean([len(set(d["country"][l])) for l in L]); nwi = np.mean([len(set(d["winery"][l])) for l in L])
        cnt = np.bincount(L.ravel(), minlength=R.n); out[t] = dict(group=float(grp), country=float(cty), province=float(prv), winery=float(win), dpoints=float(dp), price_ratio=float(pr), ild=float(ild), countries=float(nct), wineries=float(nwi),
                                                                   coverage=float((cnt > 0).mean()), hub_max=int(cnt.max()), never=float((cnt == 0).mean()), hub_top1pct_share=float(np.sort(cnt)[::-1][:R.n // 100].sum() / cnt.sum()))
        print(f"{t:8s} grape {grp:.3f} country {cty:.3f} province {prv:.3f} winery {win:.3f} dpts {dp:.2f} ild {ild:.3f} coverage {out[t]['coverage']:.3f} hub_max {cnt.max()}")
    J = {a: {b: float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(lists[a], lists[b])])) for b in TECH if b != "random"} for a in TECH if a != "random"}
    json.dump(dict(k=k, queries=nq, n_items=R.n, techniques=out, jaccard=J), open(DASH / "recsys_eval.json", "w"), indent=1); print("wrote recsys_eval.json")


def fixture(n=40, k=10):
    R = Rec(); rng = np.random.RandomState(3); rows = []
    for i in rng.choice(R.n, n, replace=False):
        r = dict(i=int(i))
        for t in ("raw", "zw", "latent", "emb", "hybrid", "rrf", "mmr"): r[t] = [int(x) for x in R.technique(t, int(i), k)]
        rows.append(r)
    json.dump(rows, open(ROOT / "outputs/recsys_parity.json", "w")); print("fixture", len(rows))


if __name__ == "__main__":
    {"build": build, "eval": evaluate, "fixture": fixture}[sys.argv[1]]()
