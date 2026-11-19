"""Guess model (LITE tier): from the description only, guess grape, place, producer, vintage, price (+ critic points).
Deterministic: fixed seeds, no sampling at inference. Features = SAT vector (guess_features.py).
  grape     softmax regression over variety classes (>= MIN_VAR reviews, else '__other__'); group = roll-up
  place     softmax over place classes (region where there are >= MIN_PLACE reviews, else province, else country), stacked on grape probabilities
  producer  P(winery | place, grape) tables x a style kernel over SAT principal components, wineries with >= MIN_WINERY reviews + 'other'
  vintage / price(log) / points   ridge on [features + log grape probs + log country probs]; intervals from out-of-fold residual quantiles
Usage: guess_model.py dev [N]      train on 4/5 of the pool, test on the held-out fifth (development numbers, NOT a blind round)"""
import sys, json, time, math, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon
from guess_features import build_spec, matrix_from_vectors

MIN_VAR, MIN_PLACE, MIN_WINERY, NPC = 150, 80, 20, 40
C_GRAPE, C_PLACE = 0.3, 0.3
rng_seed = 7


def fold_of(ids, k=5):
    return np.array([int(i[:8], 16) % k for i in ids])


def corpus(tier="lite"):
    """tier 'lite' = SAT features (browser-exportable); 'full' = SAT + MiniLM embedding (outputs/emb_minilm_all.npy, same row order)."""
    vec = pd.read_parquet(ROOT / "outputs/wset_vectors_all_nocolour.parquet")      # description-only features: matcher run WITHOUT the wine-colour label
    nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet").set_index("id").loc[vec.id]
    lex = Lexicon(); sp = ROOT / "outputs/guess_spec.json"
    spec = json.load(open(sp)) if sp.exists() else build_spec(vec, lex)
    X = matrix_from_vectors(vec, spec)
    if tier == "full":
        E = np.load(ROOT / "outputs/emb_minilm_all.npy").astype(np.float32); assert len(E) == len(X), "embedding rows do not match the corpus"
        X = np.hstack([X, E]); spec = dict(spec, names=spec["names"] + [f"emb_{i}" for i in range(E.shape[1])])
    return vec.id.values, X, spec, nm


def targets(nm):
    t = pd.DataFrame(index=nm.index)
    vc = nm.variety_norm.value_counts(); keep = set(vc[vc >= MIN_VAR].index)
    t["variety"] = np.where(nm.variety_norm.isin(keep), nm.variety_norm, "__other__")
    t["group"] = nm.variety_group.astype(str)
    prov = nm.country.astype(str) + "|" + nm.province.astype(str); trip = prov + "|" + nm.region_1.fillna("").astype(str)
    ct, pt = trip.map(trip.value_counts()), prov.map(prov.value_counts())
    t["place"] = np.where((nm.region_1.notna()) & (ct >= MIN_PLACE), trip, np.where(pt >= MIN_PLACE, prov + "|", nm.country.astype(str) + "||"))
    t["country"] = nm.country.astype(str); t["province"] = prov; t["triple"] = trip
    t["winery"] = nm.winery.astype(str); t["vintage"] = nm.vintage.astype(float); t["price"] = nm.price.astype(float); t["points"] = nm.points.astype(float)
    return t


def place_coords(t):
    from places import load_places
    pl = load_places().dropna(subset=["lat"]); pl["triple"] = pl.country.astype(str) + "|" + pl.province.astype(str) + "|" + pl.region_1.fillna("").astype(str)
    xy = pl.set_index("triple")[["lat", "lon"]]; xy = xy[~xy.index.duplicated()]
    t2 = t[["place", "triple"]].join(xy, on="triple")
    cls = t2.groupby("place")[["lat", "lon"]].mean()
    return xy, cls


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def softmax_T(logp, T):
    z = logp / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_T(P, y_idx):
    best = (1e9, 1.0); L = np.log(np.clip(P, 1e-9, 1))
    for T in (0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0):
        Q = softmax_T(L, T); ll = -np.mean(np.log(np.clip(Q[np.arange(len(y_idx)), y_idx], 1e-9, 1)))
        if ll < best[0]: best = (ll, T)
    return best[1]


def lr(C, seed=rng_seed):
    return LogisticRegression(C=C, max_iter=150, tol=1e-3, random_state=seed)


def oof_proba(Z, y, folds, C, log=print, name=""):
    """Out-of-fold class probabilities (columns = sorted classes of y). Each fold may miss rare classes -> mapped back."""
    classes = np.unique(y); out = np.zeros((len(y), len(classes)), dtype=np.float32)
    for k in range(folds):
        te = np.arange(len(y)) % folds == k; m = lr(C).fit(Z[~te], y[~te]); p = m.predict_proba(Z[te])
        out[np.ix_(te, np.searchsorted(classes, m.classes_))] = p
        log(f"  {name} oof fold {k + 1}/{folds}")
    return out, classes


def fit(X, t, idx, log=print):
    """Train on rows idx. Returns a model dict (numpy / small objects only)."""
    t0 = time.time(); M = dict(version="lite-1" if X.shape[1] < 400 else "full-1")
    mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6; Z = (X - mu) / sd; M.update(mu=mu, sd=sd)
    tr = t.iloc[idx]; Zt = Z[idx]
    # --- grape (variety) with OOF probabilities for stacking
    yv = tr.variety.values; P_oof_v, vcls = oof_proba(Zt, yv, 4, C_GRAPE, log, "variety")
    gm = lr(C_GRAPE).fit(Zt, yv); assert list(gm.classes_) == list(vcls)
    v2g = tr.groupby("variety").group.agg(lambda s: s.value_counts().index[0]).to_dict(); v2g["__other__"] = "Other"
    groups = sorted(set(v2g[c] for c in vcls)); G = np.zeros((len(vcls), len(groups)), dtype=np.float32)
    for i, c in enumerate(vcls): G[i, groups.index(v2g[c])] = 1
    Tg = fit_T(P_oof_v, np.searchsorted(vcls, yv)); log(f"  grape temperature {Tg}")
    M.update(var_classes=vcls, var_W=gm.coef_.astype(np.float32), var_b=gm.intercept_.astype(np.float32), T_var=Tg, groups=groups, G=G)
    Pg_oof = softmax_T(np.log(np.clip(P_oof_v, 1e-9, 1)), Tg) @ G                       # group probs, out-of-fold
    # --- country OOF for stacking
    yc = tr.country.values; P_oof_c, ccls = oof_proba(Zt, yc, 4, C_GRAPE, log, "country"); M["country_classes"] = ccls
    cm = lr(C_GRAPE).fit(Zt, yc); M.update(cty_W=cm.coef_.astype(np.float32), cty_b=cm.intercept_.astype(np.float32))
    S = lambda Pg, Pc: np.hstack([np.log(np.clip(Pg, 1e-4, 1)), np.log(np.clip(Pc, 1e-4, 1))]).astype(np.float32)
    Stack_tr = S(Pg_oof, P_oof_c)
    # --- place (stacked), temperature from a 90/10 split
    ypl = tr.place.values; pcls = np.unique(ypl); Zp = np.hstack([Zt, Stack_tr])
    h = (np.arange(len(ypl)) % 10) == 0
    pm0 = lr(C_PLACE).fit(Zp[~h], ypl[~h]); Pp = np.zeros((h.sum(), len(pcls)), dtype=np.float32)
    Pp[:, np.searchsorted(pcls, pm0.classes_)] = pm0.predict_proba(Zp[h]); Tp = fit_T(Pp, np.searchsorted(pcls, ypl[h])); log(f"  place temperature {Tp}")
    pm = lr(C_PLACE).fit(Zp, ypl); pc_aligned = np.searchsorted(pcls, pm.classes_)
    Wp = np.zeros((len(pcls), Zp.shape[1]), dtype=np.float32); bp = np.full(len(pcls), -30, dtype=np.float32); Wp[pc_aligned] = pm.coef_; bp[pc_aligned] = pm.intercept_
    M.update(place_classes=pcls, place_W=Wp, place_b=bp, T_place=Tp)
    xy, cls_xy = place_coords(t.iloc[idx]); M["place_xy"] = cls_xy.reindex(pcls).values.astype(np.float32)
    # --- ridge regressors with residual-quantile intervals
    reg = {}
    for name, f in (("vintage", lambda s: s), ("price", np.log), ("points", lambda s: s)):
        y = tr[name].values.astype(float); ok = ~np.isnan(y)
        if name == "vintage": ok &= (y >= 1990) & (y <= 2017)
        y = f(y); Xr = np.hstack([Zt, Stack_tr])[ok]; yy = y[ok]
        oof = np.zeros(len(yy));
        for k in range(4):
            te = np.arange(len(yy)) % 4 == k; oof[te] = Ridge(alpha=30.0).fit(Xr[~te], yy[~te]).predict(Xr[te])
        r = Ridge(alpha=30.0).fit(Xr, yy); edges = np.quantile(oof, [1 / 3, 2 / 3]); b = np.digitize(oof, edges)
        q = np.array([np.quantile((yy - oof)[b == i], [0.1, 0.5, 0.9]) for i in range(3)])
        reg[name] = dict(W=r.coef_.astype(np.float32), b=float(r.intercept_), edges=edges, q=q)
    M["reg"] = reg
    # --- producer: P(w | place, grape) tables + style kernel
    wc = tr.winery.value_counts(); W = list(wc[wc >= MIN_WINERY].index); wix = {w: i for i, w in enumerate(W)}
    pg = pd.DataFrame(dict(w=tr.winery.map(wix), p=np.searchsorted(pcls, tr.place.values), g=[list(vcls).index(v) if v in set(vcls) else -1 for v in tr.variety.values])).dropna()
    pg = pg[pg.g >= 0]; combo = pg.groupby(["p", "g"]).size().rename("n_pg"); cw = pg.dropna().groupby(["p", "g", "w"]).size().rename("n_wpg").reset_index().join(combo, on=["p", "g"])
    keys = {k: i for i, k in enumerate(combo.index)}; rows = [keys[(p, g)] for p, g in zip(cw.p, cw.g)]
    Mw = sparse.csr_matrix(((cw.n_wpg / cw.n_pg).values.astype(np.float32), (rows, cw.w.astype(int).values)), shape=(len(keys), len(W)))
    pca = PCA(NPC, random_state=1).fit(Zt); Q = pca.transform(Zt); wi = tr.winery.map(wix).values; ok = ~pd.isna(wi)
    mu_w = np.zeros((len(W), NPC), dtype=np.float32); np.add.at(mu_w, wi[ok].astype(int), Q[ok]); mu_w /= np.maximum(1, np.bincount(wi[ok].astype(int), minlength=len(W)))[:, None]
    M.update(wineries=np.array(W), combo=np.array(list(keys.keys())), Mw=Mw, pca_mean=pca.mean_.astype(np.float32), pca_comp=pca.components_.astype(np.float32), mu_w=mu_w,
             tau=8.0, p_other=float((~tr.winery.isin(W)).mean()))
    log(f"  fitted in {time.time() - t0:.0f}s: {len(vcls)} varieties, {len(pcls)} places, {len(W)} wineries")
    return M


def _lin(Z, W, b): return Z @ W.T + b


def predict(M, X, tau=None):
    Z = (X - M["mu"]) / M["sd"]; out = {}
    Lv = _lin(Z, M["var_W"], M["var_b"]); Pv = softmax_T(Lv, M["T_var"]); Pg = Pv @ M["G"]
    Pc = softmax_T(_lin(Z, M["cty_W"], M["cty_b"]), 1.0)
    Stack = np.hstack([np.log(np.clip(Pg, 1e-4, 1)), np.log(np.clip(Pc, 1e-4, 1))]).astype(np.float32); Zp = np.hstack([Z, Stack])
    Pp = softmax_T(_lin(Zp, M["place_W"], M["place_b"]), M["T_place"])
    out.update(Pv=Pv, Pg=Pg, Pp=Pp)
    for name, r in M["reg"].items():
        pr = Zp @ r["W"] + r["b"]; b = np.digitize(pr, r["edges"]); q = r["q"][b]
        out[name] = np.c_[pr + q[:, 0], pr + q[:, 1], pr + q[:, 2]]               # lo10, median, hi90 (log scale for price)
    # producer
    W = M["wineries"]; tau = tau or M["tau"]; Q = (Z - M["pca_mean"]) @ M["pca_comp"].T; Mw = M["Mw"]; cp, cg = M["combo"][:, 0], M["combo"][:, 1]
    Pw = np.zeros((len(Z), len(W)), dtype=np.float32)
    for s in range(0, len(Z), 1500):
        sl = slice(s, s + 1500); A = Pp[sl][:, cp] * Pv[sl][:, cg]; sc = (sparse.csr_matrix(A) @ Mw).toarray()
        d2 = (Q[sl] ** 2).sum(1)[:, None] - 2 * Q[sl] @ M["mu_w"].T + (M["mu_w"] ** 2).sum(1)[None]
        sc = sc * np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * tau ** 2)); Pw[sl] = sc / np.maximum(sc.sum(1, keepdims=True), 1e-12) * (1 - M["p_other"])
    out["Pw"] = Pw
    return out


def top(P, classes, k=5):
    idx = np.argsort(-P, 1)[:, :k]; return [[(classes[j], float(P[i, j])) for j in idx[i]] for i in range(len(P))]


def place_point(M, Pp, i):
    """Geographic median among candidate classes (point that minimises expected distance) + radii holding 50/80% of the mass."""
    xy = M["place_xy"]; ok = ~np.isnan(xy[:, 0]); p = Pp[i] * ok; p = p / max(p.sum(), 1e-9)
    cand = np.argsort(-p)[:15]; D = haversine(xy[cand, 0][:, None], xy[cand, 1][:, None], xy[:, 0][None], xy[:, 1][None])
    D = np.nan_to_num(D, nan=0)
    best = cand[np.argmin((D * p[None]).sum(1))]; d = haversine(xy[best, 0], xy[best, 1], xy[:, 0], xy[:, 1]); d[~ok] = 1e9
    o = np.argsort(d); cum = np.cumsum(p[o]); r50, r80 = d[o][np.searchsorted(cum, 0.5)], d[o][min(len(o) - 1, np.searchsorted(cum, 0.8))]
    return float(xy[best, 0]), float(xy[best, 1]), float(r50), float(r80)


def score(M, out, t, idx_te, cls_xy=None, xy_truth=None):
    """Metrics against truth for rows idx_te (positions in t). Returns dict."""
    T = t.iloc[idx_te]; R = {}
    vcls, pcls = M["var_classes"], M["place_classes"]
    yv = T.variety.values; tp1 = vcls[out["Pv"].argmax(1)]; R["variety_top1"] = float(np.mean(tp1 == yv))
    R["variety_top3"] = float(np.mean([y in set(vcls[np.argsort(-p)[:3]]) for y, p in zip(yv, out["Pv"])]))
    known = yv != "__other__"; R["variety_top1_known"] = float(np.mean(tp1[known] == yv[known]))
    gcls = np.array(M["groups"]); yg = T.group.values; R["group_top1"] = float(np.mean(gcls[out["Pg"].argmax(1)] == yg))
    R["group_top3"] = float(np.mean([y in set(gcls[np.argsort(-p)[:3]]) for y, p in zip(yg, out["Pg"])]))
    pl = pcls[out["Pp"].argmax(1)]; R["place_top1"] = float(np.mean(pl == T.place.values))
    R["place_top3"] = float(np.mean([y in set(pcls[np.argsort(-p)[:3]]) for y, p in zip(T.place.values, out["Pp"])]))
    ctry = np.array([c.split("|")[0] for c in pcls]); prov = np.array([c.split("|")[0] + "|" + c.split("|")[1] for c in pcls])
    for lab, arr, truth in (("country", ctry, T.country.values), ("province", prov, T.province.values)):
        u = np.unique(arr); Pm = np.zeros((len(out["Pp"]), len(u)), dtype=np.float32)
        for j, a in enumerate(arr): Pm[:, np.searchsorted(u, a)] += out["Pp"][:, j]
        R[lab + "_top1"] = float(np.mean(u[Pm.argmax(1)] == truth)); R[lab + "_top3"] = float(np.mean([y in set(u[np.argsort(-p)[:3]]) for y, p in zip(truth, Pm)]))
    # km error + interval coverage
    if xy_truth is not None:
        pts = [place_point(M, out["Pp"], i) for i in range(len(idx_te))]
        tx = xy_truth.reindex(T.triple.values).values; ok = ~np.isnan(tx[:, 0]); la, lo, r50, r80 = np.array(pts).T
        d = haversine(la[ok], lo[ok], tx[ok, 0], tx[ok, 1]); R["km_median"] = float(np.median(d)); R["km_within_100"] = float(np.mean(d <= 100)); R["km_within_500"] = float(np.mean(d <= 500))
        R["cover50"] = float(np.mean(d <= r50[ok])); R["cover80"] = float(np.mean(d <= r80[ok]))
    # producer
    W = M["wineries"]; yw = T.winery.values; inW = np.isin(yw, W)
    top5 = np.argsort(-out["Pw"], 1)[:, :5]; hit1 = np.array([W[top5[i, 0]] == yw[i] for i in range(len(yw))]); hit5 = np.array([yw[i] in set(W[top5[i]]) for i in range(len(yw))])
    R["winery_top1_all"] = float(hit1.mean()); R["winery_top5_all"] = float(hit5.mean()); R["winery_top1_in_set"] = float(hit1[inW].mean()); R["winery_top5_in_set"] = float(hit5[inW].mean()); R["winery_in_set_share"] = float(inW.mean())
    # vintage / price / points
    for name in ("vintage", "price", "points"):
        y = T[name].values.astype(float); ok = ~np.isnan(y)
        if name == "vintage": ok &= (y >= 1990) & (y <= 2017)
        y = np.log(y) if name == "price" else y; o = out[name][ok]; yy = y[ok]
        base = np.median(y[~np.isnan(y)]); R[name + "_n"] = int(ok.sum())
        R[name + "_mae"] = float(np.mean(np.abs(o[:, 1] - yy))); R[name + "_mae_baseline"] = float(np.mean(np.abs(base - yy)))
        R[name + "_skill"] = 1 - R[name + "_mae"] / R[name + "_mae_baseline"]; R[name + "_cover80"] = float(np.mean((yy >= o[:, 0]) & (yy <= o[:, 2])))
        if name == "vintage": R["vintage_within1"] = float(np.mean(np.abs(o[:, 1] - yy) <= 1))
        if name == "price": R["price_within25pct"] = float(np.mean(np.abs(o[:, 1] - yy) <= math.log(1.25)))
    return R


def dev(n=None, tier="lite"):
    ids, X, spec, nm = corpus(tier); t = targets(nm); f = fold_of(ids)
    if n: sel = np.arange(len(ids))[:n]; ids, X, t, f = ids[sel], X[sel], t.iloc[sel], f[sel]; nm = nm.iloc[sel]
    tr, te = np.where(f != 0)[0], np.where(f == 0)[0]
    print(f"dev split: train {len(tr)}, test {len(te)}")
    M = fit(X, t, tr)
    xy, _ = place_coords(t); res = {}
    out = predict(M, X[te]); R = score(M, out, t, te, xy_truth=xy[["lat", "lon"]])
    R["tau"] = M["tau"]; print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in R.items()}, indent=1))
    pickle.dump(M, open(ROOT / f"outputs/guess_{tier}_dev.pkl", "wb")); json.dump(R, open(ROOT / f"outputs/guess_dev_scores_{tier}.json", "w"), indent=1)
    return M, R


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        dev(int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None, "full" if "full" in sys.argv else "lite")
