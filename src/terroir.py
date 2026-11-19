"""Terroir analysis -> dashboard/data/terroir.json  (no review text, only aggregates).
 A  what explains the glass (adjusted R2 of grape / place / producer / vintage / weather on the WSET profile and on text embeddings)
 B  climate response curves (within place x grape: warm/cool season anomaly vs attribute, cluster-bootstrap CI)
 C  vintage explorer per place x grape + 'what a warm year tastes like' (PLS axis, leave-vintage-out CV)
 D  place taste profiles, projections (PCA/UMAP, WSET and embedding, raw and net-of-grape), colours, geography<->taste distance decay
Usage: terroir.py"""
import sys, json, glob, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon
from weather import season_metrics, cell_centre, loc_key, CACHE
from places import load_places

rng = np.random.RandomState(7)
O = ROOT / "outputs"
YEARS = range(2000, 2018)

# ------------------------------------------------------------------ 1. climate: every season 2000-2017 for every cached cell
cf = O / "climate_all_seasons.parquet"
if not cf.exists():
    rows = []
    for f in sorted(glob.glob(str(CACHE / "*.json"))):
        key = Path(f).stem; d = json.load(open(f)); lat = cell_centre(key)[0]
        for y in YEARS:
            m = season_metrics(d, lat, y)
            if m: rows.append(dict(loc=key, vintage=y, **m))
    pd.DataFrame(rows).to_parquet(cf); print("climate rows", len(rows))
cl = pd.read_parquet(cf)
for c in ("gdd", "rain_season", "diurnal", "days_gt35", "rain_harvest"):
    g = cl.groupby("loc")[c]; cl["z_" + c] = ((cl[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).clip(-3, 3)
site = cl.groupby("loc").agg(site_gdd=("gdd", "mean"), site_rain=("rain_season", "mean"), site_diurnal=("diurnal", "mean")).reset_index()
cl = cl.merge(site, on="loc")

# ------------------------------------------------------------------ 2. reviews + vectors + places
nm = pd.read_parquet(O / "reviews_norm.parquet")
vec = pd.read_parquet(O / "wset_vectors_all.parquet")
pl = load_places().dropna(subset=["lat"]).copy()
pl["loc"] = [loc_key(a, b) for a, b in zip(pl.lat, pl.lon)]
pl = pl[pl.geo_level.isin(["region_1", "region_1_alt"])]
d = nm.merge(vec, on="id").merge(pl[["country", "province", "region_1", "loc", "lat", "lon"]], on=["country", "province", "region_1"], how="inner")
d["place"] = d.region_1.astype(str) + " · " + d.province.astype(str) + ", " + d.country.astype(str)
d["grape"] = d.variety_group.astype(str)
d["grp"] = d.place + "|" + d.grape
d["ripe_state"] = d.fruit_ripeness
for s in ("ripe_fruit", "dried_fruit", "cooked_fruit", "unripe_fruit"):
    d["r_" + s] = (d.ripe_state == s).astype(float)
print("reviews with region-level coordinates:", len(d), "| places", d.place.nunique())

DESC = [c for c in vec if c.startswith("a_")]
SCALES = [c for c in vec if c.startswith("s_") and c not in ("s_development", "s_mousse", "s_texture")]
ATTR = {"points": "Critic points", "s_acidity": "Acidity (1–5)", "s_body": "Body (1–5)", "s_tannin": "Tannin (1–5)", "s_alcohol": "Alcohol (1–3)", "s_sweetness": "Sweetness (1–6)",
        "c_citrus_fruit": "Citrus fruit", "c_green_fruit": "Green fruit", "c_stone_fruit": "Stone fruit", "c_tropical_fruit": "Tropical fruit", "c_red_fruit": "Red fruit", "c_black_fruit": "Black fruit",
        "r_unripe_fruit": "Unripe fruit", "r_ripe_fruit": "Ripe fruit", "r_dried_fruit": "Dried fruit", "r_cooked_fruit": "Cooked fruit",
        "c_oak": "Oak", "c_herbaceous": "Herbaceous", "c_floral": "Floral", "c_spice": "Spice", "t_tertiary": "Tertiary notes"}
RATE = {k for k in ATTR if k[0] in "crt"}                    # shares of notes (0-1); the rest are scale/score means over notes that state them
W = d.merge(cl, on=["loc", "vintage"], how="inner")
print("with weather:", len(W), "| place x grape x vintage units:", W.groupby(["grp", "vintage"]).ngroups)

# ------------------------------------------------------------------ A. what explains the glass
def std(X):
    X = X.loc[:, X.std() > 0]; X = X.fillna(X.mean()); return (X - X.mean()) / X.std()


def adj_r2(X, g):
    g = pd.Series(g).fillna("NA").values; n = len(X); k = len(set(g)); tot = float((X.values ** 2).sum())
    ssb = float((X.groupby(g).transform("mean").values ** 2).sum()); r2 = ssb / tot
    return r2, 1 - (1 - r2) * (n - 1) / max(1, n - k), k


Xw = std(vec.set_index("id")[DESC + SCALES].astype(float)); full = nm.merge(pd.DataFrame(index=Xw.index), left_on="id", right_index=True).set_index("id").loc[Xw.index]
full["place"] = full.region_1.astype(str) + "|" + full.province.astype(str) + "|" + full.country.astype(str); full["grape"] = full.variety_group.astype(str)
full["pg"] = full.place + "|" + full.grape
fac = [("Grape", "grape"), ("Place (region)", "place"), ("Producer (winery)", "winery"), ("Place × grape", "pg"), ("Source / publication", "sources"), ("Vintage year", "vintage"), ("Price band", None), ("Critic score band", None)]
full["priceb"] = pd.qcut(full.price, 5, duplicates="drop").astype(str); full["ptsb"] = pd.cut(full.points, [0, 84, 87, 90, 100]).astype(str)
fac = [(a, "priceb" if a == "Price band" else "ptsb" if a == "Critic score band" else b) for a, b in fac]
var_w = []
for name, col in fac:
    r, a, k = adj_r2(Xw, full[col].astype(str)); var_w.append(dict(factor=name, r2=round(r, 4), adj=round(max(a, 0), 4), groups=k))
# weather: vintage warmth/wetness within place x grape (regression on within-group anomalies)
Xr = std(W.set_index("id")[DESC + SCALES].astype(float)); Wd = W.set_index("id")
Xr = Xr - Xr.groupby(Wd.grp.values).transform("mean")
Z = pd.DataFrame({"zg": Wd.z_gdd, "zr": Wd.z_rain_season, "zd": Wd.z_diurnal}); Z = Z - Z.groupby(Wd.grp.values).transform("mean"); Z["zg2"] = Z.zg ** 2; Z["zgr"] = Z.zg * Z.zr
Z = Z.fillna(0); beta = np.linalg.lstsq(Z.values, Xr.values, rcond=None)[0]; sse = float(((Xr.values - Z.values @ beta) ** 2).sum()); sst = float((Xr.values ** 2).sum())
r2w = 1 - sse / sst; p = Z.shape[1]; adjw = 1 - (1 - r2w) * (len(Xr) - 1) / (len(Xr) - p - 1)
var_w.append(dict(factor="Season weather (within a place)", r2=round(r2w, 5), adj=round(max(adjw, 0), 5), groups=p))
# embeddings (20k sample): same factors
emb = np.load(O / "emb_minilm_20000.npy"); ids20 = pd.read_parquet(O / "wset_vectors_20000.parquet", columns=["id"]).id.values
E = pd.DataFrame(emb, index=ids20); E = E.loc[:, E.std() > 0]; E = (E - E.mean()) / E.std(); e20 = full.loc[E.index]
var_e = []
for name, col in fac:
    r, a, k = adj_r2(E, e20[col].astype(str)); var_e.append(dict(factor=name, r2=round(r, 4), adj=round(max(a, 0), 4), groups=k))
print("A done:", [(v["factor"], v["adj"]) for v in var_w])

# ------------------------------------------------------------------ units (place x grape x vintage) and within-group demeaning
def units(df, attrs, extra):
    g = df.groupby(["grp", "vintage"]); U = g.agg(k=("id", "size"), **{c: (c, "first") for c in extra}).reset_index()
    for a in attrs:
        m = g[a].mean(); n = g[a].count()
        U[a] = U.set_index(["grp", "vintage"]).index.map(m).values; U["n_" + a] = U.set_index(["grp", "vintage"]).index.map(n).values
    return U


extra = ["place", "grape", "country", "lat", "lon", "z_gdd", "z_rain_season", "z_diurnal", "z_days_gt35", "gdd", "site_gdd", "site_rain", "site_diurnal"]
U = units(W, list(ATTR), extra)
nv = U[U.k >= 3].groupby("grp").vintage.transform("nunique"); U = U[(U.k >= 3)]; U = U[nv.reindex(U.index).values >= 4].copy()
for a in ATTR:                                                     # weighted within-group demeaning of the outcome and of the anomalies
    w = U["n_" + a].fillna(0).values; U["w_" + a] = w
    sw = U.assign(_w=w, _y=np.where(w > 0, U[a].fillna(0), 0) * w).groupby("grp")[["_w", "_y"]].transform("sum"); U["dm_" + a] = U[a] - sw._y / sw._w.replace(0, np.nan)
for z in ("z_gdd", "z_rain_season", "z_diurnal"):
    U["dm_" + z] = U[z] - U.groupby("grp")[z].transform("mean")
print("units used:", len(U), "| groups:", U.grp.nunique(), "| grapes:", U.grape.nunique())

# ------------------------------------------------------------------ B. climate response curves
BIN = [-3.01, -1.0, -0.35, 0.35, 1.0, 3.01]; BN = ["much cooler", "cooler", "typical", "warmer", "much warmer"]


def curve(df, a, xcol="z_gdd", reps=300):
    s = df[df["w_" + a] > 0]
    if s.grp.nunique() < 4 or len(s) < 20: return None
    s = s.assign(b=pd.cut(s[xcol], BIN, labels=False), y=s["dm_" + a], w=s["w_" + a], x=s["dm_" + xcol])
    gs = sorted(s.grp.unique()); gi = {g: i for i, g in enumerate(gs)}; s["gi"] = s.grp.map(gi); G = len(gs)
    Nb = np.zeros((G, 5)); Db = np.zeros((G, 5)); S = np.zeros((G, 5))      # per group & bin: sum w*y, sum w ; sums for slope
    for (g, b), t in s.groupby(["gi", "b"]): Nb[g, int(b)] = (t.w * t.y).sum(); Db[g, int(b)] = t.w.sum()
    for g, t in s.groupby("gi"): S[g] = [t.w.sum(), (t.w * t.x).sum(), (t.w * t.y).sum(), (t.w * t.x * t.x).sum(), (t.w * t.x * t.y).sum()]
    def stat(c):
        num, den = c @ Nb, c @ Db; sw, swx, swy, swxx, swxy = c @ S
        slope = (swxy - swx * swy / sw) / (swxx - swx ** 2 / sw) if sw > 0 and swxx - swx ** 2 / sw > 1e-9 else np.nan
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), slope
    est, sl = stat(np.ones(G)); boots = [stat(np.bincount(rng.randint(0, G, G), minlength=G).astype(float)) for _ in range(reps)]
    B = np.array([b[0] for b in boots]); SL = np.array([b[1] for b in boots])
    lo, hi = np.nanpercentile(B, 5, axis=0), np.nanpercentile(B, 95, axis=0)
    n_units = [int((s.b == i).sum()) for i in range(5)]
    return dict(m=[None if np.isnan(x) else round(float(x), 4) for x in est], lo=[None if np.isnan(x) else round(float(x), 4) for x in lo], hi=[None if np.isnan(x) else round(float(x), 4) for x in hi],
                n=n_units, slope=None if np.isnan(sl) else round(float(sl), 4), slo=round(float(np.nanpercentile(SL, 5)), 4), shi=round(float(np.nanpercentile(SL, 95)), 4), groups=G)


cnt = U.groupby("grape").grp.nunique().sort_values(ascending=False); grapes = [g for g in cnt.index if cnt[g] >= 12][:28]
curves = {}
for g in ["ALL"] + grapes:
    s = U if g == "ALL" else U[U.grape == g]; curves[g] = {}
    for xcol in ("z_gdd", "z_rain_season"):
        r = {a: curve(s, a, xcol, 200 if g == "ALL" else 150) for a in ATTR}
        curves[g][xcol] = {a: v for a, v in r.items() if v}
print("B done; grapes", len(grapes))

# site climate (between places, confounded): place-mean of attribute vs the place's mean season warmth
site_rows = []
for g in ["ALL"] + grapes:
    s = W if g == "ALL" else W[W.grape == g]
    pm = s.groupby("place").agg(n=("id", "size"), gdd=("site_gdd", "first"), **{a: (a, "mean") for a in ATTR}).query("n >= 25")
    if len(pm) < 8: continue
    site_rows.append(dict(grape=g, places=len(pm), gdd=pm.gdd.round(0).tolist(), n=pm.n.tolist(), names=pm.index.tolist() if g != "ALL" else [],
                          **{a: pm[a].round(3).where(pm[a].notna(), None).tolist() for a in ATTR}))

# ------------------------------------------------------------------ C. what a warm year tastes like (multivariate PLS) + vintage explorer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
cols = [c for c in DESC if not c.endswith("_generic")]
Ux = W.groupby(["grp", "vintage"])[cols].mean(); kk = W.groupby(["grp", "vintage"]).size()
Ux = Ux[kk.reindex(Ux.index) >= 5]; Uz = U.set_index(["grp", "vintage"]).loc[lambda t: ~t.index.duplicated()]
Ux = Ux.loc[Ux.index.intersection(Uz.index)]; Ux = Ux.loc[:, Ux.mean() > 0.02]
gi = Ux.index.get_level_values("grp").values; Xd = Ux - Ux.groupby(gi).transform("mean")
pls_out = {}
for zc, lab in (("z_gdd", "warm"), ("z_rain_season", "wet")):
    y = Uz.loc[Ux.index, zc]; y = (y - y.groupby(gi).transform("mean")).values; keep = Xd.std().values > 0
    Xs = (Xd.values[:, keep]) / Xd.values[:, keep].std(0); yrs = Ux.index.get_level_values("vintage").values
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xs, y, yrs):
        m = PLSRegression(1, scale=False).fit(Xs[tr], y[tr]); pred[te] = m.predict(Xs[te]).ravel()
    r2 = 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()
    m = PLSRegression(1, scale=False).fit(Xs, y); t = m.x_scores_.ravel()
    cor = np.array([np.corrcoef(Xs[:, j], t)[0, 1] for j in range(Xs.shape[1])]); names = np.array(Xd.columns)[keep]
    order = np.argsort(cor)
    pick = lambda idx: [dict(descriptor=names[i][2:].replace("__", ": ").replace("_", " "), r=round(float(cor[i]), 3)) for i in idx]
    pls_out[lab] = dict(cv_r2=round(float(r2), 4), n_units=int(len(y)), up=pick(order[::-1][:10]), down=pick(order[:10]))
    print(f"PLS {lab}: leave-vintage-out R2 {r2:.3f} on {len(y)} units")

# explorer: place x grape with enough vintages
EX = ["points", "s_acidity", "s_body", "s_alcohol", "c_citrus_fruit", "c_stone_fruit", "c_tropical_fruit", "c_red_fruit", "c_black_fruit", "r_ripe_fruit", "r_cooked_fruit", "c_oak", "c_herbaceous"]
Ua = units(W, EX, ["place", "grape", "country", "z_gdd", "z_rain_season", "gdd", "rain_season", "lat", "lon"]); Ua = Ua[Ua.k >= 5]
sz = Ua.groupby("grp").agg(v=("vintage", "nunique"), k=("k", "sum")); good = sz[(sz.v >= 6) & (sz.k >= 150)].index
explorer = []
for g in good:
    t = Ua[Ua.grp == g].sort_values("vintage"); row = t.iloc[0]
    wmask, cmask = t.z_gdd >= 0.4, t.z_gdd <= -0.4
    contrast = {}
    if wmask.sum() >= 2 and cmask.sum() >= 2:
        for a in EX:
            u = t[a];
            if u.notna().sum() < 6: continue
            diff = u[wmask].mean() - u[cmask].mean(); bs = []
            for _ in range(300):
                wi = rng.choice(np.where(wmask)[0], wmask.sum()); ci = rng.choice(np.where(cmask)[0], cmask.sum()); bs.append(np.nanmean(u.values[wi]) - np.nanmean(u.values[ci]))
            contrast[a] = [round(float(diff), 3), round(float(np.nanpercentile(bs, 5)), 3), round(float(np.nanpercentile(bs, 95)), 3)]
    explorer.append(dict(place=row.place, grape=row.grape, country=row.country, lat=round(float(row.lat), 2), lon=round(float(row.lon), 2), reviews=int(t.k.sum()), warm=int(wmask.sum()), cool=int(cmask.sum()),
                         vint=[[int(r.vintage), round(float(r.z_gdd), 2), round(float(r.z_rain_season), 2), int(r.k)] + [None if pd.isna(r[a]) else round(float(r[a]), 3) for a in EX] for _, r in t.iterrows()], contrast=contrast))
explorer.sort(key=lambda e: -e["reviews"]); print("explorer entries", len(explorer))

# ------------------------------------------------------------------ D. place taste profiles, projections, geography <-> taste
import umap
from sklearn.decomposition import PCA
Xa = Xw.copy(); Xa["_"] = 0; Xa = Xa.drop(columns="_")
dd = full[["place", "grape"]].join(pd.DataFrame({"x": 0}, index=full.index)).drop(columns="x")
dd["lat"] = full.index.map(d.set_index("id").lat.to_dict()); dd["lon"] = full.index.map(d.set_index("id").lon.to_dict())
dd = dd.dropna(subset=["lat"]); X0 = Xa.loc[dd.index]
Xadj = X0 - X0.groupby(dd.grape.values).transform("mean")         # net of grape: "terroir only" residual
pcount = dd.groupby("place").size(); keep_p = pcount[pcount >= 40].index
def centroid(X, k0=30):
    c = X.groupby(dd.place.values).mean().loc[keep_p]; n = pcount.loc[keep_p].values[:, None]; return c * n / (n + k0)
C_raw, C_adj = centroid(X0), centroid(Xadj)
print('NaN in centroids:', int(C_raw.isna().sum().sum()), int(C_adj.isna().sum().sum())); C_raw, C_adj = C_raw.fillna(0), C_adj.fillna(0)
geo = dd.groupby("place")[["lat", "lon"]].first().loc[keep_p]
meta = full.loc[dd.index].groupby("place").agg(country=("country", "first"), province=("province", "first"), region=("region_1", "first"), pts=("points", "mean"), price=("price", "median")).loc[keep_p]
topg = dd.groupby(["place", "grape"]).size().reset_index(name="n").sort_values("n", ascending=False).groupby("place").head(3).groupby("place").apply(lambda t: ", ".join(f"{g} ({n})" for g, n in zip(t.grape, t.n))).loc[keep_p]
names = {c: c[2:].replace("__", ": ").replace("_", " ") for c in Xa.columns}
def hexcol(x, y):
    import colorsys
    a = (math.atan2(y, x) / (2 * math.pi)) % 1; r = min(1, math.hypot(x, y)); rgb = colorsys.hls_to_rgb(a, 0.62 - 0.18 * r, 0.35 + 0.55 * r); return "#%02x%02x%02x" % tuple(int(255 * v) for v in rgb)
def proj(C, tag):
    pca = PCA(12, random_state=1).fit(C.values); P = pca.transform(C.values)
    u2 = umap.UMAP(n_components=2, n_neighbors=12, min_dist=0.35, random_state=1).fit_transform(P)
    p2 = P[:, :2]; p3 = P[:, :3]
    nz = lambda a: (a - a.mean(0)) / (np.abs(a - a.mean(0)).max(0) + 1e-9)
    ax = []
    for j in range(3):
        o = np.argsort(pca.components_[j]); ax.append(dict(ev=round(float(pca.explained_variance_ratio_[j]), 3), neg=[names[C.columns[i]] for i in o[:5]], pos=[names[C.columns[i]] for i in o[::-1][:5]]))
    return dict(pca2=np.round(p2, 3).tolist(), pca3=np.round(p3, 3).tolist(), umap2=np.round(u2, 3).tolist(), col_pca=[hexcol(*r) for r in nz(p2)], col_umap=[hexcol(*r) for r in nz(u2)], axes=ax), P
res_raw, P_raw = proj(C_raw, "raw"); res_adj, P_adj = proj(C_adj, "adj")
# distinctive descriptors per place (vs all-place mean, z-scored)
Z0 = (C_adj - C_adj.mean()) / C_adj.std()
dist = []
for p in keep_p:
    r = Z0.loc[p]; o = r.sort_values(ascending=False)
    dist.append([[names[c], round(float(v), 2)] for c, v in o.head(5).items()] + [[names[c], round(float(v), 2)] for c, v in o.tail(3).items()])
places = [dict(name=f"{meta.loc[p, 'region']} · {meta.loc[p, 'province']}, {meta.loc[p, 'country']}", country=meta.loc[p, "country"], province=meta.loc[p, "province"], lat=round(float(geo.loc[p, "lat"]), 3), lon=round(float(geo.loc[p, "lon"]), 3), n=int(pcount[p]),
               pts=round(float(meta.loc[p, "pts"]), 1), grapes=topg[p], more=dist[i]) for i, p in enumerate(keep_p)]
# embedding space (20k sample, places with >=12 notes)
e20d = e20.join(d.set_index("id")[["lat", "lon"]]).dropna(subset=["lat"]); ce = E.loc[e20d.index]
pe = e20d.groupby("place").size(); pe = pe[pe >= 12].index; Ce = ce.groupby(e20d.place.values).mean().loc[pe]
Ce = (Ce - Ce.mean()).fillna(0); emb_res = None
if len(pe) >= 30:
    Pc = PCA(12, random_state=1).fit_transform(Ce.values); u2 = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.35, random_state=1).fit_transform(Pc)
    nz = lambda a: (a - a.mean(0)) / (np.abs(a - a.mean(0)).max(0) + 1e-9)
    emb_res = dict(names=[f"{r} · {pv}, {c}" for r, pv, c in (x.split("|") for x in pe)], pca2=np.round(Pc[:, :2], 3).tolist(), umap2=np.round(u2, 3).tolist(), col_pca=[hexcol(*r) for r in nz(Pc[:, :2])], col_umap=[hexcol(*r) for r in nz(u2)])
# geography <-> taste: distance decay (+ Mantel permutation test)
la, lo_ = np.radians(geo.lat.values), np.radians(geo.lon.values)
Dkm = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None] * np.sin((lo_[:, None] - lo_[None]) / 2) ** 2))
iu = np.triu_indices(len(geo), 1); km = Dkm[iu]
EDGES = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 20000]
decay = {}
for tag, P in (("raw", P_raw), ("net_of_grape", P_adj)):
    D = np.sqrt(((P[:, None, :] - P[None]) ** 2).sum(-1)); t = D[iu]; rho = spearmanr_ = None
    from scipy.stats import spearmanr
    rho = spearmanr(km, t).correlation; perm = []
    for _ in range(300):
        q = rng.permutation(len(P)); perm.append(spearmanr(km, D[q][:, q][iu]).correlation)
    b = np.digitize(km, EDGES) - 1; med = float(np.median(t)); rows = []
    for i in range(len(EDGES) - 1):
        m = b == i
        if m.sum() >= 20: rows.append([EDGES[i], EDGES[i + 1], int(m.sum()), round(float(t[m].mean() / t.mean()), 3), round(float(np.percentile(t[m], 25) / t.mean()), 3), round(float(np.percentile(t[m], 75) / t.mean()), 3)])
    decay[tag] = dict(rho=round(float(rho), 3), p=round(float((np.sum(np.array(perm) >= rho) + 1) / 301), 4), rho_lt1600=round(float(spearmanr(km[km < 1600], t[km < 1600]).correlation), 3), bins=rows)
print("D done: places", len(places), "| decay rho", {k: v["rho"] for k, v in decay.items()})

out = dict(
    meta=dict(reviews_with_weather=int(len(W)), units=int(len(U)), place_grape_groups=int(U.grp.nunique()), places=len(places), years=[2000, 2017], bins=BN,
              attrs=ATTR, rates=sorted(RATE), grapes=grapes, notes="Associational. Weather = NASA POWER grid cells (0.5°×0.625°), growing season; anomalies vs the cell's own 2000–2017 climate."),
    variance=dict(wset=var_w, embedding=var_e, n_wset=int(len(Xw)), n_emb=int(len(E))),
    curves=curves, site=site_rows, pls=pls_out, explorer=explorer, explorer_attrs=EX,
    places=places, space=dict(raw=res_raw, net=res_adj, emb=emb_res), decay=decay)
def clean(o, path=""):
    if isinstance(o, dict): return {k: clean(v, path + "/" + str(k)) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v, path) for v in o]
    if isinstance(o, float) and not math.isfinite(o): NANS.append(path); return None
    return o
NANS = []; out = clean(out); print("NaN replaced by null at:", sorted(set(NANS))[:10], len(NANS))
json.dump(out, open(ROOT / "dashboard/data/terroir.json", "w"), separators=(",", ":"), allow_nan=False)
print("wrote terroir.json", round((ROOT / "dashboard/data/terroir.json").stat().st_size / 1024), "KB")
