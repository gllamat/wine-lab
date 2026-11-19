"""Adjustment gallery: the same question ("what does +100 growing-degree days do?") answered under different adjustment sets,
so the effect of right and wrong adjustments is visible on real data. Writes dashboard/data/causal.json.
Treatment: season growing-degree days (GDD, +100). Outcomes: critic points, acidity cue (1-5, stated notes), ripe-fruit share, body cue.
Design: fixed effects by iterative demeaning, cluster-robust SE by country x vintage (statsmodels)."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id
from weather import loc_key
from places import load_places
import causal

O = ROOT / "outputs"
nm = pd.read_parquet(O / "reviews_norm.parquet"); vec = pd.read_parquet(O / "wset_vectors_all.parquet", columns=["id", "s_acidity", "s_body", "fruit_ripeness"])
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description", "taster_name"]); raw["id"] = raw.description.map(review_id)
raw = raw.drop_duplicates("id")[["id", "taster_name"]]
pl = load_places().dropna(subset=["lat"]); pl["loc"] = [loc_key(a, b) for a, b in zip(pl.lat, pl.lon)]; pl = pl[pl.geo_level.isin(["region_1", "region_1_alt"])]
cl = pd.read_parquet(O / "climate_all_seasons.parquet")[["loc", "vintage", "gdd"]]
d = nm.merge(vec, on="id").merge(raw, on="id", how="left").merge(pl[["country", "province", "region_1", "loc"]], on=["country", "province", "region_1"]).merge(cl, on=["loc", "vintage"])
nxt = cl.assign(vintage=cl.vintage - 1).rename(columns={"gdd": "gdd_next"}); d = d.merge(nxt, on=["loc", "vintage"], how="left")
d["place"] = d.region_1.astype(str) + "|" + d.province.astype(str) + "|" + d.country.astype(str); d["grape"] = d.variety_group.astype(str)
d["pg"] = d.place + "|" + d.grape; d["winery_g"] = d.winery.astype(str) + "|" + d.grape; d["cy"] = d.country.astype(str) + "|" + d.vintage.astype(int).astype(str)
d["gdd100"] = d.gdd / 100; d["gdd100_next"] = d.gdd_next / 100; d["lprice"] = np.log(d.price); d["ripe"] = (d.fruit_ripeness == "ripe_fruit").astype(float)
d["year"] = d.vintage.astype(int).astype(str)
print("rows with region-level coordinates and weather:", len(d), "| place x grape groups", d.pg.nunique())
OUT = {"points": "Critic points", "s_acidity": "Acidity cue (1-5)", "s_body": "Body cue (1-5)", "ripe": "Ripe-fruit share"}


def demean(df, cols, fes, tol=1e-7, it=30):
    X = df[cols].astype(float).copy()
    for _ in range(it):
        prev = X.values.copy()
        for fe in fes: X = X - X.groupby(df[fe].values).transform("mean")
        if np.abs(X.values - prev).max() < tol: break
    return X


def detrend(df, cols, grp):
    """Remove a place x grape specific linear trend in vintage year from each column (groups with >=3 vintages keep their slope removed)."""
    t = df.vintage.astype(float); out = df[cols].astype(float).copy(); g = df[grp].values
    tm = t.groupby(g).transform("mean"); tc = t - tm; var = (tc ** 2).groupby(g).transform("sum")
    for c in cols:
        y = out[c]; yc = y - y.groupby(g).transform("mean"); b = (tc * yc).groupby(g).transform("sum") / var.replace(0, np.nan)
        out[c] = (yc - b.fillna(0) * tc)
    return out


def fit(sub, y, x, ctrl, fes, trend=False):
    s = sub.dropna(subset=[y, x] + ctrl)
    if trend:
        nv = s.groupby("pg").vintage.transform("nunique"); s = s[nv >= 4]
    if len(s) < 500: return None
    if trend:
        D0 = detrend(s, [y, x] + ctrl, "pg"); s = s.assign(**{c: D0[c] for c in [y, x] + ctrl}); fes = [f for f in fes if f != "pg"]
    D = demean(s, [y, x] + ctrl, fes) if fes else s[[y, x] + ctrl].astype(float)
    if not fes: D = D - D.mean()
    r = sm.OLS(D[y], D[[x] + ctrl]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.cy)[0]})
    b, se = float(r.params[x]), float(r.bse[x]); return [round(b, 4), round(b - 1.645 * se, 4), round(b + 1.645 * se, 4), int(len(s))]


# spec: id, label, fixed effects, controls, treatment column, DAG adjustment set (names in the graph), subset restriction
SPECS = [
    ("S0", "Nothing (raw comparison across all wines)", [], [], "gdd100", [], None),
    ("S1", "Grape", ["grape"], [], "gdd100", ["grape"], None),
    ("S2", "Place", ["place"], [], "gdd100", ["place"], None),
    ("S3", "Place × grape", ["pg"], [], "gdd100", ["place", "grape"], None),
    ("S4", "Place × grape + year", ["pg", "year"], [], "gdd100", ["place", "grape", "year"], None),
    ("S5", "Winery × grape + year (same winery, other vintages)", ["winery_g", "year"], [], "gdd100", ["producer", "grape", "year"], None),
    ("S6", "Place × grape + year + taster", ["pg", "year", "taster_name"], [], "gdd100", ["place", "grape", "year", "taster"], "taster"),
    ("S7", "Place × grape + year + PRICE (wrong: price is downstream)", ["pg", "year"], ["lprice"], "gdd100", ["place", "grape", "year", "price"], None),
    ("S8", "Place × grape + year + CRITIC POINTS (wrong: downstream)", ["pg", "year"], ["points"], "gdd100", ["place", "grape", "year", "points"], "notpoints"),
    ("S9", "Placebo: NEXT year's weather, place × grape + year", ["pg", "year"], [], "gdd100_next", None, None),
    ("S10", "Place × grape + year + place-specific trend", ["pg", "year"], [], "gdd100", ["place", "grape", "year"], "trend"),
    ("S11", "Placebo with trend: NEXT year's weather", ["pg", "year"], [], "gdd100_next", None, "trend"),
]
L = causal.library(); G = {l["id"]: (causal.DAG(l["nodes"], l["edges"]), l["conditioned"]) for l in L}
specs = []
for sid, label, fes, ctrl, tcol, adj, restrict in SPECS:
    rows = {}
    for y in OUT:
        if restrict == "notpoints" and y == "points": continue
        sub = d[d.taster_name.notna()] if restrict == "taster" else d
        rows[y] = fit(sub, y, tcol, ctrl, fes, trend=(restrict == "trend"))
    valid = {}
    if adj is not None:
        for did, (g, cond) in G.items():
            tgt = "note_sat"; ok, why = g.backdoor_valid("weather_anomaly", tgt, [a for a in adj if a in g.nodes], cond); valid[did] = [ok, why]
    specs.append(dict(id=sid, label=label, adjust=adj, controls=ctrl, valid=valid, rows=rows)); print(sid, label[:40], {k: v and v[0] for k, v in rows.items()})
json.dump(dict(outcomes=OUT, specs=specs, n=int(len(d)), treatment="+100 growing-degree days in the season", note="90% CI, SE clustered by country × vintage"), open(ROOT / "dashboard/data/causal.json", "w"), separators=(",", ":"))
print("wrote causal.json")
