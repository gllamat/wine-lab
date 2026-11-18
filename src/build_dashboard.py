"""Build dashboard/data/*.json from pipeline outputs (aggregates + coordinates only; NO review text).
Usage: build_dashboard.py <N> [--vec suffix]   (tolerates missing outputs; rerun as they appear)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT

n = sys.argv[1]
vsuf = sys.argv[sys.argv.index("--vec") + 1] if "--vec" in sys.argv else ""
D = ROOT / "dashboard/data"; D.mkdir(parents=True, exist_ok=True)
O = ROOT / "outputs"


def dump(name, obj):
    json.dump(obj, open(D / name, "w"), separators=(",", ":"), default=float)
    print("wrote", name, round((D / name).stat().st_size / 1024), "KB")


def clean(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else x


nm = pd.read_parquet(O / "reviews_norm.parquet")
# ---------------- overview (full-corpus matcher run if available)
full = O / "wset_vectors_all.parquet"
vec_all = pd.read_parquet(full if full.exists() else O / f"wset_vectors_{n}{vsuf}.parquet")
scales = ["acidity", "tannin", "body", "alcohol", "sweetness", "finish", "flavour_intensity", "development"]
ov = dict(
    n_reviews=int(len(vec_all)), n_countries=int(nm.country.nunique()), n_regions=int(nm.region_1.nunique()),
    n_varieties=int(nm.variety_norm.nunique()), vintage_coverage=float(nm.vintage.notna().mean()),
    coverage={s: float(vec_all["s_" + s].notna().mean()) for s in scales} | {
        "aroma_cluster": float((vec_all[[c for c in vec_all if c.startswith("c_")]].sum(axis=1) > 0).mean()),
        "drink_now": float(vec_all.r_drink_now.mean()), "can_age": float(vec_all.r_can_age.mean()),
        "year_range": float(vec_all.r_year_from.notna().mean())},
    quality=vec_all.quality_band.value_counts().to_dict(),
    clusters={c[2:]: float(vec_all[c].mean()) for c in vec_all if c.startswith("c_")},
    mean_token_coverage=float(vec_all.coverage.mean()),
)
mf = O / ("matched_forms_all.csv" if (O / "matched_forms_all.csv").exists() else f"matched_forms_{n}{vsuf}.csv")
if mf.exists():
    ov["top_forms"] = pd.read_csv(mf).head(60).fillna("").astype(str).values.tolist()
dump("overview.json", ov)
# ---------------- spaces (coordinates + metadata only)
cf = O / f"coords_{n}{vsuf}.parquet"
if cf.exists():
    co = pd.read_parquet(cf)
    m = nm.set_index("id").loc[co.id]
    q = vec_all.set_index("id").quality_band if "id" in vec_all else None
    spaces = {}
    for c in co.columns:
        if c == "id":
            continue
        sp, md, k = c.rsplit("_", 2)
        md = md.rstrip("0123456789")        # columns are ordered pca2_0,pca2_1,pca3_0,pca3_1,pca3_2
        spaces.setdefault(f"{sp}|{md}", []).append(np.round(co[c].values, 3).tolist())
    meta = {k: [clean(x) for x in m[k].tolist()] for k in ("variety_group", "country", "colour", "points", "price")}
    meta["vintage"] = [None if pd.isna(x) else int(x) for x in m.vintage.tolist()]
    meta["quality"] = [clean(x) for x in q.reindex(co.id).tolist()] if q is not None else []
    dump("spaces.json", dict(spaces=spaces, meta=meta))
# ---------------- comparison metrics
cj = O / f"compare_{n}{vsuf}.json"
if cj.exists():
    dump("compare.json", json.load(open(cj)))
# ---------------- map
pg = O / "places_geocoded.parquet"
if pg.exists():
    pl = pd.read_parquet(pg).dropna(subset=["lat"])
    key = ["country", "province", "region_1"]
    nk = nm.assign(**{k: nm[k].fillna("∅") for k in key}); pk = pl.assign(**{k: pl[k].fillna("∅") for k in key})
    st = nk.groupby(key).agg(n=("id", "size"), points=("points", "mean"), price=("price", "median"),
                             top_variety=("variety_group", lambda s: s.value_counts().index[0])).reset_index()
    sv = nk.merge(vec_all, on="id").groupby(key).agg(acidity=("s_acidity", "mean"), tannin=("s_tannin", "mean"),
                                                      body=("s_body", "mean"), sweetness=("s_sweetness", "mean")).reset_index()
    d = pk[key + ["lat", "lon", "geo_level"]].merge(st, on=key).merge(sv, on=key, how="left")
    wf = O / "weather_metrics.parquet"
    if wf.exists():
        from weather import loc_key
        d["loc"] = [loc_key(a, b) for a, b in zip(d.lat, d.lon)]
        wm = pd.read_parquet(wf).groupby("loc").agg(gdd=("gdd", "mean"), rain_season=("rain_season", "mean"), diurnal=("diurnal", "mean")).reset_index()
        d = d.merge(wm, on="loc", how="left").drop(columns="loc")
    d = d[d.n >= 10]
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].round(3)
    dump("map.json", json.loads(d.replace({np.nan: None}).to_json(orient="records")))
# ---------------- weather
jw = O / f"joint_weather_{n}{vsuf}.csv"
if jw.exists():
    dump("weather.json", json.loads(pd.read_csv(jw).round(4).replace({np.nan: None}).to_json(orient="records")))
dump("meta.json", dict(built=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), sample=n,
                       note="Aggregates and coordinates only. No review text is published."))
