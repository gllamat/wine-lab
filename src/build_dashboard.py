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
    json.dump(obj, open(D / name, "w"), separators=(",", ":"), default=float, allow_nan=False)   # NaN is invalid JSON -> fail loudly
    print("wrote", name, round((D / name).stat().st_size / 1024), "KB")


def clean(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else x


nm = pd.read_parquet(O / "reviews_norm.parquet")
# ---------------- overview (full-corpus matcher run if available)
full = O / "wset_vectors_all.parquet"
vec_all = pd.read_parquet(full if full.exists() else O / f"wset_vectors_{n}{vsuf}.parquet")
from wsetkit import Lexicon
_L = Lexicon()
scales = [x for x in _L.scales]
ov = dict(
    n_reviews=int(len(vec_all)), n_countries=int(nm.country.nunique()), n_regions=int(nm.region_1.nunique()),
    n_varieties=int(nm.variety_norm.nunique()), vintage_coverage=float(nm.vintage.notna().mean()),
    coverage={s: float(vec_all["s_" + s].notna().mean()) for s in scales} | {
        "aroma_cluster": float((vec_all[[c for c in vec_all if c.startswith("c_")]].sum(axis=1) > 0).mean()),
        "drink_now": float(vec_all.r_drink_now.mean()), "can_age": float(vec_all.r_can_age.mean()), "too_old": float(vec_all.r_too_old.mean()),
        "year_range": float(vec_all.r_year_from.notna().mean())} | {"cat_" + f: float(vec_all["cat_" + f].notna().mean()) for f in _L.cat_fields},
    tiers={t: float(vec_all["t_" + t].mean()) for t in ("primary", "secondary", "tertiary")},
    cluster_tier=_L.tier,
    scope=dict(nose=float(vec_all[[c for c in vec_all if c.startswith("cn_")]].max(axis=1).mean()), palate=float(vec_all[[c for c in vec_all if c.startswith("cp_")]].max(axis=1).mean())),
    readiness=vec_all.readiness.fillna("not_stated").value_counts().to_dict(),
    quality=vec_all.quality_band.value_counts().to_dict(),
    clusters={c[2:]: float(vec_all[c].mean()) for c in vec_all if c.startswith("c_")},
    mean_token_coverage=float(vec_all.coverage.mean()),
)
mf = O / ("matched_forms_all_extra.csv" if (O / "matched_forms_all_extra.csv").exists() else f"matched_forms_{n}{vsuf}.csv")
if mf.exists():
    ov["top_forms"] = pd.read_csv(mf).head(60).fillna("").astype(str).values.tolist()
dump("overview.json", ov)
# ---------------- lexicon (for the in-browser matcher) + browsable form table + what the lexicon misses
from export_lexicon import write_lexicon
lex, lj = write_lexicon(D / "lexicon.json")
if mf.exists():
    mc = pd.read_csv(mf).fillna("")
    cnt = {(r.kind, str(r.f1), str(r.f2), r.surface, r.tag): int(r["count"]) for _, r in mc.iterrows()}
    rows = []
    for key, targets in lj["entries"]:
        for kind, f1, f2, tag in targets:
            rows.append([key, kind, str(f1), str(f2), tag, cnt.get((kind, str(f1), str(f2), key, tag), 0)])
    for r in mc[mc.tag.isin(["bound", "bound-x"])].itertuples():
        rows.append([r.surface, r.kind, str(r.f1), str(r.f2), r.tag, int(r.count)])
    dump("forms.json", rows)
um = O / "unmatched_unigrams_all_extra.csv"
if um.exists():
    u = pd.read_csv(um).head(250)
    tm = O / "term_map_minilm.csv"
    near = pd.read_csv(tm).set_index("term") if tm.exists() else None
    out = []
    for r in u.itertuples():
        nd, sim, acc = (None, None, None)
        if near is not None and r.term in near.index:
            x = near.loc[r.term]; nd, sim, acc = x.desc, round(float(x.sim), 2), bool(x.accepted)
        out.append([r.term, int(r.count), nd, sim, acc])
    dump("unmatched.json", out)
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
from places import load_places
if pg.exists():
    pl = load_places().dropna(subset=["lat"])
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
# ---------------- missing locations (geocode / weather / vintage gaps)
if pg.exists():
    from weather import loc_key, CACHE as WCACHE, STATUS as WSTATUS
    wst = json.load(open(WSTATUS)) if WSTATUS.exists() else {}
    pl2 = load_places()
    key = ["country", "province", "region_1"]
    ag = (nm.assign(inwin=nm.vintage.between(2000, 2017)).groupby(key, dropna=False)
            .agg(n=("id", "size"), n_vint=("inwin", "sum"), n_anyvint=("vintage", lambda s: s.notna().sum())).reset_index())
    m2 = pl2[key + ["lat", "lon", "geo_level", "rejected_km", "pending_alt"]].merge(ag, on=key, how="right")
    rows, sev = [], {}
    for r in m2.itertuples():
        reasons = []
        has = pd.notna(r.lat)
        if pd.notna(r.rejected_km):
            reasons.append(f"region match rejected ({int(r.rejected_km)} km outside country)")
        if r.pending_alt is True:
            reasons.append("alternative geocode query pending")
        if not has:
            reasons.append("no coordinates"); ws = "n/a"
        else:
            if pd.notna(r.region_1) and r.geo_level not in ("region_1", "region_1_alt"):
                reasons.append(f"coarse geocode ({r.geo_level})")
            lk = loc_key(r.lat, r.lon)
            if r.n_vint == 0:
                ws = "n/a (no 2000-17 vintage)"
            elif (WCACHE / f"{lk}.json").exists():
                ws = "ok"
            elif wst.get(lk, {}).get("status") == "failed":
                ws = "failed"; reasons.append("weather download failed")
            else:
                ws = "pending"; reasons.append("weather pending")
        if r.n - r.n_anyvint > 0 and r.n_anyvint == 0:
            reasons.append("no vintage in titles")
        if reasons:
            rows.append([r.country if isinstance(r.country, str) else "(missing)", r.province if pd.notna(r.province) else "", r.region_1 if pd.notna(r.region_1) else "", int(r.n), int(r.n_vint),
                         r.geo_level if isinstance(r.geo_level, str) else "", None if not has else round(float(r.lat), 3),
                         None if not has else round(float(r.lon), 3), ws, reasons])
    rows.sort(key=lambda x: -x[3])
    tot = int(m2.n.sum())
    summary = dict(places=int(len(m2)), reviews=tot,
                   reviews_region_geocoded=int(m2[m2.geo_level.isin(["region_1", "region_1_alt"])].n.sum()),
                   reviews_province_geocoded=int(m2[m2.geo_level == "province"].n.sum()),
                   reviews_country_geocoded=int(m2[m2.geo_level == "country"].n.sum()),
                   reviews_no_coords=int(m2[m2.lat.isna()].n.sum()),
                   reviews_with_vintage_in_window=int(m2.n_vint.sum()),
                   reviews_vintage_weather_ok=int(sum(r[4] for r in rows if False) + m2.apply(
                       lambda r: r.n_vint if pd.notna(r.lat) and (WCACHE / f"{loc_key(r.lat, r.lon)}.json").exists() else 0, axis=1).sum()),
                   places_with_gaps=len(rows), weather_cells_cached=len(list(WCACHE.glob("*.json"))))
    dump("missing.json", dict(summary=summary, rows=rows))
    pd.DataFrame([r[:9] + ["; ".join(r[9])] for r in rows],
                 columns=["country", "province", "region_1", "reviews", "reviews_vintage_2000_17", "geo_level", "lat", "lon", "weather", "gaps"]
                 ).to_csv(O / "missing_locations.csv", index=False)
# ---------------- weather
jw = O / ("joint_weather_all.csv" if (O / "joint_weather_all.csv").exists() else f"joint_weather_{n}{vsuf}.csv")
if jw.exists():
    dump("weather.json", json.loads(pd.read_csv(jw).round(4).replace({np.nan: None}).to_json(orient="records")))
import key_status
key_status.run()            # refresh Jev API key health for the dashboard reminder
dump("meta.json", dict(built=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), sample=n,
                       note="Aggregates and coordinates only. No review text is published."))
