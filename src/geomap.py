"""Phase 6: interactive map of regions (folium). Usage: geomap.py <N> [--vec suffix]
Review-level stats (points, price, count) use ALL reviews; WSET-cue means use the matched sample."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, folium
from branca.colormap import LinearColormap
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT

n = sys.argv[1]
vsuf = sys.argv[sys.argv.index("--vec") + 1] if "--vec" in sys.argv else ""
pl = pd.read_parquet(ROOT / "outputs/places_geocoded.parquet").dropna(subset=["lat"])
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet")
vec = pd.read_parquet(ROOT / f"outputs/wset_vectors_{n}{vsuf}.parquet")
key = ["country", "province", "region_1"]
nm_k = nm.assign(**{k: nm[k].fillna("∅") for k in key})
pl_k = pl.assign(**{k: pl[k].fillna("∅") for k in key})
allst = nm_k.groupby(key).agg(n=("id", "size"), points=("points", "mean"), price=("price", "median"),
                              top_variety=("variety_group", lambda s: s.value_counts().index[0])).reset_index()
samp = nm_k.merge(vec, on="id").groupby(key).agg(acidity=("s_acidity", "mean"), tannin=("s_tannin", "mean"), body=("s_body", "mean"),
                                                 sweetness=("s_sweetness", "mean"), n_s=("id", "size")).reset_index()
d = pl_k[key + ["lat", "lon", "geo_level"]].merge(allst, on=key).merge(samp, on=key, how="left")
w = ROOT / "outputs/weather_metrics.parquet"
if w.exists():
    from weather import loc_key
    d["loc"] = [loc_key(a, b) for a, b in zip(d.lat, d.lon)]
    wm = pd.read_parquet(w).groupby("loc").agg(gdd=("gdd", "mean"), rain_season=("rain_season", "mean"),
                                               diurnal=("diurnal", "mean")).reset_index()
    d = d.merge(wm, on="loc", how="left")
m = folium.Map(location=[30, 0], zoom_start=2, tiles="CartoDB positron")
layers = [("avg points", "points", "viridis"), ("avg acidity cue (1-5)", "acidity", "plasma"), ("avg tannin cue (1-5)", "tannin", "plasma"),
          ("avg body cue (1-5)", "body", "plasma"), ("median price $", "price", "magma"),
          ("growing degree days", "gdd", "inferno"), ("season rain mm", "rain_season", "Blues"), ("diurnal range °C", "diurnal", "cividis")]
for label, col, _ in layers:
    if col not in d or d[col].notna().sum() < 5:
        continue
    sub = d[d[col].notna() & (d.n >= 20)]
    lo, hi = sub[col].quantile(0.02), sub[col].quantile(0.98)
    cm = LinearColormap(["#2b83ba", "#ffffbf", "#d7191c"], vmin=float(lo), vmax=float(hi), caption=label)
    fg = folium.FeatureGroup(name=label, show=(col == "points"))
    for r in sub.itertuples():
        v = getattr(r, col)
        folium.CircleMarker([r.lat, r.lon], radius=3 + 2.2 * np.log10(r.n), color=cm(min(max(v, lo), hi)), fill=True, fill_opacity=0.75, weight=0.5,
                            tooltip=f"<b>{r.region_1 if r.region_1 != '∅' else r.province}</b>, {r.province}, {r.country}<br>reviews {r.n} · avg pts {r.points:.1f}"
                                    f"<br>{label}: {v:.2f}<br>top grape: {r.top_variety}<br>geo: {r.geo_level}").add_to(fg)
    fg.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
out = ROOT / "reports" / f"map_{n}{vsuf}.html"
m.save(out)
print("saved", out, "places", len(d))
