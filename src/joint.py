"""Phase 8: does growing-season weather relate to WSET cues / score? Usage: joint.py <N> [--vec suffix]
Within-group design: outcomes and weather are demeaned within (place x grape) so we compare VINTAGES of the same wine
type in the same place (removes between-region confounding). Also reports the raw across-place correlation."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT
from weather import loc_key

n = sys.argv[1]
vsuf = sys.argv[sys.argv.index("--vec") + 1] if "--vec" in sys.argv else ""
from places import load_places
pl = load_places().dropna(subset=["lat"])
pl["loc"] = [loc_key(a, b) for a, b in zip(pl.lat, pl.lon)]
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet")
vec = pd.read_parquet(ROOT / f"outputs/wset_vectors_{n}{vsuf}.parquet")
wx = pd.read_parquet(ROOT / "outputs/weather_metrics.parquet")
d = (nm.merge(vec, on="id").merge(pl[["country", "province", "region_1", "loc", "geo_level"]], on=["country", "province", "region_1"], how="inner")
       .merge(wx, on=["loc", "vintage"], how="inner"))
d = d[d.geo_level.isin(["region_1", "region_1_alt"])] if (d.geo_level == "region_1").sum() > 500 else d   # prefer region-level coordinates
print("joined rows", len(d), "locations", d["loc"].nunique(), "vintages", d.vintage.nunique())
W = ["gdd", "huglin", "t_mean", "diurnal", "days_gt35", "rain_season", "rain_harvest", "frost_budbreak", "radiation"]
Y = ["points", "s_acidity", "s_body", "s_tannin", "s_sweetness", "c_red_fruit", "c_black_fruit", "c_citrus_fruit", "c_stone_fruit",
     "c_tropical_fruit", "c_fruit_ripeness", "c_oak", "c_herbaceous", "t_tertiary", "r_can_age"]
d["grp"] = d["loc"] + "|" + d.variety_group.astype(str)
sz = d.groupby("grp").vintage.transform("nunique")
dw = d[sz >= 4].copy()                                     # groups with >=4 vintages
rows = []
for w in W:
    dw[w + "_dm"] = dw[w] - dw.groupby("grp")[w].transform("mean")
    for y in Y:
        if y not in dw:
            continue
        ok = dw[y].notna()
        if ok.sum() < 300:
            continue
        # within place x grape: demean, then average to place-grape-VINTAGE units (reviews of one unit share one weather value)
        tmp = pd.DataFrame({"grp": dw.loc[ok, "grp"], "vintage": dw.loc[ok, "vintage"], "x": dw.loc[ok, w + "_dm"],
                            "y": dw.loc[ok, y] - dw.loc[ok].groupby("grp")[y].transform("mean")})
        u = tmp.groupby(["grp", "vintage"]).agg(x=("x", "mean"), y=("y", "mean"), k=("y", "size")).query("k >= 3")
        if len(u) < 30:
            continue
        r = spearmanr(u.x, u.y)
        tb = pd.DataFrame({"loc": d.loc[d[y].notna(), "loc"], "vintage": d.loc[d[y].notna(), "vintage"], "x": d.loc[d[y].notna(), w], "y": d.loc[d[y].notna(), y]})
        ua = tb.groupby(["loc", "vintage"]).agg(x=("x", "mean"), y=("y", "mean"), k=("y", "size")).query("k >= 3")
        ra = spearmanr(ua.x, ua.y)
        rows.append(dict(weather=w, outcome=y, n=int(len(u)), n_reviews=int(ok.sum()), rho_within=r.correlation, p_within=r.pvalue, rho_across=ra.correlation))
res = pd.DataFrame(rows)
res.to_csv(ROOT / f"outputs/joint_weather_{n}{vsuf}.csv", index=False)
import plotly.graph_objects as go
for col, name in (("rho_within", "within place×grape (vintage effect)"), ("rho_across", "across places (confounded by region)")):
    P = res.pivot(index="outcome", columns="weather", values=col)
    fig = go.Figure(go.Heatmap(z=P.values, x=P.columns, y=P.index, zmin=-0.3, zmax=0.3, colorscale="RdBu", reversescale=True,
                               text=np.round(P.values, 2), texttemplate="%{text}"))
    fig.update_layout(title=f"Spearman ρ: growing-season weather vs outcome — {name}", height=600)
    fig.write_html(ROOT / "reports" / f"weather_{col}_{n}{vsuf}.html", include_plotlyjs="cdn")
print(res.sort_values("p_within").head(15).to_string(index=False))
