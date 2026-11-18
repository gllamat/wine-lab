"""Interactive 2D/3D plots (Plotly HTML) of the WSET and embedding spaces. Usage: viz.py <N> [--vec suffix]"""
import sys, html
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id

n = sys.argv[1]
vsuf = sys.argv[sys.argv.index("--vec") + 1] if "--vec" in sys.argv else ""
co = pd.read_parquet(ROOT / f"outputs/coords_{n}{vsuf}.parquet")
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet").set_index("id")
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
raw["id"] = raw.description.map(review_id)
desc = raw.drop_duplicates("id").set_index("id").description
vec = pd.read_parquet(ROOT / f"outputs/wset_vectors_{n}{vsuf}.parquet").set_index("id")
m = nm.loc[co.id].copy()
m["description"] = desc.loc[co.id].values
m["quality"] = vec.loc[co.id, "quality_band"].values
m["hover"] = ("<b>" + m.variety_norm.fillna("?").astype(str) + "</b> · " + m.country.astype(str) + " · " +
              m.points.astype(str) + " pts · " + m.vintage.astype(str) + "<br>" +
              m.description.str.slice(0, 160).map(html.escape) + "…")
COLOURS = ["variety_group", "country", "colour", "points", "vintage", "quality"]
spaces = sorted({c.rsplit("_", 2)[0] for c in co.columns if c != "id"})


def palette(series):
    cats = sorted(series.dropna().astype(str).unique())
    base = (["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] * 10)
    return {c: base[i] for i, c in enumerate(cats)}


def figure(space, meth, dim):
    cols = [f"{space}_{meth}{dim}_{k}" for k in range(dim)]
    Z = co[cols].values
    fig = go.Figure()
    traces = []
    for ci, cname in enumerate(COLOURS):
        s = m[cname]
        if cname in ("points", "vintage"):
            vals = pd.to_numeric(s, errors="coerce")
            kw = dict(color=vals, colorscale="Viridis", showscale=True, colorbar=dict(title=cname))
        else:
            pal = palette(s)
            kw = dict(color=s.astype(str).map(pal).fillna("#cccccc"))
        mk = dict(size=3 if dim == 3 else 4, opacity=0.7, **kw)
        args = dict(mode="markers", marker=mk, text=m.hover, hoverinfo="text", visible=(ci == 0), name=cname)
        fig.add_trace(go.Scatter3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], **args) if dim == 3 else
                      go.Scattergl(x=Z[:, 0], y=Z[:, 1], **args))
    buttons = [dict(label=c, method="update", args=[{"visible": [i == j for j in range(len(COLOURS))]}]) for i, c in enumerate(COLOURS)]
    fig.update_layout(title=f"{space} · {meth.upper()} {dim}D · colour by →", updatemenus=[dict(buttons=buttons, x=0.0, y=1.1, xanchor="left")],
                      height=720, margin=dict(l=10, r=10, t=70, b=10))
    return fig


out = ROOT / "reports"; out.mkdir(exist_ok=True)
for sp in spaces:
    for meth in ("umap", "pca"):
        for dim in (2, 3):
            figure(sp, meth, dim).write_html(out / f"{sp}_{meth}{dim}d_{n}{vsuf}.html", include_plotlyjs="cdn")
print("wrote", len(list(out.glob(f'*_{n}{vsuf}.html'))), "html files")
