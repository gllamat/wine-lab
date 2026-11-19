"""Per-place attribute means for the Terroir map (colour-by-attribute, marker shape). Writes dashboard/data/place_attrs.json keyed by the place names used in terroir.json."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet"); vec = pd.read_parquet(ROOT / "outputs/wset_vectors_all.parquet")
d = nm.merge(vec[["id", "s_acidity", "s_body", "s_tannin", "c_oak", "c_citrus_fruit", "c_tropical_fruit", "c_red_fruit", "c_black_fruit", "t_tertiary", "fruit_ripeness"]], on="id")
d = d[d.region_1.notna()]; d["name"] = d.region_1.astype(str) + " · " + d.province.astype(str) + ", " + d.country.astype(str)
d["cooked"] = (d.fruit_ripeness == "cooked_fruit").astype(float); d["red"] = (d.colour == "red").astype(float)
g = d.groupby("name"); n = g.size(); keep = n[n >= 40].index
A = pd.DataFrame(dict(acidity=g.s_acidity.mean(), body=g.s_body.mean(), tannin=g.s_tannin.mean(), oak=g.c_oak.mean(), citrus=g.c_citrus_fruit.mean(), tropical=g.c_tropical_fruit.mean(),
                      red_fruit=g.c_red_fruit.mean(), black_fruit=g.c_black_fruit.mean(), cooked=g.cooked.mean(), tertiary=g.t_tertiary.mean(), price=g.price.median(), red_share=g.red.mean())).loc[keep]
out = {k: {c: (None if pd.isna(v) else round(float(v), 3)) for c, v in row.items()} for k, row in A.iterrows()}
json.dump(out, open(ROOT / "dashboard/data/place_attrs.json", "w"), separators=(",", ":")); print("places", len(out))
