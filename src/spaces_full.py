"""LOCAL-ONLY full projections of ALL reviews (the public site keeps the 8,000-review sample in spaces.json).
Writes dashboard/data/local/spaces_full_meta.json + sp_<space>_<method>.f32 (5 float32 columns: x2,y2,x3,y3,z3; UMAP 3D is not computed -> zeros).
Spaces: 'wset' (170 SAT features, standardised) and 'minilm' (384-d embeddings). Methods: PCA (2D+3D) and UMAP (2D). Deterministic (random_state=1).
Usage: spaces_full.py"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT
import guess_model as G
from sklearn.decomposition import PCA
import umap

OUT = ROOT / "dashboard/data/local"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
ids, X, spec, nm = G.corpus("lite"); E = np.load(ROOT / "outputs/emb_minilm_all.npy").astype(np.float32); n = len(ids)
log("corpus", n)
Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
vec = pd.read_parquet(ROOT / "outputs/wset_vectors_all_nocolour.parquet", columns=["id", "quality_band"]).set_index("id").loc[ids]
cols = dict(variety_group=nm.variety_group.astype(str).values, country=nm.country.astype(str).values, colour=nm.colour.astype(str).values, quality=vec.quality_band.astype(str).values)
meta = dict(n=int(n), dicts={}, codes={}, points=[None if pd.isna(v) else int(v) for v in nm.points.values], vintage=[None if pd.isna(v) else int(v) for v in nm.vintage.astype(float).values],
            price=[None if pd.isna(v) else float(v) for v in nm.price.astype(float).values])
for k, v in cols.items():
    c = pd.Series(v).astype("category"); meta["dicts"][k] = [str(x) for x in c.cat.categories]; meta["codes"][k] = c.cat.codes.astype(int).tolist()
json.dump(meta, open(OUT / "spaces_full_meta.json", "w"), separators=(",", ":")); log("meta written")
for name, M in (("wset", Xs), ("minilm", E)):
    p3 = PCA(3, random_state=1).fit_transform(M); pca = np.column_stack([p3[:, 0], p3[:, 1], p3[:, 0], p3[:, 1], p3[:, 2]]).astype(np.float32)
    pca.T.copy().tofile(OUT / f"sp_{name}_pca.f32"); log(name, "PCA done")
    red = PCA(40, random_state=1).fit_transform(M)                                   # UMAP on 40 principal components: much faster, nearly the same neighbours
    u = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=1, low_memory=True, verbose=False).fit_transform(red).astype(np.float32)
    um = np.column_stack([u[:, 0], u[:, 1], np.zeros(n), np.zeros(n), np.zeros(n)]).astype(np.float32)
    um.T.copy().tofile(OUT / f"sp_{name}_umap.f32"); log(name, "UMAP done")
json.dump({"done": True}, open(OUT / "spaces_full_done.json", "w")); log("all done")
