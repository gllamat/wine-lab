"""Phase 4/4b: compare partial WSET vectors with embeddings; PCA/UMAP 2D+3D. Deterministic (fixed seeds).
Usage: compare.py <N> <model[,model...]> [--vec suffix]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id

SEED = 0
rng = np.random.RandomState(SEED)


def wset_matrix(vec):
    A = vec[[c for c in vec if c.startswith("a_")]].values.astype("float32")
    S = vec[[c for c in vec if c.startswith("s_")]]
    Sv = ((S.fillna(3) - 3) / 2).values.astype("float32")        # missing -> neutral 0
    return np.hstack([A, Sv]), list(vec.columns[[c.startswith("a_") for c in vec.columns]]) + list(S.columns)


def cos_nn(X, k=10):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def rsa(X, Y, n=2500):
    idx = rng.choice(len(X), n, replace=False)
    def tri(M):
        Mn = M[idx] / np.maximum(np.linalg.norm(M[idx], axis=1, keepdims=True), 1e-9)
        return (Mn @ Mn.T)[np.triu_indices(n, 1)]
    return float(spearmanr(tri(X), tri(Y)).correlation)


def main():
    n = sys.argv[1]
    models = sys.argv[2].split(",")
    vsuf = sys.argv[sys.argv.index("--vec") + 1] if "--vec" in sys.argv else ""
    vec = pd.read_parquet(ROOT / f"outputs/wset_vectors_{n}{vsuf}.parquet")
    nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet").set_index("id").loc[vec.id].reset_index()
    X, xcols = wset_matrix(vec)
    keep = np.where(np.abs(X).sum(1) > 0)[0]                      # reviews with >=1 WSET signal
    print(f"WSET vector dims={X.shape[1]}; reviews with signal {len(keep)}/{len(X)}")
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    raw["id"] = raw.description.map(review_id)
    desc = raw.drop_duplicates("id").set_index("id").loc[vec.id, "description"].values
    spaces = {}
    for m in models:
        spaces[m] = np.load(ROOT / f"outputs/emb_{m}_{n}.npy")
    tf = TfidfVectorizer(min_df=5, max_features=20000, stop_words="english", sublinear_tf=True).fit_transform(desc)
    spaces["tfidf_svd"] = TruncatedSVD(128, random_state=SEED).fit_transform(tf).astype("float32")
    spaces["random"] = rng.normal(size=(len(X), 128)).astype("float32")
    res = {"n": len(X), "wset_dims": X.shape[1], "signal_rows": int(len(keep))}
    Xk = X[keep]
    # ---- RSA & neighbour overlap (on rows with signal)
    nbX = cos_nn(Xk)
    res["rsa"], res["knn_overlap@10"] = {}, {}
    for name, E in spaces.items():
        Ek = E[keep]
        res["rsa"][name] = rsa(Xk, Ek)
        res["knn_overlap@10"][name] = overlap(nbX, cos_nn(Ek))
    # ---- probes: embedding -> each WSET dimension (5-fold), per-dimension score
    kf = KFold(5, shuffle=True, random_state=SEED)
    probe = {}
    dimcols = {"c_" + c[2:].split("__")[0] for c in xcols if c.startswith("a_")}
    clus = sorted(c for c in vec if c.startswith("c_"))
    for name in [m for m in spaces if m != "random"] + ["random"]:
        E = spaces[name]
        out = {}
        for c in clus:
            y = vec[c].values
            if y.sum() < 100 or y.sum() > len(y) - 100:
                continue
            p = cross_val_predict(LogisticRegression(max_iter=300, C=1.0), E, y, cv=3, method="predict_proba")[:, 1]
            out[c] = roc_auc_score(y, p)
        for s in [c for c in vec if c.startswith("s_")]:
            ok = vec[s].notna().values
            if ok.sum() < 500:
                continue
            p = cross_val_predict(Ridge(alpha=1.0), E[ok], vec.loc[ok, s].values, cv=kf)
            out[s] = r2_score(vec.loc[ok, s].values, p)
        probe[name] = out
    res["probe_embedding_to_wset"] = probe
    # ---- probes: which space predicts metadata better? (accuracy of top class / points R2)
    meta = {}
    targets = {"variety_group": nm.variety_group, "country": nm.country.where(nm.country.map(nm.country.value_counts()) > 200, "Other"),
               "colour": nm.colour}
    spaces_m = dict(spaces); spaces_m["wset"] = X
    for name, E in spaces_m.items():
        row = {}
        for t, y in targets.items():
            p = cross_val_predict(LogisticRegression(max_iter=200), E, y.values, cv=3)
            row[t] = float((p == y.values).mean())
            row[t + "_majority"] = float(y.value_counts(normalize=True).iloc[0])
        pts = nm.points.values.astype(float)
        ok = ~np.isnan(pts)
        p = cross_val_predict(Ridge(alpha=1.0), E[ok], pts[ok], cv=kf)
        row["points_r2"] = float(r2_score(pts[ok], p))
        meta[name] = row
    res["probe_metadata"] = meta
    # ---- dimensionality reduction (2D & 3D) for WSET space and each embedding
    import umap
    sub = np.sort(rng.choice(len(X), min(8000, len(X)), replace=False))
    coords = pd.DataFrame({"id": vec.id.values[sub]})
    trust = {}
    for name, M in {**{"wset": X}, **{m: spaces[m] for m in models}}.items():
        Ms = M[sub]
        for dim in (2, 3):
            pc = PCA(dim, random_state=SEED).fit_transform(Ms)
            um = umap.UMAP(n_components=dim, n_neighbors=30, min_dist=0.2, metric="cosine", random_state=SEED).fit_transform(Ms)
            for meth, Z in (("pca", pc), ("umap", um)):
                for k in range(dim):
                    coords[f"{name}_{meth}{dim}_{k}"] = Z[:, k]
                trust[f"{name}_{meth}{dim}"] = float(trustworthiness(Ms[:2000], Z[:2000], n_neighbors=10))
    res["trustworthiness"] = trust
    coords.to_parquet(ROOT / f"outputs/coords_{n}{vsuf}.parquet", index=False)
    json.dump(res, open(ROOT / f"outputs/compare_{n}{vsuf}.json", "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("rsa", "knn_overlap@10", "probe_metadata", "trustworthiness")}, indent=1, default=float))


if __name__ == "__main__":
    main()
