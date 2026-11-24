"""Jev-aligned taste vector for the 25,000 wines of the recommender index.
Each wine gets ~224 numbers: Jev's 216 yes/no answers and the expected level of 8 structure scales (sweetness, acidity, tannin, alcohol, body, flavour intensity, finish, nose intensity).
Jev labelled only ~37% of the index, so the rest is PREDICTED from the note's words (TF-IDF + ridge, the 'word-counting' baseline of the distillation notebook).
No leakage: the 62k labelled notes are split by winery into 5 folds; a labelled wine's vector comes from the model that did NOT see its fold; unlabelled wines get the average of the 5 models.
Writes jev_vectors.npy (n_index x 224, float32, aligned with dashboard/data/recsys_meta.json ids) and jev_vectors_meta.json.
Usage: python experiments/recsys_sanity/jev_vector.py"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wsetkit import review_id

H = Path(__file__).resolve().parent
SCALES = ["sweetness", "acidity", "tannin", "alcohol", "body", "flavour_intensity", "finish", "nose_intensity"]
pack = pd.read_parquet(ROOT / "distill/pack/wset_distill_pack.parquet"); schema = json.load(open(ROOT / "distill/pack/wset_distill_schema.json"))
BIN = ["b::" + q for q in schema["binary"]]
Yb = pack[BIN].to_numpy(np.float32)
Yl = np.stack([(np.array(pack["g::level__" + s].tolist(), dtype=np.float32) * np.arange(len(schema["groups"]["level__" + s]))).sum(1) for s in SCALES], 1)
Y = np.hstack([Yb, Yl]); print("labelled notes", len(pack), "| targets", Y.shape[1])
meta = json.load(open(ROOT / "dashboard/data/recsys_meta.json")); ids = np.array(meta["ids"]); n = len(ids)
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"]); raw["id"] = raw.description.map(review_id)
raw = raw.drop_duplicates("id").set_index("id"); idx_text = raw.loc[ids, "description"].tolist()
pos = {i: k for k, i in enumerate(pack.id)}; in_pack = np.array([i in pos for i in ids]); print("index wines with Jev labels:", int(in_pack.sum()), "of", n)

t0 = time.time()
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X = tf.fit_transform(pack.description); Xi = tf.transform(idx_text); print("tfidf", X.shape, round(time.time() - t0), "s")
key = pack.winery.fillna(pack.id).astype(str); fold = np.array([int(__import__("hashlib").md5(("f" + k).encode()).hexdigest(), 16) % 5 for k in key])
V = np.zeros((n, Y.shape[1]), np.float32); acc = np.zeros((n, Y.shape[1]), np.float32); n_acc = 0
lab_rows = np.where(in_pack)[0]; lab_src = np.array([pos[ids[i]] for i in lab_rows])
for f in range(5):
    t1 = time.time(); tr = fold != f
    m = Ridge(alpha=1.0, solver="sparse_cg", max_iter=200).fit(X[tr], Y[tr])
    P = m.predict(Xi).astype(np.float32); acc += P; n_acc += 1
    sel = fold[lab_src] == f; V[lab_rows[sel]] = P[lab_rows[sel]]                      # labelled wine in this fold: prediction from the model that did not train on it
    print(f"fold {f}: {int(tr.sum())} training notes, {round(time.time() - t1)} s")
unl = ~in_pack; V[unl] = acc[unl] / n_acc
V[:, :len(BIN)] = np.clip(V[:, :len(BIN)], 0, 1)
np.save(H / "jev_vectors.npy", V)
json.dump(dict(binary=schema["binary"], scales=SCALES, n=int(n), labelled_in_index=int(in_pack.sum()), note="out-of-fold for labelled wines; mean of 5 fold models for unlabelled wines"), open(H / "jev_vectors_meta.json", "w"))
# quick check: how well do the out-of-fold predictions match Jev's true answers (labelled index wines)?
from sklearn.metrics import roc_auc_score
T = Yb[lab_src]; P = V[lab_rows][:, :len(BIN)]; a = [roc_auc_score(T[:, j] >= .5, P[:, j]) for j in range(T.shape[1]) if (T[:, j] >= .5).sum() >= 20 and (T[:, j] < .5).sum() >= 20]
print(f"out-of-fold mean AUC on the labelled wines of the index: {np.mean(a):.3f} over {len(a)} questions | level corr (sweetness) {np.corrcoef(Yl[lab_src, 0], V[lab_rows, len(BIN)])[0, 1]:.2f}, body {np.corrcoef(Yl[lab_src, 4], V[lab_rows, len(BIN) + 4])[0, 1]:.2f}")
print("done in", round(time.time() - t0), "s")
