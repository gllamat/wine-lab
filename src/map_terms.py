"""Phase 3: map unmatched review terms to WSET aroma descriptors with (old) BERT / MiniLM embeddings.
Validation: leave-one-form-out on the lexicon itself (no human labels needed). Deterministic.
Usage: map_terms.py <minilm|bert> <N>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import Lexicon, ROOT, tokenize
from embed import load, embed_texts

STOPLIKE = set("flavor flavour aroma fruit finish palate note acidity tannin nose show blend well texture offer good touch like "
               "character give full feel mouth vineyard white style made come red balance taste fine make great vintage "
               "mouthfeel delicious along slightly yet still little lead almost drink wine grape bottle year drinking"
               .split())


def main():
    name, n = sys.argv[1], sys.argv[2]
    lex = Lexicon()
    tok, mdl = load(name)
    # --- descriptor prototypes from single/multi-word forms
    forms = [(" ".join(k), v[1], v[2]) for k, v in lex.entries.items() if v[0] == "aroma"]
    F = embed_texts([f for f, _, _ in forms], tok, mdl, bs=128, maxlen=16)
    cl = np.array([c for _, c, _ in forms]); ds = np.array([f"{c}/{d}" for _, c, d in forms])
    dnames = sorted(set(ds)); didx = {d: i for i, d in enumerate(dnames)}
    dlab = np.array([didx[x] for x in ds])
    dcl = {d: d.split("/")[0] for d in dnames}

    def protos(exclude=None):
        P = np.zeros((len(dnames), F.shape[1]), dtype="float32")
        cnt = np.zeros(len(dnames))
        for i, l in enumerate(dlab):
            if exclude is not None and i == exclude:
                continue
            P[l] += F[i]; cnt[l] += 1
        P /= np.maximum(cnt, 1)[:, None]
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        return P, cnt

    # --- leave-one-out validation (single-token forms of descriptors with >=3 forms)
    res = []
    P_full, cnt_full = protos()
    for i, (f, c, d) in enumerate(forms):
        key = f"{c}/{d}"
        if len(f.split()) != 1 or cnt_full[didx[key]] < 3:
            continue
        P, cnt = protos(exclude=i)
        P[didx[key]] = (P_full[didx[key]] * cnt_full[didx[key]] - F[i]) / (cnt_full[didx[key]] - 1)
        P[didx[key]] /= np.linalg.norm(P[didx[key]])
        s = P @ F[i]
        top = np.argsort(-s)[:2]
        res.append(dict(form=f, true=key, pred=dnames[top[0]], sim=float(s[top[0]]), margin=float(s[top[0]] - s[top[1]]),
                        ok_desc=dnames[top[0]] == key, ok_cluster=dcl[dnames[top[0]]] == c))
    r = pd.DataFrame(res)
    r.to_csv(ROOT / f"outputs/loo_{name}.csv", index=False)
    print(f"[{name}] LOO n={len(r)} acc_descriptor={r.ok_desc.mean():.3f} acc_cluster={r.ok_cluster.mean():.3f}")
    # pick the lowest sim threshold with cluster-level precision >= 0.8 (else report best available)
    best = None
    for t in np.arange(0.3, 0.95, 0.01):
        sub = r[r.sim >= t]
        if len(sub) >= 30 and sub.ok_cluster.mean() >= 0.8:
            best = (float(t), len(sub) / len(r), float(sub.ok_cluster.mean()))
            break
    print(f"[{name}] threshold for >=80% cluster precision:", best)
    json.dump({"threshold": best}, open(ROOT / f"outputs/threshold_{name}.json", "w"))
    if best is None:
        print("no usable threshold; skipping term mapping"); return
    thr = best[0]
    # --- candidate terms: frequent unmatched unigrams not in stop-like or place/grape vocab
    um = pd.read_csv(ROOT / f"outputs/unmatched_unigrams_{n}.csv")
    um = um[(um["count"] >= 8)]
    nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet", columns=["country", "province", "region_1", "winery", "variety"])
    names = set()
    for col in nm:
        for v in nm[col].dropna().unique():
            names.update(tokenize(str(v)))
    cands = [t for t in um.term if t not in STOPLIKE and t not in names and (t,) not in lex.entries]
    C = embed_texts(cands, tok, mdl, bs=128, maxlen=16)
    S = C @ P_full.T
    rows = []
    for i, t in enumerate(cands):
        order = np.argsort(-S[i])[:2]
        rows.append(dict(term=t, desc=dnames[order[0]], sim=float(S[i, order[0]]), margin=float(S[i, order[0]] - S[i, order[1]]),
                         count=int(um.set_index("term").loc[t, "count"]), accepted=bool(S[i, order[0]] >= thr)))
    m = pd.DataFrame(rows)
    m.to_csv(ROOT / f"outputs/term_map_{name}.csv", index=False)
    acc = m[m.accepted]
    print(f"[{name}] candidates={len(m)} accepted={len(acc)} (sim>={thr:.2f})")
    extra = {r.term: ["aroma", r.desc.split("/")[0], r.desc.split("/")[1]] for r in acc.itertuples()}
    json.dump(extra, open(ROOT / f"outputs/extra_terms_{name}.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
