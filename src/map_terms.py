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


NONAROMA = set("balance balanced soft smooth silky velvety firm tight rich round ripe fresh bright light heavy bold lean thin juicy "
               "savory elegant complex structured structure depth richness weight quality plump polished textured rustic brisk "
               "bitter sweet dry crisp lively flavored flavoured fruity fruitiness ripeness body alcohol acid warm exotic "
               "scent smell bottling mix accent accented variety grown produced making dish skin fermented grilled "
               "hint bit deliver alongside seem best showing plenty attractive much quite add too although enough close pair "
               "even bring after aftertaste imported followed offering only provide rather".split())


def main():
    name, n = sys.argv[1], sys.argv[2]
    lex = Lexicon()
    tok, mdl = load(name)
    # --- descriptor prototypes from single/multi-word forms
    forms = [(" ".join(k), e[1], e[2]) for k, v in lex.entries.items() for e in v if e[0] == "aroma"]
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

    # --- negative class: structure cues + generic words (things that are NOT aromas)
    negf = sorted({" ".join(k) for k, v in lex.entries.items() if all(e[0] != "aroma" for e in v)} | STOPLIKE | NONAROMA)
    N = embed_texts(negf, tok, mdl, bs=128, maxlen=16)
    P_full, cnt_full = protos()

    def decide(vec, P, Nm, excl_neg=None):
        sd = P @ vec
        top = np.argsort(-sd)[:2]
        sn = Nm @ vec
        if excl_neg is not None:
            sn = np.delete(sn, excl_neg)
        return top[0], float(sd[top[0]]), float(sn.max())

    # --- leave-one-out: aroma forms (should map to right cluster) + negative forms (should be rejected)
    res = []
    for i, (f, c, d) in enumerate(forms):
        key = f"{c}/{d}"
        if len(f.split()) != 1 or cnt_full[didx[key]] < 3:
            continue
        P, cnt = protos(exclude=i)
        P[didx[key]] = (P_full[didx[key]] * cnt_full[didx[key]] - F[i]) / (cnt_full[didx[key]] - 1)
        P[didx[key]] /= np.linalg.norm(P[didx[key]])
        j, sd, sn = decide(F[i], P, N)
        res.append(dict(form=f, kind="aroma", true=key, pred=dnames[j], sim=sd, neg_sim=sn,
                        ok_cluster=dcl[dnames[j]] == c))
    for i, f in enumerate(negf):
        if len(f.split()) != 1:
            continue
        j, sd, sn = decide(N[i], P_full, N, excl_neg=i)
        res.append(dict(form=f, kind="neg", true="NONE", pred=dnames[j], sim=sd, neg_sim=sn, ok_cluster=False))
    r = pd.DataFrame(res)
    r.to_csv(ROOT / f"outputs/loo_{name}.csv", index=False)
    na, nn = (r.kind == "aroma").sum(), (r.kind == "neg").sum()
    # prior: real candidates are mostly non-aroma -> weight negatives x3 when estimating precision
    def stats(t, m):
        acc = r[(r.sim >= t) & (r.sim - r.neg_sim >= m)]
        tp = acc[(acc.kind == "aroma") & acc.ok_cluster].shape[0]
        fp = acc.shape[0] - tp
        w = 3.0 * (na / max(nn, 1))
        prec = tp / max(tp + fp * w if False else tp + (acc[acc.kind == "neg"].shape[0] * w) + acc[(acc.kind == "aroma") & ~acc.ok_cluster].shape[0], 1)
        return prec, tp / max(na, 1)
    best = None
    for t in np.arange(0.3, 0.95, 0.02):
        for m in (0.0, 0.03, 0.06, 0.1):
            prec, rec = stats(t, m)
            if prec >= 0.8 and rec >= 0.05 and (best is None or rec > best[3]):
                best = (float(t), float(m), prec, rec)
    print(f"[{name}] LOO aroma n={na} neg n={nn}; best (thr, margin, weighted precision, recall): {best}")
    json.dump({"best": best}, open(ROOT / f"outputs/threshold_{name}.json", "w"))
    if best is None:
        print("no usable threshold; skipping term mapping"); return
    thr, mar = best[0], best[1]
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
    SN = C @ N.T
    rows = []
    cnts = um.set_index("term")["count"]
    for i, t in enumerate(cands):
        j = int(np.argmax(S[i])); sd = float(S[i, j]); sn = float(SN[i].max())
        rows.append(dict(term=t, desc=dnames[j], sim=sd, neg_sim=sn, nearest_neg=negf[int(np.argmax(SN[i]))],
                         count=int(cnts[t]), accepted=bool(sd >= thr and sd - sn >= mar)))
    m = pd.DataFrame(rows)
    m.to_csv(ROOT / f"outputs/term_map_{name}.csv", index=False)
    acc = m[m.accepted]
    print(f"[{name}] candidates={len(m)} accepted={len(acc)} (sim>={thr:.2f}, margin>={mar:.2f})")
    extra = {r.term: ["aroma", r.desc.split("/")[0], r.desc.split("/")[1]] for r in acc.itertuples()}
    json.dump(extra, open(ROOT / f"outputs/extra_terms_{name}.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
