"""Export a fitted LITE guess model to JSON for the browser (dashboard/data/guess_model.json) + a Python-vs-JS parity fixture.
Usage: export_guess.py <model.pkl> [--fixture N]"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon, review_id, resolve_ripeness
from guess_features import featurize_hits
import guess_model as G


def r(a, nd=5): return np.round(np.asarray(a, dtype=float), nd).tolist()


def export(M, spec, path):
    cmb = M["combo"]; Mw = M["Mw"].tocoo()
    j = dict(version=M["version"], spec=spec, mu=r(M["mu"], 5), sd=r(M["sd"], 5),
             variety=dict(classes=[str(c) for c in M["var_classes"]], W=r(M["var_W"]), b=r(M["var_b"]), T=M["T_var"], groups=M["groups"], G=np.asarray(M["G"]).argmax(1).tolist()),
             country=dict(classes=[str(c) for c in M["country_classes"]], W=r(M["cty_W"]), b=r(M["cty_b"])),
             place=dict(classes=[str(c) for c in M["place_classes"]], W=r(M["place_W"]), b=r(M["place_b"]), T=M["T_place"], xy=[[None if np.isnan(v) else round(float(v), 3) for v in row] for row in M["place_xy"]]),
             reg={k: dict(W=r(v["W"]), b=round(v["b"], 5), edges=r(v["edges"], 5), q=r(v["q"], 5)) for k, v in M["reg"].items()},
             producer=dict(wineries=[str(w) for w in M["wineries"]], combo=cmb.tolist(), mw_row=Mw.row.tolist(), mw_col=Mw.col.tolist(), mw_val=r(Mw.data, 5), mu_w=r(M["mu_w"], 3),
                           pca_mean=r(M["pca_mean"], 5), pca_comp=r(M["pca_comp"], 5), tau=M["tau"], p_other=M["p_other"]))
    json.dump(j, open(path, "w"), separators=(",", ":"))
    return path.stat().st_size / 1e6


def fixture(M, spec, n, path):
    from guess_model import predict
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"]).sample(n, random_state=11)
    lex = Lexicon(); rows = []
    for t in raw.description:
        toks, cov, hits, yrs = lex.match(t, None)
        x = featurize_hits(hits, yrs, toks, spec, resolve_ripeness); o = predict(M, x[None])
        top = lambda P, cl, k=3: [[str(cl[j]), round(float(P[0, j]), 4)] for j in np.argsort(-P[0])[:k]]
        la, lo, r50, r80 = G.place_point(M, o["Pp"], 0)
        rows.append(dict(text=t, variety=top(o["Pv"], M["var_classes"]), place=top(o["Pp"], M["place_classes"]), group=top(o["Pg"], np.array(M["groups"])),
                         producer=top(o["Pw"], M["wineries"]), vintage=r(o["vintage"][0], 3), price=r(o["price"][0], 3), points=r(o["points"][0], 3), point=[round(la, 3), round(lo, 3), round(r50), round(r80)]))
    json.dump(rows, open(path, "w")); return len(rows)


if __name__ == "__main__":
    M = pickle.load(open(sys.argv[1], "rb")); spec = json.load(open(ROOT / "outputs/guess_spec.json"))
    print("model json MB", round(export(M, spec, ROOT / "dashboard/data/guess_model.json"), 2))
    if "--fixture" in sys.argv: print("fixture", fixture(M, spec, int(sys.argv[sys.argv.index("--fixture") + 1]), ROOT / "outputs/guess_parity.json"))
