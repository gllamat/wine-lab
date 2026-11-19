"""Export the Python lexicon for the browser matcher + a Python-vs-JS parity fixture (with wine colour)."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import Lexicon, ROOT, review_id, resolve_ripeness
import yaml


def extras():
    f = ROOT / "outputs/extra_terms_minilm.json"
    return json.load(open(f)) if f.exists() else None


def lexicon_json():
    lex = Lexicon(extras())
    s = yaml.safe_load(open(ROOT / "lexicon/structure.yaml"))
    return lex, dict(
        version=2,
        entries=[[" ".join(k), [list(t) for t in v]] for k, v in sorted(lex.entries.items())],
        bound=lex.bound, heads=lex.heads, year_re=s["readiness"]["year_range_regex"],
        scales=dict(lex.scales), cat_fields=lex.cat_fields, tier=lex.tier,
        descriptors=[[c, d, lex.tier[c]] for c, d in lex.descriptors])


def write_lexicon(path):
    lex, j = lexicon_json()
    json.dump(j, open(path, "w"), separators=(",", ":"))
    return lex, j


def parity_fixture(n=3000):
    df = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    colour = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet", columns=["id", "colour"]).set_index("id").colour
    df = df.sample(n, random_state=7).reset_index(drop=True)
    out = []
    lx_all, lx_rule = Lexicon(extras()), Lexicon()
    for t in df.description:
        c = colour.get(review_id(t))
        row = {"text": t, "colour": c}
        for name, lx in (("embed", lx_all), ("rule", lx_rule)):
            toks, cov, hits, yrs = lx.match(t, c)
            row[name] = dict(toks=toks, hits=[[h["kind"], h["f1"], h["f2"], h["surface"], h["neg"], h["tag"], h["scope"], h["amb"], h["ctiers"]] for h in hits],
                             years=[list(y) for y in yrs],
                             ripeness=list(resolve_ripeness([h["f2"] for h in hits if h["kind"] == "aroma" and not h["neg"] and (h["f1"] == "fruit_ripeness" or (h["f1"] in ("tertiary_red", "tertiary_white") and h["f2"] in ("dried_fruit", "cooked_fruit")))])))
        out.append(row)
    json.dump(out, open(ROOT / "outputs/parity.json", "w"))
    print("parity fixture", len(out))


if __name__ == "__main__":
    write_lexicon(ROOT / "dashboard/data/lexicon.json")
    parity_fixture()
