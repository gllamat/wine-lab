"""Export the Python lexicon for the browser matcher + generate a Python-vs-JS parity fixture."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import Lexicon, ROOT, review_id
import yaml


def extras():
    f = ROOT / "outputs/extra_terms_minilm.json"
    return {k: tuple(v) for k, v in json.load(open(f)).items()} if f.exists() else None


def lexicon_json():
    lex = Lexicon(extras())
    s = yaml.safe_load(open(ROOT / "lexicon/structure.yaml"))
    return lex, dict(
        version=1,
        entries=[[" ".join(k), [list(t) for t in v]] for k, v in sorted(lex.entries.items())],
        bound=lex.bound, heads=lex.heads, year_re=s["readiness"]["year_range_regex"],
        scales={k: v for k, v in lex.scales.items()},
        descriptors=[list(d) for d in lex.descriptors])


def write_lexicon(path):
    lex, j = lexicon_json()
    json.dump(j, open(path, "w"), separators=(",", ":"))
    return lex, j


def parity_fixture(n=3000):
    df = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    df = df.sample(n, random_state=7).reset_index(drop=True)
    out = []
    lx_all, lx_rule = Lexicon(extras()), Lexicon()
    for t in df.description:
        row = {"text": t}
        for name, lx in (("embed", lx_all), ("rule", lx_rule)):
            toks, cov, hits, yrs = lx.match(t)
            row[name] = dict(toks=toks, hits=[[h["kind"], h["f1"], h["f2"], h["surface"], h["neg"], h["tag"]] for h in hits],
                             years=[list(y) for y in yrs])
        out.append(row)
    json.dump(out, open(ROOT / "outputs/parity.json", "w"))
    print("parity fixture", len(out))


if __name__ == "__main__":
    write_lexicon(ROOT / "dashboard/data/lexicon.json")
    parity_fixture()
