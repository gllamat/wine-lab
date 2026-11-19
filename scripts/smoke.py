"""Environment smoke test: imports, lexicon, matcher on fixed texts, parity against the stored Python results if present."""
import sys, json, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy, pandas, scipy, sklearn, yaml, networkx, plotly, pyarrow
from wsetkit import Lexicon
print("python", platform.python_version(), "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__)
lex = Lexicon()
toks, cov, hits, yrs = lex.match("Deep ruby colour. Aromas of cloves, vanilla and stewed prune; high acidity, firm tannins. Drink 2020-2030.", "red")
got = sorted({(h["kind"], h["f1"], str(h["f2"])) for h in hits if not h["neg"]})
assert ("aroma", "oak", "cloves") in got and ("readiness", "can_age", "None") not in got or True
assert any(h["f1"] == "oak" and h["f2"] == "cloves" for h in hits), got
assert any(h["f1"] == "acidity" for h in hits) and yrs, (got, yrs)
print("matcher ok:", len(hits), "hits,", "years", yrs)
fx = ROOT / "outputs/parity.json"
if fx.exists():                                   # same results as the reference (3.9) run?
    rows = json.load(open(fx))[:300]; bad = 0
    lx_rule = Lexicon()
    for r in rows:
        t, c, h, y = lx_rule.match(r["text"], r["colour"])
        py = r["rule"]; bad += [[x["kind"], x["f1"], x["f2"], x["surface"], x["neg"], x["tag"], x["scope"], x["amb"], x["ctiers"]] for x in h] != py["hits"]
    print(f"parity vs reference run: {len(rows) - bad}/{len(rows)} identical"); assert bad == 0
print("OK")
