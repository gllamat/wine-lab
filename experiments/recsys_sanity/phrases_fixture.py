"""Reference (Python) implementation of the dashboard's typed-note path, reading the SHIPPED dashboard/data/phrases.json. Writes outputs/phrases_parity.json for dashboard/phrases_parity.js."""
import sys, json, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wsetkit import review_id
P = json.load(open(ROOT / "dashboard/data/phrases.json")); meta = json.load(open(ROOT / "dashboard/data/recsys_meta.json")); ids = meta["ids"]


def lemma_(w):
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "zes", "sses")): return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")): return w[:-1]
    return w


def phrases(text):
    t = unicodedata.normalize("NFKD", str(text).lower()); t = "".join(c for c in t if not unicodedata.combining(c)).replace("'", "").replace("-", " "); out = []
    for clause in re.split(r"[.;:,!?()]", t):
        w = [lemma_(x) for x in re.findall(r"[a-z]+", clause)]; out += w + [w[i] + " " + w[i + 1] for i in range(len(w) - 1)]
    return out


nb = len(P["questions"]); loff = np.cumsum([0] + P["levels"]); yes, lev = {}, {}
for q, r in enumerate(P["rules"]):
    for pi, pr in r: yes.setdefault(P["phrases"][pi], []).append((q, pr / 100))
for c, r in enumerate(P["lrules"]):
    for pi, pr in r: lev.setdefault(P["lphrases"][pi], []).append((c, pr / 100))


def typed_z(text, cue):
    sc = np.array(P["mu"], np.float32); cl = np.zeros(loff[-1], np.float32)
    for p in set(phrases(text)):
        for q, pr in yes.get(p, []): sc[q] = max(sc[q], pr)
        for c, pr in lev.get(p, []): cl[c] = max(cl[c], pr)
    z = np.zeros(nb + len(P["scales"]), np.float32); z[:nb] = (sc - np.array(P["mu"])) / np.array(P["sd"])
    for k, s in enumerate(P["scales"]):
        w = cl[loff[k]:loff[k + 1]]
        lv = (w * np.arange(P["levels"][k])).sum() / w.sum() if w.sum() > 0 else (cue[s] if s in cue else P["lmu"][k])
        z[nb + k] = (lv - P["lmu"][k]) / P["lsd"][k]
    return z


raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"]); raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id")
rng = np.random.RandomState(4); pick = rng.choice(len(ids), 40, replace=False); rows = []
extra = ["Pale lemon. Crisp lime, green apple, wet stones. High acidity, light body, dry. Long finish.", "Deep ruby. Black cherry, vanilla and cedar from oak, firm tannins, full body, long finish.", ""]
for t in [raw.loc[ids[i], "description"] for i in pick] + extra:
    cue = {"sweetness": 0.0, "acidity": 3.0} if t.startswith("Pale lemon") else {}
    rows.append(dict(text=t, cue=cue, tokens=phrases(t)[:60], z=[round(float(x), 4) for x in typed_z(t, cue)]))
json.dump(rows, open(ROOT / "outputs/phrases_parity.json", "w")); print("fixture", len(rows))
