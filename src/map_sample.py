"""Deterministic SAT mapping of N fresh reviews -> readable CSV/XLSX (local only: contains review text).
Excludes reviews already used in the Jev pilots. Usage: map_sample.py [N]"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import Lexicon, ROOT, review_id, quality_band, readiness_outcome

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet")
raw["id"] = raw.description.map(review_id)
raw = raw.drop_duplicates("id").set_index("id")
nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet").set_index("id")
used = set()
for f in ("outputs/jev_pilot.jsonl", "outputs/jev_pilot_v1_nonSAT.jsonl"):
    used |= {json.loads(l)["id"] for l in open(ROOT / f)}
ids = raw.index.difference(list(used)).to_series().sample(N, random_state=2026).tolist()
lex = Lexicon()
NICE = {"tired": "tired / past its best", "fully_developed": "fully developed"}
fmt = lambda n: NICE.get(n, str(n).replace("-", "(-)", 1) if str(n).endswith("-") else str(n).replace("_", " "))
fmt = lambda n: NICE.get(n, (n[:-1] + "(-)") if n.endswith("-") else (n[:-1] + "(+)") if n.endswith("+") else n.replace("_", " "))


def level(pos, s):
    v = [h["f2"] for h in pos if h["kind"] == "structure" and h["f1"] == s]
    return fmt(lex.scales[s][int(round(np.mean(v))) - 1]) if v else ""


def tiers(pos, pred):
    out = {"primary": {}, "secondary": {}, "tertiary": {}}
    for h in pos:
        if h["kind"] == "aroma" and pred(h["scope"]):
            out[lex.tier[h["f1"]]].setdefault(h["f1"].replace("_", " "), set()).add(h["f2"].replace("_", " "))
    return {t: "; ".join(f"{c}: {', '.join(sorted(d))}" for c, d in sorted(v.items())) for t, v in out.items()}


rows = []
t0 = time.perf_counter()
for i in ids:
    text = raw.loc[i, "description"]; m = nm.loc[i]
    toks, cov, hits, yrs = lex.match(text, m.colour)
    pos = [h for h in hits if not h["neg"]]
    cat = lambda f: ", ".join(sorted({h["f2"] for h in pos if h["kind"] == "cat" and h["f1"] == f}))
    fl = lambda k: any(h["kind"] == "readiness" and h["f1"] == k for h in pos)
    rd = readiness_outcome(fl("drink_now"), fl("can_age"), fl("too_old"))
    nose, pal, un = tiers(pos, lambda s: s in ("nose", "both")), tiers(pos, lambda s: s in ("palate", "both")), tiers(pos, lambda s: s == "unspecified")
    rows.append({
        "id": i, "variety": m.variety_norm, "country": m.country, "region": m.region_1, "vintage": m.vintage, "points": m.points, "wine colour": m.colour,
        "APP clarity": cat("clarity"), "APP intensity": cat("appearance_intensity"), "APP colour": cat("colour"),
        "NOSE condition": "unclean" if cat("condition") else "", "NOSE intensity": level(pos, "nose_intensity"),
        "NOSE primary": nose["primary"], "NOSE secondary": nose["secondary"], "NOSE tertiary": nose["tertiary"],
        "NOSE development": level(pos, "development"),
        "PALATE sweetness": level(pos, "sweetness"), "PALATE acidity": level(pos, "acidity"), "PALATE tannin": level(pos, "tannin"),
        "PALATE alcohol": level(pos, "alcohol"), "PALATE body": level(pos, "body"), "PALATE mousse": level(pos, "mousse"),
        "PALATE flavour intensity": level(pos, "flavour_intensity"),
        "PALATE primary": pal["primary"], "PALATE secondary": pal["secondary"], "PALATE tertiary": pal["tertiary"],
        "PALATE finish": level(pos, "finish"),
        "not attributed (nose/palate unclear) primary": un["primary"], "not attributed secondary": un["secondary"], "not attributed tertiary": un["tertiary"],
        "CONCL quality (proxy from points)": (quality_band(m.points) or "").replace("_", " "),
        "CONCL readiness": {"drink_now_potential": "can drink now, has potential", "drink_now": "drink now (potential not stated)", "not_ready_potential": "too young, has potential", "too_old": "too old", None: ""}[rd],
        "evidence (matched words)": "; ".join(sorted({h["surface"] for h in pos})), "negated words (ignored)": "; ".join(sorted({h["surface"] for h in hits if h["neg"]})),
        "review text": text})
secs = time.perf_counter() - t0
df = pd.DataFrame(rows)
df.to_csv(ROOT / "outputs/deterministic_1000.csv", index=False)
how = pd.DataFrame({"How to read this sheet": [
    "Deterministic keyword mapping (no AI model, no API cost) of reviews NOT used in the Jev pilots.",
    "Taxonomy: WSET L3 Systematic Approach to Tasting Wine (3W v2.0, 2016): appearance, nose, palate, conclusions.",
    "Blank cell = the review does not state it (not 'medium'). Level names follow the SAT scales.",
    "NOSE/PALATE aroma columns list SAT clusters (primary/secondary/tertiary) with matched descriptors; words not tied to nose or palate are in the 'not attributed' columns.",
    "Quality is a proxy from the critic's points, not a tasting judgement. Readiness uses stated cues only.",
    "Cue-to-level mappings are project heuristics, not part of the SAT, and not yet validated by a WSET-qualified taster.",
    "Contains scraped review text (Wine Enthusiast via Kaggle, reportedly non-commercial licence): keep local, do not publish."]})
with pd.ExcelWriter(ROOT / "outputs/deterministic_1000.xlsx", engine="openpyxl") as w:
    df.to_excel(w, sheet_name="Mapped reviews", index=False); how.to_excel(w, sheet_name="How to read", index=False)
    ws = w.sheets["Mapped reviews"]; ws.freeze_panes = "H2"; ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        width = min(60, max(10, max(len(str(c.value or "")) for c in col[:60]) + 2)); ws.column_dimensions[col[0].column_letter].width = width
    w.sheets["How to read"].column_dimensions["A"].width = 150
st = {c: round((df[c].astype(str).str.len() > 0).mean() * 100, 1) for c in df.columns if c.startswith(("APP", "NOSE", "PALATE", "CONCL")) and "primary" not in c and "secondary" not in c and "tertiary" not in c}
print(f"mapped {len(df)} reviews in {secs:.2f} s ({secs / len(df) * 1000:.2f} ms each) | cost $0.00 | overlap with Jev pilots: 0")
print("share of reviews where each field is stated (%):"); [print(f"   {k:34s}{v:5.1f}") for k, v in st.items()]
print("any NOSE aroma:", round((df['NOSE primary'].str.len() + df['NOSE secondary'].str.len() + df['NOSE tertiary'].str.len() > 0).mean() * 100, 1), "% | any PALATE flavour:", round((df['PALATE primary'].str.len() + df['PALATE secondary'].str.len() + df['PALATE tertiary'].str.len() > 0).mean() * 100, 1), "%")
