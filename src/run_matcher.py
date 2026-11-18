"""Apply the deterministic SAT matcher. Usage: run_matcher.py [N|all] [--extra path.json]"""
import sys, json, collections, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import Lexicon, review_id, quality_band, readiness_outcome, ageing_outcome, ROOT

SRC = ROOT.parent / "datasets/wine_reviews_deduped.parquet"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
STOP = set("a an the and or of in on with to for is are was be it its this that these those as at by from but has have had "
           "not no than then very more most some any into out up over through while which who what when where how about "
           "also just can will would could should may might one two three four five six seven eight nine ten there their "
           "they them we you your our i me my he she his her him s t wine drink now year age".split())
BOUNDS = ("|", "#")


def main():
    n = sys.argv[1] if len(sys.argv) > 1 else "20000"
    extra = None
    if "--extra" in sys.argv:
        extra = json.load(open(sys.argv[sys.argv.index("--extra") + 1]))
    df = pd.read_parquet(SRC)
    df["id"] = df.description.map(review_id)
    df = df.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    colour = pd.read_parquet(OUT / "reviews_norm.parquet", columns=["id", "colour"]).set_index("id").colour
    if n != "all":
        df["_h"] = df.id.map(lambda x: hashlib.md5(("seed42" + x).encode()).hexdigest())
        df = df.sort_values("_h").head(int(n)).sort_values("id").drop(columns="_h").reset_index(drop=True)
    lex = Lexicon(extra)
    pd.DataFrame([(a, str(b), str(c)) for a, b, c in lex.conflicts], columns=["form", "kept", "dropped"]).to_csv(OUT / "lexicon_conflicts.csv", index=False)
    descs = lex.descriptors
    clusters = list(dict.fromkeys(c for c, _ in descs))
    tiers = ["primary", "secondary", "tertiary"]
    scales = list(lex.scales)
    cats = list(lex.cat_fields)
    rows, unmatched, matched_ct, bigr = [], collections.Counter(), collections.Counter(), collections.Counter()
    for r in df.itertuples():
        toks, cov, hits, years = lex.match(r.description, colour.get(r.id))
        row = {"id": r.id}
        pos = [h for h in hits if not h["neg"]]
        ar = [h for h in pos if h["kind"] == "aroma"]
        dset = {(h["f1"], h["f2"]) for h in ar}
        for c, d in descs:
            row[f"a_{c}__{d}"] = int((c, d) in dset)
        cs = {h["f1"] for h in ar}
        csn = {h["f1"] for h in ar if h["scope"] in ("nose", "both")}
        csp = {h["f1"] for h in ar if h["scope"] in ("palate", "both")}
        for c in clusters:
            row[f"c_{c}"] = int(c in cs); row[f"cn_{c}"] = int(c in csn); row[f"cp_{c}"] = int(c in csp)
        for t in tiers:
            row[f"t_{t}"] = int(any(h["ctiers"] == t for h in ar))      # tier flag from UNAMBIGUOUS-tier evidence only
        for s in scales:
            lv = [h["f2"] for h in pos if h["kind"] == "structure" and h["f1"] == s]
            row[f"s_{s}"] = float(np.mean(lv)) if lv else np.nan
            row[f"n_{s}"] = len(lv)
        for f in cats:
            vals = [h["f2"] for h in pos if h["kind"] == "cat" and h["f1"] == f]
            row[f"cat_{f}"] = "|".join(sorted(set(vals))) if vals else None      # multi-valued allowed (e.g. two tannin natures)
        flags = {k: int(any(h["kind"] == "readiness" and h["f1"] == k for h in pos)) for k in ("drink_now", "can_age", "too_old")}
        for k, v in flags.items():
            row[f"r_{k}"] = v
        row["readiness"] = readiness_outcome(**flags)
        row["bottle_ageing"] = ageing_outcome(flags["drink_now"], flags["can_age"])
        row["r_year_from"] = years[0][0] if years else np.nan
        row["r_year_to"] = (years[0][1] if years and years[0][1] else np.nan)
        row["quality_band"] = quality_band(r.points)
        row["n_hits"] = len(pos); row["n_neg"] = len(hits) - len(pos)
        row["n_tokens"] = len([t for t in toks if t not in BOUNDS])
        row["coverage"] = sum(cov) / max(1, row["n_tokens"])
        row["evidence"] = json.dumps([[h["kind"], h["f1"], h["f2"], h["surface"], h["neg"], h["tag"], h["scope"], h["amb"]] for h in hits])
        rows.append(row)
        for h in hits:
            matched_ct[(h["kind"], str(h["f1"]), str(h["f2"]), h["surface"], h["tag"])] += 1
        prev = None
        for t, c in zip(toks, cov):
            if not c and t not in STOP and t not in BOUNDS and t.isalpha() and len(t) > 2:
                unmatched[t] += 1
                if prev:
                    bigr[prev + " " + t] += 1
                prev = t
            else:
                prev = None
    vec = pd.DataFrame(rows)
    tag = "all" if n == "all" else n
    suffix = "_extra" if extra else ""
    vec.to_parquet(OUT / f"wset_vectors_{tag}{suffix}.parquet", index=False)
    pd.DataFrame(unmatched.most_common(5000), columns=["term", "count"]).to_csv(OUT / f"unmatched_unigrams_{tag}{suffix}.csv", index=False)
    pd.DataFrame(bigr.most_common(3000), columns=["term", "count"]).to_csv(OUT / f"unmatched_bigrams_{tag}{suffix}.csv", index=False)
    pd.DataFrame([(*k, v) for k, v in matched_ct.most_common()], columns=["kind", "f1", "f2", "surface", "tag", "count"]).to_csv(OUT / f"matched_forms_{tag}{suffix}.csv", index=False)
    cc = [c for c in vec if c.startswith("c_")]
    print(f"reviews={len(vec)} mean_hits={vec.n_hits.mean():.2f} token_coverage={vec.coverage.mean():.3f} | any cluster {(vec[cc].sum(axis=1) > 0).mean():.3f} | clusters={len(clusters)} descriptors={len(descs)}")
    print("tiers:", {t: round(vec['t_' + t].mean(), 3) for t in tiers})
    print("scales filled:", {s: round(vec['s_' + s].notna().mean(), 3) for s in scales})
    print("categorical filled:", {f: round(vec['cat_' + f].notna().mean(), 3) for f in cats})
    print("readiness:", vec.readiness.value_counts(dropna=False).to_dict(), "| year range", round(vec.r_year_from.notna().mean(), 3))
    print("nose-scoped aroma:", round(vec[[c for c in vec if c.startswith('cn_')]].max(axis=1).mean(), 3), "palate-scoped:", round(vec[[c for c in vec if c.startswith('cp_')]].max(axis=1).mean(), 3))
    print("conflicts:", len(lex.conflicts))


if __name__ == "__main__":
    main()
