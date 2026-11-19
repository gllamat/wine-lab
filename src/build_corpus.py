"""Merge the public review datasets into one de-duplicated corpus: ../datasets/wine_reviews_deduped.parquet.
Sources (you must download them yourself, see docs/DATA.md; they are NOT in this repository):
  kaggle_wine_reviews/winemag-data-130k-v2.csv   Kaggle "Wine Reviews" (Wine Enthusiast, scraped 2017)
  kaggle_wine_reviews/winemag-data_first150k.csv Kaggle "Wine Reviews" (first scrape)
  hf_gronlp_winemag/{train,validation,test}.csv  Hugging Face mirror (optional; fewer columns, adds no unique reviews)
De-duplication key: wsetkit.review_id(description) (hash of the normalised text). Per column the first non-null value wins in the order 130k-v2, first150k, hf.
Usage: build_corpus.py [datasets_dir] [--out file.parquet]"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id

COLS = ["country", "description", "designation", "points", "price", "province", "region_1", "region_2", "taster_name", "taster_twitter_handle", "title", "variety", "winery"]


def main():
    argv = sys.argv[1:]
    if "--out" in argv: i = argv.index("--out"); argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    D = Path(args[0]) if args else ROOT.parent / "datasets"
    srcs = [("kaggle_130k_v2", pd.read_csv(D / "kaggle_wine_reviews/winemag-data-130k-v2.csv")),
            ("kaggle_first150k", pd.read_csv(D / "kaggle_wine_reviews/winemag-data_first150k.csv")),
            ]
    hf = [D / "hf_gronlp_winemag" / f"{s}.csv" for s in ("train", "validation", "test")]
    if all(f.exists() for f in hf):                                          # optional: adds no review that the Kaggle files lack, only the `sources` tag
        srcs.append(("hf_gronlp", pd.concat([pd.read_csv(f) for f in hf])))
    frames = []
    for name, df in srcs:
        df = df.copy(); df["id"] = df.description.map(review_id); df = df.drop_duplicates("id").set_index("id")
        for c in COLS:
            if c not in df: df[c] = pd.NA
        df = df[COLS]; df["_src"] = name; frames.append(df)
    out = frames[0][COLS].copy()
    for df in frames[1:]:
        out = out.combine_first(df[COLS])                                    # first non-null per column, earlier sources win
    src = pd.concat([f[["_src"]] for f in frames]).groupby(level=0)["_src"].agg(lambda s: ",".join(sorted(s)))
    out["sources"] = src.reindex(out.index)
    out = out.reset_index(drop=True)[COLS + ["sources"]]
    dst = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else D / "wine_reviews_deduped.parquet"; out.to_parquet(dst, index=False)
    print(f"{len(out):,} unique reviews -> {dst}")


if __name__ == "__main__":
    main()
