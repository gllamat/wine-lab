"""Build the data pack used by the distillation notebook.

Joins the critics' tasting notes with Jev's WSET answers (the `outputs/jev_*.jsonl` files)
and writes TWO small files into distill/pack/ that you upload to Google Drive:

    wset_distill_pack.parquet   one row per review: text + all Jev answers
    wset_distill_schema.json    names and order of the questions, so the notebook can rebuild everything

Jev answers come in two shapes (see src/jev_pilot.py):
  * "noul"   : one number, the probability that the note mentions the thing (216 questions)
  * "choice" / "score": a probability for EACH level (6 choices + 9 scales = 15 questions)

Usage:  python distill/make_data_pack.py                       (uses jev_1000 + jev_pilot)
        python distill/make_data_pack.py outputs/jev_full.jsonl (any jsonl files you list)

PRIVACY / LICENCE: the pack contains the review text (Wine Enthusiast, CC BY-NC-SA 4.0). Keep it private and non-commercial.
"""
import hashlib, json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def review_id(description: str) -> str:
    """Same id the labelling script used (copied from src/wsetkit.py so this file stands alone)."""
    k = re.sub(r"\s+", " ", str(description).lower().strip())
    return hashlib.md5(k.encode()).hexdigest()[:12]

OUT = Path(__file__).resolve().parent / "pack"


def load_answers(files):
    """Read every jsonl file -> {id: answers}. Later files win if an id appears twice."""
    by_id = {}
    for f in files:
        for line in open(f):
            rec = json.loads(line)
            by_id[rec["id"]] = rec["answers"]
    return by_id


def main():
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / "outputs/jev_1000.jsonl", ROOT / "outputs/jev_pilot.jsonl"]
    by_id = load_answers(files)
    first = next(iter(by_id.values()))

    # --- schema: which questions are yes/no, and which have several levels ---
    binary = [q for q, a in first.items() if a["type"] == "noul"]
    groups = {q: list(a["probabilities"]) for q, a in first.items() if a["type"] in ("choice", "score")}
    # for scales the level names live in 'legend' (e.g. 0 -> "Low"); for choices the keys are already names
    for q, a in first.items():
        if a["type"] == "score":
            groups[q] = [a["legend"][k] for k in a["probabilities"]]
    kind = {q: first[q]["type"] for q in groups}

    # --- text: the critics' notes, with the same ids as the labels ---
    cols = ["description", "title", "winery", "variety", "country", "points"]
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=cols)
    raw["id"] = raw.description.map(review_id)
    raw = raw.drop_duplicates("id").set_index("id")
    ids = [i for i in by_id if i in raw.index]
    print(f"{len(by_id)} labelled reviews, {len(ids)} found in the corpus")

    # --- one row per review ---
    new = {"id": ids}
    for q in binary:                                   # yes/no questions -> one float column each
        new["b::" + q] = [by_id[i][q]["noul"] for i in ids]
    for q in groups:                                   # multi-level questions -> a list of probabilities
        keys = list(first[q]["probabilities"])
        new["g::" + q] = [[by_id[i][q]["probabilities"][k] for k in keys] for i in ids]
    # build all columns in one go (adding hundreds of columns one by one is slow and warns)
    df = pd.concat([raw.loc[ids, cols].reset_index(drop=True), pd.DataFrame(new)], axis=1)
    df = df[["id"] + cols + [c for c in df.columns if c not in ["id"] + cols]]

    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "wset_distill_pack.parquet", index=False)
    schema = {"binary": binary, "groups": groups, "kind": kind,
              "teacher": "jev-1.14.0 (TypeSafe System One)", "n_rows": len(df)}
    (OUT / "wset_distill_schema.json").write_text(json.dumps(schema, indent=1))
    print(f"wrote {OUT}/  ({len(binary)} yes/no questions, {len(groups)} multi-level questions, {len(df)} rows)")
    write_raw_examples(files, raw, ids)


def write_raw_examples(files, raw, ids, n=3):
    """A few Jev records EXACTLY as saved from the API (all 231 answers + token usage), plus the shape of the questions we sent,
    so the notebook can show 'what arrives from Jev' before it is turned into table columns. Contains review text: keep private."""
    import random
    recs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); recs[r["id"]] = r
    rng = random.Random(7); pool = [i for i in ids if 160 <= len(raw.loc[i, "description"]) <= 420]; rng.shuffle(pool)
    picked, seen = [], set()
    for i in pool:                                           # three different wines (different grape), short notes
        v = str(raw.loc[i, "variety"])
        if v not in seen: seen.add(v); picked.append(i)
        if len(picked) == n: break
    try:
        sys.path.insert(0, str(ROOT / "src")); import jev_pilot
        qs = jev_pilot.build_questions(); sample_q = {k: qs[k] for k in ("nose__oak__vanilla", "level__tannin", "ripeness") if k in qs}
    except Exception as e:
        print("could not import the question definitions:", e); sample_q = {}
    out = {"examples": [{"id": i, "description": raw.loc[i, "description"], "title": raw.loc[i, "title"], "winery": raw.loc[i, "winery"],
                         "variety": raw.loc[i, "variety"], "country": raw.loc[i, "country"], "points": int(raw.loc[i, "points"]), "record": recs[i]} for i in picked],
           "request_example": {"model": "jev-1.14.0", "state": "<the critic's tasting note>", "questions": sample_q,
                               "note": "the real request carries all 231 questions (about 9,000 input tokens per note)"}}
    (OUT / "jev_raw_examples.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote jev_raw_examples.json:", [str(raw.loc[i, "title"])[:40] for i in picked])


if __name__ == "__main__":
    main()
