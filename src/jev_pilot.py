"""Pilot: map reviews to WSET-style fields with TypeSafe Jev (System One), then compare with the deterministic matcher.
Key is read from env TYPESAFE_API_KEY only (never stored). Usage: jev_pilot.py <N> [--model jev-1.14.0]
One request per review; all questions in one call (fan-out). Output: outputs/jev_pilot.jsonl (answers + usage; no review text)."""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, Lexicon, review_id
import key_status

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_TOK = 0.042 / 1e6
CLUSTER_GLOSS = {  # human descriptions so Jev knows what each cluster means (hand-written, WSET-style families)
    "floral": "floral notes (blossom, rose, violet, lavender, perfume)",
    "green_fruit": "green fruit (apple, pear, gooseberry, green grape, quince)",
    "citrus": "citrus (lemon, lime, grapefruit, orange peel, tangerine)",
    "stone_fruit": "stone fruit (peach, apricot, nectarine)",
    "tropical_fruit": "tropical fruit (banana, lychee, mango, melon, passion fruit, pineapple)",
    "red_fruit": "red fruit (strawberry, raspberry, red cherry, red plum, cranberry)",
    "black_fruit": "black fruit (blackcurrant/cassis, blackberry, blueberry, black cherry, black plum)",
    "herbaceous": "green/herbaceous notes (green pepper, grass, tomato leaf, asparagus, vegetal)",
    "herbal": "herbal notes (eucalyptus, mint, thyme, rosemary, fennel, anise, tea)",
    "other_primary": "other primary notes: minerality/wet stone, smoke, earth, mushroom, spice (pepper, clove, cinnamon), honey, candy, caramel, coffee/chocolate, leather, meat, nuts, petrol",
    "lees_malolactic": "winemaking aromas from lees/malolactic: butter, cream, bread dough, brioche, yeast, biscuit",
    "oak": "oak influence: vanilla, toast, cedar, coconut, smoke from oak, woody notes",
    "dried_cooked_fruit": "dried or cooked fruit (raisin, fig, prune, stewed or baked fruit, jam)",
    "oxidative_aged": "tertiary/aged notes (oxidative, sherry-like, tobacco, forest floor, dried leaves, mature complexity)",
}
SCALES = {  # scale -> (what is being rated, ordered levels low->high)
    "acidity": ("How high is the wine's acidity?", ["Low acidity: flabby, soft, round", "Medium(-) acidity", "Medium acidity: balanced, fresh", "Medium(+) acidity: crisp, zesty, bright", "High acidity: tart, searing, sharp"]),
    "tannin": ("How high are the tannins?", ["Low or no tannin: soft, silky", "Medium(-) tannin: fine, supple", "Medium tannin: structured, firm", "Medium(+) tannin: grippy, chewy", "High tannin: astringent, harsh"]),
    "body": ("How full is the body?", ["Light body: delicate, thin", "Medium(-) body", "Medium body", "Medium(+) body: rich, generous", "Full body: heavy, big, powerful"]),
    "sweetness": ("How sweet is the wine?", ["Dry", "Off-dry", "Medium-dry/medium-sweet", "Sweet", "Luscious, very sweet"]),
    "finish": ("How long is the finish?", ["Short", "Medium(-)", "Medium", "Medium(+): lingering", "Long"]),
    "flavour_intensity": ("How intense are the aromas and flavours?", ["Light, neutral, muted", "Medium(-), subtle", "Medium intensity", "Medium(+), expressive", "Pronounced, powerful"]),
    "alcohol": ("How high is the alcohol?", ["Low alcohol", "Medium alcohol", "High alcohol: hot, warming"]),
    "development": ("What stage of development does the wine show?", ["Youthful, primary fruit, needs time", "Developing, early evolution", "Fully developed, mature, at peak", "Tired, past its best, oxidised"]),
}


def build_questions():
    q = {}
    for c, g in CLUSTER_GLOSS.items():
        q[f"aroma__{c}"] = dict(type="noul", instructions=f"Does the tasting note mention {g}?",
                                criteria=dict(true=f"The note names at least one of: {g}", false="No such note is mentioned"))
    for s, (what, levels) in SCALES.items():
        q[f"stated__{s}"] = dict(type="noul", instructions=f"Does the note explicitly describe the wine's {s.replace('_', ' ')} (not merely imply it)?")
        q[f"level__{s}"] = dict(type="score", instructions=what, criteria=levels)
    q["ready__drink_now"] = dict(type="noul", instructions="Does the note say the wine is ready to drink now?")
    q["ready__can_age"] = dict(type="noul", instructions="Does the note say the wine will improve with ageing or needs time?")
    return q


def call(session, key, model, text, retries=4):
    body = dict(model=model, state=text, questions=build_questions())
    for a in range(retries):
        r = session.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (a + 1)); continue
        if r.status_code in (401, 402, 403):
            key_status.run()            # record the failure time for the dashboard reminder
        return {"error": r.status_code, "body": r.text[:300]}
    return {"error": "retries_exhausted"}


def main():
    key_status.load_env()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("set TYPESAFE_API_KEY in the environment")
    n = int(sys.argv[1]); model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "jev-latest"
    vec = pd.read_parquet(ROOT / "outputs/wset_vectors_20000.parquet", columns=["id"])
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    raw["id"] = raw.description.map(review_id)
    raw = raw.drop_duplicates("id").set_index("id")
    ids = vec.sample(n, random_state=5).id.tolist()                       # deterministic sample
    out_f = ROOT / "outputs/jev_pilot.jsonl"
    done = {json.loads(l)["id"] for l in open(out_f)} if out_f.exists() else set()
    todo = [i for i in ids if i not in done]
    s = requests.Session()
    tot_in = tot_out = errs = 0

    def work(i):
        return i, call(s, key, model, raw.loc[i, "description"])
    with ThreadPoolExecutor(6) as ex, open(out_f, "a") as f:
        for k, (i, r) in enumerate(ex.map(work, todo), 1):
            if "error" in r:
                errs += 1; print("ERR", i, r); continue
            u = r.get("usage", {}); tot_in += u.get("input_tokens", 0); tot_out += u.get("output_tokens", 0)
            f.write(json.dumps({"id": i, "model": r.get("model"), "answers": r["answers"], "usage": u}) + "\n"); f.flush()
            if k % 50 == 0:
                print(k, "of", len(todo), f"in_tok={tot_in}", flush=True)
    print(f"done {len(todo) - errs} ok, {errs} errors | input tokens {tot_in:,} | output {tot_out:,} | cost ${tot_in * PRICE_PER_TOK:.4f} "
          f"| per review {tot_in / max(1, len(todo) - errs):.0f} tok | projected full corpus ${tot_in / max(1, len(todo) - errs) * 169428 * PRICE_PER_TOK:.2f}")


if __name__ == "__main__":
    main()
