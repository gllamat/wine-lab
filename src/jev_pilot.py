"""Pilot: map reviews to the WSET L3 SAT (3W v2.0, 2016) with TypeSafe Jev (System One); compare with the deterministic matcher.
Key: env TYPESAFE_API_KEY or ./.env (never stored/printed). Usage: jev_pilot.py <N> [--model jev-latest]
One request per review, all SAT fields as typed questions (fan-out). Output: outputs/jev_pilot.jsonl (answers + usage; no review text)."""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests, yaml
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT, review_id
import key_status

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_TOK = 0.042 / 1e6
TIER_NOTE = {"primary": "primary (grape/fermentation)", "secondary": "secondary (winemaking: yeast, MLF, oak)", "tertiary": "tertiary (maturation)"}
# ---- SAT L3 scales (level names as in the SAT grid; descriptors are project glosses of the SAT level names)
SCALES = {
    "sweetness": ("How sweet is the wine?", ["Dry", "Off-dry", "Medium-dry", "Medium-sweet", "Sweet", "Luscious"]),
    "acidity": ("How high is the acidity?", ["Low", "Medium(-)", "Medium", "Medium(+)", "High"]),
    "tannin": ("How high are the tannins?", ["Low", "Medium(-)", "Medium", "Medium(+)", "High"]),
    "alcohol": ("How high is the alcohol?", ["Low", "Medium", "High"]),
    "body": ("How full is the body?", ["Light", "Medium(-)", "Medium", "Medium(+)", "Full"]),
    "mousse": ("For a sparkling wine: what is the mousse like?", ["Delicate", "Creamy", "Aggressive"]),
    "flavour_intensity": ("How intense are the flavours on the palate?", ["Light", "Medium(-)", "Medium", "Medium(+)", "Pronounced"]),
    "finish": ("How long is the finish?", ["Short", "Medium(-)", "Medium", "Medium(+)", "Long"]),
    "nose_intensity": ("How intense are the aromas on the nose?", ["Light", "Medium(-)", "Medium", "Medium(+)", "Pronounced"]),
    "development": ("What stage of development does the wine show?", ["Youthful", "Developing", "Fully developed", "Tired / past its best"]),
    "appearance_intensity": ("How deep is the colour?", ["Pale", "Medium", "Deep"]),
}
COLOURS = ["lemon-green", "lemon", "gold", "amber", "brown", "pink", "salmon", "orange", "purple", "ruby", "garnet", "tawny"]


def cluster_gloss():
    a = yaml.safe_load(open(ROOT / "lexicon/aromas.yaml"))
    out = {}
    for tier in ("primary", "secondary", "tertiary"):
        for cl, descs in a[tier].items():
            names = [d.replace("_", " ") for d in descs if not d.endswith("_generic")]
            out[cl] = (tier, cl.replace("_", " "), names)
    return out


def build_questions():
    q = {}
    q["app__clarity"] = dict(type="choice", instructions="What does the note say about the clarity of the wine?",
                             criteria=dict(clear="Described as clear or brilliant", hazy="Described as hazy, cloudy or turbid", not_stated="Clarity is not mentioned"))
    q["app__colour"] = dict(type="choice", instructions="What colour does the note describe?",
                            criteria={**{c: f"Colour described as {c}" for c in COLOURS}, "not_stated": "No colour is described"})
    q["app__intensity"] = dict(type="choice", instructions="How deep is the colour, according to the note?",
                               criteria=dict(pale="Pale colour", medium="Medium colour depth", deep="Deep, dark or inky colour", not_stated="Colour depth is not mentioned"))
    q["nose__condition"] = dict(type="noul", instructions="Does the note indicate a fault or unclean nose (cork taint, volatile acidity, brettanomyces, reduction, etc.)?")
    for cl, (tier, name, descs) in cluster_gloss().items():
        g = f"{name} ({', '.join(descs)}), a {TIER_NOTE[tier]} aroma cluster"
        q[f"nose__{cl}"] = dict(type="noul", instructions=f"Does the note describe aromas on the NOSE from the cluster: {g}?")
        q[f"palate__{cl}"] = dict(type="noul", instructions=f"Does the note describe flavours on the PALATE from the cluster: {g}?")
    for s, (what, levels) in SCALES.items():
        q[f"stated__{s}"] = dict(type="noul", instructions=f"Does the note explicitly describe the wine's {s.replace('_', ' ')} (not merely imply it)?")
        q[f"level__{s}"] = dict(type="score", instructions=what, criteria=levels)
    q["quality"] = dict(type="score", instructions="Overall, how good is the wine as implied by the note (balance, length, intensity, complexity)?",
                        criteria=["Faulty", "Poor", "Acceptable", "Good", "Very good", "Outstanding"])
    q["readiness"] = dict(type="choice", instructions="What does the note say about readiness for drinking?", criteria=dict(
        not_ready_potential="Too young or needs time, but has potential",
        drink_now_potential="Can drink now but has potential for ageing",
        drink_now="Drink now; ageing potential not stated or not suitable for ageing",
        too_old="Too old or past its best",
        not_stated="Readiness is not mentioned"))
    return q


def call(session, key, model, text, retries=4):
    body = dict(model=model, state=text, questions=build_questions())
    for a in range(retries):
        r = session.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
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
        sys.exit("set TYPESAFE_API_KEY in the environment or .env")
    n = int(sys.argv[1]); model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "jev-latest"
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    raw["id"] = raw.description.map(review_id)
    raw = raw.drop_duplicates("id").set_index("id")
    if "--ids-from" in sys.argv:                      # explicit id list (csv with an 'id' column)
        ids = pd.read_csv(sys.argv[sys.argv.index("--ids-from") + 1])["id"].tolist()[:n]
    else:
        vec = pd.read_parquet(ROOT / "outputs/wset_vectors_20000.parquet", columns=["id"])
        ids = vec.sample(n, random_state=5).id.tolist()
    out_f = ROOT / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "outputs/jev_pilot.jsonl")
    done = {json.loads(l)["id"] for l in open(out_f)} if out_f.exists() else set()
    todo = [i for i in ids if i not in done]
    s = requests.Session()
    tot_in = tot_out = errs = 0
    qn = len(build_questions())

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
    ok = max(1, len(todo) - errs)
    print(f"questions per review: {qn} | done {len(todo) - errs} ok, {errs} errors | input tokens {tot_in:,} | cost ${tot_in * PRICE_PER_TOK:.4f} "
          f"| per review {tot_in / ok:.0f} tok | projected full corpus ${tot_in / ok * 169428 * PRICE_PER_TOK:.2f}")


if __name__ == "__main__":
    main()
