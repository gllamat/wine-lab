"""Pilot: map reviews to the WSET L4 SAT card (Issue 1.2, 2023) with TypeSafe Jev (System One); compare with the deterministic matcher.
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
# ---- Questions follow the WSET Level 4 SAT card (Issue 1.2, Aug 2023): every card line, every Wine-Lexicon descriptor.
CLUSTER_HEAD = {"floral": "Floral", "green_fruit": "Green fruit", "citrus_fruit": "Citrus fruit", "stone_fruit": "Stone fruit", "tropical_fruit": "Tropical fruit",
                "red_fruit": "Red fruit", "black_fruit": "Black fruit", "herbaceous": "Herbaceous", "herbal": "Herbal", "spice": "Spice", "fruit_ripeness": "Fruit ripeness",
                "other": "Other", "yeast": "Yeast (lees, autolysis, flor)", "malolactic": "Malolactic conversion", "oak": "Oak",
                "tertiary_red": "Tertiary (maturation) aromas of RED wine", "tertiary_white": "Tertiary (maturation) aromas of WHITE wine",
                "deliberately_oxidised": "Tertiary aromas of DELIBERATELY OXIDISED wines"}
DESC_NICE = {"biscuit_graham_cracker": "biscuit / graham cracker", "green_bell_pepper": "green bell pepper (capsicum)", "black_white_pepper": "black or white pepper",
             "dried_herbs": "dried herbs (e.g. thyme, oregano)", "orange_marmalade": "orange marmalade", "simple": "simple (a plain, uncomplicated character)",
             "wet_stones": "wet stones", "unripe_fruit": "unripe fruit", "ripe_fruit": "ripe fruit", "dried_fruit": "dried fruit", "cooked_fruit": "cooked fruit",
             "cloves": "cloves", "charred_wood": "charred wood", "bread_dough": "bread dough", "wet_leaves": "wet leaves", "forest_floor": "forest floor",
             "passion_fruit": "passion fruit", "red_cherry": "red cherry", "red_plum": "red plum", "black_cherry": "black cherry", "black_plum": "black plum"}
EXTRA_NICE = {("tertiary_red", "dried_fruit"): "dried fruit (e.g. prune, raisin, fig)", ("tertiary_red", "cooked_fruit"): "cooked fruit (e.g. cooked plum, cooked cherry)",
              ("tertiary_white", "dried_fruit"): "dried fruit (e.g. dried apricot, raisin)"}
SCALES = {  # card lines with ordered levels (SAT level names)
    "sweetness": ("How sweet is the wine?", ["Dry", "Off-dry", "Medium-dry", "Medium-sweet", "Sweet", "Luscious"]),
    "acidity": ("How high is the acidity?", ["Low", "Medium(-)", "Medium", "Medium(+)", "High"]),
    "tannin": ("How high is the level of tannin?", ["Low", "Medium(-)", "Medium", "Medium(+)", "High"]),
    "alcohol": ("How high is the alcohol?", ["Low", "Medium", "High"]),
    "body": ("How full is the body?", ["Light", "Medium(-)", "Medium", "Medium(+)", "Full"]),
    "flavour_intensity": ("How intense are the flavours on the palate?", ["Light", "Medium(-)", "Medium", "Medium(+)", "Pronounced"]),
    "finish": ("How long is the finish?", ["Short", "Medium(-)", "Medium", "Medium(+)", "Long"]),
    "nose_intensity": ("How intense are the aromas on the nose?", ["Light", "Medium(-)", "Medium", "Medium(+)", "Pronounced"]),
}
COLOURS = ["lemon-green", "lemon", "gold", "amber", "brown", "pink", "pink-orange", "orange", "purple", "ruby", "garnet", "tawny"]
TANNIN_NATURE = ["ripe", "soft", "smooth", "unripe", "green", "coarse", "stalky", "chalky", "fine-grained"]
TEXTURE = ["oily", "creamy", "austere", "luscious"]
APP_OTHER = {"legs_tears": "legs or tears", "deposit": "a deposit", "petillance": "pétillance", "bubbles": "bubbles"}


def descriptors():
    a = yaml.safe_load(open(ROOT / "lexicon/aromas.yaml")); out = []
    for tier in ("primary", "secondary", "tertiary"):
        for cl, ds in a[tier].items():
            for d in ds:
                if not d.endswith("_generic"):
                    out.append((cl, d, EXTRA_NICE.get((cl, d), DESC_NICE.get(d, d.replace("_", " ")))))
    return out


def build_questions():
    q = {}
    q["app__clarity"] = dict(type="choice", instructions="What does the note say about the clarity of the wine?",
                             criteria=dict(clear="Described as clear or brilliant", hazy="Described as hazy, cloudy or turbid", not_stated="Clarity is not mentioned"))
    q["app__intensity"] = dict(type="choice", instructions="How deep is the colour, according to the note?",
                               criteria=dict(pale="Pale colour", medium="Medium colour depth", deep="Deep, dark or inky colour", not_stated="Colour depth is not mentioned"))
    q["app__colour"] = dict(type="choice", instructions="What colour does the note describe? (white wines: lemon-green, lemon, gold, amber, brown; rosé: pink, pink-orange, orange; red: purple, ruby, garnet, tawny, brown)",
                            criteria={**{c: f"Colour described as {c}" for c in COLOURS}, "not_stated": "No colour is described"})
    for k, what in APP_OTHER.items():
        q[f"appother__{k}"] = dict(type="noul", instructions=f"Does the note mention {what} when describing the appearance?")
    q["nose__condition"] = dict(type="noul", instructions="Does the note indicate a fault or unclean nose (cork taint, volatile acidity, brettanomyces, reduction, etc.)?")
    for cl, d, nice in descriptors():
        head = CLUSTER_HEAD[cl]
        q[f"nose__{cl}__{d}"] = dict(type="noul", instructions=f"Does the note describe {nice} among the aromas on the NOSE? (WSET lexicon cluster: {head})")
        q[f"palate__{cl}__{d}"] = dict(type="noul", instructions=f"Does the note describe {nice} among the flavours on the PALATE? (WSET lexicon cluster: {head})")
    for s, (what, levels) in SCALES.items():
        q[f"stated__{s}"] = dict(type="noul", instructions=f"Does the note explicitly describe the wine's {s.replace('_', ' ')} (not merely imply it)?")
        q[f"level__{s}"] = dict(type="score", instructions=what, criteria=levels)
    for n in TANNIN_NATURE:
        q[f"tnature__{n}"] = dict(type="noul", instructions=f"Does the note describe the NATURE of the tannins as {n}?")
    for n in TEXTURE:
        q[f"texture__{n}"] = dict(type="noul", instructions=f"Does the note describe the texture of the wine as {n}?")
    q["palate__petillance"] = dict(type="noul", instructions="Does the note mention pétillance (a slight prickle or spritz) on the palate of a still wine?")
    q["quality"] = dict(type="score", instructions="Overall, how good is the wine as implied by the note (balance, length, intensity, complexity)?",
                        criteria=["Faulty", "Poor", "Acceptable", "Good", "Very good", "Outstanding"])
    q["ageing"] = dict(type="choice", instructions="What does the note say about the wine's suitability for bottle ageing?", criteria=dict(
        suitable="Suitable for bottle ageing (will improve, has potential, needs time)",
        not_suitable="Not suitable for bottle ageing (to be drunk now or soon)",
        not_stated="Suitability for ageing is not mentioned"))
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
