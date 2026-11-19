"""Re-ask a subset of Jev questions (by key substring) on already-mapped reviews and patch them into the jsonl files.
Usage: jev_patch.py key1,key2,... [drop_prefix1,drop_prefix2]"""
import sys, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).parent))
import jev_pilot as J, key_status
from wsetkit import ROOT, review_id
key_status.load_env(); key = os.environ["TYPESAFE_API_KEY"]
keys = sys.argv[1].split(",")
drop = sys.argv[2].split(",") if len(sys.argv) > 2 else []   # old answer keys to delete (prefix match)
Q = {k: v for k, v in J.build_questions().items() if k in keys}
raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id")
rows = {f: [json.loads(l) for l in open(ROOT / f)] for f in ("outputs/jev_pilot.jsonl", "outputs/jev_1000.jsonl")}
s = requests.Session(); tok = errs = 0
def w(j):
    return j, s.post(J.URL, json=dict(model="jev-latest", state=raw.loc[j["id"], "description"], questions=Q), headers={"Authorization": f"Bearer {key}"}, timeout=90).json()
allj = [j for r in rows.values() for j in r]
with ThreadPoolExecutor(6) as ex:
    for j, r in ex.map(w, allj):
        if "answers" in r: j["answers"].update(r["answers"]); [j["answers"].pop(k) for k in [k for k in j["answers"] if any(k.startswith(d) for d in drop)]]; tok += r["usage"]["input_tokens"]
        else: errs += 1; print("ERR", j["id"], r)
for f, r in rows.items():
    open(ROOT / f, "w").write("".join(json.dumps(j) + "\n" for j in r))
print(f"patched {len(allj) - errs}/{len(allj)} | errors {errs} | cost ${tok * J.PRICE_PER_TOK:.4f}")
