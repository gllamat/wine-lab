"""Writes the small JSON files the dashboard's evaluation panels read (dashboard/data/recsys_sense.json, recsys_hubness.json) from the benchmark outputs in this folder."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent; D = H.parent.parent / "dashboard/data"
res = json.load(open(H / "results.json")); lists = json.load(open(H / "sanity_lists.json")); pe = json.load(open(H / "phrases_eval.json")); hub = json.load(open(H / "hubness.json")); ph = json.load(open(D / "phrases.json"))
ren = {"jev_hybrid": "jevh", "jev_hybrid+rules": "jevh+rules"}; rn = lambda k: ren.get(k, k)
keep_t = ["random", "raw", "zw", "latent", "emb", "hybrid", "rrf", "mmr", "jev_desc", "jev", "jev_hybrid", "hybrid+rules", "rrf+rules", "jev+rules", "jev_hybrid+rules"]
keep = ["lists_with_hard_clash", "style_clash_any", "taste_cosine", "grape_group", "wines_swapped_per_list", "labelled_pairs", "red_white_clash", "sparkling_still_clash", "sweetness_clash", "body_clash"]
T = {rn(k): {f: round(v, 4) for f, v in res["techniques"][k].items() if f in keep} for k in keep_t if k in res["techniques"]}
# 12 queries: 4 red, 4 white, 2 sparkling, 1 rose, 1 sweet (the benchmark wrote them in that order: 10 red, 10 white, 3 rose, 4 sparkling, 3 sweet)
sel = list(range(0, 4)) + list(range(10, 14)) + list(range(23, 25)) + [20] + [27]
LT = ["raw", "zw", "emb", "hybrid", "jev", "jev_hybrid", "hybrid+rules", "jev+rules", "jev_hybrid+rules"]; RM = ["hybrid", "jev", "jev_hybrid"]
slim = lambda w: dict(title=w["title"], variety=w["variety"], country=w["country"], colour=w["colour"], points=w["points"], sweetness=w["sweetness"], flags={k: w["flags"][k] for k in ("colour_clash", "sparkling_still", "sweetness_gap", "same_grape")})
L = []
for i in sel:
    e = lists[i]; q = e["query"]
    L.append(dict(query=dict(title=q["title"], variety=q["variety"], country=q["country"], colour=q["colour"], points=q["points"], sweetness=q["sweetness"], body=q["body"]),
                  lists={rn(t): [slim(w) for w in e["lists"][t]] for t in LT if t in e["lists"]}, removed={rn(t): [dict(title=w["title"], why=w["why"]) for w in e["removed"].get(t, [])] for t in RM}))
tk = {k: {f: round(v, 4) for f, v in d.items() if f in ("style_gap", "taste", "lists_with_hard_clash")} for k, d in pe["techniques"].items() if not k.startswith("random")}
json.dump(dict(queries=res["queries"], labelled_pairs=res["techniques"]["jev_hybrid"]["labelled_pairs"], techniques=T, typed=tk, typed_queries=pe["queries"], phrases_support=pe["min_support"], phrases_precision=pe["min_precision"], phrases_labelled=ph["n_labelled"], lists=L),
          open(D / "recsys_sense.json", "w"), separators=(",", ":"))
n = hub["n_wines"]
json.dump(dict(n_wines=n, slots=hub["slots"], lam=hub["lam"], random_never=round((1 - 10 / n) ** n, 5), plain_emb=hub["plain_emb"], plain_jev_hybrid=hub["plain_jev_hybrid"], rules=hub["rules"], rules_fix=hub["rules_fix"],
               sweep={k: v for k, v in hub["sweep"].items() if k.startswith("jev_hybrid")}, hist_rules=hub["hist_rules"], hist_rules_fix=hub["hist_rules_fix"], top_hubs=hub["top_hubs"]), open(D / "recsys_hubness.json", "w"), separators=(",", ":"))
for f in ("recsys_sense.json", "recsys_hubness.json"): print(f, round((D / f).stat().st_size / 1024), "KB")
