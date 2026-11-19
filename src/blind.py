"""Blind-tasting rounds for the guess model. One round = 500 notes never tested before.
  sample (seed 1000+round, from the corpus minus the ledger) -> seal the key -> train on EVERYTHING else (earlier revealed rounds included)
  -> predict -> freeze predictions (sha256) -> reveal -> score -> record -> append the 500 ids to the ledger.
Usage: blind.py run [--n 500] [--dry] | status
The model never sees the sealed ids in training, calibration or model selection (asserted below)."""
import sys, json, time, hashlib, platform, subprocess, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT
import guess_model as G

EXP = ROOT / "experiments/blind_tasting"
DASH = ROOT / "dashboard/data"
RECIPE = "lite-1"


def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def ledger():
    f = EXP / "ledger.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["id", "round", "tested_at"])


def git(*a):
    try: return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception: return ""


def explain(M, x, cls_i, k=5):
    """Features that pushed the guess towards class cls_i (linear model: weight x standardised value, relative to the probability-weighted mean class)."""
    z = (x - M["mu"]) / M["sd"]; W = M["var_W"]; n = len(z)
    logits = W @ z + M["var_b"]; p = G.softmax_T(logits[None], M["T_var"])[0]
    contrib = z * (W[cls_i] - p @ W); names = M["_names"]
    o = np.argsort(-contrib)[:k]
    return [[names[j], round(float(contrib[j]), 2)] for j in o if contrib[j] > 0.02]


def summarize_rounds():
    rows = []
    for d in sorted(EXP.glob("round_*")):
        f = d / "scores.json"
        if f.exists():
            s = json.load(open(f)); m = json.load(open(d / "manifest.json")); lite = s.get("lite", s)
            rows.append(dict(round=m["round"], n=m["n_sealed"], date=m["date"], recipe=m["recipe"], pool=m["n_pool"], scores=lite, scores_full=s.get("full")))
    DASH.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(DASH / "blind_rounds.json", "w"), separators=(",", ":"))     # metrics only: safe to deploy
    return rows


def run(n=500, tiers=("lite", "full"), dry=False):
    """dry=True: 25k-note subset, lite only, records into outputs/blind_dry (never touches the real ledger or dashboard files)."""
    global EXP, DASH
    if dry: EXP = ROOT / "outputs/blind_dry"; DASH = ROOT / "outputs/blind_dry/dash"; tiers = ("lite",)
    t0 = time.time(); EXP.mkdir(parents=True, exist_ok=True); DASH.mkdir(parents=True, exist_ok=True)
    ids, X, spec, nm = G.corpus("lite")
    if dry: ids, X, nm = ids[:25000], X[:25000], nm.iloc[:25000]
    t = G.targets(nm); led = ledger()
    rnd = int(led["round"].max()) + 1 if len(led) else 1; seed = 1000 + rnd
    eligible = np.where(~np.isin(ids, led.id.values))[0]
    sealed = np.sort(np.random.RandomState(seed).choice(eligible, n, replace=False))
    pool = np.setdiff1d(np.arange(len(ids)), sealed)
    assert not set(ids[sealed]) & set(ids[pool]), "sealed ids leaked into the training pool"
    assert not set(ids[sealed]) & set(led.id.values), "a sealed id was already tested"
    rd = EXP / f"round_{rnd:03d}"; rd.mkdir(exist_ok=False)
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    from wsetkit import review_id
    raw["id"] = raw.description.map(review_id); raw = raw.drop_duplicates("id").set_index("id")
    texts = raw.loc[ids[sealed], "description"].values
    pd.DataFrame(dict(id=ids[sealed], text_sha=[hashlib.sha256(x.encode()).hexdigest()[:12] for x in texts])).to_parquet(rd / "items.parquet")
    key = t.iloc[sealed].copy(); key.insert(0, "id", ids[sealed]); key["country_name"] = nm.country.values[sealed]; key.to_parquet(rd / "key.parquet")      # sealed until scoring
    print(f"round {rnd}: sealed {len(sealed)} | training pool {len(pool)} (ledger rounds so far: {led['round'].nunique()})")
    xy, _ = G.place_coords(t); S_all = {}; preds = {}; MODELS = {}
    for tier in tiers:
        Xt, spect = (X, spec) if tier == "lite" else G.corpus(tier)[1:3]
        M = G.fit(Xt, t, pool); M["_names"] = spect["names"]; out = G.predict(M, Xt[sealed]); MODELS[tier] = (M, out, Xt)
        vcls, pcls, W = M["var_classes"], M["place_classes"], M["wineries"]; rows = []
        for i, ix in enumerate(sealed):
            la, lo, r50, r80 = G.place_point(M, out["Pp"], i)
            rows.append(dict(id=ids[ix], variety=json.dumps(G.top(out["Pv"][i:i + 1], vcls)[0]), group=json.dumps(G.top(out["Pg"][i:i + 1], np.array(M["groups"]))[0]),
                             place=json.dumps(G.top(out["Pp"][i:i + 1], pcls)[0]), producer=json.dumps(G.top(out["Pw"][i:i + 1], W)[0]),
                             lat=la, lon=lo, r50_km=r50, r80_km=r80, vintage_lo=out["vintage"][i, 0], vintage=out["vintage"][i, 1], vintage_hi=out["vintage"][i, 2],
                             price_lo=float(np.exp(out["price"][i, 0])), price=float(np.exp(out["price"][i, 1])), price_hi=float(np.exp(out["price"][i, 2])),
                             points_lo=out["points"][i, 0], points=out["points"][i, 1], points_hi=out["points"][i, 2]))
        pred = pd.DataFrame(rows); pred.to_parquet(rd / f"predictions_{tier}.parquet"); preds[tier] = pred
    frozen = {k: sha(rd / f"predictions_{k}.parquet") for k in tiers}
    print("predictions frozen", frozen, "-> opening the key")
    for tier in tiers:
        M, out, _ = MODELS[tier]; S = G.score(M, out, t, sealed, xy_truth=xy[["lat", "lon"]]); S["n"] = int(len(sealed)); S_all[tier] = S
    S = S_all["lite"]; M = MODELS["lite"][0]; out = MODELS["lite"][1]; pred = preds["lite"]
    json.dump(S_all, open(rd / "scores.json", "w"), indent=1)
    man = dict(round=rnd, recipe=RECIPE, seed=seed, n_sealed=int(len(sealed)), n_pool=int(len(pool)), date=datetime.datetime.now().isoformat(timespec="seconds"),
               git_commit=git("rev-parse", "--short", "HEAD"), git_dirty=bool(git("status", "--porcelain")), python=platform.python_version(),
               packages={k: __import__(k).__version__ for k in ("numpy", "pandas", "sklearn", "scipy")},
               tiers=list(tiers), hashes=dict(vectors=sha(ROOT / "outputs/wset_vectors_all_nocolour.parquet"), spec=sha(ROOT / "outputs/guess_spec.json"), corpus=sha(ROOT / "outputs/reviews_norm.parquet"),
                           requirements=sha(ROOT / "requirements/core.txt"), predictions=frozen, embeddings=sha(ROOT / "outputs/emb_minilm_all.npy") if "full" in tiers else None),
               hyper=dict(MIN_VAR=G.MIN_VAR, MIN_PLACE=G.MIN_PLACE, MIN_WINERY=G.MIN_WINERY, NPC=G.NPC, C_GRAPE=G.C_GRAPE, C_PLACE=G.C_PLACE, tau=M["tau"], T_var=M["T_var"], T_place=M["T_place"]),
               seconds=round(time.time() - t0))
    json.dump(man, open(rd / "manifest.json", "w"), indent=1)
    # local-only reveal browser (contains review text: gitignored and not deployed)
    ex = []
    for i, ix in enumerate(sealed):
        r = pred.iloc[i]; tt = t.iloc[ix]; why = explain(M, X[ix], int(np.argmax(out["Pv"][i])))
        ex.append(dict(text=texts[i], round=rnd, truth=dict(variety=nm.variety_norm.values[ix], group=tt.group, country=tt.country, province=nm.province.values[ix], region=nm.region_1.values[ix] if isinstance(nm.region_1.values[ix], str) else None,
                                                             winery=tt.winery, vintage=None if np.isnan(tt.vintage) else int(tt.vintage), price=None if np.isnan(tt.price) else float(tt.price), points=int(tt.points)),
                       guess=dict(variety=json.loads(r.variety), place=json.loads(r.place), producer=json.loads(r.producer), point=[round(r.lat, 2), round(r.lon, 2), round(r.r50_km), round(r.r80_km)],
                                  vintage=[round(r.vintage_lo, 1), round(r.vintage, 1), round(r.vintage_hi, 1)], price=[round(r.price_lo, 1), round(r.price, 1), round(r.price_hi, 1)], points=[round(r.points_lo, 1), round(r.points, 1), round(r.points_hi, 1)]),
                       why=why))
    json.dump(ex, open(DASH / "blind_examples.json", "w"), separators=(",", ":"))
    # CLOSE: the revealed ids join the ledger (never drawn again) and the training pool of the next round
    new = pd.DataFrame(dict(id=ids[sealed], round=rnd, tested_at=man["date"])); pd.concat([led, new]).to_csv(EXP / "ledger.csv", index=False)
    summarize_rounds()
    report = [f"# Blind round {rnd}", f"{n} notes, recipe `{RECIPE}`, seed {seed}, pool {len(pool)}, {man['seconds']} s", "", "| metric | " + " | ".join(tiers) + " |", "|---|" + "---|" * len(tiers)] + [f"| {k} | " + " | ".join(str(round(S_all[t_][k], 3)) if isinstance(S_all[t_][k], float) else str(S_all[t_][k]) for t_ in tiers) + " |" for k in S]
    open(rd / "report.md", "w").write("\n".join(report))
    print(json.dumps({t_: {k: round(v, 3) if isinstance(v, float) else v for k, v in S_all[t_].items() if k in ("variety_top1", "country_top1", "place_top3", "vintage_skill", "price_skill")} for t_ in tiers}, indent=1)); print(f"round {rnd} closed; ledger now {len(led) + len(new)} ids")


def status():
    led = ledger(); print(f"ledger: {len(led)} tested ids in {led['round'].nunique() if len(led) else 0} rounds")
    for r in summarize_rounds(): s = r["scores"]; print(f"  round {r['round']}: n={r['n']} grape top1 {s['variety_top1']:.3f} | country {s['country_top1']:.3f} | place top3 {s['place_top3']:.3f} | vintage MAE {s['vintage_mae']:.2f} | price skill {s['price_skill']:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "run": run(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 500, dry="--dry" in sys.argv)
    else: status()
