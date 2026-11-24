"""Jev-aligned taste vector for the recommender (extends src/recsys.py).
Every wine of the 25,000-wine index gets 224 numbers: Jev's 216 yes/no answers + the level of 8 structure scales, predicted from the note's words
(experiments/recsys_sanity/jev_vector.py: TF-IDF + ridge, out-of-fold for labelled wines). This module
  build  : writes the browser data  dashboard/data/recsys_j.bin (int8 z-scores), recsys_hub.bin (uint8 hub offsets) and adds a 'jev' block to recsys_meta.json
  JevRec : the same maths as the browser (dashboard/recsys.js), on the QUANTISED data, for evaluation and the parity fixture
New ingredients over the cue-based recommender:
  * jev / jevh  : cosine in the Jev space (levels weighted LW x), and its fusion with the text embedding
  * taster's rules : same colour family and predicted sweetness within SWMAX levels of the query (applied before the top-k)
  * hub correction (CSLS): score - lam * r_c, r_c = mean cosine of wine c to its 10 nearest wines; lam = 0.5
Usage: recsys_jev.py build | fixture | eval"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT
import recsys as RS

DASH = ROOT / "dashboard/data"; EXP = ROOT / "experiments/recsys_sanity"
NB, SJ, LW, SWMAX, LAM = 216, 20.0, 3.0, 1.0, 0.5
SCALES = ["sweetness", "acidity", "tannin", "alcohol", "body", "flavour_intensity", "finish", "nose_intensity"]


def build():
    V = np.load(EXP / "jev_vectors.npy"); off = json.load(open(EXP / "hub_offsets.json")); n = len(V)
    schema = json.load(open(ROOT / "distill/pack/wset_distill_schema.json"))
    mu_b, sd_b = V[:, :NB].mean(0), V[:, :NB].std(0) + 1e-6; mu_l, sd_l = V[:, NB:].mean(0), V[:, NB:].std(0) + 1e-6
    Z = np.hstack([(V[:, :NB] - mu_b) / sd_b, (V[:, NB:] - mu_l) / sd_l]); Zq = np.clip(np.round(Z * SJ), -127, 127).astype(np.int8)
    Zq.tofile(DASH / "recsys_j.bin")
    hub = np.concatenate([off["jev"], off["emb"]]); hq = np.clip(np.round((np.array(hub) - 0.3) / 0.7 * 255), 0, 255).astype(np.uint8); hq.tofile(DASH / "recsys_hub.bin")
    meta = json.load(open(DASH / "recsys_meta.json"))
    meta["jev"] = dict(dim=NB + len(SCALES), nb=NB, scale=SJ, lw=LW, swmax=SWMAX, lam=LAM, hub_lo=0.3, hub_span=0.7, questions=schema["binary"], scales=SCALES,
                       level_names=[schema["groups"]["level__" + s] for s in SCALES], lev_mu=[round(float(x), 5) for x in mu_l], lev_sd=[round(float(x), 5) for x in sd_l],
                       mu=[round(float(x), 5) for x in mu_b], sd=[round(float(x), 5) for x in sd_b])
    json.dump(meta, open(DASH / "recsys_meta.json", "w"), separators=(",", ":"))
    print("wrote recsys_j.bin", round(Zq.nbytes / 1e6, 1), "MB | recsys_hub.bin", round(hq.nbytes / 1e3), "KB | meta +jev block")


class JevRec(RS.Rec):
    """Same maths as dashboard/recsys.js, on the quantised data the browser loads."""
    def __init__(self):
        super().__init__()
        meta = json.load(open(DASH / "recsys_meta.json")); j = meta["jev"]; self.jm = j; n = self.n
        Zq = np.fromfile(DASH / "recsys_j.bin", dtype=np.int8).reshape(n, j["dim"]); self.Zj = Zq.astype(np.float32) / j["scale"]
        W = np.ones(j["dim"], np.float32); W[j["nb"]:] = j["lw"]; self.M["jev"] = RS.l2(self.Zj * W)
        hq = np.fromfile(DASH / "recsys_hub.bin", dtype=np.uint8).astype(np.float32); hub = j["hub_lo"] + hq / 255 * j["hub_span"]; self.hub = dict(jev=hub[:n], emb=hub[n:])
        self.sweet = self.Zj[:, j["nb"]] * j["lev_sd"][0] + j["lev_mu"][0]; self.colour = self.d["colour"].astype(str)

    def cosh(self, space, q, hub=False):
        s = self.M[space] @ q[space]
        return s - LAM * self.hub[space] if hub and space in self.hub else s

    def qj(self, i): return dict(self.query(i), jev=self.M["jev"][i])

    def mask(self, i): return (self.colour == self.colour[i]) & (np.abs(self.sweet - self.sweet[i]) < SWMAX)

    def tech(self, name, i, k=10, rules=False, hub=False, alpha=0.5):
        q = self.qj(i); mk = self.mask(i) if rules else None
        if name == "jev": s = self.cosh("jev", q, hub)
        elif name == "jevh": s = alpha * self.zs(self.cosh("jev", q, hub)) + (1 - alpha) * self.zs(self.cosh("emb", q, hub))
        elif name == "emb": s = self.cosh("emb", q, hub)
        else: raise ValueError(name)
        return self.rank(s, k, (i,), mk)


def fixture(n=40, k=10):
    R = JevRec(); rng = np.random.RandomState(8); rows = []
    for i in rng.choice(R.n, n, replace=False):
        r = dict(i=int(i))
        for key, kw in dict(jev={}, jevh={}, jev_rules=dict(rules=True), jevh_rules=dict(rules=True), jevh_hub=dict(hub=True), jevh_rules_hub=dict(rules=True, hub=True), emb_hub=dict(hub=True)).items():
            r[key] = [int(x) for x in R.tech(key.split("_")[0], int(i), k, **kw)]
        rows.append(r)
    json.dump(rows, open(ROOT / "outputs/recsys_jev_parity.json", "w")); print("fixture", len(rows))


def evaluate(nq=3000, ne=8000, k=10):
    """Same proxy metrics as src/recsys.py evaluate (3,000 queries) for the old AND the new techniques. Coverage / hub numbers use 8,000 queries (80,000 list slots for 25,000 wines):
    the old 3,000-query figure could not show more than 30,000 slots, so 'never recommended' was mostly arithmetic, not hubness."""
    R = JevRec(); d = R.d; rng = np.random.RandomState(11); Q = rng.choice(R.n, nq, replace=False); QE = np.random.RandomState(12).choice(R.n, ne, replace=False)
    price, pts = d["price"].astype(float), d["points"].astype(float)
    TECH = [("random", lambda i: R.technique("random", i, k)), ("raw", lambda i: R.technique("raw", i, k)), ("zw", lambda i: R.technique("zw", i, k)), ("latent", lambda i: R.technique("latent", i, k)),
            ("emb", lambda i: R.technique("emb", i, k)), ("hybrid", lambda i: R.technique("hybrid", i, k)), ("rrf", lambda i: R.technique("rrf", i, k)), ("mmr", lambda i: R.technique("mmr", i, k)),
            ("jev", lambda i: R.tech("jev", i, k)), ("jevh", lambda i: R.tech("jevh", i, k)), ("jevh+rules", lambda i: R.tech("jevh", i, k, rules=True)), ("jevh+rules+fair", lambda i: R.tech("jevh", i, k, rules=True, hub=True))]
    out = {}
    for t, f in TECH:
        L = np.array([f(int(i)) for i in Q]); E = R.M["emb"]
        grp = (d["group"][L] == d["group"][Q][:, None]).mean(); cty = (d["country"][L] == d["country"][Q][:, None]).mean(); prv = (d["province"][L] == d["province"][Q][:, None]).mean(); win = (d["winery"][L] == d["winery"][Q][:, None]).mean()
        dp = np.abs(pts[L] - pts[Q][:, None]).mean(); pr = np.nanmedian(price[L] / price[Q][:, None]); ild = np.mean([1 - ((E[l] @ E[l].T).sum() - k) / (k * (k - 1)) for l in L])
        nct = np.mean([len(set(d["country"][l])) for l in L]); nwi = np.mean([len(set(d["winery"][l])) for l in L])
        LE = np.array([f(int(i)) for i in QE]); cnt = np.bincount(LE.ravel(), minlength=R.n); srt = np.sort(cnt)[::-1]; cum = np.cumsum(np.sort(cnt)); gini = float(1 - 2 * (cum / cnt.sum()).sum() / R.n + 1 / R.n)
        out[t] = dict(group=float(grp), country=float(cty), province=float(prv), winery=float(win), dpoints=float(dp), price_ratio=float(pr), ild=float(ild), countries=float(nct), wineries=float(nwi),
                      coverage=float((cnt > 0).mean()), hub_max=int(cnt.max()), never=float((cnt == 0).mean()), hub_top1pct_share=float(srt[:R.n // 100].sum() / cnt.sum()), gini=gini)
        print(f"{t:16s} grape {grp:.3f} country {cty:.3f} ild {ild:.3f} | coverage {out[t]['coverage']:.3f} never {out[t]['never']:.3f} hub_max {cnt.max()} gini {gini:.3f}", flush=True)
    old = json.load(open(DASH / "recsys_eval.json"))
    json.dump(dict(k=k, queries=nq, coverage_queries=ne, n_items=R.n, techniques=out, jaccard=old.get("jaccard")), open(DASH / "recsys_eval.json", "w"), indent=1); print("wrote recsys_eval.json")


if __name__ == "__main__":
    {"build": build, "fixture": fixture, "eval": evaluate}[sys.argv[1]]()
