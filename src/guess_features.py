"""Feature layer of the guess model. ONE definition, two code paths that must agree (and are tested to):
  matrix_from_vectors(vec, spec)  training path, from outputs/wset_vectors_*.parquet columns
  featurize_hits(...)             single-text path (same logic the browser will run), from matcher hits
Inputs are the SAT vector only (rule-based, no embedding-mapped terms) + two text-shape cues (note length, year hints).
Nothing about the wine's metadata (colour label, country, price...) is ever an input: the model sees the description only."""
import math, json
import numpy as np, pandas as pd

BOUNDS = ("|", "#")
YEAR0, YEARSD = 2005.0, 10.0


def build_spec(vec, lex, min_cat=100, min_na=0.01):
    cols = [c for c in vec if c.startswith("a_")]
    scales = [c[2:] for c in vec if c.startswith("s_")]
    na = [s for s in scales if vec["s_" + s].notna().mean() > min_na]
    cats = []
    for c in [c for c in vec if c.startswith("cat_")]:
        vc = vec[c].dropna().str.split("|").explode().value_counts()
        cats += [(c[4:], v) for v in vc[vc >= min_cat].index]
    names = cols + ["s_" + s for s in scales] + ["s_" + s + "_na" for s in na] + [f"cat_{f}={v}" for f, v in cats] + \
            ["r_drink_now", "r_can_age", "r_too_old", "yf", "yf_na", "yt", "yt_na", "len_log"]
    return dict(names=names, descriptors=cols, scales={s: len(lex.scales[s]) for s in scales}, na_scales=na, cats=[[f, v] for f, v in cats])


def matrix_from_vectors(vec, spec):
    n = len(vec); X = np.zeros((n, len(spec["names"])), dtype=np.float32); ix = {c: i for i, c in enumerate(spec["names"])}
    for c in spec["descriptors"]:
        X[:, ix[c]] = vec[c].values
    for s, L in spec["scales"].items():
        m = vec["s_" + s].values.astype(float); ok = ~np.isnan(m)
        X[ok, ix["s_" + s]] = (m[ok] - 1) / max(1, L - 1)
        if s in spec["na_scales"]:
            X[:, ix["s_" + s + "_na"]] = (~ok).astype(np.float32)
    for f, v in spec["cats"]:
        X[:, ix[f"cat_{f}={v}"]] = vec["cat_" + f].fillna("").map(lambda s: v in s.split("|")).values
    for k in ("r_drink_now", "r_can_age", "r_too_old"):
        X[:, ix[k]] = vec[k].values
    for k, col in (("yf", "r_year_from"), ("yt", "r_year_to")):
        y = vec[col].values.astype(float); ok = ~np.isnan(y)
        X[ok, ix[k]] = np.clip((y[ok] - YEAR0) / YEARSD, -2, 3); X[:, ix[k + "_na"]] = (~ok).astype(np.float32)
    X[:, ix["len_log"]] = np.log1p(vec["n_tokens"].values)
    return X


def featurize_hits(hits, years, toks, spec, resolve_ripeness):
    """Same features from one text's matcher output. hits: list of dicts (kind,f1,f2,neg,...)."""
    ix = {c: i for i, c in enumerate(spec["names"])}; x = np.zeros(len(spec["names"]), dtype=np.float32)
    pos = [h for h in hits if not h["neg"]]
    ar = [h for h in pos if h["kind"] == "aroma"]
    for h in ar:
        if h["f1"] != "fruit_ripeness":
            k = f"a_{h['f1']}__{h['f2']}"
            if k in ix: x[ix[k]] = 1
    rip, _ = resolve_ripeness([h["f2"] for h in ar if h["f1"] == "fruit_ripeness" or (h["f1"] in ("tertiary_red", "tertiary_white") and h["f2"] in ("dried_fruit", "cooked_fruit"))])
    if rip: x[ix[f"a_fruit_ripeness__{rip}"]] = 1
    for s, L in spec["scales"].items():
        lv = [h["f2"] for h in pos if h["kind"] == "structure" and h["f1"] == s]
        if lv:
            x[ix["s_" + s]] = (float(np.mean(lv)) - 1) / max(1, L - 1)
        elif s in spec["na_scales"]:
            x[ix["s_" + s + "_na"]] = 1
    for f, v in spec["cats"]:
        if any(h["kind"] == "cat" and h["f1"] == f and h["f2"] == v for h in pos): x[ix[f"cat_{f}={v}"]] = 1
    for k in ("drink_now", "can_age", "too_old"):
        if any(h["kind"] == "readiness" and h["f1"] == k for h in pos): x[ix["r_" + k]] = 1
    for k, val in (("yf", years[0][0] if years else None), ("yt", years[0][1] if years and years[0][1] else None)):
        if val is None: x[ix[k + "_na"]] = 1
        else: x[ix[k]] = min(3, max(-2, (val - YEAR0) / YEARSD))
    x[ix["len_log"]] = math.log1p(len([t for t in toks if t not in BOUNDS]))
    return x
