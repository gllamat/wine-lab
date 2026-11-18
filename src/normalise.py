"""Phase 3b: grape / vintage / colour normalisation -> outputs/reviews_norm.parquet"""
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import review_id, ROOT

ALIAS = {"shiraz": "Syrah", "pinot grigio": "Pinot Gris", "garnacha": "Grenache", "garnacha tintorera": "Grenache",
         "grenache blanc": "Grenache Blanc", "monastrell": "Mourvèdre", "mourvedre": "Mourvèdre", "carmenère": "Carmenère",
         "primitivo": "Zinfandel", "spatburgunder": "Pinot Noir", "spätburgunder": "Pinot Noir", "weissburgunder": "Pinot Blanc",
         "grauburgunder": "Pinot Gris", "tinta de toro": "Tempranillo", "tinto fino": "Tempranillo", "cencibel": "Tempranillo",
         "aragonez": "Tempranillo", "tinta roriz": "Tempranillo", "pinot nero": "Pinot Noir", "grüner veltliner": "Grüner Veltliner",
         "gruner veltliner": "Grüner Veltliner", "sauvignon": "Sauvignon Blanc", "fumé blanc": "Sauvignon Blanc",
         "malbec": "Malbec", "rosé": "Rosé", "rosato": "Rosé", "rosado": "Rosé"}
BLEND = re.compile(r"blend|style|cuv[eé]e|\bred\b|\bwhite\b|sparkling|champagne|port|portuguese|rhône|bordeaux", re.I)
RED = re.compile(r"cabernet|merlot|pinot noir|syrah|shiraz|malbec|tempranillo|sangiovese|nebbiolo|zinfandel|grenache|red|"
                 r"mourv|carmen|barbera|tannat|pinotage|nero|gamay|petite sirah|touriga|montepulciano|aglianico|"
                 r"mencía|monastrell|bonarda|corvina|primitivo|port|rhône|bordeaux-style red", re.I)
WHITE = re.compile(r"chardonnay|sauvignon blanc|riesling|pinot gris|pinot blanc|grüner|chenin|viognier|albariño|"
                   r"gewürztraminer|white|moscato|muscat|semillon|sémillon|verdejo|vermentino|garganega|trebbiano|"
                   r"fiano|greco|grillo|gruner|marsanne|roussanne|torrontés|furmint|pinot grigio|sparkling|champagne|prosecco|"
                   r"cava|glera|picpoul|melon|godello|arneis|cortese|falanghina", re.I)


def vintage_from_title(t):
    if not isinstance(t, str):
        return None
    t = re.sub(r"\(.*?\)", " ", t)
    for m in re.finditer(r"\b((?:19[5-9]\d|20[01]\d))\b", t):
        y = int(m.group(1))
        if 1950 <= y <= 2017:
            return y
    return None


def colour(v):
    if not isinstance(v, str):
        return "unknown"
    if re.search(r"ros[eé]", v, re.I):
        return "rose"
    if re.search(r"sparkling|champagne|prosecco|cava|glera", v, re.I):
        return "sparkling"
    if WHITE.search(v) and not re.search(r"red", v, re.I):
        return "white"
    if RED.search(v):
        return "red"
    return "other"


def main():
    df = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet")
    df["id"] = df.description.map(review_id)
    df = df.drop_duplicates("id").reset_index(drop=True)
    v = df.variety.astype("string")
    df["variety_norm"] = v.map(lambda x: ALIAS.get(x.lower(), x) if isinstance(x, str) else None)
    df["is_blend"] = v.map(lambda x: bool(BLEND.search(x)) if isinstance(x, str) else False)
    df["colour"] = df.variety_norm.map(colour)
    df["vintage"] = df.title.map(vintage_from_title).astype("Int64")
    # fallback: year mentioned in description as "2013 vintage" / "the 2012"
    m = df.description.str.extract(r"\b(?:the |this )?((?:19[5-9]\d|20[01]\d)) (?:vintage|harvest)\b")[0]
    df["vintage_src"] = df.vintage.notna().map({True: "title", False: None})
    miss = df.vintage.isna() & m.notna()
    df.loc[miss, "vintage"] = m[miss].astype(int)
    df.loc[miss, "vintage_src"] = "description"
    top = df.variety_norm.value_counts()
    df["variety_group"] = df.variety_norm.where(df.variety_norm.map(top) >= 400, "Other")
    keep = ["id", "country", "province", "region_1", "region_2", "winery", "variety", "variety_norm", "variety_group",
            "is_blend", "colour", "vintage", "vintage_src", "points", "price", "sources"]
    out = df[keep]
    out.to_parquet(ROOT / "outputs/reviews_norm.parquet", index=False)
    print("rows", len(out), "vintage coverage", round(out.vintage.notna().mean(), 3), out.vintage_src.value_counts().to_dict())
    print(out.colour.value_counts().to_dict())
    print("variety groups", out.variety_group.nunique())


if __name__ == "__main__":
    main()
