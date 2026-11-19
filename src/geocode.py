"""Geocode unique (country, province, region_1) places via OpenStreetMap Nominatim (1 req/s, cached, resumable).
Only place names are sent. Fallback chain: region_1 -> province -> country centroid."""
import json, re, sys, time
from pathlib import Path
import pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache/nominatim.json"
CACHE.parent.mkdir(exist_ok=True)
UA = "wine-research-geocoder/0.1 (non-commercial research)"
CMAP = {"US": "United States", "England": "United Kingdom"}


def clean(x):
    if pd.isna(x):
        return None
    return re.sub(r"\s*\(.*?\)", "", str(x)).strip() or None


def query(q, cache):
    if q in cache:
        return cache[q]
    for attempt in range(3):
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": q, "format": "json", "limit": 1, "addressdetails": 0},
                             headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200:
                j = r.json()
                cache[q] = ({"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "name": j[0]["display_name"],
                             "type": j[0].get("type")} if j else None)
                time.sleep(1.1)
                return cache[q]
            time.sleep(5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None  # not cached -> retried next run


def main():
    df = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet")
    places = (df.groupby(["country", "province", "region_1"], dropna=False).size().rename("n").reset_index()
              .sort_values("n", ascending=False).reset_index(drop=True))
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    res = {}

    def q_for(p, lvl):
        if pd.isna(p.country):
            return None
        c = CMAP.get(p.country, p.country)
        prov, reg = clean(p.province), clean(p.region_1)
        parts = {"region_1": [reg, prov, c], "province": [prov, c], "country": [c]}[lvl]
        return None if (lvl == "region_1" and not reg) or (lvl == "province" and not prov) else ", ".join(x for x in parts if x)

    def write():
        rows = []
        for p in places.itertuples():
            best = None
            for lvl in ("region_1", "province", "country"):
                q = q_for(p, lvl)
                h = cache.get(q) if q else None
                if h:
                    best = (lvl, h); break
            rows.append(dict(country=p.country, province=p.province, region_1=p.region_1, n=p.n,
                             lat=best[1]["lat"] if best else None, lon=best[1]["lon"] if best else None,
                             geo_level=best[0] if best else None, geo_name=best[1]["name"] if best else None))
        out = pd.DataFrame(rows)
        out.to_parquet(ROOT / "outputs/places_geocoded.parquet", index=False)
        json.dump(cache, open(CACHE, "w"))
        return out

    # pass A: province (fallback country) for everything; pass B: region_1 by descending review count
    for lvl in ("province", "country", "region_1"):
        done = 0
        for p in places.itertuples():
            q = q_for(p, lvl)
            if q is None or (lvl == "country" and cache.get(q_for(p, "province"))):
                continue
            if lvl == "region_1" and cache.get(q_for(p, "region_1"), 0) != 0:
                continue
            query(q, cache)
            done += 1
            if done % 25 == 0:
                out = write()
                print(lvl, done, f"resolved={out.geo_level.value_counts().to_dict()}", flush=True)
    out = write()
    print("final", out.geo_level.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
