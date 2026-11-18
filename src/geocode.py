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
    places = df[["country", "province", "region_1"]].drop_duplicates().reset_index(drop=True)
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    rows = []
    for i, p in enumerate(places.itertuples(), 1):
        c = CMAP.get(p.country, p.country)
        prov, reg = clean(p.province), clean(p.region_1)
        hit, level = None, None
        for lvl, parts in (("region_1", [reg, prov, c]), ("province", [prov, c]), ("country", [c])):
            if lvl == "region_1" and not reg:
                continue
            if lvl == "province" and not prov:
                continue
            hit = query(", ".join(x for x in parts if x), cache)
            if hit:
                level = lvl
                break
        rows.append(dict(country=p.country, province=p.province, region_1=p.region_1,
                         lat=hit["lat"] if hit else None, lon=hit["lon"] if hit else None,
                         geo_level=level, geo_name=hit["name"] if hit else None))
        if i % 25 == 0:
            json.dump(cache, open(CACHE, "w"))
            print(f"{i}/{len(places)}", flush=True)
    json.dump(cache, open(CACHE, "w"))
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "outputs/places_geocoded.parquet", index=False)
    print(out.geo_level.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
