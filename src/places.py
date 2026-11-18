"""Single source of truth for place coordinates: built from the Nominatim cache with validation.
Preference: region (full query) > region (simple 'region, country' query) > province > country.
A region hit is REJECTED if it lies > MAX_KM from its province anchor (or > MAX_KM_COUNTRY from the country anchor
when the province itself did not resolve) -> falls back and is reported as a gap."""
import json, math
from pathlib import Path
import pandas as pd
from geocode import clean, CMAP, CACHE

ROOT = Path(__file__).resolve().parent.parent
# plausibility limit: max distance of a hit from its COUNTRY centroid (catches wrong-country matches only;
# wrong-region-within-country errors, e.g. a different "Carneros", cannot be detected this way).
LIMIT_KM = {"US": 3600, "Chile": 2600, "Australia": 3200, "Argentina": 2200, "Canada": 4500}
DEFAULT_LIMIT_KM = 1100
PROV_OVERRIDE = {("US", "Washington"): "Washington State, United States", ("US", "New York"): "New York State, United States"}


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def queries(country, province, region_1):
    c = CMAP.get(country, country) if isinstance(country, str) else None
    prov, reg = clean(province), clean(region_1)
    return dict(country=c, prov=prov, reg=reg,
                q_full=", ".join(x for x in [reg, prov, c] if x) if reg else None,
                q_alt=f"{reg}, {c}" if reg and c else None,
                q_prov=PROV_OVERRIDE.get((country, province)) or (", ".join(x for x in [prov, c] if x) if prov else None))


def load_places():
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    df = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["country", "province", "region_1"])
    pl = df.groupby(["country", "province", "region_1"], dropna=False).size().rename("n").reset_index()
    out = []
    for r in pl.itertuples():
        q = queries(r.country, r.province, r.region_1)
        hp = cache.get(q["q_prov"]) if q["q_prov"] else None
        hc = cache.get(q["country"]) if q["country"] else None
        anchor, lim = hc, LIMIT_KM.get(r.country, DEFAULT_LIMIT_KM)
        chosen, level, rejected_km = None, None, None
        for lvl, qq in (("region_1", q["q_full"]), ("region_1_alt", q["q_alt"])):
            h = cache.get(qq) if qq else None
            if not h:
                continue
            d = km(h, anchor) if anchor else 0.0
            if d <= lim:
                chosen, level = h, lvl
                break
            rejected_km = d if rejected_km is None else min(rejected_km, d)
        if chosen is None and hp:
            d = km(hp, anchor) if anchor else 0.0
            if d <= lim:
                chosen, level = hp, "province"
            else:
                rejected_km = d if rejected_km is None else min(rejected_km, d)
        if chosen is None and hc:
            chosen, level = hc, "country"
        out.append(dict(country=r.country, province=r.province, region_1=r.region_1, n=r.n,
                        lat=chosen["lat"] if chosen else None, lon=chosen["lon"] if chosen else None,
                        geo_level=level, geo_name=chosen["name"] if chosen else None,
                        rejected_km=None if rejected_km is None else round(rejected_km),
                        pending_alt=bool(q["reg"] and level not in ("region_1", "region_1_alt") and q["q_alt"] not in cache)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    p = load_places()
    print(p.geo_level.value_counts(dropna=False).to_dict())
    print("rejected region hits:", int(p.rejected_km.notna().sum()), "| pending alt queries:", int(p.pending_alt.sum()))
    print(p[p.rejected_km.notna()].sort_values("n", ascending=False).head(8)[["province", "region_1", "n", "rejected_km"]].to_string(index=False))
