"""Second geocoding pass: simpler 'region, country' query for regions the full query missed or got rejected.
Runs after geocode.py (shares the Nominatim 1 req/s budget). Most-reviewed first; resumable."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from geocode import query, CACHE
from places import load_places, queries

cache = json.load(open(CACHE))
pl = load_places()
from places import PROV_OVERRIDE
for q in PROV_OVERRIDE.values():          # ambiguous province names (e.g. Washington -> D.C.)
    if q not in cache:
        query(q, cache)
json.dump(cache, open(CACHE, "w"))
todo = pl[pl.pending_alt].sort_values("n", ascending=False)
print("alt queries to run:", len(todo), flush=True)
for i, r in enumerate(todo.itertuples(), 1):
    qq = queries(r.country, r.province, r.region_1)["q_alt"]
    if qq and qq not in cache:
        query(qq, cache)
    if i % 25 == 0:
        json.dump(cache, open(CACHE, "w")); print(i, flush=True)
json.dump(cache, open(CACHE, "w"))
print("done")
