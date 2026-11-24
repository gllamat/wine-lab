# Wine WSET dashboard (static)

Pure static site: `index.html` + `data/*.json`. No build step on Vercel.

Rebuild data from the pipeline (from repo root):
    .venv/bin/python src/build_dashboard.py 20000

Deploy (run yourself; needs your Vercel login):
    cd dashboard && npx vercel --prod

Data policy: only aggregates, coordinates and metadata are published, never review text
(source reviews are scraped Wine Enthusiast text, reportedly CC BY-NC-SA). Weather: NASA POWER (MERRA-2 reanalysis; NASA open data).
Geocoding: (c) OpenStreetMap contributors via Nominatim.

## Jev API key
Stored in `.env` (gitignored, mode 600) as `TYPESAFE_API_KEY`; optional `TYPESAFE_KEY_EXPIRES=YYYY-MM-DD` for an expiry warning.
`src/key_status.py` probes the key (tiny real call, ~$0.000006) and writes `data/key_status.json` (status + timestamps only, never the key).
The dashboard banner turns red when the key stops working, and amber within 7 days of the expiry date.
Hourly checks: `sh scripts/install_keycheck.sh` (remove with `sh scripts/install_keycheck.sh remove`). Checks also run on every `build_dashboard.py` and on auth errors in `jev_pilot.py`.

## Recommender: Jev taste vector, taster's rules, hub correction, learned phrases
Extra data files for the recommender tab (all small; no review text):
- `data/recsys_j.bin` (int8, 25,000 x 224): Jev's 216 yes/no answers + 8 structure levels, predicted from each note; `data/recsys_hub.bin` (uint8): hub offsets; the `jev` block of `recsys_meta.json` holds the scales and names.
- `data/phrases.json` + `phrases.js`: phrases mined from labelled notes, to turn a typed description into the same vector in the browser.
- `data/recsys_sense.json`, `data/recsys_hubness.json` + `sense.js`: the "Does it make sense?" and hubness panels; `data/recsys_eval.json` also lists the new techniques.
Rebuild (from the repo root, in this order):
    .venv/bin/python experiments/recsys_sanity/jev_vector.py      # ~10 min CPU: out-of-fold ridge predictions (needs distill/pack)
    .venv/bin/python experiments/recsys_sanity/benchmark_hub.py   # hub offsets + hubness numbers
    .venv/bin/python src/recsys_jev.py build                      # writes recsys_j.bin, recsys_hub.bin, meta
    .venv/bin/python experiments/recsys_sanity/mine_phrases.py 20 0.6
    .venv/bin/python experiments/recsys_sanity/benchmark.py 1500 && .venv/bin/python experiments/recsys_sanity/export_site.py
    .venv/bin/python src/recsys_jev.py eval                       # recsys_eval.json
Parity (Python vs JavaScript): `.venv/bin/python src/recsys_jev.py fixture && node dashboard/recsys_jev_parity.js`;
`.venv/bin/python experiments/recsys_sanity/phrases_fixture.py && node dashboard/phrases_parity.js`.
Licence note: the vectors and phrases are derived from critics' notes (CC BY-NC-SA) and from answers of the Jev model; the site is non-commercial.
