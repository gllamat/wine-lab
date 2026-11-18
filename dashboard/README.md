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
