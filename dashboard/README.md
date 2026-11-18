# Wine WSET dashboard (static)

Pure static site: `index.html` + `data/*.json`. No build step on Vercel.

Rebuild data from the pipeline (from repo root):
    .venv/bin/python src/build_dashboard.py 20000

Deploy (run yourself; needs your Vercel login):
    cd dashboard && npx vercel --prod

Data policy: only aggregates, coordinates and metadata are published, never review text
(source reviews are scraped Wine Enthusiast text, reportedly CC BY-NC-SA). Weather: Open-Meteo (CC BY 4.0, non-commercial free tier).
Geocoding: (c) OpenStreetMap contributors via Nominatim.
