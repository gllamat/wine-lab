"""Phase 7: growing-season weather per (place, vintage) from the Open-Meteo historical archive (ERA5-based).
Only coordinates are sent. Cached per location, resumable. Data: Open-Meteo.com (CC BY 4.0; free tier is non-commercial).
Usage: weather.py [--limit K]"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache/weather"; CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,shortwave_radiation_sum"
START, END = "2000-01-01", "2017-12-31"


def loc_key(lat, lon):
    return f"{round(lat, 1):.1f}_{round(lon, 1):.1f}"


def fetch(lat, lon, tries=5):
    k = loc_key(lat, lon)
    f = CACHE / f"{k}.json"
    if f.exists():
        return json.load(open(f))
    for a in range(tries):
        try:
            r = requests.get(URL, params=dict(latitude=round(lat, 1), longitude=round(lon, 1), start_date=START, end_date=END,
                                              daily=DAILY, timezone="UTC"), timeout=120)
            if r.status_code == 200:
                d = r.json()["daily"]
                json.dump(d, open(f, "w"))
                time.sleep(0.4)
                return d
            if r.status_code == 429:
                time.sleep(30 * (a + 1))        # rate limit: back off
            else:
                time.sleep(5)
        except requests.RequestException:
            time.sleep(5)
    return None


def huglin_k(lat):
    a = abs(lat)
    return 1.0 if a < 40 else 1.02 if a < 42 else 1.03 if a < 44 else 1.04 if a < 46 else 1.05 if a < 48 else 1.06


def season_metrics(d, lat, vintage):
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"])
    nh = lat >= 0
    if nh:
        s, e = pd.Timestamp(vintage, 4, 1), pd.Timestamp(vintage, 10, 31)
        hv = (pd.Timestamp(vintage, 9, 1), pd.Timestamp(vintage, 10, 31))
        bud = (pd.Timestamp(vintage, 3, 15), pd.Timestamp(vintage, 4, 30))
        hug = (pd.Timestamp(vintage, 4, 1), pd.Timestamp(vintage, 9, 30))
    else:  # southern hemisphere: vintage = harvest year, season starts previous October
        s, e = pd.Timestamp(vintage - 1, 10, 1), pd.Timestamp(vintage, 4, 30)
        hv = (pd.Timestamp(vintage, 2, 1), pd.Timestamp(vintage, 4, 30))
        bud = (pd.Timestamp(vintage - 1, 9, 15), pd.Timestamp(vintage - 1, 10, 31))
        hug = (pd.Timestamp(vintage - 1, 10, 1), pd.Timestamp(vintage, 3, 31))
    w = df[(df.time >= s) & (df.time <= e)]
    if len(w) < 150:
        return None
    h = df[(df.time >= hug[0]) & (df.time <= hug[1])]
    tm, tx, tn = w.temperature_2m_mean, w.temperature_2m_max, w.temperature_2m_min
    return dict(
        gdd=float(np.clip(tm - 10, 0, None).sum()),
        huglin=float(huglin_k(lat) * (((h.temperature_2m_mean - 10).clip(lower=0) + (h.temperature_2m_max - 10).clip(lower=0)) / 2).sum()),
        t_mean=float(tm.mean()), t_max_mean=float(tx.mean()), diurnal=float((tx - tn).mean()),
        days_gt35=int((tx > 35).sum()), days_gt30=int((tx > 30).sum()),
        rain_season=float(w.precipitation_sum.sum()),
        rain_harvest=float(df[(df.time >= hv[0]) & (df.time <= hv[1])].precipitation_sum.sum()),
        rain_days=int((w.precipitation_sum >= 1).sum()),
        frost_budbreak=int((df[(df.time >= bud[0]) & (df.time <= bud[1])].temperature_2m_min < 0).sum()),
        radiation=float(w.shortwave_radiation_sum.sum()),
    )


def main():
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    pl = pd.read_parquet(ROOT / "outputs/places_geocoded.parquet").dropna(subset=["lat"])
    pl["loc"] = [loc_key(a, b) for a, b in zip(pl.lat, pl.lon)]
    locs = pl.drop_duplicates("loc")[["loc", "lat", "lon"]]
    if lim:
        locs = locs.head(lim)
    for i, r in enumerate(locs.itertuples(), 1):
        fetch(r.lat, r.lon)
        if i % 25 == 0:
            print(f"{i}/{len(locs)}", flush=True)
    rows = []
    nm = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet", columns=["country", "province", "region_1", "vintage"])
    vint = nm.dropna(subset=["vintage"]).merge(pl[["country", "province", "region_1", "loc", "lat"]],
                                               on=["country", "province", "region_1"], how="inner")
    pairs = vint[["loc", "lat", "vintage"]].drop_duplicates()
    for r in pairs.itertuples():
        if not (2000 <= r.vintage <= 2017) or not (CACHE / f"{r.loc}.json").exists():
            continue
        m = season_metrics(json.load(open(CACHE / f"{r.loc}.json")), r.lat, int(r.vintage))
        if m:
            rows.append(dict(loc=r.loc, vintage=int(r.vintage), **m))
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "outputs/weather_metrics.parquet", index=False)
    print("place-vintage rows", len(out))


if __name__ == "__main__":
    main()
