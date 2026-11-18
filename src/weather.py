"""Phase 7: growing-season weather per (place, vintage) from NASA POWER (MERRA-2 reanalysis, 0.5x0.625 deg, free, no key).
Only coordinates are sent. Cached per grid cell, resumable. (Open-Meteo was tried first: its minutely quota allows ~1 location/min.)
Usage: weather.py [--limit K]"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache/weather_power"; CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START, END = "20000101", "20171231"
# NASA POWER (MERRA-2 based, public domain, no key). Native grid 0.5 deg lat x 0.625 deg lon -> one request per cell.
LAT_STEP, LON_STEP = 0.5, 0.625


def loc_key(lat, lon):
    import math
    return f"{math.floor(lat / LAT_STEP)}_{math.floor(lon / LON_STEP)}"


def cell_centre(key):
    i, j = key.split("_")
    return (int(i) + 0.5) * LAT_STEP, (int(j) + 0.5) * LON_STEP


def fetch(lat, lon, tries=5):
    k = loc_key(lat, lon)
    f = CACHE / f"{k}.json"
    if f.exists():
        return json.load(open(f))
    clat, clon = cell_centre(k)
    for a in range(tries):
        try:
            r = requests.get(URL, params=dict(parameters="T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN", community="AG",
                                              longitude=round(clon, 3), latitude=round(clat, 3), start=START, end=END, format="JSON"),
                             timeout=120)
            if r.status_code == 200:
                P = r.json()["properties"]["parameter"]
                days = sorted(P["T2M"])
                g = lambda name, mult=1.0: [None if P[name][d] == -999 else P[name][d] * mult for d in days]
                d = {"time": [f"{x[:4]}-{x[4:6]}-{x[6:]}" for x in days], "temperature_2m_mean": g("T2M"),
                     "temperature_2m_max": g("T2M_MAX"), "temperature_2m_min": g("T2M_MIN"), "precipitation_sum": g("PRECTOTCORR"),
                     "shortwave_radiation_sum": g("ALLSKY_SFC_SW_DWN", 3.6)}      # kWh/m2/day -> MJ/m2/day
                json.dump(d, open(f, "w"))
                time.sleep(1.0)
                return d
            fetch.reason = "rate_limited" if r.status_code == 429 else f"http_{r.status_code}"
            time.sleep(15 * (a + 1))
        except (requests.RequestException, KeyError, ValueError):
            fetch.reason = "network_or_parse"
            time.sleep(5)
    return None


fetch.reason = None
STATUS = ROOT / "cache/weather_status.json"


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
    from places import load_places
    pl = load_places().dropna(subset=["lat"])
    pl["loc"] = [loc_key(a, b) for a, b in zip(pl.lat, pl.lon)]
    nm0 = pd.read_parquet(ROOT / "outputs/reviews_norm.parquet", columns=["country", "province", "region_1", "vintage"])
    nv = (nm0[nm0.vintage.between(2000, 2017)].groupby(["country", "province", "region_1"], dropna=False).size()
          .rename("n_v").reset_index())
    pl = pl.merge(nv, on=["country", "province", "region_1"], how="left")
    pl["n_v"] = pl.n_v.fillna(0)
    locs = (pl[pl.n_v > 0].groupby("loc").agg(lat=("lat", "first"), lon=("lon", "first"), n_v=("n_v", "sum"))
            .sort_values("n_v", ascending=False).reset_index())          # most-reviewed first
    if lim:
        locs = locs.head(lim)
    status = json.load(open(STATUS)) if STATUS.exists() else {}
    consecutive = 0
    for i, r in enumerate(locs.itertuples(), 1):
        if (CACHE / f"{r.loc}.json").exists():
            status[r.loc] = {"status": "ok"}
            continue
        fetch.reason = None
        d = fetch(r.lat, r.lon)
        if d is not None:
            status[r.loc] = {"status": "ok"}; consecutive = 0
        else:
            status[r.loc] = {"status": "failed", "reason": fetch.reason, "at": time.strftime("%Y-%m-%d %H:%M")}
            consecutive += 1
            if fetch.reason == "rate_limited" and consecutive >= 3:
                print("Rate limit persists (likely the daily quota). Stopping; rerun later to resume.", flush=True)
                break
        if i % 25 == 0:
            json.dump(status, open(STATUS, "w"))
            print(f"{i}/{len(locs)} ok={sum(v['status']=='ok' for v in status.values())}", flush=True)
    json.dump(status, open(STATUS, "w"))
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
