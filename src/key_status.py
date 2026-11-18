"""TypeSafe API key health check. Reads TYPESAFE_API_KEY from env or ./.env (gitignored). Never prints/stores the key.
Writes dashboard/data/key_status.json (status + timestamps only) and appends outputs/key_status_history.jsonl.
Run: key_status.py  (hourly via launchd; also called by jev_pilot on auth errors)."""
import os, sys, json, datetime as dt
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "dashboard/data/key_status.json"
HIST = ROOT / "outputs/key_status_history.jsonl"
URL = "https://api.typesafe.ai/v1/systemone"
FAILING = {"invalid_or_expired", "payment_or_quota"}


def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def probe(key):
    body = dict(model="jev-latest", state="ping", questions={"p": dict(type="noul", instructions="Is this a greeting?")})
    try:
        r = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    except requests.RequestException as e:
        return "unreachable", None, type(e).__name__
    c = r.status_code
    if c == 200:
        return "ok", c, "key works"
    if c in (401, 403):
        return "invalid_or_expired", c, "key rejected (expired, revoked or wrong)"
    if c == 402:
        return "payment_or_quota", c, "credits or quota exhausted"
    if c == 429:
        return "rate_limited", c, "rate limited (key is valid)"
    return "error", c, f"unexpected HTTP {c}"


def run():
    load_env()
    key = os.environ.get("TYPESAFE_API_KEY")
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not key:
        status, code, msg = "missing", None, "TYPESAFE_API_KEY not set (.env)"
    else:
        status, code, msg = probe(key)
    st = dict(prev)
    st.update(status=status, http=code, message=msg, checked_at=now(), expires_on=os.environ.get("TYPESAFE_KEY_EXPIRES") or None)
    if status in ("ok", "rate_limited"):
        st["last_ok_at"] = st["checked_at"]; st["failing_since"] = None; st["consecutive_failures"] = 0
    elif status in FAILING or status == "missing":
        st["failing_since"] = prev.get("failing_since") or st["checked_at"]
        st["consecutive_failures"] = int(prev.get("consecutive_failures") or 0) + 1
    st.setdefault("last_ok_at", None); st.setdefault("failing_since", None); st.setdefault("consecutive_failures", 0)
    st["renew_url"] = "https://console.typesafe.ai/keys"
    STATE.parent.mkdir(parents=True, exist_ok=True); HIST.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))
    with open(HIST, "a") as f:
        f.write(json.dumps({k: st[k] for k in ("checked_at", "status", "http")}) + "\n")
    return st


if __name__ == "__main__":
    s = run()
    print(f"{s['checked_at']}  {s['status']}  http={s['http']}  last_ok={s['last_ok_at']}  failing_since={s['failing_since']}")
    sys.exit(0 if s["status"] in ("ok", "rate_limited") else 1)
