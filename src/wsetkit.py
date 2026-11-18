"""Deterministic text normalisation + lexicon matcher for wine reviews -> partial WSET vectors."""
import re, unicodedata, hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
NEG = {"no", "not", "without", "lacks", "lacking", "lack", "devoid", "minus", "neither", "nor",
       "never", "free", "absent", "hardly"}
BOUNDARY = {"|", "but", "yet", "although", "though", "however", "while", "with", "and"}
NEG_WINDOW = 3
LEMMA_EXC = {"molasses": "molasses", "citrus": "citrus", "asparagus": "asparagus", "grass": "grass",
             "christmas": "christmas", "lees": "lees", "bias": "bias", "canvas": "canvas"}

# Wine Enthusiast points -> WSET-style quality band (project assumption, documented in README)
QUALITY_BANDS = [(82, "acceptable"), (86, "good"), (90, "very_good"), (101, "outstanding")]


def lemma(t: str) -> str:
    if t in LEMMA_EXC:
        return t
    if len(t) <= 3 or not t.isalpha():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    if t.endswith(("ches", "shes", "xes", "zes")):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str):
    t = unicodedata.normalize("NFKD", str(text).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("'", "")
    t = re.sub(r"[-–—/]", " ", t)
    t = re.sub(r"[.;:,!?()\[\]\"“”]", " | ", t)
    t = re.sub(r"[^a-z0-9| ]", " ", t)
    return [lemma(x) for x in t.split()]


def review_id(description: str) -> str:
    k = re.sub(r"\s+", " ", str(description).lower().strip())
    return hashlib.md5(k.encode()).hexdigest()[:12]


def quality_band(points):
    try:
        p = float(points)
    except (TypeError, ValueError):
        return None
    for hi, name in QUALITY_BANDS:
        if p <= hi:
            return name
    return None


class Lexicon:
    def __init__(self, extra=None):
        self.entries = {}      # token tuple -> list of (kind, field, value, tag); one surface may set several kinds
        self.conflicts = []
        self.descriptors = []  # (cluster, descriptor)
        self.scales = {}
        a = yaml.safe_load(open(ROOT / "lexicon/aromas.yaml"))
        s = yaml.safe_load(open(ROOT / "lexicon/structure.yaml"))
        for group in ("primary", "secondary", "oak", "tertiary"):
            for cluster, descs in a[group].items():
                for desc, forms in descs.items():
                    self.descriptors.append((cluster, desc))
                    for f in forms:
                        self._add(f, ("aroma", cluster, desc))
        for scale, d in s["scales"].items():
            self.scales[scale] = d["levels"]
            for lvl, forms in d["cues"].items():
                for f in forms:
                    self._add(f, ("structure", scale, int(lvl)))
        for f in s["readiness"]["drink_now"]:
            self._add(f, ("readiness", "drink_now", 1))
        for f in s["readiness"]["can_age"]:
            self._add(f, ("readiness", "can_age", 1))
        ov = yaml.safe_load(open(ROOT / "lexicon/overrides.yaml"))
        for f in ov["drop"]:
            self.entries.pop(tuple(tokenize(f)), None)
        self.heads = {tokenize(k)[0]: v for k, v in ov["heads"].items()}
        self.bound = {tokenize(k)[0]: int(v) for k, v in ov["bound"].items()}
        self.year_re = re.compile(s["readiness"]["year_range_regex"])
        for f, tgt in (extra or {}).items():       # BERT-derived additions
            self._add(f, tuple(tgt), tag="embed")
        self.maxlen = max(len(k) for k in self.entries)

    def _add(self, form, target, tag="rule"):
        key = tuple(tokenize(form))
        if not key:
            return
        cur = self.entries.setdefault(key, [])
        for t in cur:
            if t[:3] == target:
                return
            if t[0] == target[0]:      # same kind, different value -> real conflict, keep first
                self.conflicts.append((" ".join(key), t[:3], target))
                return
        cur.append(target + (tag,))

    def _find_head(self, toks, i):
        """Head noun for a bound adjective at i. Returns (head_idx, direct).
        direct: adjective + <=1 token + head in same clause -> consume through head.
        cross: 'firm, grippy tannins' / 'soft and silky tannins' -> adjective only, tolerate , and + other cue words."""
        for j in range(i + 1, min(len(toks), i + 3)):
            if toks[j] == "|":
                break
            if toks[j] in self.heads:
                return j, True
        nonskip = 0
        for j in range(i + 1, min(len(toks), i + 7)):
            t = toks[j]
            if t in self.heads:
                return j, False
            if t in ("|", "and"):
                continue
            ok = t in self.bound or any(e[0] == "structure" for e in self.entries.get((t,), []))
            if not ok:
                break
            nonskip += 1
            if nonskip > 2:
                break
        return None, False

    def match(self, text: str):
        toks = tokenize(text)
        i, out, covered = 0, [], [False] * len(toks)
        while i < len(toks):
            if toks[i] in self.bound:
                head, direct = self._find_head(toks, i)
                if head is not None:
                    neg = any(toks[k] in NEG for k in range(max(0, i - NEG_WINDOW), i)
                              if not any(t in BOUNDARY for t in toks[k:i]))
                    out.append(dict(kind="structure", f1=self.heads[toks[head]], f2=self.bound[toks[i]],
                                    surface=toks[i] + " " + toks[head] if direct else toks[i] + " … " + toks[head],
                                    neg=neg, tag="bound" if direct else "bound-x"))
                    if direct:
                        for j in range(i, head + 1):
                            covered[j] = True
                        i = head + 1
                    else:
                        covered[i] = True
                        i += 1
                    continue
            hit = None
            for n in range(min(self.maxlen, len(toks) - i), 0, -1):
                key = tuple(toks[i:i + n])
                if key in self.entries:
                    hit = (n, key)
                    break
            if hit:
                n, key = hit
                neg = False
                for j in range(i - 1, max(-1, i - 1 - NEG_WINDOW), -1):
                    if toks[j] in BOUNDARY:
                        break
                    if toks[j] in NEG:
                        neg = True
                        break
                for kind, a, b, tag in self.entries[key]:
                    out.append(dict(kind=kind, f1=a, f2=b, surface=" ".join(key), neg=neg, tag=tag))
                for j in range(i, i + n):
                    covered[j] = True
                i += n
            else:
                i += 1
        low = str(text).lower()
        years = []
        for m in self.year_re.finditer(low):
            y1 = int(m.group(1))
            y2 = m.group(2)
            if y2 is not None:
                y2 = int(y2) if len(y2) == 4 else int(str(y1)[:2] + y2)
            years.append((y1, y2))
        return toks, covered, out, years
