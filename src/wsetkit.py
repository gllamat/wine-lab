"""Deterministic text normalisation + lexicon matcher: wine review -> partial WSET L3 SAT (3W v2.0, 2016) vector.
v2: SAT-aligned clusters, nose/palate scope, colour-resolved tertiary clusters, categorical fields (colour, clarity, condition)."""
import re, unicodedata, hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
NEG = {"no", "not", "without", "lacks", "lacking", "lack", "devoid", "minus", "neither", "nor",
       "never", "free", "absent", "hardly"}
BOUNDARY = {"|", "#", "but", "yet", "although", "though", "however", "while", "with", "and"}
NEG_WINDOW = 3
LEMMA_EXC = {"molasses": "molasses", "citrus": "citrus", "asparagus": "asparagus", "grass": "grass",
             "christmas": "christmas", "lees": "lees", "bias": "bias", "canvas": "canvas"}
NOSE_TRIG = {"aroma", "nose", "bouquet", "smell", "scent"}
PALATE_TRIG = {"palate", "flavor", "flavour", "taste", "mouth", "mouthfeel", "sip"}
SCOPE_WINDOW = 6
# SAT quality scale has 6 levels (faulty..outstanding). We only PROXY it from critic points (80-100 in this corpus):
QUALITY_BANDS = [(82, "acceptable"), (86, "good"), (90, "very_good"), (101, "outstanding")]
SAT_QUALITY_LEVELS = ["faulty", "poor", "acceptable", "good", "very_good", "outstanding"]


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
    t = re.sub(r"[.;!?]", " # ", t)               # sentence boundary
    t = re.sub(r"[:,()\[\]\"“”]", " | ", t)       # clause boundary
    t = re.sub(r"[^a-z0-9|# ]", " ", t)
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


def colour_group(colour):
    """wine colour (from variety) -> SAT lexicon variant: red / white (rose & sparkling use the white lists) / None"""
    return {"red": "red", "white": "white", "rose": "white", "sparkling": "white"}.get(colour)


class Lexicon:
    def __init__(self, extra=None):
        self.entries = {}      # token tuple -> list of (kind, f1, f2, tag); a surface may set several targets
        self.conflicts = []
        self.descriptors = []  # (cluster, descriptor)
        self.tier = {}         # cluster -> primary|secondary|tertiary
        self.scales = {}
        self.cat_fields = {}   # categorical field -> sorted values
        a = yaml.safe_load(open(ROOT / "lexicon/aromas.yaml"))
        s = yaml.safe_load(open(ROOT / "lexicon/structure.yaml"))
        for tier in ("primary", "secondary", "tertiary"):
            for cluster, descs in a[tier].items():
                self.tier[cluster] = tier
                for desc, forms in descs.items():
                    self.descriptors.append((cluster, desc))
                    for f in forms:
                        self._add(f, ("aroma", cluster, desc))
        for scale, d in s["scales"].items():
            self.scales[scale] = d["levels"]
            for lvl, forms in (d["cues"] or {}).items():
                for f in forms:
                    self._add(f, ("structure", scale, int(lvl)))
        for item in s.get("categorical", []):
            for field, value in item["set"].items():
                self._add(item["form"], ("cat", field, value))
                self.cat_fields.setdefault(field, set()).add(value)
        self.cat_fields = {k: sorted(v) for k, v in self.cat_fields.items()}
        for flag in ("drink_now", "can_age", "too_old"):
            for f in s["readiness"][flag]:
                self._add(f, ("readiness", flag, 1))
        ov = yaml.safe_load(open(ROOT / "lexicon/overrides.yaml"))
        for f in ov["drop"]:
            self.entries.pop(tuple(tokenize(f)), None)
        self.heads = {tokenize(k)[0]: v for k, v in ov["heads"].items()}
        self.bound = {tokenize(k)[0]: int(v) for k, v in ov["bound"].items()}
        self.year_re = re.compile(s["readiness"]["year_range_regex"])
        for f, tgt in (extra or {}).items():       # embedding-mapped additions
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
            if t[0] == target[0] and target[0] != "aroma":   # aroma: several clusters per surface is intended (SAT is ambiguous)
                if target[0] in ("cat", "readiness") and t[1] != target[1]:
                    continue                                   # different categorical field / readiness flag: both apply
                self.conflicts.append((" ".join(key), t[:3], target))
                return
        cur.append(target + (tag,))

    # ---- helpers
    def _find_head(self, toks, i):
        """Head noun for a bound adjective at i -> (head_idx, direct)."""
        for j in range(i + 1, min(len(toks), i + 3)):
            if toks[j] in ("|", "#"):
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
            if t == "#":
                break
            ok = t in self.bound or any(e[0] == "structure" for e in self.entries.get((t,), []))
            if not ok:
                break
            nonskip += 1
            if nonskip > 2:
                break
        return None, False

    def _negated(self, toks, i):
        for j in range(i - 1, max(-1, i - 1 - NEG_WINDOW), -1):
            if toks[j] in BOUNDARY:
                return False
            if toks[j] in NEG:
                return True
        return False

    @staticmethod
    def _scope(toks, i):
        """nose / palate / both / unspecified, from trigger words in the same sentence."""
        s = i
        while s > 0 and toks[s - 1] != "#":
            s -= 1
        e = i
        while e < len(toks) and toks[e] != "#":
            e += 1
        pre = [(j, "nose" if toks[j] in NOSE_TRIG else "palate") for j in range(s, i) if toks[j] in NOSE_TRIG or toks[j] in PALATE_TRIG]
        if pre:
            recent = {t for j, t in pre if i - j <= SCOPE_WINDOW}
            if len(recent) == 2:
                return "both"
            return pre[-1][1]
        post = [(j, "nose" if toks[j] in NOSE_TRIG else "palate") for j in range(i + 1, e) if toks[j] in NOSE_TRIG or toks[j] in PALATE_TRIG]
        return post[0][1] if post else "unspecified"

    @staticmethod
    def _variant(t):
        if t[0] != "aroma":
            return None
        return "white" if t[1].endswith("_white") else "red" if t[1].endswith("_red") else None

    def _resolve(self, targets, cg):
        """Drop a _white/_red cluster that contradicts the wine colour, unless nothing else would remain for that surface."""
        if cg is None:
            return targets
        keep = []
        for t in targets:
            v = self._variant(t)
            if v is None or v == cg:
                keep.append(t)
                continue
            others = [u for u in targets if u is not t and not (self._variant(u) not in (None, cg))]
            if not others:
                keep.append(t)
        return keep

    def match(self, text: str, colour=None):
        toks = tokenize(text)
        cg = colour_group(colour)
        i, out, covered = 0, [], [False] * len(toks)

        def emit(kind, f1, f2, surface, neg, tag, i0, amb=0, ctiers=""):
            sc = self._scope(toks, i0)
            base = dict(kind=kind, f1=f1, f2=f2, surface=surface, neg=neg, tag=tag, scope=sc, amb=amb, ctiers=ctiers)
            if kind == "structure" and f1 == "flavour_intensity" and sc in ("nose", "both"):
                out.append(dict(base, f1="nose_intensity"))
                if sc == "both":
                    out.append(base)
            else:
                out.append(base)

        while i < len(toks):
            lex_n = 0
            for n in range(min(self.maxlen, len(toks) - i), 1, -1):       # longest multi-word lexicon phrase (>=2 tokens) at i
                if tuple(toks[i:i + n]) in self.entries:
                    lex_n = n
                    break
            if toks[i] in self.bound and not lex_n:
                head, direct = self._find_head(toks, i)
                if head is not None:
                    neg = any(toks[k] in NEG for k in range(max(0, i - NEG_WINDOW), i)
                              if not any(t in BOUNDARY for t in toks[k:i]))
                    f1 = self.heads[toks[head]]
                    lvl = (self.bound[toks[i]] - 1) * len(self.scales[f1]) // 5 + 1     # generic 1-5 -> this scale's own length
                    emit("structure", f1, lvl,
                         toks[i] + " " + toks[head] if direct else toks[i] + " … " + toks[head], neg,
                         "bound" if direct else "bound-x", i)
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
                neg = self._negated(toks, i)
                res = self._resolve(self.entries[key], cg)
                at = [x for x in res if x[0] == "aroma"]
                amb = len({x[1] for x in at})
                ctiers = ",".join(sorted({self.tier[x[1]] for x in at}))
                for kind, a, b, tag in res:
                    emit(kind, a, b, " ".join(key), neg, tag, i, amb if kind == "aroma" else 0, ctiers if kind == "aroma" else "")
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


def readiness_outcome(drink_now, can_age, too_old):
    """SAT readiness outcomes, from stated cues only (None = not stated)."""
    if too_old:
        return "too_old"
    if drink_now and can_age:
        return "drink_now_potential"       # can drink now, but has potential for ageing
    if drink_now:
        return "drink_now"                 # ageing potential not stated
    if can_age:
        return "not_ready_potential"       # too young / needs time
    return None


def ageing_outcome(drink_now, can_age):
    """L4 card 'Bottle ageing': suitable / not suitable. From stated cues only; 'drink now' without ageing cues is read as NOT suitable (implied)."""
    if can_age:
        return "suitable"
    if drink_now:
        return "not_suitable"
    return None
