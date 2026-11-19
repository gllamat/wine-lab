// Browser/Node port of src/wsetkit.py v2 (deterministic keyword -> WSET L3 SAT matcher).
// Parity-tested against Python (src/export_lexicon.py parity fixture + dashboard/parity.js).
(function (root) {
  const LEMMA_EXC = new Set(["molasses", "citrus", "asparagus", "grass", "christmas", "lees", "bias", "canvas"]);
  const NEG = new Set(["no", "not", "without", "lacks", "lacking", "lack", "devoid", "minus", "neither", "nor", "never", "free", "absent", "hardly"]);
  const BOUNDARY = new Set(["|", "#", "but", "yet", "although", "though", "however", "while", "with", "and"]);
  const NOSE = new Set(["aroma", "nose", "bouquet", "smell", "scent"]);
  const PALATE = new Set(["palate", "flavor", "flavour", "taste", "mouth", "mouthfeel", "sip"]);
  const NEG_WINDOW = 3, SCOPE_WINDOW = 6;
  const CG = { red: "red", white: "white", rose: "white", sparkling: "white" };

  function lemma(t) {
    if (LEMMA_EXC.has(t)) return t;
    if (t.length <= 3 || !/^[a-z]+$/.test(t)) return t;
    if (t.endsWith("ies") && t.length > 4) return t.slice(0, -3) + "y";
    if (t.endsWith("sses")) return t.slice(0, -2);
    if (/(ches|shes|xes|zes)$/.test(t)) return t.slice(0, -2);
    if (t.endsWith("s") && !/(ss|us|is)$/.test(t)) return t.slice(0, -1);
    return t;
  }
  function tokenize(text) {
    let t = String(text).toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "");
    t = t.replace(/'/g, "");
    t = t.replace(/[-–—\/]/g, " ");
    t = t.replace(/[.;!?]/g, " # ");
    t = t.replace(/[:,()\[\]"“”]/g, " | ");
    t = t.replace(/[^a-z0-9|# ]/g, " ");
    return t.split(/\s+/).filter(Boolean).map(lemma);
  }

  class Matcher {
    constructor(lex, opts = {}) {
      this.tier = lex.tier;
      this.entries = new Map();
      for (const [key, targets] of lex.entries) {
        const kept = targets.filter(t => opts.embed !== false || t[3] !== "embed");
        if (kept.length) this.entries.set(key, kept);
      }
      this.bound = new Map(Object.entries(lex.bound));
      this.heads = new Map(Object.entries(lex.heads));
      this.scales = lex.scales;
      this.yearRe = lex.year_re;
      this.maxlen = 0;
      for (const k of this.entries.keys()) this.maxlen = Math.max(this.maxlen, k.split(" ").length);
    }
    findHead(toks, i) {
      for (let j = i + 1; j < Math.min(toks.length, i + 3); j++) {
        if (toks[j] === "|" || toks[j] === "#") break;
        if (this.heads.has(toks[j])) return [j, true];
      }
      let nonskip = 0;
      for (let j = i + 1; j < Math.min(toks.length, i + 7); j++) {
        const t = toks[j];
        if (this.heads.has(t)) return [j, false];
        if (t === "|" || t === "and") continue;
        if (t === "#") break;
        const ok = this.bound.has(t) || (this.entries.get(t) || []).some(e => e[0] === "structure");
        if (!ok) break;
        nonskip++;
        if (nonskip > 2) break;
      }
      return [null, false];
    }
    negated(toks, i) {
      for (let j = i - 1; j > Math.max(-1, i - 1 - NEG_WINDOW); j--) {
        if (BOUNDARY.has(toks[j])) return false;
        if (NEG.has(toks[j])) return true;
      }
      return false;
    }
    scope(toks, i) {
      let s = i; while (s > 0 && toks[s - 1] !== "#") s--;
      let e = i; while (e < toks.length && toks[e] !== "#") e++;
      const trig = j => NOSE.has(toks[j]) ? "nose" : PALATE.has(toks[j]) ? "palate" : null;
      const pre = [];
      for (let j = s; j < i; j++) { const t = trig(j); if (t) pre.push([j, t]); }
      if (pre.length) {
        const recent = new Set(pre.filter(([j]) => i - j <= SCOPE_WINDOW).map(x => x[1]));
        if (recent.size === 2) return "both";
        return pre[pre.length - 1][1];
      }
      for (let j = i + 1; j < e; j++) { const t = trig(j); if (t) return t; }
      return "unspecified";
    }
    variant(t) { return t[0] !== "aroma" ? null : t[1].endsWith("_white") ? "white" : t[1].endsWith("_red") ? "red" : null; }
    resolve(targets, cg) {
      if (!cg) return targets;
      const keep = [];
      for (const t of targets) {
        const v = this.variant(t);
        if (v === null || v === cg) { keep.push(t); continue; }
        const others = targets.filter(u => u !== t && !(this.variant(u) !== null && this.variant(u) !== cg));
        if (!others.length) keep.push(t);
      }
      return keep;
    }
    match(text, colour) {
      const toks = tokenize(text), cg = CG[colour] || null, covered = new Array(toks.length).fill(false), hits = [];
      const emit = (kind, f1, f2, surface, neg, tag, i0, start, end, amb = 0, ctiers = "") => {
        const sc = this.scope(toks, i0);
        const base = { kind, f1, f2, surface, neg, tag, scope: sc, amb, ctiers, start, end };
        if (kind === "structure" && f1 === "flavour_intensity" && (sc === "nose" || sc === "both")) {
          hits.push({ ...base, f1: "nose_intensity" });
          if (sc === "both") hits.push(base);
        } else hits.push(base);
      };
      let i = 0;
      while (i < toks.length) {
        let lexN = 0;
        for (let n = Math.min(this.maxlen, toks.length - i); n > 1; n--) {
          if (this.entries.has(toks.slice(i, i + n).join(" "))) { lexN = n; break; }
        }
        if (this.bound.has(toks[i]) && !lexN) {
          const [head, direct] = this.findHead(toks, i);
          if (head !== null) {
            let neg = false;
            for (let k = Math.max(0, i - NEG_WINDOW); k < i; k++) {
              if (NEG.has(toks[k]) && !toks.slice(k, i).some(t => BOUNDARY.has(t))) { neg = true; break; }
            }
            const f1b = this.heads.get(toks[head]), lvlb = Math.floor((this.bound.get(toks[i]) - 1) * this.scales[f1b].length / 5) + 1;
            emit("structure", f1b, lvlb,
              direct ? toks[i] + " " + toks[head] : toks[i] + " … " + toks[head], neg, direct ? "bound" : "bound-x", i, i, direct ? head + 1 : i + 1);
            if (direct) { for (let j = i; j <= head; j++) covered[j] = true; i = head + 1; }
            else { covered[i] = true; i += 1; }
            continue;
          }
        }
        let hit = null;
        for (let n = Math.min(this.maxlen, toks.length - i); n > 0; n--) {
          const key = toks.slice(i, i + n).join(" ");
          if (this.entries.has(key)) { hit = [n, key]; break; }
        }
        if (hit) {
          const [n, key] = hit, neg = this.negated(toks, i), res = this.resolve(this.entries.get(key), cg);
          const at = res.filter(x => x[0] === "aroma");
          const amb = new Set(at.map(x => x[1])).size, ctiers = [...new Set(at.map(x => this.tier[x[1]]))].sort().join(",");
          for (const [kind, a, b, tag] of res) emit(kind, a, b, key, neg, tag, i, i, i + n, kind === "aroma" ? amb : 0, kind === "aroma" ? ctiers : "");
          for (let j = i; j < i + n; j++) covered[j] = true;
          i += n;
        } else i += 1;
      }
      const years = [], re = new RegExp(this.yearRe, "g"), low = String(text).toLowerCase();
      for (const m of low.matchAll(re)) {
        const y1 = parseInt(m[1], 10); let y2 = m[2];
        if (y2 !== undefined) y2 = y2.length === 4 ? parseInt(y2, 10) : parseInt(String(y1).slice(0, 2) + y2, 10);
        years.push([y1, y2 === undefined ? null : y2]);
      }
      return { toks, covered, hits, years };
    }
  }
  // Assumption A1 (lexicon/structure.yaml): fruit ripeness is one-of per note -> [state|null, mixed]
  function resolveRipeness(hits) {
    const s = new Set(hits.filter(h => h.kind === "aroma" && !h.neg && (h.f1 === "fruit_ripeness" || ((h.f1 === "tertiary_red" || h.f1 === "tertiary_white") && (h.f2 === "dried_fruit" || h.f2 === "cooked_fruit")))).map(h => h.f2));
    if (s.size > 1) s.delete("ripe_fruit");
    return s.size === 1 ? [[...s][0], false] : [null, s.size > 0];
  }
  const api = { tokenize, lemma, Matcher, resolveRipeness };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetMatcher = api;
})(typeof self !== "undefined" ? self : this);
