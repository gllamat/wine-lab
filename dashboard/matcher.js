// Browser/Node port of src/wsetkit.py (deterministic keyword -> WSET matcher).
// Parity-tested against the Python implementation (see src/parity_test.py + dashboard/parity.js).
(function (root) {
  const LEMMA_EXC = new Set(["molasses", "citrus", "asparagus", "grass", "christmas", "lees", "bias", "canvas"]);
  const NEG = new Set(["no", "not", "without", "lacks", "lacking", "lack", "devoid", "minus", "neither", "nor", "never", "free", "absent", "hardly"]);
  const BOUNDARY = new Set(["|", "but", "yet", "although", "though", "however", "while", "with", "and"]);
  const NEG_WINDOW = 3;

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
    t = t.replace(/[.;:,!?()\[\]"“”]/g, " | ");
    t = t.replace(/[^a-z0-9| ]/g, " ");
    return t.split(/\s+/).filter(Boolean).map(lemma);
  }

  class Matcher {
    // lex = parsed data/lexicon.json; opts.embed: include embedding-mapped terms
    constructor(lex, opts = {}) {
      this.entries = new Map();
      for (const [key, targets] of lex.entries) {
        const kept = targets.filter(t => opts.embed !== false || t[3] !== "embed");
        if (kept.length) this.entries.set(key, kept);
      }
      this.bound = new Map(Object.entries(lex.bound));
      this.heads = new Map(Object.entries(lex.heads));
      this.yearRe = lex.year_re;
      this.maxlen = 0;
      for (const k of this.entries.keys()) this.maxlen = Math.max(this.maxlen, k.split(" ").length);
    }
    findHead(toks, i) {
      for (let j = i + 1; j < Math.min(toks.length, i + 3); j++) {
        if (toks[j] === "|") break;
        if (this.heads.has(toks[j])) return [j, true];
      }
      let nonskip = 0;
      for (let j = i + 1; j < Math.min(toks.length, i + 7); j++) {
        const t = toks[j];
        if (this.heads.has(t)) return [j, false];
        if (t === "|" || t === "and") continue;
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
    match(text) {
      const toks = tokenize(text), covered = new Array(toks.length).fill(false), hits = [];
      let i = 0;
      while (i < toks.length) {
        if (this.bound.has(toks[i])) {
          const [head, direct] = this.findHead(toks, i);
          if (head !== null) {
            // bound-adjective negation: scan back over window, stopping at boundary (same as Python any(...) form)
            let neg = false;
            for (let k = Math.max(0, i - NEG_WINDOW); k < i; k++) {
              if (NEG.has(toks[k]) && !toks.slice(k, i).some(t => BOUNDARY.has(t))) { neg = true; break; }
            }
            hits.push({ kind: "structure", f1: this.heads.get(toks[head]), f2: this.bound.get(toks[i]),
              surface: direct ? toks[i] + " " + toks[head] : toks[i] + " … " + toks[head], neg, tag: direct ? "bound" : "bound-x", start: i, end: direct ? head + 1 : i + 1 });
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
          const [n, key] = hit, neg = this.negated(toks, i);
          for (const [kind, a, b, tag] of this.entries.get(key))
            hits.push({ kind, f1: a, f2: b, surface: key, neg, tag, start: i, end: i + n });
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
  const api = { tokenize, lemma, Matcher };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetMatcher = api;
})(typeof self !== "undefined" ? self : this);
