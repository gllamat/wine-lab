// Learned phrases: typed note -> Jev-aligned taste vector, with no model and no server. Rules were mined from labelled notes (experiments/recsys_sanity/mine_phrases.py):
// a phrase (word or two-word phrase) that Jev answered "yes" to often enough (precision) in enough notes (support). Parity: dashboard/phrases_parity.js
(function (root) {
  const lemma = w => (w.length > 4 && w.endsWith("ies")) ? w.slice(0, -3) + "y" : (w.length > 4 && /(ches|shes|xes|zes|sses)$/.test(w)) ? w.slice(0, -2) : (w.length > 3 && w.endsWith("s") && !/(ss|us|is)$/.test(w)) ? w.slice(0, -1) : w;
  function phrases(text) {
    const t = String(text).toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/'/g, "").replace(/-/g, " "), out = [];
    for (const clause of t.split(/[.;:,!?()]/)) { const w = (clause.match(/[a-z]+/g) || []).map(lemma); for (const x of w) out.push(x); for (let i = 0; i + 1 < w.length; i++) out.push(w[i] + " " + w[i + 1]); }
    return out;
  }
  class PhraseVec {
    constructor(P) { this.P = P; this.yes = new Map(); this.lev = new Map();
      P.rules.forEach((r, q) => r.forEach(([pi, pr]) => { const k = P.phrases[pi]; if (!this.yes.has(k)) this.yes.set(k, []); this.yes.get(k).push([q, pr / 100]); }));
      P.lrules.forEach((r, c) => r.forEach(([pi, pr]) => { const k = P.lphrases[pi]; if (!this.lev.has(k)) this.lev.set(k, []); this.lev.get(k).push([c, pr / 100]); }));
      this.loff = [0]; for (const l of P.levels) this.loff.push(this.loff[this.loff.length - 1] + l); }
    // cue: optional {scale: stated level (0-based, from the WSET cue matcher)}; returns z-scored (unweighted) 224-vector and the phrases that fired
    vector(text, cue = {}) {
      const P = this.P, nb = P.questions.length, sc = Float32Array.from(P.mu), fired = [], cl = new Float32Array(this.loff[this.loff.length - 1]), seen = new Set(phrases(text));
      const best = new Float32Array(nb);
      for (const p of seen) { const a = this.yes.get(p); if (a) for (const [q, pr] of a) { if (pr > best[q]) { best[q] = pr; } if (pr > sc[q]) sc[q] = pr; } const b = this.lev.get(p); if (b) for (const [c, pr] of b) if (pr > cl[c]) cl[c] = pr; }
      for (const p of seen) { const a = this.yes.get(p); if (a) for (const [q, pr] of a) if (pr === best[q]) fired.push({ phrase: p, question: P.questions[q], precision: pr }); }
      const z = new Float32Array(nb + P.scales.length); for (let q = 0; q < nb; q++) z[q] = (sc[q] - P.mu[q]) / P.sd[q];
      P.scales.forEach((s, k) => { let w = 0, a = 0; for (let l = 0; l < P.levels[k]; l++) { const v = cl[this.loff[k] + l]; w += v; a += v * l; }
        const lv = w > 0 ? a / w : (cue[s] != null ? cue[s] : P.lmu[k]); z[nb + k] = (lv - P.lmu[k]) / P.lsd[k]; });
      const seenQ = new Set(), uniq = fired.filter(f => { const key = f.question; if (seenQ.has(key)) return false; seenQ.add(key); return true; }).sort((x, y) => y.precision - x.precision);
      return { z, fired: uniq };
    }
  }
  const api = { phrases, PhraseVec }; if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetPhrases = api;
})(typeof self !== "undefined" ? self : this);
