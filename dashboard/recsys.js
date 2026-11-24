// Recommender engine (port of src/recsys.py). Content-based: cosine kNN in raw / standardised / latent (SVD) WSET spaces and in the embedding space,
// score fusion, reciprocal rank fusion, MMR diversification, taste profiles (Rocchio) and critiquing. Parity: dashboard/recsys_parity.js
// Two storage modes with identical results: normal (float arrays, fast, ~25k wines) and compact (int8 arrays + on-the-fly maths, ~170k wines in ~120 MB).
(function (root) {
  class RecIndex {
    constructor(meta, X8, E8, opts = {}) {
      this.meta = meta; const n = this.n = meta.n, dx = this.dx = meta.dim_x, de = this.de = meta.dim_e, nl = this.nl = meta.nlat, sx = meta.sx, se = meta.se;
      this.compact = opts.compact !== undefined ? opts.compact : n > 60000; this.X8 = X8; this.E8 = E8;
      this.mu = Float32Array.from(meta.mu); this.sd = Float32Array.from(meta.sd); this.zmean = Float32Array.from(meta.zmean); this.comps = meta.comps.map(c => Float32Array.from(c));
      this.L = new Float32Array(n * nl); this.nrm = { raw: new Float32Array(n), zw: new Float32Array(n), latent: new Float32Array(n), emb: new Float32Array(n) };
      const full = !this.compact; if (full) { this.X = new Float32Array(n * dx); this.Z = new Float32Array(n * dx); this.E = new Float32Array(n * de); }
      const z = new Float32Array(dx);
      for (let i = 0; i < n; i++) {
        let a = 0, b = 0; for (let j = 0; j < dx; j++) { const x = X8[i * dx + j] / sx, zz = (x - this.mu[j]) / this.sd[j]; z[j] = zz; a += x * x; b += zz * zz; if (full) { this.X[i * dx + j] = x; this.Z[i * dx + j] = zz; } }
        this.nrm.raw[i] = Math.sqrt(a) || 1e-9; this.nrm.zw[i] = Math.sqrt(b) || 1e-9;
        let c = 0; for (let k = 0; k < nl; k++) { let s = 0; const cp = this.comps[k]; for (let j = 0; j < dx; j++) s += (z[j] - this.zmean[j]) * cp[j]; this.L[i * nl + k] = s; c += s * s; } this.nrm.latent[i] = Math.sqrt(c) || 1e-9;
        let d = 0; for (let j = 0; j < de; j++) { const e = E8[i * de + j] / se; if (full) this.E[i * de + j] = e; d += e * e; } this.nrm.emb[i] = Math.sqrt(d) || 1e-9;
      }
    }
    xval(i, j) { return this.X8[i * this.dx + j] / this.meta.sx; }
    vec(space, i) {
      const dx = this.dx, de = this.de, sx = this.meta.sx, se = this.meta.se;
      if (space === "latent") return this.L.subarray(i * this.nl, (i + 1) * this.nl);
      if (space === "jev") return this.J.subarray(i * this.jm.dim, (i + 1) * this.jm.dim);
      if (!this.compact) { const D = space === "emb" ? de : dx, A = { raw: this.X, zw: this.Z, emb: this.E }[space]; return A.subarray(i * D, (i + 1) * D); }
      if (space === "raw") return Float32Array.from({ length: dx }, (_, j) => this.X8[i * dx + j] / sx);
      if (space === "zw") return Float32Array.from({ length: dx }, (_, j) => (this.X8[i * dx + j] / sx - this.mu[j]) / this.sd[j]);
      return Float32Array.from({ length: de }, (_, j) => this.E8[i * de + j] / se);
    }
    query(i) { const q = { raw: Float32Array.from(this.vec("raw", i)), zw: Float32Array.from(this.vec("zw", i)), latent: Float32Array.from(this.vec("latent", i)), emb: Float32Array.from(this.vec("emb", i)) }; if (this.J) q.jev = Float32Array.from(this.J.subarray(i * this.jm.dim, (i + 1) * this.jm.dim)); return q; }
    // ---- Jev-aligned taste vector (224 numbers per wine: Jev's 216 yes/no answers + 8 structure levels), taster's rules and hub correction. Data: recsys_j.bin (int8 z-scores), recsys_hub.bin (uint8).
    attachJev(J8, H8) {
      const jm = this.jm = this.meta.jev; if (!jm || this.compact) return false; const n = this.n, D = jm.dim; this.J8 = J8; this.J = new Float32Array(n * D); this.nrm.jev = new Float32Array(n).fill(1);
      for (let i = 0; i < n; i++) { let a = 0; for (let j = 0; j < D; j++) { const v = J8[i * D + j] / jm.scale * (j >= jm.nb ? jm.lw : 1); this.J[i * D + j] = v; a += v * v; } a = Math.sqrt(a) || 1e-9; for (let j = 0; j < D; j++) this.J[i * D + j] /= a; }
      if (H8) { const dec = k => Float32Array.from({ length: n }, (_, i) => jm.hub_lo + H8[k * n + i] / 255 * jm.hub_span); this.hubOff = { jev: dec(0), emb: dec(1) }; }
      this.hubOn = false; this.colourCodes = this.meta.codes.colour; return true;
    }
    sweet(i) { const jm = this.jm; return this.J8[i * jm.dim + jm.nb] / jm.scale * jm.lev_sd[0] + jm.lev_mu[0]; }
    level(i, s) { const jm = this.jm; return this.J8[i * jm.dim + jm.nb + s] / jm.scale * jm.lev_sd[s] + jm.lev_mu[s]; }
    // the taster's rules: same colour family, sweetness within swmax levels (predicted). qSweet overrides the query's sweetness (typed notes); qColour null = unknown (typed notes)
    rulesMask(i, qSweet = null, qColour = undefined) {
      const jm = this.jm, sw = qSweet != null ? qSweet : (i != null ? this.sweet(i) : null), col = qColour !== undefined ? qColour : (i != null ? this.colourCodes[i] : null);
      return j => (col == null || this.colourCodes[j] === col) && (sw == null || Math.abs(this.sweet(j) - sw) < jm.swmax);
    }
    // a typed note -> query vector from z-scored (unweighted) Jev dimensions
    queryFromJev(z) { const jm = this.jm, v = Float32Array.from(z, (x, j) => x * (j >= jm.nb ? jm.lw : 1)); let a = 0; for (const x of v) a += x * x; a = Math.sqrt(a) || 1e-9; return v.map(x => x / a); }
    sharedJ(i, j, max = 6) { const jm = this.jm, D = jm.dim, out = []; for (let c = 0; c < jm.nb; c++) { const a = this.J8[i * D + c] / jm.scale, b = this.J8[j * D + c] / jm.scale, m = Math.min(a, b); if (m > 1.2 && !jm.questions[c].startsWith("stated__")) out.push([jm.questions[c], m]); } return out.sort((x, y) => y[1] - x[1]).slice(0, max); }
    hybridJ(q, alpha = 0.5) { const a = this.zs(this.cos("jev", q)), e = q.emb ? this.zs(this.cos("emb", q)) : null; return Float32Array.from(a, (v, i) => e ? alpha * v + (1 - alpha) * e[i] : v); }
    queryFromFeatures(x) { const z = Float32Array.from(x, (v, j) => (v - this.mu[j]) / this.sd[j]), lat = new Float32Array(this.nl); for (let k = 0; k < this.nl; k++) { let s = 0; for (let j = 0; j < this.dx; j++) s += (z[j] - this.zmean[j]) * this.comps[k][j]; lat[k] = s; } return { raw: Float32Array.from(x), zw: z, latent: lat, emb: null }; }
    cos(space, q) {
      const o = this._cos(space, q); if (o && this.hubOn && this.hubOff && this.hubOff[space]) { const h = this.hubOff[space], lam = this.jm.lam; for (let i = 0; i < this.n; i++) o[i] -= lam * h[i]; } return o;
    }
    _cos(space, q) {
      const v = q[space]; if (!v) return null; const D = v.length, nr = this.nrm[space], out = new Float32Array(this.n); let qn = 0; for (let j = 0; j < D; j++) qn += v[j] * v[j]; qn = Math.sqrt(qn) || 1e-9;
      if (!this.compact || space === "latent") { const A = { raw: this.X, zw: this.Z, latent: this.L, emb: this.E, jev: this.J }[space];
        for (let i = 0; i < this.n; i++) { let s = 0; const o = i * D; for (let j = 0; j < D; j++) s += A[o + j] * v[j]; out[i] = s / (nr[i] * qn); } return out; }
      const dx = this.dx, de = this.de, sx = this.meta.sx, se = this.meta.se, w = new Float32Array(D); let c0 = 0;
      if (space === "raw") for (let j = 0; j < D; j++) w[j] = v[j] / sx;
      else if (space === "zw") for (let j = 0; j < D; j++) { w[j] = v[j] / (sx * this.sd[j]); c0 += this.mu[j] * v[j] / this.sd[j]; }
      else for (let j = 0; j < D; j++) w[j] = v[j] / se;
      const A = space === "emb" ? this.E8 : this.X8;
      for (let i = 0; i < this.n; i++) { let s = 0; const o = i * D; for (let j = 0; j < D; j++) s += A[o + j] * w[j]; out[i] = (s - c0) / (nr[i] * qn); }
      return out;
    }
    zs(s) { let m = 0; for (const v of s) m += v; m /= s.length; let q = 0; for (const v of s) q += (v - m) ** 2; const sd = Math.sqrt(q / s.length) + 1e-9; return Float32Array.from(s, v => (v - m) / sd); }
    hybrid(q, alpha = 0.5) { const a = this.zs(this.cos("zw", q)), e = q.emb ? this.zs(this.cos("emb", q)) : null; return Float32Array.from(a, (v, i) => e ? alpha * v + (1 - alpha) * e[i] : v); }
    ranks(s) { const o = Array.from({ length: this.n }, (_, i) => i).sort((a, b) => s[b] - s[a] || a - b), r = new Float32Array(this.n); o.forEach((i, k) => r[i] = k + 1); return r; }
    rrf(q, k0 = 60) { const sc = new Float32Array(this.n); for (const sp of ["zw", "emb"]) { const c = this.cos(sp, q); if (!c) continue; const r = this.ranks(c); for (let i = 0; i < this.n; i++) sc[i] += 1 / (k0 + r[i]); } return sc; }
    // top-k by score (ties: lower index first), skipping excluded and masked items; O(n·k) worst case but fast because most scores fail the threshold
    rank(s, k, excl = [], mask = null) {
      const ex = new Set(excl), top = []; let thr = -Infinity;
      for (let i = 0; i < this.n; i++) { const v = s[i]; if (top.length >= k && v <= thr) continue; if (ex.has(i) || (mask && !mask(i))) continue;
        let p = top.length; while (p > 0 && s[top[p - 1]] < v) p--; top.splice(p, 0, i); if (top.length > k) top.pop(); if (top.length >= k) thr = s[top[top.length - 1]]; }
      return top;
    }
    embSim(a, b) { const de = this.de, A = this.compact ? this.E8 : this.E, sc = this.compact ? this.meta.se * this.meta.se : 1; let s = 0; for (let j = 0; j < de; j++) s += A[a * de + j] * A[b * de + j]; return s / sc / (this.nrm.emb[a] * this.nrm.emb[b]); }
    mmr(rel, k, lam = 0.7, pool = 100, excl = [], mask = null) {
      let cand = this.rank(rel, pool, excl, mask); const chosen = [];
      while (chosen.length < k && cand.length) { let best;
        if (!chosen.length) best = cand[0]; else { let mn = Infinity, mx = -Infinity; for (const c of cand) { mn = Math.min(mn, rel[c]); mx = Math.max(mx, rel[c]); } let bv = -Infinity;
          for (const c of cand) { let ms = -Infinity; for (const s of chosen) ms = Math.max(ms, this.embSim(s, c)); const v = lam * (rel[c] - mn) / (mx - mn + 1e-9) - (1 - lam) * ms; if (v > bv) { bv = v; best = c; } } }
        chosen.push(best); cand = cand.filter(c => c !== best); }
      return chosen;
    }
    technique(name, i, k = 10, o = {}) { const q = this.query(i), ex = [i], mk = o.rules ? this.rulesMask(i) : null; if (o.hub !== undefined) this.hubOn = !!o.hub;
      if (["raw", "zw", "latent", "emb", "jev"].includes(name)) return this.rank(this.cos(name, q), k, ex, mk);
      if (name === "jevh") return this.rank(this.hybridJ(q, o.alpha ?? 0.5), k, ex, mk);
      if (name === "hybrid") return this.rank(this.hybrid(q, o.alpha ?? 0.5), k, ex, mk);
      if (name === "rrf") return this.rank(this.rrf(q), k, ex, mk);
      if (name === "mmr") return this.mmr(this.hybrid(q, o.alpha ?? 0.5), k, o.lam ?? 0.7, 100, ex, mk); throw new Error(name); }
    // taste profile (Rocchio): mean of liked minus beta * mean of disliked, in every space
    profile(liked, disliked, beta = 0.5) { const q = {}; for (const sp of ["raw", "zw", "latent", "emb"]) { const D = { raw: this.dx, zw: this.dx, latent: this.nl, emb: this.de }[sp], v = new Float32Array(D);
        for (const i of liked) { const a = this.vec(sp, i), n = this.nrm[sp][i]; for (let j = 0; j < D; j++) v[j] += a[j] / n / Math.max(1, liked.length); }
        for (const i of disliked) { const a = this.vec(sp, i), n = this.nrm[sp][i]; for (let j = 0; j < D; j++) v[j] -= beta * a[j] / n / Math.max(1, disliked.length); } q[sp] = v; } return q; }
    critique(q, delta) { const z = Float32Array.from(q.zw); for (const [name, d] of Object.entries(delta)) { const j = this.meta.names.indexOf(name); if (j >= 0) z[j] += d; if (name.startsWith("s_")) { const k = this.meta.names.indexOf(name + "_na"); if (k >= 0) z[k] = (0 - this.mu[k]) / this.sd[k]; } } const x = Float32Array.from(z, (v, j) => v * this.sd[j] + this.mu[j]); const o = this.queryFromFeatures(x); o.emb = null; return o; }
    shared(i, j) { const names = this.meta.names, out = []; for (let c = 0; c < this.dx; c++) if (names[c].startsWith("a_") && this.xval(i, c) > 0.5 && this.xval(j, c) > 0.5) out.push([names[c], this.sd[c]]); return out.sort((a, b) => (a[0].endsWith('_generic') - b[0].endsWith('_generic')) || (a[1] - b[1])).map(a => a[0]); }
  }
  const api = { RecIndex }; if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetRec = api;
})(typeof self !== "undefined" ? self : this);
