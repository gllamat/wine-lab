// Recommender engine (port of src/recsys.py). Content-based: cosine kNN in raw / standardised / latent (SVD) WSET spaces and in the embedding space,
// score fusion, reciprocal rank fusion, MMR diversification, taste profiles (Rocchio) and critiquing. Parity: dashboard/recsys_parity.js
(function (root) {
  class RecIndex {
    constructor(meta, X8, E8) {
      this.meta = meta; const n = this.n = meta.n, dx = this.dx = meta.dim_x, de = this.de = meta.dim_e, nl = this.nl = meta.nlat, sx = meta.sx, se = meta.se;
      this.mu = Float32Array.from(meta.mu); this.sd = Float32Array.from(meta.sd); this.zmean = Float32Array.from(meta.zmean); this.comps = meta.comps.map(c => Float32Array.from(c));
      this.X = new Float32Array(n * dx); this.Z = new Float32Array(n * dx); this.L = new Float32Array(n * nl); this.E = X8 && E8 ? new Float32Array(n * de) : null;
      this.nrm = { raw: new Float32Array(n), zw: new Float32Array(n), latent: new Float32Array(n), emb: new Float32Array(n) };
      for (let i = 0; i < n; i++) {
        let a = 0, b = 0; for (let j = 0; j < dx; j++) { const x = X8[i * dx + j] / sx, z = (x - this.mu[j]) / this.sd[j]; this.X[i * dx + j] = x; this.Z[i * dx + j] = z; a += x * x; b += z * z; }
        this.nrm.raw[i] = Math.sqrt(a) || 1e-9; this.nrm.zw[i] = Math.sqrt(b) || 1e-9;
        let c = 0; for (let k = 0; k < nl; k++) { let s = 0; const cp = this.comps[k]; for (let j = 0; j < dx; j++) s += (this.Z[i * dx + j] - this.zmean[j]) * cp[j]; this.L[i * nl + k] = s; c += s * s; } this.nrm.latent[i] = Math.sqrt(c) || 1e-9;
        let d = 0; for (let j = 0; j < de; j++) { const e = E8[i * de + j] / se; this.E[i * de + j] = e; d += e * e; } this.nrm.emb[i] = Math.sqrt(d) || 1e-9;
      }
    }
    vec(space, i) { const D = { raw: this.dx, zw: this.dx, latent: this.nl, emb: this.de }[space], A = { raw: this.X, zw: this.Z, latent: this.L, emb: this.E }[space]; return A.subarray(i * D, (i + 1) * D); }
    query(i) { return { raw: Float32Array.from(this.vec("raw", i)), zw: Float32Array.from(this.vec("zw", i)), latent: Float32Array.from(this.vec("latent", i)), emb: Float32Array.from(this.vec("emb", i)) }; }
    queryFromFeatures(x) { const z = Float32Array.from(x, (v, j) => (v - this.mu[j]) / this.sd[j]), lat = new Float32Array(this.nl); for (let k = 0; k < this.nl; k++) { let s = 0; for (let j = 0; j < this.dx; j++) s += (z[j] - this.zmean[j]) * this.comps[k][j]; lat[k] = s; } return { raw: Float32Array.from(x), zw: z, latent: lat, emb: null }; }
    cos(space, q) {
      const v = q[space]; if (!v) return null; const D = v.length, A = { raw: this.X, zw: this.Z, latent: this.L, emb: this.E }[space], nr = this.nrm[space], out = new Float32Array(this.n); let qn = 0; for (let j = 0; j < D; j++) qn += v[j] * v[j]; qn = Math.sqrt(qn) || 1e-9;
      for (let i = 0; i < this.n; i++) { let s = 0; const o = i * D; for (let j = 0; j < D; j++) s += A[o + j] * v[j]; out[i] = s / (nr[i] * qn); } return out;
    }
    zs(s) { let m = 0; for (const v of s) m += v; m /= s.length; let q = 0; for (const v of s) q += (v - m) ** 2; const sd = Math.sqrt(q / s.length) + 1e-9; return Float32Array.from(s, v => (v - m) / sd); }
    hybrid(q, alpha = 0.5) { const a = this.zs(this.cos("zw", q)), e = q.emb ? this.zs(this.cos("emb", q)) : null; return Float32Array.from(a, (v, i) => e ? alpha * v + (1 - alpha) * e[i] : v); }
    ranks(s) { const o = Array.from({ length: this.n }, (_, i) => i).sort((a, b) => s[b] - s[a] || a - b), r = new Float32Array(this.n); o.forEach((i, k) => r[i] = k + 1); return r; }
    rrf(q, k0 = 60) { const sc = new Float32Array(this.n); for (const sp of ["zw", "emb"]) { const c = this.cos(sp, q); if (!c) continue; const r = this.ranks(c); for (let i = 0; i < this.n; i++) sc[i] += 1 / (k0 + r[i]); } return sc; }
    rank(s, k, excl = [], mask = null) { const ex = new Set(excl), idx = []; for (let i = 0; i < this.n; i++) if (!ex.has(i) && (!mask || mask(i))) idx.push(i); idx.sort((a, b) => s[b] - s[a] || a - b); return idx.slice(0, k); }
    embSim(a, b) { const de = this.de, A = this.E; let s = 0; for (let j = 0; j < de; j++) s += A[a * de + j] * A[b * de + j]; return s / (this.nrm.emb[a] * this.nrm.emb[b]); }
    mmr(rel, k, lam = 0.7, pool = 100, excl = [], mask = null) {
      let cand = this.rank(rel, pool, excl, mask); const chosen = [];
      while (chosen.length < k && cand.length) { let best;
        if (!chosen.length) best = cand[0]; else { let mn = Infinity, mx = -Infinity; for (const c of cand) { mn = Math.min(mn, rel[c]); mx = Math.max(mx, rel[c]); } let bv = -Infinity;
          for (const c of cand) { let ms = -Infinity; for (const s of chosen) ms = Math.max(ms, this.embSim(s, c)); const v = lam * (rel[c] - mn) / (mx - mn + 1e-9) - (1 - lam) * ms; if (v > bv) { bv = v; best = c; } } }
        chosen.push(best); cand = cand.filter(c => c !== best); }
      return chosen;
    }
    technique(name, i, k = 10, o = {}) { const q = this.query(i), ex = [i];
      if (["raw", "zw", "latent", "emb"].includes(name)) return this.rank(this.cos(name, q), k, ex);
      if (name === "hybrid") return this.rank(this.hybrid(q, o.alpha ?? 0.5), k, ex);
      if (name === "rrf") return this.rank(this.rrf(q), k, ex);
      if (name === "mmr") return this.mmr(this.hybrid(q, o.alpha ?? 0.5), k, o.lam ?? 0.7, 100, ex); throw new Error(name); }
    // taste profile (Rocchio): mean of liked minus beta * mean of disliked, in every space
    profile(liked, disliked, beta = 0.5) { const q = {}; for (const sp of ["raw", "zw", "latent", "emb"]) { const D = { raw: this.dx, zw: this.dx, latent: this.nl, emb: this.de }[sp], v = new Float32Array(D);
        for (const i of liked) { const a = this.vec(sp, i), n = this.nrm[sp][i]; for (let j = 0; j < D; j++) v[j] += a[j] / n / Math.max(1, liked.length); }
        for (const i of disliked) { const a = this.vec(sp, i), n = this.nrm[sp][i]; for (let j = 0; j < D; j++) v[j] -= beta * a[j] / n / Math.max(1, disliked.length); } q[sp] = v; } return q; }
    critique(q, delta) { const z = Float32Array.from(q.zw); for (const [name, d] of Object.entries(delta)) { const j = this.meta.names.indexOf(name); if (j >= 0) z[j] += d; if (name.startsWith("s_")) { const k = this.meta.names.indexOf(name + "_na"); if (k >= 0) z[k] = (0 - this.mu[k]) / this.sd[k]; } } const x = Float32Array.from(z, (v, j) => v * this.sd[j] + this.mu[j]); const o = this.queryFromFeatures(x); o.emb = null; return o; }
    shared(i, j) { const names = this.meta.names, out = []; for (let c = 0; c < this.dx; c++) if (names[c].startsWith("a_") && this.X[i * this.dx + c] > 0.5 && this.X[j * this.dx + c] > 0.5) out.push([names[c], this.sd[c]]); return out.sort((a, b) => (a[0].endsWith('_generic') - b[0].endsWith('_generic')) || (a[1] - b[1])).map(a => a[0]); }
  }
  const api = { RecIndex }; if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetRec = api;
})(typeof self !== "undefined" ? self : this);
