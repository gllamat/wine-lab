// Browser/Node port of src/guess_model.py (LITE tier) inference + src/guess_features.py featurize_hits.
// Parity-tested against Python with dashboard/guess_parity.js. Deterministic: no sampling.
(function (root) {
  const BOUNDS = new Set(["|", "#"]);
  const YEAR0 = 2005, YEARSD = 10;
  const clip = (v, a, b) => Math.min(b, Math.max(a, v));
  function featurize(r, spec, resolveRipeness) {
    const ix = spec.index || (spec.index = Object.fromEntries(spec.names.map((n, i) => [n, i]))), x = new Float32Array(spec.names.length);
    const pos = r.hits.filter(h => !h.neg), ar = pos.filter(h => h.kind === "aroma");
    for (const h of ar) if (h.f1 !== "fruit_ripeness") { const k = `a_${h.f1}__${h.f2}`; if (k in ix) x[ix[k]] = 1; }
    const [rip] = resolveRipeness(r.hits); if (rip) x[ix[`a_fruit_ripeness__${rip}`]] = 1;
    for (const [s, L] of Object.entries(spec.scales)) {
      const lv = pos.filter(h => h.kind === "structure" && h.f1 === s).map(h => +h.f2);
      if (lv.length) x[ix["s_" + s]] = (lv.reduce((a, b) => a + b, 0) / lv.length - 1) / Math.max(1, L - 1);
      else if (spec.na_scales.includes(s)) x[ix["s_" + s + "_na"]] = 1;
    }
    for (const [f, v] of spec.cats) if (pos.some(h => h.kind === "cat" && h.f1 === f && h.f2 === v)) x[ix[`cat_${f}=${v}`]] = 1;
    for (const k of ["drink_now", "can_age", "too_old"]) if (pos.some(h => h.kind === "readiness" && h.f1 === k)) x[ix["r_" + k]] = 1;
    const y = r.years && r.years.length ? r.years[0] : null;
    for (const [k, val] of [["yf", y ? y[0] : null], ["yt", y && y[1] ? y[1] : null]]) { if (val === null) x[ix[k + "_na"]] = 1; else x[ix[k]] = clip((val - YEAR0) / YEARSD, -2, 3); }
    x[ix["len_log"]] = Math.log1p(r.toks.filter(t => !BOUNDS.has(t)).length);
    return x;
  }
  const dot = (W, z, n) => W.map(w => { let s = 0; for (let i = 0; i < n; i++) s += w[i] * z[i]; return s; });
  function softmaxT(l, T) { const z = l.map(v => v / T), m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
  const digitize = (v, edges) => { let i = 0; while (i < edges.length && v >= edges[i]) i++; return i; };
  function haversine(la1, lo1, la2, lo2) { const R = Math.PI / 180, a1 = la1 * R, o1 = lo1 * R, a2 = la2 * R, o2 = lo2 * R, h = Math.sin((a2 - a1) / 2) ** 2 + Math.cos(a1) * Math.cos(a2) * Math.sin((o2 - o1) / 2) ** 2; return 6371 * 2 * Math.asin(Math.sqrt(h)); }
  function predict(M, x) {
    const n = M.mu.length, z = Array.from(x, (v, i) => (v - M.mu[i]) / M.sd[i]), out = {};
    const Pv = softmaxT(dot(M.variety.W, z, n).map((v, i) => v + M.variety.b[i]), M.variety.T);
    const Pg = new Array(M.variety.groups.length).fill(0); Pv.forEach((p, i) => Pg[M.variety.G[i]] += p);
    const Pc = softmaxT(dot(M.country.W, z, n).map((v, i) => v + M.country.b[i]), 1);
    const lg = p => Math.log(clip(p, 1e-4, 1)), zp = z.concat(Pg.map(lg), Pc.map(lg)), np_ = zp.length;
    const Pp = softmaxT(dot(M.place.W, zp, np_).map((v, i) => v + M.place.b[i]), M.place.T);
    Object.assign(out, { Pv, Pg, Pp });
    for (const [k, r] of Object.entries(M.reg)) { let s = r.b; for (let i = 0; i < np_; i++) s += r.W[i] * zp[i]; const q = r.q[digitize(s, r.edges)]; out[k] = [s + q[0], s + q[1], s + q[2]]; }
    const P = M.producer, nw = P.wineries.length, sc = new Float64Array(nw);
    for (let e = 0; e < P.mw_row.length; e++) { const c = P.mw_row[e]; sc[P.mw_col[e]] += Pp[P.combo[c][0]] * Pv[P.combo[c][1]] * P.mw_val[e]; }
    const q = P.pca_comp.map(c => { let s = 0; for (let i = 0; i < n; i++) s += c[i] * (z[i] - P.pca_mean[i]); return s; });
    const d2 = new Float64Array(nw); let dmin = Infinity;
    for (let w = 0; w < nw; w++) { let s = 0; const m = P.mu_w[w]; for (let i = 0; i < q.length; i++) { const d = q[i] - m[i]; s += d * d; } d2[w] = s; if (s < dmin) dmin = s; }
    let tot = 0; for (let w = 0; w < nw; w++) { sc[w] *= Math.exp(-(d2[w] - dmin) / (2 * P.tau * P.tau)); tot += sc[w]; }
    out.Pw = Array.from(sc, v => v / Math.max(tot, 1e-12) * (1 - P.p_other));
    return out;
  }
  const top = (P, classes, k = 5) => P.map((p, i) => [classes[i], p]).sort((a, b) => b[1] - a[1]).slice(0, k);
  function placePoint(M, Pp) {
    const xy = M.place.xy, ok = xy.map(v => v[0] !== null); let p = Pp.map((v, i) => ok[i] ? v : 0); const s = p.reduce((a, b) => a + b, 0) || 1e-9; p = p.map(v => v / s);
    const cand = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 15).map(a => a[1]);
    const dist = (i, j) => (xy[i][0] === null || xy[j][0] === null) ? 0 : haversine(xy[i][0], xy[i][1], xy[j][0], xy[j][1]);
    let best = cand[0], bv = Infinity; for (const c of cand) { let e = 0; for (let j = 0; j < p.length; j++) e += dist(c, j) * p[j]; if (e < bv) { bv = e; best = c; } }
    const d = xy.map((v, j) => ok[j] ? haversine(xy[best][0], xy[best][1], v[0], v[1]) : 1e9), o = d.map((v, i) => i).sort((a, b) => d[a] - d[b]);
    let cum = 0, r50 = null, r80 = null; for (const j of o) { cum += p[j]; if (r50 === null && cum >= 0.5) r50 = d[j]; if (r80 === null && cum >= 0.8) { r80 = d[j]; break; } }
    return [xy[best][0], xy[best][1], r50, r80 ?? d[o[o.length - 1]]];
  }
  const api = { featurize, predict, top, placePoint, haversine };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetGuess = api;
})(typeof self !== "undefined" ? self : this);
