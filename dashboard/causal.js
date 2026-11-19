// Causal graph toolkit (port of src/causal.py): d-separation, back-door / front-door, adjustment sets, the three do-calculus rules.
// Parity-tested against Python with dashboard/causal_parity.js.
(function (root) {
  class DAG {
    constructor(nodes, edges) {
      this.nodes = nodes; this.edges = edges.map(e => [e[0], e[1]]);
      this.par = {}; this.ch = {}; Object.keys(nodes).forEach(n => { this.par[n] = new Set(); this.ch[n] = new Set(); });
      for (const [a, b] of this.edges) { if (!this.ch[a] || !this.par[b]) continue; this.ch[a].add(b); this.par[b].add(a); }
    }
    sub(into = [], out = []) { const I = new Set(into), O = new Set(out); return new DAG(this.nodes, this.edges.filter(([a, b]) => !I.has(b) && !O.has(a))); }
    anc(S, incl = true) { const seen = new Set(incl ? S : []), st = [...S]; while (st.length) for (const p of this.par[st.pop()]) if (!seen.has(p)) { seen.add(p); st.push(p); } return seen; }
    desc(S, incl = false) { const seen = new Set(incl ? S : []), st = [...S]; while (st.length) for (const c of this.ch[st.pop()]) if (!seen.has(c)) { seen.add(c); st.push(c); } return seen; }
    dsep(X, Y, Z) {
      const Zs = new Set(Z), keep = this.anc([...X, ...Y, ...Z]), und = {}; keep.forEach(n => und[n] = new Set());
      for (const b of keep) { const ps = [...this.par[b]].filter(p => keep.has(p)); for (const p of ps) { und[p].add(b); und[b].add(p); } for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) { und[ps[i]].add(ps[j]); und[ps[j]].add(ps[i]); } }
      const seen = new Set(), st = [...X].filter(x => !Zs.has(x));
      while (st.length) { const n = st.pop(); if (seen.has(n) || Zs.has(n)) continue; seen.add(n); for (const m of und[n]) if (!seen.has(m)) st.push(m); }
      return ![...Y].some(y => seen.has(y));
    }
    paths(x, y, maxlen = 9, cap = 4000) {
      const out = [], nb = {}; Object.keys(this.nodes).forEach(n => nb[n] = [...new Set([...this.par[n], ...this.ch[n]])].sort());
      const dfs = p => { if (out.length >= cap) return; const last = p[p.length - 1]; if (last === y) { out.push([...p]); return; } if (p.length >= maxlen) return; for (const m of nb[last]) if (!p.includes(m)) { p.push(m); dfs(p); p.pop(); } };
      dfs([x]); return out;
    }
    pathStatus(p, Z) {
      const Zs = new Set(Z), dz = Zs.size ? this.desc([...Zs], true) : new Set(), blockers = [];
      for (let i = 1; i < p.length - 1; i++) { const a = p[i - 1], v = p[i], b = p[i + 1], coll = this.par[v].has(a) && this.par[v].has(b);
        if (coll) { if (!dz.has(v)) blockers.push(`${v} is a collider and neither it nor a descendant is conditioned on`); } else if (Zs.has(v)) blockers.push(`${v} is conditioned on (non-collider)`); }
      const causal = p.every((n, i) => i === p.length - 1 || this.ch[n].has(p[i + 1]));
      return { path: p, kind: causal ? "causal" : (this.par[p[0]].has(p[1]) ? "backdoor" : "other"), open: blockers.length === 0, blockers };
    }
    adjustable(x, cond = []) { const C = new Set(cond); return Object.entries(this.nodes).filter(([n, v]) => ["observed", "derived", "new_data_acquired"].includes(v.kind) && n !== x && !C.has(n)).map(a => a[0]); }
    backdoorValid(x, y, Z, cond = []) {
      const bad = [...Z, ...cond].filter(v => this.desc([x]).has(v)); if (bad.length) return [false, `a descendant of the treatment is conditioned on: ${[...new Set(bad)].sort().join(", ")}`];
      const ok = this.sub([], [x]).dsep([x], [y], [...Z, ...cond]); return [ok, ok ? "all back-door paths are blocked" : "an open back-door path remains"];
    }
    adjustmentSets(x, y, cand, cond = [], maxsize = 3, limit = 12) {
      const dx = this.desc([x]), cs = cand.filter(c => !dx.has(c) && c !== x && c !== y).sort(), found = [];
      const comb = (arr, k, st = 0, cur = []) => k === 0 ? [cur] : arr.slice(st).flatMap((v, i) => comb(arr, k - 1, st + i + 1, [...cur, v]));
      for (let k = 0; k <= maxsize; k++) for (const S of comb(cs, k)) { if (found.some(f => f.every(v => S.includes(v)))) continue; if (this.backdoorValid(x, y, S, cond)[0]) found.push(S); if (found.length >= limit) return found; }
      return found;
    }
    frontdoorValid(x, y, M, cond = []) {
      const Ms = new Set(M); if ([...cond].some(c => this.desc([x]).has(c))) return [false, "a descendant of X is conditioned on"];
      const g = new DAG(Object.fromEntries(Object.entries(this.nodes).filter(([n]) => !Ms.has(n))), this.edges.filter(([a, b]) => !Ms.has(a) && !Ms.has(b)));
      if (g.desc([x]).has(y)) return [false, "some directed path from X to Y avoids M"];
      if (!this.sub([], [x]).dsep([x], [...M], cond)) return [false, "an open back-door path from X to M"];
      if (!this.sub([], [...M]).dsep([...M], [y], [x, ...cond])) return [false, "a back-door path from M to Y is not blocked by X"];
      return [true, "front-door criterion satisfied"];
    }
    frontdoorSets(x, y, cand, cond = [], maxsize = 2) {
      const out = [], cs = cand.filter(c => c !== x && c !== y).sort(), comb = (arr, k, st = 0, cur = []) => k === 0 ? [cur] : arr.slice(st).flatMap((v, i) => comb(arr, k - 1, st + i + 1, [...cur, v]));
      for (let k = 1; k <= maxsize; k++) for (const S of comb(cs, k)) { if (out.some(f => f.every(v => S.includes(v)))) continue; if (this.frontdoorValid(x, y, S, cond)[0]) out.push(S); }
      return out;
    }
    rule1(Y, X, Z, W) { return this.sub(X, []).dsep(Y, Z, [...X, ...W]); }
    rule2(Y, X, Z, W) { return this.sub(X, Z).dsep(Y, Z, [...X, ...W]); }
    rule3(Y, X, Z, W) { const gx = this.sub(X, []), aw = gx.anc(W), zw = Z.filter(z => !aw.has(z)); return this.sub([...X, ...zw], []).dsep(Y, Z, [...X, ...W]); }
    identify(x, y, cond = [], maxsize = 3) { const cand = this.adjustable(x, cond); const bd = this.adjustmentSets(x, y, cand, cond, maxsize), fd = this.frontdoorSets(x, y, cand, cond); return { backdoor: bd, frontdoor: fd, identified: bd.length > 0 || fd.length > 0 }; }
    layers() { const lv = {}, order = []; const indeg = {}; Object.keys(this.nodes).forEach(n => indeg[n] = this.par[n].size); let q = Object.keys(this.nodes).filter(n => !indeg[n]);
      while (q.length) { const n = q.shift(); order.push(n); lv[n] = Math.max(0, ...[...this.par[n]].map(p => lv[p] + 1)); for (const c of this.ch[n]) if (--indeg[c] === 0) q.push(c); } return lv; }
  }
  const api = { DAG }; if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WsetCausal = api;
})(typeof self !== "undefined" ? self : this);
